import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


KAGGLE_DATASET = "teejmahal20/airline-passenger-satisfaction"
DEFAULT_REFERENCE_REAL = "balanced_passenger_survey_dataset/passenger_survey_balanced.csv"
DEFAULT_OUTPUT_DIR = "synthetic_tabular_detection_audit"
DEFAULT_TARGET_COLUMN = "satisfaction"
DEFAULT_DROP_COLUMNS = [
    "Unnamed: 0",
    "id",
    "__split",
]


@dataclass
class DetectorResult:
    view: str
    setup: str
    generator: str
    auc: float | None
    separability_auc: float | None
    accuracy: float | None
    n_real: int
    n_synthetic: int
    interpretation: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper-inspired synthetic tabular data audit for the Kaggle Airline "
            "Passenger Satisfaction dataset. Implements the row text linearization "
            "+ character trigram logistic regression baseline described in "
            "arXiv:2412.13227, plus same-table C2ST sanity checks."
        )
    )
    parser.add_argument(
        "--target-train",
        default=None,
        help="Path to Kaggle train.csv. If omitted with --download-kaggle, the dataset is downloaded via kagglehub.",
    )
    parser.add_argument(
        "--target-test",
        default=None,
        help="Optional path to Kaggle test.csv.",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help=f"Download {KAGGLE_DATASET} through kagglehub and load train.csv/test.csv.",
    )
    parser.add_argument(
        "--reference-real",
        default=DEFAULT_REFERENCE_REAL,
        help=(
            "Reference real table for cross-table scoring. By default this uses the "
            "Brazilian passenger survey dataset built in this project."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON/CSV/Markdown audit outputs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12000,
        help="Maximum rows sampled from target and reference tables for speed.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column in Kaggle dataset. Used for feature-only view.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=DEFAULT_DROP_COLUMNS,
        help="Columns dropped as identifiers/export artifacts.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=100000,
        help="Maximum character trigram features for CountVectorizer.",
    )
    return parser.parse_args()


def load_csv(path: str, split_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    if split_name is not None:
        df["__split"] = split_name
    return df


def load_target_dataset(args) -> pd.DataFrame:
    if args.download_kaggle:
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for --download-kaggle. Install it with "
                "python3 -m pip install 'kagglehub[pandas-datasets]'"
            ) from exc

        dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
        train_path = os.path.join(dataset_dir, "train.csv")
        test_path = os.path.join(dataset_dir, "test.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Downloaded dataset does not contain train.csv: {dataset_dir}")
        frames = [load_csv(train_path, "train")]
        if os.path.exists(test_path):
            frames.append(load_csv(test_path, "test"))
        return pd.concat(frames, ignore_index=True)

    if not args.target_train:
        raise ValueError("Provide --target-train or use --download-kaggle.")

    frames = [load_csv(args.target_train, "train")]
    if args.target_test:
        frames.append(load_csv(args.target_test, "test"))
    return pd.concat(frames, ignore_index=True)


def sample_rows(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def strip_columns(df: pd.DataFrame, drop_columns: Iterable[str]) -> pd.DataFrame:
    keep_cols = [col for col in df.columns if col not in set(drop_columns)]
    return df.loc[:, keep_cols].copy()


def build_views(df: pd.DataFrame, args, table_name: str) -> dict[str, pd.DataFrame]:
    base = strip_columns(df, args.drop_columns)
    views = {"full_content": base}
    if args.target_column in base.columns:
        views["feature_only"] = base.drop(columns=[args.target_column])
    else:
        views["feature_only"] = base.copy()

    cleaned = {}
    for view_name, view_df in views.items():
        view_df = view_df.copy()
        view_df.columns = [f"{table_name}__{str(col).strip()}" for col in view_df.columns]
        cleaned[view_name] = view_df
    return cleaned


def format_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "<NA>"
        return f"{float(value):.6g}"
    text = str(value).strip()
    return text if text else "<EMPTY>"


def linearize_rows(df: pd.DataFrame, random_state: int, shuffle_columns: bool = True) -> list[str]:
    rng = np.random.default_rng(random_state)
    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        row_columns = columns.copy()
        if shuffle_columns:
            rng.shuffle(row_columns)
        parts = [f"{col}:{format_value(row[col])}" for col in row_columns]
        rows.append(",".join(parts))
    return rows


def make_detector(max_features: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "trigrams",
                CountVectorizer(
                    analyzer="char",
                    ngram_range=(3, 3),
                    lowercase=False,
                    min_df=2,
                    max_features=max_features,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def generate_independent_marginals(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    generated = {}
    n_rows = len(df)
    for col in df.columns:
        values = df[col].to_numpy()
        generated[col] = rng.choice(values, size=n_rows, replace=True)
    return pd.DataFrame(generated, columns=df.columns)


def generate_column_permutations(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    generated = {}
    for col in df.columns:
        values = df[col].to_numpy().copy()
        rng.shuffle(values)
        generated[col] = values
    return pd.DataFrame(generated, columns=df.columns)


def generate_numeric_smoothed_marginals(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    generated = {}
    n_rows = len(df)
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_rate = float(numeric.notna().mean())
        if numeric_rate >= 0.95 and numeric.nunique(dropna=True) > 10:
            values = numeric.dropna().to_numpy(dtype=float)
            if len(values) == 0:
                generated[col] = rng.choice(df[col].to_numpy(), size=n_rows, replace=True)
                continue
            sampled = rng.choice(values, size=n_rows, replace=True)
            std = float(np.nanstd(values))
            noise = rng.normal(loc=0.0, scale=0.03 * std if std > 0 else 0.0, size=n_rows)
            generated[col] = sampled + noise
        else:
            values = df[col].to_numpy()
            generated[col] = rng.choice(values, size=n_rows, replace=True)
    return pd.DataFrame(generated, columns=df.columns)


def synthetic_controls(df: pd.DataFrame, random_state: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    return {
        "independent_marginals": generate_independent_marginals(df, rng),
        "column_permutation": generate_column_permutations(df, rng),
        "numeric_smoothed_marginals": generate_numeric_smoothed_marginals(df, rng),
    }


def evaluate_detector(texts: list[str], labels: np.ndarray, max_features: int, random_state: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, test_idx = next(splitter.split(texts, labels))
    x_train = [texts[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    detector = make_detector(max_features)
    detector.fit(x_train, y_train)
    proba = detector.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))
    return {
        "auc": auc,
        "separability_auc": float(max(auc, 1.0 - auc)),
        "orientation_flipped": bool(auc < 0.5),
        "accuracy": float(accuracy_score(y_test, pred)),
        "detector": detector,
    }


def run_same_table_c2st(
    view_name: str,
    target_df: pd.DataFrame,
    args,
) -> tuple[list[DetectorResult], list[dict]]:
    results = []
    rows = []
    real_texts = linearize_rows(target_df, args.random_state)
    for generator_name, synthetic_df in synthetic_controls(target_df, args.random_state + 11).items():
        synthetic_texts = linearize_rows(synthetic_df, args.random_state + 17)
        texts = real_texts + synthetic_texts
        labels = np.array([0] * len(real_texts) + [1] * len(synthetic_texts))
        metrics = evaluate_detector(texts, labels, args.max_features, args.random_state)
        interpretation = (
            "Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a "
            "simple synthetic control generated from the same table. This does not prove the "
            "Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts."
        )
        results.append(
            DetectorResult(
                view=view_name,
                setup="same_table_c2st",
                generator=generator_name,
                auc=metrics["auc"],
                separability_auc=metrics["separability_auc"],
                accuracy=metrics["accuracy"],
                n_real=len(real_texts),
                n_synthetic=len(synthetic_texts),
                interpretation=interpretation,
            )
        )
        rows.append(
            {
                "view": view_name,
                "setup": "same_table_c2st",
                "generator": generator_name,
                "auc": metrics["auc"],
                "separability_auc": metrics["separability_auc"],
                "orientation_flipped": metrics["orientation_flipped"],
                "accuracy": metrics["accuracy"],
                "n_real": len(real_texts),
                "n_synthetic": len(synthetic_texts),
            }
        )
    return results, rows


def align_reference_to_target(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    # The paper's text baseline is table-agnostic, but a detector trained on one
    # schema can still overfit to column names. Prefixes created in build_views
    # intentionally keep schemas distinct; this function only trims empty columns.
    reference_df = reference_df.dropna(axis=1, how="all")
    target_df = target_df.dropna(axis=1, how="all")
    return reference_df, target_df


def run_cross_table_reference_detector(
    view_name: str,
    target_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    args,
) -> tuple[DetectorResult, pd.DataFrame, dict]:
    reference_df, target_df = align_reference_to_target(reference_df, target_df)
    control_frames = synthetic_controls(reference_df, args.random_state + 31)
    synthetic_ref = pd.concat(control_frames.values(), ignore_index=True)

    reference_texts = linearize_rows(reference_df, args.random_state + 37)
    synthetic_texts = linearize_rows(synthetic_ref, args.random_state + 41)
    labels = np.array([0] * len(reference_texts) + [1] * len(synthetic_texts))
    texts = reference_texts + synthetic_texts

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=args.random_state,
        stratify=labels,
    )

    detector = make_detector(args.max_features)
    detector.fit(x_train, y_train)

    holdout_proba = detector.predict_proba(x_holdout)[:, 1]
    holdout_pred = (holdout_proba >= 0.5).astype(int)
    holdout_auc = float(roc_auc_score(y_holdout, holdout_proba))
    orientation_flipped = bool(holdout_auc < 0.5)
    holdout_separability_auc = float(max(holdout_auc, 1.0 - holdout_auc))
    holdout_accuracy = float(accuracy_score(y_holdout, holdout_pred))

    target_texts = linearize_rows(target_df, args.random_state + 43)
    target_raw_proba = detector.predict_proba(target_texts)[:, 1]
    target_oriented_proba = 1.0 - target_raw_proba if orientation_flipped else target_raw_proba

    oriented_holdout_proba = 1.0 - holdout_proba if orientation_flipped else holdout_proba

    ref_real_holdout = np.asarray([p for p, y in zip(oriented_holdout_proba, y_holdout) if y == 0])
    ref_synth_holdout = np.asarray([p for p, y in zip(oriented_holdout_proba, y_holdout) if y == 1])
    summary = {
        "target_oriented_synthetic_probability_mean": float(np.mean(target_oriented_proba)),
        "target_oriented_synthetic_probability_median": float(np.median(target_oriented_proba)),
        "target_oriented_synthetic_probability_q10": float(np.quantile(target_oriented_proba, 0.10)),
        "target_oriented_synthetic_probability_q90": float(np.quantile(target_oriented_proba, 0.90)),
        "target_raw_synthetic_probability_mean": float(np.mean(target_raw_proba)),
        "target_raw_synthetic_probability_median": float(np.median(target_raw_proba)),
        "reference_real_holdout_mean": float(np.mean(ref_real_holdout)) if len(ref_real_holdout) else None,
        "reference_synthetic_holdout_mean": float(np.mean(ref_synth_holdout)) if len(ref_synth_holdout) else None,
        "holdout_auc": holdout_auc,
        "holdout_separability_auc": holdout_separability_auc,
        "orientation_flipped": orientation_flipped,
        "holdout_accuracy": holdout_accuracy,
    }

    scores_df = pd.DataFrame(
        {
            "view": view_name,
            "row_id": np.arange(len(target_raw_proba)),
            "raw_synthetic_probability": target_raw_proba,
            "oriented_synthetic_probability": target_oriented_proba,
            "orientation_flipped": orientation_flipped,
        }
    )

    interpretation = (
        "Cross-table detector inspired by arXiv:2412.13227. It is trained on a reference real "
        "passenger survey table and simple synthetic controls, then applied to the Kaggle table. "
        "Because the paper reports weak cross-table performance for the 3-gram logistic baseline "
        "(AUC about 0.58 under cross-table shift), this score is only a risk indicator, not proof "
        "of synthetic origin."
    )
    result = DetectorResult(
        view=view_name,
        setup="cross_table_reference_detector",
        generator="all_simple_controls",
        auc=holdout_auc,
        separability_auc=holdout_separability_auc,
        accuracy=holdout_accuracy,
        n_real=len(reference_texts),
        n_synthetic=len(synthetic_texts),
        interpretation=interpretation,
    )
    return result, scores_df, summary


def result_to_dict(result: DetectorResult) -> dict:
    return {
        "view": result.view,
        "setup": result.setup,
        "generator": result.generator,
        "auc": result.auc,
        "separability_auc": result.separability_auc,
        "accuracy": result.accuracy,
        "n_real": result.n_real,
        "n_synthetic": result.n_synthetic,
        "interpretation": result.interpretation,
    }


def write_markdown_report(output_dir: str, summary: dict):
    report_path = os.path.join(output_dir, "synthetic_detection_report.md")
    lines = [
        "# Synthetic Tabular Detection Audit",
        "",
        "This audit is inspired by Kindji et al., `Cross-table Synthetic Tabular Data Detection`, arXiv:2412.13227.",
        "",
        "Important limitation: the paper reports that cross-table synthetic tabular detection is challenging. "
        "For the simple character-trigram logistic-regression baseline, the reported cross-table AUC is about 0.58. "
        "Therefore this script provides risk indicators, not a proof that the Kaggle dataset is synthetic.",
        "",
        "## Inputs",
        "",
        f"- Target rows used: {summary['inputs']['target_rows_used']}",
        f"- Reference real rows used: {summary['inputs'].get('reference_rows_used')}",
        f"- Reference real path: `{summary['inputs'].get('reference_real_path')}`",
        "",
        "## Detector Results",
        "",
        "| View | Setup | Generator | AUC | Separability AUC | Accuracy | Interpretation |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for item in summary["detector_results"]:
        auc = "" if item["auc"] is None else f"{item['auc']:.4f}"
        separability_auc = "" if item["separability_auc"] is None else f"{item['separability_auc']:.4f}"
        accuracy = "" if item["accuracy"] is None else f"{item['accuracy']:.4f}"
        lines.append(
            f"| {item['view']} | {item['setup']} | {item['generator']} | {auc} | {separability_auc} | {accuracy} | "
            f"{item['interpretation']} |"
        )

    if summary.get("cross_table_scores"):
        lines.extend(["", "## Cross-table Target Scores", ""])
        for view_name, scores in summary["cross_table_scores"].items():
            lines.extend(
                [
                    f"### {view_name}",
                    "",
                    f"- Target mean oriented synthetic probability: {scores['target_oriented_synthetic_probability_mean']:.4f}",
                    f"- Target median oriented synthetic probability: {scores['target_oriented_synthetic_probability_median']:.4f}",
                    f"- Target mean raw synthetic probability: {scores['target_raw_synthetic_probability_mean']:.4f}",
                    f"- Holdout separability AUC: {scores['holdout_separability_auc']:.4f}",
                    f"- Orientation flipped: {scores['orientation_flipped']}",
                    f"- Reference real holdout mean: {scores['reference_real_holdout_mean']}",
                    f"- Reference synthetic holdout mean: {scores['reference_synthetic_holdout_mean']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## How to Use This Result",
            "",
            "A high score should be described as an additional warning signal, especially together with missing dataset provenance. "
            "It should not be written as definitive evidence that the Kaggle dataset is fake.",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    target_df = load_target_dataset(args)
    target_df = sample_rows(target_df, args.max_rows, args.random_state)
    target_views = build_views(target_df, args, "target")

    reference_df = None
    reference_views = {}
    if args.reference_real and os.path.exists(args.reference_real):
        reference_df = pd.read_csv(args.reference_real, encoding="utf-8-sig")
        reference_df.columns = [str(col).strip() for col in reference_df.columns]
        reference_df = sample_rows(reference_df, args.max_rows, args.random_state + 5)
        reference_views = build_views(reference_df, args, "reference")

    detector_results = []
    c2st_rows = []
    cross_table_scores = {}
    score_frames = []

    for view_name, target_view in target_views.items():
        same_results, same_rows = run_same_table_c2st(view_name, target_view, args)
        detector_results.extend(same_results)
        c2st_rows.extend(same_rows)

        if reference_views and view_name in reference_views:
            cross_result, scores_df, score_summary = run_cross_table_reference_detector(
                view_name,
                target_view,
                reference_views[view_name],
                args,
            )
            detector_results.append(cross_result)
            cross_table_scores[view_name] = score_summary
            score_frames.append(scores_df)

    pd.DataFrame(c2st_rows).to_csv(
        os.path.join(args.output_dir, "same_table_c2st_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if score_frames:
        pd.concat(score_frames, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "cross_table_target_row_scores.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "paper": {
            "title": "Cross-table Synthetic Tabular Data Detection",
            "arxiv": "https://arxiv.org/abs/2412.13227",
            "method_used": "Row linearization as <column>:<value> strings + character trigram logistic regression baseline.",
            "paper_reported_cross_table_baseline": {
                "model": "3grm-LReg",
                "auc": 0.58,
                "accuracy": 0.55,
                "note": "Reported for All Models vs Real, All Tables, Cross-table shift.",
            },
            "local_metric_note": (
                "separability_auc=max(AUC, 1-AUC). It measures whether two samples are separable "
                "even when the detector's positive-class orientation is reversed."
            ),
        },
        "inputs": {
            "target_rows_used": int(len(target_df)),
            "target_columns": list(target_df.columns),
            "reference_real_path": args.reference_real,
            "reference_rows_used": int(len(reference_df)) if reference_df is not None else None,
        },
        "detector_results": [result_to_dict(result) for result in detector_results],
        "cross_table_scores": cross_table_scores,
        "conclusion_guardrail": (
            "Synthetic-tabular detection is not an authenticity proof. Use the output as "
            "supporting evidence together with provenance checks, leakage checks, and dataset documentation."
        ),
    }
    with open(os.path.join(args.output_dir, "synthetic_detection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown_report(args.output_dir, summary)
    print(f"Audit complete. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
