import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency, ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


TARGET_COLUMN = "satisfaction"
INDEX_COLUMNS = ["Unnamed: 0"]
ID_COLUMNS = ["id"]
RATING_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]


@dataclass
class Finding:
    check: str
    severity: str
    value: object
    interpretation: str


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(train_path: str, test_path: str | None) -> pd.DataFrame:
    frames = []

    train_df = pd.read_csv(train_path)
    train_df["__split"] = "train"
    frames.append(train_df)

    if test_path:
        test_df = pd.read_csv(test_path)
        test_df["__split"] = "test"
        frames.append(test_df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(col).strip() for col in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} was not found.")

    return df


def add_finding(findings, check, severity, value, interpretation):
    findings.append(Finding(check, severity, value, interpretation))


def check_schema_and_cleanliness(df: pd.DataFrame, findings: list[Finding]):
    missing_counts = df.isna().sum()
    missing_columns = missing_counts[missing_counts > 0].to_dict()

    add_finding(
        findings,
        "missing_values",
        "info" if missing_columns else "low",
        missing_columns,
        "Very low missingness is not proof of synthetic data, but a nearly clean public dataset "
        "should be treated as curated rather than raw operational data.",
    )

    rating_ranges = {}
    for col in RATING_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        rating_ranges[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "n_unique": int(values.nunique()),
            "unique_values": sorted(values.unique().tolist()),
        }

    invalid_rating_columns = {
        col: stats
        for col, stats in rating_ranges.items()
        if stats["min"] < 0 or stats["max"] > 5
    }

    add_finding(
        findings,
        "rating_value_ranges",
        "high" if invalid_rating_columns else "low",
        invalid_rating_columns or rating_ranges,
        "Ratings outside the expected 0-5 range would be a strong integrity issue. "
        "A clean 0-5 range is expected for survey data and does not prove authenticity.",
    )


def check_identifier_artifacts(df: pd.DataFrame, findings: list[Finding]):
    if "Unnamed: 0" in df.columns:
        sequential_by_split = {}
        for split_name, split_df in df.groupby("__split"):
            index_values = pd.to_numeric(split_df["Unnamed: 0"], errors="coerce")
            expected = np.arange(len(split_df))
            sequential_by_split[split_name] = bool(
                len(index_values) == len(expected)
                and np.array_equal(index_values.to_numpy(), expected)
            )

        add_finding(
            findings,
            "csv_export_index_artifact",
            "medium" if any(sequential_by_split.values()) else "low",
            sequential_by_split,
            "A sequential 'Unnamed: 0' column is a CSV export artifact. It does not prove fake data, "
            "but it shows that the Kaggle file is a processed export, not a raw source table.",
        )

    if "id" in df.columns:
        duplicate_id_count = int(df["id"].duplicated().sum())
        id_monotonic_by_split = {}
        for split_name, split_df in df.groupby("__split"):
            id_values = pd.to_numeric(split_df["id"], errors="coerce")
            id_monotonic_by_split[split_name] = bool(id_values.is_monotonic_increasing)

        add_finding(
            findings,
            "id_integrity",
            "high" if duplicate_id_count else "low",
            {
                "duplicate_id_count": duplicate_id_count,
                "id_monotonic_by_split": id_monotonic_by_split,
            },
            "Duplicate IDs across train/test would be suspicious. Non-monotonic IDs are common "
            "after random splitting and are not evidence of fabrication.",
        )


def check_duplicates(df: pd.DataFrame, findings: list[Finding], output_dir: str):
    ignored_columns = set(INDEX_COLUMNS + ID_COLUMNS + ["__split"])
    content_columns = [col for col in df.columns if col not in ignored_columns]
    duplicate_mask = df.duplicated(subset=content_columns, keep=False)
    duplicate_rows = df.loc[duplicate_mask, ["__split", *content_columns]]

    duplicate_profile_count = int(df.duplicated(subset=content_columns).sum())
    duplicate_profile_rate = float(duplicate_profile_count / len(df))

    if len(duplicate_rows) > 0:
        duplicate_rows.head(200).to_csv(
            os.path.join(output_dir, "duplicate_profiles_sample.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    add_finding(
        findings,
        "duplicate_content_profiles",
        "medium" if duplicate_profile_rate > 0.01 else "low",
        {
            "duplicate_profile_count": duplicate_profile_count,
            "duplicate_profile_rate": duplicate_profile_rate,
            "sample_file": "duplicate_profiles_sample.csv" if len(duplicate_rows) else None,
        },
        "Many identical passenger profiles can indicate templating or repeated records, but survey "
        "data with low-cardinality answers can naturally contain repeated profiles.",
    )


def check_train_test_distribution_shift(df: pd.DataFrame, findings: list[Finding], output_dir: str):
    if df["__split"].nunique() < 2:
        return

    train_df = df[df["__split"] == "train"]
    test_df = df[df["__split"] == "test"]
    rows = []

    for col in df.columns:
        if col in {"__split", *INDEX_COLUMNS}:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            train_values = pd.to_numeric(train_df[col], errors="coerce").dropna()
            test_values = pd.to_numeric(test_df[col], errors="coerce").dropna()
            if len(train_values) == 0 or len(test_values) == 0:
                continue
            statistic, p_value = ks_2samp(train_values, test_values)
            rows.append({
                "feature": col,
                "test": "ks_2samp",
                "statistic": statistic,
                "p_value": p_value,
            })
        else:
            contingency = pd.crosstab(df["__split"], df[col])
            if contingency.shape[1] < 2:
                continue
            statistic, p_value, _, _ = chi2_contingency(contingency)
            rows.append({
                "feature": col,
                "test": "chi2_contingency",
                "statistic": statistic,
                "p_value": p_value,
            })

    drift_df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    drift_df.to_csv(
        os.path.join(output_dir, "train_test_distribution_tests.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    very_small_p_values = int((drift_df["p_value"] < 1e-6).sum()) if len(drift_df) else 0

    add_finding(
        findings,
        "train_test_distribution_shift",
        "medium" if very_small_p_values else "low",
        {
            "features_with_p_lt_1e-6": very_small_p_values,
            "details_file": "train_test_distribution_tests.csv",
        },
        "A public train/test split should usually be distributionally similar. Strong drift can "
        "indicate non-random splitting, post-processing, or mixed sources, not necessarily fake data.",
    )


def check_target_predictability(df: pd.DataFrame, findings: list[Finding], output_dir: str):
    feature_df = df.drop(columns=[TARGET_COLUMN, "__split", *[c for c in INDEX_COLUMNS + ID_COLUMNS if c in df.columns]])
    y = df[TARGET_COLUMN].astype(str)

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]), categorical_cols),
        ],
        sparse_threshold=0.0,
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted_labels = cross_val_predict(model, feature_df, y, cv=cv, n_jobs=1)
    predicted_proba = cross_val_predict(
        model,
        feature_df,
        y,
        cv=cv,
        n_jobs=1,
        method="predict_proba",
    )

    positive_class = sorted(y.unique())[-1]
    positive_index = list(model.named_steps["model"].classes_) if False else None
    classes = sorted(y.unique())
    proba_positive_index = classes.index(positive_class)

    f1_macro = float(f1_score(y, predicted_labels, average="macro"))
    roc_auc = float(roc_auc_score((y == positive_class).astype(int), predicted_proba[:, proba_positive_index]))

    add_finding(
        findings,
        "target_predictability",
        "medium" if f1_macro > 0.95 else "low",
        {
            "cross_validated_f1_macro": f1_macro,
            "cross_validated_roc_auc": roc_auc,
        },
        "Extremely high predictability could indicate leakage or rule-generated labels. The threshold "
        "here is intentionally conservative; high quality alone does not prove synthetic data.",
    )

    encoded = preprocessor.fit_transform(feature_df)
    feature_names = preprocessor.get_feature_names_out()
    y_binary = (y == positive_class).astype(int)
    mi = mutual_info_classif(encoded, y_binary, discrete_features=False, random_state=42)
    mi_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "mutual_information": mi,
    }).sort_values("mutual_information", ascending=False)
    mi_df.head(100).to_csv(
        os.path.join(output_dir, "top_mutual_information_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def check_low_order_rating_templates(df: pd.DataFrame, findings: list[Finding], output_dir: str):
    available_rating_cols = [col for col in RATING_COLUMNS if col in df.columns]
    if not available_rating_cols:
        return

    rating_profiles = df[available_rating_cols].astype(str)
    profile_counts = rating_profiles.value_counts().reset_index(name="count")
    top_count = int(profile_counts["count"].iloc[0])
    top_share = float(top_count / len(df))
    repeated_share = float(profile_counts.loc[profile_counts["count"] > 1, "count"].sum() / len(df))

    profile_counts.head(100).to_csv(
        os.path.join(output_dir, "top_rating_profiles.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    add_finding(
        findings,
        "rating_template_repetition",
        "medium" if top_share > 0.01 or repeated_share > 0.5 else "low",
        {
            "top_profile_count": top_count,
            "top_profile_share": top_share,
            "repeated_profile_share": repeated_share,
            "details_file": "top_rating_profiles.csv",
        },
        "High repetition of full rating profiles can be a sign of templating. However, this survey "
        "contains many low-cardinality rating fields, so repeated profiles are expected to some degree.",
    )


def check_numeric_digit_patterns(df: pd.DataFrame, findings: list[Finding], output_dir: str):
    numeric_cols = [
        col for col in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
        if col in df.columns
    ]
    rows = []

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        if len(values) == 0:
            continue

        last_digits = values.abs() % 10
        observed = last_digits.value_counts().reindex(range(10), fill_value=0).to_numpy()
        expected = np.full(10, observed.sum() / 10)
        statistic, p_value = chisquare(observed, expected)

        rows.append({
            "feature": col,
            "chi_square_statistic": statistic,
            "p_value_uniform_last_digit": p_value,
            "zero_last_digit_share": float((last_digits == 0).mean()),
        })

    digit_df = pd.DataFrame(rows)
    digit_df.to_csv(
        os.path.join(output_dir, "numeric_digit_pattern_tests.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    suspicious_rows = digit_df[
        (digit_df["p_value_uniform_last_digit"] < 1e-12)
        & (digit_df["zero_last_digit_share"] > 0.25)
    ] if len(digit_df) else pd.DataFrame()

    add_finding(
        findings,
        "numeric_digit_patterns",
        "medium" if len(suspicious_rows) else "low",
        {
            "suspicious_feature_count": int(len(suspicious_rows)),
            "details_file": "numeric_digit_pattern_tests.csv",
        },
        "Unnatural digit patterns may indicate rounding or generation. For delays and distances, "
        "rounding/heaping can also occur in real operational systems.",
    )


def write_report(findings: list[Finding], output_dir: str, train_path: str, test_path: str | None):
    findings_df = pd.DataFrame([finding.__dict__ for finding in findings])
    findings_df.to_csv(
        os.path.join(output_dir, "authenticity_findings.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    severity_score = {"low": 0, "info": 0, "medium": 1, "high": 2}
    score = sum(severity_score.get(finding.severity, 0) for finding in findings)

    if score >= 5:
        verdict = "high_suspicion"
    elif score >= 2:
        verdict = "some_suspicion"
    else:
        verdict = "no_strong_suspicion_from_these_tests"

    summary = {
        "train_path": train_path,
        "test_path": test_path,
        "verdict": verdict,
        "suspicion_score": score,
        "important_limitation": (
            "These tests cannot prove that the dataset is fake. They can only identify "
            "internal inconsistencies, processing artifacts, distribution anomalies, and "
            "patterns that would justify further source verification."
        ),
        "findings": [finding.__dict__ for finding in findings],
    }

    with open(os.path.join(output_dir, "authenticity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    markdown_lines = [
        "# Airline Passenger Satisfaction Dataset Authenticity Audit",
        "",
        f"Train file: `{train_path}`",
        f"Test file: `{test_path}`" if test_path else "Test file: not provided",
        "",
        f"Verdict: **{verdict}**",
        f"Suspicion score: **{score}**",
        "",
        "Important limitation: these checks cannot strictly prove that the dataset is fake. "
        "They only provide evidence of processing artifacts, inconsistencies, or suspicious "
        "statistical patterns.",
        "",
        "## Findings",
        "",
    ]

    for finding in findings:
        markdown_lines.extend([
            f"### {finding.check}",
            f"- Severity: `{finding.severity}`",
            f"- Value: `{finding.value}`",
            f"- Interpretation: {finding.interpretation}",
            "",
        ])

    with open(os.path.join(output_dir, "authenticity_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Kaggle Airline Passenger Satisfaction dataset for signs of synthetic "
            "or heavily processed data. This script cannot prove that a dataset is fake."
        )
    )
    parser.add_argument("--train", default="train.csv", help="Path to Kaggle train.csv")
    parser.add_argument("--test", default="test.csv", help="Path to Kaggle test.csv")
    parser.add_argument("--output-dir", default="airline_authenticity_audit")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    test_path = args.test if args.test and os.path.exists(args.test) else None

    df = load_dataset(args.train, test_path)
    findings = []

    check_schema_and_cleanliness(df, findings)
    check_identifier_artifacts(df, findings)
    check_duplicates(df, findings, args.output_dir)
    check_train_test_distribution_shift(df, findings, args.output_dir)
    check_low_order_rating_templates(df, findings, args.output_dir)
    check_numeric_digit_patterns(df, findings, args.output_dir)
    check_target_predictability(df, findings, args.output_dir)
    write_report(findings, args.output_dir, args.train, test_path)

    print(f"Audit completed. Report: {os.path.join(args.output_dir, 'authenticity_report.md')}")


if __name__ == "__main__":
    main()
