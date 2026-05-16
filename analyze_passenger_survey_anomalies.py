from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "balanced_passenger_survey_dataset"
FULL_PATH = DATA_DIR / "passenger_survey_full.csv"
BALANCED_PATH = DATA_DIR / "passenger_survey_balanced.csv"
REPORT_JSON_PATH = DATA_DIR / "passenger_survey_balanced_report.json"
OUT_DIR = ROOT / "passenger_survey_anomaly_audit"
TARGET = "liked"


EXPECTED_BINARY = {0, 1}
EXPECTED_RATING_MIN = 1
EXPECTED_RATING_MAX = 5
TIME_COLUMNS = {"arrival_lead_time", "connection_wait_time"}
NON_RATING_NUMERIC = TIME_COLUMNS | {TARGET}


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def load_report() -> dict:
    if REPORT_JSON_PATH.exists():
        return json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
    return {}


def dataset_overview(name: str, df: pd.DataFrame) -> dict:
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows_excluding_target": int(
            df.drop(columns=[TARGET]).duplicated().sum()
        ),
        "target_0": int((df[TARGET] == 0).sum()),
        "target_1": int((df[TARGET] == 1).sum()),
    }


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    indicator_cols = [c for c in numeric_cols if c.endswith("_is_applicable")]
    rating_cols = [
        c
        for c in numeric_cols
        if c not in indicator_cols and c not in NON_RATING_NUMERIC
    ]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, rating_cols, indicator_cols, categorical_cols


def numeric_range_issues(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    _, rating_cols, indicator_cols, _ = split_columns(df)
    rows = []

    for col in rating_cols:
        s = df[col]
        invalid = s.notna() & ((s < EXPECTED_RATING_MIN) | (s > EXPECTED_RATING_MAX))
        non_integer = s.notna() & (np.abs(s - np.round(s)) > 1e-9)
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "kind": "rating",
                "min": float(s.min()),
                "max": float(s.max()),
                "invalid_outside_1_5": int(invalid.sum()),
                "non_integer_count": int(non_integer.sum()),
                "unique_count": int(s.nunique(dropna=True)),
            }
        )

    for col in indicator_cols + [TARGET]:
        s = df[col]
        values = set(pd.Series(s.dropna().unique()).astype(int).tolist())
        invalid = sorted(values - EXPECTED_BINARY)
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "kind": "binary",
                "min": float(s.min()),
                "max": float(s.max()),
                "invalid_values": ",".join(map(str, invalid)),
                "unique_count": int(s.nunique(dropna=True)),
            }
        )

    for col in TIME_COLUMNS & set(df.columns):
        s = df[col]
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "kind": "time_seconds",
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.quantile(0.50)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "negative_count": int((s < 0).sum()),
                "over_24h_count": int((s > 24 * 3600).sum()),
                "unique_count": int(s.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def column_quality(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        top_share = float(vc.iloc[0] / len(df)) if len(vc) else np.nan
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_share": float(df[col].isna().mean()),
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_value": str(vc.index[0]) if len(vc) else "",
                "top_value_count": int(vc.iloc[0]) if len(vc) else 0,
                "top_value_share": top_share,
            }
        )
    return pd.DataFrame(rows)


def categorical_rare_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    _, _, _, categorical_cols = split_columns(df)
    rows = []
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        for value, count in vc.items():
            share = count / len(df)
            if count < 30 or share < 0.001:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "column": col,
                        "value": str(value),
                        "count": int(count),
                        "share": float(share),
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "column", "count"])


def contradictory_feature_duplicates(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    features = [c for c in df.columns if c != TARGET]
    grouped = (
        df.groupby(features, dropna=False)[TARGET]
        .agg(["count", "nunique", "mean"])
        .reset_index()
    )
    repeated = grouped[grouped["count"] > 1].copy()
    bad = repeated[repeated["nunique"] > 1].copy()
    pure = repeated[repeated["nunique"] == 1].copy()
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "unique_feature_profiles": [int(len(grouped))],
            "repeated_feature_profiles": [int(len(repeated))],
            "rows_in_repeated_feature_profiles": [int(repeated["count"].sum()) if len(repeated) else 0],
            "pure_repeated_feature_profiles": [int(len(pure))],
            "rows_in_pure_repeated_feature_profiles": [int(pure["count"].sum()) if len(pure) else 0],
            "contradictory_feature_profiles": [int(len(bad))],
            "rows_in_contradictory_feature_profiles": [int(bad["count"].sum()) if len(bad) else 0],
            "max_repeated_profile_count": [int(repeated["count"].max()) if len(repeated) else 0],
        }
    )


def duplicate_profile_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    features = [c for c in df.columns if c != TARGET]
    grouped = (
        df.groupby(features, dropna=False)[TARGET]
        .agg(count="count", positives="sum", target_mean="mean")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    rows = []
    for i, (_, row) in enumerate(grouped.head(20).iterrows(), start=1):
        rows.append(
            {
                "dataset": dataset_name,
                "rank": i,
                "count": int(row["count"]),
                "positives": int(row["positives"]),
                "target_mean": float(row["target_mean"]),
                "profile_preview": json.dumps(
                    {col: row[col] for col in features[:15]},
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )
    return pd.DataFrame(rows)


def target_association(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    numeric_cols, _, _, categorical_cols = split_columns(df)
    rows = []
    y = df[TARGET]

    for col in numeric_cols:
        if col == TARGET:
            continue
        s = df[col]
        corr = s.corr(y)
        grouped = y.groupby(s).agg(["count", "mean"])
        if len(grouped) > 1:
            gap = float(grouped["mean"].max() - grouped["mean"].min())
        else:
            gap = 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "type": "numeric",
                "pearson_corr_with_target": float(corr) if pd.notna(corr) else np.nan,
                "max_target_rate_gap_by_value": gap,
                "unique_count": int(s.nunique(dropna=True)),
            }
        )

    for col in categorical_cols:
        grouped = y.groupby(df[col], dropna=False).agg(["count", "mean"])
        if len(grouped) > 1:
            # Ignore tiny groups when computing practical target gap.
            large = grouped[grouped["count"] >= 100]
            source = large if len(large) >= 2 else grouped
            gap = float(source["mean"].max() - source["mean"].min())
        else:
            gap = 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "type": "categorical",
                "pearson_corr_with_target": np.nan,
                "max_target_rate_gap_by_value": gap,
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )

    out = pd.DataFrame(rows)
    out["abs_corr"] = out["pearson_corr_with_target"].abs()
    return out.sort_values(
        ["abs_corr", "max_target_rate_gap_by_value"], ascending=False
    ).drop(columns=["abs_corr"])


def process_consistency(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    if {"process", "connection"}.issubset(df.columns):
        tab = pd.crosstab(df["process"], df["connection"])
        for process, data in tab.iterrows():
            for connection, count in data.items():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "check": "process_vs_connection",
                        "left": process,
                        "right": connection,
                        "count": int(count),
                    }
                )

    if {"process", "disembarkation_method_used"}.issubset(df.columns):
        tab = pd.crosstab(df["process"], df["disembarkation_method_used"])
        for process, data in tab.iterrows():
            for value, count in data.items():
                if count:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "check": "process_vs_disembarkation_method_used",
                            "left": process,
                            "right": value,
                            "count": int(count),
                        }
                    )
    return pd.DataFrame(rows)


def balanced_vs_full_positive_shift(full: pd.DataFrame, balanced: pd.DataFrame) -> pd.DataFrame:
    full_pos = full[full[TARGET] == 1]
    bal_pos = balanced[balanced[TARGET] == 1]
    _, rating_cols, indicator_cols, categorical_cols = split_columns(full)
    rows = []

    for col in rating_cols + list(TIME_COLUMNS & set(full.columns)) + indicator_cols:
        if col not in bal_pos.columns:
            continue
        rows.append(
            {
                "column": col,
                "type": "numeric",
                "full_positive_mean": float(full_pos[col].mean()),
                "balanced_positive_mean": float(bal_pos[col].mean()),
                "absolute_mean_shift": float(abs(full_pos[col].mean() - bal_pos[col].mean())),
                "full_positive_p95": float(full_pos[col].quantile(0.95)),
                "balanced_positive_p95": float(bal_pos[col].quantile(0.95)),
            }
        )

    for col in categorical_cols:
        full_dist = full_pos[col].value_counts(normalize=True, dropna=False)
        bal_dist = bal_pos[col].value_counts(normalize=True, dropna=False)
        idx = full_dist.index.union(bal_dist.index)
        tvd = 0.5 * (full_dist.reindex(idx, fill_value=0) - bal_dist.reindex(idx, fill_value=0)).abs().sum()
        rows.append(
            {
                "column": col,
                "type": "categorical",
                "full_positive_mean": np.nan,
                "balanced_positive_mean": np.nan,
                "absolute_mean_shift": float(tvd),
                "full_positive_p95": np.nan,
                "balanced_positive_p95": np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("absolute_mean_shift", ascending=False)


def impossible_applicability_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for ind in [c for c in df.columns if c.endswith("_is_applicable")]:
        base = ind.removesuffix("_is_applicable")
        if base not in df.columns:
            continue
        base_values_when_not_applicable = df.loc[df[ind] == 0, base].value_counts(dropna=False)
        if len(base_values_when_not_applicable):
            top_value = base_values_when_not_applicable.index[0]
            rows.append(
                {
                    "dataset": dataset_name,
                    "indicator": ind,
                    "base_column": base,
                    "not_applicable_rows": int((df[ind] == 0).sum()),
                    "distinct_imputed_values_when_not_applicable": int(base_values_when_not_applicable.size),
                    "top_imputed_value_when_not_applicable": str(top_value),
                    "top_imputed_value_share": float(base_values_when_not_applicable.iloc[0] / base_values_when_not_applicable.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["not_applicable_rows", "distinct_imputed_values_when_not_applicable"],
        ascending=False,
    )


def write_markdown(
    overview: pd.DataFrame,
    range_issues: pd.DataFrame,
    quality: pd.DataFrame,
    rare: pd.DataFrame,
    assoc: pd.DataFrame,
    shift: pd.DataFrame,
    dup_summary: pd.DataFrame,
) -> None:
    major_range = range_issues[
        (
            (range_issues.get("invalid_outside_1_5", 0).fillna(0) > 0)
            | (range_issues.get("negative_count", 0).fillna(0) > 0)
            | (range_issues.get("over_24h_count", 0).fillna(0) > 0)
            | (range_issues.get("invalid_values", "").fillna("") != "")
        )
    ]
    quasi_constant = quality[(quality["unique_count"] <= 1) | (quality["top_value_share"] >= 0.99)]
    high_assoc = assoc[
        (assoc["pearson_corr_with_target"].abs() >= 0.55)
        | (assoc["max_target_rate_gap_by_value"] >= 0.75)
    ].head(20)

    lines = [
        "# Passenger Survey Anomaly Audit",
        "",
        "## Overview",
        overview.to_markdown(index=False),
        "",
        "## Main Findings",
        f"- Range/binary/time violations found: {len(major_range)}.",
        f"- Columns with one value or >=99% dominant value: {len(quasi_constant)}.",
        f"- Rare categorical values listed: {len(rare)}.",
        f"- High target-association columns flagged: {len(high_assoc)}.",
        f"- Largest positive-sample shift after balancing: {shift['absolute_mean_shift'].max():.6f}.",
        "",
        "## Range, Binary, And Time Issues",
        major_range.head(30).to_markdown(index=False) if len(major_range) else "No hard range/binary/time violations.",
        "",
        "## Quasi-Constant Columns",
        quasi_constant.sort_values("top_value_share", ascending=False).head(30).to_markdown(index=False)
        if len(quasi_constant)
        else "No quasi-constant columns by the selected threshold.",
        "",
        "## Strongest Target Associations",
        high_assoc.to_markdown(index=False) if len(high_assoc) else "No columns crossed the high-association threshold.",
        "",
        "## Largest Shifts From Full Positive Class To Balanced Positive Subsample",
        shift.head(20).to_markdown(index=False),
        "",
        "## Most Repeated Feature Profiles",
        dup_summary.head(20).to_markdown(index=False),
        "",
        "Detailed CSV files are in `passenger_survey_anomaly_audit/`.",
        "",
    ]
    (OUT_DIR / "anomaly_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    full = pd.read_csv(FULL_PATH, low_memory=False)
    balanced = pd.read_csv(BALANCED_PATH, low_memory=False)
    report = load_report()

    overview = pd.DataFrame(
        [
            dataset_overview("full", full),
            dataset_overview("balanced", balanced),
        ]
    )
    write_csv(overview, "overview.csv")

    range_issues = pd.concat(
        [
            numeric_range_issues(full, "full"),
            numeric_range_issues(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(range_issues, "numeric_range_binary_time_checks.csv")

    quality = pd.concat(
        [
            column_quality(full, "full"),
            column_quality(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(quality, "column_quality.csv")

    rare = pd.concat(
        [
            categorical_rare_values(full, "full"),
            categorical_rare_values(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(rare, "rare_categorical_values.csv")

    assoc = pd.concat(
        [
            target_association(full, "full"),
            target_association(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(assoc, "target_association_screen.csv")

    consistency = pd.concat(
        [
            process_consistency(full, "full"),
            process_consistency(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(consistency, "process_consistency_tables.csv")

    applicability = pd.concat(
        [
            impossible_applicability_values(full, "full"),
            impossible_applicability_values(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(applicability, "applicability_imputed_value_patterns.csv")

    shift = balanced_vs_full_positive_shift(full, balanced)
    write_csv(shift, "balanced_vs_full_positive_shift.csv")

    duplicates = pd.concat(
        [
            contradictory_feature_duplicates(full, "full"),
            contradictory_feature_duplicates(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(duplicates, "contradictory_duplicate_summary.csv")

    dup_summary = pd.concat(
        [
            duplicate_profile_summary(full, "full"),
            duplicate_profile_summary(balanced, "balanced"),
        ],
        ignore_index=True,
    )
    write_csv(dup_summary, "most_repeated_feature_profiles.csv")

    metadata = {
        "source_report_random_state": report.get("random_state"),
        "source_report_balancing": report.get("balancing"),
        "dropped_columns": report.get("dropped_columns", []),
    }
    (OUT_DIR / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown(overview, range_issues, quality, rare, assoc, shift, dup_summary)
    print(f"Audit written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
