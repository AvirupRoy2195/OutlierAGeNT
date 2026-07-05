"""Exploratory data analysis tools exposed to Google ADK."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from .data_io import (
    load_table,
    numeric_feature_columns,
    parse_columns,
    records_preview,
    resolve_output_path,
    sanitize_json,
    write_json_report,
)


def generate_eda_report(
    file_path: str,
    columns: Optional[str] = None,
    target_column: Optional[str] = None,
    sheet_name: Optional[str] = None,
    report_path: Optional[str] = None,
    max_lag: int = 20,
    top_correlation_pairs: int = 25,
) -> dict[str, Any]:
    """Generate an EDA report with summaries, moments, correlations, and ACF.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to analyze.
        columns: Optional comma-separated columns. Leave blank for all columns.
        target_column: Optional Y/target column for target relationship summaries.
        sheet_name: Optional Excel sheet name.
        report_path: Optional JSON report path. Defaults beside the input file.
        max_lag: Maximum autocorrelation lag to compute for numeric columns.
        top_correlation_pairs: Number of strongest numeric correlation pairs to return.

    Returns:
        Structured EDA report with robust statistics and saved report path.
    """
    df, path = load_table(file_path, sheet_name=sheet_name)
    selected_columns = parse_columns(columns, df)
    selected = df[selected_columns].copy()
    numeric_columns = numeric_feature_columns(selected)
    categorical_columns = [
        column
        for column in selected.columns
        if column not in numeric_columns and not _looks_datetime(selected[column])
    ]
    datetime_columns = [
        column for column in selected.columns if _looks_datetime(selected[column])
    ]

    numeric_summary = {
        column: _numeric_summary(selected[column])
        for column in numeric_columns
    }
    categorical_summary = {
        column: _categorical_summary(selected[column])
        for column in categorical_columns
    }
    datetime_summary = {
        column: _datetime_summary(selected[column])
        for column in datetime_columns
    }
    correlations = _correlation_summary(
        selected,
        numeric_columns,
        top_n=top_correlation_pairs,
    )
    acf = {
        column: _acf(selected[column], max_lag=max_lag)
        for column in numeric_columns
    }
    target_summary = _target_summary(selected, target_column, numeric_columns)

    report_output = resolve_output_path(report_path, path, "_eda_report.json")
    report = {
        "status": "success",
        "file_path": str(path),
        "report_path": str(report_output),
        "row_count": int(len(selected)),
        "column_count": int(len(selected.columns)),
        "selected_columns": list(selected.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "datetime_summary": datetime_summary,
        "correlations": correlations,
        "autocorrelation": {
            "note": "ACF uses the current row order. Sort by time before using ACF for time-series interpretation.",
            "max_lag": int(max_lag),
            "columns": acf,
        },
        "target_summary": target_summary,
        "preview_records": records_preview(selected),
    }
    write_json_report(report, report_output)
    return sanitize_json(report)


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {
            "count": 0,
            "error": "No numeric values available.",
        }

    q1 = float(numeric.quantile(0.25))
    q2 = float(numeric.quantile(0.50))
    q3 = float(numeric.quantile(0.75))
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    iqr = q3 - q1
    mean_value = float(numeric.mean())
    centered = numeric - mean_value
    mad = float((numeric - q2).abs().median())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    robust_scale = mad * 1.4826

    return {
        "count": int(numeric.count()),
        "five_point_summary": {
            "min": min_value,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "max": max_value,
        },
        "mean": mean_value,
        "variance": float(numeric.var(ddof=1)) if len(numeric) > 1 else 0.0,
        "standard_deviation": std,
        "coefficient_of_variation": _safe_divide(std, mean_value),
        "moments": {
            "second_central_moment": float((centered**2).mean()),
            "third_central_moment": float((centered**3).mean()),
            "fourth_central_moment": float((centered**4).mean()),
            "skewness": float(numeric.skew()) if len(numeric) > 2 else None,
            "excess_kurtosis": float(numeric.kurt()) if len(numeric) > 3 else None,
        },
        "robust_statistics": {
            "iqr": iqr,
            "mad": mad,
            "mad_normalized": robust_scale,
            "robust_cv": _safe_divide(robust_scale, q2),
            "trimmed_mean_10_percent": _trimmed_mean(numeric, proportion=0.10),
            "winsorized_mean_10_percent": _winsorized_mean(numeric, proportion=0.10),
            "p05": float(numeric.quantile(0.05)),
            "p95": float(numeric.quantile(0.95)),
        },
    }


def _categorical_summary(series: pd.Series) -> dict[str, Any]:
    observed = series.dropna()
    value_counts = observed.astype(str).value_counts().head(10)
    probabilities = value_counts / value_counts.sum() if value_counts.sum() else value_counts
    entropy = -float(sum(p * math.log2(p) for p in probabilities if p > 0))
    return {
        "count": int(observed.count()),
        "unique_count": int(observed.nunique()),
        "mode": None if observed.mode().empty else observed.mode().iloc[0],
        "top_values": value_counts.to_dict(),
        "entropy_top_10": entropy,
    }


def _datetime_summary(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return {
            "count": 0,
            "error": "No parseable datetime values available.",
        }
    return {
        "count": int(parsed.count()),
        "min": parsed.min(),
        "max": parsed.max(),
        "range_days": float((parsed.max() - parsed.min()).total_seconds() / 86400),
    }


def _correlation_summary(
    df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    top_n: int,
) -> dict[str, Any]:
    if len(numeric_columns) < 2:
        return {
            "pearson": {},
            "spearman": {},
            "kendall": {},
            "top_absolute_pearson_pairs": [],
        }

    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    kendall = numeric.corr(method="kendall")
    pairs = []
    for left_index, left in enumerate(numeric_columns):
        for right in numeric_columns[left_index + 1 :]:
            value = pearson.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": float(value),
                    "absolute_pearson": abs(float(value)),
                }
            )
    pairs.sort(key=lambda item: item["absolute_pearson"], reverse=True)
    return {
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
        "kendall": kendall.to_dict(),
        "top_absolute_pearson_pairs": pairs[:top_n],
    }


def _acf(series: pd.Series, *, max_lag: int) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    max_lag = max(0, int(max_lag))
    if len(numeric) < 3 or max_lag == 0:
        return {"count": int(len(numeric)), "lags": {}}

    usable_max_lag = min(max_lag, len(numeric) - 2)
    return {
        "count": int(len(numeric)),
        "lags": {
            str(lag): float(numeric.autocorr(lag=lag))
            for lag in range(1, usable_max_lag + 1)
        },
    }


def _target_summary(
    df: pd.DataFrame,
    target_column: Optional[str],
    numeric_columns: list[str],
) -> dict[str, Any]:
    if not target_column:
        return {"target_column": None}
    if target_column not in df.columns:
        return {
            "target_column": target_column,
            "status": "error",
            "error_message": "Target column is not present in the selected data.",
        }

    target_numeric = pd.to_numeric(df[target_column], errors="coerce")
    target_is_numeric = df[target_column].notna().any() and target_numeric[
        df[target_column].notna()
    ].notna().all()
    if not target_is_numeric:
        return {
            "target_column": target_column,
            "target_type": "categorical",
            "class_balance": df[target_column].astype(str).value_counts().to_dict(),
        }

    relationships = {}
    for column in numeric_columns:
        if column == target_column:
            continue
        feature = pd.to_numeric(df[column], errors="coerce")
        relationships[column] = {
            "pearson_with_target": feature.corr(target_numeric, method="pearson"),
            "spearman_with_target": feature.corr(target_numeric, method="spearman"),
        }
    return {
        "target_column": target_column,
        "target_type": "numeric",
        "numeric_relationships": relationships,
    }

def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    observed = series.dropna()
    if observed.empty:
        return False
    sample = observed.astype(str).head(50)
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def _trimmed_mean(series: pd.Series, *, proportion: float) -> Optional[float]:
    if series.empty:
        return None
    lower = series.quantile(proportion)
    upper = series.quantile(1 - proportion)
    trimmed = series[(series >= lower) & (series <= upper)]
    if trimmed.empty:
        return None
    return float(trimmed.mean())


def _winsorized_mean(series: pd.Series, *, proportion: float) -> Optional[float]:
    if series.empty:
        return None
    lower = series.quantile(proportion)
    upper = series.quantile(1 - proportion)
    return float(series.clip(lower=lower, upper=upper).mean())


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)
