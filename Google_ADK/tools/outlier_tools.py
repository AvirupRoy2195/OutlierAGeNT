"""Outlier detection and imputation tools exposed to Google ADK."""

from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Any, Optional

import pandas as pd

from .data_io import (
    load_table,
    numeric_feature_columns,
    records_preview,
    resolve_output_path,
    sanitize_json,
    save_table,
    write_json_report,
)


SUPPORTED_OUTLIER_METHODS = {"iqr", "zscore", "modified_zscore"}
SUPPORTED_OUTLIER_IMPUTATIONS = {"none", "median", "mean", "cap", "drop"}


def detect_outliers(
    file_path: str,
    columns: Optional[str] = None,
    method: str = "iqr",
    sheet_name: Optional[str] = None,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    modified_zscore_threshold: float = 3.5,
) -> dict[str, Any]:
    """Detect outliers in numeric columns of a tabular data file.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to inspect.
        columns: Optional comma-separated column names. Leave blank for all numeric feature columns.
        method: Detection method: iqr, zscore, or modified_zscore.
        sheet_name: Optional Excel sheet name.
        iqr_multiplier: IQR fence multiplier, usually 1.5.
        zscore_threshold: Absolute z-score threshold, usually 3.0.
        modified_zscore_threshold: Absolute modified z-score threshold, usually 3.5.

    Returns:
        Structured report with bounds, counts, row indices, and preview records.
    """
    method = method.lower().strip()
    if method not in SUPPORTED_OUTLIER_METHODS:
        return {
            "status": "error",
            "error_message": f"Unsupported method '{method}'.",
            "supported_methods": sorted(SUPPORTED_OUTLIER_METHODS),
        }

    df, path = load_table(file_path, sheet_name=sheet_name)
    numeric_columns = numeric_feature_columns(df, columns)
    column_reports = [
        _detect_column(
            df,
            column,
            method=method,
            iqr_multiplier=iqr_multiplier,
            zscore_threshold=zscore_threshold,
            modified_zscore_threshold=modified_zscore_threshold,
        )
        for column in numeric_columns
    ]
    report = {
        "status": "success",
        "file_path": str(path),
        "row_count": int(len(df)),
        "numeric_columns": numeric_columns,
        "method": method,
        "total_outliers": int(sum(item["outlier_count"] for item in column_reports)),
        "columns": column_reports,
        "preview_records": records_preview(df),
    }
    return sanitize_json(report)


def impute_outliers(
    file_path: str,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    columns: Optional[str] = None,
    method: str = "iqr",
    imputation: str = "median",
    sheet_name: Optional[str] = None,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    modified_zscore_threshold: float = 3.5,
) -> dict[str, Any]:
    """Detect and impute outliers in numeric columns of a tabular data file.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to clean.
        output_path: Optional cleaned output path. Defaults beside the input file.
        report_path: Optional JSON report path. Defaults beside the output file.
        columns: Optional comma-separated column names. Leave blank for all numeric feature columns.
        method: Detection method: iqr, zscore, or modified_zscore.
        imputation: Imputation strategy: none, median, mean, cap, or drop.
        sheet_name: Optional Excel sheet name.
        iqr_multiplier: IQR fence multiplier, usually 1.5.
        zscore_threshold: Absolute z-score threshold, usually 3.0.
        modified_zscore_threshold: Absolute modified z-score threshold, usually 3.5.

    Returns:
        Structured report with output paths and every changed cell or dropped row.
    """
    method = method.lower().strip()
    imputation = imputation.lower().strip()
    if method not in SUPPORTED_OUTLIER_METHODS:
        return {
            "status": "error",
            "error_message": f"Unsupported method '{method}'.",
            "supported_methods": sorted(SUPPORTED_OUTLIER_METHODS),
        }
    if imputation not in SUPPORTED_OUTLIER_IMPUTATIONS:
        return {
            "status": "error",
            "error_message": f"Unsupported imputation '{imputation}'.",
            "supported_imputations": sorted(SUPPORTED_OUTLIER_IMPUTATIONS),
        }

    df, path = load_table(file_path, sheet_name=sheet_name)
    cleaned = df.copy()
    numeric_columns = numeric_feature_columns(cleaned, columns)
    column_reports: list[dict[str, Any]] = []
    changed_cells: list[dict[str, Any]] = []
    dropped_rows: set[int] = set()

    for column in numeric_columns:
        column_report = _detect_column(
            cleaned,
            column,
            method=method,
            iqr_multiplier=iqr_multiplier,
            zscore_threshold=zscore_threshold,
            modified_zscore_threshold=modified_zscore_threshold,
        )
        column_reports.append(column_report)
        row_indices = column_report["outlier_indices"]
        if imputation == "none" or not row_indices:
            continue
        if imputation == "drop":
            dropped_rows.update(row_indices)
            continue

        series = pd.to_numeric(cleaned[column], errors="coerce")
        inliers = series.drop(index=row_indices, errors="ignore").dropna()
        if inliers.empty:
            inliers = series.dropna()
        replacement = _replacement_value(
            inliers.tolist(),
            imputation=imputation,
        )

        for row_index in row_indices:
            old_value = cleaned.at[row_index, column]
            numeric_old_value = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
            if pd.isna(numeric_old_value):
                continue
            new_value = (
                _cap_value(
                    float(numeric_old_value),
                    lower_bound=column_report["lower_bound"],
                    upper_bound=column_report["upper_bound"],
                )
                if imputation == "cap"
                else replacement
            )
            cleaned.at[row_index, column] = new_value
            changed_cells.append(
                {
                    "row_index": int(row_index),
                    "column": column,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )

    if dropped_rows:
        cleaned = cleaned.drop(index=sorted(dropped_rows)).reset_index(drop=True)

    output = resolve_output_path(output_path, path, "_outliers_imputed.csv")
    report_output = resolve_output_path(
        report_path,
        output,
        "_report.json",
    )
    save_table(cleaned, output)

    report = {
        "status": "success",
        "file_path": str(path),
        "output_path": str(output),
        "report_path": str(report_output),
        "row_count_before": int(len(df)),
        "row_count_after": int(len(cleaned)),
        "numeric_columns": numeric_columns,
        "method": method,
        "imputation": imputation,
        "total_outliers": int(sum(item["outlier_count"] for item in column_reports)),
        "columns": column_reports,
        "changed_cells": changed_cells,
        "dropped_rows": sorted(dropped_rows),
    }
    write_json_report(report, report_output)
    return sanitize_json(report)


def _detect_column(
    df: pd.DataFrame,
    column: str,
    *,
    method: str,
    iqr_multiplier: float,
    zscore_threshold: float,
    modified_zscore_threshold: float,
) -> dict[str, Any]:
    series = pd.to_numeric(df[column], errors="coerce")
    values = series.dropna()
    if len(values) < 4:
        return {
            "column": column,
            "method": method,
            "lower_bound": None,
            "upper_bound": None,
            "center": None,
            "spread": None,
            "outlier_count": 0,
            "outlier_indices": [],
        }

    if method == "iqr":
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        spread = q3 - q1
        lower_bound = q1 - iqr_multiplier * spread
        upper_bound = q3 + iqr_multiplier * spread
        mask = (series < lower_bound) | (series > upper_bound)
        center = float(values.median())
    elif method == "zscore":
        numeric_values = values.astype(float).tolist()
        center = float(mean(numeric_values))
        spread = float(pstdev(numeric_values))
        if spread == 0:
            mask = pd.Series(False, index=df.index)
            lower_bound = None
            upper_bound = None
        else:
            lower_bound = center - zscore_threshold * spread
            upper_bound = center + zscore_threshold * spread
            mask = ((series - center).abs() / spread) > zscore_threshold
    else:
        center = float(values.median())
        deviations = (values - center).abs()
        spread = float(deviations.median())
        if spread == 0:
            mask = pd.Series(False, index=df.index)
            lower_bound = None
            upper_bound = None
        else:
            modified_scores = 0.6745 * (series - center) / spread
            mask = modified_scores.abs() > modified_zscore_threshold
            lower_bound = center - (modified_zscore_threshold * spread / 0.6745)
            upper_bound = center + (modified_zscore_threshold * spread / 0.6745)

    outlier_indices = [int(index) for index in series[mask.fillna(False)].index.tolist()]
    return {
        "column": column,
        "method": method,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "center": center,
        "spread": spread,
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices,
    }


def _replacement_value(
    values: list[float],
    *,
    imputation: str,
) -> Optional[float]:
    if not values:
        return None
    if imputation == "median":
        return float(median(values))
    if imputation == "mean":
        return float(mean(values))
    return None


def _cap_value(
    value: float,
    *,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
) -> float:
    if lower_bound is not None and value < lower_bound:
        return float(lower_bound)
    if upper_bound is not None and value > upper_bound:
        return float(upper_bound)
    return float(value)
