"""Missing-data inspection and imputation tools exposed to Google ADK."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .data_io import (
    load_table,
    parse_columns,
    records_preview,
    resolve_output_path,
    sanitize_json,
    save_table,
    write_json_report,
)


SUPPORTED_MISSING_STRATEGIES = {
    "auto",
    "mean",
    "median",
    "mode",
    "constant",
    "drop_rows",
    "drop_columns",
    "forward_fill",
    "backward_fill",
}


def inspect_missing_data(
    file_path: str,
    columns: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> dict[str, Any]:
    """Inspect missing values in a tabular data file.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to inspect.
        columns: Optional comma-separated column names. Leave blank for all columns.
        sheet_name: Optional Excel sheet name.

    Returns:
        Structured missingness report by column and row.
    """
    df, path = load_table(file_path, sheet_name=sheet_name)
    selected_columns = parse_columns(columns, df)
    target = df[selected_columns]
    missing_mask = target.isna()
    row_missing_counts = missing_mask.sum(axis=1)
    column_reports = []

    for column in selected_columns:
        missing_count = int(missing_mask[column].sum())
        column_reports.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": missing_count,
                "missing_percent": round(
                    (missing_count / len(df) * 100) if len(df) else 0.0,
                    4,
                ),
                "non_missing_count": int(df[column].notna().sum()),
            }
        )

    report = {
        "status": "success",
        "file_path": str(path),
        "row_count": int(len(df)),
        "column_count": int(len(selected_columns)),
        "total_missing_cells": int(missing_mask.sum().sum()),
        "rows_with_missing": int((row_missing_counts > 0).sum()),
        "complete_rows": int((row_missing_counts == 0).sum()),
        "columns": column_reports,
        "sample_rows_with_missing": [
            int(index) for index in row_missing_counts[row_missing_counts > 0].head(20).index
        ],
        "preview_records": records_preview(df),
    }
    return sanitize_json(report)


def impute_missing_data(
    file_path: str,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    columns: Optional[str] = None,
    strategy: str = "auto",
    fill_value: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> dict[str, Any]:
    """Check and impute missing values in a tabular data file.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to clean.
        output_path: Optional cleaned output path. Defaults beside the input file.
        report_path: Optional JSON report path. Defaults beside the output file.
        columns: Optional comma-separated column names. Leave blank for all columns.
        strategy: Imputation strategy: auto, mean, median, mode, constant, drop_rows, drop_columns, forward_fill, or backward_fill.
        fill_value: Replacement value for the constant strategy.
        sheet_name: Optional Excel sheet name.

    Returns:
        Structured report with missingness summary, output paths, and changed counts.
    """
    strategy = strategy.lower().strip()
    if strategy not in SUPPORTED_MISSING_STRATEGIES:
        return {
            "status": "error",
            "error_message": f"Unsupported missing-data strategy '{strategy}'.",
            "supported_strategies": sorted(SUPPORTED_MISSING_STRATEGIES),
        }

    df, path = load_table(file_path, sheet_name=sheet_name)
    selected_columns = parse_columns(columns, df)
    cleaned = df.copy()
    before_mask = cleaned[selected_columns].isna()
    column_reports: list[dict[str, Any]] = []
    dropped_rows: list[int] = []
    dropped_columns: list[str] = []

    if strategy == "drop_rows":
        row_mask = before_mask.any(axis=1)
        dropped_rows = [int(index) for index in cleaned[row_mask].index.tolist()]
        cleaned = cleaned.drop(index=dropped_rows).reset_index(drop=True)
    elif strategy == "drop_columns":
        dropped_columns = [
            column for column in selected_columns if bool(before_mask[column].any())
        ]
        cleaned = cleaned.drop(columns=dropped_columns)
    elif strategy in {"forward_fill", "backward_fill"}:
        cleaned[selected_columns] = (
            cleaned[selected_columns].ffill()
            if strategy == "forward_fill"
            else cleaned[selected_columns].bfill()
        )
    else:
        for column in selected_columns:
            missing_count = int(before_mask[column].sum())
            replacement = _choose_replacement(
                cleaned[column],
                strategy=strategy,
                fill_value=fill_value,
            )
            if missing_count:
                cleaned[column] = cleaned[column].fillna(replacement)
            column_reports.append(
                {
                    "column": column,
                    "strategy": strategy,
                    "replacement_value": replacement,
                    "missing_count_before": missing_count,
                    "missing_count_after": int(cleaned[column].isna().sum()),
                }
            )

    if strategy in {"drop_rows", "drop_columns", "forward_fill", "backward_fill"}:
        available_columns = [column for column in selected_columns if column in cleaned.columns]
        for column in available_columns:
            column_reports.append(
                {
                    "column": column,
                    "strategy": strategy,
                    "replacement_value": None,
                    "missing_count_before": int(before_mask[column].sum()),
                    "missing_count_after": int(cleaned[column].isna().sum()),
                }
            )

    output = resolve_output_path(output_path, path, "_missing_imputed.csv")
    report_output = resolve_output_path(report_path, output, "_report.json")
    save_table(cleaned, output)

    report = {
        "status": "success",
        "file_path": str(path),
        "output_path": str(output),
        "report_path": str(report_output),
        "strategy": strategy,
        "row_count_before": int(len(df)),
        "row_count_after": int(len(cleaned)),
        "column_count_before": int(len(df.columns)),
        "column_count_after": int(len(cleaned.columns)),
        "total_missing_before": int(before_mask.sum().sum()),
        "total_missing_after": int(
            cleaned[[column for column in selected_columns if column in cleaned.columns]]
            .isna()
            .sum()
            .sum()
        ),
        "columns": column_reports,
        "dropped_rows": dropped_rows,
        "dropped_columns": dropped_columns,
    }
    write_json_report(report, report_output)
    return sanitize_json(report)


def _choose_replacement(
    series: pd.Series,
    *,
    strategy: str,
    fill_value: Optional[str],
) -> Any:
    if strategy == "constant":
        return "" if fill_value is None else fill_value

    numeric = pd.to_numeric(series, errors="coerce")
    is_numeric = series.notna().any() and numeric[series.notna()].notna().all()

    if strategy == "mean":
        return float(numeric.mean()) if is_numeric else _mode_or_empty(series)
    if strategy == "median":
        return float(numeric.median()) if is_numeric else _mode_or_empty(series)
    if strategy == "mode":
        return _mode_or_empty(series)
    if strategy == "auto":
        if is_numeric:
            return float(numeric.median())
        return _mode_or_empty(series)

    return _mode_or_empty(series)


def _mode_or_empty(series: pd.Series) -> Any:
    modes = series.dropna().mode()
    if modes.empty:
        return ""
    return modes.iloc[0]
