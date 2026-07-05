"""Dataset intake and downstream planning tools exposed to Google ADK."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .data_io import (
    PROJECT_DIR,
    is_identifier_column,
    load_table,
    numeric_feature_columns,
    parse_columns,
    records_preview,
    resolve_output_path,
    sanitize_json,
    write_json_report,
)


TARGET_NAME_HINTS = {
    "target",
    "y",
    "label",
    "class",
    "outcome",
    "response",
    "dependent",
    "result",
}
SUPPORTED_TASK_TYPES = {
    "auto",
    "regression",
    "classification",
    "time_series",
    "clustering",
    "descriptive",
}


def intake_dataset_for_downstream(
    file_path: str,
    target_column: Optional[str] = None,
    x_columns: Optional[str] = None,
    task_type: str = "auto",
    sheet_name: Optional[str] = None,
    copy_to_uploads: bool = True,
    report_path: Optional[str] = None,
) -> dict[str, Any]:
    """Register a dataset and plan downstream modeling or analysis tasks.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to register.
        target_column: Optional Y/dependent variable. If blank, the tool infers a likely target when possible.
        x_columns: Optional comma-separated X/feature columns. If blank, features are inferred from all non-target non-ID columns.
        task_type: auto, regression, classification, time_series, clustering, or descriptive.
        sheet_name: Optional Excel sheet name.
        copy_to_uploads: Whether to copy the source file into Google_ADK/uploads.
        report_path: Optional JSON manifest/report path.

    Returns:
        Dataset manifest with inferred X/Y variables and downstream plan.
    """
    task_type = task_type.lower().strip()
    if task_type not in SUPPORTED_TASK_TYPES:
        return {
            "status": "error",
            "error_message": f"Unsupported task_type '{task_type}'.",
            "supported_task_types": sorted(SUPPORTED_TASK_TYPES),
        }

    df, path = load_table(file_path, sheet_name=sheet_name)
    uploaded_path = _copy_to_uploads(path) if copy_to_uploads else path

    try:
        y_column = _resolve_target_column(df, target_column)
        x_variable_report = _resolve_x_columns(df, x_columns, y_column)
    except ValueError as exc:
        return {
            "status": "error",
            "file_path": str(path),
            "error_message": str(exc),
            "available_columns": list(df.columns),
        }
    inferred_task_type = _infer_task_type(df, y_column, task_type)
    downstream_plan = _downstream_plan(
        inferred_task_type,
        has_target=y_column is not None,
    )
    requirements = _requirements(
        inferred_task_type,
        has_target=y_column is not None,
        y_column=y_column,
    )

    output_report = resolve_output_path(
        report_path,
        uploaded_path,
        "_intake_manifest.json",
    )
    report = {
        "status": "success",
        "source_file_path": str(path),
        "registered_file_path": str(uploaded_path),
        "report_path": str(output_report),
        "sheet_name": sheet_name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "task_type": inferred_task_type,
        "y_variable": y_column,
        "x_variables": x_variable_report["x_variables"],
        "excluded_columns": x_variable_report["excluded_columns"],
        "schema": _schema(df),
        "requirements_for_downstream": requirements,
        "recommended_plan": downstream_plan,
        "preview_records": records_preview(df),
    }
    write_json_report(report, output_report)
    return sanitize_json(report)


def _copy_to_uploads(path: Path) -> Path:
    uploads_dir = PROJECT_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = uploads_dir / path.name
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination.resolve()


def _resolve_target_column(df: pd.DataFrame, target_column: Optional[str]) -> Optional[str]:
    if target_column:
        requested = target_column.strip()
        matched = _match_column(df, requested)
        if matched is None:
            raise ValueError(f"Target column not found: {requested}")
        return matched

    normalized_map = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }
    for hint in TARGET_NAME_HINTS:
        if hint in normalized_map:
            return str(normalized_map[hint])
    for normalized, original in normalized_map.items():
        if normalized.endswith("_target") or normalized.endswith("_label"):
            return str(original)
    return None


def _match_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    if requested in df.columns:
        return requested
    normalized = requested.strip().lower()
    matches = [column for column in df.columns if str(column).strip().lower() == normalized]
    if len(matches) == 1:
        return str(matches[0])
    return None


def _resolve_x_columns(
    df: pd.DataFrame,
    x_columns: Optional[str],
    y_column: Optional[str],
) -> dict[str, Any]:
    if x_columns:
        requested = parse_columns(x_columns, df)
    else:
        requested = list(df.columns)

    x_variables: list[str] = []
    excluded_columns: list[dict[str, str]] = []
    for column in requested:
        reason = None
        if y_column and column == y_column:
            reason = "target_variable"
        elif not x_columns and is_identifier_column(column):
            reason = "identifier_like_column"
        elif df[column].nunique(dropna=True) <= 1:
            reason = "constant_or_empty_column"

        if reason:
            excluded_columns.append({"column": column, "reason": reason})
        else:
            x_variables.append(column)

    return {
        "x_variables": x_variables,
        "excluded_columns": excluded_columns,
    }


def _infer_task_type(
    df: pd.DataFrame,
    y_column: Optional[str],
    requested_task_type: str,
) -> str:
    requested = requested_task_type.lower().strip()
    if requested != "auto":
        return requested
    if y_column is None:
        return "descriptive"

    target = df[y_column]
    numeric = pd.to_numeric(target, errors="coerce")
    observed_target = target.dropna()
    if observed_target.empty:
        return "supervised_unknown"
    is_numeric = numeric[target.notna()].notna().all()
    unique_count = int(observed_target.nunique())
    if is_numeric and unique_count > min(20, max(5, int(len(observed_target) * 0.05))):
        return "regression"
    return "classification"


def _schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    numeric_columns = set(numeric_feature_columns(df))
    rows = []
    for column in df.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "semantic_type": _semantic_type(df[column], column, numeric_columns),
                "unique_count": int(df[column].nunique(dropna=True)),
                "sample_values": df[column].dropna().head(5).tolist(),
            }
        )
    return rows


def _semantic_type(
    series: pd.Series,
    column: str,
    numeric_columns: set[str],
) -> str:
    if is_identifier_column(column):
        return "identifier"
    if column in numeric_columns:
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    parsed = pd.to_datetime(series.dropna().astype(str).head(50), errors="coerce")
    if len(parsed) and parsed.notna().mean() >= 0.9:
        return "datetime"
    return "categorical_or_text"


def _requirements(
    task_type: str,
    *,
    has_target: bool,
    y_column: Optional[str],
) -> list[str]:
    requirements = [
        "Confirm that the registered file is the intended analysis dataset.",
        "Confirm the X variables and Y variable before downstream processing.",
        "Call missing_data_agent for missing-value inspection or imputation.",
        "Call outlier_agent for outlier detection or imputation.",
        "Call eda_agent for summaries, moments, robust statistics, correlations, and ACF.",
        "Call modeling_agent for model comparison only after X/Y variables and cleaned data are confirmed.",
    ]
    if task_type in {"regression", "classification", "supervised_unknown"}:
        if has_target:
            requirements.append(f"Confirm that '{y_column}' is the correct Y variable.")
        else:
            requirements.append("Provide a Y/target column for supervised modeling.")
        requirements.append("Confirm that no X variable leaks information from the Y variable.")
    else:
        requirements.append("Provide a target column later if supervised modeling is required.")
    return requirements


def _downstream_plan(
    task_type: str,
    *,
    has_target: bool,
) -> list[dict[str, Any]]:
    plan = [
        {
            "step": 1,
            "agent": "data_qc_input_agent",
            "action": "Use this manifest to confirm file path, X variables, Y variable, and task type.",
        },
        {
            "step": 2,
            "agent": "missing_data_agent",
            "action": "Inspect and impute missing values if the user requests missing-data handling.",
        },
        {
            "step": 3,
            "agent": "eda_agent",
            "action": "Generate EDA statistics, correlation matrices, and ACF diagnostics.",
        },
        {
            "step": 4,
            "agent": "outlier_agent",
            "action": "Detect and impute/cap outliers if the user requests outlier handling.",
        },
    ]
    if has_target and task_type in {"regression", "classification"}:
        plan.append(
            {
                "step": 5,
                "agent": "modeling_agent",
                "action": f"Train and compare candidate {task_type} models using confirmed X and Y variables.",
            }
        )
    else:
        plan.append(
            {
                "step": 5,
                "agent": "analysis_or_downstream_agent",
                "action": "Proceed with descriptive analysis, clustering, or request a target column.",
            }
        )
    return plan
