"""Shared table I/O helpers for the ADK data-quality tools."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent


def resolve_data_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECT_DIR / path,
        WORKSPACE_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def resolve_output_path(
    output_path: Optional[str],
    input_path: Path,
    default_suffix: str,
) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
        if path.is_absolute():
            resolved = path
        else:
            resolved = PROJECT_DIR / path
    else:
        resolved = input_path.with_name(f"{input_path.stem}{default_suffix}")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def load_table(file_path: str, sheet_name: Optional[str] = None) -> tuple[pd.DataFrame, Path]:
    path = resolve_data_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), path
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0), path
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            payload = payload["records"]
        return pd.DataFrame(payload), path

    raise ValueError("Supported input formats are .csv, .json, .xlsx, and .xlsm.")


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
        return
    if suffix in {".xlsx", ".xlsm"}:
        df.to_excel(output_path, index=False)
        return
    raise ValueError("Supported output formats are .csv, .json, .xlsx, and .xlsm.")


def parse_columns(columns: Optional[str], df: pd.DataFrame) -> list[str]:
    if not columns:
        return list(df.columns)
    requested = [column.strip() for column in columns.split(",") if column.strip()]
    missing = [column for column in requested if column not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}")
    return requested


def numeric_feature_columns(df: pd.DataFrame, columns: Optional[str] = None) -> list[str]:
    requested = parse_columns(columns, df)
    numeric_columns: list[str] = []
    explicit = columns is not None
    for column in requested:
        if not explicit and is_identifier_column(column):
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        populated = df[column].notna()
        if populated.any() and series[populated].notna().all():
            numeric_columns.append(column)
    return numeric_columns


def is_identifier_column(column: str) -> bool:
    normalized = column.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"id", "index", "row_id", "record_id"} or normalized.endswith(
        "_id"
    )


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        try:
            return sanitize_json(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def records_preview(df: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    return sanitize_json(df.head(limit).to_dict(orient="records"))


def write_json_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(sanitize_json(report), indent=2),
        encoding="utf-8",
    )
