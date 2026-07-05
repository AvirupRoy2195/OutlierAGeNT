"""Run the data workflow without ADK or Gemini calls.

This script reuses the same deterministic tools that the ADK agents call. Use it
when Gemini quota is exhausted or when you want a reproducible batch run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.data_qc_tools import intake_dataset_for_downstream
from tools.eda_tools import generate_eda_report
from tools.missing_data_tools import impute_missing_data, inspect_missing_data
from tools.modeling_tools import train_and_select_model
from tools.outlier_tools import detect_outliers, impute_outliers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the data workflow without LLM calls.")
    parser.add_argument("input", help="Input .csv, .json, .xlsx, or .xlsm file.")
    parser.add_argument("--target", required=True, help="Y/target column.")
    parser.add_argument(
        "--x-columns",
        help="Comma-separated X/feature columns. Omit to use all non-target non-ID columns.",
    )
    parser.add_argument(
        "--task-type",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Modeling task type.",
    )
    parser.add_argument("--sheet-name", help="Excel sheet name.")
    parser.add_argument("--output-dir", default="offline_outputs", help="Output directory.")
    parser.add_argument(
        "--missing-strategy",
        default="auto",
        help="Missing-data strategy passed to impute_missing_data.",
    )
    parser.add_argument(
        "--outlier-method",
        default="iqr",
        choices=["iqr", "zscore", "modified_zscore"],
        help="Outlier detection method.",
    )
    parser.add_argument(
        "--outlier-imputation",
        default="median",
        choices=["none", "median", "mean", "cap", "drop"],
        help="Outlier imputation method.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing-data imputation.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA report.")
    parser.add_argument("--skip-outliers", action="store_true", help="Skip outlier handling.")
    parser.add_argument("--skip-modeling", action="store_true", help="Skip model training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.input
    summary: dict[str, Any] = {
        "input": source_path,
        "target": args.target,
        "x_columns": args.x_columns,
        "steps": {},
    }

    intake = intake_dataset_for_downstream(
        source_path,
        target_column=args.target,
        x_columns=args.x_columns,
        task_type=args.task_type,
        sheet_name=args.sheet_name,
        copy_to_uploads=False,
        report_path=str(output_dir / "intake_manifest.json"),
    )
    summary["steps"]["intake"] = _compact_status(intake)
    if intake.get("status") != "success":
        _write_summary(output_dir, summary)
        print(json.dumps(summary, indent=2))
        return

    working_path = source_path
    x_columns = args.x_columns or ",".join(intake["x_variables"])

    missing_check = inspect_missing_data(
        working_path,
        columns=x_columns + "," + args.target,
        sheet_name=args.sheet_name,
    )
    summary["steps"]["missing_check"] = _compact_status(missing_check)

    if not args.skip_missing:
        missing = impute_missing_data(
            working_path,
            output_path=str(output_dir / "missing_imputed.csv"),
            report_path=str(output_dir / "missing_imputed_report.json"),
            columns=x_columns + "," + args.target,
            strategy=args.missing_strategy,
            sheet_name=args.sheet_name,
        )
        summary["steps"]["missing_imputation"] = _compact_status(missing)
        if missing.get("status") == "success":
            working_path = missing["output_path"]

    if not args.skip_eda:
        eda = generate_eda_report(
            working_path,
            columns=x_columns + "," + args.target,
            target_column=args.target,
            report_path=str(output_dir / "eda_report.json"),
        )
        summary["steps"]["eda"] = _compact_status(eda)

    outlier_check = detect_outliers(
        working_path,
        columns=x_columns,
        method=args.outlier_method,
        sheet_name=args.sheet_name if Path(working_path).suffix.lower() in {".xlsx", ".xlsm"} else None,
    )
    summary["steps"]["outlier_check"] = _compact_status(outlier_check)

    if not args.skip_outliers:
        outliers = impute_outliers(
            working_path,
            output_path=str(output_dir / "outliers_imputed.csv"),
            report_path=str(output_dir / "outliers_imputed_report.json"),
            columns=x_columns,
            method=args.outlier_method,
            imputation=args.outlier_imputation,
            sheet_name=args.sheet_name if Path(working_path).suffix.lower() in {".xlsx", ".xlsm"} else None,
        )
        summary["steps"]["outlier_imputation"] = _compact_status(outliers)
        if outliers.get("status") == "success":
            working_path = outliers["output_path"]

    if not args.skip_modeling:
        model = train_and_select_model(
            working_path,
            target_column=args.target,
            x_columns=x_columns,
            task_type=args.task_type,
            report_path=str(output_dir / "model_report.json"),
            model_output_path=str(output_dir / "best_model.joblib"),
        )
        summary["steps"]["modeling"] = _compact_status(model)

    summary["final_dataset"] = working_path
    _write_summary(output_dir, summary)
    print(json.dumps(summary, indent=2))


def _compact_status(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "report_path",
        "output_path",
        "model_output_path",
        "task_type",
        "best_model",
        "best_metrics",
        "total_missing_cells",
        "total_missing_before",
        "total_missing_after",
        "total_outliers",
        "row_count",
        "row_count_used",
        "error_message",
    ]
    return {key: report[key] for key in keys if key in report}


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
