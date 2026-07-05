"""Outlier detection and imputation utilities.

The tool works on tabular records represented as ``list[dict]`` and keeps the
implementation dependency-free so it can run in constrained environments.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable


Number = float
Record = dict[str, Any]


@dataclass(frozen=True)
class ColumnStats:
    column: str
    method: str
    lower_bound: Number | None
    upper_bound: Number | None
    center: Number | None
    spread: Number | None
    outlier_count: int
    outlier_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class OutlierReport:
    method: str
    imputation: str
    row_count: int
    numeric_columns: list[str]
    column_stats: list[ColumnStats]
    changed_cells: list[dict[str, Any]]
    dropped_rows: list[int] = field(default_factory=list)

    @property
    def total_outliers(self) -> int:
        return sum(stats.outlier_count for stats in self.column_stats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "imputation": self.imputation,
            "row_count": self.row_count,
            "numeric_columns": self.numeric_columns,
            "total_outliers": self.total_outliers,
            "column_stats": [
                {
                    "column": stats.column,
                    "method": stats.method,
                    "lower_bound": stats.lower_bound,
                    "upper_bound": stats.upper_bound,
                    "center": stats.center,
                    "spread": stats.spread,
                    "outlier_count": stats.outlier_count,
                    "outlier_indices": stats.outlier_indices,
                }
                for stats in self.column_stats
            ],
            "changed_cells": self.changed_cells,
            "dropped_rows": self.dropped_rows,
        }


def load_csv(path: str | Path) -> list[Record]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: str | Path, records: list[Record]) -> None:
    if not records:
        Path(path).write_text("", encoding="utf-8")
        return

    fieldnames = list(records[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


class OutlierTool:
    """Detect and impute outliers in numeric tabular columns."""

    METHODS = {"iqr", "zscore", "modified_zscore"}
    IMPUTATIONS = {"none", "median", "mean", "cap", "drop"}

    def detect_and_impute(
        self,
        records: list[Record],
        *,
        columns: Iterable[str] | None = None,
        method: str = "iqr",
        imputation: str = "median",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        modified_zscore_threshold: float = 3.5,
    ) -> tuple[list[Record], OutlierReport]:
        method = method.lower().strip()
        imputation = imputation.lower().strip()
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {sorted(self.METHODS)}.")
        if imputation not in self.IMPUTATIONS:
            raise ValueError(
                f"Unknown imputation '{imputation}'. Choose from {sorted(self.IMPUTATIONS)}."
            )

        cleaned = [dict(row) for row in records]
        numeric_columns = self._resolve_numeric_columns(cleaned, columns)
        all_stats: list[ColumnStats] = []
        changed_cells: list[dict[str, Any]] = []
        rows_to_drop: set[int] = set()

        for column in numeric_columns:
            values_by_index = self._numeric_values_by_index(cleaned, column)
            values = list(values_by_index.values())
            stats = self._detect_column(
                column,
                values_by_index,
                method=method,
                iqr_multiplier=iqr_multiplier,
                zscore_threshold=zscore_threshold,
                modified_zscore_threshold=modified_zscore_threshold,
            )
            all_stats.append(stats)

            if not stats.outlier_indices or imputation == "none":
                continue

            inlier_values = [
                value
                for row_index, value in values_by_index.items()
                if row_index not in stats.outlier_indices
            ] or values
            replacement_value = self._replacement_value(
                inlier_values,
                imputation=imputation,
                lower_bound=stats.lower_bound,
                upper_bound=stats.upper_bound,
            )
            for row_index in stats.outlier_indices:
                if imputation == "drop":
                    rows_to_drop.add(row_index)
                    continue

                old_value = cleaned[row_index].get(column)
                numeric_old_value = self._to_float(old_value)
                if numeric_old_value is None:
                    continue

                new_value = (
                    self._cap_value(
                        numeric_old_value,
                        lower_bound=stats.lower_bound,
                        upper_bound=stats.upper_bound,
                    )
                    if imputation == "cap"
                    else replacement_value
                )
                cleaned[row_index][column] = self._format_like_original(old_value, new_value)
                changed_cells.append(
                    {
                        "row_index": row_index,
                        "column": column,
                        "old_value": old_value,
                        "new_value": cleaned[row_index][column],
                    }
                )

        if rows_to_drop:
            cleaned = [
                row for index, row in enumerate(cleaned) if index not in rows_to_drop
            ]

        return cleaned, OutlierReport(
            method=method,
            imputation=imputation,
            row_count=len(records),
            numeric_columns=numeric_columns,
            column_stats=all_stats,
            changed_cells=changed_cells,
            dropped_rows=sorted(rows_to_drop),
        )

    def detect(
        self,
        records: list[Record],
        *,
        columns: Iterable[str] | None = None,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        modified_zscore_threshold: float = 3.5,
    ) -> OutlierReport:
        _, report = self.detect_and_impute(
            records,
            columns=columns,
            method=method,
            imputation="none",
            iqr_multiplier=iqr_multiplier,
            zscore_threshold=zscore_threshold,
            modified_zscore_threshold=modified_zscore_threshold,
        )
        return report

    def _detect_column(
        self,
        column: str,
        values_by_index: dict[int, Number],
        *,
        method: str,
        iqr_multiplier: float,
        zscore_threshold: float,
        modified_zscore_threshold: float,
    ) -> ColumnStats:
        values = list(values_by_index.values())
        if len(values) < 4:
            return ColumnStats(column, method, None, None, None, None, 0, [])

        if method == "iqr":
            q1 = percentile(values, 25)
            q3 = percentile(values, 75)
            spread = q3 - q1
            lower_bound = q1 - iqr_multiplier * spread
            upper_bound = q3 + iqr_multiplier * spread
            outlier_indices = [
                index
                for index, value in values_by_index.items()
                if value < lower_bound or value > upper_bound
            ]
            return ColumnStats(
                column=column,
                method=method,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                center=median(values),
                spread=spread,
                outlier_count=len(outlier_indices),
                outlier_indices=outlier_indices,
            )

        if method == "zscore":
            center = mean(values)
            spread = pstdev(values)
            if spread == 0:
                return ColumnStats(column, method, None, None, center, spread, 0, [])
            lower_bound = center - zscore_threshold * spread
            upper_bound = center + zscore_threshold * spread
            outlier_indices = [
                index
                for index, value in values_by_index.items()
                if abs((value - center) / spread) > zscore_threshold
            ]
            return ColumnStats(
                column,
                method,
                lower_bound,
                upper_bound,
                center,
                spread,
                len(outlier_indices),
                outlier_indices,
            )

        center = median(values)
        absolute_deviations = [abs(value - center) for value in values]
        spread = median(absolute_deviations)
        if spread == 0:
            return ColumnStats(column, method, None, None, center, spread, 0, [])
        outlier_indices = [
            index
            for index, value in values_by_index.items()
            if abs(0.6745 * (value - center) / spread) > modified_zscore_threshold
        ]
        lower_bound = center - (modified_zscore_threshold * spread / 0.6745)
        upper_bound = center + (modified_zscore_threshold * spread / 0.6745)
        return ColumnStats(
            column,
            method,
            lower_bound,
            upper_bound,
            center,
            spread,
            len(outlier_indices),
            outlier_indices,
        )

    def _resolve_numeric_columns(
        self, records: list[Record], columns: Iterable[str] | None
    ) -> list[str]:
        if not records:
            return []
        explicit_columns = columns is not None
        requested = list(columns) if columns else list(records[0].keys())
        numeric_columns: list[str] = []
        for column in requested:
            if not explicit_columns and self._is_identifier_column(column):
                continue
            numeric_count = 0
            populated_count = 0
            for row in records:
                value = row.get(column)
                if value in (None, ""):
                    continue
                populated_count += 1
                if self._to_float(value) is not None:
                    numeric_count += 1
            if populated_count and numeric_count == populated_count:
                numeric_columns.append(column)
        return numeric_columns

    def _is_identifier_column(self, column: str) -> bool:
        normalized = column.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized in {"id", "index", "row_id", "record_id"} or normalized.endswith(
            "_id"
        )

    def _numeric_values_by_index(
        self, records: list[Record], column: str
    ) -> dict[int, Number]:
        values_by_index: dict[int, Number] = {}
        for index, row in enumerate(records):
            value = self._to_float(row.get(column))
            if value is not None and math.isfinite(value):
                values_by_index[index] = value
        return values_by_index

    def _replacement_value(
        self,
        values: list[Number],
        *,
        imputation: str,
        lower_bound: Number | None,
        upper_bound: Number | None,
    ) -> Number | None:
        if imputation == "median":
            return median(values)
        if imputation == "mean":
            return mean(values)
        if imputation == "cap":
            if lower_bound is None or upper_bound is None:
                return None
            return None
        return None

    def _cap_value(
        self,
        value: Number,
        *,
        lower_bound: Number | None,
        upper_bound: Number | None,
    ) -> Number:
        if lower_bound is not None and value < lower_bound:
            return lower_bound
        if upper_bound is not None and value > upper_bound:
            return upper_bound
        return value

    def _format_like_original(self, original: Any, value: Number | None) -> Any:
        if value is None:
            return original
        if isinstance(original, int):
            return int(round(value))
        if isinstance(original, float):
            return float(value)
        if isinstance(original, str):
            stripped = original.strip()
            if stripped and stripped.replace("-", "", 1).isdigit():
                return str(int(round(value)))
            return f"{value:.10g}"
        return value

    def _to_float(self, value: Any) -> Number | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(str(value).strip())
        except ValueError:
            return None


def percentile(values: list[Number], percentile_value: float) -> Number:
    if not values:
        raise ValueError("Cannot calculate percentile for an empty list.")
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile_value / 100
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[int(rank)]
    lower_weight = upper_index - rank
    upper_weight = rank - lower_index
    return (
        sorted_values[lower_index] * lower_weight
        + sorted_values[upper_index] * upper_weight
    )
