"""Model training and model-selection tools exposed to Google ADK."""

from __future__ import annotations

from inspect import signature
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .data_io import (
    is_identifier_column,
    load_table,
    parse_columns,
    resolve_output_path,
    sanitize_json,
    write_json_report,
)


SUPPORTED_TASK_TYPES = {"auto", "classification", "regression"}


def train_and_select_model(
    file_path: str,
    target_column: str,
    x_columns: Optional[str] = None,
    task_type: str = "auto",
    sheet_name: Optional[str] = None,
    report_path: Optional[str] = None,
    model_output_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train candidate models and return the best-performing one.

    Args:
        file_path: CSV, JSON, XLSX, or XLSM file to train from. This should usually be the cleaned file after missing-data imputation.
        target_column: Y/dependent variable.
        x_columns: Optional comma-separated X/feature columns. Leave blank to use all non-target non-ID non-constant columns.
        task_type: auto, classification, or regression.
        sheet_name: Optional Excel sheet name.
        report_path: Optional JSON report output path.
        model_output_path: Optional joblib model output path.
        test_size: Validation split size between 0 and 0.5.
        random_state: Random seed for reproducible splits and models.

    Returns:
        Structured model-selection report with metrics, best model, and saved model path.
    """
    task_type = task_type.lower().strip()
    if task_type not in SUPPORTED_TASK_TYPES:
        return {
            "status": "error",
            "error_message": f"Unsupported task_type '{task_type}'.",
            "supported_task_types": sorted(SUPPORTED_TASK_TYPES),
        }

    df, path = load_table(file_path, sheet_name=sheet_name)
    if target_column not in df.columns:
        return {
            "status": "error",
            "error_message": f"Target column not found: {target_column}",
            "available_columns": list(df.columns),
        }

    features = _resolve_feature_columns(df, target_column, x_columns)
    if not features:
        return {
            "status": "error",
            "error_message": "No usable X variables were found.",
            "target_column": target_column,
        }

    modeling_df = df[features + [target_column]].dropna(subset=[target_column]).copy()
    if len(modeling_df) < 10:
        return {
            "status": "error",
            "error_message": "At least 10 rows with a target value are required for modeling.",
            "row_count": int(len(modeling_df)),
        }

    inferred_task_type = _infer_task_type(modeling_df[target_column], task_type)
    x = modeling_df[features]
    y_raw = modeling_df[target_column]
    y, label_mapping = _prepare_target(y_raw, inferred_task_type)
    stratify = _stratify_target(y, inferred_task_type)
    test_size = min(max(float(test_size), 0.1), 0.5)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = _build_preprocessor(x_train)
    candidates = _candidate_models(inferred_task_type, random_state=random_state)
    model_reports: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in candidates:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            metrics = (
                _classification_metrics(y_test, predictions)
                if inferred_task_type == "classification"
                else _regression_metrics(y_test, predictions)
            )
            model_reports.append(
                {
                    "model": name,
                    "status": "success",
                    "metrics": metrics,
                }
            )
            fitted_models[name] = pipeline
        except Exception as exc:
            model_reports.append(
                {
                    "model": name,
                    "status": "failed",
                    "error_message": str(exc),
                }
            )

    successful = [item for item in model_reports if item["status"] == "success"]
    if not successful:
        return {
            "status": "error",
            "error_message": "All candidate models failed.",
            "models": model_reports,
        }

    best_model_name = _best_model_name(successful, inferred_task_type)
    model_output = resolve_output_path(
        model_output_path,
        path,
        f"_{best_model_name}_model.joblib",
    )
    report_output = resolve_output_path(report_path, path, "_model_report.json")
    joblib.dump(
        {
            "pipeline": fitted_models[best_model_name],
            "task_type": inferred_task_type,
            "target_column": target_column,
            "x_columns": features,
            "label_mapping": label_mapping,
        },
        model_output,
    )

    report = {
        "status": "success",
        "file_path": str(path),
        "report_path": str(report_output),
        "model_output_path": str(model_output),
        "task_type": inferred_task_type,
        "target_column": target_column,
        "x_columns": features,
        "row_count_used": int(len(modeling_df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_size": test_size,
        "best_model": best_model_name,
        "best_metrics": next(
            item["metrics"] for item in successful if item["model"] == best_model_name
        ),
        "models": model_reports,
        "label_mapping": label_mapping,
        "notes": [
            "Run this after intake, missing-data handling, and EDA.",
            "Outlier handling should be applied before modeling when the confirmed workflow requires it.",
            "This is a holdout validation comparison, not a full hyperparameter search.",
        ],
    }
    write_json_report(report, report_output)
    return sanitize_json(report)


def _resolve_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    x_columns: Optional[str],
) -> list[str]:
    if x_columns:
        requested = parse_columns(x_columns, df)
    else:
        requested = list(df.columns)

    features = []
    for column in requested:
        if column == target_column:
            continue
        if not x_columns and is_identifier_column(column):
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        features.append(column)
    return features


def _infer_task_type(series: pd.Series, requested_task_type: str) -> str:
    if requested_task_type != "auto":
        return requested_task_type
    numeric = pd.to_numeric(series, errors="coerce")
    observed = series.dropna()
    is_numeric = numeric[series.notna()].notna().all()
    unique_count = int(observed.nunique())
    if is_numeric and unique_count > min(20, max(5, int(len(observed) * 0.05))):
        return "regression"
    return "classification"


def _prepare_target(series: pd.Series, task_type: str) -> tuple[pd.Series, dict[str, Any]]:
    if task_type == "regression":
        return pd.to_numeric(series, errors="coerce"), {}

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series.astype(str))
    mapping = {
        str(index): label
        for index, label in enumerate(encoder.classes_.tolist())
    }
    return pd.Series(encoded, index=series.index), mapping


def _stratify_target(series: pd.Series, task_type: str) -> Optional[pd.Series]:
    if task_type != "classification":
        return None
    counts = series.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return series


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in x.columns
        if pd.to_numeric(x[column], errors="coerce")[x[column].notna()].notna().all()
    ]
    categorical_columns = [column for column in x.columns if column not in numeric_columns]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def _one_hot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in signature(OneHotEncoder).parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def _candidate_models(task_type: str, *, random_state: int) -> list[tuple[str, Any]]:
    if task_type == "classification":
        models: list[tuple[str, Any]] = [
            (
                "logistic_regression",
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=250,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingClassifier(random_state=random_state),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(random_state=random_state),
            ),
        ]
        xgb_classifier = _xgb_classifier(random_state)
        if xgb_classifier is not None:
            models.append(("xgboost", xgb_classifier))
        return models

    models = [
        ("ridge", Ridge()),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=250,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=250,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(random_state=random_state),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(random_state=random_state),
        ),
    ]
    xgb_regressor = _xgb_regressor(random_state)
    if xgb_regressor is not None:
        models.append(("xgboost", xgb_regressor))
    return models


def _xgb_classifier(random_state: int) -> Optional[Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _xgb_regressor(random_state: int) -> Optional[Any]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    average = "binary" if len(pd.Series(y_true).unique()) == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mse**0.5),
    }


def _best_model_name(model_reports: list[dict[str, Any]], task_type: str) -> str:
    if task_type == "classification":
        return max(
            model_reports,
            key=lambda item: (
                item["metrics"]["accuracy"],
                item["metrics"]["balanced_accuracy"],
                item["metrics"]["f1"],
            ),
        )["model"]
    return max(
        model_reports,
        key=lambda item: (
            item["metrics"]["r2"],
            -item["metrics"]["rmse"],
            -item["metrics"]["mae"],
        ),
    )["model"]
