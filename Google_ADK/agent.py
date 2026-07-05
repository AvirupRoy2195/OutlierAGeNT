"""Google ADK data-quality agent team.

Run from the workspace parent with:

    adk run Google_ADK
    adk web --port 8000
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.genai import types

from .tools.data_qc_tools import intake_dataset_for_downstream
from .tools.eda_tools import generate_eda_report
from .tools.missing_data_tools import impute_missing_data, inspect_missing_data
from .tools.modeling_tools import train_and_select_model
from .tools.outlier_tools import detect_outliers, impute_outliers


AGENT_MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-2.0-flash")
GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(initial_delay=2, attempts=3)
    )
)


data_qc_input_agent = Agent(
    name="data_qc_input_agent",
    model=AGENT_MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Registers uploaded datasets, identifies X and Y variables, "
        "infers task type, and creates a downstream analysis plan."
    ),
    instruction=(
        "You are the dataset intake and planning specialist. Use "
        "intake_dataset_for_downstream when the user uploads, registers, profiles, "
        "or prepares a dataset for downstream work. Ask for a file path if the "
        "user has not provided one. Ask for the target/Y column and feature/X "
        "columns when they are not supplied. Only infer likely X and Y variables "
        "when the user explicitly asks you to infer them. If the user provides a "
        "target/Y column, pass it as target_column. If the user provides feature/X "
        "columns, pass them as x_columns. "
        "Always report the registered file path, inferred task type, Y variable, "
        "X variables, and recommended downstream plan."
    ),
    tools=[intake_dataset_for_downstream],
)


eda_agent = Agent(
    name="eda_agent",
    model=AGENT_MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Generates exploratory data analysis reports including five-point "
        "summaries, moments, robust statistics, correlations, and autocorrelation."
    ),
    instruction=(
        "You are the EDA specialist. Use generate_eda_report when the user asks "
        "for exploratory analysis, five-point summary, moments, robust statistics, "
        "correlation, or ACF. Explain that ACF uses the current row order and is "
        "only time-series meaningful after sorting by time. Always include the "
        "report path, numeric/categorical column counts, strongest correlations, "
        "and important robust-statistics observations."
    ),
    tools=[generate_eda_report],
)


outlier_agent = Agent(
    name="outlier_agent",
    model=AGENT_MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Detects numeric outliers and imputes or caps outlier values in tabular "
        "CSV, JSON, XLSX, and XLSM datasets."
    ),
    instruction=(
        "You are the outlier specialist. Use detect_outliers when the user asks "
        "to inspect, find, count, or explain outliers. Use impute_outliers when "
        "the user asks to clean, impute, cap, drop, or write an outlier-cleaned "
        "dataset. Prefer IQR for general skewed data, modified_zscore for small "
        "or noisy sensor/finance-style data, and zscore only when the user says "
        "the data is approximately normal. Always report the method, columns, "
        "output path when created, and any changed cells or dropped rows."
    ),
    tools=[detect_outliers, impute_outliers],
)


missing_data_agent = Agent(
    name="missing_data_agent",
    model=AGENT_MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Checks missing values and imputes missing data in tabular CSV, JSON, "
        "XLSX, and XLSM datasets."
    ),
    instruction=(
        "You are the missing-data specialist. Use inspect_missing_data when the "
        "user asks to check, profile, count, summarize, or diagnose missing "
        "values. Use impute_missing_data when the user asks to fill, impute, "
        "drop, forward-fill, backward-fill, or write a cleaned dataset. Prefer "
        "strategy='auto' unless the user specifies a strategy: numeric columns "
        "use median and non-numeric columns use mode. Always report missing "
        "counts before and after, output path when created, and any dropped rows "
        "or columns."
    ),
    tools=[inspect_missing_data, impute_missing_data],
)


modeling_agent = Agent(
    name="modeling_agent",
    model=AGENT_MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Trains a candidate list of classification or regression models, "
        "including XGBoost when available, selects the best validation model, "
        "and writes model-selection reports."
    ),
    instruction=(
        "You are the modeling specialist. Use train_and_select_model only after "
        "the user has confirmed the dataset path, X variables, and Y variable. "
        "Prefer the cleaned dataset produced after missing-data imputation, and "
        "after outlier handling if the workflow requested it. If the user has "
        "not provided target_column or x_columns, ask for them instead of "
        "guessing. For classification, report best accuracy plus balanced "
        "accuracy and F1. For regression, report R2, RMSE, and MAE. Always "
        "include the best model name, report path, model artifact path, and a "
        "short caution that this is holdout validation, not final production "
        "validation."
    ),
    tools=[train_and_select_model],
)


root_agent = Agent(
    name="data_quality_coordinator",
    model=AGENT_MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Coordinates dataset intake, missing-data handling, EDA, outlier work, "
        "and model selection by delegating to specialized data agents."
    ),
    instruction=(
        "You are the coordinator for a data-quality agent team. Decide which "
        "specialist should handle the user's request. Delegate dataset upload, "
        "X/Y variable selection, and downstream planning to "
        "data_qc_input_agent. Delegate EDA, five-point summary, moments, robust "
        "statistics, correlation, and ACF requests to eda_agent. Delegate "
        "outlier detection or outlier imputation tasks to outlier_agent. Delegate "
        "missing-value inspection or missing-data imputation tasks to "
        "missing_data_agent. Delegate model training, XGBoost/model comparison, "
        "and best-accuracy selection to modeling_agent. For multi-step requests, "
        "use this order: intake, missing-data handling, EDA, outlier handling "
        "when requested, then modeling. Use the file path provided by the user "
        "exactly when calling tools. If the user does not provide a file path, "
        "ask for one. Keep responses concise and include generated file paths "
        "and report paths."
    ),
    sub_agents=[
        data_qc_input_agent,
        missing_data_agent,
        eda_agent,
        outlier_agent,
        modeling_agent,
    ],
)
