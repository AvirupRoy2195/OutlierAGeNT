# Code and Agent Logic Review

## Scope

Reviewed the Google ADK data-quality implementation in this folder:

- `agent.py`
- `tools/data_io.py`
- `tools/missing_data_tools.py`
- `tools/outlier_tools.py`
- newly added `tools/eda_tools.py`
- newly added `tools/data_qc_tools.py`
- newly added `tools/modeling_tools.py`

## Findings

### Medium: coordinator can delegate, but deterministic multi-step workflow is not guaranteed

ADK `sub_agents` give the root agent a clean routing mechanism, but a free-form LLM coordinator is not the same as a strict pipeline executor. For requests such as "impute missing data, then run outlier handling, then EDA", the coordinator is instructed to follow an order, but the sequence is still LLM-mediated.

Mitigation added: `data_qc_input_agent` now creates an explicit downstream plan and manifest. A future hardening step would be a deterministic workflow agent or one orchestration tool that calls the specialists in fixed order.

### Medium: ADK runtime dependency can conflict with other Google GenAI packages

The project compiles, deterministic tools were smoke-tested locally, and the ADK root agent imports with `google-adk` 2.3.0. Installing ADK upgraded `google-genai` to 2.10.0, which conflicts with `langchain-google-genai` in this environment because that package requires `google-genai<2.0.0`.

Mitigation: keep this ADK project in its own virtual environment if other projects need older `google-genai` versions.

### Low: ACF depends on row order

Autocorrelation is only meaningful when rows are sorted by time or sequence. The EDA tool computes ACF on current row order.

Mitigation added: the EDA report explicitly includes this warning.

### Low: modified z-score can miss edge cases when MAD is zero

The outlier tool currently returns no modified-z-score outliers if the median absolute deviation is zero. This is defensible for a stable column but can miss "mostly constant plus one spike" data.

Recommended follow-up: add a fallback to IQR or exact-deviation detection when MAD is zero and unique values are greater than one.

### Low: generated datasets and reports should stay out of source control

The project can generate cleaned CSVs, reports, and uploaded dataset copies.

Mitigation added: `.gitignore` excludes generated reports, imputed outputs, Excel workbooks, and `Google_ADK/uploads/`.

## Logical Review

### Agent boundaries

The current split is sound:

- LLM agents decide intent, routing, and explanation.
- Python tools perform deterministic data operations.
- Reports are structured dictionaries that can be saved and reused downstream.

This avoids using an LLM for numerical calculations, which is the correct design for statistical work.

### Data intake and planning

The intake agent is the first step in the data workflow. It registers the dataset, captures or infers X/Y variables, infers task type, and proposes the downstream agent order. It does not inspect missing values or detect outliers; those tasks are delegated to the specialist agents.

### Missing-data agent

The missing-data agent handles inspection and common imputation strategies. The `auto` strategy uses median for numeric columns and mode for non-numeric columns, which is a reasonable baseline. Dropping rows/columns remains explicit.

### EDA agent

The EDA agent covers five-point summaries, central moments, skewness/kurtosis, robust descriptive measures, Pearson/Spearman/Kendall correlations, top correlation pairs, categorical summaries, datetime summaries, and ACF. It does not produce missing-value or outlier reports.

### Outlier agent

The outlier agent supports IQR, z-score, and modified z-score, with median/mean/capping/drop strategies. It is auditable because reports include bounds, counts, changed cells, and dropped rows.

### Modeling agent

The modeling agent is a downstream specialist. It consumes a confirmed dataset, X variables, and Y variable after intake, missing-data handling, EDA, and optional outlier handling. It compares a shortlist of scikit-learn models and XGBoost when installed, then saves the best holdout-validation pipeline and a model-selection report.

## Recommended Next Improvements

- Add a deterministic pipeline runner for multi-step requests.
- Add tests with small synthetic datasets for missingness, outliers, EDA, and intake planning.
- Add time-column-aware sorting before ACF when a datetime column is supplied.
- Add optional visualization outputs for EDA.
- Add a manifest schema version so downstream agents can depend on stable keys.
- Add cross-validation and hyperparameter search for the modeling agent before using results for production decisions.
