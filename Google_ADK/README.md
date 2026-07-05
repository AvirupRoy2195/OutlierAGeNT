# Google ADK Data Quality Agents

This folder contains a Google ADK agent team for tabular data-quality work.

## Agents

- `data_quality_coordinator`: root ADK agent that delegates work.
- `data_qc_input_agent`: registers/uploads datasets, infers X/Y variables, and creates a downstream plan.
- `eda_agent`: computes EDA summaries, moments, robust statistics, correlations, and ACF.
- `outlier_agent`: detects and imputes numeric outliers.
- `missing_data_agent`: checks and imputes missing values.
- `modeling_agent`: trains candidate models, including XGBoost when available, and reports the best validation model.

## Setup

From this folder:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_ADK_MODEL=gemini-2.0-flash
```

If the web agent returns `429 RESOURCE_EXHAUSTED`, your Gemini quota is exhausted
for the selected model. Wait for quota reset, enable billing/request quota, or
switch `GOOGLE_ADK_MODEL` to another available Gemini model. The deterministic
tools can also be run without any Gemini calls through `run_offline_pipeline.py`.

## Run

From the parent folder that contains `Google_ADK`:

```powershell
adk run Google_ADK
```

Or use the browser UI:

```powershell
adk web --port 8000
```

Run the deterministic pipeline without Gemini or ADK:

```powershell
python Google_ADK/run_offline_pipeline.py "Google_ADK/Forest Fire Smoke Dataset.xlsx" --target fire_label
```

## Example Prompts

```text
Upload Forest Fire Smoke Dataset.xlsx, use fire_label as Y, use all other non-ID columns as X, and create a downstream plan.
```

```text
Generate an EDA report for Forest Fire Smoke Dataset.xlsx with five point summary, moments, robust statistics, correlations, and ACF.
```

```text
Check missing data in Forest Fire Smoke Dataset.xlsx.
```

```text
Impute missing data in Forest Fire Smoke Dataset.xlsx using auto strategy and save it as fire_missing_cleaned.csv.
```

```text
Detect outliers in Forest Fire Smoke Dataset.xlsx using modified_zscore.
```

```text
First impute missing data in Forest Fire Smoke Dataset.xlsx, then detect outliers in the cleaned file.
```

```text
Train classification models on Forest Fire Smoke Dataset.xlsx using fire_label as Y and all other columns as X, then report the best accuracy.
```

Supported file formats: `.csv`, `.json`, `.xlsx`, and `.xlsm`.
