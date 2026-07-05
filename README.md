# OutlierAGeNT

OutlierAGeNT detects, explains, and imputes outliers in tabular datasets. It
can run fully offline with rule-based planning, or use an LLM as the planning
brain when API settings are configured.

## Files

- `Outlier_agent.py` - agent brain and CLI.
- `outlier.py` - deterministic detection and imputation tool.
- `context.json` - agent defaults, LLM settings, and domain profiles.
- `skills.md` - subject-matter method guidance.
- `.env.example` - template for local LLM credentials.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Then fill in `.env`:

```env
OUTLIER_AGENT_API_KEY=your_api_key
OUTLIER_AGENT_ENDPOINT=https://api.openai.com/v1
OUTLIER_AGENT_MODEL=gpt-4.1-mini
```

## Usage

Run the built-in demo without an LLM:

```powershell
python Outlier_agent.py --no-llm --print-records
```

Run on a dataset:

```powershell
python Outlier_agent.py "data.xlsx" --output cleaned.csv --report report.json
```

Force a specific method and imputation:

```powershell
python Outlier_agent.py "data.csv" --method iqr --imputation median
```

Supported input formats: `.csv`, `.json`, `.xlsx`, and `.xlsm`.

