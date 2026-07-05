# Data Quality Agents — OpenAI + A2A edition

This is the OpenAI / LiteLLM port of the Google ADK data-quality agent team.
The deterministic tools under `tools/` are unchanged — only the LLM driver
and the A2A transport layer are swapped in.

## What changed vs. the Gemini version

| Concern | Before (Gemini) | After (OpenAI + A2A) |
|---|---|---|
| LLM driver | `google.adk.agents.Agent(model="gemini-2.0-flash")` | `Agent(model=LiteLlm(model="openai/gpt-4o-mini"))` |
| Auth | `GOOGLE_API_KEY` | `OPENAI_API_KEY` (or Azure / custom endpoint) |
| `generate_content_config` | `google.genai.types.GenerateContentConfig` | _dropped — LiteLLM handles provider-specific config internally_ |
| Cross-agent transport | `sub_agents=[...]` (intra-process) | `sub_agents=[...]` **plus** `to_a2a(root_agent)` for cross-process A2A |

The agent team itself (coordinator + 5 specialists + their tools) is
identical — same tools, same prompts, same reports.

## Agents

- `data_quality_coordinator` — root agent, delegates to specialists.
- `data_qc_input_agent` — registers datasets, infers X/Y, builds a plan.
- `eda_agent` — EDA summaries, moments, robust stats, correlations, ACF.
- `outlier_agent` — outlier detection / imputation (IQR, zscore, modified zscore).
- `missing_data_agent` — missing-value inspection / imputation.
- `modeling_agent` — trains candidate models (incl. XGBoost if installed), picks best on holdout.

## Setup

```bash
cd Google_ADK
python -m pip install -r requirements.txt
cp .env.example .env
# fill in your OPENAI_API_KEY (and optionally OPENAI_BASE_URL / ADK_MODEL)
```

### Supported model identifiers

Anything LiteLLM understands under the `openai/` prefix works:

```env
ADK_MODEL=openai/gpt-4o-mini        # default — cheap + capable
ADK_MODEL=openai/gpt-4o             # stronger reasoning
ADK_MODEL=openai/o4-mini            # reasoning model
ADK_MODEL=azure/openai/<deployment> # Azure OpenAI
ADK_MODEL=openai/<name>             # any OpenAI-compatible endpoint behind OPENAI_BASE_URL
```

You can also decouple coordinator and specialists:

```env
ADK_COORDINATOR_MODEL=openai/gpt-4o
ADK_SPECIALIST_MODEL=openai/gpt-4o-mini
```

## Run the ADK dev UI (intra-process agents)

From the parent folder containing `Google_ADK/`:

```bash
adk run Google_ADK
# or
adk web --port 8000
```

## Run as an A2A server (cross-process / cross-vendor agents)

```bash
# quick local start
python -m Google_ADK.a2a_server --port 8001

# production-style
uvicorn Google_ADK.a2a_server:a2a_app --host 0.0.0.0 --port 8001
```

Once running, the Agent Card is served at:

```
http://localhost:8001/.well-known/agent-card.json
```

Other A2A-compatible agents (LangGraph, CrewAI, AG2, custom…) can discover
and invoke the coordinator via the standard `tasks/send` /
`message/send` JSON-RPC endpoints.

> Install the A2A extras explicitly if needed:
> `pip install 'google-adk[a2a]' uvicorn[standard]`

## Deterministic pipeline (no LLM at all)

The tool layer doesn't require an LLM. To run the full pipeline offline
without OpenAI / Gemini / any agent runtime:

```bash
python Google_ADK/run_offline_pipeline.py "Google_ADK/Forest Fire Smoke Dataset.xlsx" --target fire_label
```

## Example prompts

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

Supported input formats: `.csv`, `.json`, `.xlsx`, `.xlsm`.

## Troubleshooting

- **`AuthenticationError: No API key provided`** — `OPENAI_API_KEY` is not
  set in the environment. Source `.env` before running, or `export` it in
  your shell.
- **`NotFoundError: model 'openai/...'`** — LiteLLM couldn't resolve the
  model name. Check `ADK_MODEL`, your `OPENAI_API_KEY` permissions, or
  `OPENAI_BASE_URL` if you're using a custom endpoint.
- **`to_a2a()` import fails** — install the A2A extras:
  `pip install 'google-adk[a2a]'`
- **429 from OpenAI** — rate limit hit. Switch to a smaller model
  (`openai/gpt-4o-mini`) or back off.