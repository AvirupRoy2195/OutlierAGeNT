# OutlierAGeNT: End-to-End Tabular Data Quality & Modeling

OutlierAGeNT is a multi-agent framework designed to automate dataset profiling, missing-value imputation, exploratory data analysis (EDA), outlier handling, and machine learning model selection on tabular datasets. It combines **LLM-assisted planning** (via the Google Agent Development Kit - ADK) with **deterministic statistical tools** to ensure accuracy, transparency, and safety.

---

## 1. System Architecture (Agent-to-Agent / A2A)

The core architecture follows an **Agent-to-Agent (A2A) orchestration model**. A root coordinator agent acts as the brain, parsing user instructions and delegating specialized subtasks to independent, narrow-scope specialist agents. This modular division prevents LLM context bloat and ensures high-quality planning.

### Architecture Diagram

```mermaid
graph TD
    User([User Query / CLI / Web UI]) --> Coordinator[data_quality_coordinator <br/> Root Agent]
    
    subgraph A2A ["Agent-to-Agent (A2A) Coordination Layer"]
        Coordinator --> InputAgent[data_qc_input_agent]
        Coordinator --> MissingAgent[missing_data_agent]
        Coordinator --> EDAAgent[eda_agent]
        Coordinator --> OutlierAgent[outlier_agent]
        Coordinator --> ModelingAgent[modeling_agent]
    end

    subgraph Tools ["Deterministic Tool Execution Layer"]
        InputAgent -->|Calls| T_intake[data_qc_tools.py]
        MissingAgent -->|Calls| T_missing[missing_data_tools.py]
        EDAAgent -->|Calls| T_eda[eda_tools.py]
        OutlierAgent -->|Calls| T_outlier[outlier_tools.py]
        ModelingAgent -->|Calls| T_model[modeling_tools.py]
    end

    subgraph Data ["Data Storage & Manifests"]
        T_intake -->|Generates| Manifest[intake_manifest.json]
        T_missing -->|Saves| CleanedMissing[missing_imputed.csv]
        T_eda -->|Saves| EDAReport[eda_report.json]
        T_outlier -->|Saves| CleanedOutliers[outliers_imputed.csv]
        T_model -->|Saves| ModelReport[model_report.json & best_model.joblib]
    end

    classDef agent fill:#d1e7dd,stroke:#0f5132,stroke-width:2px;
    classDef tool fill:#fff3cd,stroke:#664d03,stroke-width:1px;
    classDef data fill:#cfe2ff,stroke:#084298,stroke-width:1px;
    
    class Coordinator,InputAgent,MissingAgent,EDAAgent,OutlierAgent,ModelingAgent agent;
    class T_intake,T_missing,T_eda,T_outlier,T_model tool;
    class Manifest,CleanedMissing,EDAReport,CleanedOutliers,ModelReport data;
```

### Agent Roles & Specialist Capabilities

1. **`data_quality_coordinator` (Root Coordinator)**: The entry point. It understands natural language requests, determines the sequential workflow dependencies, and orchestrates execution by routing requests to correct specialists.
2. **`data_qc_input_agent` (Dataset Intake)**: Registers raw datasets, infers column schemas, matches or recommends target (Y) and feature (X) variables, and publishes a structured downstream manifest.
3. **`missing_data_agent` (Missing Data Specialist)**: Identifies missing entries and applies appropriate imputation strategies (e.g., median for numeric, mode for categorical, or row-dropping).
4. **`eda_agent` (EDA Specialist)**: Generates descriptive summaries, skewness, kurtosis, robust metrics, Pearson/Spearman correlations, and Autocorrelation (ACF) diagnostics.
5. **`outlier_agent` (Outlier Specialist)**: Detects anomalies using IQR, Z-score, or Modified Z-score methods, and cleans them using mean/median imputation, winsorization (capping), or row drop.
6. **`modeling_agent` (ML Specialist)**: Evaluates a shortlist of scikit-learn models (and XGBoost when available) using holdout cross-validation, selecting and exporting the highest-performing pipeline.

---

## 2. End-to-End Data Flow

Data flows sequentially between agents via structured file handoffs. Each agent updates the manifest and outputs a new state of the dataset, preventing downstream contamination.

### Data Flow Diagram

```mermaid
flowchart LR
    Raw[Raw Dataset] -->|Intake & Profile| InputAgent[data_qc_input_agent]
    InputAgent -->|Generates| Manifest[intake_manifest.json]

    Raw -->|Read file & columns| MissingAgent[missing_data_agent]
    MissingAgent -->|Impute missing values| ImputedCSV[(missing_imputed.csv)]

    ImputedCSV -->|Generate stats & ACF| EDAAgent[eda_agent]
    EDAAgent -->|Write summary| EDAReport[eda_report.json]

    ImputedCSV -->|Detect & clean outliers| OutlierAgent[outlier_agent]
    OutlierAgent -->|Impute/cap values| CleanedCSV[(outliers_imputed.csv)]

    CleanedCSV -->|Train pipelines & XGBoost| ModelingAgent[modeling_agent]
    ModelingAgent -->|Validate & select| BestModel[best_model.joblib]
    ModelingAgent -->|Write metrics| ModelReport[model_report.json]
```

---

## 3. Execution Paths

The framework supports three operating modes:

### Path A: LLM-Driven Multi-Agent Execution (ADK)
Enables natural language team collaboration where agents autonomously call Python tools.
```powershell
# From the project root
adk run Google_ADK "Upload Forest Fire Smoke Dataset.xlsx, use fire_label as Y, and build a downstream plan"
```

### Path B: Deterministic Offline Pipeline (Batch mode)
Runs the identical underlying tools in a hardcoded pipeline without any LLM API requirements.
```powershell
python Google_ADK/run_offline_pipeline.py "Google_ADK/Forest Fire Smoke Dataset.xlsx" --target fire_label
```

### Path C: Single-File Outlier Specialist (Offline/LLM Hybrid)
A standalone CLI tool focused purely on tabular outliers (no modeling/missing-data features).
```powershell
# Run the local sensor demo
python Outlier_agent.py --no-llm --print-records
```

---

## 4. Setup & Installation

### Prerequisite Installation

```powershell
# Install core dependencies
pip install -r requirements.txt

# Install Google ADK specific dependencies
pip install -r Google_ADK/requirements.txt
```

### Environment Configuration
Copy the env templates and provide your API credentials:

```powershell
# Root special outlier agent env
Copy-Item .env.example .env

# Google ADK agent env
Copy-Item Google_ADK/.env.example Google_ADK/.env
```

Fill in `Google_ADK/.env` for Gemini model access:
```env
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_ADK_MODEL=gemini-2.0-flash
```

Or fill in `.env` for root OpenAI client access:
```env
OUTLIER_AGENT_API_KEY=your_openai_api_key
OUTLIER_AGENT_ENDPOINT=https://api.openai.com/v1
OUTLIER_AGENT_MODEL=gpt-4.1-mini
```
