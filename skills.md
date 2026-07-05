# Outlier Agent Skills

This file describes subject-matter method choices that the agent can use when
planning outlier detection and imputation.

## Core Statistical Skills

### IQR Rule

- Use for skewed, ordinal-like, or unknown numeric distributions.
- Works well as a conservative default for tabular business data.
- Typical threshold: 1.5 times IQR.
- Imputation options: median replacement, capping to bounds, or detection only.

### Z-Score

- Use when the variable is approximately normally distributed.
- Needs enough rows for the mean and standard deviation to be meaningful.
- Typical threshold: absolute z-score greater than 3.0.
- Avoid when a few extreme points can distort the mean and standard deviation.

### Modified Z-Score

- Use when the dataset is small, noisy, or already contaminated by extreme values.
- Uses median and median absolute deviation.
- Typical threshold: absolute modified z-score greater than 3.5.
- Good for finance, sensor data, and operational telemetry with spikes.

## Domain Skills

### General Tabular Data

- Default method: IQR.
- Default imputation: median.
- Rationale: robust and easy to explain without assuming normality.

### Finance

- Prefer modified z-score or domain-specific winsorization.
- Prefer capping over deletion when preserving time periods or instruments matters.
- Never remove extreme market moves without confirming whether they are data errors.

### Healthcare and Clinical Data

- Prefer detection-only unless a clinician or domain rule approves imputation.
- Flag values with context because rare values can be clinically meaningful.
- Keep an audit report of every changed cell.

### Sensor, IoT, and Telemetry

- Prefer modified z-score for isolated spikes.
- Consider rolling-window methods for time series when timestamp order is available.
- Median replacement is acceptable for isolated faults; longer faulty intervals should be flagged.

### Manufacturing and Quality Control

- Prefer z-score when the process is stable and approximately normal.
- Prefer domain control limits when they are available.
- Capping can be acceptable for downstream modeling, but reporting should retain original values.

### Survey and Behavioral Data

- Prefer IQR.
- Treat bounded scales carefully because high or low values can be valid responses.
- Median replacement is usually safer than mean replacement.

### Environmental Data

- Prefer modified z-score for sensor spikes and IQR for seasonal aggregates.
- Do not impute extreme weather, pollution, or event readings without external validation.
- Record whether the outlier is likely a sensor error or a real event.

## Agent Behavior

- Select a method from the data profile, domain profile, and user instructions.
- Explain why the method and imputation strategy were selected.
- Preserve the original dataset in memory and write cleaned output separately.
- Return a structured report with bounds, outlier counts, changed cells, and dropped rows.
- If the LLM is unavailable, use the context defaults and domain profiles.
