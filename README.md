# Industrial Motor Condition Monitoring

A monitoring system for identifying motors that show signs of degradation, determining which motors need attention, and tracking how their condition changes over time. The project combines data simulation, signal analysis, alarm evaluation, and data preparation for Power BI.

The current implementation processes local CSV files. The simulator generates synthetic data for 30 three-phase induction motors over 90 days, starting June 1, 2026. It uses internal 10-minute simulation steps, exports hourly readings, and uses a random seed of 42.

## Features

- Track temperature, vibration, load, phase currents, electrical unbalance, and operating hours.
- Detect bearing wear, misalignment, mechanical overload, cooling degradation, electrical unbalance, and temperature sensor anomalies.
- Classify motor condition and summarize the latest state of each motor.
- Evaluate detections, false alarms, alarm persistence, and lead time before simulated corrective maintenance.
- Consolidate alarm events and export tables for Power BI.

## Scripts

| File | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| `generate_motor_data.py` | Simulate the fleet, progressive faults, and maintenance. | Parameters defined in the code. | `motors.csv`, `motor_readings.csv`, `maintenance_events.csv`. |
| `analyze_motor_condition.py` | Estimate normal behavior and apply detection rules. | The three simulator CSV files. | `motor_readings_enriched.csv`, `fleet_latest_condition.csv`. |
| `comparison_model_events.py` | Evaluate detections against simulated faults. | Enriched readings and maintenance records. | `detection_evaluation.csv`, `detection_metrics.csv`, `rule_performance.csv`, `healthy_motor_false_alarms.csv`. |
| `consolidate_alarm_events_powerbi.py` | Group alarms into events and prepare the data model. | Enriched readings, motor metadata, maintenance, and the first three evaluation CSV files. | Eight CSV tables in `industrial_motor_data/powerbi_model/`. |
| `python_plot_motor.py` | Plot the history of the six motors with simulated faults. | Enriched readings and maintenance records. | Matplotlib windows; `plot_motor(..., save=True)` also saves PNG files. |

The first three stages write their CSV files to `industrial_motor_data/`. The project also includes `industrial_motor_dashboard.pbix`.

## Installation

Python 3 and pip are required. External dependencies are NumPy, pandas, and Matplotlib. `pathlib` and `dataclasses` are part of the Python standard library. `requirements.txt` does not pin package versions because the project does not declare a validated version set.

From the project root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Power BI Desktop is required to open the `.pbix` file; it is not installed through pip.

## Running the Pipeline

Run the scripts from the project root because data paths are relative. Use this order:

```powershell
.\.venv\Scripts\python.exe generate_motor_data.py
.\.venv\Scripts\python.exe analyze_motor_condition.py
.\.venv\Scripts\python.exe comparison_model_events.py
.\.venv\Scripts\python.exe consolidate_alarm_events_powerbi.py
```

Each stage overwrites existing output files with the same names. The simulator may take some time because it computes internal steps for the entire fleet. The scripts do not accept command-line arguments; their parameters are defined in the code.

### Plots

`python_plot_motor.py` uses `load_pct_med24`, `current_pct_rated_med24`, and `current_unbalance_pct_med24`. If these columns are missing from the enriched CSV, it reconstructs them from the original signals, sorted by motor and timestamp, using rolling medians over 24 samples with at least 12 observations, matching the analysis script. Existing median columns are preserved.

After running the analysis, open the plots with:

```powershell
.\.venv\Scripts\python.exe python_plot_motor.py
```

The current plotting calls use `save=False`, so images are not saved automatically.

## Interpreting Results

The analysis builds a baseline from the first 30 days. It fits vibration against load and temperature rise against power losses, then calculates robust deviations. It uses rolling medians over 24 samples and trends over 72 samples; for hourly data, these correspond to 24 and 72 hours.

`condition_score` ranges from 0 to 100: higher values indicate greater deterioration. `condition_state` is assigned as follows:

| State | Score |
| --- | --- |
| `Healthy` | 0 through 20, inclusive. |
| `Warning` | Greater than 20 and up to 50. |
| `Critical` | Greater than 50 and up to 100. |

`alarm` indicates that at least one rule is active, and `detected_condition` describes the detected conditions. Alarms and the score are calculated separately. `fleet_latest_condition.csv` sorts the latest reading from each motor by descending score to support review.

The simulator also generates `health_score`, where higher values indicate better health. This is a separate index from `condition_score` and uses different thresholds.

The evaluation requires six consecutive hourly samples with the expected rule active to count as a persistent detection. A healthy motor is counted as a false positive if it accumulates at least six alarm samples. Simulated fault labels are used for evaluation; the analysis rules rely on measured signals.

Event consolidation opens an event when a rule becomes active and closes it after 12 consecutive samples with that rule inactive, excluding those final samples from the event interval. Event severity uses its own thresholds: `Advisory` below 20, `Warning` from 20 to below 50, and `Critical` from 50 onward. `MAX_GAP_HOURS` and `MIN_EVENT_DURATION_H` are declared but are not applied by the current consolidation logic.

## Power BI Model

The following files are generated in `industrial_motor_data/powerbi_model/`:

- `dim_motors.csv`: motor metadata and event summaries.
- `dim_date.csv`: daily calendar.
- `fact_motor_readings.csv`: signals and condition indicators for each reading.
- `fact_alarm_events.csv`: events, duration, severity, and maintenance lead time.
- `fact_maintenance.csv`: maintenance interventions.
- `fact_detection_evaluation.csv`: evaluation for each motor with a simulated fault.
- `fact_detection_metrics.csv`: overall metrics.
- `fact_rule_performance.csv`: performance by rule.

Configure one-to-many relationships from `dim_motors[motor_id]` to readings, alarms, and maintenance. Relate `dim_date[date]` to `fact_motor_readings[date]`, `fact_alarm_events[start_date]`, and `fact_maintenance[date]`. Review the data source paths in the `.pbix` file so they match the local project location.

## Scope

This is a demonstration using synthetic data and project-specific thresholds. It does not include real-time acquisition or sensor connections. Windows and durations assume regular hourly readings. Before using real measurements, review the sampling frequency, data quality, initial baseline, and thresholds.
