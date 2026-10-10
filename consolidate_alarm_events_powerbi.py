from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("industrial_motor_data")
READINGS_FILE = DATA_DIR / "motor_readings_enriched.csv"
MAINT_FILE = DATA_DIR / "maintenance_events.csv"
MOTORS_FILE = DATA_DIR / "motors.csv"
EVALUATION_FILE = DATA_DIR / "detection_evaluation.csv"
METRICS_FILE = DATA_DIR / "detection_metrics.csv"
RULE_PERFOMANCE_FILE = DATA_DIR / "rule_performance.csv"

POWERBI_DIR = DATA_DIR / "powerbi_model"
POWERBI_DIR.mkdir(parents=True, exist_ok=True)

maintenance = pd.read_csv(MAINT_FILE, parse_dates=["event_date"])
df = pd.read_csv(READINGS_FILE,parse_dates=["timestamp"])
motors = pd.read_csv(MOTORS_FILE)
evaluation = pd.read_csv(EVALUATION_FILE)
metrics = pd.read_csv(METRICS_FILE)
rule_performance = pd.read_csv(RULE_PERFOMANCE_FILE)


MAX_GAP_HOURS = 6
MIN_EVENT_DURATION_H = 6
CLOSE_PERSISTENCE_HOURS = 12

RULE_CONFIG = {
    "rule_bearing": {
        "condition": "Bearing degradation",
        "category": "Mechanical",
        "recommended_action": "Inspect bearings, lubrication, coupling and mechanical condition.",
    },
    "rule_misalignment": {
        "condition": "Mechanical vibration / misalignment",
        "category": "Mechanical",
        "recommended_action": "Inspect alignment, coupling, supports and mechanical looseness.",
    },
    "rule_overload": {
        "condition": "Mechanical overload",
        "category": "Load",
        "recommended_action": "Inspect driven load, process demand and mechanical resistance.",
    },
    "rule_cooling": {
        "condition": "Cooling degradation",
        "category": "Thermal",
        "recommended_action": "Inspect cooling fan, fins, airflow path and ventilation.",
    },
    "rule_unbalance": {
        "condition": "Electrical unbalance",
        "category": "Electrical",
        "recommended_action": "Inspect supply phases, terminals, connections and phase loading.",
    },
    "rule_temp_sensor": {
        "condition": "Temperature sensor anomaly",
        "category": "Instrumentation",
        "recommended_action": "Inspect temperature sensor, wiring and signal integrity.",
    },
}

def severity_from_score(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 50:
        return "Critical"
    if score >= 20:
        return "Warning"
    return "Advisory"

def get_next_corrective_maintenance(motor_id, event_start):
    candidates = maintenance[
        (maintenance["motor_id"] == motor_id)
        & (maintenance["planned"] == False)
        & (maintenance["event_date"] >= event_start)
    ].sort_values("event_date")
    if candidates.empty:
        return None
    return candidates.iloc[0]

def consolidate_rule_events_hysteresis(df_input, rule_col, cfg):
    events = []

    for motor_id, g in df_input.groupby("motor_id"):
        g = g.sort_values("timestamp").copy()

        event_open = False
        event_start = None
        event_rows = []
        normal_counter = 0

        for _, row in g.iterrows():

            active = bool(row[rule_col])

            if not event_open:

                if active:
                    event_open = True
                    event_start = row["timestamp"]
                    event_rows = [row]
                    normal_counter = 0

            else:

                event_rows.append(row)

                if active:
                    normal_counter = 0

                else:
                    normal_counter += 1

                    if normal_counter >= CLOSE_PERSISTENCE_HOURS:

                        # exclude final normal persistence samples
                        useful_rows = event_rows[:-CLOSE_PERSISTENCE_HOURS]

                        if useful_rows:

                            evt = pd.DataFrame(useful_rows)

                            start_time = evt["timestamp"].min()
                            end_time = evt["timestamp"].max()

                            duration_h = (
                                end_time - start_time
                            ).total_seconds() / 3600 + 1

                            peak_idx = evt["condition_score"].idxmax()
                            peak_row = evt.loc[peak_idx]

                            maint = get_next_corrective_maintenance(
                                motor_id,
                                start_time
                            )

                            if maint is not None:
                                maintenance_date = maint["event_date"]
                                lead_time_days = (
                                    maintenance_date - start_time
                                ).total_seconds() / 86400
                            else:
                                maintenance_date = pd.NaT
                                lead_time_days = np.nan

                            events.append({
                                "motor_id": motor_id,
                                "area": peak_row["area"],
                                "criticality": peak_row["criticality"],
                                "condition_category": cfg["category"],
                                "condition": cfg["condition"],
                                "rule": rule_col,
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration_h": round(duration_h, 1),
                                "max_condition_score": round(
                                    evt["condition_score"].max(), 2
                                ),
                                "severity": severity_from_score(
                                    evt["condition_score"].max()
                                ),
                                "lead_time_days": round(
                                    lead_time_days, 2
                                ) if pd.notna(lead_time_days) else np.nan,
                            })

                        event_open = False
                        event_start = None
                        event_rows = []
                        normal_counter = 0

        # event still open at dataset end
        if event_open and event_rows:

            evt = pd.DataFrame(event_rows)

            start_time = evt["timestamp"].min()
            end_time = evt["timestamp"].max()

            duration_h = (
                end_time - start_time
            ).total_seconds() / 3600 + 1

            peak_idx = evt["condition_score"].idxmax()
            peak_row = evt.loc[peak_idx]

            events.append({
                "motor_id": motor_id,
                "area": peak_row["area"],
                "criticality": peak_row["criticality"],
                "condition_category": cfg["category"],
                "condition": cfg["condition"],
                "rule": rule_col,
                "start_time": start_time,
                "end_time": end_time,
                "duration_h": round(duration_h, 1),
                "max_condition_score": round(
                    evt["condition_score"].max(), 2
                ),
                "severity": severity_from_score(
                    evt["condition_score"].max()
                ),
                "lead_time_days": np.nan,
            })

    return pd.DataFrame(events)

# 1) Consolidate all rule events
event_tables = []

for rule_col, cfg in RULE_CONFIG.items():
    rule_events = consolidate_rule_events_hysteresis(df, rule_col, cfg)
    if not rule_events.empty:
        event_tables.append(rule_events)

alarm_events = (
    pd.concat(event_tables, ignore_index=True)
    if event_tables else
    pd.DataFrame()
)

# 2) Unique event IDs
if not alarm_events.empty:
    alarm_events = alarm_events.sort_values(
        ["start_time", "motor_id", "condition"]
    ).reset_index(drop=True)

    alarm_events["event_id"] = (
        "EVT-" + (alarm_events.index + 1).astype(str).str.zfill(5)
    )

    alarm_events = alarm_events[
        ["event_id"] + [c for c in alarm_events.columns if c != "event_id"]
    ]

# 3) Event status
dataset_end = df["timestamp"].max()

if not alarm_events.empty:
    alarm_events["event_status"] = np.where(
        alarm_events["end_time"] >= dataset_end,
        "Active",
        "Closed"
    )

# 4) Motor-level event summary
if not alarm_events.empty:
    motor_event_summary = (
        alarm_events.groupby("motor_id", as_index=False)
        .agg(
            total_alarm_events=("event_id", "count"),
            critical_events=("severity", lambda s: (s == "Critical").sum()),
            warning_events=("severity", lambda s: (s == "Warning").sum()),
            total_alarm_hours=("duration_h", "sum"),
            max_event_score=("max_condition_score", "max"),
            avg_event_lead_time_days=("lead_time_days", "mean"),
        )
    )
else:
    motor_event_summary = pd.DataFrame({"motor_id": motors["motor_id"]})

# 5) Motor dimension
dim_motors = motors.copy().merge(
    motor_event_summary,
    on="motor_id",
    how="left"
)

for col in ["total_alarm_events", "critical_events", "warning_events", "total_alarm_hours"]:
    if col in dim_motors.columns:
        dim_motors[col] = dim_motors[col].fillna(0)

# 6) Fact motor readings
FACT_READING_COLS = [
    "timestamp",
    "motor_id",
    "status",
    "load_pct",
    "voltage_v",
    "current_a",
    "current_a_phase_a",
    "current_a_phase_b",
    "current_a_phase_c",
    "current_unbalance_pct",
    "current_pct_rated",
    "power_factor",
    "speed_rpm",
    "torque_nm",
    "p_in_kw",
    "p_out_kw",
    "p_loss_kw",
    "ambient_temp_c",
    "temperature_c",
    "temperature_rise_c",
    "vibration_rms_mm_s",
    "runtime_hours",
    "vibration_expected",
    "thermal_rise_expected",
    "vibration_z",
    "thermal_z",
    "vib_slope_day",
    "temp_rise_slope_day",
    "condition_score",
    "condition_state",
    "alarm",
]

fact_motor_readings = df[
    [c for c in FACT_READING_COLS if c in df.columns]
].copy()

# 7) Fact maintenance
fact_maintenance = (
    maintenance.copy()
    .sort_values("event_date")
    .reset_index(drop=True)
)

fact_maintenance["maintenance_event_id"] = (
    "MNT-" + (fact_maintenance.index + 1).astype(str).str.zfill(5)
)

fact_maintenance = fact_maintenance[
    ["maintenance_event_id"]
    + [c for c in fact_maintenance.columns if c != "maintenance_event_id"]
]

# 8) Date dimension
date_start = df["timestamp"].min().normalize()
date_end = df["timestamp"].max().normalize()

dim_date = pd.DataFrame({
    "date": pd.date_range(date_start, date_end, freq="D")
})

dim_date["year"] = dim_date["date"].dt.year
dim_date["month_number"] = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.month_name()
dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
dim_date["day"] = dim_date["date"].dt.day
dim_date["day_of_week_number"] = dim_date["date"].dt.dayofweek + 1
dim_date["day_of_week"] = dim_date["date"].dt.day_name()
dim_date["week"] = dim_date["date"].dt.isocalendar().week.astype(int)
dim_date["quarter"] = "Q" + dim_date["date"].dt.quarter.astype(str)
dim_date["is_weekend"] = dim_date["day_of_week_number"] >= 6

# 9) Date keys
fact_motor_readings["date"] = fact_motor_readings["timestamp"].dt.normalize()

if not alarm_events.empty:
    alarm_events["start_date"] = alarm_events["start_time"].dt.normalize()
    alarm_events["end_date"] = alarm_events["end_time"].dt.normalize()

fact_maintenance["date"] = fact_maintenance["event_date"].dt.normalize()

# 10) Export Power BI model
dim_motors.to_csv(POWERBI_DIR / "dim_motors.csv", index=False)
dim_date.to_csv(POWERBI_DIR / "dim_date.csv", index=False)
fact_motor_readings.to_csv(POWERBI_DIR / "fact_motor_readings.csv", index=False)
alarm_events.to_csv(POWERBI_DIR / "fact_alarm_events.csv", index=False)
fact_maintenance.to_csv(POWERBI_DIR / "fact_maintenance.csv", index=False)
evaluation.to_csv(POWERBI_DIR / "fact_detection_evaluation.csv", index=False)
metrics.to_csv(POWERBI_DIR / "fact_detection_metrics.csv", index=False)
rule_performance.to_csv(POWERBI_DIR / "fact_rule_performance.csv", index=False)

# 11) Validation summary
print("\n======================================")
print("CONSOLIDATED ALARM EVENTS")
print("======================================\n")

if not alarm_events.empty:
    print(
        alarm_events[
            [
                "event_id",
                "motor_id",
                "condition_category",
                "condition",
                "start_time",
                "end_time",
                "duration_h",
                "severity",
                "max_condition_score",
                "lead_time_days",
            ]
        ]
        .sort_values("start_time")
        .to_string(index=False)
    )
else:
    print("No alarm events found.")

print("\n======================================")
print("EVENT SUMMARY")
print("======================================\n")

if not alarm_events.empty:
    print("Total events:", len(alarm_events))
    print("\nEvents by condition:")
    print(alarm_events["condition"].value_counts())

    print("\nEvents by severity:")
    print(alarm_events["severity"].value_counts())

    print("\nEvents by motor:")
    print(alarm_events["motor_id"].value_counts().sort_index())

    valid_lead = alarm_events["lead_time_days"].dropna()

    if not valid_lead.empty:
        print(
            "\nAverage event lead time:",
            round(valid_lead.mean(), 2),
            "days"
        )

print("\n======================================")
print("POWER BI FILES CREATED")
print("======================================\n")

for file in sorted(POWERBI_DIR.glob("*.csv")):
    print("-", file.name)

print("\nRecommended Power BI relationships:")
print("dim_motors[motor_id] 1 -> * fact_motor_readings[motor_id]")
print("dim_motors[motor_id] 1 -> * fact_alarm_events[motor_id]")
print("dim_motors[motor_id] 1 -> * fact_maintenance[motor_id]")
print("dim_date[date] 1 -> * fact_motor_readings[date]")
print("dim_date[date] 1 -> * fact_alarm_events[start_date]")
print("dim_date[date] 1 -> * fact_maintenance[date]")
