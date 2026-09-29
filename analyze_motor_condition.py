from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("industrial_motor_data")
READINGS_FILE = DATA_DIR / "motor_readings.csv"
MOTORS_FILE = DATA_DIR / "motors.csv"
MAINT_FILE = DATA_DIR / "maintenance_events.csv"

BASELINE_DAYS = 30
ROLL_H = 24
TREND_H = 72

TH = {
    "vib_z_warning": 3.0,
    "vib_z_critical": 5.0,
    "thermal_z_warning": 3.0,
    "thermal_z_critical": 5.0,
    "current_pct_warning": 105.0,
    "current_pct_critical": 120.0,
    "unbalance_warning": 5.0,
    "unbalance_critical": 10.0,
    "temp_jump_c_per_h": 15.0,
    "temp_min_c": 5.0,
    "temp_max_c": 130.0,
    "trend_vib_mm_s_day": 0.10,
    "trend_temp_c_day": 0.50,
}

def robust_scale_from_baseline(s):
    med = s.median()
    mad = (s - med).abs().median()
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = max(float(s.std(ddof=0)), 1e-6)
    return float(med), float(sigma)

def rolling_slope(values):
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < max(6, len(y) // 2):
        return np.nan
    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return ((x - x_mean) * (y - y_mean)).sum() / denom

def add_rolling_features(g):
    motor = g.name
    if "motor_id" not in g.columns:
        g["motor_id"] = motor

    g = g.sort_values("timestamp").copy()

    g["temperature_rise_c"] = g["temperature_c"] - g["ambient_temp_c"]
    g["phase_current_spread_a"] = (
        g[["current_a_phase_a", "current_a_phase_b", "current_a_phase_c"]].max(axis=1)
        - g[["current_a_phase_a", "current_a_phase_b", "current_a_phase_c"]].min(axis=1)
    )

    cols = [
        "temperature_rise_c",
        "vibration_rms_mm_s",
        "current_pct_rated",
        "current_unbalance_pct",
        "load_pct",
        "p_loss_kw",
    ]

    for col in cols:
        g[f"{col}_med24"] = g[col].rolling(
            ROLL_H, min_periods=max(6, ROLL_H // 2)
        ).median()

    g["vib_slope_h"] = g["vibration_rms_mm_s"].rolling(
        TREND_H, min_periods=24
    ).apply(rolling_slope, raw=False)

    g["temp_rise_slope_h"] = g["temperature_rise_c"].rolling(
        TREND_H, min_periods=24
    ).apply(rolling_slope, raw=False)

    g["vib_slope_day"] = 24 * g["vib_slope_h"]
    g["temp_rise_slope_day"] = 24 * g["temp_rise_slope_h"]

    dt_h = g["timestamp"].diff().dt.total_seconds().div(3600)
    g["temp_rate_c_h"] = g["temperature_c"].diff().div(dt_h)

    return g

df = pd.read_csv(READINGS_FILE, parse_dates=["timestamp"])
motors = pd.read_csv(MOTORS_FILE)
maintenance = pd.read_csv(MAINT_FILE, parse_dates=["event_date"])

df = (
    df.sort_values(["motor_id", "timestamp"])
      .groupby("motor_id", group_keys=False)
      .apply(add_rolling_features)
      .reset_index(drop=True)
)

start_date = df["timestamp"].min()
baseline_end = start_date + pd.Timedelta(days=BASELINE_DAYS)

baseline_rows = []

for motor_id, g in df.groupby("motor_id"):
    b = g[
        (g["timestamp"] < baseline_end)
        & (g["status"] == "Running")
    ].copy()

    vib_fit = b.dropna(subset=["load_pct", "vibration_rms_mm_s"])
    if len(vib_fit) >= 20:
        vib_coef = np.polyfit(vib_fit["load_pct"], vib_fit["vibration_rms_mm_s"], 1)
    else:
        vib_coef = [0.0, b["vibration_rms_mm_s"].median()]

    thermal_fit = b.dropna(subset=["p_loss_kw", "temperature_rise_c"])
    if len(thermal_fit) >= 20:
        thermal_coef = np.polyfit(thermal_fit["p_loss_kw"], thermal_fit["temperature_rise_c"], 1)
    else:
        thermal_coef = [0.0, b["temperature_rise_c"].median()]

    vib_pred = np.polyval(vib_coef, vib_fit["load_pct"])
    vib_resid = vib_fit["vibration_rms_mm_s"] - vib_pred
    vib_resid_med, vib_resid_sigma = robust_scale_from_baseline(vib_resid)

    th_pred = np.polyval(thermal_coef, thermal_fit["p_loss_kw"])
    th_resid = thermal_fit["temperature_rise_c"] - th_pred
    th_resid_med, th_resid_sigma = robust_scale_from_baseline(th_resid)

    baseline_rows.append({
        "motor_id": motor_id,
        "vib_load_a": vib_coef[0],
        "vib_load_b": vib_coef[1],
        "vib_resid_med": vib_resid_med,
        "vib_resid_sigma": vib_resid_sigma,
        "thermal_loss_a": thermal_coef[0],
        "thermal_loss_b": thermal_coef[1],
        "thermal_resid_med": th_resid_med,
        "thermal_resid_sigma": th_resid_sigma,
    })

baseline = pd.DataFrame(baseline_rows)
df = df.merge(baseline, on="motor_id", how="left")

df["vibration_expected"] = df["vib_load_a"] * df["load_pct"] + df["vib_load_b"]
df["thermal_rise_expected"] = df["thermal_loss_a"] * df["p_loss_kw"] + df["thermal_loss_b"]

df["vibration_residual"] = df["vibration_rms_mm_s"] - df["vibration_expected"]
df["thermal_residual"] = df["temperature_rise_c"] - df["thermal_rise_expected"]

df["vibration_z"] = (
    (df["vibration_residual"] - df["vib_resid_med"]) / df["vib_resid_sigma"]
)
df["thermal_z"] = (
    (df["thermal_residual"] - df["thermal_resid_med"]) / df["thermal_resid_sigma"]
)

df["vibration_z_med24"] = (
    df.groupby("motor_id")["vibration_z"]
      .transform(lambda s: s.rolling(24, min_periods=8).median())
)
df["thermal_z_med24"] = (
    df.groupby("motor_id")["thermal_z"]
      .transform(lambda s: s.rolling(24, min_periods=8).median())
)

running = df["status"].eq("Running")

df["rule_bearing"] = (
    running
    & (df["vibration_z_med24"] >= TH["vib_z_warning"])
    & (df["vib_slope_day"] >= TH["trend_vib_mm_s_day"])
    & (
        (df["thermal_z_med24"] >= 1.5)
        | (df["temp_rise_slope_day"] >= TH["trend_temp_c_day"])
    )
    & (
        df["current_unbalance_pct_med24"]
        < TH["unbalance_warning"]
    )
)

df["rule_misalignment"] = (
    running
    & (df["vibration_z_med24"] >= TH["vib_z_warning"])
    & (df["vibration_rms_mm_s_med24"] >= 3.5)
    & (df["current_unbalance_pct_med24"] < TH["unbalance_warning"])
)

df["current_pct_rated_med6"] = (
    df.groupby("motor_id")["current_pct_rated"]
      .transform(lambda s: s.rolling(6, min_periods=3).median())
)

df["load_pct_med6"] = (
    df.groupby("motor_id")["load_pct"]
      .transform(lambda s: s.rolling(6, min_periods=3).median())
)

df["rule_overload"] = (
    running
    & (
        (df["current_pct_rated_med6"] >= 103)
        | (df["load_pct_med6"] >= 98)
    )
    & (
        (df["thermal_z_med24"] >= 0.5)
        | (df["temperature_rise_c"] >= 40)
    )
)

df["rule_cooling"] = (
    running
    & (df["thermal_z_med24"] >= TH["thermal_z_warning"])
    & (df["current_pct_rated_med24"] < TH["current_pct_warning"])
    & (df["vibration_z_med24"] < TH["vib_z_warning"])
)

df["rule_unbalance"] = (
    running
    & (df["current_unbalance_pct_med24"] >= TH["unbalance_warning"])
)

df["running_3h"] = (
    df.groupby("motor_id")["status"]
      .transform(
          lambda s:
          s.eq("Running")
           .astype(int)
           .rolling(3, min_periods=3)
           .sum()
      )
)

df["stable_running"] = (
    df["running_3h"] == 3
)

df["temp_jump_c"] = (
    df.groupby("motor_id")["temperature_c"]
      .diff()
)

df["ambient_jump_c"] = (
    df.groupby("motor_id")["ambient_temp_c"]
      .diff()
)

df["unexplained_temp_jump_c"] = (
    df["temp_jump_c"] - df["ambient_jump_c"]
)

df["rule_temp_sensor"] = (
    (df["temperature_c"] < 5)
    | (df["temperature_c"] > 130)
    | (
        df["stable_running"]
        & (df["unexplained_temp_jump_c"].abs() >= 15)
    )
)

rule_cols = [
    "rule_bearing",
    "rule_misalignment",
    "rule_overload",
    "rule_cooling",
    "rule_unbalance",
    "rule_temp_sensor",
]
df["n_rules_active"] = df[rule_cols].sum(axis=1)

def detected_condition(row):
    hits = []
    if row["rule_bearing"]:
        hits.append("Bearing degradation")
    if row["rule_misalignment"]:
        hits.append("Misalignment / mechanical vibration")
    if row["rule_overload"]:
        hits.append("Mechanical overload")
    if row["rule_cooling"]:
        hits.append("Cooling degradation")
    if row["rule_unbalance"]:
        hits.append("Electrical unbalance")
    if row["rule_temp_sensor"]:
        hits.append("Temperature sensor anomaly")
    return " | ".join(hits) if hits else "Normal"

df["detected_condition"] = df.apply(detected_condition, axis=1)
df["alarm"] = df["n_rules_active"] > 0

df["condition_score"] = 0.0
df["condition_score"] += np.clip(df["vibration_z_med24"].fillna(0), 0, 10) * 4
df["condition_score"] += np.clip(df["thermal_z_med24"].fillna(0), 0, 10) * 3
df["condition_score"] += np.clip(
    (df["current_pct_rated_med24"].fillna(0) - 100) / 5, 0, 10
) * 3
df["condition_score"] += np.clip(
    (df["current_unbalance_pct_med24"].fillna(0) - 2), 0, 15
) * 3
df["condition_score"] += df["rule_temp_sensor"].astype(int) * 25
df["condition_score"] = df["condition_score"].clip(0, 100)

df["condition_state"] = pd.cut(
    df["condition_score"],
    bins=[-0.1, 20, 50, 100],
    labels=["Healthy", "Warning", "Critical"]
)

def fleet_summary():
    latest = (
        df.sort_values("timestamp")
          .groupby("motor_id", as_index=False)
          .tail(1)
    )
    return latest[[
        "motor_id",
        "area",
        "criticality",
        "condition_score",
        "condition_state",
        "detected_condition",
        "temperature_c",
        "temperature_rise_c",
        "vibration_rms_mm_s",
        "current_pct_rated",
        "current_unbalance_pct",
        "load_pct",
    ]].sort_values("condition_score", ascending=False)

EXPORT_COLS = [
    "timestamp",
    "motor_id",
    "area",
    "criticality",
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
    "rule_bearing",
    "rule_misalignment",
    "rule_overload",
    "rule_cooling",
    "rule_unbalance",
    "rule_temp_sensor",
    "n_rules_active",
    "detected_condition",
    "alarm",
    "condition_score",
    "condition_state",
]

df[EXPORT_COLS].to_csv(DATA_DIR / "motor_readings_enriched.csv", index=False)
fleet_summary().to_csv(DATA_DIR / "fleet_latest_condition.csv", index=False)

print("\nBlind detection count by rule:")
print(df[rule_cols].sum().sort_values(ascending=False))

print("\nLatest fleet condition:")
print(fleet_summary().head(15).to_string(index=False))

# ============================================================
# FINAL EVALUATION
# ============================================================

# Ground truth ONLY for evaluation.
# Do not use this dictionary anywhere in the detection logic.
GROUND_TRUTH = {
    "MTR-005": {
        "fault": "Electrical unbalance",
        "expected_rule": "rule_unbalance",
    },
    "MTR-007": {
        "fault": "Cooling degradation",
        "expected_rule": "rule_cooling",
    },
    "MTR-013": {
        "fault": "Bearing wear",
        "expected_rule": "rule_bearing",
    },
    "MTR-019": {
        "fault": "Mechanical overload",
        "expected_rule": "rule_overload",
    },
    "MTR-024": {
        "fault": "Misalignment",
        "expected_rule": "rule_misalignment",
    },
    "MTR-028": {
        "fault": "Temperature sensor fault",
        "expected_rule": "rule_temp_sensor",
    },
}

RULE_LABELS = {
    "rule_bearing": "Bearing degradation",
    "rule_misalignment": "Mechanical vibration / misalignment",
    "rule_overload": "Mechanical overload",
    "rule_cooling": "Cooling degradation",
    "rule_unbalance": "Electrical unbalance",
    "rule_temp_sensor": "Temperature sensor anomaly",
}


def get_failure_event_date(motor_id):
    """
    Finds the corrective event for each simulated fault motor.
    Preventive maintenance events are ignored.
    """
    evt = maintenance[
        (maintenance["motor_id"] == motor_id)
        & (maintenance["planned"] == False)
    ].sort_values("event_date")

    if evt.empty:
        return pd.NaT

    return evt.iloc[0]["event_date"]


def first_persistent_detection(g, rule_col, persistence_h=6):
    """
    Finds the first moment in which a rule remains active
    for at least `persistence_h` consecutive hourly samples.

    This avoids considering one isolated spike as a valid alarm.
    """
    g = g.sort_values("timestamp").copy()

    active = g[rule_col].fillna(False).astype(bool)

    # Detect consecutive groups
    groups = (active != active.shift()).cumsum()

    intervals = (
        g.assign(_active=active, _group=groups)
        .groupby("_group")
        .agg(
            active=("_active", "first"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            samples=("_active", "size"),
        )
        .reset_index(drop=True)
    )

    intervals = intervals[
        (intervals["active"] == True)
        & (intervals["samples"] >= persistence_h)
    ]

    if intervals.empty:
        return pd.NaT

    return intervals.iloc[0]["start"]


def calculate_alarm_persistence(g, rule_col):
    """
    Total number of hourly samples for which a rule is active.
    With hourly data, this is approximately alarm persistence in hours.
    """
    return int(g[rule_col].fillna(False).sum())


# ------------------------------------------------------------
# 1. FAULT MOTOR EVALUATION
# ------------------------------------------------------------

evaluation_rows = []

for motor_id, gt in GROUND_TRUTH.items():

    g = df[df["motor_id"] == motor_id].copy()

    event_date = get_failure_event_date(motor_id)

    expected_rule = gt["expected_rule"]

    first_detection = first_persistent_detection(
        g,
        expected_rule,
        persistence_h=6
    )

    if pd.notna(first_detection) and pd.notna(event_date):
        lead_time_h = (
            event_date - first_detection
        ).total_seconds() / 3600

        lead_time_days = lead_time_h / 24

        detected_before_event = lead_time_h > 0

    else:
        lead_time_h = np.nan
        lead_time_days = np.nan
        detected_before_event = False

    # First detection from ANY rule
    any_alarm = g[g["alarm"] == True]

    if not any_alarm.empty:
        first_any_detection = any_alarm["timestamp"].min()

        rules_at_first_detection = []

        first_rows = g[
            g["timestamp"] == first_any_detection
        ]

        for rule in rule_cols:
            if first_rows[rule].any():
                rules_at_first_detection.append(
                    RULE_LABELS[rule]
                )

        detected_rules = " | ".join(rules_at_first_detection)

    else:
        first_any_detection = pd.NaT
        detected_rules = "None"

    persistence_h = calculate_alarm_persistence(
        g,
        expected_rule
    )

    max_condition_score = g["condition_score"].max()

    evaluation_rows.append({

        "motor_id": motor_id,

        "simulated_fault": gt["fault"],

        "expected_rule": RULE_LABELS[
            expected_rule
        ],

        "first_expected_detection":
            first_detection,

        "first_any_detection":
            first_any_detection,

        "event_date":
            event_date,

        "lead_time_hours":
            round(lead_time_h, 1)
            if pd.notna(lead_time_h)
            else np.nan,

        "lead_time_days":
            round(lead_time_days, 2)
            if pd.notna(lead_time_days)
            else np.nan,

        "detected_before_event":
            detected_before_event,

        "alarm_persistence_hours":
            persistence_h,

        "max_condition_score":
            round(
                max_condition_score,
                2
            ),

        "rules_at_first_alarm":
            detected_rules,
    })


evaluation = pd.DataFrame(
    evaluation_rows
)

print("\n======================================")
print("FAULT MOTOR EVALUATION")
print("======================================\n")

print(
    evaluation.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 2. DETECTION RATE
# ------------------------------------------------------------

n_fault_motors = len(
    GROUND_TRUTH
)

n_detected = (
    evaluation[
        "detected_before_event"
    ]
    .sum()
)

detection_rate = (
    n_detected
    / n_fault_motors
    * 100
)


# ------------------------------------------------------------
# 3. FALSE POSITIVE RATE
# ------------------------------------------------------------

fault_motor_ids = set(
    GROUND_TRUTH.keys()
)

all_motor_ids = set(
    df["motor_id"].unique()
)

healthy_motor_ids = (
    all_motor_ids
    - fault_motor_ids
)

false_positive_motors = []

healthy_alarm_hours = {}

for motor_id in healthy_motor_ids:

    g = df[
        df["motor_id"] == motor_id
    ]

    alarm_hours = int(
        g["alarm"]
        .fillna(False)
        .sum()
    )

    healthy_alarm_hours[
        motor_id
    ] = alarm_hours

    # Require at least 6 hours
    # of alarms to count as FP motor.
    if alarm_hours >= 6:
        false_positive_motors.append(
            motor_id
        )

n_healthy = len(
    healthy_motor_ids
)

n_false_positive = len(
    false_positive_motors
)

false_positive_rate = (
    n_false_positive
    / n_healthy
    * 100
    if n_healthy > 0
    else np.nan
)


# ------------------------------------------------------------
# 4. AVERAGE LEAD TIME
# ------------------------------------------------------------

valid_lead_times = (
    evaluation.loc[
        evaluation[
            "detected_before_event"
        ],
        "lead_time_days"
    ]
)

average_lead_time_days = (
    valid_lead_times.mean()
    if not valid_lead_times.empty
    else np.nan
)


# ------------------------------------------------------------
# 5. ALARM PERSISTENCE
# ------------------------------------------------------------

average_alarm_persistence_h = (
    evaluation[
        "alarm_persistence_hours"
    ]
    .mean()
)


# ------------------------------------------------------------
# 6. MOTORS AT RISK
# ------------------------------------------------------------

# Latest available condition
latest_per_motor = (
    df.sort_values(
        "timestamp"
    )
    .groupby(
        "motor_id",
        as_index=False
    )
    .tail(1)
)

motors_at_risk = (
    latest_per_motor[
        "condition_state"
    ]
    .isin(
        [
            "Warning",
            "Critical"
        ]
    )
    .sum()
)


# ------------------------------------------------------------
# 7. RULE-SPECIFIC PERFORMANCE
# ------------------------------------------------------------

rule_performance = []

for motor_id, gt in GROUND_TRUTH.items():

    expected_rule = (
        gt[
            "expected_rule"
        ]
    )

    g_fault = df[
        df["motor_id"] == motor_id
    ]

    event_date = (
        get_failure_event_date(
            motor_id
        )
    )

    # Was the expected rule triggered
    # before the intervention?
    pre_event = g_fault[
        g_fault[
            "timestamp"
        ] < event_date
    ]

    true_positive = bool(
        pre_event[
            expected_rule
        ]
        .fillna(False)
        .any()
    )

    # Check same rule against
    # all healthy motors.
    fp_motors_for_rule = []

    for healthy_id in healthy_motor_ids:

        gh = df[
            df["motor_id"]
            == healthy_id
        ]

        active_hours = int(
            gh[
                expected_rule
            ]
            .fillna(False)
            .sum()
        )

        if active_hours >= 6:

            fp_motors_for_rule.append(
                healthy_id
            )

    rule_performance.append({

        "fault_motor":
            motor_id,

        "simulated_fault":
            gt["fault"],

        "rule":
            RULE_LABELS[
                expected_rule
            ],

        "true_positive":
            true_positive,

        "false_positive_motors":
            len(
                fp_motors_for_rule
            ),

        "false_positive_motor_ids":
            ", ".join(
                fp_motors_for_rule
            )
            if fp_motors_for_rule
            else "None",
    })


rule_performance = pd.DataFrame(
    rule_performance
)


# ------------------------------------------------------------
# 8. GLOBAL METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame({

    "metric": [
        "Fault Motors",
        "Fault Motors Detected",
        "Detection Rate %",
        "Healthy Motors",
        "False Positive Motors",
        "False Positive Rate %",
        "Average Lead Time Days",
        "Average Alarm Persistence Hours",
        "Motors At Risk - Latest State",
    ],

    "value": [
        n_fault_motors,
        n_detected,
        round(
            detection_rate,
            2
        ),
        n_healthy,
        n_false_positive,
        round(
            false_positive_rate,
            2
        ),
        round(
            average_lead_time_days,
            2
        )
        if pd.notna(
            average_lead_time_days
        )
        else np.nan,
        round(
            average_alarm_persistence_h,
            2
        ),
        int(
            motors_at_risk
        ),
    ],
})


print("\n======================================")
print("GLOBAL DETECTION METRICS")
print("======================================\n")

print(
    metrics.to_string(
        index=False
    )
)


print("\n======================================")
print("RULE PERFORMANCE")
print("======================================\n")

print(
    rule_performance.to_string(
        index=False
    )
)


print("\nFalse positive motors:")

if false_positive_motors:

    print(
        ", ".join(
            sorted(
                false_positive_motors
            )
        )
    )

else:

    print(
        "None"
    )


# ------------------------------------------------------------
# 9. EXPORT FOR POWER BI
# ------------------------------------------------------------

evaluation.to_csv(
    DATA_DIR
    / "detection_evaluation.csv",
    index=False
)

metrics.to_csv(
    DATA_DIR
    / "detection_metrics.csv",
    index=False
)

rule_performance.to_csv(
    DATA_DIR
    / "rule_performance.csv",
    index=False
)

healthy_fp_df = pd.DataFrame({

    "motor_id":
        list(
            healthy_alarm_hours.keys()
        ),

    "alarm_hours":
        list(
            healthy_alarm_hours.values()
        ),
})

healthy_fp_df = (
    healthy_fp_df
    .sort_values(
        "alarm_hours",
        ascending=False
    )
)

healthy_fp_df.to_csv(
    DATA_DIR
    / "healthy_motor_false_alarms.csv",
    index=False
)


print("\nEvaluation files exported:")
print("- detection_evaluation.csv")
print("- detection_metrics.csv")
print("- rule_performance.csv")
print("- healthy_motor_false_alarms.csv")