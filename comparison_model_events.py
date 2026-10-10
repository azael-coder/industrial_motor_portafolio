from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("industrial_motor_data")
READINGS_FILE = DATA_DIR / "motor_readings_enriched.csv"
MAINT_FILE = DATA_DIR / "maintenance_events.csv"

df = pd.read_csv(READINGS_FILE,parse_dates=["timestamp"])
maintenance = pd.read_csv(MAINT_FILE, parse_dates=["event_date"])

rule_cols = [
    "rule_bearing",
    "rule_misalignment",
    "rule_overload",
    "rule_cooling",
    "rule_unbalance",
    "rule_temp_sensor",
]

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