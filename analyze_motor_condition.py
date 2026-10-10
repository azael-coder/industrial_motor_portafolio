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

