from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("industrial_motor_data")
READINGS_FILE = DATA_DIR / "motor_readings_enriched.csv"
MAINT_FILE = DATA_DIR / "maintenance_events.csv"

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

df = pd.read_csv(READINGS_FILE,parse_dates=["timestamp"])
maintenance = pd.read_csv(MAINT_FILE, parse_dates=["event_date"])

def plot_motor(motor_id, save=False):
    g = df[df["motor_id"] == motor_id].copy()
    if g.empty:
        raise ValueError(f"{motor_id} not found")

    event_dates = maintenance.loc[
        maintenance["motor_id"].eq(motor_id), "event_date"
    ].tolist()

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"diagnostic dasboard - {motor_id}", fontsize=16, fontweight="bold", y=0.92)

    ax[0].plot(g["timestamp"], g["temperature_rise_c"], label="Temperature rise")
    ax[0].plot(g["timestamp"], g["thermal_rise_expected"], label="Expected from baseline")
    for d in event_dates: ax[0].axvline(d, linestyle="--", alpha=0.6)
    ax[0].set_title("Thermal behavior", loc="left")
    ax[0].set_ylabel("Rise [°C]")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.2)

    ax[1].plot(g["timestamp"], g["vibration_rms_mm_s"], label="Measured vibration")
    ax[1].plot(g["timestamp"], g["vibration_expected"], label="Expected from baseline")
    for d in event_dates: ax[1].axvline(d, linestyle="--", alpha=0.6)
    ax[1].set_title("Vibration", loc="left")
    ax[1].set_ylabel("RMS [mm/s]")
    ax[1].legend(loc="upper left")
    ax[1].grid(alpha=0.2)

    ax[2].plot(g["timestamp"], g["load_pct_med24"], label="Load % - 24h median")
    ax[2].plot(g["timestamp"], g["current_pct_rated_med24"], label="Current % rated - 24h")
    ax[2].axhline(100, linestyle="--", color="red", alpha=0.6)
    for d in event_dates: ax[2].axvline(d, linestyle="--", alpha=0.6)
    ax[2].set_title("Mechanical / electrical load", loc="left")
    ax[2].set_ylabel("%")
    ax[2].legend(loc="upper left")
    ax[2].grid(alpha=0.2)

    ax[3].plot(g["timestamp"], g["current_unbalance_pct_med24"], label="Current unbalance - 24h")
    ax[3].axhline(TH["unbalance_warning"], linestyle="--", color="orange", label="Warning")
    ax[3].axhline(TH["unbalance_critical"], linestyle="--", color="red", label="Critical")
    for d in event_dates: ax[3].axvline(d, linestyle="--", alpha=0.6)
    ax[3].set_title("Current unbalance", loc="left")
    ax[3].set_ylabel("%")
    ax[3].legend(loc="upper left")
    ax[3].grid(alpha=0.2)

    ax[4].plot(g["timestamp"], g["condition_score"], label="Condition score")
    ax[4].axhline(20, linestyle="--", color="orange", alpha=0.6)
    ax[4].axhline(50, linestyle="--", color="red", alpha=0.6)
    alarm_points = g[g["alarm"]]
    ax[4].scatter(alarm_points["timestamp"], alarm_points["condition_score"], s=15, color="red", label="Detected alarm")
    for d in event_dates: ax[4].axvline(d, linestyle="--", alpha=0.6)
    ax[4].set_title("Blind detection score", loc="left")
    ax[4].set_ylabel("Score [0-100]")
    ax[4].legend(loc="upper left")
    ax[4].grid(alpha=0.2)

    plt.subplots_adjust(hspace=0.4)

    if save:
        plt.savefig(f"{motor_id}_dashboard.png", dpi=160, bbox_inches="tight")

for motor in ["MTR-005","MTR-007","MTR-013","MTR-019","MTR-024","MTR-028"]:
    plot_motor(motor)

plt.show()