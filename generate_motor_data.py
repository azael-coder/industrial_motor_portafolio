import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# INDUSTRIAL MOTOR FLEET SIMULATOR
# 30 three-phase induction motors, 90 days
# Internal physics step: 10 min
# Export resolution: 1 h
#
# Main physical relationships:
#   P_out = load * P_rated
#   I ~= P_in / (sqrt(3) * V * PF)
#   P_in = P_out / eta
#   P_loss = P_in - P_out
#   omega = 2*pi*n/60
#   Torque = P_out / omega
#   Thermal RC model:
#      C_th * d(theta)/dt = P_loss - theta/R_th
#   Induction-motor speed:
#      n = n_sync * (1 - slip)
#
# Fault signatures are modeled as progressive modifiers of
# friction, thermal resistance, vibration and/or load behavior.
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

N_MOTORS = 30
DAYS = 90
INTERNAL_MINUTES = 10
EXPORT_MINUTES = 60
START = pd.Timestamp("2026-06-01 00:00:00")

V_LL = 460.0          # line-line RMS voltage [V]
FREQ_HZ = 60.0
POLES = 4
N_SYNC = 120 * FREQ_HZ / POLES  # 1800 rpm

DT_H = INTERNAL_MINUTES / 60.0
STEPS = int(DAYS * 24 / DT_H)
EXPORT_EVERY = EXPORT_MINUTES // INTERNAL_MINUTES

AREAS = ["Cooling Table", "Conveyors", "Pumps", "Finishing", "Utilities"]
CRITICALITY = ["Low", "Medium", "High"]

@dataclass
class Motor:
    motor_id: str
    area: str
    criticality: str
    rated_kw: float
    efficiency_nom: float
    pf_nom: float
    slip_full_load: float
    ambient_base_c: float
    rth_c_per_kw: float
    tau_h: float
    vib_base: float
    installation_year: int
    load_base: float
    duty_cycle: float
    phase_offset: float

def make_motor(i: int) -> Motor:
    rated_choices = np.array([7.5, 11, 15, 18.5, 22, 30, 37, 45, 55])
    rated_kw = float(rng.choice(rated_choices))
    area = AREAS[i % len(AREAS)]
    crit = rng.choice(CRITICALITY, p=[0.2, 0.5, 0.3])

    # Typical premium/general industrial ranges, varied by unit
    efficiency = float(np.clip(rng.normal(0.91, 0.02), 0.86, 0.95))
    pf = float(np.clip(rng.normal(0.86, 0.035), 0.76, 0.92))
    slip = float(np.clip(rng.normal(0.022, 0.005), 0.012, 0.038))

    # Thermal resistance is inversely related to size:
    # larger motors tend to reject more absolute heat for similar temperature rise.
    rth = float(np.clip(1.15 * (15.0 / rated_kw) ** 0.28, 0.45, 1.6))
    tau = float(np.clip(rng.normal(2.5, 0.55), 1.3, 4.2))  # h
    vib = float(np.clip(rng.normal(1.35, 0.22), 0.75, 1.9))
    load_base = float(rng.uniform(0.48, 0.82))
    duty = float(rng.uniform(0.72, 0.98))

    return Motor(
        motor_id=f"MTR-{i+1:03d}",
        area=area,
        criticality=crit,
        rated_kw=rated_kw,
        efficiency_nom=efficiency,
        pf_nom=pf,
        slip_full_load=slip,
        ambient_base_c=float(rng.uniform(23, 31)),
        rth_c_per_kw=rth,
        tau_h=tau,
        vib_base=vib,
        installation_year=int(rng.integers(2016, 2025)),
        load_base=load_base,
        duty_cycle=duty,
        phase_offset=float(rng.uniform(0, 2*np.pi)),
    )

motors = [make_motor(i) for i in range(N_MOTORS)]

# ------------------------------------------------------------
# Fault plan
# start_day is when measurable degradation begins.
# severity grows from 0 to 1 until event day.
# ------------------------------------------------------------
faults = {
    "MTR-007": {"type": "cooling_degradation", "start_day": 52, "event_day": 76},
    "MTR-013": {"type": "bearing_wear",        "start_day": 40, "event_day": 72},
    "MTR-019": {"type": "mechanical_overload","start_day": 58, "event_day": 79},
    "MTR-024": {"type": "misalignment",        "start_day": 47, "event_day": 74},
    "MTR-028": {"type": "sensor_fault",        "start_day": 64, "event_day": 82},
    "MTR-005": {"type": "voltage_unbalance",   "start_day": 61, "event_day": 84},
}

def fault_severity(motor_id: str, day: float):
    cfg = faults.get(motor_id)
    if cfg is None or day < cfg["start_day"]:
        return None, 0.0
    # At the event day we assume maintenance/corrective intervention resets
    # the degradation, which lets the dataset show before/after behavior.
    if day >= cfg["event_day"]:
        return None, 0.0
    x = (day - cfg["start_day"]) / max(cfg["event_day"] - cfg["start_day"], 1e-9)
    # Progressive degradation with acceleration near intervention/failure.
    sev = float(np.clip(x, 0, 1) ** 1.35)
    return cfg["type"], sev

def ambient_temperature(m: Motor, t_h: float):
    # daily sinusoid, peak around 15:00 + very slow weather drift
    hour = t_h % 24
    daily = 4.3 * np.sin(2*np.pi*(hour - 9)/24)
    slow = 1.7 * np.sin(2*np.pi*t_h/(24*11) + m.phase_offset)
    return m.ambient_base_c + daily + slow + rng.normal(0, 0.25)

def operating_state(m: Motor, t_h: float):
    """
    Industrial-ish production schedule:
    - higher run probability Mon-Sat
    - reduced Sunday operation
    - random short stops
    """
    ts = START + pd.Timedelta(hours=t_h)
    dow = ts.dayofweek
    p_run = m.duty_cycle * (0.72 if dow == 6 else 1.0)

    # Area-specific shift modulation
    shift_wave = 0.04 * np.sin(2*np.pi*(t_h % 24)/8 + m.phase_offset)
    p_run = np.clip(p_run + shift_wave, 0.45, 0.995)
    return rng.random() < p_run

def load_fraction(m: Motor, t_h: float, running: bool, fault_type: str, sev: float):
    if not running:
        return 0.0

    # Process demand varies with 8h shifts + 24h cycle
    cyc1 = 0.08*np.sin(2*np.pi*t_h/8 + m.phase_offset)
    cyc2 = 0.05*np.sin(2*np.pi*t_h/24 + 0.5*m.phase_offset)
    load = m.load_base + cyc1 + cyc2 + rng.normal(0, 0.035)

    if fault_type == "mechanical_overload":
        # Progressive extra driven-load / mechanical resistance.
        # At severe condition the motor is pushed close to or above nameplate.
        load += 0.52 * sev + 0.07 * sev * np.sin(2*np.pi*t_h/2)

    return float(np.clip(load, 0.18, 1.28))

def motor_electrical(m: Motor, load: float, running: bool, fault_type: str, sev: float):
    if not running:
        # motor off: tiny measurement noise, speed 0
        return {
            "voltage_v": V_LL + rng.normal(0, 1.0),
            "current_a": max(0.0, rng.normal(0.12, 0.05)),
            "current_a_phase_a": max(0.0, rng.normal(0.12, 0.05)),
            "current_a_phase_b": max(0.0, rng.normal(0.12, 0.05)),
            "current_a_phase_c": max(0.0, rng.normal(0.12, 0.05)),
            "pf": 0.05,
            "efficiency": m.efficiency_nom,
            "speed_rpm": 0.0,
            "torque_nm": 0.0,
            "p_out_kw": 0.0,
            "p_in_kw": 0.0,
            "p_loss_kw": 0.0,
        }

    # Efficiency falls at light load and degrades slightly under faults.
    eta = m.efficiency_nom - 0.025*(1-load)**2
    if fault_type in ("bearing_wear", "misalignment", "mechanical_overload"):
        eta -= 0.020*sev
    eta = float(np.clip(eta, 0.78, 0.96))

    # PF is poorer at light load.
    pf = m.pf_nom - 0.16*max(0, 0.65-load)
    pf = float(np.clip(pf, 0.55, 0.94))

    p_out_kw = m.rated_kw * load
    p_in_kw = p_out_kw / eta
    p_loss_kw = p_in_kw - p_out_kw

    # Fault-related mechanical/friction losses.
    if fault_type == "bearing_wear":
        p_loss_kw += m.rated_kw * 0.025 * sev
        p_in_kw = p_out_kw + p_loss_kw
    elif fault_type == "misalignment":
        p_loss_kw += m.rated_kw * 0.018 * sev
        p_in_kw = p_out_kw + p_loss_kw

    voltage = V_LL + rng.normal(0, 2.0)

    # Balanced RMS current approximation from three-phase power.
    current = (p_in_kw * 1000) / (np.sqrt(3) * voltage * pf)

    # Slip increases with load and friction.
    slip = 0.004 + (m.slip_full_load - 0.004) * min(load, 1.0)
    if load > 1:
        slip += 0.020*(load - 1.0)
    if fault_type in ("bearing_wear", "misalignment"):
        slip += 0.005 * sev

    speed = N_SYNC * (1 - slip) + rng.normal(0, 1.6)
    omega = 2*np.pi*max(speed, 1)/60
    torque = (p_out_kw*1000) / omega

    # Three-phase currents. Voltage unbalance induces much larger
    # current unbalance than the original voltage deviation.
    ia = current * (1 + rng.normal(0, 0.006))
    ib = current * (1 + rng.normal(0, 0.006))
    ic = current * (1 + rng.normal(0, 0.006))

    if fault_type == "voltage_unbalance":
        # approximate signature: asymmetry grows progressively
        # without claiming an exact NEMA transfer ratio.
        ia *= (1 + 0.12*sev)
        ib *= (1 - 0.07*sev)
        ic *= (1 - 0.02*sev)

    current_mean = (ia + ib + ic)/3
    current_unbalance_pct = 100 * max(abs(ia-current_mean), abs(ib-current_mean), abs(ic-current_mean)) / max(current_mean, 1e-6)

    return {
        "voltage_v": voltage,
        "current_a": current_mean,
        "current_a_phase_a": ia,
        "current_a_phase_b": ib,
        "current_a_phase_c": ic,
        "current_unbalance_pct": current_unbalance_pct,
        "pf": pf,
        "efficiency": eta,
        "speed_rpm": max(speed, 0),
        "torque_nm": torque,
        "p_out_kw": p_out_kw,
        "p_in_kw": p_in_kw,
        "p_loss_kw": p_loss_kw,
    }

def vibration_rms(m: Motor, load: float, running: bool, fault_type: str, sev: float):
    if not running:
        return max(0.05, rng.normal(0.12, 0.03))

    # Baseline RMS increases mildly with load.
    vib = m.vib_base * (0.82 + 0.30*load) + rng.normal(0, 0.08)

    if fault_type == "bearing_wear":
        # progressive broadband/high-frequency energy represented as RMS growth
        vib += 5.2 * sev**1.35 + rng.normal(0, 0.18*sev)
    elif fault_type == "misalignment":
        # strong 1x/2x mechanical signature; compressed here into overall RMS
        vib += 3.7 * sev + 0.35*sev*np.sin(2*np.pi*load*3)
    elif fault_type == "mechanical_overload":
        vib += 1.25 * sev
    elif fault_type == "voltage_unbalance":
        # electromagnetic forcing can increase vibration modestly
        vib += 0.85 * sev

    return float(max(0.05, vib))

def health_score(temp_c, ambient_c, vib, current_pct, current_unbalance, running, sensor_quality):
    """
    Transparent demo health index (0-100), not an industry standard.
    Uses measured condition variables only; it does NOT use the hidden
    simulated fault label or fault severity.
    """
    if not running:
        return np.nan, "Stopped"

    temp_rise = max(temp_c - ambient_c, 0)

    # Piecewise penalties chosen to create realistic separation between
    # normal operation, developing degradation, and severe condition.
    p_t = np.interp(temp_rise, [20, 35, 50, 65, 85], [0, 0, 8, 25, 45])
    p_v = np.interp(vib,       [1.0, 2.5, 4.0, 6.0, 8.5], [0, 0, 12, 35, 58])
    p_i = np.interp(current_pct,[60, 90, 105, 120, 140], [0, 0, 7, 20, 38])
    p_u = np.interp(current_unbalance, [0.5, 2, 5, 10, 18], [0, 0, 8, 28, 48])

    p_q = 0
    if sensor_quality == "Suspect":
        p_q = 15
    elif sensor_quality == "Bad":
        p_q = 45

    score = float(np.clip(100 - (p_t + p_v + p_i + p_u + p_q), 0, 100))
    if score >= 80:
        state = "Healthy"
    elif score >= 60:
        state = "Warning"
    else:
        state = "Critical"
    return score, state

# ------------------------------------------------------------
# Simulate
# ------------------------------------------------------------
rows = []
maintenance_events = []
state = {}

for m in motors:
    state[m.motor_id] = {
        "theta_c": rng.uniform(8, 18),  # winding/body rise above ambient
        "runtime_h": rng.uniform(1000, 14000),
    }

for k in range(STEPS):
    t_h = k * DT_H
    day = t_h / 24.0
    ts = START + pd.Timedelta(hours=t_h)

    for m in motors:
        ftype, sev = fault_severity(m.motor_id, day)
        running = operating_state(m, t_h)
        load = load_fraction(m, t_h, running, ftype, sev)
        elec = motor_electrical(m, load, running, ftype, sev)
        ambient = ambient_temperature(m, t_h)

        # Thermal model.
        # theta is motor temperature rise above ambient.
        # At steady-state: theta_ss ~= P_loss * Rth.
        # Convert motor thermal time constant tau = Rth*Cth.
        rth = m.rth_c_per_kw
        if ftype == "cooling_degradation":
            # clogged fan/fins or airflow restriction: heat rejection worsens
            rth *= 1 + 3.2*sev

        theta = state[m.motor_id]["theta_c"]
        target_rise = elec["p_loss_kw"] * rth * 18.0  # calibrated lumped thermal gain
        dtheta = (target_rise - theta) * (DT_H / m.tau_h)
        theta = theta + dtheta + rng.normal(0, 0.06)
        theta = max(ambient - ambient, theta)
        state[m.motor_id]["theta_c"] = theta

        temp = ambient + theta

        # Bearing friction contributes local heating beyond average losses.
        if ftype == "bearing_wear" and running:
            temp += 7.5 * sev**1.25
        if ftype == "misalignment" and running:
            temp += 4.5 * sev

        vib = vibration_rms(m, load, running, ftype, sev)

        # Sensor fault affects measurement, not actual physical temperature.
        measured_temp = temp
        sensor_quality = "Good"
        if ftype == "sensor_fault" and day >= faults[m.motor_id]["start_day"]:
            p_spike = 0.015 + 0.18*sev
            if rng.random() < p_spike:
                measured_temp += rng.choice([-1, 1]) * rng.uniform(18, 55)
                sensor_quality = "Suspect"
            elif rng.random() < 0.035*sev:
                measured_temp = 0.0
                sensor_quality = "Bad"

        # Nameplate current derived near rated load using same physical model.
        rated_current = (m.rated_kw*1000 / m.efficiency_nom) / (np.sqrt(3)*V_LL*m.pf_nom)
        current_pct = 100*elec["current_a"]/rated_current if running else 0

        score, hstate = health_score(
            measured_temp,
            ambient,
            vib,
            current_pct,
            elec.get("current_unbalance_pct", 0.0),
            running,
            sensor_quality
        )

        if running:
            state[m.motor_id]["runtime_h"] += DT_H

        # Export once per hour
        if k % EXPORT_EVERY == 0:
            rows.append({
                "timestamp": ts,
                "motor_id": m.motor_id,
                "area": m.area,
                "criticality": m.criticality,
                "status": "Running" if running else "Stopped",
                "load_pct": round(load*100, 2),
                "voltage_v": round(elec["voltage_v"], 2),
                "current_a": round(elec["current_a"], 3),
                "current_a_phase_a": round(elec["current_a_phase_a"], 3),
                "current_a_phase_b": round(elec["current_a_phase_b"], 3),
                "current_a_phase_c": round(elec["current_a_phase_c"], 3),
                "current_unbalance_pct": round(elec.get("current_unbalance_pct", 0.0), 3),
                "current_pct_rated": round(current_pct, 2),
                "power_factor": round(elec["pf"], 3),
                "efficiency": round(elec["efficiency"], 4),
                "speed_rpm": round(elec["speed_rpm"], 2),
                "torque_nm": round(elec["torque_nm"], 2),
                "p_in_kw": round(elec["p_in_kw"], 3),
                "p_out_kw": round(elec["p_out_kw"], 3),
                "p_loss_kw": round(elec["p_loss_kw"], 3),
                "ambient_temp_c": round(ambient, 2),
                "temperature_c": round(measured_temp, 2),
                "vibration_rms_mm_s": round(vib, 3),
                "runtime_hours": round(state[m.motor_id]["runtime_h"], 1),
                "sensor_quality": sensor_quality,
                "fault_type_simulated": ftype if sev > 0 else "none",
                "fault_severity": round(sev, 4),
                "health_score": None if np.isnan(score) else round(score, 2),
                "health_status": hstate,
            })

# Motor master
df_motors = pd.DataFrame([{
    "motor_id": m.motor_id,
    "area": m.area,
    "criticality": m.criticality,
    "rated_kw": m.rated_kw,
    "rated_voltage_v": V_LL,
    "frequency_hz": FREQ_HZ,
    "poles": POLES,
    "synchronous_speed_rpm": N_SYNC,
    "efficiency_nominal": round(m.efficiency_nom, 4),
    "power_factor_nominal": round(m.pf_nom, 3),
    "full_load_slip": round(m.slip_full_load, 4),
    "installation_year": m.installation_year,
} for m in motors])

df = pd.DataFrame(rows)

# Maintenance events aligned with simulated degradation.
event_descriptions = {
    "bearing_wear": ("Bearing", "Inspect/replace bearing; verify lubrication and alignment"),
    "misalignment": ("Misalignment", "Laser alignment inspection and coupling check"),
    "mechanical_overload": ("Overload", "Inspect driven load and mechanical resistance"),
    "cooling_degradation": ("Cooling", "Inspect fan, fins and airflow path"),
    "sensor_fault": ("Instrumentation", "Inspect temperature sensor and wiring"),
    "voltage_unbalance": ("Electrical", "Inspect supply voltage, terminals and phase loading"),
}

for motor_id, cfg in faults.items():
    evt_ts = START + pd.Timedelta(days=cfg["event_day"])
    cat, action = event_descriptions[cfg["type"]]
    maintenance_events.append({
        "event_date": evt_ts,
        "motor_id": motor_id,
        "event_category": cat,
        "fault_mode": cfg["type"],
        "action": action,
        "downtime_h": round(float(rng.uniform(1.5, 6.0)), 1),
        "planned": False,
    })

# Add routine PM events
for m in rng.choice([x.motor_id for x in motors], size=12, replace=False):
    day = int(rng.integers(8, 87))
    maintenance_events.append({
        "event_date": START + pd.Timedelta(days=day),
        "motor_id": m,
        "event_category": "Preventive",
        "fault_mode": "routine_pm",
        "action": "Routine inspection, cleaning and electrical/mechanical checks",
        "downtime_h": round(float(rng.uniform(0.5, 2.0)), 1),
        "planned": True,
    })

df_events = pd.DataFrame(maintenance_events).sort_values("event_date")

# Output
OUT = Path("industrial_motor_data")
OUT.mkdir(exist_ok=True)

df_motors.to_csv(OUT / "motors.csv", index=False)
df.to_csv(OUT / "motor_readings.csv", index=False)
df_events.to_csv(OUT / "maintenance_events.csv", index=False)

print(f"Generated {len(df):,} hourly motor records")
print(f"Motors: {len(df_motors)}")
print(f"Maintenance events: {len(df_events)}")
print("\nFault motors:")
print(df[df["fault_severity"] > 0][["motor_id","fault_type_simulated"]].drop_duplicates().to_string(index=False))
print("\nHealth status distribution (running samples):")
print(df.loc[df.status=="Running","health_status"].value_counts(normalize=True).round(3))
