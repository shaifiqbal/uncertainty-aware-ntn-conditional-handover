from skyfield.api import load, wgs84
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from link_budget import cnr_db, CNR_MIN_DB
from uncertainty_model import SIGMA_MARGIN_DB, SIGMA_DELAY_S, SIGMA_MEAS_S, MIN_DCNR_DT_DB_S, SIGMA_T_CAP_S

GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65
MIN_ELEVATION_DEG = 0.0
MIN_MAX_ELEV_FOR_CANDIDACY = 20.0
SEARCH_HOURS = 72
N_TRIALS = 40
SEED_BASE = 42
TOTAL_OVERHEAD_NOMINAL_S = 3.1

Z_ALPHA_SWEEP = [0.0, 0.5, 0.84, 1.0, 1.28, 1.645, 2.0, 2.33]  # 0=no margin ... 2.33=~99% bound

observer = wgs84.latlon(GROUND_LAT, GROUND_LON, GROUND_ALT_M)
ts = load.timescale()
satellites = load.tle_file("data/tle_archive/starlink.tle")
subset_names = ["STARLINK-1008", "STARLINK-1012", "STARLINK-1017",
                 "STARLINK-1020", "STARLINK-1036", "STARLINK-1039"]
by_name = {sat.name: sat for sat in satellites}
subset = [by_name[n] for n in subset_names if n in by_name]

t0 = ts.now()
t1 = ts.utc(t0.utc_datetime() + timedelta(hours=SEARCH_HOURS))
all_passes = []
for sat in subset:
    times, events = sat.find_events(observer, t0, t1, altitude_degrees=MIN_ELEVATION_DEG)
    i = 0
    while i < len(events):
        if events[i] == 0 and i + 2 < len(events) and events[i+1] == 1 and events[i+2] == 2:
            rise_t, culminate_t, set_t = times[i], times[i+1], times[i+2]
            alt, az, dist = (sat - observer).at(culminate_t).altaz()
            if alt.degrees >= MIN_MAX_ELEV_FOR_CANDIDACY:
                all_passes.append({"sat": sat, "rise": rise_t, "set": set_t, "max_elev": alt.degrees})
        i += 1
print(f"Candidate pool: {len(all_passes)} passes")


def build_true_profile(p):
    sat = p["sat"]
    rise_dt = p["rise"].utc_datetime()
    set_dt = p["set"].utc_datetime()
    total_s = int((set_dt - rise_dt).total_seconds())
    fine_times = ts.utc([rise_dt + timedelta(seconds=s) for s in range(0, total_s + 1)])
    alt, az, dist = (sat - observer).at(fine_times).altaz()
    df = pd.DataFrame({"seconds_from_rise": list(range(0, total_s + 1)),
                        "slant_range_km": dist.km})
    df["cnr_db_true"] = df["slant_range_km"].apply(cnr_db)
    return df


def crossing_time(series_val, threshold, start_idx=None):
    sub = series_val if start_idx is None else series_val.loc[start_idx:]
    below = sub[sub <= threshold]
    return below.index[0] if not below.empty else None


# --- Precompute per-trial: sampled pass, noisy profile, actual overhead (fixed across z_alpha sweep for fair comparison) ---
trials = []
rng_master = np.random.default_rng(SEED_BASE)
for i in range(N_TRIALS):
    seed = SEED_BASE + i
    rng = np.random.default_rng(seed)
    p = all_passes[rng.integers(0, len(all_passes))]
    df = build_true_profile(p)
    peak_idx = df["cnr_db_true"].idxmax()
    true_expiry_idx = crossing_time(df["cnr_db_true"], CNR_MIN_DB, start_idx=peak_idx)
    if true_expiry_idx is None:
        continue
    true_expiry_s = df.loc[true_expiry_idx, "seconds_from_rise"]

    margin_noise_db = rng.normal(0, SIGMA_MARGIN_DB)
    df["cnr_db_obs"] = df["cnr_db_true"] + margin_noise_db
    actual_overhead_s = max(0.1, TOTAL_OVERHEAD_NOMINAL_S + rng.normal(0, SIGMA_DELAY_S))

    obs_expiry_idx = crossing_time(df["cnr_db_obs"], CNR_MIN_DB, start_idx=peak_idx)
    obs_expiry_s = (df.loc[obs_expiry_idx, "seconds_from_rise"]
                     if obs_expiry_idx is not None else df["seconds_from_rise"].iloc[-1])

    descending = df.loc[peak_idx:].copy()
    descending["remaining_obs_s"] = obs_expiry_s - descending["seconds_from_rise"]

    dt = descending["seconds_from_rise"].diff()
    dCNR = descending["cnr_db_obs"].diff()
    dCNR_dt = (dCNR / dt).replace([np.inf, -np.inf], np.nan)
    dCNR_dt_safe = dCNR_dt.where(dCNR_dt.abs() >= MIN_DCNR_DT_DB_S)
    sigma_T_predict = (SIGMA_MARGIN_DB / dCNR_dt_safe.abs())
    descending["sigma_T_s"] = np.sqrt(
        sigma_T_predict**2 + SIGMA_MEAS_S**2 + SIGMA_DELAY_S**2
    ).clip(upper=SIGMA_T_CAP_S)

    trials.append({"descending": descending, "true_expiry_s": true_expiry_s,
                    "actual_overhead_s": actual_overhead_s})

print(f"{len(trials)} of {N_TRIALS} trials usable across sweep")

sweep_rows = []
for z_alpha in Z_ALPHA_SWEEP:
    outcomes = []
    excess_margins = []
    for t in trials:
        descending = t["descending"]
        T_safe = descending["remaining_obs_s"] - z_alpha * descending["sigma_T_s"]
        hit = descending[T_safe.notna() & (T_safe <= TOTAL_OVERHEAD_NOMINAL_S)]
        if hit.empty:
            outcomes.append(True)  # treat as failure (never triggers in time)
            continue
        trigger_s = hit.iloc[0]["seconds_from_rise"]
        true_remaining_at_trigger = t["true_expiry_s"] - trigger_s
        margin_true = true_remaining_at_trigger - t["actual_overhead_s"]
        outcomes.append(margin_true < 0)
        excess_margins.append(max(0, margin_true))

    sweep_rows.append({
        "z_alpha": z_alpha,
        "failure_probability": np.mean(outcomes),
        "avg_excess_margin_s": np.mean(excess_margins) if excess_margins else np.nan,
        "n_trials": len(trials),
    })

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv("results/experiment3_risk_sweep.csv", index=False)
print(sweep_df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(sweep_df["z_alpha"], sweep_df["failure_probability"], marker="o", label="Proposed (swept)")
ax1.set_xlabel("Risk parameter z_alpha")
ax1.set_ylabel("Handover failure probability")
ax1.set_title("Risk-parameter sensitivity")
ax1.grid(alpha=0.3)

ax2.plot(sweep_df["failure_probability"], sweep_df["avg_excess_margin_s"], marker="o", label="Proposed (swept)")
for _, r in sweep_df.iterrows():
    ax2.annotate(f"{r['z_alpha']}", (r["failure_probability"], r["avg_excess_margin_s"]), fontsize=7)
ax2.set_xlabel("Handover failure probability")
ax2.set_ylabel("Avg. excess margin (s)")
ax2.set_title("Pareto: reliability vs. excess margin")
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("figures/experiment3_risk_sweep.png", dpi=150)
print("Saved results/experiment3_risk_sweep.csv and figures/experiment3_risk_sweep.png")