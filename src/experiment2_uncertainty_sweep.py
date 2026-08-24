from skyfield.api import load, wgs84
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from link_budget import cnr_db, CNR_MIN_DB
from uncertainty_model import SIGMA_DELAY_S, SIGMA_MEAS_S, MIN_DCNR_DT_DB_S, SIGMA_T_CAP_S

GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65
MIN_ELEVATION_DEG = 0.0
MIN_MAX_ELEV_FOR_CANDIDACY = 20.0
SEARCH_HOURS = 72
N_TRIALS = 40
SEED_BASE = 42
TOTAL_OVERHEAD_NOMINAL_S = 3.1
Z_ALPHA_HEADLINE = 0.84  # operating point identified in Experiment 3

SIGMA_MARGIN_SWEEP_DB = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SIGMA_MARGIN_ASSUMED_DB = 1.5  # fixed - what the policy's uncertainty model always assumes
# (matches the documented value in config/scenario_config.yaml). The sweep values above
# represent the ACTUAL environmental noise, which may differ from this assumption -
# this tests robustness to model mismatch, not just scaling a perfectly-known parameter.
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
    df = pd.DataFrame({"seconds_from_rise": list(range(0, total_s + 1)), "slant_range_km": dist.km})
    df["cnr_db_true"] = df["slant_range_km"].apply(cnr_db)
    return df


def crossing_time(series_val, threshold, start_idx=None):
    sub = series_val if start_idx is None else series_val.loc[start_idx:]
    below = sub[sub <= threshold]
    return below.index[0] if not below.empty else None


# Pre-sample the same passes and delay-overhead realizations for every sweep value
# (only the margin-noise draw and sigma_margin itself change) - fair comparison, same
# principle used in Experiment 3.
base_trials = []
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
    actual_overhead_s = max(0.1, TOTAL_OVERHEAD_NOMINAL_S + rng.normal(0, SIGMA_DELAY_S))
    # store the RNG state seed so each sweep value draws its own margin noise, reproducibly
    base_trials.append({"df": df, "peak_idx": peak_idx, "true_expiry_s": true_expiry_s,
                         "actual_overhead_s": actual_overhead_s, "seed": seed})

print(f"{len(base_trials)} of {N_TRIALS} trials usable")


def evaluate_policy_at_sigma(sigma_margin_db, policy):
    """policy: 'proposed', 'guard15', or 'conservative60'"""
    premature_flags = []
    interruptions = []
    outcomes = []
    for t in base_trials:
        df, peak_idx = t["df"], t["peak_idx"]
        rng = np.random.default_rng(t["seed"] + 10000)  # separate stream for margin noise, reproducible
        margin_noise_db = rng.normal(0, sigma_margin_db)
        df = df.copy()
        df["cnr_db_obs"] = df["cnr_db_true"] + margin_noise_db

        obs_expiry_idx = crossing_time(df["cnr_db_obs"], CNR_MIN_DB, start_idx=peak_idx)
        obs_expiry_s = (df.loc[obs_expiry_idx, "seconds_from_rise"]
                         if obs_expiry_idx is not None else df["seconds_from_rise"].iloc[-1])

        descending = df.loc[peak_idx:].copy()
        descending["remaining_obs_s"] = obs_expiry_s - descending["seconds_from_rise"]

        if policy == "guard15":
            hit = descending[descending["remaining_obs_s"] <= 15]
        elif policy == "conservative60":
            hit = descending[descending["remaining_obs_s"] <= 60]
        else:  # proposed
            dt = descending["seconds_from_rise"].diff()
            dCNR = descending["cnr_db_obs"].diff()
            dCNR_dt = (dCNR / dt).replace([np.inf, -np.inf], np.nan)
            dCNR_dt_safe = dCNR_dt.where(dCNR_dt.abs() >= MIN_DCNR_DT_DB_S)
            sigma_T_predict = (SIGMA_MARGIN_ASSUMED_DB / dCNR_dt_safe.abs())
            sigma_T_s = np.sqrt(sigma_T_predict**2 + SIGMA_MEAS_S**2 + SIGMA_DELAY_S**2).clip(upper=SIGMA_T_CAP_S)
            T_safe = descending["remaining_obs_s"] - Z_ALPHA_HEADLINE * sigma_T_s
            hit = descending[T_safe.notna() & (T_safe <= TOTAL_OVERHEAD_NOMINAL_S)]

        if hit.empty:
            outcomes.append(True)  # failure: never triggered
            continue
        trigger_s = hit.iloc[0]["seconds_from_rise"]
        true_remaining_at_trigger = t["true_expiry_s"] - trigger_s
        margin_true = true_remaining_at_trigger - t["actual_overhead_s"]
        failure = margin_true < 0
        outcomes.append(failure)
        premature_flags.append((not failure) and (margin_true > 30.0))
        interruptions.append(abs(margin_true) if failure else 0.0)
    return np.mean(outcomes), np.mean(premature_flags) if premature_flags else np.nan, np.mean(interruptions) if interruptions else 0.0


rows = []
for sigma in SIGMA_MARGIN_SWEEP_DB:
    for policy in ["proposed", "guard15", "conservative60"]:
        fail_prob, premature_rate, avg_interruption = evaluate_policy_at_sigma(sigma, policy)
        rows.append({"sigma_margin_db": sigma, "policy": policy, "failure_probability": fail_prob,
                     "premature_rate": premature_rate, "avg_interruption_s": avg_interruption})

sweep_df = pd.DataFrame(rows)
sweep_df.to_csv("results/experiment2_uncertainty_sweep.csv", index=False)
print(sweep_df.pivot(index="sigma_margin_db", columns="policy", values="failure_probability").to_string())

fig, ax = plt.subplots(figsize=(7, 5))
for policy, label in [("proposed", "Proposed (uncertainty-aware)"),
                       ("guard15", "Fixed guard-15s"),
                       ("conservative60", "Conservative fixed-margin (60s)")]:
    sub = sweep_df[sweep_df["policy"] == policy]
    ax.plot(sub["sigma_margin_db"], sub["failure_probability"], marker="o", label=label)
ax.set_xlabel("Link-margin measurement uncertainty, sigma_margin (dB)")
ax.set_ylabel("Handover failure probability")
ax.set_title("Robustness to increasing measurement uncertainty")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/experiment2_uncertainty_sweep.png", dpi=150)
print("Saved results/experiment2_uncertainty_sweep.csv and figures/experiment2_uncertainty_sweep.png")