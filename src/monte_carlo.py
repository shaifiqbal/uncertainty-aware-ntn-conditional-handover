from skyfield.api import load, wgs84
import pandas as pd
import numpy as np
from datetime import timedelta
import sys

sys.path.insert(0, "src")
from link_budget import cnr_db, CNR_MIN_DB
from uncertainty_model import (SIGMA_MARGIN_DB, SIGMA_DELAY_S, SIGMA_MEAS_S,
                                Z_ALPHA_DEFAULT, MIN_DCNR_DT_DB_S, SIGMA_T_CAP_S)

GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65
MIN_ELEVATION_DEG = 0.0
MIN_MAX_ELEV_FOR_CANDIDACY = 20.0
SEARCH_HOURS = 72
N_TRIALS = 40
SEED_BASE = 42
TOTAL_OVERHEAD_NOMINAL_S = 3.1
PREMATURE_THRESHOLD_S = 30.0

observer = wgs84.latlon(GROUND_LAT, GROUND_LON, GROUND_ALT_M)
ts = load.timescale()
satellites = load.tle_file("data/tle_archive/starlink.tle")
subset_names = ["STARLINK-1008", "STARLINK-1012", "STARLINK-1017",
                 "STARLINK-1020", "STARLINK-1036", "STARLINK-1039"]
by_name = {sat.name: sat for sat in satellites}
subset = [by_name[n] for n in subset_names if n in by_name]

# --- Build candidate pass pool ---
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
print(f"Candidate pool: {len(all_passes)} passes (max_elev >= {MIN_MAX_ELEV_FOR_CANDIDACY} deg)")


def build_true_profile(p):
    sat = p["sat"]
    rise_dt = p["rise"].utc_datetime()
    set_dt = p["set"].utc_datetime()
    total_s = int((set_dt - rise_dt).total_seconds())
    fine_times = ts.utc([rise_dt + timedelta(seconds=s) for s in range(0, total_s + 1)])
    alt, az, dist = (sat - observer).at(fine_times).altaz()
    df = pd.DataFrame({
        "seconds_from_rise": list(range(0, total_s + 1)),
        "elevation_deg": alt.degrees,
        "slant_range_km": dist.km,
    })
    df["cnr_db_true"] = df["slant_range_km"].apply(cnr_db)
    return df


def crossing_time(series_val, threshold, start_idx=None):
    sub = series_val if start_idx is None else series_val.loc[start_idx:]
    below = sub[sub <= threshold]
    return below.index[0] if not below.empty else None


def run_trial(seed):
    rng = np.random.default_rng(seed)
    p = all_passes[rng.integers(0, len(all_passes))]
    df = build_true_profile(p)

    peak_idx = df["cnr_db_true"].idxmax()

    true_expiry_idx = crossing_time(df["cnr_db_true"], CNR_MIN_DB, start_idx=peak_idx)
    if true_expiry_idx is None:
        return None
    true_expiry_s = df.loc[true_expiry_idx, "seconds_from_rise"]

    margin_noise_db = rng.normal(0, SIGMA_MARGIN_DB)
    df["cnr_db_obs"] = df["cnr_db_true"] + margin_noise_db

    actual_overhead_s = max(0.1, TOTAL_OVERHEAD_NOMINAL_S + rng.normal(0, SIGMA_DELAY_S))

    obs_expiry_idx = crossing_time(df["cnr_db_obs"], CNR_MIN_DB, start_idx=peak_idx)
    obs_expiry_s = (df.loc[obs_expiry_idx, "seconds_from_rise"]
                     if obs_expiry_idx is not None else df["seconds_from_rise"].iloc[-1])

    descending = df.loc[peak_idx:].copy()
    descending["remaining_obs_s"] = obs_expiry_s - descending["seconds_from_rise"]
    descending["remaining_true_s"] = true_expiry_s - descending["seconds_from_rise"]

    dt = descending["seconds_from_rise"].diff()
    dCNR = descending["cnr_db_obs"].diff()
    dCNR_dt = (dCNR / dt).replace([np.inf, -np.inf], np.nan)
    dCNR_dt_safe = dCNR_dt.where(dCNR_dt.abs() >= MIN_DCNR_DT_DB_S)
    sigma_T_predict = (SIGMA_MARGIN_DB / dCNR_dt_safe.abs())
    descending["sigma_T_s"] = np.sqrt(
        sigma_T_predict**2 + SIGMA_MEAS_S**2 + SIGMA_DELAY_S**2
    ).clip(upper=SIGMA_T_CAP_S)
    descending["T_safe_s"] = descending["remaining_obs_s"] - Z_ALPHA_DEFAULT * descending["sigma_T_s"]

    def first_trigger(condition_series):
        hit = descending[condition_series]
        return hit.iloc[0]["seconds_from_rise"] if not hit.empty else None

    policies = {
        "Fixed guard-15s": first_trigger(descending["remaining_obs_s"] <= 15),
        "Point-prediction (no margin)": first_trigger(descending["remaining_obs_s"] <= TOTAL_OVERHEAD_NOMINAL_S),
        "Quality threshold (3dB)": first_trigger(descending["cnr_db_obs"] <= 3.0),
        "Conservative fixed-margin (60s)": first_trigger(descending["remaining_obs_s"] <= 60),
        "Proposed (uncertainty-aware)": first_trigger(
            descending["T_safe_s"].notna() & (descending["T_safe_s"] <= TOTAL_OVERHEAD_NOMINAL_S)),
    }

    rows = []
    for policy_name, trigger_s in policies.items():
        if trigger_s is None:
            rows.append({"seed": seed, "satellite": p["sat"].name, "max_elev": p["max_elev"],
                         "policy": policy_name, "outcome": "NO_TRIGGER", "failure": True,
                         "premature": False, "interruption_s": np.nan,
                         "excess_margin_s": np.nan, "retained_link_time_s": np.nan})
            continue
        true_remaining_at_trigger = true_expiry_s - trigger_s
        margin_true = true_remaining_at_trigger - actual_overhead_s
        failure = margin_true < 0
        premature = (not failure) and (margin_true > PREMATURE_THRESHOLD_S)
        rows.append({
            "seed": seed, "satellite": p["sat"].name, "max_elev": p["max_elev"],
            "policy": policy_name,
            "outcome": "FAILURE" if failure else "SUCCESS",
            "failure": failure, "premature": premature,
            "interruption_s": abs(margin_true) if failure else 0.0,
            "excess_margin_s": max(0, margin_true) if not failure else 0.0,
            "retained_link_time_s": trigger_s,
        })
    return rows


all_rows = []
n_excluded = 0
for i in range(N_TRIALS):
    seed = SEED_BASE + i
    result = run_trial(seed)
    if result:
        all_rows.extend(result)
    else:
        n_excluded += 1

print(f"{n_excluded} of {N_TRIALS} trials excluded (sampled pass never dropped below CNR_min)")

results_df = pd.DataFrame(all_rows)
results_df.to_csv("results/experiment1_trials.csv", index=False)

summary = results_df.groupby("policy").agg(
    n_trials=("failure", "count"),
    failure_probability=("failure", "mean"),
    premature_rate=("premature", "mean"),
    avg_interruption_s=("interruption_s", lambda x: x[results_df.loc[x.index, "failure"]].mean()
                         if results_df.loc[x.index, "failure"].any() else 0.0),
    avg_excess_margin_s=("excess_margin_s", "mean"),
    avg_retained_link_time_s=("retained_link_time_s", "mean"),
).reset_index()
summary.to_csv("results/experiment1_summary.csv", index=False)

print(summary.to_string(index=False))
print(f"\nSaved results/experiment1_trials.csv ({len(results_df)} rows) and results/experiment1_summary.csv")