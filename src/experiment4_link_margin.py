from skyfield.api import load, wgs84
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from link_budget import (EIRP_DBW, UE_GT_DB, K_BOLTZMANN_DBW_K_HZ,
                          free_space_path_loss_db, BANDWIDTH_DBHZ, CNR_MIN_DB)
from uncertainty_model import SIGMA_MARGIN_DB, SIGMA_DELAY_S, SIGMA_MEAS_S, MIN_DCNR_DT_DB_S, SIGMA_T_CAP_S

GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65
MIN_ELEVATION_DEG = 0.0
SEARCH_HOURS = 72
N_TRIALS = 40
SEED_BASE = 42
TOTAL_OVERHEAD_NOMINAL_S = 3.1
Z_ALPHA_HEADLINE = 0.84

# Conditions to sweep: extra implementation/atmospheric loss (dB) representing
# progressively weaker link environments (e.g. foliage, rain fade margin, aging hardware)
EXTRA_LOSS_SWEEP_DB = [0.0, 1.0, 2.0, 3.0, 5.0]
MIN_MAX_ELEV_CANDIDACY_SWEEP = [10.0, 20.0, 30.0, 40.0]

observer = wgs84.latlon(GROUND_LAT, GROUND_LON, GROUND_ALT_M)
ts = load.timescale()
satellites = load.tle_file("data/tle_archive/starlink.tle")
subset_names = ["STARLINK-1008", "STARLINK-1012", "STARLINK-1017",
                 "STARLINK-1020", "STARLINK-1036", "STARLINK-1039"]
by_name = {sat.name: sat for sat in satellites}
subset = [by_name[n] for n in subset_names if n in by_name]


def find_passes(min_max_elev):
    t0 = ts.now()
    t1 = ts.utc(t0.utc_datetime() + timedelta(hours=SEARCH_HOURS))
    passes = []
    for sat in subset:
        times, events = sat.find_events(observer, t0, t1, altitude_degrees=MIN_ELEVATION_DEG)
        i = 0
        while i < len(events):
            if events[i] == 0 and i + 2 < len(events) and events[i+1] == 1 and events[i+2] == 2:
                rise_t, culminate_t, set_t = times[i], times[i+1], times[i+2]
                alt, az, dist = (sat - observer).at(culminate_t).altaz()
                if alt.degrees >= min_max_elev:
                    passes.append({"sat": sat, "rise": rise_t, "set": set_t, "max_elev": alt.degrees})
            i += 1
    return passes


def cnr_db_with_extra_loss(slant_range_km, extra_loss_db):
    """Same CNR formula as link_budget.py, with an additional configurable loss term
    representing weaker-link conditions (documented sensitivity parameter, not a new
    fixed constant)."""
    pl_fs = free_space_path_loss_db(slant_range_km)
    return (EIRP_DBW + UE_GT_DB - K_BOLTZMANN_DBW_K_HZ - pl_fs - 0.5 - extra_loss_db - BANDWIDTH_DBHZ)


def build_true_profile(p, extra_loss_db):
    sat = p["sat"]
    rise_dt = p["rise"].utc_datetime()
    set_dt = p["set"].utc_datetime()
    total_s = int((set_dt - rise_dt).total_seconds())
    fine_times = ts.utc([rise_dt + timedelta(seconds=s) for s in range(0, total_s + 1)])
    alt, az, dist = (sat - observer).at(fine_times).altaz()
    df = pd.DataFrame({"seconds_from_rise": list(range(0, total_s + 1)), "slant_range_km": dist.km})
    df["cnr_db_true"] = df["slant_range_km"].apply(lambda d: cnr_db_with_extra_loss(d, extra_loss_db))
    return df


def crossing_time(series_val, threshold, start_idx=None):
    sub = series_val if start_idx is None else series_val.loc[start_idx:]
    below = sub[sub <= threshold]
    return below.index[0] if not below.empty else None


def run_experiment_for_condition(extra_loss_db, min_max_elev):
    all_passes = find_passes(min_max_elev)
    if not all_passes:
        return {"extra_loss_db": extra_loss_db, "min_max_elev": min_max_elev,
                "n_candidates": 0, "proposed_failure": np.nan, "guard15_failure": np.nan}

    outcomes = {"proposed": [], "guard15": []}
    n_usable = 0
    for i in range(N_TRIALS):
        seed = SEED_BASE + i
        rng = np.random.default_rng(seed)
        p = all_passes[rng.integers(0, len(all_passes))]
        df = build_true_profile(p, extra_loss_db)
        peak_idx = df["cnr_db_true"].idxmax()
        true_expiry_idx = crossing_time(df["cnr_db_true"], CNR_MIN_DB, start_idx=peak_idx)
        if true_expiry_idx is None:
            continue
        n_usable += 1
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
        sigma_T_s = np.sqrt(sigma_T_predict**2 + SIGMA_MEAS_S**2 + SIGMA_DELAY_S**2).clip(upper=SIGMA_T_CAP_S)
        T_safe = descending["remaining_obs_s"] - Z_ALPHA_HEADLINE * sigma_T_s

        for policy_name, hit in [
            ("proposed", descending[T_safe.notna() & (T_safe <= TOTAL_OVERHEAD_NOMINAL_S)]),
            ("guard15", descending[descending["remaining_obs_s"] <= 15]),
        ]:
            if hit.empty:
                outcomes[policy_name].append(True)
                continue
            trigger_s = hit.iloc[0]["seconds_from_rise"]
            margin_true = (true_expiry_s - trigger_s) - actual_overhead_s
            outcomes[policy_name].append(margin_true < 0)

    return {
        "extra_loss_db": extra_loss_db, "min_max_elev": min_max_elev,
        "n_candidates": len(all_passes), "n_usable_trials": n_usable,
        "proposed_failure": np.mean(outcomes["proposed"]) if outcomes["proposed"] else np.nan,
        "guard15_failure": np.mean(outcomes["guard15"]) if outcomes["guard15"] else np.nan,
    }


print("--- Sweep 1: extra implementation/atmospheric loss (fixed candidacy=20deg) ---")
loss_rows = [run_experiment_for_condition(loss, 20.0) for loss in EXTRA_LOSS_SWEEP_DB]
loss_df = pd.DataFrame(loss_rows)
print(loss_df.to_string(index=False))
loss_df.to_csv("results/experiment4_extra_loss_sweep.csv", index=False)

print("\n--- Sweep 2: minimum elevation candidacy (fixed extra_loss=0dB) ---")
elev_rows = [run_experiment_for_condition(0.0, elev) for elev in MIN_MAX_ELEV_CANDIDACY_SWEEP]
elev_df = pd.DataFrame(elev_rows)
print(elev_df.to_string(index=False))
elev_df.to_csv("results/experiment4_elevation_candidacy_sweep.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(loss_df["extra_loss_db"], loss_df["proposed_failure"], marker="o", label="Proposed")
ax1.plot(loss_df["extra_loss_db"], loss_df["guard15_failure"], marker="s", label="Fixed guard-15s")
ax1.set_xlabel("Extra implementation/atmospheric loss (dB)")
ax1.set_ylabel("Handover failure probability")
ax1.set_title("Sensitivity to link degradation")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(elev_df["min_max_elev"], elev_df["proposed_failure"], marker="o", label="Proposed")
ax2.plot(elev_df["min_max_elev"], elev_df["guard15_failure"], marker="s", label="Fixed guard-15s")
ax2.set_xlabel("Minimum candidacy elevation (deg)")
ax2.set_ylabel("Handover failure probability")
ax2.set_title("Sensitivity to pass geometry (weak vs strong passes)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/experiment4_link_margin_conditions.png", dpi=150)
print("Saved figures/experiment4_link_margin_conditions.png")