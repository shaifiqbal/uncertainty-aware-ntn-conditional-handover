from skyfield.api import load, wgs84
import pandas as pd
import numpy as np
from datetime import timedelta
import sys

sys.path.insert(0, "src")
from link_budget import cnr_db, CNR_MIN_DB
from remaining_link_time import compute_remaining_link_time
from uncertainty_model import compute_uncertainty_sigma_T, compute_safe_time, Z_ALPHA_DEFAULT
from baselines import (baseline_fixed_guard, baseline_quality_threshold,
                        proposed_policy_trigger, evaluate_trigger, TOTAL_OVERHEAD_S)

GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65
MIN_ELEVATION_DEG = 0.0
SEARCH_HOURS = 72

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
            all_passes.append({
                "sat": sat, "rise": rise_t, "set": set_t,
                "duration_s": (set_t.utc_datetime() - rise_t.utc_datetime()).total_seconds(),
                "max_elevation_deg": alt.degrees
            })
        i += 1

print(f"Found {len(all_passes)} total passes over next {SEARCH_HOURS}h across {len(subset)} satellites")

# Pick 5 diverse passes by max elevation: min, 25th pct, median, 75th pct, max
MIN_MAX_ELEV_FOR_CANDIDACY = 10.0  # from config/scenario_config.yaml
all_passes = [p for p in all_passes if p["max_elevation_deg"] >= MIN_MAX_ELEV_FOR_CANDIDACY]
print(f"{len(all_passes)} passes remain after filtering to max_elevation >= {MIN_MAX_ELEV_FOR_CANDIDACY} deg candidacy threshold")
all_passes.sort(key=lambda p: p["max_elevation_deg"])
n = len(all_passes)
if n < 5:
    print(f"WARNING: only {n} passes found, using all of them")
    selected = all_passes
else:
    idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    selected = [all_passes[i] for i in sorted(set(idxs))]

print(f"Selected {len(selected)} passes for manual sanity check:\n")

summary_rows = []
for p in selected:
    sat = p["sat"]
    rise_dt = p["rise"].utc_datetime()
    set_dt = p["set"].utc_datetime()
    total_s = int((set_dt - rise_dt).total_seconds())
    fine_times = ts.utc([rise_dt + timedelta(seconds=s) for s in range(0, total_s + 1)])

    difference = sat - observer
    alt, az, dist = difference.at(fine_times).altaz()

    df = pd.DataFrame({
        "seconds_from_rise": list(range(0, total_s + 1)),
        "elevation_deg": alt.degrees,
        "azimuth_deg": az.degrees,
        "slant_range_km": dist.km,
    })
    df["cnr_db"] = df["slant_range_km"].apply(cnr_db)
    df["usable"] = df["cnr_db"] >= CNR_MIN_DB

    df, start_idx, end_idx = compute_remaining_link_time(df)
    df = compute_uncertainty_sigma_T(df)
    df = compute_safe_time(df)

    row = {"satellite": sat.name, "max_elevation_deg": round(p["max_elevation_deg"], 2),
           "duration_s": round(p["duration_s"], 1)}

    proposed = evaluate_trigger(proposed_policy_trigger(df))
    guard15 = evaluate_trigger(baseline_fixed_guard(df, 15))
    quality = evaluate_trigger(baseline_quality_threshold(df, 3.0))

    row["proposed_status"] = proposed.get("status") if isinstance(proposed, dict) else proposed
    row["proposed_trigger_s"] = proposed.get("trigger_time_s") if isinstance(proposed, dict) else None
    row["guard15_status"] = guard15.get("status") if isinstance(guard15, dict) else guard15
    row["quality3db_status"] = quality.get("status") if isinstance(quality, dict) else quality

    summary_rows.append(row)
    print(f"{sat.name}: max_elev={p['max_elevation_deg']:.1f} deg, duration={p['duration_s']:.0f}s")
    print(f"  Proposed: {proposed}")
    print(f"  Guard-15s: {guard15}")
    print(f"  Quality-3dB: {quality}")
    print()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/multi_pass_sanity_check.csv", index=False)
print("Saved results/multi_pass_sanity_check.csv")