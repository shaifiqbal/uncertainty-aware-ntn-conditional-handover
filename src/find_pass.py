from skyfield.api import load, wgs84
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65
MIN_ELEVATION_DEG = 0.0   # search threshold for a "usable" pass start/end

observer = wgs84.latlon(GROUND_LAT, GROUND_LON, GROUND_ALT_M)
ts = load.timescale()
satellites = load.tle_file("data/tle_archive/starlink.tle")

subset_names = ["STARLINK-1008", "STARLINK-1012", "STARLINK-1017",
                 "STARLINK-1020", "STARLINK-1036", "STARLINK-1039"]
by_name = {sat.name: sat for sat in satellites}
subset = [by_name[n] for n in subset_names if n in by_name]

t0 = ts.now()
t1 = ts.utc(t0.utc_datetime() + timedelta(hours=24))

best_pass = None

for sat in subset:
    times, events = sat.find_events(observer, t0, t1, altitude_degrees=MIN_ELEVATION_DEG)
    # events: 0 = rise above threshold, 1 = culminate (max elevation), 2 = set below threshold
    i = 0
    while i < len(events):
        if events[i] == 0:  # found a rise
            # find matching culminate and set
            if i + 2 < len(events) and events[i+1] == 1 and events[i+2] == 2:
                rise_t, culminate_t, set_t = times[i], times[i+1], times[i+2]
                duration_s = (set_t.utc_datetime() - rise_t.utc_datetime()).total_seconds()
                alt, az, dist = (sat - observer).at(culminate_t).altaz()
                candidate = {
                    "sat": sat, "rise": rise_t, "culminate": culminate_t, "set": set_t,
                    "duration_s": duration_s, "max_elevation_deg": alt.degrees
                }
                if best_pass is None or candidate["max_elevation_deg"] > best_pass["max_elevation_deg"]:
                    best_pass = candidate
        i += 1

if best_pass is None:
    print(f"No pass above {MIN_ELEVATION_DEG} deg found for any satellite in next 24h. "
          f"Try lowering MIN_ELEVATION_DEG.")
else:
    sat = best_pass["sat"]
    print(f"Best pass found: {sat.name}")
    print(f"  Rise: {best_pass['rise'].utc_iso()}")
    print(f"  Culminate: {best_pass['culminate'].utc_iso()} (max elev {best_pass['max_elevation_deg']:.2f} deg)")
    print(f"  Set: {best_pass['set'].utc_iso()}")
    print(f"  Duration: {best_pass['duration_s']:.1f} s")

    # Generate fine-resolution profile across the full pass (1-second steps)
    rise_dt = best_pass["rise"].utc_datetime()
    set_dt = best_pass["set"].utc_datetime()
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
    df.to_csv("results/full_pass_example.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["seconds_from_rise"], df["elevation_deg"])
    ax1.axhline(MIN_ELEVATION_DEG, color="red", linestyle="--", linewidth=0.7, label=f"{MIN_ELEVATION_DEG} deg threshold")
    ax1.set_ylabel("Elevation (deg)")
    ax1.set_title(f"Full pass example: {sat.name} over Aberdeen")
    ax1.legend()

    ax2.plot(df["seconds_from_rise"], df["slant_range_km"])
    ax2.set_ylabel("Slant range (km)")
    ax2.set_xlabel("Seconds from rise (AOS)")

    plt.tight_layout()
    plt.savefig("figures/full_pass_example.png", dpi=150)
    print("Saved results/full_pass_example.csv and figures/full_pass_example.png")