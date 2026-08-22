from skyfield.api import load, wgs84, EarthSatellite
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

# Ground station: Aberdeen, Scotland
GROUND_LAT = 57.1497
GROUND_LON = -2.0943
GROUND_ALT_M = 65

observer = wgs84.latlon(GROUND_LAT, GROUND_LON, GROUND_ALT_M)

# Load our documented small subset directly from the archived file
ts = load.timescale()
satellites = load.tle_file("data/tle_archive/starlink.tle")
subset_names = ["STARLINK-1008", "STARLINK-1012", "STARLINK-1017",
                 "STARLINK-1020", "STARLINK-1036", "STARLINK-1039"]
by_name = {sat.name: sat for sat in satellites}
subset = [by_name[n] for n in subset_names if n in by_name]

print(f"Loaded {len(subset)} of {len(subset_names)} requested satellites")

# Diagnostic pass: track one satellite over a 20-minute window at fine resolution
sat = subset[0]
t0 = ts.now()
seconds_offsets = range(0, 20 * 60, 10)  # every 10 seconds for 20 minutes
times = ts.utc([(t0.utc_datetime() + timedelta(seconds=s)) for s in seconds_offsets])

difference = sat - observer
topocentric = difference.at(times)
alt, az, distance = topocentric.altaz()

df = pd.DataFrame({
    "seconds_from_start": list(seconds_offsets),
    "elevation_deg": alt.degrees,
    "azimuth_deg": az.degrees,
    "slant_range_km": distance.km,
})

df.to_csv("results/diagnostic_pass.csv", index=False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(df["seconds_from_start"], df["elevation_deg"])
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax1.set_ylabel("Elevation (deg)")
ax1.set_title(f"Diagnostic pass: {sat.name} over Aberdeen")

ax2.plot(df["seconds_from_start"], df["slant_range_km"])
ax2.set_ylabel("Slant range (km)")
ax2.set_xlabel("Seconds from start")

plt.tight_layout()
plt.savefig("figures/diagnostic_pass.png", dpi=150)
print("Saved results/diagnostic_pass.csv and figures/diagnostic_pass.png")
print(df.head(10))