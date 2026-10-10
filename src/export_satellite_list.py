"""
Writes the exact satellite list used by camera_ready_mc.py (same selection rule) and summarises
inclination / altitude. Run from repo root:  python src/export_satellite_list.py
"""
import numpy as np
from skyfield.api import load
import camera_ready_mc as m          # reuses TLE_FILE, SAT_NAMES, N_SATS

sats = {s.name: s for s in load.tle_file(m.TLE_FILE)}
names = m.SAT_NAMES + [n for n in sats if n not in m.SAT_NAMES][:max(0, m.N_SATS - len(m.SAT_NAMES))]
MU = 398600.4418; RE = 6378.137
rows = []
for n in names:
    mod = sats[n].model
    inc = np.degrees(mod.inclo)
    a = (MU / (mod.no_kozai / 60.0) ** 2) ** (1/3)
    rows.append((n, sats[n].model.satnum, inc, a - RE))
with open("data/satellite_list.txt", "w") as f:
    f.write("name,norad_id,inclination_deg,mean_altitude_km\n")
    for r in rows: f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.0f}\n")
inc = np.array([r[2] for r in rows]); alt = np.array([r[3] for r in rows])
print(f"{len(rows)} satellites written to data/satellite_list.txt")
print("Inclination (deg): min %.2f  max %.2f" % (inc.min(), inc.max()))
print("Inclination values (rounded) and counts:", dict(zip(*np.unique(np.round(inc, 0), return_counts=True))))
print("Mean altitude (km): min %.0f  max %.0f" % (alt.min(), alt.max()))
print("First NORAD id: %d  last: %d" % (rows[0][1], rows[-1][1]))