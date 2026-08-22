import requests, hashlib, json
from datetime import datetime, timezone

URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
OUT_TLE = "data/tle_archive/starlink.tle"
OUT_META = "data/tle_archive/starlink_meta.json"

resp = requests.get(URL, timeout=30)
resp.raise_for_status()
content = resp.text

with open(OUT_TLE, "w") as f:
    f.write(content)

checksum = hashlib.sha256(content.encode()).hexdigest()
meta = {
    "source_url": URL,
    "group": "starlink",
    "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "sha256_checksum": checksum,
}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved {OUT_TLE}, checksum {checksum[:12]}...")