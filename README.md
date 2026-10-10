# Uncertainty-Aware Conditional Handover Timing in 5G NR LEO NTN

Simulation code and reproducibility materials for a risk-aware conditional-handover
timing policy in 5G NR LEO non-terrestrial networks. The policy triggers on a lower
confidence bound of the predicted remaining usable link time, and is evaluated by Monte
Carlo simulation against fixed-guard-time, point-prediction, quality-threshold and
conservative fixed-margin baselines.

**Status:** paper accepted at ISETC 2026 (17th International Symposium on Electronics and
Telecommunications, Timisoara, Romania). The camera-ready results are in the tag
`v2.0-camera-ready`.

Author: Shaif Khan (Independent Researcher, Aberdeen, UK)

All results are simulation-based, for one ground location (Aberdeen), using public Starlink
TLE data. They do not represent a commercial NTN deployment.

## Repository structure
- `data/tle_archive/` - archived TLE data (source, timestamp and checksum logged)
- `data/satellite_list.txt` - the 100 satellites used (name, NORAD id, inclination, mean altitude)
- `src/` - simulation code
- `config/` - scenario configuration files
- `results/` - raw simulation outputs (CSV)
- `results/camera_ready/` - outputs behind the camera-ready paper
- `figures/` - generated plots (`figures/camera_ready/` for the paper figures)
- `tests/` - sanity checks
- `literature/` - literature matrix export

## Reproducing the camera-ready results
Environment: Python with `skyfield`, `numpy`, `pandas`, `scipy` and `matplotlib`.
Run from the repository root.

```
python src/camera_ready_mc.py          # main Monte Carlo run (about 15-25 minutes)
python src/camera_ready_analysis.py    # paired, pass-clustered comparison
python src/camera_ready_figures.py     # risk sweep, sigma sweep, coverage figures
python src/export_satellite_list.py    # writes data/satellite_list.txt
python src/camera_ready_fig2.py        # pass example figure
python src/camera_ready_fig3.py        # trigger-point example figure
```

Key settings: 100 satellites, 72 h window, 1 s time step, minimum peak elevation 20 degrees,
calibration seed 100000 and test seed 42. The calibration and test halves use disjoint
passes. The TLE file was downloaded from CelesTrak on 2026-08-22 (first epoch
2026-08-21). Its SHA-256 checksum is
`1f68ffe53e866a9ae2649957fa05fcf54b07f6fce21112f08d91c3ae6eb2b86f` and is also recorded in
`results/camera_ready/run_metadata.json`.

## Limitations
One location, a documented satellite subset that includes satellites outside the
operational shell, assumed (not measured) uncertainty values, and no target-satellite
availability or signalling model. Field validation on a testbed is future work.