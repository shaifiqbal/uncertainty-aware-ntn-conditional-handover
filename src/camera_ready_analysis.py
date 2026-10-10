"""
Pass-clustered analysis of results/camera_ready/test_raw.csv (run AFTER camera_ready_mc.py).
Run from repo root:  python src/camera_ready_analysis.py
"""
import numpy as np, pandas as pd

RAW = "results/camera_ready/test_raw.csv"
PROPS = [1.0, 0.84]
REFS = [("fixed", 60.0), ("fixed", 30.0), ("qual", 3.0)]
PREM_S, B = 30.0, 5000
rng = np.random.default_rng(2026)

df = pd.read_csv(RAW)
def sel(pol, par):
    return df[(df.policy == pol) & np.isclose(df.param, par)].set_index("trial")

def boot_ratio(num, den, grp):
    g = pd.DataFrame({"n": num, "d": den, "g": grp}).groupby("g").sum()
    N, D = g.n.values, g.d.values
    idx = rng.integers(0, len(g), (B, len(g)))
    r = N[idx].sum(1) / D[idx].sum(1)
    return N.sum()/D.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)

base = sel("prop", PROPS[0])
print(f"Valid trials: {len(base)}   distinct passes behind them: {base['pass'].nunique()}")
print(f"Trials per distinct pass: mean {len(base)/base['pass'].nunique():.1f}\n")

rows = []
for z in PROPS:
    p = sel("prop", z)
    for ref in REFS:
        r = sel(*ref).reindex(p.index)
        ok = (p.fail == 0) & (r.fail == 0)
        d = (r.excess_s - p.excess_s)[ok]; grp = p["pass"][ok]
        m, lo, hi = boot_ratio(d.values, np.ones(len(d)), grp.values)
        rel, rlo, rhi = boot_ratio(d.values, r.excess_s[ok].values, grp.values)
        rows.append(dict(prop_z=z, reference=f"{ref[0]}-{ref[1]:g}", n_pairs=int(ok.sum()),
                         prop_fails=int(p.fail.sum()), ref_fails=int(r.fail.sum()),
                         saving_s=m, saving_lo=lo, saving_hi=hi, rel=rel, rel_lo=rlo, rel_hi=rhi))
out = pd.DataFrame(rows)
print("Paired excess-margin saving of proposed vs reference (pass-clustered 95% CI):")
print(out.round(3).to_string(index=False))

print("\nPremature-handover rate (excess > %.0f s), pass-clustered 95%% CI:" % PREM_S)
items = [(f"prop-{z:g}", sel("prop", z)) for z in PROPS] + [(f"{a}-{b:g}", sel(a, b).reindex(base.index)) for a, b in REFS]
for name, q in items:
    prem = ((q.fail == 0) & (q.excess_s > PREM_S)).astype(float).values
    m, lo, hi = boot_ratio(prem, np.ones(len(prem)), base["pass"].values)
    print(f"  {name:12s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")

print("\nFailures (valid trials):", {n: int(q.fail.sum()) for n, q in items})
out.to_csv("results/camera_ready/paired_saving_clustered.csv", index=False)
print("Saved results/camera_ready/paired_saving_clustered.csv")