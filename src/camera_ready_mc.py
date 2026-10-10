"""
Camera-ready Monte Carlo (ISETC'26 revision).
Run from repo root:  python src/camera_ready_mc.py
"""
import os, re, json, hashlib, time
import numpy as np, pandas as pd
from scipy.stats import beta, norm
from skyfield.api import load, wgs84

# ------------------------------------------------------------------ settings
TLE_FILE   = "data/tle_archive/starlink.tle"
CFG_FILE   = "config/scenario_config.yaml"
SAT_NAMES  = ["STARLINK-1008", "STARLINK-1012", "STARLINK-1017",
              "STARLINK-1020", "STARLINK-1036", "STARLINK-1039"]
LAT, LON, ALT_M = 57.1497, -2.0943, 20.0       # overridden by scenario_config.yaml if keys found
N_SATS     = 100            # total satellites used (the six above + next ones in the TLE file)
WINDOW_H   = 72             # pass-pool window (hours from first TLE epoch)
DT_S       = 1.0            # simulation time step (s)
MIN_PEAK_ELEV = 20.0        # candidacy filter (deg)
FC_MHZ, EIRP_DBW, GT_DB, K_DB, L_ATM, BW_HZ, CNR_MIN = 2000.0, 34+10*np.log10(30), -36.6, -228.6, 0.5, 30e6, 0.0
SIG_MARGIN, SIG_MEAS, SIG_DELAY = 1.5, 0.2, 1.0
T_OVERHEAD = 0.5 + 2.0 + 0.1 + 0.5                 # 3.1 s
SLOPE_FLOOR, SIGMA_CAP = 0.01, 60.0
PREMATURE_S = 30.0
Z_OP = 1.0
TARGET_FAIL = 0.01
N_TEST, N_CAL, SEED_TEST, SEED_CAL = 1000, 1000, 42, 100000
Z_GRID  = [0, 0.25, 0.5, 0.84, 1.0, 1.28, 1.645, 2.0, 2.33, 3.0]
G_GRID  = [3.1, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120]
OUT = "results/camera_ready"

def cp_ci(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else beta.ppf(1-a/2, k+1, n-k)
    return lo, hi

# ------------------------------------------------------------------ passes
def read_cfg():
    global LAT, LON, ALT_M
    if not os.path.exists(CFG_FILE): return
    txt = open(CFG_FILE).read()
    for name, pat in (("LAT", r"\b(?:latitude|lat)(?:_deg)?\s*:\s*(-?[\d.]+)"),
                      ("LON", r"\b(?:longitude|lon|lng)(?:_deg)?\s*:\s*(-?[\d.]+)"),
                      ("ALT", r"\b(?:altitude|alt|elevation)(?:_m|_km)?\s*:\s*(-?[\d.]+)")):
        m = re.search(pat, txt, re.I)
        if m:
            v = float(m.group(1))
            if name == "LAT": LAT = v
            elif name == "LON": LON = v
            else: ALT_M = v if v > 5 else v*1000   # tolerate km

def cnr_db(d_km):
    fspl = 32.45 + 20*np.log10(FC_MHZ) + 20*np.log10(d_km)
    return EIRP_DBW + GT_DB - fspl - L_ATM - K_DB - 10*np.log10(BW_HZ)

def build_passes():
    global SAT_NAMES
    ts = load.timescale()
    sats = {s.name: s for s in load.tle_file(TLE_FILE)}
    SAT_NAMES = SAT_NAMES + [n for n in sats if n not in SAT_NAMES][:max(0, N_SATS-len(SAT_NAMES))]
    missing = [n for n in SAT_NAMES if n not in sats]
    if missing: raise SystemExit(f"Satellites not in TLE file: {missing}")
    t0 = sats[SAT_NAMES[0]].epoch
    n = int(WINDOW_H*3600/DT_S)
    obs = wgs84.latlon(LAT, LON, elevation_m=ALT_M)
    CH = 6*3600                                   # 6-h chunks to limit memory
    passes = []
    for nm in SAT_NAMES:
        els, ds = [], []
        for c0 in range(0, n, CH):
            m = min(CH, n-c0)
            t = ts.tt_jd(t0.tt + (c0+np.arange(m))*DT_S/86400.0)
            alt, az, dist = (sats[nm]-obs).at(t).altaz()
            els.append(alt.degrees); ds.append(dist.km)
        el, d = np.concatenate(els), np.concatenate(ds)
        up = el > 0
        edges = np.diff(up.astype(int), prepend=0, append=0)
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if s == 0 or e >= n: continue          # skip truncated passes
            if el[s:e].max() < MIN_PEAK_ELEV: continue
            passes.append(dict(sat=nm, start=s, el=el[s:e], d=d[s:e], cnr=cnr_db(d[s:e])))
    return passes, t0.utc_iso()

def prep(p):
    c = p["cnr"]; pk = int(np.argmax(c)); p["pk"] = pk
    below = np.where(c[pk:] < CNR_MIN)[0]
    p["exp"] = pk + int(below[0]) if len(below) else None     # true expiry index
    p["slope"] = np.abs(np.gradient(c, DT_S))
    return p

# ------------------------------------------------------------------ one trial, all policies
def draw_cases(passes, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(passes), n)
    return [(int(i), rng.normal(0, SIG_MARGIN), rng.normal(0, SIG_DELAY)) for i in idx]

def policy_triggers(p, bias, params):
    """All policies see the same observed CNR (true + bias)."""
    c, pk = p["cnr"], p["pk"]
    obs = c + bias
    idx = np.arange(pk, len(c))
    cb = np.where(obs[pk:] < CNR_MIN)[0]
    crossing = pk + int(cb[0]) if len(cb) else len(c)          # fallback: pass end
    That = (crossing - idx) * DT_S                              # predicted remaining usable time
    sig_geom = params["sig_margin_assumed"] / np.maximum(p["slope"][pk:], params["slope_floor"])
    var = (sig_geom**2 if params["use_geom"] else 0) \
        + (SIG_MEAS**2 if params["use_meas"] else 0) + (SIG_DELAY**2 if params["use_delay"] else 0)
    sigT = np.minimum(np.sqrt(var), params["cap"])
    out = {}
    def first(mask):
        w = np.where(mask)[0]
        return pk + int(w[0]) if len(w) else None
    for z in params["z_list"]:
        out[("prop", z)] = first(That - z*sigT <= T_OVERHEAD)
    for g in params["g_list"]:
        out[("fixed", g)] = first(That <= g)
    out[("qual", 3.0)] = first(obs[pk:] <= 3.0)
    return out

def score(p, trig, overhead):
    """-> (fail, interruption_s, retained_s, excess_s)"""
    e = p["exp"]
    if trig is None: return 1, overhead, 0.0, 0.0
    retained = (e - trig) * DT_S                                # true serving-link time left at trigger
    if retained < overhead: return 1, overhead - retained, retained, 0.0
    return 0, 0.0, retained, retained - overhead

def run(cases, passes, params):
    rows = []
    for k, (pi, bias, dn) in enumerate(cases):
        p = passes[pi]
        if p["exp"] is None: continue                            # excluded: never drops below CNR_min after peak
        ov = max(0.1, T_OVERHEAD + dn)
        for pol, trig in policy_triggers(p, bias, params).items():
            f, itr, ret, ex = score(p, trig, ov)
            rows.append((k, pi, pol[0], pol[1], f, itr, ret, ex))
    return pd.DataFrame(rows, columns=["trial","pass","policy","param","fail","interrupt_s","retained_s","excess_s"])

def summarise(df, n_total):
    out = []
    for (pol, par), g in df.groupby(["policy","param"]):
        n = len(g); k = int(g.fail.sum()); lo, hi = cp_ci(k, n)
        ok = g[g.fail == 0]
        m = lambda s: (s.mean(), 1.96*s.std(ddof=1)/np.sqrt(len(s))) if len(s) > 1 else (np.nan, np.nan)
        ex, exc = m(ok.excess_s); rt, rtc = m(g.retained_s)
        out.append(dict(policy=pol, param=par, n_total=n_total, n_valid=n, fails=k, fail_rate=k/n,
            fail_lo=lo, fail_hi=hi, interrupt_mean=g.interrupt_s.mean(),
            retained_mean=rt, retained_ci=rtc, excess_mean=ex, excess_ci=exc,
            premature_rate=(ok.excess_s > PREMATURE_S).sum()/n))
    return pd.DataFrame(out)

def base_params(**kw):
    p = dict(sig_margin_assumed=SIG_MARGIN, slope_floor=SLOPE_FLOOR, cap=SIGMA_CAP,
             use_geom=True, use_meas=True, use_delay=True, z_list=Z_GRID, g_list=G_GRID)
    p.update(kw); return p

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t_start = time.time(); os.makedirs(OUT, exist_ok=True)
    read_cfg()
    passes, epoch = build_passes()
    passes = [prep(p) for p in passes]
    order = np.random.default_rng(7).permutation(len(passes))      # disjoint calibration / test passes
    cal_p = [passes[i] for i in order[: len(passes)//2]]
    tst_p = [passes[i] for i in order[len(passes)//2:]]
    sha = hashlib.sha256(open(TLE_FILE,"rb").read()).hexdigest()
    meta = dict(n_sats=len(SAT_NAMES), lat=LAT, lon=LON, alt_m=ALT_M, window_h=WINDOW_H, dt_s=DT_S, tle_sha256=sha, first_epoch=epoch,
                n_pass_pool=len(passes), n_cal_passes=len(cal_p), n_test_passes=len(tst_p),
                n_excluded_pool=int(sum(p["exp"] is None for p in passes)), n_test=N_TEST, n_cal=N_CAL,
                seeds=dict(test=SEED_TEST, cal=SEED_CAL), z_op=Z_OP, target_fail=TARGET_FAIL)
    print(json.dumps(meta, indent=1))

    # 1) calibration on calibration passes only
    cal = summarise(run(draw_cases(cal_p, N_CAL, SEED_CAL), cal_p, base_params()), N_CAL)
    cal.to_csv(f"{OUT}/calibration_summary.csv", index=False)
    def tuned(pol):
        c = cal[(cal.policy == pol) & (cal.fail_rate <= TARGET_FAIL)]
        return float(c.param.min()) if len(c) else None
    z_star, g_star = tuned("prop"), tuned("fixed")
    meta.update(z_tuned=z_star, g_tuned=g_star); print("tuned z*:", z_star, " tuned fixed guard G*:", g_star)

    # 2) test set (independent passes), every policy on identical cases
    tdf = run(draw_cases(tst_p, N_TEST, SEED_TEST), tst_p, base_params())
    tdf.to_csv(f"{OUT}/test_raw.csv", index=False)
    ts_ = summarise(tdf, N_TEST); ts_.to_csv(f"{OUT}/test_summary_all.csv", index=False)
    sel = [("prop", 0.84), ("prop", Z_OP)] + ([("prop", z_star)] if z_star is not None else []) + \
          [("fixed", g) for g in (5.0, 15.0, 30.0, 60.0)] + ([("fixed", g_star)] if g_star is not None else []) + \
          [("fixed", 3.1), ("qual", 3.0)]
    tab = pd.concat([ts_[(ts_.policy == a) & np.isclose(ts_.param, b)] for a, b in sel]).drop_duplicates(["policy","param"])
    tab.to_csv(f"{OUT}/table2_operating_point.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nTABLE II (test set):\n", tab.round(4).to_string(index=False))

    # 3) empirical coverage of the lower bound vs z
    rows = []
    for lead in (30, 60, 120):
        cov = {z: [] for z in Z_GRID}
        for pi, bias, _ in draw_cases(tst_p, N_TEST, SEED_TEST):
            p = tst_p[pi]
            if p["exp"] is None: continue
            i = p["exp"] - lead
            if i < p["pk"]: continue
            obs = p["cnr"] + bias
            cb = np.where(obs[p["pk"]:] < CNR_MIN)[0]
            cr = p["pk"] + int(cb[0]) if len(cb) else len(p["cnr"])
            That = (cr - i)*DT_S
            sg = min(SIGMA_CAP, np.sqrt((SIG_MARGIN/max(p["slope"][i], SLOPE_FLOOR))**2 + SIG_MEAS**2 + SIG_DELAY**2))
            for z in Z_GRID: cov[z].append(lead >= That - z*sg)   # true remaining >= T_safe
        for z in Z_GRID:
            v = np.array(cov[z]); k = int(v.sum()); lo, hi = cp_ci(k, len(v))
            rows.append(dict(lead_s=lead, z=z, n=len(v), empirical_cov=k/len(v), lo=lo, hi=hi, nominal=norm.cdf(z)))
    pd.DataFrame(rows).to_csv(f"{OUT}/coverage_vs_z.csv", index=False)
    print("\nCOVERAGE:\n", pd.DataFrame(rows).round(3).to_string(index=False))

    # 4) sensitivity to slope floor and sigma_T cap
    rows = []
    cases = draw_cases(tst_p, N_TEST, SEED_TEST)
    for fl in (0.005, 0.01, 0.02, 0.05):
        for cap in (30.0, 60.0, 120.0, 1e9):
            s = summarise(run(cases, tst_p, base_params(slope_floor=fl, cap=cap, z_list=[Z_OP], g_list=[])), N_TEST)
            r = s[s.policy == "prop"].iloc[0].to_dict(); r.update(slope_floor=fl, cap=cap); rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/sensitivity_floor_cap.csv", index=False)
    print("\nSENSITIVITY:\n", pd.DataFrame(rows)[["slope_floor","cap","fails","fail_rate","fail_lo","fail_hi","excess_mean","premature_rate"]].round(4).to_string(index=False))

    # 5) ablation at z = Z_OP
    rows = []
    for name, kw in [("Full", {}), ("No uncertainty (z=0)", {"z_list":[0.0]}), ("No meas. term", {"use_meas":False}),
                     ("No delay term", {"use_delay":False}), ("No geometry/margin term", {"use_geom":False})]:
        kw = dict(kw); kw.setdefault("z_list", [Z_OP]); kw["g_list"] = []
        s = summarise(run(cases, tst_p, base_params(**kw)), N_TEST)
        r = s[s.policy == "prop"].iloc[0].to_dict(); r["variant"] = name; rows.append(r)
    ab = pd.DataFrame(rows); ab.to_csv(f"{OUT}/ablation.csv", index=False)
    print("\nABLATION:\n", ab[["variant","n_valid","fails","fail_rate","fail_lo","fail_hi","excess_mean","premature_rate"]].round(4).to_string(index=False))

    # 6) uncertainty sweep: TRUE margin sigma varies, policy assumes 1.5 dB
    rows = []
    for sm in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        rng = np.random.default_rng(SEED_TEST)
        cs = [(pi, rng.normal(0, sm), dn) for pi, _, dn in cases]
        s = summarise(run(cs, tst_p, base_params(z_list=[Z_OP], g_list=[15.0, 30.0, 60.0])), N_TEST)
        s["sigma_margin_true"] = sm; rows.append(s)
    pd.concat(rows).to_csv(f"{OUT}/exp2_uncertainty_sweep.csv", index=False)
    print("\nEXP2:\n", pd.concat(rows)[["sigma_margin_true","policy","param","n_valid","fails","fail_rate","fail_hi","premature_rate","excess_mean"]].round(4).to_string(index=False))

    meta["runtime_s"] = round(time.time() - t_start, 1)
    json.dump(meta, open(f"{OUT}/run_metadata.json", "w"), indent=1)
    print("\nDone. Outputs in", OUT, " runtime", meta["runtime_s"], "s")