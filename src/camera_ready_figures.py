"""
Camera-ready figures from results/camera_ready/*.csv (run AFTER camera_ready_mc.py).
Run from repo root:  python src/camera_ready_figures.py
Writes PNGs (300 dpi, 3.5-inch column width, large labels, grids) to figures/camera_ready/
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

R, F = "results/camera_ready", "figures/camera_ready"
os.makedirs(F, exist_ok=True)
Z_OP = json.load(open(f"{R}/run_metadata.json"))["z_op"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.4, "lines.markersize": 4.5})
W = 3.5

# ---- Fig A: uncertainty sweep (true margin sigma), distinct markers/styles + exact CIs
e = pd.read_csv(f"{R}/exp2_uncertainty_sweep.csv")
specs = [("prop", Z_OP, f"Proposed ($z_\\alpha$={Z_OP:g})", "o", "-", "#1f4e9c", -0.05),
         ("fixed", 60.0, "Fixed guard 60 s", "s", "--", "#d9822b", 0.05),
         ("fixed", 30.0, "Fixed guard 30 s", "^", ":", "#6a9955", 0.0),
         ("qual", 3.0, "Quality thr. 3 dB", "D", "-.", "#8b5fbf", 0.0)]
fig, ax = plt.subplots(2, 1, figsize=(W, 4.1), sharex=True)
for pol, par, lab, mk, ls, col, dx in specs:
    g = e[(e.policy == pol) & np.isclose(e.param, par)].sort_values("sigma_margin_true")
    x = g.sigma_margin_true + dx
    ax[0].errorbar(x, g.fail_rate*100, yerr=[(g.fail_rate-g.fail_lo)*100, (g.fail_hi-g.fail_rate)*100],
                   marker=mk, ls=ls, color=col, capsize=2, label=lab)
    ax[1].errorbar(x, g.excess_mean, yerr=g.excess_ci, marker=mk, ls=ls, color=col, capsize=2)
ax[0].set_yscale("symlog", linthresh=1.0); ax[0].set_ylim(0, 60)
ax[0].set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50]); ax[0].set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50"])
ax[0].set_ylabel("Failure probability (%)"); ax[1].set_ylabel("Mean excess margin (s)")
ax[1].set_xlabel("True $\\sigma_{margin}$ (dB); policy assumes 1.5 dB")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, loc="upper center", frameon=False, handlelength=2.4)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(f"{F}/fig_sigma_sweep.png", dpi=300); plt.close(fig)

# ---- Fig B: coverage vs z
c = pd.read_csv(f"{R}/coverage_vs_z.csv")
fig, ax = plt.subplots(figsize=(W, 2.5))
for lead, mk, ls in zip((30, 60, 120), ("o", "s", "^"), ("-", "--", ":")):
    g = c[c.lead_s == lead]
    ax.errorbar(g.z, g.empirical_cov, yerr=[g.empirical_cov-g.lo, g.hi-g.empirical_cov], marker=mk, ls=ls,
                capsize=2, label=f"Empirical, {lead} s ahead")
zz = np.linspace(0, 3, 200)
ax.plot(zz, norm.cdf(zz), "k-", lw=1.0, alpha=0.7, label="Nominal $\\Phi(z_\\alpha)$")
ax.set_xlabel("Risk parameter $z_\\alpha$"); ax.set_ylabel("Coverage of $T_{safe}$")
ax.set_ylim(0.4, 1.02); ax.legend(loc="lower right", frameon=True)
fig.tight_layout(); fig.savefig(f"{F}/fig_coverage.png", dpi=300); plt.close(fig)

# ---- Fig C: risk-parameter sweep (test set)
t = pd.read_csv(f"{R}/test_summary_all.csv")
p = t[t.policy == "prop"].sort_values("param")
ref = t[(t.policy == "fixed") & np.isclose(t.param, 60.0)].iloc[0]
fig, ax = plt.subplots(2, 1, figsize=(W, 3.7), sharex=True)
ax[0].errorbar(p.param, p.fail_rate*100, yerr=[(p.fail_rate-p.fail_lo)*100, (p.fail_hi-p.fail_rate)*100],
               marker="o", color="#1f4e9c", capsize=2, label="Proposed")
ax[0].axhline(ref.fail_rate*100, color="#d9822b", ls="--", label="Fixed guard 60 s")
ax[0].set_yscale("symlog", linthresh=1.0); ax[0].set_ylim(0, 70)
ax[0].set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50]); ax[0].set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50"])
ax[0].set_ylabel("Failure probability (%)"); ax[0].legend(frameon=True, loc="upper right")
ax[1].errorbar(p.param, p.excess_mean, yerr=p.excess_ci, marker="o", color="#1f4e9c", capsize=2)
ax[1].axhline(ref.excess_mean, color="#d9822b", ls="--")
ax[1].set_ylabel("Mean excess margin (s)"); ax[1].set_xlabel("Risk parameter $z_\\alpha$")
fig.tight_layout(); fig.savefig(f"{F}/fig_risk_sweep.png", dpi=300); plt.close(fig)

# ---- Fig D: ablation
a = pd.read_csv(f"{R}/ablation.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.12})
lab = [v.replace(" term", "").replace("No ", "w/o ") for v in a.variant]
y = np.arange(len(a))
ax[0].barh(y, a.fail_rate*100, xerr=[(a.fail_rate-a.fail_lo)*100, (a.fail_hi-a.fail_rate)*100], color="#1f4e9c", capsize=2)
ax[0].set_yticks(y); ax[0].set_yticklabels(lab); ax[0].invert_yaxis(); ax[0].set_xlabel("Failure (%)")
ax[1].barh(y, a.excess_mean, xerr=a.excess_ci, color="#d9822b", capsize=2)
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].set_xlabel("Excess (s)")
fig.subplots_adjust(left=0.36, right=0.98, bottom=0.2, top=0.97)
fig.savefig(f"{F}/fig_ablation.png", dpi=300); plt.close(fig)
print("Saved figures to", F, ":", sorted(os.listdir(F)))