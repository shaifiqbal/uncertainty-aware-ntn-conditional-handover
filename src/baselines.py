import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

T_MEAS_S = 0.5
T_RRC_S = 2.0
T_EXEC_S = 0.1
T_MARGIN_S = 0.5
TOTAL_OVERHEAD_S = T_MEAS_S + T_RRC_S + T_EXEC_S + T_MARGIN_S

Z_ALPHA_DEFAULT = 1.645


def _descending_half(df):
    peak_idx = df["cnr_db"].idxmax()
    return df.loc[peak_idx:]


def baseline_fixed_guard(df, t_fixed_s, label=None):
    triggered = df[df["remaining_link_time_s"] <= t_fixed_s]
    if triggered.empty:
        return None
    row = triggered.iloc[0]
    return {
        "policy": label or f"Fixed guard-time ({t_fixed_s}s)",
        "trigger_time_s": row["seconds_from_rise"],
        "remaining_link_time_at_trigger_s": row["remaining_link_time_s"],
        "cnr_at_trigger_db": row["cnr_db"],
    }


def baseline_quality_threshold(df, cnr_trigger_db):
    descending = _descending_half(df)
    triggered = descending[descending["cnr_db"] <= cnr_trigger_db]
    if triggered.empty:
        return None
    row = triggered.iloc[0]
    return {
        "policy": f"Quality threshold ({cnr_trigger_db} dB)",
        "trigger_time_s": row["seconds_from_rise"],
        "remaining_link_time_at_trigger_s": row["remaining_link_time_s"],
        "cnr_at_trigger_db": row["cnr_db"],
    }


def proposed_policy_trigger(df, z_alpha=Z_ALPHA_DEFAULT, total_overhead_s=TOTAL_OVERHEAD_S):
    descending = _descending_half(df).dropna(subset=["T_safe_s"])
    triggered = descending[descending["T_safe_s"] <= total_overhead_s]
    if triggered.empty:
        return None
    row = triggered.iloc[0]
    return {
        "policy": f"Proposed (uncertainty-aware, z_alpha={z_alpha})",
        "trigger_time_s": row["seconds_from_rise"],
        "remaining_link_time_at_trigger_s": row["remaining_link_time_s"],
        "cnr_at_trigger_db": row["cnr_db"],
        "sigma_T_s": row["sigma_T_s"],
    }


def evaluate_trigger(result, total_overhead_s=TOTAL_OVERHEAD_S):
    if result is None:
        return "NO TRIGGER (guard/threshold never reached in this pass)"
    margin = result["remaining_link_time_at_trigger_s"] - total_overhead_s
    result["execution_overhead_s"] = total_overhead_s
    result["margin_after_execution_s"] = margin
    result["status"] = "SUCCESS (executes before expiry)" if margin >= 0 else "LATE (handover would fail)"
    return result


if __name__ == "__main__":
    df = pd.read_csv("results/full_pass_with_uncertainty.csv")

    results = []
    # Baseline 1: fixed guard-time (low, medium, conservative-ish)
    for t_fixed in [5, 15, 30]:
        results.append(evaluate_trigger(baseline_fixed_guard(df, t_fixed)))

    # Baseline 2: point-prediction (no margin at all - trigger exactly at overhead)
    results.append(evaluate_trigger(
        baseline_fixed_guard(df, TOTAL_OVERHEAD_S, label="Point-prediction (no margin)")))

    # Baseline 3: quality threshold
    results.append(evaluate_trigger(baseline_quality_threshold(df, cnr_trigger_db=3.0)))

    # Baseline 4: conservative fixed-margin (intentionally large, e.g. 60s)
    results.append(evaluate_trigger(baseline_fixed_guard(df, 60, label="Conservative fixed-margin (60s)")))

    # Proposed: uncertainty-aware
    results.append(evaluate_trigger(proposed_policy_trigger(df)))

    for r in results:
        print(r)
        print("---")

    results_df = pd.DataFrame([r for r in results if isinstance(r, dict)])
    results_df.to_csv("results/all_policies_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["seconds_from_rise"], df["cnr_db"], label="CNR(t)", color="black")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.7, label="CNR_min (outage)")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    for i, r in enumerate([x for x in results if isinstance(x, dict)]):
        ax.axvline(r["trigger_time_s"], color=colors[i % len(colors)], linestyle=":", label=r["policy"])

    ax.set_xlabel("Seconds from rise (AOS)")
    ax.set_ylabel("CNR (dB)")
    ax.set_title("All policy trigger points: STARLINK-1017 over Aberdeen")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/all_policies_triggers.png", dpi=150)
    print("Saved results/all_policies_comparison.csv and figures/all_policies_triggers.png")