import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Handover execution overhead (documented assumption - refine with real values later)
T_MEAS_S = 0.5      # measurement reporting delay
T_RRC_S = 2.0        # control-plane (RRC) delay - CHOMET paper notes CHO gaps up to ~9-10s, using conservative mid value
T_EXEC_S = 0.1        # execution delay
T_MARGIN_S = 0.5      # extra safety margin
TOTAL_OVERHEAD_S = T_MEAS_S + T_RRC_S + T_EXEC_S + T_MARGIN_S


def baseline_fixed_guard(df, t_fixed_s):
    """
    Baseline 1: trigger handover at the first timestep where the NOMINAL
    (point-estimate) remaining link time drops to or below a fixed guard time.
    """
    triggered = df[df["remaining_link_time_s"] <= t_fixed_s]
    if triggered.empty:
        return None  # never triggers within this pass (guard time longer than window)
    trigger_row = triggered.iloc[0]
    return {
        "policy": f"Fixed guard-time ({t_fixed_s}s)",
        "trigger_time_s": trigger_row["seconds_from_rise"],
        "remaining_link_time_at_trigger_s": trigger_row["remaining_link_time_s"],
        "cnr_at_trigger_db": trigger_row["cnr_db"],
    }


def baseline_quality_threshold(df, cnr_trigger_db):
    """
    Baseline 3: trigger handover when instantaneous CNR crosses below a
    quality threshold on the DESCENDING side of the pass only (link degrading
    toward expiry), not during initial acquisition while CNR is still rising.
    """
    peak_idx = df["cnr_db"].idxmax()
    descending = df.loc[peak_idx:]  # only consider post-peak (degrading link)
    triggered = descending[descending["cnr_db"] <= cnr_trigger_db]
    if triggered.empty:
        return None
    trigger_row = triggered.iloc[0]
    return {
        "policy": f"Quality threshold ({cnr_trigger_db} dB)",
        "trigger_time_s": trigger_row["seconds_from_rise"],
        "remaining_link_time_at_trigger_s": trigger_row["remaining_link_time_s"],
        "cnr_at_trigger_db": trigger_row["cnr_db"],
    }


def evaluate_trigger(result, total_overhead_s=TOTAL_OVERHEAD_S):
    """
    Checks whether the trigger leaves enough time to complete handover
    execution before the link actually expires (success), or not (late/failed).
    """
    if result is None:
        return "NO TRIGGER (guard/threshold never reached in this pass)"
    margin = result["remaining_link_time_at_trigger_s"] - total_overhead_s
    status = "SUCCESS (executes before expiry)" if margin >= 0 else "LATE (handover would fail)"
    result["execution_overhead_s"] = total_overhead_s
    result["margin_after_execution_s"] = margin
    result["status"] = status
    return result


if __name__ == "__main__":
    df = pd.read_csv("results/full_pass_with_remaining_time.csv")

    results = []
    # Baseline 1: three fixed guard-time values (low, medium, conservative)
    for t_fixed in [5, 15, 30]:
        r = baseline_fixed_guard(df, t_fixed)
        results.append(evaluate_trigger(r))

    # Baseline 3: quality threshold set 3 dB above the 0 dB outage floor
    r = baseline_quality_threshold(df, cnr_trigger_db=3.0)
    results.append(evaluate_trigger(r))

    for r in results:
        print(r)
        print("---")

    results_df = pd.DataFrame([r for r in results if isinstance(r, dict)])
    results_df.to_csv("results/baseline_trigger_comparison.csv", index=False)

    # Plot trigger points on the pass profile
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["seconds_from_rise"], df["cnr_db"], label="CNR(t)", color="black")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.7, label="CNR_min (outage)")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for i, r in enumerate([x for x in results if isinstance(x, dict)]):
        ax.axvline(r["trigger_time_s"], color=colors[i % len(colors)],
                   linestyle=":", label=r["policy"])

    ax.set_xlabel("Seconds from rise (AOS)")
    ax.set_ylabel("CNR (dB)")
    ax.set_title("Baseline trigger points: STARLINK-1017 over Aberdeen")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/baseline_triggers.png", dpi=150)
    print("Saved results/baseline_trigger_comparison.csv and figures/baseline_triggers.png")