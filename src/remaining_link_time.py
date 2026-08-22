import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_remaining_link_time(df, usability_col="usable"):
    """
    For each timestep, compute T_link_hat(t): time remaining until the
    link becomes unusable (i.e. until 'usable' flag drops to False for
    the last time in this pass). Only defined within the contiguous
    'usable' window; NaN outside it (not yet connected, or already expired).
    """
    usable_idx = df.index[df[usability_col]].tolist()
    if not usable_idx:
        df["remaining_link_time_s"] = np.nan
        return df

    start_idx, end_idx = usable_idx[0], usable_idx[-1]
    end_time = df.loc[end_idx, "seconds_from_rise"]

    remaining = pd.Series(np.nan, index=df.index)
    for i in range(start_idx, end_idx + 1):
        remaining.loc[i] = end_time - df.loc[i, "seconds_from_rise"]

    df["remaining_link_time_s"] = remaining
    return df, start_idx, end_idx


if __name__ == "__main__":
    df = pd.read_csv("results/full_pass_with_linkbudget.csv")
    df, start_idx, end_idx = compute_remaining_link_time(df)
    df.to_csv("results/full_pass_with_remaining_time.csv", index=False)

    usable_start_s = df.loc[start_idx, "seconds_from_rise"]
    usable_end_s = df.loc[end_idx, "seconds_from_rise"]
    print(f"Usable-link window: {usable_start_s:.1f}s to {usable_end_s:.1f}s "
          f"(duration {usable_end_s - usable_start_s:.1f}s)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["seconds_from_rise"], df["cnr_db"])
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=0.7, label="CNR_min")
    ax1.set_ylabel("CNR (dB)")
    ax1.set_title("CNR and remaining usable-link time: STARLINK-1017 over Aberdeen")
    ax1.legend()

    ax2.plot(df["seconds_from_rise"], df["remaining_link_time_s"])
    ax2.set_ylabel("Remaining link time (s)")
    ax2.set_xlabel("Seconds from rise (AOS)")

    plt.tight_layout()
    plt.savefig("figures/remaining_link_time.png", dpi=150)
    print("Saved results/full_pass_with_remaining_time.csv and figures/remaining_link_time.png")