import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Sourced link-budget constants (S-band, 2 GHz) ---
# 3GPP TR 38.821 Table 6.1.1.1-1 (Set-2 S-band satellite parameters)
EIRP_DENSITY_DBW_MHZ = 34.0
BANDWIDTH_MHZ = 30.0
EIRP_DBW = EIRP_DENSITY_DBW_MHZ + 10 * np.log10(BANDWIDTH_MHZ)

# 3GPP TR 38.811 Table 4.4-1 (UE equivalent G/T under satellite coverage)
UE_GT_DB = -36.6

# Standard physical constant
K_BOLTZMANN_DBW_K_HZ = -228.6

# Documented assumptions
ATMOSPHERIC_LOSS_DB = 0.5   # clear-sky S-band, small
CARRIER_FREQ_HZ = 2e9

BANDWIDTH_HZ = BANDWIDTH_MHZ * 1e6
BANDWIDTH_DBHZ = 10 * np.log10(BANDWIDTH_HZ)

# Usable-link threshold: matches closest related work (Qureshi et al., IJERT 2026)
CNR_MIN_DB = 0.0


def free_space_path_loss_db(slant_range_km, freq_hz=CARRIER_FREQ_HZ):
    """3GPP TR 38.811 clause 6.6.2 formula: L_FS = 32.45 + 20log10(fc_MHz) + 20log10(d_km)"""
    freq_mhz = freq_hz / 1e6
    return 32.45 + 20 * np.log10(freq_mhz) + 20 * np.log10(slant_range_km)


def cnr_db(slant_range_km, shadow_margin_db=0.0, scintillation_loss_db=0.0):
    """3GPP TR 38.821 Section 6.1.3.1 CNR formula."""
    pl_fs = free_space_path_loss_db(slant_range_km)
    cnr = (EIRP_DBW + UE_GT_DB - K_BOLTZMANN_DBW_K_HZ
           - pl_fs - ATMOSPHERIC_LOSS_DB - shadow_margin_db
           - scintillation_loss_db - BANDWIDTH_DBHZ)
    return cnr


if __name__ == "__main__":
    # Apply link budget to the full pass we generated on Day 3
    df = pd.read_csv("results/full_pass_example.csv")
    df["cnr_db"] = df["slant_range_km"].apply(cnr_db)
    df["usable"] = df["cnr_db"] >= CNR_MIN_DB

    df.to_csv("results/full_pass_with_linkbudget.csv", index=False)

    print(f"EIRP: {EIRP_DBW:.1f} dBW")
    print(f"CNR range across pass: {df['cnr_db'].min():.2f} to {df['cnr_db'].max():.2f} dB")
    print(f"Usable-link threshold: {CNR_MIN_DB} dB")
    print(f"Fraction of pass usable: {df['usable'].mean()*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["seconds_from_rise"], df["elevation_deg"])
    ax1.set_ylabel("Elevation (deg)")
    ax1.set_title("Full pass with link budget: STARLINK-1017 over Aberdeen")

    ax2.plot(df["seconds_from_rise"], df["cnr_db"])
    ax2.axhline(CNR_MIN_DB, color="red", linestyle="--", label=f"CNR_min = {CNR_MIN_DB} dB")
    ax2.set_ylabel("CNR (dB)")
    ax2.set_xlabel("Seconds from rise (AOS)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("figures/full_pass_with_linkbudget.png", dpi=150)
    print("Saved results/full_pass_with_linkbudget.csv and figures/full_pass_with_linkbudget.png")