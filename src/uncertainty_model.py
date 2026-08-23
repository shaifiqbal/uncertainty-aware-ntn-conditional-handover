import pandas as pd
import numpy as np
SIGMA_T_CAP_S = 60.0  # ceiling on time-uncertainty estimate — beyond this, the local-linearization
# conversion is no longer physically meaningful; documented assumption, same rationale as the
# MIN_DCNR_DT_DB_S floor (both guard against the same slope-based instability)
# Documented uncertainty parameter ranges (sensitivity-analysis basis,
# consistent with noise levels used in the closest related work - Qureshi et al., IJERT 2026)
SIGMA_ELEV_DEG = 1.0      # elevation-angle estimation error (deg)
SIGMA_MARGIN_DB = 1.5     # link-margin/SNR estimation error (dB)
SIGMA_MEAS_S = 0.2        # measurement-report delay uncertainty (s)
SIGMA_DELAY_S = 1.0       # control-plane + execution delay uncertainty (s)

Z_ALPHA_DEFAULT = 1.645   # ~95% one-sided confidence bound (risk parameter - swept in Experiment 3)


MIN_DCNR_DT_DB_S = 0.01  # below this slope, local-linearization is invalid - exclude, don't extrapolate

def compute_uncertainty_sigma_T(df):
    dt = df["seconds_from_rise"].diff()
    dCNR = df["cnr_db"].diff()
    dElev = df["elevation_deg"].diff()

    dCNR_dt = (dCNR / dt).replace([np.inf, -np.inf], np.nan)
    dCNR_delev = (dCNR / dElev).replace([np.inf, -np.inf], np.nan)

    # Exclude near-zero-slope points (e.g. near culmination) - the dB-to-time
    # conversion via local linearization is invalid there, not just "large"
    dCNR_dt_safe = dCNR_dt.where(dCNR_dt.abs() >= MIN_DCNR_DT_DB_S)

    sigma_cnr_from_geom = (dCNR_delev.abs() * SIGMA_ELEV_DEG)
    sigma_cnr_total = np.sqrt(sigma_cnr_from_geom**2 + SIGMA_MARGIN_DB**2)

    sigma_T_predict = (sigma_cnr_total / dCNR_dt_safe.abs())
    sigma_T = np.sqrt(sigma_T_predict**2 + SIGMA_MEAS_S**2 + SIGMA_DELAY_S**2)

    df["sigma_T_s"] = sigma_T
    return df


def compute_safe_time(df, z_alpha=Z_ALPHA_DEFAULT):
    df["T_safe_s"] = df["remaining_link_time_s"] - z_alpha * df["sigma_T_s"]
    return df


if __name__ == "__main__":
    df = pd.read_csv("results/full_pass_with_remaining_time.csv")
    df = compute_uncertainty_sigma_T(df)
    df = compute_safe_time(df)
    df.to_csv("results/full_pass_with_uncertainty.csv", index=False)
    print(df[["seconds_from_rise", "cnr_db", "remaining_link_time_s",
               "sigma_T_s", "T_safe_s"]].dropna().head(15))
    print("...")
    print(df[["seconds_from_rise", "cnr_db", "remaining_link_time_s",
               "sigma_T_s", "T_safe_s"]].dropna().tail(15))