"""
Pure NumPy/SciPy mathematical and physical engine for Module 6 (Danam Systems Navigation R&D SW Specialization):
- RAIM (Receiver Autonomous Integrity Monitoring) Chi-Square WSSE Test & FDE (Fault Detection and Exclusion)
- Pull-off Spoofing Correlation Peak Hijacking & SQM (Signal Quality Monitoring) Asymmetry Detection
- EGI (Embedded GPS/INS) Lever-Arm Offset & Time-Tag Latency Error Analysis under High-G Maneuvers
- STAP (Space-Time Adaptive Processing) vs SAP Broadband Multipath Jamming Suppression
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import chi2

from core.coords import ecef_to_enu, llh_to_ecef
from core.pvt_dop import generate_constellation_at_reference, solve_gauss_newton_wls


@dataclass(frozen=True)
class EgiRaimSpoofResult:
    """Comprehensive output container for Module 6 (EGI / RAIM / Anti-Spoofing / Lever-Arm / STAP)."""

    # 1. RAIM & FDE
    sat_prns: List[int]
    fault_prn: int
    fault_bias_m: float
    normalized_residuals: np.ndarray
    raim_wsse_stat: float
    raim_chi2_threshold: float
    raim_Gaussian_5sigma_thresh: float
    raim_alarm_triggered: bool
    detected_fault_prn: int
    pos_err_before_fde_m: float
    pos_err_after_fde_m: float

    # 2. Pull-Off Spoofing & SQM
    tau_chips: np.ndarray
    corr_authentic: np.ndarray
    corr_spoofer: np.ndarray
    corr_composite: np.ndarray
    sqm_delta_metric: float
    sqm_threshold: float
    spoof_alarm_triggered: bool

    # 3. 3D Pull-Off Spoofing Time-Evolution Surface R(tau, t)
    pulloff_time_s: np.ndarray
    pulloff_tau_chips: np.ndarray
    pulloff_surface: np.ndarray  # Shape (len(t), len(tau))
    sqm_history: np.ndarray

    # 4. EGI Lever-Arm & Time-Tag Latency Error + STAP Broadband Suppression
    maneuver_t_s: np.ndarray
    uncomp_lever_error_m: np.ndarray
    uncomp_latency_error_m: np.ndarray
    uncomp_total_error_m: np.ndarray
    comp_residual_error_m: np.ndarray
    freq_offset_mhz: np.ndarray
    sap_broadband_null_db: np.ndarray
    stap_broadband_null_db: np.ndarray


def _tri_corr(tau: np.ndarray) -> np.ndarray:
    """Ideal BPSK C/A correlation triangle envelope."""
    return np.maximum(0.0, 1.0 - np.abs(tau))


def run_egi_raim_spoof_simulation(
    fault_prn_bias_m: float = 45.0,
    spoofer_delay_chips: float = 0.45,
    spoofer_power_ratio_db: float = 4.0,
    lever_arm_m: float = 1.2,
    latency_ms: float = 40.0,
    stap_taps: int = 5,
    seed: int = 606,
) -> EgiRaimSpoofResult:
    """
    Execute the complete Danam Navigation R&D specialization simulation pipeline:
    1. RAIM Chi-Square WSSE & Normalized Residual FDE (Fault Detection & Exclusion)
    2. Pull-off Spoofing Composite Correlation Distortion & SQM Ratio/Delta Test
    3. 3D Pull-off Spoofing Time-Evolution Surface R(tau, t)
    4. High-G Missile/UAV EGI Lever-Arm & Time-Tag Latency Compensation + STAP vs SAP
    """
    rng = np.random.default_rng(seed)
    ref_lat, ref_lon, ref_alt = 37.5665, 126.9780, 100.0
    true_ecef = llh_to_ecef(ref_lat, ref_lon, ref_alt)
    true_clock_m = 5000.0

    # --------------------------------------------------------------------------
    # 1. RAIM & FDE (Fault Detection and Exclusion) on 9 visible satellites
    # --------------------------------------------------------------------------
    sats = [s for s in generate_constellation_at_reference(ref_lat, ref_lon, ref_alt, mask_angle_deg=12.0) if s.is_visible]
    n_sats = len(sats)
    sat_prns = [s.prn for s in sats]
    sat_ecef_mat = np.vstack([s.ecef_m for s in sats])
    true_ranges = np.array([s.true_range_m for s in sats], dtype=np.float64)

    sigma_uere = 1.5
    meas_noise = rng.normal(0.0, sigma_uere, size=n_sats)
    rho_clean = true_ranges + true_clock_m + meas_noise

    # Inject fault/spoofing bias on satellite index 3 (PRN #12)
    fault_idx = min(3, n_sats - 1)
    fault_prn = sat_prns[fault_idx]
    rho_faulty = rho_clean.copy()
    rho_faulty[fault_idx] += float(fault_prn_bias_m)

    weights = np.ones(n_sats, dtype=np.float64) / (sigma_uere ** 2)
    w_mat = np.diag(weights)

    wls_faulty = solve_gauss_newton_wls(
        sat_ecef_matrix=sat_ecef_mat,
        pseudoranges_m=rho_faulty,
        weights=weights,
        init_ecef_m=true_ecef + np.array([200.0, -200.0, 100.0]),
        true_ecef_m=true_ecef,
    )
    est_ecef_before = wls_faulty["est_ecef_m"]
    est_b_before = float(wls_faulty["est_clock_bias_m"])
    h_mat = wls_faulty["H_ecef"]

    pos_err_before = float(np.linalg.norm(ecef_to_enu(est_ecef_before, ref_lat, ref_lon, ref_alt)))

    # Post-fit pseudorange residual vector r_hat = rho_meas - rho_pred
    pred_ranges = np.linalg.norm(sat_ecef_mat - est_ecef_before[None, :], axis=1) + est_b_before
    residuals = rho_faulty - pred_ranges

    # Parity / Projection Matrix S = I - H (H^T W H)^-1 H^T W
    ht_w_h_inv = np.linalg.inv(h_mat.T @ w_mat @ h_mat + 1e-10 * np.eye(4))
    p_proj = h_mat @ ht_w_h_inv @ h_mat.T @ w_mat
    s_proj = np.eye(n_sats) - p_proj

    # Weighted Sum of Squared Errors (WSSE) ~ Chi-Square(df = N - 4)
    wsse_stat = float(residuals.T @ w_mat @ residuals)
    dof_raim = max(n_sats - 4, 1)
    # False alarm probability P_fa = 1e-4 -> Chi-square threshold
    chi2_thresh = float(chi2.ppf(1.0 - 1e-4, df=dof_raim))
    raim_alarm = bool(wsse_stat > chi2_thresh)

    # Normalized residual per satellite: w_i = |r_i| / (sigma_i * sqrt(S_ii))
    s_diag = np.maximum(np.diag(s_proj), 1e-6)
    norm_res = np.abs(residuals) / (sigma_uere * np.sqrt(s_diag))
    detected_idx = int(np.argmax(norm_res))
    detected_prn = sat_prns[detected_idx]
    norm_res_thresh = 3.29  # ~99.9% Gaussian threshold (3.29 sigma)

    # FDE: Exclude detected satellite if RAIM alarm triggered and re-solve WLS
    if raim_alarm and n_sats >= 5:
        keep_mask = np.ones(n_sats, dtype=bool)
        keep_mask[detected_idx] = False
        wls_after = solve_gauss_newton_wls(
            sat_ecef_matrix=sat_ecef_mat[keep_mask],
            pseudoranges_m=rho_faulty[keep_mask],
            weights=weights[keep_mask],
            init_ecef_m=true_ecef,
            true_ecef_m=true_ecef,
        )
        pos_err_after = float(np.linalg.norm(ecef_to_enu(wls_after["est_ecef_m"], ref_lat, ref_lon, ref_alt)))
    else:
        pos_err_after = pos_err_before

    # --------------------------------------------------------------------------
    # 2. Pull-Off Spoofing Correlation Peak Distortion & SQM Asymmetry Metric
    # --------------------------------------------------------------------------
    tau_chips = np.linspace(-1.8, 2.2, 401)
    sp_amp = 10.0 ** (float(spoofer_power_ratio_db) / 20.0) if spoofer_power_ratio_db > 0.0 else 0.0
    corr_auth = _tri_corr(tau_chips)
    corr_spoof = sp_amp * _tri_corr(tau_chips - spoofer_delay_chips)
    corr_comp = corr_auth + corr_spoof

    # Receiver prompt tracker tracks the dominant peak of the composite correlation function
    track_idx = int(np.argmax(corr_comp))
    tau_prompt = float(tau_chips[track_idx])
    d_half = 0.25  # d = 0.5 chip E-L spacing
    r_p_val = float(np.interp(tau_prompt, tau_chips, corr_comp))
    r_e_val = float(np.interp(tau_prompt - d_half, tau_chips, corr_comp))
    r_l_val = float(np.interp(tau_prompt + d_half, tau_chips, corr_comp))

    # SQM Asymmetry Delta Metric: |R_E - R_L| / (2 * R_P) + excess power ratio check
    sqm_delta = abs(r_e_val - r_l_val) / max(2.0 * r_p_val, 1e-6)
    sqm_thresh = 0.045
    spoof_alarm = bool(sqm_delta >= sqm_thresh)

    # --------------------------------------------------------------------------
    # 3. 3D Pull-Off Spoofing Time-Evolution Surface R(tau, t) over t in [0..10s]
    # --------------------------------------------------------------------------
    t_pull = np.linspace(0.0, 10.0, 45)
    tau_pull = np.linspace(-1.4, 2.0, 85)
    surf_pull = np.empty((len(t_pull), len(tau_pull)), dtype=np.float64)
    sqm_hist = np.empty(len(t_pull), dtype=np.float64)

    for idx_t, t_val in enumerate(t_pull):
        # Spoofer stays aligned until t=1.5s, then linearly pulls off at 0.14 chips/sec
        cur_delay = 0.0 if t_val < 1.5 else 0.15 * (t_val - 1.5)
        c_a = _tri_corr(tau_pull)
        c_s = sp_amp * _tri_corr(tau_pull - cur_delay)
        c_tot = c_a + c_s
        surf_pull[idx_t, :] = c_tot

        pk_tau = float(tau_pull[int(np.argmax(c_tot))])
        rp = float(np.interp(pk_tau, tau_pull, c_tot))
        re = float(np.interp(pk_tau - d_half, tau_pull, c_tot))
        rl = float(np.interp(pk_tau + d_half, tau_pull, c_tot))
        sqm_hist[idx_t] = abs(re - rl) / max(2.0 * rp, 1e-6)

    # --------------------------------------------------------------------------
    # 4. EGI Lever-Arm & Time-Tag Latency under High-G Flight Maneuver + STAP
    # --------------------------------------------------------------------------
    maneuver_t = np.linspace(0.0, 10.0, 250)
    flight_speed_mps = 220.0  # High-speed tactical UAV / missile (~Mach 0.65)
    # Dynamic S-turn yaw rate (up to 45 deg/s) and pitch oscillation
    yaw_rate_rad_s = np.deg2rad(45.0) * np.sin(2.0 * np.pi * 0.25 * maneuver_t)
    yaw_angle_rad = np.cumsum(yaw_rate_rad_s) * (maneuver_t[1] - maneuver_t[0])

    # Uncompensated Lever-Arm position error magnitude when attitude rotates: ||C_b^n l^b - l^b_init||
    lever_err_m = float(lever_arm_m) * np.abs(2.0 * np.sin(0.5 * yaw_angle_rad)) + float(lever_arm_m) * 0.3 * np.abs(yaw_rate_rad_s)
    # Uncompensated Time-Tag Latency error during acceleration & turn: ||v * dt_lat + 0.5 * a * dt_lat^2||
    dt_lat_s = float(latency_ms) * 1e-3
    lat_accel_mps2 = flight_speed_mps * np.abs(yaw_rate_rad_s)
    latency_err_m = (0.35 * flight_speed_mps + lat_accel_mps2) * dt_lat_s

    total_uncomp_m = np.hypot(lever_err_m, latency_err_m)
    # With EGI SW Lever-Arm compensation (C_b^n * l^b) + IMU Ring-Buffer 1PPS Time-Tag matching:
    comp_res_m = 0.08 + 0.04 * np.abs(np.sin(2.0 * np.pi * 0.5 * maneuver_t))

    # Broadband Jammer Null Depth across [-10 MHz, +10 MHz] around L1 (1575.42 MHz)
    # Compare Spatial-Only CRPA (1 tap) vs L-tap STAP
    freq_off_mhz = np.linspace(-10.0, 10.0, 121)
    frac_bw = np.abs(freq_off_mhz) / 1575.42
    # Spatial-only null depth degrades proportional to 20*log10(pi * (d/lambda) * sin(theta_j) * df/f0)
    sap_null_db = np.minimum(-12.0, 20.0 * np.log10(np.maximum(1.8 * frac_bw, 1e-3)))
    # L-tap STAP adds temporal degrees of freedom, suppressing wideband dispersion by L-order
    stap_gain_factor = max(int(stap_taps), 1)
    stap_null_db = np.maximum(-62.0, sap_null_db - 9.5 * (stap_gain_factor - 1) * (1.0 - 0.3 * (freq_off_mhz / 10.0) ** 2))

    return EgiRaimSpoofResult(
        sat_prns=sat_prns,
        fault_prn=fault_prn,
        fault_bias_m=float(fault_prn_bias_m),
        normalized_residuals=norm_res,
        raim_wsse_stat=wsse_stat,
        raim_chi2_threshold=chi2_thresh,
        raim_Gaussian_5sigma_thresh=norm_res_thresh,
        raim_alarm_triggered=raim_alarm,
        detected_fault_prn=detected_prn,
        pos_err_before_fde_m=pos_err_before,
        pos_err_after_fde_m=pos_err_after,
        tau_chips=tau_chips,
        corr_authentic=corr_auth,
        corr_spoofer=corr_spoof,
        corr_composite=corr_comp,
        sqm_delta_metric=sqm_delta,
        sqm_threshold=sqm_thresh,
        spoof_alarm_triggered=spoof_alarm,
        pulloff_time_s=t_pull,
        pulloff_tau_chips=tau_pull,
        pulloff_surface=surf_pull,
        sqm_history=sqm_hist,
        maneuver_t_s=maneuver_t,
        uncomp_lever_error_m=lever_err_m,
        uncomp_latency_error_m=latency_err_m,
        uncomp_total_error_m=total_uncomp_m,
        comp_residual_error_m=comp_res_m,
        freq_offset_mhz=freq_off_mhz,
        sap_broadband_null_db=sap_null_db,
        stap_broadband_null_db=stap_null_db,
    )
