"""
Pure NumPy/SciPy mathematical and array signal processing engine for Module 4:
- N-Element Uniform Linear Array (ULA) Steering Vector & Array Manifold
- Multi-Jammer + Satellite + Thermal Noise Spatial Covariance Matrix (R_xx) & Diagonal Loading
- Power Minimization (PI / Power Inversion) Adaptive Nulling
- MVDR (Minimum Variance Distortionless Response / Capon) Beamformer
- Conventional Delay-and-Sum (DAS) Beamformer
- Eigenvalue Subspace Analysis, Output SINR, Null Depth, and Effective Post-Beamforming C/N0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class CrpaSimulationResult:
    """Comprehensive output container for CRPA Anti-Jamming Array Signal Processing."""

    angles_deg: np.ndarray
    pattern_das_db: np.ndarray
    pattern_pm_db: np.ndarray
    pattern_mvdr_db: np.ndarray
    weights_das: np.ndarray
    weights_pm: np.ndarray
    weights_mvdr: np.ndarray
    eigenvalues_db: np.ndarray
    sat_angle_deg: float
    jammer_angles_deg: List[float]
    jammer_jsr_db: List[float]
    null_depths_pm_db: List[float]
    null_depths_mvdr_db: List[float]
    sat_gain_pm_db: float
    sat_gain_mvdr_db: float
    sinr_single_db: float
    sinr_pm_db: float
    sinr_mvdr_db: float
    eff_cn0_single_dbhz: float
    eff_cn0_pm_dbhz: float
    eff_cn0_mvdr_dbhz: float
    jsr_sweep_db: np.ndarray
    cn0_sweep_single_dbhz: np.ndarray
    cn0_sweep_pm_dbhz: np.ndarray
    cn0_sweep_mvdr_dbhz: np.ndarray


def steering_vector_ula(
    angle_deg: float | np.ndarray,
    num_elements: int = 4,
    d_lambda: float = 0.5,
) -> np.ndarray:
    """
    Compute the Uniform Linear Array (ULA) steering vector a(theta) for incident angle(s)
    `angle_deg` measured from array boresight (-90° to +90°).

    Parameters
    ----------
    angle_deg : float or np.ndarray
        Incident angle(s) in degrees. Scalar returns shape (N,), 1D array of length M returns (N, M).
    num_elements : int
        Number of antenna elements N >= 2.
    d_lambda : float
        Inter-element spacing normalized by carrier wavelength (d / lambda), typically 0.5.
    """
    num_elements = max(int(num_elements), 2)
    d_lambda = float(np.clip(d_lambda, 0.05, 2.0))
    theta_rad = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    elem_idx = np.arange(num_elements, dtype=np.float64)

    if theta_rad.ndim == 0:
        phase = 2.0 * np.pi * d_lambda * elem_idx * np.sin(theta_rad)
        return np.exp(1j * phase)

    phase = 2.0 * np.pi * d_lambda * elem_idx[:, None] * np.sin(theta_rad[None, :])
    return np.exp(1j * phase)


def compute_spatial_covariance(
    num_elements: int,
    d_lambda: float,
    sat_angle_deg: float,
    jammer_angles_deg: List[float],
    jammer_jsr_db: List[float],
    snr_pre_db: float = -20.0,
    diagonal_loading_db: float = -10.0,
    num_snapshots: int = 256,
    seed: int = 404,
) -> Dict[str, np.ndarray]:
    """
    Construct both the theoretical and finite-snapshot sample spatial covariance matrices
    R_xx (N x N Hermitian positive-definite) for a satellite signal (submerged below thermal noise)
    and multiple high-power continuous/broadband jammers.

    Reference power normalization:
    - Thermal noise power per element: sigma_n^2 = 1.0 (0 dB)
    - Satellite pre-correlation signal power: sigma_s^2 = 10^(snr_pre_db / 10) (~0.01 for -20 dB)
    - Jammer m power: sigma_jm^2 = sigma_s^2 * 10^(JSR_m_dB / 10)
    """
    rng = np.random.default_rng(seed)
    num_elements = max(int(num_elements), 2)
    num_snapshots = max(int(num_snapshots), 16)

    sigma_n2 = 1.0
    sigma_s2 = 10.0 ** (float(snr_pre_db) / 10.0)
    diag_load_lin = sigma_n2 * (10.0 ** (float(diagonal_loading_db) / 10.0))

    a_sat = steering_vector_ula(sat_angle_deg, num_elements, d_lambda)
    r_signal = sigma_s2 * np.outer(a_sat, np.conj(a_sat))

    r_jam_plus_noise = sigma_n2 * np.eye(num_elements, dtype=np.complex128)

    # Generate finite-sample snapshot matrix X (N x K)
    noise_snapshots = (
        rng.normal(0.0, np.sqrt(sigma_n2 / 2.0), size=(num_elements, num_snapshots))
        + 1j * rng.normal(0.0, np.sqrt(sigma_n2 / 2.0), size=(num_elements, num_snapshots))
    )
    sat_wave = np.sqrt(sigma_s2 / 2.0) * (
        rng.normal(0.0, 1.0, size=num_snapshots) + 1j * rng.normal(0.0, 1.0, size=num_snapshots)
    )
    x_snapshots = a_sat[:, None] * sat_wave[None, :] + noise_snapshots

    for j_ang, jsr_db in zip(jammer_angles_deg, jammer_jsr_db):
        sigma_j2 = sigma_s2 * (10.0 ** (float(jsr_db) / 10.0))
        a_jam = steering_vector_ula(j_ang, num_elements, d_lambda)
        r_jam_plus_noise += sigma_j2 * np.outer(a_jam, np.conj(a_jam))

        jam_wave = np.sqrt(sigma_j2 / 2.0) * (
            rng.normal(0.0, 1.0, size=num_snapshots) + 1j * rng.normal(0.0, 1.0, size=num_snapshots)
        )
        x_snapshots += a_jam[:, None] * jam_wave[None, :]

    r_theo = r_signal + r_jam_plus_noise
    r_sample = (x_snapshots @ np.conj(x_snapshots.T)) / float(num_snapshots)
    # Enforce exact Hermitian symmetry and apply diagonal loading
    r_sample_loaded = 0.5 * (r_sample + np.conj(r_sample.T)) + diag_load_lin * np.eye(num_elements)

    return {
        "R_sample_loaded": r_sample_loaded,
        "R_theo": r_theo,
        "R_signal": r_signal,
        "R_jam_plus_noise": r_jam_plus_noise,
        "a_sat": a_sat,
    }


def compute_adaptive_weights(
    r_xx: np.ndarray,
    a_sat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute complex weight vectors w (N,) for:
    1. Conventional Delay-and-Sum (DAS): w_DAS = a(theta_s) / N
    2. Power Minimization (PI / Power Inversion): w_PM = (R_xx^-1 * delta_1) / (delta_1^H * R_xx^-1 * delta_1)
    3. MVDR (Capon Distortionless Beamformer): w_MVDR = (R_xx^-1 * a_s) / (a_s^H * R_xx^-1 * a_s)
    """
    n_elem = r_xx.shape[0]
    w_das = a_sat / float(n_elem)

    delta_1 = np.zeros(n_elem, dtype=np.complex128)
    delta_1[0] = 1.0 + 0.0j

    try:
        r_inv_delta = np.linalg.solve(r_xx, delta_1)
        r_inv_asat = np.linalg.solve(r_xx, a_sat)
    except np.linalg.LinAlgError:
        r_pinv = np.linalg.pinv(r_xx)
        r_inv_delta = r_pinv @ delta_1
        r_inv_asat = r_pinv @ a_sat

    denom_pm = np.vdot(delta_1, r_inv_delta)
    if abs(denom_pm) < 1e-15:
        denom_pm = 1e-15
    w_pm = r_inv_delta / denom_pm

    denom_mvdr = np.vdot(a_sat, r_inv_asat)
    if abs(denom_mvdr) < 1e-15:
        denom_mvdr = 1e-15
    w_mvdr = r_inv_asat / denom_mvdr

    return {
        "w_das": w_das,
        "w_pm": w_pm,
        "w_mvdr": w_mvdr,
    }


def evaluate_sinr_db(w: np.ndarray, r_signal: np.ndarray, r_jam_plus_noise: np.ndarray) -> float:
    """Compute output Signal-to-Interference-plus-Noise Ratio (SINR) in dB for weight vector `w`."""
    p_sig = float(np.real(np.vdot(w, r_signal @ w)))
    p_in = float(np.real(np.vdot(w, r_jam_plus_noise @ w)))
    ratio = max(p_sig, 1e-20) / max(p_in, 1e-20)
    return float(10.0 * np.log10(ratio))


def run_crpa_simulation(
    num_elements: int = 4,
    d_lambda: float = 0.5,
    sat_angle_deg: float = 15.0,
    jam1_angle_deg: float = -40.0,
    jam1_jsr_db: float = 60.0,
    jam2_enabled: bool = True,
    jam2_angle_deg: float = 50.0,
    jam2_jsr_db: float = 55.0,
    diagonal_loading_db: float = -10.0,
    nominal_cn0_dbhz: float = 46.0,
) -> CrpaSimulationResult:
    """
    Execute full CRPA Anti-Jamming simulation:
    - Array patterns across [-90°, +90°] for DAS, Power-Min, and MVDR
    - Eigenvalue spectrum of spatial covariance matrix R_xx
    - Null depths at each jammer angle and output SINR / Effective C/N0
    - J/S sweep (30 to 90 dB) comparing Single Antenna vs Power-Min vs MVDR
    """
    num_elements = int(np.clip(num_elements, 2, 16))
    d_lambda = float(np.clip(d_lambda, 0.2, 1.0))

    jam_angles: List[float] = [float(jam1_angle_deg)]
    jam_jsrs: List[float] = [float(jam1_jsr_db)]
    if jam2_enabled:
        jam_angles.append(float(jam2_angle_deg))
        jam_jsrs.append(float(jam2_jsr_db))

    # Pre-correlation SNR in ~2 MHz bandwidth for C/N0 = nominal_cn0_dbhz (46 dB-Hz - 63 dB = -17 dB)
    bw_db = 10.0 * np.log10(2.046e6)  # ~63.11 dB
    snr_pre_db = nominal_cn0_dbhz - bw_db

    cov_data = compute_spatial_covariance(
        num_elements=num_elements,
        d_lambda=d_lambda,
        sat_angle_deg=sat_angle_deg,
        jammer_angles_deg=jam_angles,
        jammer_jsr_db=jam_jsrs,
        snr_pre_db=snr_pre_db,
        diagonal_loading_db=diagonal_loading_db,
    )

    r_xx = cov_data["R_sample_loaded"]
    r_sig = cov_data["R_signal"]
    r_jn = cov_data["R_jam_plus_noise"]
    a_sat = cov_data["a_sat"]

    weights = compute_adaptive_weights(r_xx, a_sat)
    w_das = weights["w_das"]
    w_pm = weights["w_pm"]
    w_mvdr = weights["w_mvdr"]

    # Evaluate radiation patterns over [-90, +90] degrees (721 points, 0.25 deg resolution)
    angles_deg = np.linspace(-90.0, 90.0, 721)
    a_grid = steering_vector_ula(angles_deg, num_elements, d_lambda)  # (N, 721)

    resp_das = np.abs(np.conj(w_das) @ a_grid)
    resp_pm = np.abs(np.conj(w_pm) @ a_grid)
    resp_mvdr = np.abs(np.conj(w_mvdr) @ a_grid)

    # Normalize Power-Min pattern so peak or reference is clear, but keep true array response in dB
    pattern_das_db = 20.0 * np.log10(np.maximum(resp_das, 1e-5))
    pattern_pm_db = 20.0 * np.log10(np.maximum(resp_pm / max(np.max(resp_pm), 1e-6), 1e-5))
    pattern_mvdr_db = 20.0 * np.log10(np.maximum(resp_mvdr, 1e-5))

    # Exact gain / null depth at satellite and jammer angles
    pm_norm_factor = max(float(np.max(resp_pm)), 1e-6)
    sat_gain_pm_db = float(20.0 * np.log10(max(abs(np.vdot(w_pm, a_sat)) / pm_norm_factor, 1e-5)))
    sat_gain_mvdr_db = float(20.0 * np.log10(max(abs(np.vdot(w_mvdr, a_sat)), 1e-5)))

    nulls_pm_db: List[float] = []
    nulls_mvdr_db: List[float] = []
    for j_ang in jam_angles:
        a_j = steering_vector_ula(j_ang, num_elements, d_lambda)
        g_pm = 20.0 * np.log10(max(abs(np.vdot(w_pm, a_j)) / pm_norm_factor, 1e-6))
        g_mvdr = 20.0 * np.log10(max(abs(np.vdot(w_mvdr, a_j)), 1e-6))
        nulls_pm_db.append(float(g_pm))
        nulls_mvdr_db.append(float(g_mvdr))

    # Eigenvalue spectrum of R_xx (sorted descending in dB relative to thermal noise floor = 0 dB)
    eigvals = np.linalg.eigvalsh(cov_data["R_theo"])
    eigvals_sorted = np.sort(np.real(eigvals))[::-1]
    eigenvalues_db = 10.0 * np.log10(np.maximum(eigvals_sorted, 1e-12))

    # Single element reference weight w_single = [1, 0, ..., 0]^T
    w_single = np.zeros(num_elements, dtype=np.complex128)
    w_single[0] = 1.0

    sinr_single_db = evaluate_sinr_db(w_single, r_sig, r_jn)
    sinr_pm_db = evaluate_sinr_db(w_pm, r_sig, r_jn)
    sinr_mvdr_db = evaluate_sinr_db(w_mvdr, r_sig, r_jn)

    eff_cn0_single = sinr_single_db + bw_db
    eff_cn0_pm = sinr_pm_db + bw_db
    eff_cn0_mvdr = sinr_mvdr_db + bw_db

    # Sweep J/S Ratio from 25 dB to 85 dB to plot Effective C/N0 curves
    jsr_sweep = np.linspace(25.0, 85.0, 25)
    cn0_single_arr = np.empty_like(jsr_sweep)
    cn0_pm_arr = np.empty_like(jsr_sweep)
    cn0_mvdr_arr = np.empty_like(jsr_sweep)

    sigma_s2 = 10.0 ** (snr_pre_db / 10.0)
    diag_load_lin = 10.0 ** (diagonal_loading_db / 10.0)
    a_jams = [steering_vector_ula(ja, num_elements, d_lambda) for ja in jam_angles]
    delta_j_offsets = [j_db - jam1_jsr_db for j_db in jam_jsrs]

    for idx, jsr_val in enumerate(jsr_sweep):
        r_jn_sw = np.eye(num_elements, dtype=np.complex128)
        for a_j, d_off in zip(a_jams, delta_j_offsets):
            p_j = sigma_s2 * (10.0 ** ((jsr_val + d_off) / 10.0))
            r_jn_sw += p_j * np.outer(a_j, np.conj(a_j))
        r_tot_sw = r_sig + r_jn_sw + diag_load_lin * np.eye(num_elements)
        w_sw = compute_adaptive_weights(r_tot_sw, a_sat)

        cn0_single_arr[idx] = evaluate_sinr_db(w_single, r_sig, r_jn_sw) + bw_db
        cn0_pm_arr[idx] = evaluate_sinr_db(w_sw["w_pm"], r_sig, r_jn_sw) + bw_db
        cn0_mvdr_arr[idx] = evaluate_sinr_db(w_sw["w_mvdr"], r_sig, r_jn_sw) + bw_db

    return CrpaSimulationResult(
        angles_deg=angles_deg,
        pattern_das_db=pattern_das_db,
        pattern_pm_db=pattern_pm_db,
        pattern_mvdr_db=pattern_mvdr_db,
        weights_das=w_das,
        weights_pm=w_pm,
        weights_mvdr=w_mvdr,
        eigenvalues_db=eigenvalues_db,
        sat_angle_deg=sat_angle_deg,
        jammer_angles_deg=jam_angles,
        jammer_jsr_db=jam_jsrs,
        null_depths_pm_db=nulls_pm_db,
        null_depths_mvdr_db=nulls_mvdr_db,
        sat_gain_pm_db=sat_gain_pm_db,
        sat_gain_mvdr_db=sat_gain_mvdr_db,
        sinr_single_db=sinr_single_db,
        sinr_pm_db=sinr_pm_db,
        sinr_mvdr_db=sinr_mvdr_db,
        eff_cn0_single_dbhz=eff_cn0_single,
        eff_cn0_pm_dbhz=eff_cn0_pm,
        eff_cn0_mvdr_dbhz=eff_cn0_mvdr,
        jsr_sweep_db=jsr_sweep,
        cn0_sweep_single_dbhz=cn0_single_arr,
        cn0_sweep_pm_dbhz=cn0_pm_arr,
        cn0_sweep_mvdr_dbhz=cn0_mvdr_arr,
    )
