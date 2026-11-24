"""
Pure NumPy/SciPy mathematical and physical engine for Module 5:
- 2D/Planar Vehicle/UAV Dynamic Trajectory & True Kinematics Generator (50 Hz IMU / 5 Hz GNSS)
- Realistic MEMS IMU Error Model (Accelerometer & Gyroscope Biases + White Noise)
- Pure INS Dead-Reckoning Mechanization (Uncalibrated Drift Demonstration)
- 8-State Error-State Extended Kalman Filter (ESKF / Indirect EKF) with Joseph-Form Covariance Update
- GNSS Outage (Jamming / Tunnel Blackout) Coasting & Recovery Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class EkfSimulationResult:
    """Comprehensive output container for GNSS/INS Error-State EKF simulation."""

    t_sec: np.ndarray
    true_pos_enu_m: np.ndarray       # Shape (N, 2) [East, North]
    true_vel_enu_mps: np.ndarray     # Shape (N, 2) [v_E, v_N]
    true_yaw_deg: np.ndarray         # Shape (N,)
    ins_only_pos_enu_m: np.ndarray   # Shape (N, 2) Uncalibrated Pure INS Dead-Reckoning
    ekf_pos_enu_m: np.ndarray        # Shape (N, 2) Error-State EKF Fused Trajectory
    ekf_vel_enu_mps: np.ndarray      # Shape (N, 2)
    ekf_yaw_deg: np.ndarray          # Shape (N,)
    gnss_meas_t_sec: np.ndarray      # Timestamps of active GNSS measurements
    gnss_meas_pos_enu_m: np.ndarray  # Shape (M, 2) Active GNSS position measurements
    outage_mask: np.ndarray          # Boolean mask (N,) True during GNSS Outage
    ins_pos_err_m: np.ndarray        # Shape (N,) 2D position error of Pure INS
    ekf_pos_err_m: np.ndarray        # Shape (N,) 2D position error of EKF
    ekf_pos_3sigma_m: np.ndarray     # Shape (N,) 3-sigma position uncertainty bound
    ins_yaw_err_deg: np.ndarray      # Shape (N,) Heading error of Pure INS
    ekf_yaw_err_deg: np.ndarray      # Shape (N,) Heading error of EKF
    ekf_yaw_3sigma_deg: np.ndarray   # Shape (N,) 3-sigma heading uncertainty bound
    est_bias_ax_mps2: np.ndarray     # Shape (N,) EKF estimated forward accel bias
    est_bias_ay_mps2: np.ndarray     # Shape (N,) EKF estimated lateral accel bias
    est_bias_gz_deg_s: np.ndarray    # Shape (N,) EKF estimated gyro Z bias [deg/s]
    true_bias_ax_mps2: float
    true_bias_ay_mps2: float
    true_bias_gz_deg_s: float
    max_ins_err_m: float
    max_ekf_outage_err_m: float
    steady_ekf_rms_m: float


def wrap_angle_rad(angle_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) in radians to [-pi, +pi]."""
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def generate_true_trajectory(
    duration_sec: float = 60.0,
    dt_sec: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    Generate a smooth, dynamically rich Figure-8 / curved reconnaissance trajectory
    with analytically exact velocity, acceleration, heading, body-frame specific force
    [a_x^b, a_y^b], and yaw rate omega_z.
    """
    duration_sec = max(float(duration_sec), 10.0)
    dt_sec = float(np.clip(dt_sec, 0.005, 0.1))
    t = np.arange(0.0, duration_sec + 0.5 * dt_sec, dt_sec)
    w0 = 2.0 * np.pi / duration_sec  # 1 full Figure-8 loop over duration_sec

    # Parametric Lissajous / Figure-8 trajectory (scale: 160m East x 110m North)
    scale_e = 160.0
    scale_n = 110.0

    p_e = scale_e * np.sin(w0 * t)
    p_n = scale_n * np.sin(2.0 * w0 * t)

    v_e = scale_e * w0 * np.cos(w0 * t)
    v_n = scale_n * (2.0 * w0) * np.cos(2.0 * w0 * t)

    a_e = -scale_e * (w0 ** 2) * np.sin(w0 * t)
    a_n = -scale_n * ((2.0 * w0) ** 2) * np.sin(2.0 * w0 * t)

    j_e = -scale_e * (w0 ** 3) * np.cos(w0 * t)
    j_n = -scale_n * ((2.0 * w0) ** 3) * np.cos(2.0 * w0 * t)

    # Heading psi measured counter-clockwise from East axis: v_E = V cos(psi), v_N = V sin(psi)
    psi = np.arctan2(v_n, v_e)
    v_sq = np.maximum(v_e ** 2 + v_n ** 2, 1e-6)

    # Exact analytical yaw rate: d(psi)/dt = (v_E * a_N - v_N * a_E) / (v_E^2 + v_N^2)
    omega_z = (v_e * a_n - v_n * a_e) / v_sq

    # Project world ENU acceleration [a_E, a_N] into Body frame [a_x^b (forward), a_y^b (left)]
    # R(psi) = [[cos, -sin], [sin, cos]] => a_body = R(psi)^T @ a_enu
    cos_p = np.cos(psi)
    sin_p = np.sin(psi)
    ax_b = cos_p * a_e + sin_p * a_n
    ay_b = -sin_p * a_e + cos_p * a_n

    return {
        "t": t,
        "pos_enu": np.column_stack([p_e, p_n]),
        "vel_enu": np.column_stack([v_e, v_n]),
        "acc_enu": np.column_stack([a_e, a_n]),
        "jerk_enu": np.column_stack([j_e, j_n]),
        "psi_rad": psi,
        "omega_z_rad_s": omega_z,
        "acc_body": np.column_stack([ax_b, ay_b]),
    }


def run_gnss_ins_ekf_simulation(
    accel_bias_mps2: float = 0.25,
    gyro_bias_deg_s: float = 1.2,
    accel_noise_mps2: float = 0.08,
    gyro_noise_deg_s: float = 0.15,
    gnss_pos_sigma_m: float = 1.8,
    gnss_vel_sigma_mps: float = 0.15,
    outage_start_sec: float = 22.0,
    outage_duration_sec: float = 18.0,
    ekf_q_scale: float = 1.0,
    ekf_r_scale: float = 1.0,
    duration_sec: float = 60.0,
    seed: int = 505,
) -> EkfSimulationResult:
    """
    Execute the 8-State Error-State EKF (ESKF) and Pure INS Dead-Reckoning simulation
    with a configurable GNSS Outage (Jamming / Blackout) window.

    State vector (8 states):
      Nominal: [p_E, p_N, v_E, v_N, psi, b_ax, b_ay, b_gz]^T
      Error  : [dp_E, dp_N, dv_E, dv_N, dpsi, db_ax, db_ay, db_gz]^T
    """
    rng = np.random.default_rng(seed)
    dt = 0.02  # 50 Hz IMU rate
    gnss_step_interval = 10  # 5 Hz GNSS update rate (every 0.20 s)

    traj = generate_true_trajectory(duration_sec=duration_sec, dt_sec=dt)
    t_arr = traj["t"]
    n_steps = len(t_arr)

    true_pos = traj["pos_enu"]
    true_vel = traj["vel_enu"]
    true_psi = traj["psi_rad"]
    true_omega_z = traj["omega_z_rad_s"]
    true_acc_b = traj["acc_body"]

    # True constant/slowly-varying sensor biases
    true_b_ax = float(accel_bias_mps2)
    true_b_ay = float(-0.7 * accel_bias_mps2)
    true_b_gz = float(np.deg2rad(gyro_bias_deg_s))

    sigma_a = max(float(accel_noise_mps2), 1e-4)
    sigma_g = max(float(np.deg2rad(gyro_noise_deg_s)), 1e-5)
    sigma_p_gnss = max(float(gnss_pos_sigma_m), 0.05)
    sigma_v_gnss = max(float(gnss_vel_sigma_mps), 0.01)

    # Synthesize corrupted IMU measurements
    meas_acc_b = true_acc_b + np.array([true_b_ax, true_b_ay]) + rng.normal(0.0, sigma_a, size=(n_steps, 2))
    meas_gyro_z = true_omega_z + true_b_gz + rng.normal(0.0, sigma_g, size=n_steps)

    # Outage window mask
    outage_end_sec = outage_start_sec + max(float(outage_duration_sec), 0.0)
    outage_mask = (t_arr >= outage_start_sec) & (t_arr <= outage_end_sec)

    # --------------------------------------------------------------------------
    # 1. Pure INS Dead-Reckoning (No GNSS Aiding, Uncalibrated Biases)
    # --------------------------------------------------------------------------
    ins_pos = np.zeros((n_steps, 2), dtype=np.float64)
    ins_vel = np.zeros((n_steps, 2), dtype=np.float64)
    ins_psi = np.zeros(n_steps, dtype=np.float64)

    ins_pos[0] = true_pos[0]
    ins_vel[0] = true_vel[0]
    ins_psi[0] = true_psi[0]

    for k in range(1, n_steps):
        psi_prev = ins_psi[k - 1]
        psi_mid = psi_prev + 0.5 * meas_gyro_z[k - 1] * dt
        cos_m, sin_m = np.cos(psi_mid), np.sin(psi_mid)
        ax_m, ay_m = meas_acc_b[k - 1, 0], meas_acc_b[k - 1, 1]
        a_e = cos_m * ax_m - sin_m * ay_m
        a_n = sin_m * ax_m + cos_m * ay_m

        ins_pos[k, 0] = ins_pos[k - 1, 0] + ins_vel[k - 1, 0] * dt + 0.5 * a_e * (dt ** 2)
        ins_pos[k, 1] = ins_pos[k - 1, 1] + ins_vel[k - 1, 1] * dt + 0.5 * a_n * (dt ** 2)
        ins_vel[k, 0] = ins_vel[k - 1, 0] + a_e * dt
        ins_vel[k, 1] = ins_vel[k - 1, 1] + a_n * dt
        ins_psi[k] = float(wrap_angle_rad(psi_prev + meas_gyro_z[k - 1] * dt))

    # --------------------------------------------------------------------------
    # 2. 8-State Error-State EKF (ESKF) with Joseph-Form Update
    # --------------------------------------------------------------------------
    ekf_pos = np.zeros((n_steps, 2), dtype=np.float64)
    ekf_vel = np.zeros((n_steps, 2), dtype=np.float64)
    ekf_psi = np.zeros(n_steps, dtype=np.float64)
    est_b_ax = np.zeros(n_steps, dtype=np.float64)
    est_b_ay = np.zeros(n_steps, dtype=np.float64)
    est_b_gz = np.zeros(n_steps, dtype=np.float64)

    ekf_pos_3sigma = np.zeros(n_steps, dtype=np.float64)
    ekf_yaw_3sigma_deg = np.zeros(n_steps, dtype=np.float64)

    # Initialize nominal state
    p_nom = true_pos[0].copy()
    v_nom = true_vel[0].copy()
    psi_nom = float(true_psi[0])
    b_ax_nom = 0.0
    b_ay_nom = 0.0
    b_gz_nom = 0.0

    # Initial Error Covariance P (8x8)
    p_cov = np.diag(
        [
            sigma_p_gnss ** 2,
            sigma_p_gnss ** 2,
            sigma_v_gnss ** 2,
            sigma_v_gnss ** 2,
            np.deg2rad(3.0) ** 2,
            (0.5) ** 2,
            (0.5) ** 2,
            np.deg2rad(2.0) ** 2,
        ]
    )

    # Discrete Process Noise Covariance Q_k (scaled by user slider ekf_q_scale)
    q_mult = max(float(ekf_q_scale), 0.05)
    q_cov = q_mult * np.diag(
        [
            (0.5 * sigma_a * dt ** 2) ** 2,
            (0.5 * sigma_a * dt ** 2) ** 2,
            (sigma_a * dt) ** 2,
            (sigma_a * dt) ** 2,
            (sigma_g * dt) ** 2,
            (1e-3 * dt) ** 2,
            (1e-3 * dt) ** 2,
            (2e-4 * dt) ** 2,
        ]
    )

    # Measurement matrix H (4x8) observing [dp_E, dp_N, dv_E, dv_N]
    h_mat = np.zeros((4, 8), dtype=np.float64)
    h_mat[:4, :4] = np.eye(4)

    # Measurement Noise Covariance R_GNSS (scaled by user slider ekf_r_scale)
    r_mult = max(float(ekf_r_scale), 0.05)
    r_cov = r_mult * np.diag(
        [
            sigma_p_gnss ** 2,
            sigma_p_gnss ** 2,
            sigma_v_gnss ** 2,
            sigma_v_gnss ** 2,
        ]
    )

    gnss_t_list = []
    gnss_pos_list = []

    ekf_pos[0] = p_nom
    ekf_vel[0] = v_nom
    ekf_psi[0] = psi_nom
    ekf_pos_3sigma[0] = 3.0 * np.sqrt(p_cov[0, 0] + p_cov[1, 1])
    ekf_yaw_3sigma_deg[0] = 3.0 * np.rad2deg(np.sqrt(p_cov[4, 4]))

    eye8 = np.eye(8, dtype=np.float64)

    for k in range(1, n_steps):
        # 2.1 Bias-Compensated INS Mechanization (Nominal State Propagation)
        ax_corr = meas_acc_b[k - 1, 0] - b_ax_nom
        ay_corr = meas_acc_b[k - 1, 1] - b_ay_nom
        gz_corr = meas_gyro_z[k - 1] - b_gz_nom

        psi_mid = psi_nom + 0.5 * gz_corr * dt
        cos_p, sin_p = np.cos(psi_mid), np.sin(psi_mid)

        a_e = cos_p * ax_corr - sin_p * ay_corr
        a_n = sin_p * ax_corr + cos_p * ay_corr

        p_nom[0] += v_nom[0] * dt + 0.5 * a_e * (dt ** 2)
        p_nom[1] += v_nom[1] * dt + 0.5 * a_n * (dt ** 2)
        v_nom[0] += a_e * dt
        v_nom[1] += a_n * dt
        psi_nom = float(wrap_angle_rad(psi_nom + gz_corr * dt))

        # 2.2 Error-State Jacobian & State Transition Matrix F_k = I + A * dt
        # dv_dot = (J_psi * a_body) * dpsi - R(psi) * db_a
        # J_psi * a_body = [-sin_p * ax - cos_p * ay, cos_p * ax - sin_p * ay]^T = [-a_n, a_e]^T
        f_mat = eye8.copy()
        f_mat[0, 2] = dt
        f_mat[1, 3] = dt

        f_mat[2, 4] = (-sin_p * ax_corr - cos_p * ay_corr) * dt
        f_mat[3, 4] = (cos_p * ax_corr - sin_p * ay_corr) * dt

        f_mat[2, 5] = -cos_p * dt
        f_mat[2, 6] = sin_p * dt
        f_mat[3, 5] = -sin_p * dt
        f_mat[3, 6] = -cos_p * dt

        f_mat[4, 7] = -dt

        # Covariance Prediction
        p_cov = f_mat @ p_cov @ f_mat.T + q_cov

        # 2.3 GNSS Measurement Update (if on 5 Hz epoch AND not inside GNSS Outage window)
        if (k % gnss_step_interval == 0) and (not outage_mask[k]):
            gnss_p = true_pos[k] + rng.normal(0.0, sigma_p_gnss, size=2)
            gnss_v = true_vel[k] + rng.normal(0.0, sigma_v_gnss, size=2)

            gnss_t_list.append(t_arr[k])
            gnss_pos_list.append(gnss_p)

            # Innovation z = y_GNSS - y_nom
            z_innov = np.array(
                [
                    gnss_p[0] - p_nom[0],
                    gnss_p[1] - p_nom[1],
                    gnss_v[0] - v_nom[0],
                    gnss_v[1] - v_nom[1],
                ],
                dtype=np.float64,
            )

            s_mat = h_mat @ p_cov @ h_mat.T + r_cov
            try:
                k_gain = p_cov @ h_mat.T @ np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                k_gain = p_cov @ h_mat.T @ np.linalg.pinv(s_mat)

            dx = k_gain @ z_innov

            # Joseph Stabilized Covariance Update: P+ = (I - K H) P- (I - K H)^T + K R K^T
            ikh = eye8 - (k_gain @ h_mat)
            p_cov = (ikh @ p_cov @ ikh.T) + (k_gain @ r_cov @ k_gain.T)
            p_cov = 0.5 * (p_cov + p_cov.T)  # Enforce exact symmetry

            # Error Injection into Nominal State & implicit reset of error state dx <- 0
            p_nom[0] += dx[0]
            p_nom[1] += dx[1]
            v_nom[0] += dx[2]
            v_nom[1] += dx[3]
            psi_nom = float(wrap_angle_rad(psi_nom + dx[4]))
            b_ax_nom += dx[5]
            b_ay_nom += dx[6]
            b_gz_nom += dx[7]

        # Store step trajectories & covariance envelopes
        ekf_pos[k] = p_nom
        ekf_vel[k] = v_nom
        ekf_psi[k] = psi_nom
        est_b_ax[k] = b_ax_nom
        est_b_ay[k] = b_ay_nom
        est_b_gz[k] = np.rad2deg(b_gz_nom)

        ekf_pos_3sigma[k] = 3.0 * np.sqrt(max(p_cov[0, 0] + p_cov[1, 1], 0.0))
        ekf_yaw_3sigma_deg[k] = 3.0 * np.rad2deg(np.sqrt(max(p_cov[4, 4], 0.0)))

    # Error metrics
    ins_pos_err = np.linalg.norm(ins_pos - true_pos, axis=1)
    ekf_pos_err = np.linalg.norm(ekf_pos - true_pos, axis=1)

    ins_yaw_err_deg = np.rad2deg(np.abs(wrap_angle_rad(ins_psi - true_psi)))
    ekf_yaw_err_deg = np.rad2deg(np.abs(wrap_angle_rad(ekf_psi - true_psi)))

    max_ins_err = float(np.max(ins_pos_err))
    max_ekf_outage_err = float(np.max(ekf_pos_err[outage_mask])) if np.any(outage_mask) else float(np.max(ekf_pos_err))
    non_outage_mask = ~outage_mask
    steady_ekf_rms = float(np.sqrt(np.mean(ekf_pos_err[non_outage_mask] ** 2))) if np.any(non_outage_mask) else float(np.mean(ekf_pos_err))

    gnss_meas_pos_arr = np.vstack(gnss_pos_list) if gnss_pos_list else np.zeros((1, 2))
    gnss_meas_t_arr = np.array(gnss_t_list, dtype=np.float64)

    return EkfSimulationResult(
        t_sec=t_arr,
        true_pos_enu_m=true_pos,
        true_vel_enu_mps=true_vel,
        true_yaw_deg=np.rad2deg(true_psi),
        ins_only_pos_enu_m=ins_pos,
        ekf_pos_enu_m=ekf_pos,
        ekf_vel_enu_mps=ekf_vel,
        ekf_yaw_deg=np.rad2deg(ekf_psi),
        gnss_meas_t_sec=gnss_meas_t_arr,
        gnss_meas_pos_enu_m=gnss_meas_pos_arr,
        outage_mask=outage_mask,
        ins_pos_err_m=ins_pos_err,
        ekf_pos_err_m=ekf_pos_err,
        ekf_pos_3sigma_m=ekf_pos_3sigma,
        ins_yaw_err_deg=ins_yaw_err_deg,
        ekf_yaw_err_deg=ekf_yaw_err_deg,
        ekf_yaw_3sigma_deg=ekf_yaw_3sigma_deg,
        est_bias_ax_mps2=est_b_ax,
        est_bias_ay_mps2=est_b_ay,
        est_bias_gz_deg_s=est_b_gz,
        true_bias_ax_mps2=true_b_ax,
        true_bias_ay_mps2=true_b_ay,
        true_bias_gz_deg_s=float(gyro_bias_deg_s),
        max_ins_err_m=max_ins_err,
        max_ekf_outage_err_m=max_ekf_outage_err,
        steady_ekf_rms_m=steady_ekf_rms,
    )
