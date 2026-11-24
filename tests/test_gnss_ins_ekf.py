"""Unit tests for core/gnss_ins_ekf.py validating trajectory kinematics, Error-State EKF bias estimation, and GNSS Outage coasting."""

import numpy as np
import pytest

from core.gnss_ins_ekf import generate_true_trajectory, run_gnss_ins_ekf_simulation


def test_true_trajectory_kinematic_consistency():
    """Verify numerical derivative of position matches analytical velocity and body acceleration norms match ENU."""
    traj = generate_true_trajectory(duration_sec=40.0, dt_sec=0.02)
    num_vel = np.gradient(traj["pos_enu"], 0.02, axis=0)
    assert np.allclose(num_vel[5:-5], traj["vel_enu"][5:-5], atol=0.15)

    acc_enu_norm = np.linalg.norm(traj["acc_enu"], axis=1)
    acc_body_norm = np.linalg.norm(traj["acc_body"], axis=1)
    assert np.allclose(acc_enu_norm, acc_body_norm, atol=1e-9)


def test_eskf_imu_bias_convergence_and_tracking():
    """
    Verify that the 8-State Error-State EKF accurately estimates the unknown
    gyroscope bias (deg/s) and accelerometer bias (m/s^2) and maintains < 2m RMS error.
    """
    res = run_gnss_ins_ekf_simulation(
        accel_bias_mps2=0.30,
        gyro_bias_deg_s=1.5,
        outage_duration_sec=0.0,  # Continuous GNSS to test full convergence
        duration_sec=50.0,
    )

    assert res.steady_ekf_rms_m < 1.8
    assert res.max_ins_err_m > 50.0  # Pure INS drifts massively (>50 m)

    # Check final 10 seconds average of estimated gyro and accel biases
    final_bg_est = float(np.mean(res.est_bias_gz_deg_s[-200:]))
    final_bax_est = float(np.mean(res.est_bias_ax_mps2[-200:]))

    assert pytest.approx(final_bg_est, abs=0.25) == 1.5
    assert pytest.approx(final_bax_est, abs=0.10) == 0.30


def test_gnss_outage_covariance_growth_and_recovery():
    """
    Verify behavior during a 16-second GNSS Outage (t = 22s to 38s):
    - Uncertainty bound (3-sigma) grows during the outage and drops sharply after t=38s recovery.
    - Bias-calibrated EKF coasting error during outage is vastly smaller than uncalibrated Pure INS drift.
    """
    res = run_gnss_ins_ekf_simulation(
        accel_bias_mps2=0.25,
        gyro_bias_deg_s=1.2,
        outage_start_sec=22.0,
        outage_duration_sec=16.0,
        duration_sec=60.0,
    )

    idx_before = int(21.0 / 0.02)
    idx_peak_outage = int(37.8 / 0.02)
    idx_recovered = int(42.0 / 0.02)

    assert res.ekf_pos_3sigma_m[idx_peak_outage] > 2.5 * res.ekf_pos_3sigma_m[idx_before]
    assert res.ekf_pos_3sigma_m[idx_recovered] < 0.4 * res.ekf_pos_3sigma_m[idx_peak_outage]
    assert res.max_ekf_outage_err_m < 0.15 * res.max_ins_err_m
