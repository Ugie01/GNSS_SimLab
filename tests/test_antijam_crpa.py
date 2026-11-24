"""Unit tests for core/antijam_crpa.py validating ULA steering vectors, MVDR distortionless constraint, and CRPA nulling."""

import numpy as np
import pytest

from core.antijam_crpa import (
    compute_adaptive_weights,
    compute_spatial_covariance,
    run_crpa_simulation,
    steering_vector_ula,
)


def test_steering_vector_ula_boresight_and_norm():
    """Verify ULA steering vector at boresight (0 deg) is all ones and norm^2 == N."""
    a0 = steering_vector_ula(0.0, num_elements=6, d_lambda=0.5)
    assert a0.shape == (6,)
    assert np.allclose(a0, np.ones(6, dtype=np.complex128))
    assert pytest.approx(float(np.vdot(a0, a0).real), abs=1e-9) == 6.0


def test_mvdr_distortionless_constraint_and_power_min_constraint():
    """
    Verify fundamental algebraic constraints:
    - MVDR: w_MVDR^H * a(theta_s) == 1.0 (0 dB distortionless response at satellite direction)
    - Power-Min: w_PM[0] == 1.0 (first reference element weight clamped to unity)
    """
    cov = compute_spatial_covariance(
        num_elements=4,
        d_lambda=0.5,
        sat_angle_deg=20.0,
        jammer_angles_deg=[-35.0, 50.0],
        jammer_jsr_db=[60.0, 55.0],
    )
    w_dict = compute_adaptive_weights(cov["R_sample_loaded"], cov["a_sat"])

    mvdr_sat_resp = np.vdot(w_dict["w_mvdr"], cov["a_sat"])
    assert pytest.approx(float(np.abs(mvdr_sat_resp)), abs=1e-6) == 1.0
    assert pytest.approx(float(np.real(w_dict["w_pm"][0])), abs=1e-6) == 1.0


def test_crpa_deep_nulling_and_cn0_restoration():
    """
    Verify that under 60 dB J/S jamming:
    - Single antenna effective C/N0 collapses below 10 dB-Hz.
    - Both Power-Min and MVDR place deep nulls (<-25 dB) at jammer directions
      and restore effective C/N0 above 40 dB-Hz.
    - MVDR achieves higher effective C/N0 than Power-Min due to main-beam array gain.
    """
    res = run_crpa_simulation(
        num_elements=4,
        d_lambda=0.5,
        sat_angle_deg=15.0,
        jam1_angle_deg=-40.0,
        jam1_jsr_db=60.0,
        jam2_enabled=True,
        jam2_angle_deg=50.0,
        jam2_jsr_db=55.0,
        nominal_cn0_dbhz=46.0,
    )

    assert res.eff_cn0_single_dbhz < 10.0
    assert res.eff_cn0_pm_dbhz > 40.0
    assert res.eff_cn0_mvdr_dbhz > res.eff_cn0_pm_dbhz
    assert all(nd < -25.0 for nd in res.null_depths_mvdr_db)
    assert all(nd < -20.0 for nd in res.null_depths_pm_db)


def test_crpa_degrees_of_freedom_limit():
    """
    Verify eigenvalue subspace property:
    With N=4 elements and 2 independent jammers, the top 2 eigenvalues of R_xx
    are much larger (>30 dB) than the remaining N-2 noise-floor eigenvalues.
    """
    res = run_crpa_simulation(
        num_elements=4,
        d_lambda=0.5,
        jam1_angle_deg=-35.0,
        jam1_jsr_db=60.0,
        jam2_enabled=True,
        jam2_angle_deg=45.0,
        jam2_jsr_db=55.0,
    )
    assert res.eigenvalues_db[0] > 30.0
    assert res.eigenvalues_db[1] > 25.0
    assert abs(res.eigenvalues_db[2]) < 3.0
    assert abs(res.eigenvalues_db[3]) < 3.0
