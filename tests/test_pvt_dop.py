"""Unit tests for core/pvt_dop.py validating DOP computation, Gauss-Newton WLS convergence, and SBAS improvement."""

import numpy as np
import pytest

from core.coords import llh_to_ecef
from core.pvt_dop import (
    compute_dop_from_az_el,
    generate_constellation_at_reference,
    ionospheric_obliquity_factor,
    run_pvt_simulation_pipeline,
    solve_gauss_newton_wls,
)


def test_ionospheric_obliquity_factor_monotonic():
    """Verify obliquity factor is 1.0 at Zenith (90°) and increases monotonically as elevation drops."""
    f_90 = float(ionospheric_obliquity_factor(90.0))
    f_45 = float(ionospheric_obliquity_factor(45.0))
    f_10 = float(ionospheric_obliquity_factor(10.0))

    assert pytest.approx(f_90, abs=1e-6) == 1.0
    assert 1.0 < f_45 < f_10
    assert f_10 > 2.5


def test_dop_relationships_and_degradation_under_masking():
    """
    Verify fundamental algebraic identities of DOP:
    - PDOP^2 == HDOP^2 + VDOP^2
    - GDOP^2 == PDOP^2 + TDOP^2
    - Raising elevation mask angle reduces visible satellites and increases (degrades) PDOP.
    """
    sats_open = [s for s in generate_constellation_at_reference(37.5665, 126.9780, 65.0, mask_angle_deg=5.0) if s.is_visible]
    sats_high_mask = [s for s in generate_constellation_at_reference(37.5665, 126.9780, 65.0, mask_angle_deg=30.0) if s.is_visible]

    dop_open = compute_dop_from_az_el([s.az_deg for s in sats_open], [s.el_deg for s in sats_open])
    dop_masked = compute_dop_from_az_el([s.az_deg for s in sats_high_mask], [s.el_deg for s in sats_high_mask])

    assert pytest.approx(dop_open.pdop ** 2, rel=1e-5) == (dop_open.hdop ** 2 + dop_open.vdop ** 2)
    assert pytest.approx(dop_open.gdop ** 2, rel=1e-5) == (dop_open.pdop ** 2 + dop_open.tdop ** 2)
    assert len(sats_open) > len(sats_high_mask)
    assert dop_masked.pdop > dop_open.pdop


def test_gauss_newton_wls_exact_convergence_from_earth_center():
    """
    Verify Gauss-Newton WLS solver converges from Earth center [0, 0, 0] (~6,378 km away!)
    to the exact true user ECEF position and clock bias within 6 iterations when noise is zero.
    """
    ref_lat, ref_lon, ref_alt = 37.5665, 126.9780, 100.0
    true_ecef = llh_to_ecef(ref_lat, ref_lon, ref_alt)
    true_clock_m = 8450.25

    sats = [s for s in generate_constellation_at_reference(ref_lat, ref_lon, ref_alt, mask_angle_deg=10.0) if s.is_visible]
    sat_ecef_mat = np.vstack([s.ecef_m for s in sats])
    noise_free_rho = np.array([s.true_range_m + true_clock_m for s in sats], dtype=np.float64)

    res = solve_gauss_newton_wls(
        sat_ecef_matrix=sat_ecef_mat,
        pseudoranges_m=noise_free_rho,
        init_ecef_m=np.zeros(3),
        true_ecef_m=true_ecef,
        max_iter=8,
    )

    assert np.allclose(res["est_ecef_m"], true_ecef, atol=1e-4)
    assert pytest.approx(res["est_clock_bias_m"], abs=1e-4) == true_clock_m
    assert res["step_norms_m"][-1] < 1e-5


def test_sbas_reduces_positioning_rms_error():
    """Verify SBAS (KASS/WAAS) augmentation significantly reduces both Horizontal and Vertical RMS errors."""
    sim = run_pvt_simulation_pipeline(
        ref_lat_deg=37.5665,
        ref_lon_deg=126.9780,
        ref_alt_m=65.0,
        mask_angle_deg=10.0,
        zenith_iono_m=5.0,
        base_uere_noise_m=1.5,
        num_mc_runs=80,
    )
    assert sim.sbas_h_rms_m < 0.6 * sim.standalone_h_rms_m
    assert sim.sbas_v_rms_m < 0.5 * sim.standalone_v_rms_m


def test_underdetermined_constellation_handling():
    """Verify graceful handling when fewer than 4 satellites are visible (e.g. extreme mask angle)."""
    with pytest.raises(ValueError):
        solve_gauss_newton_wls(np.zeros((3, 3)), np.zeros(3))

    sim_under = run_pvt_simulation_pipeline(
        mask_angle_deg=55.0,
        urban_canyon_deg=65.0,
    )
    assert len(sim_under.active_satellites) < 4
    assert sim_under.dop.gdop == 99.9
    assert sim_under.standalone_h_rms_m == 99.9

