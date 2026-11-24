"""Unit tests for core/coords.py validating mathematical accuracy of coordinate and time algorithms."""

from datetime import datetime, timezone

import numpy as np
import pytest

from core.coords import (
    WGS84_A,
    WGS84_B,
    analyze_coordinate_pipeline,
    ecef_to_enu,
    ecef_to_llh,
    enu_to_az_el_range,
    enu_to_ecef,
    generate_wgs84_3d_globe_and_enu_triad,
    llh_to_ecef,
    prime_vertical_radius,
    simulate_1pps_and_mcu_clock,
    simulate_float32_vs_float64_trajectory,
    utc_to_gpst,
)


def test_wgs84_equator_and_poles():
    """Verify LLH -> ECEF at the equator (0, 0, 0) and North Pole (90, 0, 0)."""
    ecef_eq = llh_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(ecef_eq, [WGS84_A, 0.0, 0.0], atol=1e-6)

    ecef_np = llh_to_ecef(90.0, 0.0, 0.0)
    assert np.allclose(ecef_np, [0.0, 0.0, WGS84_B], atol=1e-6)

    lat, lon, alt = ecef_to_llh(ecef_np)
    assert pytest.approx(lat, abs=1e-6) == 90.0
    assert pytest.approx(alt, abs=1e-4) == 0.0


@pytest.mark.parametrize(
    "lat_deg,lon_deg,alt_m",
    [
        (37.5665, 126.9780, 65.0),     # Seoul
        (36.3721, 127.3604, 92.0),     # Daejeon
        (-33.8688, 151.2093, 25.0),    # Sydney
        (0.0, -78.4678, 2850.0),       # Quito (Equator, high altitude)
        (64.1466, -21.9426, 1200.0),   # Reykjavik (High latitude)
    ],
)
def test_llh_ecef_round_trip(lat_deg: float, lon_deg: float, alt_m: float):
    """Ensure LLH -> ECEF -> LLH round-trip error is below 1e-8 deg and 1e-4 m."""
    ecef = llh_to_ecef(lat_deg, lon_deg, alt_m)
    lat_out, lon_out, alt_out = ecef_to_llh(ecef)

    assert pytest.approx(lat_out, abs=1e-9) == lat_deg
    assert pytest.approx(lon_out, abs=1e-9) == lon_deg
    assert pytest.approx(alt_out, abs=1e-4) == alt_m


def test_enu_ecef_round_trip_and_orthogonality():
    """Verify ENU <-> ECEF round-trip accuracy and azimuth/elevation calculation."""
    ref_lat, ref_lon, ref_alt = 37.5665, 126.9780, 50.0
    target_enu = np.array([300.0, 400.0, 500.0], dtype=np.float64)

    target_ecef = enu_to_ecef(target_enu, ref_lat, ref_lon, ref_alt)
    recovered_enu = ecef_to_enu(target_ecef, ref_lat, ref_lon, ref_alt)

    assert np.allclose(recovered_enu, target_enu, atol=1e-7)

    az, el, slant = enu_to_az_el_range(target_enu)
    assert pytest.approx(slant, abs=1e-6) == np.sqrt(300.0**2 + 400.0**2 + 500.0**2)
    assert pytest.approx(el, abs=1e-6) == 45.0  # horiz = 500, up = 500 -> el = 45 deg
    assert pytest.approx(az, abs=1e-6) == np.rad2deg(np.arctan2(300.0, 400.0))


def test_float32_quantization_error_detection():
    """Verify that float32 ECEF quantization produces noticeable sub-meter errors (~0.1 to 1.0 m)."""
    res = analyze_coordinate_pipeline(37.5665, 126.9780, 65.0, 123.45, 234.56, 12.34)
    assert res.prime_vertical_radius_n_m > WGS84_A

    traj = simulate_float32_vs_float64_trajectory(37.5665, 126.9780, 65.0, radius_m=2.5, num_points=120)
    max_f32_err = float(np.max(traj["ecef_f32_error_3d_m"]))
    # At ~6.37e6 m magnitude, float32 ULP is 0.25 ~ 0.5 m, so 3D quantization error max must exceed 0.1 m
    assert 0.05 < max_f32_err < 2.0


def test_utc_to_gpst_conversion():
    """Verify GPS Week and TOW calculation against known epoch offset."""
    # GPS Epoch: 1980-01-06 00:00:00 UTC with 0 leap seconds -> Week 0, TOW 0
    ep = utc_to_gpst(datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc), leap_seconds=0)
    assert ep["gps_week_full"] == 0
    assert ep["tow_seconds"] == 0.0

    # Exactly 1 week + 18 leap seconds later
    dt = datetime(1980, 1, 13, 0, 0, 0, tzinfo=timezone.utc)
    res = utc_to_gpst(dt, leap_seconds=18)
    assert res["gps_week_full"] == 1
    assert pytest.approx(res["tow_seconds"], abs=1e-6) == 18.0


def test_1pps_simulation_shapes_and_drift():
    """Verify 1PPS and MCU drift simulation output arrays and physical trend."""
    sim = simulate_1pps_and_mcu_clock(duration_sec=10, pps_jitter_ns=20.0, mcu_drift_ppm=25.0)
    assert len(sim["sec_axis"]) == 11
    # After 10 seconds at 25 ppm, free running error should be around 250 microseconds
    assert 200.0 < sim["free_running_err_us"][-1] < 300.0
    # Disciplined clock error must remain well under 5 microseconds
    assert np.max(np.abs(sim["disciplined_fll_err_us"])) < 5.0


def test_coords_invalid_inputs_and_edge_cases():
    """Verify exception handling for NaN/Inf inputs and zero-vector ENU Az/El calculation."""
    with pytest.raises(ValueError):
        llh_to_ecef(float("nan"), 127.0, 50.0)

    with pytest.raises(ValueError):
        ecef_to_llh(np.array([1.0, 2.0]))  # Less than 3 elements

    az, el, slant = enu_to_az_el_range(np.array([0.0, 0.0, 0.0]))
    assert (az, el, slant) == (0.0, 0.0, 0.0)


def test_wgs84_3d_globe_and_enu_triad_orthogonality():
    """Verify 3D WGS-84 globe mesh generation and orthogonality of the 3D ENU basis triad vectors."""
    g = generate_wgs84_3d_globe_and_enu_triad(37.5665, 126.9780, 65.0, triad_length_km=2000.0)
    ev, nv, uv = g["east_vec_km"], g["north_vec_km"], g["up_vec_km"]
    assert pytest.approx(float(np.dot(ev, nv)), abs=1e-6) == 0.0
    assert pytest.approx(float(np.dot(ev, uv)), abs=1e-6) == 0.0
    assert pytest.approx(float(np.dot(nv, uv)), abs=1e-6) == 0.0
    assert pytest.approx(float(np.linalg.norm(ev)), rel=1e-6) == 2000.0


