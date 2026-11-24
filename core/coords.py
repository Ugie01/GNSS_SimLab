"""
Pure NumPy/SciPy mathematical and physical engine for Module 1:
- WGS-84 Geodetic (LLH) <-> Earth-Centered Earth-Fixed (ECEF) <-> Local Tangent Plane (ENU/NED)
- IEEE 754 float32 vs float64 coordinate quantization error simulation
- GPS Time (Week, TOW), Leap Seconds, and 1PPS hardware interrupt vs MCU oscillator drift simulation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import numpy as np

# ==============================================================================
# WGS-84 Ellipsoid Constants (IS-GPS-200 Standard)
# ==============================================================================
WGS84_A: float = 6378137.0  # Semi-major axis (Equatorial radius) [m]
WGS84_F: float = 1.0 / 298.257223563  # Flattening
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis (Polar radius) [m]
WGS84_E2: float = 2.0 * WGS84_F - (WGS84_F ** 2)  # First eccentricity squared e^2
WGS84_EP2: float = (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)  # Second eccentricity squared e'^2

GPS_EPOCH_UTC = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
SECONDS_PER_WEEK: int = 604800


@dataclass(frozen=True)
class CoordinateConversionResult:
    """Comprehensive output container for LLH / ECEF / ENU conversion and round-trip metrics."""

    ref_llh_deg_m: Tuple[float, float, float]
    ref_ecef_m: np.ndarray
    prime_vertical_radius_n_m: float
    target_enu_m: np.ndarray
    target_ned_m: np.ndarray
    target_ecef_f64_m: np.ndarray
    target_llh_f64_deg_m: Tuple[float, float, float]
    target_ecef_f32_m: np.ndarray
    target_enu_from_f32_m: np.ndarray
    f32_ecef_error_vec_m: np.ndarray
    f32_ecef_error_norm_m: float
    f32_enu_error_vec_m: np.ndarray
    f32_enu_error_norm_m: float
    azimuth_deg: float
    elevation_deg: float
    slant_range_m: float


def prime_vertical_radius(lat_rad: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the Prime Vertical Radius of Curvature N(phi) on the WGS-84 ellipsoid.

    N(phi) = a / sqrt(1 - e^2 * sin^2(phi))
    """
    sin_lat = np.sin(lat_rad)
    return WGS84_A / np.sqrt(1.0 - WGS84_E2 * (sin_lat ** 2))


def llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """
    Convert WGS-84 geodetic coordinates (Latitude [deg], Longitude [deg], Ellipsoidal Height [m])
    into Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates [X, Y, Z] in meters (float64).
    """
    if not (np.isfinite(lat_deg) and np.isfinite(lon_deg) and np.isfinite(alt_m)):
        raise ValueError("LLH coordinates must be finite real numbers.")
    clamped_lat = float(np.clip(lat_deg, -90.0, 90.0))
    lat = np.deg2rad(np.float64(clamped_lat))
    lon = np.deg2rad(np.float64(lon_deg))
    h = np.float64(alt_m)

    n_val = prime_vertical_radius(lat)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    x = (n_val + h) * cos_lat * cos_lon
    y = (n_val + h) * cos_lat * sin_lon
    z = (n_val * (1.0 - WGS84_E2) + h) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


def ecef_to_llh(ecef_m: np.ndarray, max_iter: int = 8, tol_rad: float = 1e-14) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates [X, Y, Z] (meters) into WGS-84 geodetic coordinates
    (lat_deg, lon_deg, alt_m) using Bowring-initialized iterative refinement.
    """
    arr = np.asarray(ecef_m, dtype=np.float64).ravel()
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        raise ValueError("ecef_m must be a finite 3-element array [X, Y, Z].")
    x, y, z = arr[0], arr[1], arr[2]
    p = np.hypot(x, y)

    if p < 1e-10:
        lat = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        lon = 0.0
        alt = abs(z) - WGS84_B
        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)

    lon = np.arctan2(y, x)

    # Bowring's initial approximation
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * (np.sin(theta) ** 3),
        p - WGS84_E2 * WGS84_A * (np.cos(theta) ** 3),
    )

    # High-precision iterative refinement
    for _ in range(max_iter):
        n_val = prime_vertical_radius(lat)
        lat_new = np.arctan2(z + WGS84_E2 * n_val * np.sin(lat), p)
        if abs(lat_new - lat) < tol_rad:
            lat = lat_new
            break
        lat = lat_new

    n_val = prime_vertical_radius(lat)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)

    if abs(cos_lat) > 1e-6:
        alt = p / cos_lat - n_val
    else:
        alt = z / sin_lat - n_val * (1.0 - WGS84_E2)

    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def ecef_to_enu_matrix(ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    """
    Return the 3x3 orthogonal rotation matrix R_ecef2enu that transforms
    an ECEF difference vector dX into local [East, North, Up]^T at (ref_lat_deg, ref_lon_deg).
    """
    phi = np.deg2rad(np.float64(ref_lat_deg))
    lam = np.deg2rad(np.float64(ref_lon_deg))

    sin_p, cos_p = np.sin(phi), np.cos(phi)
    sin_l, cos_l = np.sin(lam), np.cos(lam)

    return np.array(
        [
            [-sin_l, cos_l, 0.0],
            [-sin_p * cos_l, -sin_p * sin_l, cos_p],
            [cos_p * cos_l, cos_p * sin_l, sin_p],
        ],
        dtype=np.float64,
    )


def ecef_to_enu(target_ecef_m: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    """Transform target ECEF coordinates [X, Y, Z] into local ENU [E, N, U] relative to a reference LLH."""
    ref_ecef = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    dx = np.asarray(target_ecef_m, dtype=np.float64) - ref_ecef
    rot = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    return rot @ dx


def enu_to_ecef(enu_m: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    """Transform local ENU coordinates [E, N, U] relative to a reference LLH into ECEF [X, Y, Z]."""
    ref_ecef = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    rot = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    return ref_ecef + (rot.T @ np.asarray(enu_m, dtype=np.float64))


def enu_to_az_el_range(enu_m: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Azimuth [0~360 deg clockwise from North], Elevation [-90~+90 deg],
    and Slant Range [m] from an ENU vector [E, N, U].
    """
    e, n, u = float(enu_m[0]), float(enu_m[1]), float(enu_m[2])
    horiz = np.hypot(e, n)
    slant = float(np.hypot(horiz, u))
    if slant < 1e-12:
        return 0.0, 0.0, 0.0
    az = float(np.rad2deg(np.arctan2(e, n)) % 360.0)
    el = float(np.rad2deg(np.arctan2(u, horiz)))
    return az, el, slant


def analyze_coordinate_pipeline(
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
    east_m: float,
    north_m: float,
    up_m: float,
) -> CoordinateConversionResult:
    """
    Perform complete coordinate conversion and compare IEEE 754 float64 vs float32
    quantization errors when storing ECEF coordinates in 32-bit float variables.
    """
    ref_ecef = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    n_radius = float(prime_vertical_radius(np.deg2rad(ref_lat_deg)))
    target_enu = np.array([east_m, north_m, up_m], dtype=np.float64)
    target_ned = np.array([north_m, east_m, -up_m], dtype=np.float64)

    target_ecef_f64 = enu_to_ecef(target_enu, ref_lat_deg, ref_lon_deg, ref_alt_m)
    target_llh_f64 = ecef_to_llh(target_ecef_f64)

    # Simulate Embedded C `float` (32-bit IEEE 754) storage of ECEF coordinates
    target_ecef_f32 = target_ecef_f64.astype(np.float32).astype(np.float64)
    ref_ecef_f32 = ref_ecef.astype(np.float32).astype(np.float64)

    # In embedded C, subtracting two float32 ECEF vectors loses sub-meter precision
    rot = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    target_enu_from_f32 = rot @ (target_ecef_f32 - ref_ecef_f32)

    f32_ecef_err_vec = target_ecef_f32 - target_ecef_f64
    f32_ecef_err_norm = float(np.linalg.norm(f32_ecef_err_vec))

    f32_enu_err_vec = target_enu_from_f32 - target_enu
    f32_enu_err_norm = float(np.linalg.norm(f32_enu_err_vec))

    az, el, slant = enu_to_az_el_range(target_enu)

    return CoordinateConversionResult(
        ref_llh_deg_m=(ref_lat_deg, ref_lon_deg, ref_alt_m),
        ref_ecef_m=ref_ecef,
        prime_vertical_radius_n_m=n_radius,
        target_enu_m=target_enu,
        target_ned_m=target_ned,
        target_ecef_f64_m=target_ecef_f64,
        target_llh_f64_deg_m=target_llh_f64,
        target_ecef_f32_m=target_ecef_f32,
        target_enu_from_f32_m=target_enu_from_f32,
        f32_ecef_error_vec_m=f32_ecef_err_vec,
        f32_ecef_error_norm_m=f32_ecef_err_norm,
        f32_enu_error_vec_m=f32_enu_err_vec,
        f32_enu_error_norm_m=f32_enu_err_norm,
        azimuth_deg=az,
        elevation_deg=el,
        slant_range_m=slant,
    )


def simulate_float32_vs_float64_trajectory(
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
    radius_m: float = 3.0,
    num_points: int = 240,
) -> Dict[str, np.ndarray]:
    """
    Generate a smooth circular trajectory in local ENU (e.g. a small drone or RTK rover moving
    along a circle of radius `radius_m`) and demonstrate the stair-stepping quantization effect
    when ECEF coordinates are stored in IEEE 754 float32 vs float64.
    """
    safe_radius = max(float(radius_m), 0.01)
    safe_pts = max(int(num_points), 8)
    angles = np.linspace(0.0, 2.0 * np.pi, safe_pts, endpoint=True)
    true_east = safe_radius * np.cos(angles)
    true_north = safe_radius * np.sin(angles)
    true_up = 0.5 * np.sin(2.0 * angles)

    ref_ecef_f64 = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    ref_ecef_f32 = ref_ecef_f64.astype(np.float32).astype(np.float64)
    rot = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)

    enu_true = np.vstack([true_east, true_north, true_up])  # (3, N)
    ecef_f64 = ref_ecef_f64[:, None] + (rot.T @ enu_true)

    # Quantize to float32 in ECEF frame (typical embedded mistake)
    ecef_f32 = ecef_f64.astype(np.float32).astype(np.float64)
    enu_f32 = rot @ (ecef_f32 - ref_ecef_f32[:, None])

    pos_error_3d_m = np.linalg.norm(enu_f32 - enu_true, axis=0)
    horiz_error_m = np.hypot(enu_f32[0] - enu_true[0], enu_f32[1] - enu_true[1])

    # Also compute float32 direct LLH storage quantization error (in meters)
    cos_lat_safe = max(abs(float(np.cos(np.deg2rad(np.clip(ref_lat_deg, -89.99, 89.99))))), 1e-4)
    lat_arr = ref_lat_deg + (true_north / 111132.0)
    lon_arr = ref_lon_deg + (true_east / (111320.0 * cos_lat_safe))
    lat_f32 = lat_arr.astype(np.float32).astype(np.float64)
    lon_f32 = lon_arr.astype(np.float32).astype(np.float64)
    llh_f32_horiz_err_m = np.hypot(
        (lat_f32 - lat_arr) * 111132.0,
        (lon_f32 - lon_arr) * 111320.0 * cos_lat_safe,
    )

    return {
        "angles_rad": angles,
        "true_east_m": true_east,
        "true_north_m": true_north,
        "f32_east_m": enu_f32[0],
        "f32_north_m": enu_f32[1],
        "ecef_f32_error_3d_m": pos_error_3d_m,
        "ecef_f32_error_horiz_m": horiz_error_m,
        "llh_f32_error_horiz_m": llh_f32_horiz_err_m,
    }


def utc_to_gpst(utc_dt: datetime, leap_seconds: int = 18) -> Dict[str, float | int | str]:
    """
    Convert a UTC datetime to GPS Time (Continuous GPS Week, Rollover Week [0..1023],
    Time of Week [TOW, s], and GPST timestamp).
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    else:
        utc_dt = utc_dt.astimezone(timezone.utc)

    gpst_dt = utc_dt + timedelta(seconds=int(leap_seconds))
    elapsed_sec = (gpst_dt - GPS_EPOCH_UTC).total_seconds()

    full_week = int(elapsed_sec // SECONDS_PER_WEEK)
    tow_sec = float(elapsed_sec - full_week * SECONDS_PER_WEEK)
    mod1024_week = full_week % 1024
    rollover_epoch = full_week // 1024
    day_of_week = int(tow_sec // 86400)

    return {
        "utc_iso": utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC",
        "gpst_iso": gpst_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " GPST",
        "leap_seconds": int(leap_seconds),
        "gps_week_full": full_week,
        "gps_week_mod1024": mod1024_week,
        "rollover_count": rollover_epoch,
        "tow_seconds": tow_sec,
        "day_of_week": day_of_week,
    }


def simulate_1pps_and_mcu_clock(
    duration_sec: int = 12,
    pps_jitter_ns: float = 25.0,
    mcu_drift_ppm: float = 20.0,
    nmea_uart_delay_ms: float = 180.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate 1PPS hardware pulse edges, UART NMEA packet arrival timing,
    and free-running MCU crystal oscillator drift vs 1PPS-disciplined MCU clock error.

    Parameters
    ----------
    duration_sec : int
        Simulation duration in integer seconds.
    pps_jitter_ns : float
        1-sigma hardware timing jitter of the GNSS receiver's 1PPS output [ns].
    mcu_drift_ppm : float
        MCU crystal oscillator frequency error in parts-per-million [ppm] (1 ppm = 1 us/sec).
    nmea_uart_delay_ms : float
        Average serial transmission delay of NMEA/UBX packet after the 1PPS rising edge [ms].
    seed : int
        Random seed for deterministic reproducibility.
    """
    duration_sec = max(int(duration_sec), 1)
    pps_jitter_ns = max(float(pps_jitter_ns), 0.0)
    nmea_uart_delay_ms = float(np.clip(nmea_uart_delay_ms, 0.0, 950.0))

    rng = np.random.default_rng(seed)
    sec_axis = np.arange(0, duration_sec + 1, dtype=np.float64)

    # 1. Free-running MCU oscillator error without 1PPS synchronization [microseconds]
    # 1 ppm = 1 us drift per second, plus small thermal walk
    thermal_walk_us = np.cumsum(rng.normal(0.0, 0.05, size=len(sec_axis)))
    thermal_walk_us[0] = 0.0
    free_running_err_us = sec_axis * mcu_drift_ppm + thermal_walk_us

    # 2. 1PPS-disciplined MCU clock sawtooth / PLL error [microseconds]
    # Sub-second fine time axis (20 samples per second) to visualize the sawtooth reset at each 1PPS edge
    fine_t = np.linspace(0.0, float(duration_sec), duration_sec * 25 + 1)
    pps_edge_jitter_us = rng.normal(0.0, pps_jitter_ns * 1e-3, size=duration_sec + 1)

    disciplined_err_us = np.zeros_like(fine_t)
    for idx, t_val in enumerate(fine_t):
        sec_idx = min(int(np.floor(t_val)), duration_sec)
        frac_sec = t_val - sec_idx
        # Between 1PPS edges, if undisciplined rate is not compensated, sawtooth builds up and resets at edge;
        # With software FLL/PLL timer capture calibration, residual error stays near 1PPS jitter + tiny residual ppm
        residual_ppm = mcu_drift_ppm * 0.02  # 98% drift calibrated out by measuring timer ticks between 1PPS
        disciplined_err_us[idx] = pps_edge_jitter_us[sec_idx] + frac_sec * residual_ppm

    sawtooth_uncalibrated_us = np.zeros_like(fine_t)
    for idx, t_val in enumerate(fine_t):
        sec_idx = min(int(np.floor(t_val)), duration_sec)
        frac_sec = t_val - sec_idx
        sawtooth_uncalibrated_us[idx] = pps_edge_jitter_us[sec_idx] + frac_sec * mcu_drift_ppm

    # 3. Single-second timing diagram waveform (0.0 to 2.2 seconds) showing 1PPS pulse vs UART NMEA arrival
    waveform_t_ms = np.linspace(0.0, 2200.0, 2201)
    pps_wave = np.zeros_like(waveform_t_ms)
    uart_wave = np.zeros_like(waveform_t_ms)

    for k in range(3):
        edge_ms = k * 1000.0
        # 1PPS pulse width: 100 ms high for visualization clarity
        pps_mask = (waveform_t_ms >= edge_ms) & (waveform_t_ms <= edge_ms + 90.0)
        pps_wave[pps_mask] = 1.0

        # UART packet arrives after nmea_uart_delay_ms and lasts ~75 ms
        uart_start = edge_ms + nmea_uart_delay_ms + rng.uniform(-12.0, 12.0)
        uart_mask = (waveform_t_ms >= uart_start) & (waveform_t_ms <= uart_start + 80.0)
        uart_wave[uart_mask] = 1.0

    return {
        "sec_axis": sec_axis,
        "free_running_err_us": free_running_err_us,
        "fine_t": fine_t,
        "sawtooth_reset_only_us": sawtooth_uncalibrated_us,
        "disciplined_fll_err_us": disciplined_err_us,
        "pps_edge_jitter_ns": pps_edge_jitter_us * 1e3,
        "waveform_t_ms": waveform_t_ms,
        "pps_wave": pps_wave,
        "uart_wave": uart_wave,
    }


def generate_wgs84_3d_globe_and_enu_triad(
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float = 0.0,
    triad_length_km: float = 2400.0,
) -> Dict[str, np.ndarray]:
    """
    Generate 3D WGS-84 Earth ellipsoid surface mesh, equator/meridian curves,
    reference ECEF vector, local ENU tangent plane patch, and orthogonal [East, North, Up]
    basis vectors in kilometers for 3D visualization.
    """
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 19)
    u_grid, v_grid = np.meshgrid(u, v)

    a_km = WGS84_A * 1e-3
    b_km = WGS84_B * 1e-3

    globe_x = a_km * np.cos(v_grid) * np.cos(u_grid)
    globe_y = a_km * np.cos(v_grid) * np.sin(u_grid)
    globe_z = b_km * np.sin(v_grid)

    eq_theta = np.linspace(0.0, 2.0 * np.pi, 120)
    equator_xyz = np.vstack([a_km * np.cos(eq_theta), a_km * np.sin(eq_theta), np.zeros_like(eq_theta)])

    mer_phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 90)
    meridian_xyz = np.vstack([a_km * np.cos(mer_phi), np.zeros_like(mer_phi), b_km * np.sin(mer_phi)])

    ref_ecef_km = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m) * 1e-3
    rot_ecef2enu = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    # Rows of rot_ecef2enu are the unit East, North, Up vectors expressed in ECEF
    e_unit = rot_ecef2enu[0]
    n_unit = rot_ecef2enu[1]
    u_unit = rot_ecef2enu[2]

    # Local tangent plane patch around ref_ecef_km
    patch_half = triad_length_km * 0.55
    s_grid, t_grid = np.meshgrid([-patch_half, patch_half], [-patch_half, patch_half])
    plane_x = ref_ecef_km[0] + s_grid * e_unit[0] + t_grid * n_unit[0]
    plane_y = ref_ecef_km[1] + s_grid * e_unit[1] + t_grid * n_unit[1]
    plane_z = ref_ecef_km[2] + s_grid * e_unit[2] + t_grid * n_unit[2]

    return {
        "globe_x_km": globe_x,
        "globe_y_km": globe_y,
        "globe_z_km": globe_z,
        "equator_xyz_km": equator_xyz,
        "meridian_xyz_km": meridian_xyz,
        "ref_ecef_km": ref_ecef_km,
        "east_vec_km": e_unit * triad_length_km,
        "north_vec_km": n_unit * triad_length_km,
        "up_vec_km": u_unit * triad_length_km,
        "plane_x_km": plane_x,
        "plane_y_km": plane_y,
        "plane_z_km": plane_z,
    }

