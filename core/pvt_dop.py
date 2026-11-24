"""
Pure NumPy/SciPy mathematical and physical engine for Module 3:
- Multi-satellite GNSS sky constellation generation & Elevation Mask / Urban Canyon filtering
- Geometry Matrix (H) and Dilution of Precision (GDOP, PDOP, HDOP, VDOP, TDOP)
- Physical Pseudorange Error Budget (Ionospheric Obliquity, Troposphere, Multipath, Clock Bias)
- SBAS (KASS / WAAS) Wide-Area Differential & Ionospheric Grid Correction Model
- Gauss-Newton Weighted Least Squares (WLS) Iterative PVT Solver & Monte Carlo Scatter Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.coords import WGS84_A, ecef_to_enu, ecef_to_enu_matrix, enu_to_ecef, llh_to_ecef

SPEED_OF_LIGHT_MPS: float = 299792458.0
GPS_ORBIT_RADIUS_M: float = 26560000.0  # ~20,180 km altitude
IONO_SHELL_HEIGHT_M: float = 350000.0   # 350 km thin-shell ionosphere height

# Default 11-satellite constellation (PRN, Azimuth [deg], Elevation [deg])
DEFAULT_SKY_CONSTELLATION: List[Tuple[int, float, float]] = [
    (2, 18.0, 74.0),
    (5, 62.0, 48.0),
    (7, 104.0, 29.0),
    (9, 138.0, 61.0),
    (12, 175.0, 22.0),
    (15, 212.0, 53.0),
    (18, 254.0, 34.0),
    (21, 286.0, 16.0),
    (24, 315.0, 66.0),
    (27, 346.0, 39.0),
    (30, 82.0, 12.0),
]


@dataclass(frozen=True)
class SatelliteInfo:
    """State and measurement breakdown for a single GNSS satellite."""

    prn: int
    az_deg: float
    el_deg: float
    ecef_m: np.ndarray
    true_range_m: float
    iono_delay_m: float
    tropo_residual_m: float
    multipath_sigma_m: float
    is_visible: bool


@dataclass(frozen=True)
class DopMetrics:
    """Dilution of Precision metrics in local ENU frame."""

    gdop: float
    pdop: float
    hdop: float
    vdop: float
    tdop: float
    edop: float
    ndop: float
    cov_enu_t: np.ndarray  # 4x4 (H_enu^T H_enu)^-1


@dataclass(frozen=True)
class WlsPvtResult:
    """Comprehensive output of Gauss-Newton WLS positioning and Monte Carlo SBAS comparison."""

    all_satellites: List[SatelliteInfo]
    active_satellites: List[SatelliteInfo]
    dop: DopMetrics
    iter_step_norms_m: np.ndarray
    iter_pos_errors_m: np.ndarray
    standalone_enu_err_m: np.ndarray
    sbas_enu_err_m: np.ndarray
    est_clock_bias_m: float
    true_clock_bias_m: float
    mc_standalone_enu_m: np.ndarray  # Shape (N_mc, 3)
    mc_sbas_enu_m: np.ndarray        # Shape (N_mc, 3)
    standalone_h_rms_m: float
    standalone_v_rms_m: float
    sbas_h_rms_m: float
    sbas_v_rms_m: float


def ionospheric_obliquity_factor(el_deg: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the thin-shell ionospheric mapping (obliquity) function F(El):
    F(El) = 1 / sqrt(1 - (R_e * cos(El) / (R_e + h_iono))^2)
    Ranges from 1.0 at Zenith (El=90°) to ~3.0 at low elevation (El=5°).
    """
    el_rad = np.deg2rad(np.maximum(el_deg, 2.0))
    ratio = (WGS84_A * np.cos(el_rad)) / (WGS84_A + IONO_SHELL_HEIGHT_M)
    return 1.0 / np.sqrt(1.0 - ratio ** 2)


def generate_constellation_at_reference(
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
    mask_angle_deg: float = 10.0,
    urban_canyon_deg: float = 0.0,
    zenith_iono_m: float = 4.5,
    base_uere_noise_m: float = 1.5,
) -> List[SatelliteInfo]:
    """
    Generate 3D ECEF satellite coordinates and error models around `(ref_lat_deg, ref_lon_deg, ref_alt_m)`.

    `urban_canyon_deg`: Additional elevation cutoff applied to East/West sectors (Azimuth 45°~135° and 225°~315°)
    simulating tall buildings flanking a North-South street.
    """
    ref_ecef = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    slant_dist = GPS_ORBIT_RADIUS_M - WGS84_A

    sat_list: List[SatelliteInfo] = []
    for prn, az_deg, el_deg in DEFAULT_SKY_CONSTELLATION:
        az_rad = np.deg2rad(az_deg)
        el_rad = np.deg2rad(el_deg)

        # Unit LOS vector in local ENU: [East, North, Up]
        u_east = np.cos(el_rad) * np.sin(az_rad)
        u_north = np.cos(el_rad) * np.cos(az_rad)
        u_up = np.sin(el_rad)
        sat_enu = slant_dist * np.array([u_east, u_north, u_up], dtype=np.float64)

        sat_ecef = enu_to_ecef(sat_enu, ref_lat_deg, ref_lon_deg, ref_alt_m)
        true_range = float(np.linalg.norm(sat_ecef - ref_ecef))

        # Check visibility against general elevation mask & urban canyon E/W building mask
        in_ew_sector = (45.0 <= az_deg <= 135.0) or (225.0 <= az_deg <= 315.0)
        effective_mask = max(mask_angle_deg, urban_canyon_deg) if in_ew_sector else mask_angle_deg
        is_vis = bool(el_deg >= effective_mask)

        # Physical atmospheric & multipath error terms
        # Mid-latitude daytime ionosphere has higher TEC toward the South-Southwest (Equatorial Anomaly gradient)
        obliq = float(ionospheric_obliquity_factor(el_deg))
        horiz_tec_gradient = 1.0 - 0.28 * np.cos(az_rad) - 0.15 * np.sin(az_rad)
        iono_m = zenith_iono_m * obliq * horiz_tec_gradient
        tropo_res_m = 0.25 / np.sin(np.deg2rad(max(el_deg, 5.0)))
        mp_sigma_m = base_uere_noise_m * (0.45 + 0.65 / np.sin(np.deg2rad(max(el_deg, 5.0))))

        sat_list.append(
            SatelliteInfo(
                prn=prn,
                az_deg=az_deg,
                el_deg=el_deg,
                ecef_m=sat_ecef,
                true_range_m=true_range,
                iono_delay_m=iono_m,
                tropo_residual_m=tropo_res_m,
                multipath_sigma_m=mp_sigma_m,
                is_visible=is_vis,
            )
        )

    return sat_list


def compute_dop_from_az_el(az_deg_list: List[float], el_deg_list: List[float]) -> DopMetrics:
    """
    Compute exact Dilution of Precision (GDOP, PDOP, HDOP, VDOP, TDOP, EDOP, NDOP)
    from visible satellite Azimuth and Elevation angles in the local ENU frame.
    """
    n_sats = len(az_deg_list)
    if n_sats < 4:
        inf_val = 99.9
        return DopMetrics(inf_val, inf_val, inf_val, inf_val, inf_val, inf_val, inf_val, np.eye(4) * 999.0)

    h_enu = np.empty((n_sats, 4), dtype=np.float64)
    for i, (az_d, el_d) in enumerate(zip(az_deg_list, el_deg_list)):
        az = np.deg2rad(az_d)
        el = np.deg2rad(el_d)
        h_enu[i, 0] = -np.cos(el) * np.sin(az)  # East LOS component
        h_enu[i, 1] = -np.cos(el) * np.cos(az)  # North LOS component
        h_enu[i, 2] = -np.sin(el)               # Up LOS component
        h_enu[i, 3] = 1.0                       # Clock bias term

    # Tikhonov diagonal loading (1e-9) prevents LinAlgError if geometry is nearly singular
    gram = h_enu.T @ h_enu + 1e-9 * np.eye(4)
    try:
        q_mat = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        q_mat = np.linalg.pinv(gram)

    q_e = max(float(q_mat[0, 0]), 0.0)
    q_n = max(float(q_mat[1, 1]), 0.0)
    q_u = max(float(q_mat[2, 2]), 0.0)
    q_t = max(float(q_mat[3, 3]), 0.0)

    edop = float(np.sqrt(q_e))
    ndop = float(np.sqrt(q_n))
    hdop = float(np.sqrt(q_e + q_n))
    vdop = float(np.sqrt(q_u))
    pdop = float(np.sqrt(q_e + q_n + q_u))
    tdop = float(np.sqrt(q_t))
    gdop = float(np.sqrt(q_e + q_n + q_u + q_t))

    return DopMetrics(
        gdop=min(gdop, 99.9),
        pdop=min(pdop, 99.9),
        hdop=min(hdop, 99.9),
        vdop=min(vdop, 99.9),
        tdop=min(tdop, 99.9),
        edop=min(edop, 99.9),
        ndop=min(ndop, 99.9),
        cov_enu_t=q_mat,
    )


def solve_gauss_newton_wls(
    sat_ecef_matrix: np.ndarray,
    pseudoranges_m: np.ndarray,
    weights: np.ndarray | None = None,
    init_ecef_m: np.ndarray | None = None,
    true_ecef_m: np.ndarray | None = None,
    max_iter: int = 8,
    tol_m: float = 1e-6,
) -> Dict[str, np.ndarray | float]:
    """
    Solve non-linear pseudorange equations for receiver ECEF position [X, Y, Z] and
    clock bias b = c * dt_u [m] using iterative Gauss-Newton Weighted Least Squares (WLS).

    Returns estimated ECEF, estimated clock bias, final Geometry Matrix H,
    and per-iteration step norms ||dx_k|| and true position errors.
    """
    n_sats = sat_ecef_matrix.shape[0]
    if n_sats < 4:
        raise ValueError("At least 4 visible satellites are required for 3D PVT + Clock solution.")

    if init_ecef_m is None:
        # Start from Earth center [0, 0, 0] to demonstrate global 6,378 km convergence in 5~6 steps!
        x_est = np.zeros(3, dtype=np.float64)
    else:
        x_est = np.array(init_ecef_m, dtype=np.float64).copy()

    b_est = 0.0

    if weights is None:
        w_mat = np.eye(n_sats, dtype=np.float64)
    else:
        w_mat = np.diag(np.asarray(weights, dtype=np.float64))

    step_norms: List[float] = []
    pos_errors: List[float] = []

    if true_ecef_m is not None:
        pos_errors.append(float(np.linalg.norm(x_est - true_ecef_m)))

    h_mat = np.zeros((n_sats, 4), dtype=np.float64)

    for _ in range(max_iter):
        diff = sat_ecef_matrix - x_est[None, :]  # (N, 3) vector from receiver to each satellite
        ranges = np.linalg.norm(diff, axis=1)
        ranges = np.maximum(ranges, 1.0)

        # Predicted pseudorange
        rho_pred = ranges + b_est
        delta_rho = pseudoranges_m - rho_pred

        # Jacobian / Geometry Matrix H in ECEF frame: [-e_x, -e_y, -e_z, 1]
        h_mat[:, :3] = -diff / ranges[:, None]
        h_mat[:, 3] = 1.0

        # Normal equations with slight diagonal regularization for numerical safety
        ht_w = h_mat.T @ w_mat
        normal_mat = ht_w @ h_mat + 1e-10 * np.eye(4)
        try:
            dx = np.linalg.solve(normal_mat, ht_w @ delta_rho)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(normal_mat, ht_w @ delta_rho, rcond=None)[0]

        x_est += dx[:3]
        b_est += float(dx[3])

        step_norm = float(np.linalg.norm(dx))
        step_norms.append(step_norm)
        if true_ecef_m is not None:
            pos_errors.append(float(np.linalg.norm(x_est - true_ecef_m)))

        if step_norm < tol_m:
            break

    return {
        "est_ecef_m": x_est,
        "est_clock_bias_m": b_est,
        "step_norms_m": np.array(step_norms, dtype=np.float64),
        "pos_errors_m": np.array(pos_errors, dtype=np.float64),
        "H_ecef": h_mat,
    }


def run_pvt_simulation_pipeline(
    ref_lat_deg: float = 37.5665,
    ref_lon_deg: float = 126.9780,
    ref_alt_m: float = 65.0,
    mask_angle_deg: float = 10.0,
    urban_canyon_deg: float = 0.0,
    zenith_iono_m: float = 4.5,
    base_uere_noise_m: float = 1.5,
    use_elevation_weighting: bool = True,
    num_mc_runs: int = 100,
    seed: int = 2026,
) -> WlsPvtResult:
    """
    Execute the complete Module 3 simulation pipeline:
    1. Constellation visibility & Skyplot data
    2. DOP calculation
    3. Gauss-Newton WLS iterative convergence from Earth center [0,0,0]
    4. Standalone GPS vs SBAS (KASS/WAAS) positioning error & Monte Carlo scatter comparison
    """
    rng = np.random.default_rng(seed)
    true_ecef = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    true_clock_bias_m = 12500.0  # +12.5 km (~41.7 us receiver clock offset)

    all_sats = generate_constellation_at_reference(
        ref_lat_deg=ref_lat_deg,
        ref_lon_deg=ref_lon_deg,
        ref_alt_m=ref_alt_m,
        mask_angle_deg=mask_angle_deg,
        urban_canyon_deg=urban_canyon_deg,
        zenith_iono_m=zenith_iono_m,
        base_uere_noise_m=base_uere_noise_m,
    )
    active_sats = [s for s in all_sats if s.is_visible]

    dop = compute_dop_from_az_el(
        [s.az_deg for s in active_sats],
        [s.el_deg for s in active_sats],
    )

    if len(active_sats) < 4:
        zeros3 = np.zeros(3)
        return WlsPvtResult(
            all_satellites=all_sats,
            active_satellites=active_sats,
            dop=dop,
            iter_step_norms_m=np.array([0.0]),
            iter_pos_errors_m=np.array([0.0]),
            standalone_enu_err_m=zeros3,
            sbas_enu_err_m=zeros3,
            est_clock_bias_m=0.0,
            true_clock_bias_m=true_clock_bias_m,
            mc_standalone_enu_m=np.zeros((10, 3)),
            mc_sbas_enu_m=np.zeros((10, 3)),
            standalone_h_rms_m=99.9,
            standalone_v_rms_m=99.9,
            sbas_h_rms_m=99.9,
            sbas_v_rms_m=99.9,
        )

    sat_ecef_mat = np.vstack([s.ecef_m for s in active_sats])
    true_ranges = np.array([s.true_range_m for s in active_sats], dtype=np.float64)
    iono_delays = np.array([s.iono_delay_m for s in active_sats], dtype=np.float64)
    tropo_res = np.array([s.tropo_residual_m for s in active_sats], dtype=np.float64)
    mp_sigmas = np.array([s.multipath_sigma_m for s in active_sats], dtype=np.float64)
    el_rads = np.deg2rad(np.array([s.el_deg for s in active_sats], dtype=np.float64))

    if use_elevation_weighting:
        weights = (np.sin(el_rads) ** 2) / (np.maximum(base_uere_noise_m, 0.1) ** 2)
    else:
        weights = np.ones(len(active_sats), dtype=np.float64)

    # Single-epoch representative measurement for WLS iteration convergence plot
    ep_noise = rng.normal(0.0, mp_sigmas)
    rho_standalone = true_ranges + true_clock_bias_m + iono_delays + tropo_res + ep_noise
    # SBAS (KASS/WAAS) removes ~88% of ionospheric delay & 75% of tropo/ephemeris bias, plus 40% carrier-smoothing
    rho_sbas = (
        true_ranges
        + true_clock_bias_m
        + 0.12 * iono_delays
        + 0.25 * tropo_res
        + 0.60 * ep_noise
    )

    wls_standalone = solve_gauss_newton_wls(
        sat_ecef_matrix=sat_ecef_mat,
        pseudoranges_m=rho_standalone,
        weights=weights,
        init_ecef_m=np.zeros(3),  # Start from Earth center [0,0,0]
        true_ecef_m=true_ecef,
        max_iter=8,
    )
    wls_sbas = solve_gauss_newton_wls(
        sat_ecef_matrix=sat_ecef_mat,
        pseudoranges_m=rho_sbas,
        weights=weights,
        init_ecef_m=true_ecef + np.array([1000.0, -1000.0, 500.0]),
        true_ecef_m=true_ecef,
        max_iter=6,
    )

    standalone_enu_err = ecef_to_enu(wls_standalone["est_ecef_m"], ref_lat_deg, ref_lon_deg, ref_alt_m)
    sbas_enu_err = ecef_to_enu(wls_sbas["est_ecef_m"], ref_lat_deg, ref_lon_deg, ref_alt_m)

    # Fast linearized WLS estimator around true_ecef for 100-epoch Monte Carlo scatter cloud
    h_ecef = wls_sbas["H_ecef"]
    w_mat = np.diag(weights)
    pinv_wls = np.linalg.solve(h_ecef.T @ w_mat @ h_ecef + 1e-10 * np.eye(4), h_ecef.T @ w_mat)
    rot_ecef2enu = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)

    mc_standalone = np.empty((num_mc_runs, 3), dtype=np.float64)
    mc_sbas = np.empty((num_mc_runs, 3), dtype=np.float64)

    for m in range(num_mc_runs):
        noise_vec = rng.normal(0.0, mp_sigmas)
        drho_std = iono_delays + tropo_res + noise_vec
        drho_sb = 0.12 * iono_delays + 0.25 * tropo_res + 0.60 * noise_vec

        dx_std = pinv_wls @ drho_std
        dx_sb = pinv_wls @ drho_sb

        mc_standalone[m] = rot_ecef2enu @ dx_std[:3]
        mc_sbas[m] = rot_ecef2enu @ dx_sb[:3]

    std_h_rms = float(np.sqrt(np.mean(mc_standalone[:, 0] ** 2 + mc_standalone[:, 1] ** 2)))
    std_v_rms = float(np.sqrt(np.mean(mc_standalone[:, 2] ** 2)))
    sbas_h_rms = float(np.sqrt(np.mean(mc_sbas[:, 0] ** 2 + mc_sbas[:, 1] ** 2)))
    sbas_v_rms = float(np.sqrt(np.mean(mc_sbas[:, 2] ** 2)))

    return WlsPvtResult(
        all_satellites=all_sats,
        active_satellites=active_sats,
        dop=dop,
        iter_step_norms_m=wls_standalone["step_norms_m"],
        iter_pos_errors_m=wls_standalone["pos_errors_m"],
        standalone_enu_err_m=standalone_enu_err,
        sbas_enu_err_m=sbas_enu_err,
        est_clock_bias_m=float(wls_standalone["est_clock_bias_m"]),
        true_clock_bias_m=true_clock_bias_m,
        mc_standalone_enu_m=mc_standalone,
        mc_sbas_enu_m=mc_sbas,
        standalone_h_rms_m=std_h_rms,
        standalone_v_rms_m=std_v_rms,
        sbas_h_rms_m=sbas_h_rms,
        sbas_v_rms_m=sbas_v_rms,
    )
