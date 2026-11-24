"""
Pure NumPy/SciPy mathematical and signal processing engine for Module 2:
- Authentic GPS L1 C/A Gold Code generator (IS-GPS-200 G1/G2 10-stage LFSR for PRN 1..32)
- Circular Auto-Correlation & Cross-Correlation analysis
- 2D FFT Parallel Code Phase Search (PCPS) Signal Acquisition under configurable C/N0
- DLL (Early-Minus-Late S-Curve) and Costas PLL / DLL Closed-Loop Tracking Simulation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ==============================================================================
# IS-GPS-200 GPS L1 C/A PRN G2 Phase Selector Tap Assignments (1-indexed stages)
# ==============================================================================
GPS_CA_G2_TAPS: Dict[int, Tuple[int, int]] = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10),  10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7),  14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6),  20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7),  26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8),  32: (4, 9),
}

CA_CODE_LENGTH: int = 1023
CA_CHIP_RATE_HZ: float = 1.023e6
CA_CODE_PERIOD_SEC: float = 1e-3  # 1 ms


@dataclass(frozen=True)
class AcquisitionResult:
    """Output container for 2D FFT PCPS Acquisition."""

    doppler_bins_hz: np.ndarray
    code_phase_chips: np.ndarray
    corr_grid: np.ndarray  # Shape: (num_doppler_bins, num_samples)
    est_doppler_hz: float
    est_code_phase_chips: float
    peak_metric: float
    threshold_metric: float
    is_detected: bool
    doppler_cut: np.ndarray
    code_cut: np.ndarray


def generate_ca_code(prn: int) -> np.ndarray:
    """
    Generate the authentic 1023-chip GPS L1 C/A Gold Code sequence for PRN (1..32)
    in bipolar BPSK format (+1.0 for chip 0, -1.0 for chip 1, or vice versa: +/-1.0).

    G1 polynomial: 1 + X^3 + X^10
    G2 polynomial: 1 + X^2 + X^3 + X^6 + X^8 + X^9 + X^10
    """
    if prn not in GPS_CA_G2_TAPS:
        raise ValueError(f"Invalid GPS PRN {prn}. Must be between 1 and 32.")

    t1, t2 = GPS_CA_G2_TAPS[prn]
    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    ca_bits = np.empty(CA_CODE_LENGTH, dtype=np.int8)

    for i in range(CA_CODE_LENGTH):
        g1_out = g1[9]
        g2_out = g2[t1 - 1] ^ g2[t2 - 1]
        ca_bits[i] = g1_out ^ g2_out

        # Shift feedback
        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        g1[1:] = g1[:-1]
        g1[0] = g1_fb

        g2[1:] = g2[:-1]
        g2[0] = g2_fb

    # Convert {0, 1} bits to BPSK {+1.0, -1.0}
    return 1.0 - 2.0 * ca_bits.astype(np.float64)


def sample_ca_code(prn: int, num_samples: int) -> np.ndarray:
    """Sample a 1-ms 1023-chip C/A code into `num_samples` discrete samples."""
    chips = generate_ca_code(prn)
    sample_indices = (np.arange(num_samples) * CA_CODE_LENGTH // num_samples) % CA_CODE_LENGTH
    return chips[sample_indices]


def compute_prn_correlations(
    prn_main: int = 1,
    prn_other: int = 2,
    samples_per_chip: int = 4,
    max_lag_chips: int = 25,
) -> Dict[str, np.ndarray | float | bool]:
    """
    Compute circular auto-correlation of `prn_main` and cross-correlation between
    `prn_main` and `prn_other` using FFT, returning a zoomed window around 0 lag
    as well as full-period correlation statistics.
    """
    samples_per_chip = max(int(samples_per_chip), 1)
    max_lag_chips = max(int(max_lag_chips), 1)
    n_total = CA_CODE_LENGTH * samples_per_chip
    code1 = sample_ca_code(prn_main, n_total)
    code2 = sample_ca_code(prn_other, n_total)

    f1 = np.fft.fft(code1)
    f2 = np.fft.fft(code2)

    auto_full = np.real(np.fft.ifft(f1 * np.conj(f1))) / n_total
    cross_full = np.real(np.fft.ifft(f1 * np.conj(f2))) / n_total

    auto_shifted = np.fft.fftshift(auto_full)
    cross_shifted = np.fft.fftshift(cross_full)
    lags_chips_full = (np.arange(n_total) - (n_total // 2)) / float(samples_per_chip)

    mask = np.abs(lags_chips_full) <= max_lag_chips
    return {
        "lags_chips": lags_chips_full[mask],
        "auto_corr": auto_shifted[mask],
        "cross_corr": cross_shifted[mask],
        "auto_peak": float(np.max(auto_full)),
        "cross_max_abs": float(np.max(np.abs(cross_full))),
        "is_same_prn": bool(prn_main == prn_other),
    }


def run_2d_fft_acquisition(
    prn: int = 1,
    true_code_delay_chips: float = 340.0,
    true_doppler_hz: float = 1500.0,
    cn0_dbhz: float = 44.0,
    coh_ms: int = 1,
    doppler_max_hz: float = 5000.0,
    doppler_step_hz: float = 250.0,
    samples_per_ms: int = 2046,
    seed: int = 101,
) -> AcquisitionResult:
    """
    Simulate baseband GPS L1 C/A signal with AWGN at specified C/N0 [dB-Hz] and execute
    Parallel Code Phase Search (PCPS) 2D FFT acquisition over Doppler and Code Phase.

    Post-correlation SNR scales linearly with coherent integration time `T_coh = coh_ms * 1 ms`
    and `10^(cn0_dbhz / 10)`.
    """
    rng = np.random.default_rng(seed)
    fs = samples_per_ms / CA_CODE_PERIOD_SEC  # e.g. 2.046 MHz (2 samples/chip)

    # Local replica code for 1 ms
    local_code = sample_ca_code(prn, samples_per_ms)
    local_code_fft_conj = np.conj(np.fft.fft(local_code))

    # Received signal with exact fractional code phase delay and Doppler shift
    chip_axis = (np.arange(samples_per_ms, dtype=np.float64) * CA_CODE_LENGTH / samples_per_ms)
    delayed_chip_idx = np.floor((chip_axis - true_code_delay_chips) % CA_CODE_LENGTH).astype(int)
    raw_chips = generate_ca_code(prn)
    rx_code = raw_chips[delayed_chip_idx]

    t_axis = np.arange(samples_per_ms, dtype=np.float64) / fs
    carrier = np.exp(1j * 2.0 * np.pi * true_doppler_hz * t_axis)

    # Complex baseband noise power scaling for C/N0:
    # After correlating `samples_per_ms` samples over `T_coh = coh_ms * 1e-3` s:
    # Post-correlation SNR = 2 * T_coh * 10^(CN0/10) for real, or T_coh * 10^(CN0/10) for complex power
    t_coh_sec = float(coh_ms) * CA_CODE_PERIOD_SEC
    cn0_lin = 10.0 ** (cn0_dbhz / 10.0)
    # Per-sample complex noise variance sigma_n^2 = fs / cn0_lin (scaled by 1/coh_ms for coherent averaging)
    noise_std_per_comp = np.sqrt(fs / (2.0 * cn0_lin * max(coh_ms, 1)))
    noise = rng.normal(0.0, noise_std_per_comp, size=samples_per_ms) + 1j * rng.normal(
        0.0, noise_std_per_comp, size=samples_per_ms
    )

    rx_signal = rx_code * carrier + noise

    # 2D PCPS Search across Doppler bins
    doppler_bins = np.arange(-doppler_max_hz, doppler_max_hz + 0.5 * doppler_step_hz, doppler_step_hz)
    num_dopp = len(doppler_bins)
    corr_grid = np.empty((num_dopp, samples_per_ms), dtype=np.float64)

    for d_idx, f_d in enumerate(doppler_bins):
        lo_wipeoff = np.exp(-1j * 2.0 * np.pi * f_d * t_axis)
        mixed_fft = np.fft.fft(rx_signal * lo_wipeoff)
        corr_time = np.fft.ifft(mixed_fft * local_code_fft_conj) / samples_per_ms
        corr_grid[d_idx, :] = np.abs(corr_time) ** 2

    # Normalize correlation power grid by mean off-peak noise floor so metric = Peak-to-Noise Ratio
    flat_sorted = np.sort(corr_grid.ravel())
    noise_floor = float(np.mean(flat_sorted[: int(0.85 * len(flat_sorted))]))
    if noise_floor > 1e-15:
        corr_grid_norm = corr_grid / noise_floor
    else:
        corr_grid_norm = corr_grid

    peak_flat_idx = int(np.argmax(corr_grid_norm))
    peak_d_idx, peak_c_idx = np.unravel_index(peak_flat_idx, corr_grid_norm.shape)

    code_phase_axis = np.arange(samples_per_ms, dtype=np.float64) * CA_CODE_LENGTH / samples_per_ms
    est_doppler = float(doppler_bins[peak_d_idx])
    est_code_phase = float(code_phase_axis[peak_c_idx])
    peak_metric = float(corr_grid_norm[peak_d_idx, peak_c_idx])

    # Standard CFAR acquisition threshold (Peak-to-mean noise ratio ~ 12.0 => ~10.8 dB)
    threshold_metric = 12.0
    is_detected = bool(peak_metric >= threshold_metric)

    return AcquisitionResult(
        doppler_bins_hz=doppler_bins,
        code_phase_chips=code_phase_axis,
        corr_grid=corr_grid_norm,
        est_doppler_hz=est_doppler,
        est_code_phase_chips=est_code_phase,
        peak_metric=peak_metric,
        threshold_metric=threshold_metric,
        is_detected=is_detected,
        doppler_cut=corr_grid_norm[:, peak_c_idx],
        code_cut=corr_grid_norm[peak_d_idx, :],
    )


def compute_dll_s_curve(
    correlator_spacing_chips: float = 0.5,
    num_points: int = 401,
) -> Dict[str, np.ndarray | float]:
    """
    Compute the analytical DLL Early-Prompt-Late triangle auto-correlation envelopes
    R_E(e), R_P(e), R_L(e) and Normalized Early-Minus-Late Power/Envelope discriminator
    S-curve D(e) = (R_E - R_L) / (R_E + R_L + eps) for tracking error e in [-1.5, +1.5] chips.
    """
    correlator_spacing_chips = float(np.clip(correlator_spacing_chips, 0.02, 1.98))
    num_points = max(int(num_points), 21)
    err_chips = np.linspace(-1.5, 1.5, num_points)
    half_d = 0.5 * correlator_spacing_chips

    def tri_corr(tau: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, 1.0 - np.abs(tau))

    r_prompt = tri_corr(err_chips)
    r_early = tri_corr(err_chips - half_d)
    r_late = tri_corr(err_chips + half_d)

    # Non-coherent Early-minus-Late envelope discriminator (scaled by (2 - d)/2 so slope=1 near 0)
    denom = r_early + r_late + 1e-9
    disc_norm = (1.0 - half_d) * (r_early - r_late) / denom
    # Outside correlation support where both E and L are zero, discriminator drops to 0
    support_mask = (r_early + r_late) > 1e-4
    disc_norm = np.where(support_mask, disc_norm, 0.0)

    # Also compute coherent Early-minus-Late: (R_E - R_L) / 2
    disc_coherent = 0.5 * (r_early - r_late)

    return {
        "error_chips": err_chips,
        "r_early": r_early,
        "r_prompt": r_prompt,
        "r_late": r_late,
        "s_curve_norm": disc_norm,
        "s_curve_coherent": disc_coherent,
        "linear_pull_in_chips": float(half_d),
    }


def simulate_dll_pll_tracking(
    cn0_dbhz: float = 42.0,
    pll_bw_hz: float = 18.0,
    dll_bw_hz: float = 2.0,
    correlator_spacing_chips: float = 0.5,
    initial_freq_err_hz: float = 35.0,
    initial_code_err_chips: float = 0.35,
    duration_ms: int = 300,
    seed: int = 77,
) -> Dict[str, np.ndarray | float | bool]:
    """
    Simulate 1-ms epoch-by-epoch closed-loop 2nd-order Costas PLL and 2nd-order DLL
    tracking loops, including 50 bps BPSK navigation data bit flips every 20 ms.
    """
    rng = np.random.default_rng(seed)
    correlator_spacing_chips = float(np.clip(correlator_spacing_chips, 0.02, 1.98))
    pll_bw_hz = max(float(pll_bw_hz), 0.5)
    dll_bw_hz = max(float(dll_bw_hz), 0.05)
    duration_ms = max(int(duration_ms), 20)
    dt = CA_CODE_PERIOD_SEC  # 1 ms loop update rate
    t_ms = np.arange(duration_ms, dtype=np.float64)

    cn0_lin = 10.0 ** (cn0_dbhz / 10.0)
    # Prompt post-correlation SNR per 1-ms coherent integration: SNR_1ms = 2 * C/N0 * T_coh
    snr_post = 2.0 * cn0_lin * dt
    noise_std = 1.0 / np.sqrt(max(snr_post, 1e-6))

    # 2nd-order loop filter parameters (damping ratio zeta = 0.707)
    zeta = 0.70710678
    wn_pll = (8.0 * zeta * pll_bw_hz) / (4.0 * zeta**2 + 1.0)
    wn_dll = (8.0 * zeta * dll_bw_hz) / (4.0 * zeta**2 + 1.0)
    fll_gain = min(0.35, 0.02 * pll_bw_hz)  # FLL-assist gain for rapid frequency pull-in

    # Generate 50 bps navigation data bits (+1 or -1, changing every 20 ms)
    num_nav_bits = (duration_ms // 20) + 2
    nav_bits_seq = rng.choice([-1.0, 1.0], size=num_nav_bits)
    # Ensure alternating pattern in first few bits for clear visual intuition
    if len(nav_bits_seq) >= 6:
        nav_bits_seq[:6] = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

    # State variables
    true_phase_rad = 0.5  # Initial phase error [rad]
    true_freq_hz = initial_freq_err_hz
    nco_phase_rad = 0.0
    nco_freq_hz = 0.0
    prev_ip = 0.0
    prev_qp = 0.0
    prev_nav_idx = 0

    code_err_chips = initial_code_err_chips
    half_d = 0.5 * correlator_spacing_chips
    dll_vel_chips_s = 0.0

    ip_hist = np.empty(duration_ms, dtype=np.float64)
    qp_hist = np.empty(duration_ms, dtype=np.float64)
    phase_err_deg_hist = np.empty(duration_ms, dtype=np.float64)
    freq_err_hz_hist = np.empty(duration_ms, dtype=np.float64)
    code_err_chips_hist = np.empty(duration_ms, dtype=np.float64)

    for k in range(duration_ms):
        nav_idx = k // 20
        nav_bit = nav_bits_seq[nav_idx]
        phase_err = true_phase_rad - nco_phase_rad

        # Prompt amplitude degraded by residual code error
        r_p = max(0.0, 1.0 - abs(code_err_chips))
        r_e = max(0.0, 1.0 - abs(code_err_chips - half_d))
        r_l = max(0.0, 1.0 - abs(code_err_chips + half_d))

        # Prompt I_P, Q_P correlator outputs
        ip = nav_bit * r_p * np.cos(phase_err) + rng.normal(0.0, noise_std)
        qp = nav_bit * r_p * np.sin(phase_err) + rng.normal(0.0, noise_std)

        # Early and Late envelope magnitudes
        e_noise_std = noise_std * np.sqrt(max(0.2, correlator_spacing_chips))
        env_e = np.hypot(r_e * np.cos(phase_err) + rng.normal(0.0, e_noise_std), r_e * np.sin(phase_err))
        env_l = np.hypot(r_l * np.cos(phase_err) + rng.normal(0.0, e_noise_std), r_l * np.sin(phase_err))

        ip_hist[k] = ip
        qp_hist[k] = qp
        # Wrap phase error to [-90, +90] deg (Costas 180-deg ambiguity domain)
        wrapped_err_rad = np.arctan(np.tan(phase_err))
        phase_err_deg_hist[k] = np.rad2deg(wrapped_err_rad)
        freq_err_hz_hist[k] = true_freq_hz - nco_freq_hz
        code_err_chips_hist[k] = code_err_chips

        # 1. Costas PLL Discriminator: atan(Q_P / I_P), insensitive to 180° nav bit flips
        denom_ip = ip if abs(ip) > 1e-6 else (1e-6 if ip >= 0.0 else -1e-6)
        pll_disc_rad = np.arctan(qp / denom_ip)

        # FLL-assist cross-product discriminator within same navigation bit period
        fll_disc_hz = 0.0
        if k > 0 and nav_idx == prev_nav_idx:
            cross = prev_ip * qp - prev_qp * ip
            dot = prev_ip * ip + prev_qp * qp
            if abs(dot) > 1e-6:
                fll_disc_hz = np.arctan(cross / dot) / (2.0 * np.pi * dt)
        prev_ip, prev_qp, prev_nav_idx = ip, qp, nav_idx

        # 2nd-order FLL-assisted PLL loop filter update
        nco_freq_hz += ((wn_pll ** 2) * pll_disc_rad * dt / (2.0 * np.pi)) + fll_gain * fll_disc_hz
        phase_rate_rad_s = 2.0 * np.pi * nco_freq_hz + 2.0 * zeta * wn_pll * pll_disc_rad
        nco_phase_rad += phase_rate_rad_s * dt
        true_phase_rad += 2.0 * np.pi * true_freq_hz * dt

        # 2. Normalized Early-Minus-Late DLL Discriminator
        dll_disc_chips = (1.0 - half_d) * (env_e - env_l) / (env_e + env_l + 1e-6)
        dll_vel_chips_s += (wn_dll ** 2) * dll_disc_chips * dt
        code_rate_chips_s = dll_vel_chips_s + 2.0 * zeta * wn_dll * dll_disc_chips
        code_err_chips -= code_rate_chips_s * dt

    # Theoretical thermal phase jitter (1-sigma degrees) with squaring loss
    squaring_loss = 1.0 + 1.0 / (2.0 * cn0_lin * dt)
    sigma_pll_deg = float(np.rad2deg(np.sqrt((pll_bw_hz / cn0_lin) * squaring_loss)))

    # Theoretical thermal DLL jitter (1-sigma meters, 1 chip ≈ 293.05 m)
    sigma_dll_chips = float(
        np.sqrt(
            (dll_bw_hz / (2.0 * cn0_lin))
            * correlator_spacing_chips
            * (1.0 + 2.0 / (cn0_lin * dt * (2.0 - correlator_spacing_chips)))
        )
    )
    sigma_dll_m = sigma_dll_chips * 293.05

    return {
        "t_ms": t_ms,
        "ip": ip_hist,
        "qp": qp_hist,
        "phase_err_deg": phase_err_deg_hist,
        "freq_err_hz": freq_err_hz_hist,
        "code_err_chips": code_err_chips_hist,
        "code_err_m": code_err_chips_hist * 293.05,
        "sigma_pll_deg": sigma_pll_deg,
        "sigma_dll_m": sigma_dll_m,
        "pll_locked": bool(sigma_pll_deg < 15.0),
    }
