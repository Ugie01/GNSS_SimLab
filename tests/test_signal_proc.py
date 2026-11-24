"""Unit tests for core/signal_proc.py validating GPS C/A Gold Code, PCPS 2D FFT acquisition, and DLL/PLL tracking."""

import numpy as np
import pytest

from core.signal_proc import (
    CA_CODE_LENGTH,
    compute_dll_s_curve,
    compute_prn_correlations,
    generate_ca_code,
    run_2d_fft_acquisition,
    simulate_dll_pll_tracking,
)


def test_ca_code_length_and_first_10_chips_prn1():
    """
    Verify GPS L1 C/A Gold Code for PRN 1 matches the IS-GPS-200 specification:
    First 10 chips of PRN 1 in octal 1440 -> binary 1,1,0,0,1,0,0,0,0,0 -> BPSK [-1,-1,+1,+1,-1,+1,+1,+1,+1,+1].
    """
    code1 = generate_ca_code(1)
    assert len(code1) == CA_CODE_LENGTH
    assert set(np.unique(code1)) == {-1.0, 1.0}

    expected_first_10_bits = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    expected_first_10_bpsk = 1.0 - 2.0 * expected_first_10_bits
    assert np.allclose(code1[:10], expected_first_10_bpsk)


def test_gold_code_auto_and_cross_correlation_bounds():
    """
    Verify Gold Code correlation properties:
    - Normalized auto-correlation peak at lag 0 is 1.0 (1023/1023).
    - Cross-correlation between two distinct PRNs is bounded by 65/1023 (~0.06354).
    """
    res = compute_prn_correlations(prn_main=1, prn_other=2, samples_per_chip=1, max_lag_chips=50)
    assert pytest.approx(res["auto_peak"], abs=1e-6) == 1.0
    assert res["cross_max_abs"] <= (65.0 / 1023.0) + 1e-6


def test_2d_fft_pcps_acquisition_accuracy():
    """Verify 2D FFT PCPS acquisition accurately recovers true code delay and Doppler at high C/N0."""
    true_delay = 412.5
    true_dopp = -1750.0
    acq = run_2d_fft_acquisition(
        prn=7,
        true_code_delay_chips=true_delay,
        true_doppler_hz=true_dopp,
        cn0_dbhz=48.0,
        coh_ms=2,
        doppler_max_hz=4000.0,
        doppler_step_hz=250.0,
    )
    assert acq.is_detected is True
    assert abs(acq.est_doppler_hz - true_dopp) <= 250.0
    assert abs(acq.est_code_phase_chips - true_delay) <= 1.0


def test_dll_s_curve_symmetry_and_zero_crossing():
    """Verify DLL S-curve passes through 0 at 0 error and is odd-symmetric: D(-e) = -D(e)."""
    s_data = compute_dll_s_curve(correlator_spacing_chips=0.5, num_points=201)
    mid = len(s_data["error_chips"]) // 2
    assert pytest.approx(s_data["s_curve_norm"][mid], abs=1e-7) == 0.0
    assert np.allclose(s_data["s_curve_norm"], -s_data["s_curve_norm"][::-1], atol=1e-6)


def test_dll_pll_closed_loop_convergence():
    """Verify 2nd-order Costas PLL and DLL converge from initial frequency and code errors at 45 dB-Hz."""
    sim = simulate_dll_pll_tracking(
        cn0_dbhz=45.0,
        pll_bw_hz=20.0,
        dll_bw_hz=3.0,
        correlator_spacing_chips=0.5,
        initial_freq_err_hz=30.0,
        initial_code_err_chips=0.3,
        duration_ms=250,
    )
    assert sim["pll_locked"] is True
    # Steady-state code error in final 50 ms should be much smaller than initial 0.3 chips
    steady_code_err = float(np.mean(np.abs(sim["code_err_chips"][-50:])))
    assert steady_code_err < 0.08
    # Steady-state Q_P energy should be much smaller than I_P energy once Costas PLL locks
    steady_ip_power = float(np.mean(sim["ip"][-80:] ** 2))
    steady_qp_power = float(np.mean(sim["qp"][-80:] ** 2))
    assert steady_ip_power > 4.0 * steady_qp_power


def test_signal_proc_edge_cases_and_invalid_prn():
    """Verify invalid PRN raises ValueError and same-PRN correlation sets is_same_prn=True."""
    with pytest.raises(ValueError):
        generate_ca_code(0)
    with pytest.raises(ValueError):
        generate_ca_code(33)

    same_res = compute_prn_correlations(prn_main=5, prn_other=5, samples_per_chip=2, max_lag_chips=10)
    assert same_res["is_same_prn"] is True
    assert pytest.approx(same_res["cross_max_abs"], abs=1e-6) == 1.0

