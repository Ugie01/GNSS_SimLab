"""Unit tests for core/egi_raim_spoof.py validating RAIM/FDE fault exclusion, SQM spoofing detection, Lever-Arm/Latency, and STAP."""

import numpy as np

from core.egi_raim_spoof import run_egi_raim_spoof_simulation


def test_raim_fault_detection_and_fde_exclusion():
    """Verify RAIM detects a 50m single-satellite pseudorange fault and FDE restores accuracy (<3m)."""
    res = run_egi_raim_spoof_simulation(fault_prn_bias_m=50.0)
    assert res.raim_alarm_triggered is True
    assert res.detected_fault_prn == res.fault_prn
    assert res.pos_err_before_fde_m > 10.0
    assert res.pos_err_after_fde_m < 3.0


def test_raim_no_false_alarm_when_bias_zero():
    """Verify RAIM does not trigger a false alarm under nominal 0m fault bias."""
    res = run_egi_raim_spoof_simulation(fault_prn_bias_m=0.0)
    assert res.raim_alarm_triggered is False


def test_sqm_spoofing_asymmetry_detection():
    """Verify SQM Delta metric detects correlation peak distortion during intermediate pull-off (0.25~0.55 chips)."""
    res_spoof = run_egi_raim_spoof_simulation(spoofer_delay_chips=0.35, spoofer_power_ratio_db=4.0)
    assert res_spoof.spoof_alarm_triggered is True

    res_clean = run_egi_raim_spoof_simulation(spoofer_delay_chips=0.35, spoofer_power_ratio_db=0.0)
    assert res_clean.spoof_alarm_triggered is False


def test_lever_arm_latency_and_stap_broadband_improvement():
    """Verify EGI Lever-Arm/Time-Tag compensation and STAP multi-tap broadband nulling improvement."""
    res = run_egi_raim_spoof_simulation(lever_arm_m=1.5, latency_ms=50.0, stap_taps=5)
    assert np.max(res.uncomp_total_error_m) > 4.0
    assert np.max(res.comp_residual_error_m) < 0.25
    # Across off-center frequencies, 5-tap STAP achieves much deeper nulling than 1-tap SAP
    assert np.mean(res.stap_broadband_null_db) < np.mean(res.sap_broadband_null_db) - 15.0
