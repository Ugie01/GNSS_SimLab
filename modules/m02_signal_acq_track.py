"""
Module 02: CDMA PRN Gold Code Correlation, 2D FFT Parallel Code Phase Search (PCPS)
Acquisition, and DLL / Costas PLL Closed-Loop Signal Tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.signal_proc import (
    compute_dll_s_curve,
    compute_prn_correlations,
    run_2d_fft_acquisition,
    simulate_dll_pll_tracking,
)
from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Module02SignalAcqTrack(BaseLabModule):
    """Module 2: GPS L1 C/A PRN Correlation, 2D FFT PCPS Acquisition, and DLL/PLL Tracking."""

    module_id = "m02_signal_acq_track"
    module_title = "M02. PRN 골드코드 상관 · 2D FFT 획득 · DLL/PLL 추적"
    module_subtitle = "CDMA 자기/상호상관 특성과 병렬 코드위상 탐색(PCPS), Early-Late S-Curve 및 코스타스 루프 추적"
    module_badge = "MOD 02"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.run_simulation()

    def build_parameter_widget(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        group_style = """
            QGroupBox {
                font-weight: 700;
                font-size: 13px;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: #f8fafc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

        # 1. CDMA PRN & Signal Quality Group
        g_prn = QGroupBox("1. 위성 PRN 번호 · 수신 신호 품질 (C/N0)")
        g_prn.setStyleSheet(group_style)
        f_prn = QFormLayout(g_prn)
        f_prn.setSpacing(7)

        self.spin_prn_main = QSpinBox()
        self.spin_prn_main.setRange(1, 32)
        self.spin_prn_main.setValue(1)
        self.spin_prn_main.setPrefix("PRN #")
        self.spin_prn_main.valueChanged.connect(lambda _: self.run_simulation())
        f_prn.addRow("목표 위성 (Target):", self.spin_prn_main)

        self.spin_prn_other = QSpinBox()
        self.spin_prn_other.setRange(1, 32)
        self.spin_prn_other.setValue(7)
        self.spin_prn_other.setPrefix("PRN #")
        self.spin_prn_other.valueChanged.connect(lambda _: self.run_simulation())
        f_prn.addRow("간섭 타 위성 (Other):", self.spin_prn_other)

        self.slider_cn0 = QSlider(Qt.Orientation.Horizontal)
        self.slider_cn0.setRange(26, 54)  # 26 to 54 dB-Hz
        self.slider_cn0.setValue(46)
        self.lbl_cn0 = QLabel("46 dB-Hz")
        self.lbl_cn0.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_cn0.valueChanged.connect(self._on_cn0_changed)

        cn0_row = QHBoxLayout()
        cn0_row.addWidget(self.slider_cn0, stretch=1)
        cn0_row.addWidget(self.lbl_cn0)
        f_prn.addRow("신호 품질 (C/N0):", cn0_row)
        layout.addWidget(g_prn)

        # 2. 2D FFT PCPS Acquisition Parameters
        g_acq = QGroupBox("2. 2D FFT 신호 획득 (PCPS Acquisition)")
        g_acq.setStyleSheet(group_style)
        f_acq = QFormLayout(g_acq)
        f_acq.setSpacing(7)

        self.spin_code_delay = QDoubleSpinBox()
        self.spin_code_delay.setRange(0.0, 1022.5)
        self.spin_code_delay.setDecimals(1)
        self.spin_code_delay.setSingleStep(25.0)
        self.spin_code_delay.setValue(340.0)
        self.spin_code_delay.setSuffix(" chips")
        self.spin_code_delay.valueChanged.connect(lambda _: self.run_simulation())
        f_acq.addRow("실제 코드 위상 지연:", self.spin_code_delay)

        self.spin_doppler = QDoubleSpinBox()
        self.spin_doppler.setRange(-4500.0, 4500.0)
        self.spin_doppler.setDecimals(0)
        self.spin_doppler.setSingleStep(250.0)
        self.spin_doppler.setValue(1250.0)
        self.spin_doppler.setSuffix(" Hz")
        self.spin_doppler.valueChanged.connect(lambda _: self.run_simulation())
        f_acq.addRow("실제 도플러 편이:", self.spin_doppler)

        self.combo_coh_ms = QComboBox()
        for ms in [1, 2, 4, 8, 10]:
            self.combo_coh_ms.addItem(f"{ms} ms (적분 이득 +{10*np.log10(ms):.1f} dB)", userData=ms)
        self.combo_coh_ms.currentIndexChanged.connect(lambda _: self.run_simulation())
        f_acq.addRow("동기 적분 시간 (T_coh):", self.combo_coh_ms)
        layout.addWidget(g_acq)

        # 3. DLL & Costas PLL Tracking Loop Parameters
        g_trk = QGroupBox("3. DLL / Costas PLL 추적 루프 설계")
        g_trk.setStyleSheet(group_style)
        f_trk = QFormLayout(g_trk)
        f_trk.setSpacing(7)

        self.slider_spacing = QSlider(Qt.Orientation.Horizontal)
        self.slider_spacing.setRange(1, 10)  # 0.1 to 1.0 chips
        self.slider_spacing.setValue(5)      # 0.5 chips
        self.lbl_spacing = QLabel("0.5 chip")
        self.lbl_spacing.setStyleSheet("font-weight:700; color:#16a34a;")
        self.slider_spacing.valueChanged.connect(self._on_spacing_changed)

        sp_row = QHBoxLayout()
        sp_row.addWidget(self.slider_spacing, stretch=1)
        sp_row.addWidget(self.lbl_spacing)
        f_trk.addRow("E-L 상관기 간격 (d):", sp_row)

        self.slider_pll_bw = QSlider(Qt.Orientation.Horizontal)
        self.slider_pll_bw.setRange(5, 45)  # 5 to 45 Hz
        self.slider_pll_bw.setValue(18)
        self.lbl_pll_bw = QLabel("18 Hz")
        self.lbl_pll_bw.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_pll_bw.valueChanged.connect(self._on_pll_bw_changed)

        p_row = QHBoxLayout()
        p_row.addWidget(self.slider_pll_bw, stretch=1)
        p_row.addWidget(self.lbl_pll_bw)
        f_trk.addRow("PLL 루프 대역폭:", p_row)

        self.slider_dll_bw = QSlider(Qt.Orientation.Horizontal)
        self.slider_dll_bw.setRange(5, 60)  # 0.5 to 6.0 Hz
        self.slider_dll_bw.setValue(20)     # 2.0 Hz
        self.lbl_dll_bw = QLabel("2.0 Hz")
        self.lbl_dll_bw.setStyleSheet("font-weight:700; color:#d97706;")
        self.slider_dll_bw.valueChanged.connect(self._on_dll_bw_changed)

        d_row = QHBoxLayout()
        d_row.addWidget(self.slider_dll_bw, stretch=1)
        d_row.addWidget(self.lbl_dll_bw)
        f_trk.addRow("DLL 루프 대역폭:", d_row)

        self.spin_init_ferr = QDoubleSpinBox()
        self.spin_init_ferr.setRange(-80.0, 80.0)
        self.spin_init_ferr.setDecimals(1)
        self.spin_init_ferr.setSingleStep(5.0)
        self.spin_init_ferr.setValue(25.0)
        self.spin_init_ferr.setSuffix(" Hz")
        self.spin_init_ferr.valueChanged.connect(lambda _: self.run_simulation())
        f_trk.addRow("초기 주파수 풀인 오차:", self.spin_init_ferr)

        layout.addWidget(g_trk)
        return container

    def _on_cn0_changed(self, val: int) -> None:
        self.lbl_cn0.setText(f"{val} dB-Hz")
        self.run_simulation()

    def _on_spacing_changed(self, val: int) -> None:
        self.lbl_spacing.setText(f"{val / 10.0:.1f} chip")
        self.run_simulation()

    def _on_pll_bw_changed(self, val: int) -> None:
        self.lbl_pll_bw.setText(f"{val} Hz")
        self.run_simulation()

    def _on_dll_bw_changed(self, val: int) -> None:
        self.lbl_dll_bw.setText(f"{val / 10.0:.1f} Hz")
        self.run_simulation()

    def build_simulation_Plot_widget(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        cards_frame = QFrame()
        cards_frame.setStyleSheet("QFrame { background-color: #0f172a; border-radius: 8px; }")
        grid = QGridLayout(cards_frame)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(18)

        self.lbl_cdma_readout = QLabel()
        self.lbl_cdma_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#e2e8f0; border:none;"
        )
        self.lbl_acq_readout = QLabel()
        self.lbl_acq_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#7dd3fc; border:none;"
        )
        self.lbl_trk_readout = QLabel()
        self.lbl_trk_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#86efac; border:none;"
        )

        grid.addWidget(self.lbl_cdma_readout, 0, 0)
        grid.addWidget(self.lbl_acq_readout, 0, 1)
        grid.addWidget(self.lbl_trk_readout, 0, 2)
        vbox.addWidget(cards_frame)

        self.fig = Figure(figsize=(10.2, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)
        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        widgets = [
            self.spin_prn_main,
            self.spin_prn_other,
            self.slider_cn0,
            self.spin_code_delay,
            self.spin_doppler,
            self.combo_coh_ms,
            self.slider_spacing,
            self.slider_pll_bw,
            self.slider_dll_bw,
            self.spin_init_ferr,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.spin_prn_main.setValue(int(preset_data.get("prn_main", 1)))
        self.spin_prn_other.setValue(int(preset_data.get("prn_other", 7)))

        cn0_val = int(preset_data.get("cn0_dbhz", 46))
        self.slider_cn0.setValue(cn0_val)
        self.lbl_cn0.setText(f"{cn0_val} dB-Hz")

        self.spin_code_delay.setValue(float(preset_data.get("true_code_delay_chips", 340.0)))
        self.spin_doppler.setValue(float(preset_data.get("true_doppler_hz", 1250.0)))

        coh_ms = int(preset_data.get("coh_ms", 1))
        idx = self.combo_coh_ms.findData(coh_ms)
        if idx >= 0:
            self.combo_coh_ms.setCurrentIndex(idx)

        sp_val = int(round(float(preset_data.get("correlator_spacing_chips", 0.5)) * 10))
        self.slider_spacing.setValue(sp_val)
        self.lbl_spacing.setText(f"{sp_val / 10.0:.1f} chip")

        pll_val = int(preset_data.get("pll_bw_hz", 18))
        self.slider_pll_bw.setValue(pll_val)
        self.lbl_pll_bw.setText(f"{pll_val} Hz")

        dll_val = int(round(float(preset_data.get("dll_bw_hz", 2.0)) * 10))
        self.slider_dll_bw.setValue(dll_val)
        self.lbl_dll_bw.setText(f"{dll_val / 10.0:.1f} Hz")

        self.spin_init_ferr.setValue(float(preset_data.get("init_freq_err_hz", 25.0)))

        for w in widgets:
            w.blockSignals(False)

    def run_simulation(self) -> None:
        prn_main = self.spin_prn_main.value()
        prn_other = self.spin_prn_other.value()
        cn0_dbhz = float(self.slider_cn0.value())
        true_delay = self.spin_code_delay.value()
        true_dopp = self.spin_doppler.value()
        coh_ms = int(self.combo_coh_ms.currentData() or 1)
        spacing_chips = self.slider_spacing.value() / 10.0
        pll_bw = float(self.slider_pll_bw.value())
        dll_bw = self.slider_dll_bw.value() / 10.0
        init_ferr = self.spin_init_ferr.value()

        # 1. CDMA Gold Code Correlation
        corr_res = compute_prn_correlations(prn_main=prn_main, prn_other=prn_other, samples_per_chip=6, max_lag_chips=6)

        # 2. 2D FFT PCPS Acquisition
        acq_res = run_2d_fft_acquisition(
            prn=prn_main,
            true_code_delay_chips=true_delay,
            true_doppler_hz=true_dopp,
            cn0_dbhz=cn0_dbhz,
            coh_ms=coh_ms,
        )

        # 3. DLL S-Curve & Closed-Loop Costas PLL/DLL Tracking
        s_curve = compute_dll_s_curve(correlator_spacing_chips=spacing_chips)
        trk_res = simulate_dll_pll_tracking(
            cn0_dbhz=cn0_dbhz,
            pll_bw_hz=pll_bw,
            dll_bw_hz=dll_bw,
            correlator_spacing_chips=spacing_chips,
            initial_freq_err_hz=init_ferr,
            initial_code_err_chips=0.35,
            duration_ms=240,
        )

        # Update top telemetric readout cards
        if corr_res.get("is_same_prn"):
            cross_line = "동일 PRN 선택 (상호상관=자기상관 1.0)"
        else:
            cross_db = 20.0 * np.log10(max(float(corr_res["cross_max_abs"]), 1e-6))
            cross_line = f"상호상관 최대 {corr_res['cross_max_abs']:.4f} ({cross_db:.1f} dB 억제)"

        self.lbl_cdma_readout.setText(
            f"<div style='font-size:11.5px; color:#1E3A2F; font-weight:800;'>Gold Code 상관 (PRN #{prn_main} vs #{prn_other})</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:800; margin-top:3px;'>"
            f"1023 chips (1 ms) · 확산 이득 +30.10 dB</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>{cross_line}</div>"
        )

        det_color = "#16A34A" if acq_res.is_detected else "#DC2626"
        det_str = "신호 획득 성공 (DETECTED)" if acq_res.is_detected else "획득 실패 (잡음 바닥 매몰)"
        self.lbl_acq_readout.setText(
            f"<div style='font-size:11.5px; color:#B45309; font-weight:800;'>2D FFT PCPS 신호 획득</div>"
            f"<div style='font-size:13.5px; color:{det_color}; font-weight:900; margin-top:3px;'>{det_str}</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"τ={acq_res.est_code_phase_chips:.1f}chip · fd={acq_res.est_doppler_hz:+.0f}Hz · SNR={acq_res.peak_metric:.1f}/{acq_res.threshold_metric:.1f}</div>"
        )

        lock_color = "#1D4ED8" if trk_res["pll_locked"] else "#DC2626"
        lock_str = "LOCK 유지 (<15°)" if trk_res["pll_locked"] else "LOSS OF LOCK 위험 (>15°)"
        self.lbl_trk_readout.setText(
            f"<div style='font-size:11.5px; color:#1D4ED8; font-weight:800;'>DLL / Costas PLL 추적</div>"
            f"<div style='font-size:13.5px; color:{lock_color}; font-weight:900; margin-top:3px;'>{lock_str}</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"PLL 1σ={trk_res['sigma_pll_deg']:.2f}° · DLL 1σ={trk_res['sigma_dll_m']:.2f}m · 풀인 ±{s_curve['linear_pull_in_chips']:.2f}c</div>"
        )

        acq_coach = (
            f"신호 세기(<b>{cn0_dbhz:.0f} dB-Hz</b>, 적분 <b>{coh_ms}ms</b>)가 충분해서 2D 그래프 2 · 3D 탭에서 <b>상관 피크({acq_res.peak_metric:.1f})가 빨간 기준선({acq_res.threshold_metric:.1f}) 위로 솟아올라 위성을 찾았습니다.</b>"
            if acq_res.is_detected
            else f"신호가 약해(<b>{cn0_dbhz:.0f} dB-Hz</b>) 2D 그래프 2에서 피크가 잡음 바닥에 묻혀 위성을 놓쳤습니다. 좌측 <b>[동기 적분 시간(T_coh)]</b>을 4~10ms로 늘려보세요."
        )
        self.set_easy_coach_message(
            f"<b>[사수의 실시간 쉬운 해설]</b> (1) {acq_coach} &nbsp;|&nbsp; "
            f"(2) 2D 그래프 4에서 빨간선(<code>Q_P</code> 위상 오차)이 0으로 수렴하고 <b>파란선(<code>I_P</code>)이 +1과 -1 사이를 오가며 50bps 항법 데이터를 복원</b>합니다."
        )

        # Tab 0: Clean 2D Graphs
        self.fig.clear()
        ax1, ax2, ax3, ax4 = self.allocate_2d_axes(self.fig)
        self._draw_2d_plots(
            ax1, ax2, ax3, ax4,
            corr_res, acq_res, s_curve, trk_res,
            prn_main, prn_other, cn0_dbhz, coh_ms, spacing_chips, pll_bw, dll_bw,
        )
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        # Tab 1: Dedicated 3D PCPS Surface
        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            self._draw_3d_pcps_surface(ax_3d, acq_res, prn_main, cn0_dbhz, coh_ms)
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def _draw_2d_plots(
        self,
        ax1,
        ax2,
        ax3,
        ax4,
        corr_res: Dict[str, Any],
        acq_res,
        s_curve: Dict[str, Any],
        trk_res: Dict[str, Any],
        prn_main: int,
        prn_other: int,
        cn0_dbhz: float,
        coh_ms: int,
        spacing_chips: float,
        pll_bw: float,
        dll_bw: float,
    ) -> None:
        # Subplot 1: PRN Auto-Correlation Triangle vs Cross-Correlation
        if ax1 is not None:
            ax1.plot(
                corr_res["lags_chips"],
                corr_res["auto_corr"],
                color="#2563eb",
                linewidth=2.2,
                label=f"PRN #{prn_main} 자기상관 R(τ)",
            )
            ax1.plot(
                corr_res["lags_chips"],
                corr_res["cross_corr"],
                color="#dc2626",
                linewidth=1.3,
                alpha=0.85,
                label=f"PRN #{prn_main} × PRN #{prn_other} 상호상관",
            )
            ax1.axhline(65.0 / 1023.0, color="#94a3b8", linestyle=":", linewidth=1.1, label="Gold 상한 (±65/1023)")
            ax1.axhline(-65.0 / 1023.0, color="#94a3b8", linestyle=":", linewidth=1.1)
            ax1.set_title("1. C/A Gold Code 자기상관 vs 상호상관 (±6 chips)", fontsize=10.5, fontweight="bold", pad=8)
            ax1.set_xlabel("코드 지연 오차 τ [chips]", fontsize=9.0)
            ax1.set_ylabel("정규화 상관값 R(τ)", fontsize=9.0)
            ax1.set_ylim(-0.22, 1.12)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8.0)

        # Subplot 2: 2D FFT PCPS Acquisition Code-Phase Cut at Best Doppler Bin + Doppler Profile Inset
        if ax2 is not None:
            ax2.plot(
                acq_res.code_phase_chips,
                acq_res.code_cut,
                color="#0284c7",
                linewidth=1.4,
                label=f"코드위상 탐색 ({acq_res.est_doppler_hz:+.0f} Hz)",
            )
            ax2.axhline(
                acq_res.threshold_metric,
                color="#dc2626",
                linestyle="--",
                linewidth=1.5,
                label=f"CFAR 임계값 ({acq_res.threshold_metric:.1f})",
            )
            ax2.scatter(
                [acq_res.est_code_phase_chips],
                [acq_res.peak_metric],
                color="#16a34a" if acq_res.is_detected else "#dc2626",
                s=60,
                zorder=5,
                label=f"피크 ({acq_res.est_code_phase_chips:.1f} chip, SNR={acq_res.peak_metric:.1f})",
            )
            ax2.set_title(
                f"2. PCPS 코드위상/도플러 단면 (C/N0={cn0_dbhz:.0f}dB-Hz)",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax2.set_xlabel("탐색 코드 위상 [0 ~ 1023 chips]", fontsize=9.0)
            ax2.set_ylabel("정규화 상관 전력 (Peak/Noise)", fontsize=9.0)
            ax2.set_xlim(0, 1023)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper left", fontsize=7.8)

            ax2_in = ax2.inset_axes([0.62, 0.45, 0.35, 0.46])
            ax2_in.plot(acq_res.doppler_bins_hz / 1000.0, acq_res.doppler_cut, color="#7c3aed", linewidth=1.3)
            ax2_in.axvline(acq_res.est_doppler_hz / 1000.0, color="#16a34a", linestyle="--", linewidth=1.0)
            ax2_in.set_title(f"도플러 ({acq_res.est_doppler_hz:+.0f}Hz)", fontsize=7.5, fontweight="bold")
            ax2_in.set_xlabel("f_d [kHz]", fontsize=7.0)
            ax2_in.tick_params(labelsize=6.8)
            ax2_in.grid(True, linestyle=":", alpha=0.5)

        # Subplot 3: DLL Early / Prompt / Late Correlation Envelopes & Discriminator S-Curve
        if ax3 is not None:
            err_c = s_curve["error_chips"]
            ax3.plot(err_c, s_curve["r_early"], color="#f59e0b", linestyle="--", linewidth=1.2, label="Early 포락선 R_E")
            ax3.plot(err_c, s_curve["r_prompt"], color="#94a3b8", linestyle=":", linewidth=1.2, label="Prompt 포락선 R_P")
            ax3.plot(err_c, s_curve["r_late"], color="#8b5cf6", linestyle="--", linewidth=1.2, label="Late 포락선 R_L")
            ax3.plot(
                err_c,
                s_curve["s_curve_norm"],
                color="#16a34a",
                linewidth=2.2,
                label=f"E-L 판별기 S-Curve (d={spacing_chips:.1f})",
            )
            ax3.axvspan(
                -s_curve["linear_pull_in_chips"],
                s_curve["linear_pull_in_chips"],
                color="#dcfce7",
                alpha=0.55,
                label=f"선형 구간 (±{s_curve['linear_pull_in_chips']:.2f} chip)",
            )
            ax3.axhline(0.0, color="#475569", linewidth=0.8)
            ax3.axvline(0.0, color="#475569", linewidth=0.8)
            ax3.set_title("3. DLL Early-Prompt-Late 상관기 · S-Curve", fontsize=10.5, fontweight="bold", pad=8)
            ax3.set_xlabel("코드 추적 위상 오차 δτ [chips]", fontsize=9.0)
            ax3.set_ylabel("상관 진폭 / 판별기 출력", fontsize=9.0)
            ax3.set_xlim(-1.5, 1.5)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="lower right", fontsize=7.8)

        # Subplot 4: Closed-Loop Costas PLL Prompt I_P / Q_P (50 bps nav bit demod) & Code Error convergence
        if ax4 is not None:
            t_ms = trk_res["t_ms"]
            ax4.plot(t_ms, trk_res["ip"], color="#2563eb", linewidth=1.5, label="Prompt I_P (50bps 항법비트)")
            ax4.plot(t_ms, trk_res["qp"], color="#dc2626", linewidth=1.1, alpha=0.8, label="Prompt Q_P (→ 0 수렴)")
            ax4.plot(
                t_ms,
                trk_res["code_err_chips"],
                color="#16a34a",
                linewidth=1.9,
                linestyle="-.",
                label="DLL 코드 오차 [chips]",
            )
            ax4.axhline(0.0, color="#64748b", linestyle=":", linewidth=1.0)
            ax4.set_title(
                f"4. Costas PLL ({pll_bw:.0f}Hz) I_P/Q_P 복조 · DLL ({dll_bw:.1f}Hz) 수렴",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax4.set_xlabel("추적 경과 시간 [ms]", fontsize=9.0)
            ax4.set_ylabel("상관기 출력 / 코드 오차", fontsize=9.0)
            ax4.grid(True, linestyle="--", alpha=0.5)
            ax4.legend(loc="lower right", fontsize=7.8)

    def _draw_3d_pcps_surface(
        self,
        ax_3d,
        acq_res,
        prn_main: int,
        cn0_dbhz: float,
        coh_ms: int,
    ) -> None:
        """Render the 3D Parallel Code Phase Search (PCPS) correlation surface over (Code Phase x Doppler)."""
        # Subsample code phase axis to ~90 columns while preserving the exact peak column
        n_code = len(acq_res.code_phase_chips)
        peak_c_idx = int(np.argmax(acq_res.code_cut))
        stride = max(1, n_code // 85)
        col_indices = np.unique(np.append(np.arange(0, n_code, stride), [peak_c_idx, max(0, peak_c_idx - 1), min(n_code - 1, peak_c_idx + 1)]))

        code_sub = acq_res.code_phase_chips[col_indices]
        dopp_khz = acq_res.doppler_bins_hz / 1000.0
        z_sub = acq_res.corr_grid[:, col_indices]

        c_grid, d_grid = np.meshgrid(code_sub, dopp_khz)
        ax_3d.plot_surface(
            c_grid,
            d_grid,
            z_sub,
            cmap="plasma",
            edgecolor="none",
            alpha=0.88,
            antialiased=True,
        )

        # Highlight detected 3D correlation peak
        ax_3d.scatter(
            [acq_res.est_code_phase_chips],
            [acq_res.est_doppler_hz / 1000.0],
            [acq_res.peak_metric],
            color="#16a34a" if acq_res.is_detected else "#dc2626",
            s=70,
            zorder=10,
            label=f"피크 ({acq_res.est_code_phase_chips:.0f}chip, {acq_res.est_doppler_hz/1000.0:+.2f}kHz)",
        )

        ax_3d.set_xlabel("코드 위상 τ [chips]", fontsize=8.5)
        ax_3d.set_ylabel("도플러 주파수 f_d [kHz]", fontsize=8.5)
        ax_3d.set_zlabel("상관 전력 S(f_d, τ)", fontsize=8.5)
        ax_3d.set_title(
            f"[3D] 2D FFT PCPS 상관 평면 S(f_d, τ) (PRN #{prn_main}, C/N0={cn0_dbhz:.0f}dB-Hz, T_coh={coh_ms}ms)",
            fontsize=11.0,
            fontweight="bold",
            pad=10,
        )
        ax_3d.view_init(elev=28, azim=-52)
        ax_3d.legend(loc="upper right", fontsize=7.5)

    def get_lv1_easy_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">1. 일상생활 비유로 30초 만에 이해하기</h3>
            <ul>
                <li><b>PRN 골드 코드 (비밀 도장 맞추기)</b>:
                    우주에서 온 GPS 전파는 주변 소음(열잡음)보다 100배나 작아서 그냥 들으면 '치이익~' 하는 라디오 잡음밖에 안 들립니다.
                    하지만 위성마다 고유한 <b>1023칸짜리 비밀 톱니바퀴(PRN 코드)</b>가 있어서, 내 수신기의 톱니바퀴와 <b>아귀가 딱 맞는 순간에만 에너지가 1023배(+30 dB)로 뻥튀기</b>되어 뾰족한 산(피크)이 솟아오릅니다!</li>
                <li><b>신호 획득 (Acquisition: 처음 위성 찾기)</b>:
                    위성이 어디쯤 있는지 처음 찾을 때, 1023칸의 시간 지연(코드 위상)과 위성 속도에 따른 목소리 변화(도플러 주파수)를 <b>2차원 바둑판처럼 한 번에 뒤져서(2D FFT) 가장 높게 솟은 산봉우리</b>를 찾는 과정입니다.</li>
                <li><b>신호 추적 (DLL · PLL: 찾은 위성 계속 따라가기)</b>:
                    <ul>
                        <li><b>DLL (코드 추적)</b>: 산봉우리 꼭대기(Prompt)를 놓치지 않으려고 양손으로 왼쪽 어깨(<b>Early</b>)와 오른쪽 어깨(<b>Late</b>)를 짚어보고, 두 어깨 높이가 같아지도록(<code>Early - Late = 0</code>) 중심을 잡는 것입니다.</li>
                        <li><b>PLL (반송파 위상 추적)</b>: 위성 전파의 물결(위상)에 내 박자를 딱 맞춰서, <b>위성이 1초에 50번씩 보내는 0과 1 디지털 글자(항법 메시지)를 받아 적는 과정</b>입니다.</li>
                    </ul>
                </li>
            </ul>

            <h3 style="color:#1E3A2F;">2. 그래프 어디를 봐야 하나요?</h3>
            <ul>
                <li><b>2D 그래프 1</b>: 파란색 뾰족한 산이 <b>내 위성과 아귀가 맞았을 때(자기상관)</b>이고, 빨간색 잔물결은 <b>다른 위성과 곱했을 때(상호상관, 0 근처로 무시됨)</b>입니다.</li>
                <li><b>2D 그래프 2 · 3D 입체 탭</b>: 잡음 바다 한가운데에 <b>우뚝 솟은 피크 하나</b>가 바로 찾아낸 위성입니다. 빨간 점선(검출 기준선)보다 높아야 위성을 찾은 것입니다.</li>
                <li><b>2D 그래프 3</b>: 초록색 S자 곡선(<b>S-Curve</b>)이 가운데 <code>0</code>을 지나는 직선 구간에서 양손(Early, Late)이 산봉우리 중심을 꽉 잡고 있습니다.</li>
                <li><b>2D 그래프 4</b>: 추적이 성공하면 <b>빨간선(<code>Q_P</code>)은 0으로 잔잔해지고, 파란선(<code>I_P</code>)이 위(+1)와 아래(-1)로 오르내리며 데이터 비트(0과 1)를 복원</b>합니다.</li>
            </ul>
        </div>
        """

    def get_lv1_easy_c_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">3. C 코드 3단계 한글 흐름도 (1ms 인터럽트에서 하는 일)</h3>
            <pre style="background-color:#FAFBF9; border:1px solid #D8E2DB; color:#14532d; padding:12px; border-radius:8px; font-family:'Malgun Gothic',monospace; font-size:13px;">
[매 0.001초(1ms)마다 하드웨어 상관기가 다 합산했다는 인터럽트 발생!]
1. FPGA 레지스터에서 6개 숫자 읽어오기:
   왼쪽어깨(I_E, Q_E), 꼭대기(I_P, Q_P), 오른쪽어깨(I_L, Q_L)

2. 양쪽 어깨 높이 비교해서 코드 속도 조절 (DLL):
   어깨차이 = (왼쪽어깨_크기 - 오른쪽어깨_크기) / (왼쪽 + 오른쪽);
   만약 왼쪽이 더 높으면 -> "내 코드가 너무 빠르네, 속도를 조금 늦추자!"

3. 꼭대기(I_P, Q_P) 각도 비교해서 주파수 조절 (Costas PLL):
   각도오차 = atan(Q_P / I_P);  // Q_P를 0으로 만들도록 주파수(NCO) 보정!
   데이터비트 = (I_P > 0) ? 1 : 0; // I_P 부호로 위성이 보낸 0/1 비트 읽기!</pre>
        </div>
        """

    def get_theory_html(self) -> str:
        return """
        <div style="line-height: 1.68;">
            <h2 style="color:#1E3A2F; margin-top:0;">1. CDMA 대역확산 이득과 PRN 골드 코드 직교성</h2>
            <p>
                고도 20,200 km 궤도에서 수신되는 L1 C/A 신호 전력은 약 <b>-130 dBm</b>으로 수신단 열잡음(-110 dBm)보다 ~20 dB 아래에 존재합니다.
                따라서 기저대역 신호처리단은 <a href="glossary://cdma" style="color:#1E3A2F; font-weight:700;">CDMA 역확산(Despreading)</a>과
                <a href="glossary://prn_code" style="color:#1E3A2F; font-weight:700;">PRN 골드 코드(Gold Code)</a>의 이진 상관 특성을 통해 SNR을 복구합니다.
            </p>
            <ul>
                <li>
                    <b>자기상관 처리 이득 (Processing Gain)</b>: 주기 1023 칩(1 ms) 시퀀스는 복제 코드와 ±1 칩(약 ±293 m) 이내로 동기화될 때
                    <code>10·log10(1023) = +30.10 dB</code>의 상관 피크를 형성합니다.
                </li>
                <li>
                    <b>3-값 상호상관 상한</b>: Gold 수열의 주기 상호상관 크기는 <code>{-65, -1, +63}/1023</code>으로 유계화(≤ -23.9 dB)되어
                    동일 중심주파수(1575.42 MHz)를 공유하는 다중 위성 간 채널 간섭을 억제합니다.
                </li>
            </ul>

            <h2 style="color:#1E3A2F;">2. 2D FFT 병렬 코드 위상 탐색 (PCPS Acquisition)</h2>
            <p>
                궤도 운동에 따른 <a href="glossary://doppler_shift" style="color:#1E3A2F; font-weight:700;">도플러 천이(±5 kHz 이상)</a>와
                미지의 코드 지연 τ를 시간 영역에서 전수 상관하면 연산량이 O(N²)으로 폭증합니다.
                <a href="glossary://pcps_acquisition" style="color:#1E3A2F; font-weight:700;">PCPS(Parallel Code Phase Search)</a>는
                순환 상관 정리를 이용해 <code>IFFT(FFT(x) · conj(FFT(code)))</code> 한 번으로 모든 코드 위상 빈(Bin)을 O(N log N)에 병렬 계산합니다.
            </p>

            <h2 style="color:#1E3A2F;">3. 비간섭 DLL 판별기와 Costas PLL 폐루프 추적 이론</h2>
            <ul>
                <li>
                    <b><a href="glossary://dll" style="color:#1E3A2F; font-weight:700;">DLL (Delay Lock Loop)</a></b>:
                    정규화된 Early-Minus-Late 판별기는 ±d/2 칩 선형 영역에서 신호 진폭 변동과 반송파 위상에 독립적인 코드 지연 오차를 출력합니다.
                </li>
                <li>
                    <b><a href="glossary://pll" style="color:#1E3A2F; font-weight:700;">Costas PLL (Phase Lock Loop)</a></b>:
                    50 bps 항법 데이터 비트 반전(180° 위상 점프)에 불변인 2사분면 판별기 <code>atan(Q_P / I_P)</code>를 적용해
                    잔여 도플러를 추적하고 Q 채널 잡음 전력을 최소화하며 I<sub>P</sub> 채널에서 항법 비트를 복조합니다.
                </li>
            </ul>
        </div>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. 2D FFT 병렬 코드 위상 탐색 (PCPS Acquisition) 상관 방정식",
                latex_expr=(
                    r"S(f_d, \tau) = \frac{1}{N^2} \left| \mathrm{IFFT}\left\{ \mathrm{FFT}\left( x[n] \cdot e^{-j 2\pi f_d n / f_s} \right) \cdot \mathrm{FFT}^*\left( c_{\mathrm{PRN}}[n] \right) \right\} \right|^2"
                ),
                intuitive_summary="입력 신호에서 후보 도플러 주파수 f_d를 먼저 제거(Carrier Wipe-off)한 뒤, FFT와 로컬 PRN 코드의 켤레복소수 곱셈 및 IFFT를 통해 모든 코드 지연 τ에 대한 상관 전력을 한 번에 계산합니다.",
                symbols=[
                    SymbolSpec("x[n]", "기저대역 복소 수신 샘플", "V", "ADC 샘플링된 I/Q 디지털 수신 신호", "cdma"),
                    SymbolSpec("f_d, f_s", "탐색 도플러 주파수 및 샘플링율", "Hz", "위성 시선 속도에 의한 반송파 도플러 편이 및 ADC 샘플링 주파수", "doppler_shift"),
                    SymbolSpec(r"c_{\mathrm{PRN}}[n]", "로컬 PRN 골드 코드", "±1", "찾고자 하는 위성의 1023 칩 C/A 코드 시퀀스", "prn_code"),
                    SymbolSpec(r"S(f_d, \tau)", "2차원 상관 전력 그리드", "-", "도플러 빈과 코드 위상 평면의 상관 피크 에너지", "pcps_acquisition"),
                ],
            ),
            FormulaSpec(
                title="수식 2. 정규화 Early-Minus-Late DLL 판별기(S-Curve) 및 열잡음 코드 지터",
                latex_expr=(
                    r"e_{\mathrm{DLL}}(\delta\tau) = \left(1 - \frac{d}{2}\right) \frac{R_E - R_L}{R_E + R_L}, \qquad \sigma_{\mathrm{DLL}} = \sqrt{\frac{B_{\mathrm{DLL}} \cdot d}{2 \cdot C/N_0}\left(1 + \frac{2}{T_{\mathrm{coh}} \cdot C/N_0 (2 - d)}\right)}"
                ),
                intuitive_summary="Early와 Late 상관 진폭의 차이를 합으로 나누어 신호 세기 변동에 무관한 코드 오차 δτ를 구하며, 상관기 간격 d와 대역폭 B_DLL이 작을수록 코드 잡음 지터 σ_DLL이 감소합니다.",
                symbols=[
                    SymbolSpec("R_E, R_L", "Early 및 Late 상관 크기", "-", "Prompt 기준 ±d/2 칩 앞뒤에 배치된 상관기의 포락선 크기", "dll"),
                    SymbolSpec("d", "E-L 상관기 간격", "chips", "Early와 Late 사이의 칩 간격 (표준 1.0~0.5, Narrow 0.1~0.2 chip)", "dll"),
                    SymbolSpec(r"B_{\mathrm{DLL}}", "DLL 루프 노이즈 대역폭", "Hz", "코드 추적 루프 필터 대역폭 (보통 0.5 ~ 2 Hz)", "dll"),
                    SymbolSpec("C/N_0", "반송파 대 잡음밀도 비", "Hz (선형)", "선형 비율로 변환된 신호 품질: 10^((C/N0_dBHz)/10)", "cn0"),
                ],
            ),
            FormulaSpec(
                title="수식 3. 코스타스(Costas) PLL 위상 판별기 및 열잡음 위상 지터",
                latex_expr=(
                    r"e_{\mathrm{PLL}} = \arctan\left(\frac{Q_P}{I_P}\right), \qquad \sigma_{\mathrm{PLL}} = \frac{180^\circ}{\pi} \sqrt{\frac{B_{\mathrm{PLL}}}{C/N_0}\left(1 + \frac{1}{2 \cdot T_{\mathrm{coh}} \cdot C/N_0}\right)} \leq 15^\circ"
                ),
                intuitive_summary="50 bps 항법 데이터 비트 반전(±180°)에 영향을 받지 않는 atan(Q_P/I_P)로 반송파 위상 오차를 검출하며, 1σ 위상 지터가 15°를 넘으면 사이클 슬립 및 Lock 상실이 발생합니다.",
                symbols=[
                    SymbolSpec("I_P, Q_P", "Prompt 동위상/직교위상 상관값", "-", "정시(Prompt) 상관기의 실수부(항법비트 포함)와 허수부(위상 오차 성분)", "pll"),
                    SymbolSpec(r"e_{\mathrm{PLL}}", "반송파 위상 오차 추정치", "rad", "범위 [-π/2, +π/2] 내에서 검출된 반송파 위상 오차", "pll"),
                    SymbolSpec(r"B_{\mathrm{PLL}}", "PLL 루프 대역폭", "Hz", "반송파 추적 루프 대역폭 (보통 10 ~ 25 Hz)", "pll"),
                    SymbolSpec(r"\sigma_{\mathrm{PLL}}", "열잡음 위상 지터 (1σ)", "deg", "15° 이하 유지 시 안정적 반송파 추적 가능 (3σ = 45° 임계)", "pll"),
                ],
            ),
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <b style="color:#1E3A2F; font-size:14px;">슬라이더 조작 실습 미션 (Observation Guide)</b>
            <ol style="margin-top:8px; margin-bottom:4px; padding-left:20px;">
                <li style="margin-bottom:6px;">
                    <b>미션 1 (약신호 획득과 동기 적분 시간):</b> <b>[신호 품질 (C/N0)]</b>를 <code style="color:#dc2626;">30 dB-Hz</code>로 낮추면 2D 그래프 2에서 피크가 빨간 임계선 아래로 묻혀 획득에 실패합니다.
                    이때 <b>[동기 적분 시간 (T_coh)]</b>을 <code style="color:#2563eb;">8 ms</code>로 늘려 피크가 다시 솟아오르는지 확인하세요.
                </li>
                <li style="margin-bottom:6px;">
                    <b>미션 2 (Narrow Correlator 효과):</b> <b>[E-L 상관기 간격 (d)]</b>을 <code style="color:#16a34a;">1.0 chip → 0.2 chip</code>으로 줄여 상단 <b>[그래프 3 · 4 크게 보기]</b>로 전환해보세요.
                    2D 그래프 3의 선형 구간 폭과 상단 KPI 카드의 <b>DLL 거리 지터(σ_DLL)</b> 변화를 비교하세요.
                </li>
                <li>
                    <b>미션 3 (PLL 대역폭 트레이드오프):</b> <b>[초기 주파수 풀인 오차]</b>를 <code style="color:#d97706;">55 Hz</code>로 높인 상태에서
                    <b>[PLL 루프 대역폭]</b>이 좁을 때(<code style="color:#dc2626;">8 Hz</code>)와 넓을 때(<code style="color:#2563eb;">28 Hz</code>) 2D 그래프 4의 <code>Q_P → 0</code> 수렴 속도를 비교하세요.
                </li>
            </ol>
        </div>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <h2 style="color:#1E3A2F; margin-top:0;">실전 임베디드 C/DSP 포팅 가이드 (FPGA / STM32 / TI C6000)</h2>

            <div style="background-color:#F4F7F4; border-left:4px solid #1E3A2F; padding:12px 14px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#1E3A2F;">[핵심 아키텍처] 하드웨어 상관기(FPGA/ASIC) + 소프트웨어 루프 필터(MCU/DSP)</b><br>
                수 MHz 속도로 들어오는 IF 샘플과 로컬 코드/반송파를 곱해 <code>I_E, Q_E, I_P, Q_P, I_L, Q_L</code> 6개 적분값을 만드는
                <b>Correlator + 32-bit NCO</b>는 FPGA(또는 전용 베이스밴드 칩)에서 수행하고,
                <b>1 ms (1 kHz) 인터럽트</b>마다 MCU가 6개 적분 레지스터를 읽어 DLL/PLL 판별기와 루프 필터를 계산해 NCO 주파수 워드를 갱신합니다.
            </div>

            <h3 style="color:#0f172a;">1. 32비트 정수 위상 누산기(Phase Accumulator) 기반 NCO 구현</h3>
            <p>
                부동소수점 <code>fmod()</code>로 위상을 돌리면 누적 오차와 연산 지연이 발생합니다.
                32비트 부호 없는 정수(<code>uint32_t</code>)의 자연 오버플로우(Modulo 2³²)를 이용하면 단 1클럭 덧셈으로 완벽한 NCO가 구현됩니다.
            </p>
            <pre style="background-color:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#include &lt;stdint.h&gt;
#include &lt;math.h&gt;

/* 주파수 f_hz를 32-bit NCO Tuning Word로 변환: FCW = round(f_hz * 2^32 / f_sample) */
static inline uint32_t GNSS_HzToNcoWord(double freq_hz, double fs_hz) {
    return (uint32_t)((freq_hz / fs_hz) * 4294967296.0);
}

/* 1 ms (1000 Hz) 추적 루프 상태 구조체 */
typedef struct {
    float pll_vel_acc_hz;   /* 2차 PLL 적분기 상태 (도플러 주파수 추정치) */
    float dll_vel_acc_cps;  /* 2차 DLL 적분기 상태 (코드 도플러 추정치) */
    float wn_pll;           /* PLL 고유주파수: 8*zeta*B_pll / (4*zeta^2 + 1) */
    float wn_dll;           /* DLL 고유주파수 */
} TrackingChannel_t;

/* 1 ms Correlation Dump ISR에서 호출되는 Costas PLL + Early-Late DLL 루프 갱신 */
void GNSS_UpdateTrackingLoop_1ms(TrackingChannel_t *ch,
                                 float i_e, float q_e,
                                 float i_p, float q_p,
                                 float i_l, float q_l,
                                 float *out_carrier_dopp_hz,
                                 float *out_code_dopp_cps) {
    const float dt = 0.001f;       /* 1 ms */
    const float zeta = 0.70710678f;

    /* 1. Costas PLL 판별기: atan(Q_P / I_P) -> 50 bps 항법비트 180도 위상 반전에 불변 */
    float safe_ip = (fabsf(i_p) &gt; 1e-6f) ? i_p : ((i_p &gt;= 0.0f) ? 1e-6f : -1e-6f);
    float pll_err_rad = atanf(q_p / safe_ip);

    /* 2차 PLL 루프 필터 */
    ch-&gt;pll_vel_acc_hz += (ch-&gt;wn_pll * ch-&gt;wn_pll * pll_err_rad * dt) / 6.2831853f;
    *out_carrier_dopp_hz = ch-&gt;pll_vel_acc_hz + (2.0f * zeta * ch-&gt;wn_pll * pll_err_rad) / 6.2831853f;

    /* 2. 비동기 Early-Minus-Late Envelope DLL 판별기 */
    float env_e = sqrtf(i_e * i_e + q_e * q_e);
    float env_l = sqrtf(i_l * i_l + q_l * q_l);
    float dll_err_chips = 0.75f * (env_e - env_l) / (env_e + env_l + 1e-6f); /* d = 0.5 chip 기준 */

    /* 2차 DLL 루프 필터 + Carrier-Aiding (반송파 도플러로 코드 도플러의 99%를 선행 보상!) */
    const float scale_l1 = 1.023e6f / 1575.42e6f; /* 1 / 1540 */
    ch-&gt;dll_vel_acc_cps += (ch-&gt;wn_dll * ch-&gt;wn_dll) * dll_err_chips * dt;
    float dll_feedback = ch-&gt;dll_vel_acc_cps + 2.0f * zeta * ch-&gt;wn_dll * dll_err_chips;
    *out_code_dopp_cps = (*out_carrier_dopp_hz * scale_l1) + dll_feedback;
}</pre>
        </div>
        """
