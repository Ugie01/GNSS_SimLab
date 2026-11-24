"""
Module 06 (Danam Systems Navigation Technology R&D Institute SW Team Specialization):
- RAIM (Receiver Autonomous Integrity Monitoring) & FDE (Fault Detection and Exclusion)
- Pull-Off Spoofing Correlation Peak Hijacking & SQM (Signal Quality Monitoring)
- EGI (Embedded GPS/INS) Lever-Arm & Time-Tag Latency Compensation under High-G Maneuvers
- STAP (Space-Time Adaptive Processing) vs SAP Broadband Jamming Suppression
- Strapdown INS Coning & Sculling + RTOS SW Architecture Guide
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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

from core.egi_raim_spoof import run_egi_raim_spoof_simulation
from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Module06EgiRaimSpoof(BaseLabModule):
    """Module 6: Danam Navigation R&D SW Specialization — RAIM/FDE, Anti-Spoofing SQM, EGI Lever-Arm/Latency, STAP."""

    module_id = "m06_egi_raim_spoof"
    module_title = "M06. [항법연구소 특화] EGI 실무 · RAIM/항기만 · STAP/레버암"
    module_subtitle = "단암시스템즈 항법기술연구소 SW팀 핵심 실무: RAIM/FDE 고장분리, Pull-off 항기만(SQM), 레버암/타임태그 보정, STAP"
    module_badge = "DANAM R&D"

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

        # 1. RAIM & FDE Fault Injection Group
        g_raim = QGroupBox("1. RAIM 무결성 감시 · FDE 고장 위성 분리")
        g_raim.setStyleSheet(group_style)
        f_raim = QFormLayout(g_raim)
        f_raim.setSpacing(7)

        self.slider_fault_bias = QSlider(Qt.Orientation.Horizontal)
        self.slider_fault_bias.setRange(0, 100)  # 0 to 100 m fault bias on PRN #12
        self.slider_fault_bias.setValue(45)
        self.lbl_fault_bias = QLabel("45 m (고장/기만)")
        self.lbl_fault_bias.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_fault_bias.valueChanged.connect(self._on_fault_bias_changed)

        fb_row = QHBoxLayout()
        fb_row.addWidget(self.slider_fault_bias, stretch=1)
        fb_row.addWidget(self.lbl_fault_bias)
        f_raim.addRow("G12 의사거리 오차 주입:", fb_row)
        layout.addWidget(g_raim)

        # 2. Pull-Off Spoofing & SQM Parameters
        g_spoof = QGroupBox("2. Pull-Off 스푸핑(기만) · SQM 비대칭 탐지")
        g_spoof.setStyleSheet(group_style)
        f_spoof = QFormLayout(g_spoof)
        f_spoof.setSpacing(7)

        self.slider_spoof_delay = QSlider(Qt.Orientation.Horizontal)
        self.slider_spoof_delay.setRange(0, 150)  # 0.00 to 1.50 chips
        self.slider_spoof_delay.setValue(45)      # 0.45 chips
        self.lbl_spoof_delay = QLabel("0.45 chip")
        self.lbl_spoof_delay.setStyleSheet("font-weight:700; color:#d97706;")
        self.slider_spoof_delay.valueChanged.connect(self._on_spoof_delay_changed)

        sd_row = QHBoxLayout()
        sd_row.addWidget(self.slider_spoof_delay, stretch=1)
        sd_row.addWidget(self.lbl_spoof_delay)
        f_spoof.addRow("기만 신호 코드 오프셋:", sd_row)

        self.spin_spoof_pwr = QDoubleSpinBox()
        self.spin_spoof_pwr.setRange(0.0, 15.0)
        self.spin_spoof_pwr.setDecimals(1)
        self.spin_spoof_pwr.setSingleStep(0.5)
        self.spin_spoof_pwr.setValue(4.0)
        self.spin_spoof_pwr.setSuffix(" dB (S/A 우세비)")
        self.spin_spoof_pwr.valueChanged.connect(lambda _: self.run_simulation())
        f_spoof.addRow("기만 대 정상 신호비:", self.spin_spoof_pwr)
        layout.addWidget(g_spoof)

        # 3. EGI Lever-Arm, Time-Tag Latency & STAP Taps
        g_egi = QGroupBox("3. EGI 레버암/타임태그 지연 · 광대역 STAP 탭 수")
        g_egi.setStyleSheet(group_style)
        f_egi = QFormLayout(g_egi)
        f_egi.setSpacing(7)

        self.spin_lever = QDoubleSpinBox()
        self.spin_lever.setRange(0.0, 5.0)
        self.spin_lever.setDecimals(2)
        self.spin_lever.setSingleStep(0.2)
        self.spin_lever.setValue(1.20)
        self.spin_lever.setSuffix(" m (IMU↔안테나)")
        self.spin_lever.valueChanged.connect(lambda _: self.run_simulation())
        f_egi.addRow("장착 레버암 거리 (|l^b|):", self.spin_lever)

        self.slider_latency = QSlider(Qt.Orientation.Horizontal)
        self.slider_latency.setRange(0, 120)  # 0 to 120 ms
        self.slider_latency.setValue(40)
        self.lbl_latency = QLabel("40 ms")
        self.lbl_latency.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_latency.valueChanged.connect(self._on_latency_changed)

        lat_row = QHBoxLayout()
        lat_row.addWidget(self.slider_latency, stretch=1)
        lat_row.addWidget(self.lbl_latency)
        f_egi.addRow("GNSS 통신/연산 지연:", lat_row)

        self.spin_stap_taps = QSpinBox()
        self.spin_stap_taps.setRange(1, 9)
        self.spin_stap_taps.setValue(5)
        self.spin_stap_taps.setSuffix(" 탭 (STAP FIR)")
        self.spin_stap_taps.valueChanged.connect(lambda _: self.run_simulation())
        f_egi.addRow("STAP 시간 지연 탭 (L):", self.spin_stap_taps)

        layout.addWidget(g_egi)
        return container

    def _on_fault_bias_changed(self, val: int) -> None:
        self.lbl_fault_bias.setText(f"{val} m (고장/기만)" if val > 0 else "0 m (정상)")
        self.run_simulation()

    def _on_spoof_delay_changed(self, val: int) -> None:
        self.lbl_spoof_delay.setText(f"{val / 100.0:.2f} chip")
        self.run_simulation()

    def _on_latency_changed(self, val: int) -> None:
        self.lbl_latency.setText(f"{val} ms")
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

        self.lbl_raim_readout = QLabel()
        self.lbl_raim_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#e2e8f0; border:none;"
        )
        self.lbl_spoof_readout = QLabel()
        self.lbl_spoof_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#7dd3fc; border:none;"
        )
        self.lbl_egi_readout = QLabel()
        self.lbl_egi_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#86efac; border:none;"
        )

        grid.addWidget(self.lbl_raim_readout, 0, 0)
        grid.addWidget(self.lbl_spoof_readout, 0, 1)
        grid.addWidget(self.lbl_egi_readout, 0, 2)
        vbox.addWidget(cards_frame)

        self.fig = Figure(figsize=(10.2, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)
        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        widgets = [
            self.slider_fault_bias,
            self.slider_spoof_delay,
            self.spin_spoof_pwr,
            self.spin_lever,
            self.slider_latency,
            self.spin_stap_taps,
        ]
        for w in widgets:
            w.blockSignals(True)

        fb = int(round(float(preset_data.get("fault_prn_bias_m", 45.0))))
        self.slider_fault_bias.setValue(fb)
        self.lbl_fault_bias.setText(f"{fb} m (고장/기만)" if fb > 0 else "0 m (정상)")

        sd = int(round(float(preset_data.get("spoofer_delay_chips", 0.45)) * 100))
        self.slider_spoof_delay.setValue(sd)
        self.lbl_spoof_delay.setText(f"{sd / 100.0:.2f} chip")

        self.spin_spoof_pwr.setValue(float(preset_data.get("spoofer_power_ratio_db", 4.0)))
        self.spin_lever.setValue(float(preset_data.get("lever_arm_m", 1.2)))

        lat = int(round(float(preset_data.get("latency_ms", 40.0))))
        self.slider_latency.setValue(lat)
        self.lbl_latency.setText(f"{lat} ms")

        self.spin_stap_taps.setValue(int(preset_data.get("stap_taps", 5)))

        for w in widgets:
            w.blockSignals(False)

    def run_simulation(self) -> None:
        fault_bias = float(self.slider_fault_bias.value())
        spoof_delay = self.slider_spoof_delay.value() / 100.0
        spoof_pwr = self.spin_spoof_pwr.value()
        lever_m = self.spin_lever.value()
        latency_ms = float(self.slider_latency.value())
        stap_taps = self.spin_stap_taps.value()

        res = run_egi_raim_spoof_simulation(
            fault_prn_bias_m=fault_bias,
            spoofer_delay_chips=spoof_delay,
            spoofer_power_ratio_db=spoof_pwr,
            lever_arm_m=lever_m,
            latency_ms=latency_ms,
            stap_taps=stap_taps,
        )

        raim_color = "#DC2626" if res.raim_alarm_triggered else "#16A34A"
        raim_badge = f"고장 검출 · G{res.detected_fault_prn:02d} 자동 배제" if res.raim_alarm_triggered else "전 위성 무결성 정상"
        self.lbl_raim_readout.setText(
            f"<div style='font-size:11.5px; color:#1E3A2F; font-weight:800;'>RAIM 무결성 · FDE 고장 배제</div>"
            f"<div style='font-size:13.5px; color:{raim_color}; font-weight:900; margin-top:3px;'>{raim_badge}</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"WSSE {res.raim_wsse_stat:.1f}/{res.raim_chi2_threshold:.1f} · 오차 {res.pos_err_before_fde_m:.1f}m→{res.pos_err_after_fde_m:.2f}m</div>"
        )

        sq_color = "#DC2626" if res.spoof_alarm_triggered else "#16A34A"
        sq_badge = "Pull-Off 스푸핑 경보" if res.spoof_alarm_triggered else "상관 피크 대칭 정상"
        self.lbl_spoof_readout.setText(
            f"<div style='font-size:11.5px; color:#B45309; font-weight:800;'>항기만 SQM 상관 품질 감시</div>"
            f"<div style='font-size:13.5px; color:{sq_color}; font-weight:900; margin-top:3px;'>{sq_badge} (M_Δ={res.sqm_delta_metric:.4f})</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"임계 {res.sqm_threshold:.3f} · 오프셋 {spoof_delay:.2f}c (+{spoof_pwr:.1f}dB)</div>"
        )

        max_uncomp = float(np.max(res.uncomp_total_error_m))
        max_comp = float(np.max(res.comp_residual_error_m))
        red_pct = (1 - max_comp / max(max_uncomp, 1e-3)) * 100
        mean_stap_db = float(np.mean(res.stap_broadband_null_db))
        self.lbl_egi_readout.setText(
            f"<div style='font-size:11.5px; color:#1D4ED8; font-weight:800;'>EGI 레버암·타임태그 · STAP 항재밍</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:900; margin-top:3px;'>"
            f"보정 오차 {max_uncomp:.1f}m→{max_comp:.2f}m (-{red_pct:.0f}%)</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"SAP {np.mean(res.sap_broadband_null_db):.1f}dB vs {stap_taps}탭 STAP {mean_stap_db:.1f}dB</div>"
        )

        raim_coach = (
            f"현재 <code>G{res.fault_prn:02d}</code> 위성이 <b>+{fault_bias:.0f}m 오차</b>를 주입하고 있어 그대로 측위하면 <b>{res.pos_err_before_fde_m:.1f}m</b> 틀어지지만, 2D 그래프 1의 <b>RAIM 검사</b>가 <code>G{res.detected_fault_prn:02d}</code>를 분리해(<b>FDE</b>) 오차를 <b>{res.pos_err_after_fde_m:.2f}m</b>로 복원했습니다."
            if res.raim_alarm_triggered
            else "현재 모든 위성 측정치가 정상 범위로 RAIM 무결성이 유지되고 있습니다."
        )
        self.set_easy_coach_message(
            f"<b>[사수의 실시간 쉬운 해설]</b> (1) {raim_coach} &nbsp;|&nbsp; "
            f"(2) 2D 그래프 3에서 급선회 중 안테나 레버암(<b>{lever_m:.1f}m</b>)과 통신 지연(<b>{latency_ms:.0f}ms</b>)으로 생긴 <b>{max_uncomp:.1f}m 오차</b>를 SW 링버퍼 보정으로 <b>{max_comp:.2f}m</b>까지 줄였습니다."
        )

        # Tab 0: Clean 2D Graphs
        self.fig.clear()
        ax1, ax2, ax3, ax4 = self.allocate_2d_axes(self.fig)
        self._draw_2d_plots(
            ax1, ax2, ax3, ax4,
            res, fault_bias, spoof_delay, spoof_pwr, lever_m, latency_ms, stap_taps,
        )
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        # Tab 1: Dedicated 3D Pull-Off Spoofing Surface
        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            self._draw_3d_spoofing_pulloff_surface(ax_3d, res, spoof_pwr)
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def _draw_2d_plots(
        self,
        ax1,
        ax2,
        ax3,
        ax4,
        res,
        fault_bias: float,
        spoof_delay: float,
        spoof_pwr: float,
        lever_m: float,
        latency_ms: float,
        stap_taps: int,
    ) -> None:
        # Subplot 1: RAIM Normalized Pseudorange Residuals & FDE
        if ax1 is not None:
            prn_labels = [f"G{p:02d}" for p in res.sat_prns]
            bar_cols = [
                "#dc2626" if (res.raim_alarm_triggered and p == res.detected_fault_prn) else "#2563eb"
                for p in res.sat_prns
            ]
            bars = ax1.bar(prn_labels, res.normalized_residuals, color=bar_cols, width=0.55, alpha=0.88)
            ax1.axhline(
                res.raim_Gaussian_5sigma_thresh,
                color="#dc2626",
                linestyle="--",
                linewidth=1.5,
                label=f"FDE 고장 분리 임계값 ({res.raim_Gaussian_5sigma_thresh:.2f}σ)",
            )
            for b, val in zip(bars, res.normalized_residuals):
                ax1.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.15,
                    f"{val:.1f}σ",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    fontweight="bold",
                )
            ax1.set_ylim(0, max(float(np.max(res.normalized_residuals)) * 1.22, 5.5))
            ax1.set_title(
                f"1. RAIM 위성별 정규화 잔차 (FDE {res.pos_err_before_fde_m:.1f}m → {res.pos_err_after_fde_m:.2f}m)",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax1.set_ylabel("정규화 잔차 |r_i| / (σ·√S_ii)", fontsize=9.0)
            ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=7.8)

        # Subplot 2: Pull-Off Spoofing Correlation Peak Distortion & SQM
        if ax2 is not None:
            ax2.plot(res.tau_chips, res.corr_authentic, color="#2563eb", linestyle="--", linewidth=1.4, label="정상 위성 피크 R_auth")
            ax2.plot(res.tau_chips, res.corr_spoofer, color="#dc2626", linestyle=":", linewidth=1.5, label=f"기만 신호 R_spoof (+{spoof_pwr:.1f}dB)")
            ax2.plot(
                res.tau_chips,
                res.corr_composite,
                color="#0f172a",
                linewidth=2.2,
                label=f"합성 상관 피크 (SQM M_Δ={res.sqm_delta_metric:.3f})",
            )
            pk_idx = int(np.argmax(res.corr_composite))
            pk_tau = float(res.tau_chips[pk_idx])
            for off_c, col_c, lbl_c in [(-0.25, "#f59e0b", "Early"), (0.0, "#16a34a", "Prompt"), (0.25, "#8b5cf6", "Late")]:
                val_c = float(np.interp(pk_tau + off_c, res.tau_chips, res.corr_composite))
                ax2.scatter([pk_tau + off_c], [val_c], color=col_c, s=55, zorder=6, label=f"{lbl_c} 상관기")
            ax2.set_title("2. Pull-Off 스푸핑 상관 피크 납치 · SQM 비대칭", fontsize=10.5, fontweight="bold", pad=8)
            ax2.set_xlabel("코드 위상 지연 τ [chips]", fontsize=9.0)
            ax2.set_ylabel("정규화 상관 진폭", fontsize=9.0)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right", fontsize=7.6, ncol=2)

        # Subplot 3: EGI High-G Maneuver Lever-Arm & Time-Tag Latency Error
        if ax3 is not None:
            t_m = res.maneuver_t_s
            ax3.plot(t_m, res.uncomp_total_error_m, color="#dc2626", linewidth=2.0, label="미보상 총 이노베이션 오차 [m]")
            ax3.plot(t_m, res.uncomp_latency_error_m, color="#d97706", linestyle="--", linewidth=1.3, label=f"타임태그 지연 오차 ({latency_ms:.0f}ms)")
            ax3.plot(t_m, res.uncomp_lever_error_m, color="#7c3aed", linestyle=":", linewidth=1.3, label=f"레버암 회전 오차 ({lever_m:.1f}m)")
            ax3.plot(t_m, res.comp_residual_error_m, color="#16a34a", linewidth=2.1, label="SW 링버퍼+레버암 보상 후 잔차")
            ax3.set_title("3. 고기동 선회 시 EGI 레버암 · 타임태그 지연 보정", fontsize=10.5, fontweight="bold", pad=8)
            ax3.set_xlabel("비행 기동 시간 [초] (마하 0.65 S-Turn 선회)", fontsize=9.0)
            ax3.set_ylabel("칼만필터 측정치 왜곡 오차 [m]", fontsize=9.0)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right", fontsize=7.6)

        # Subplot 4: STAP (Space-Time) vs SAP (Spatial-Only) Broadband Jammer Null Depth
        if ax4 is not None:
            ax4.plot(
                res.freq_offset_mhz,
                res.sap_broadband_null_db,
                color="#dc2626",
                linestyle="--",
                linewidth=1.8,
                label="단순 공간 CRPA (SAP, 1탭: 대역 외곽 널 열화)",
            )
            ax4.plot(
                res.freq_offset_mhz,
                res.stap_broadband_null_db,
                color="#2563eb",
                linewidth=2.2,
                label=f"시공간 적응처리 (STAP, {stap_taps}탭: 광대역 깊은 널 유지)",
            )
            ax4.axhline(-40.0, color="#16a34a", linestyle=":", linewidth=1.3, label="전술 광대역 억제 목표 (-40 dB)")
            ax4.set_title(f"4. 광대역/다중경로 재밍 대역폭별 널 깊이 (SAP vs {stap_taps}탭 STAP)", fontsize=10.5, fontweight="bold", pad=8)
            ax4.set_xlabel("중심주파수 대비 오프셋 Δf [MHz] (GPS L1 ±10 MHz 대역)", fontsize=9.0)
            ax4.set_ylabel("광대역 재머 널 깊이 [dB]", fontsize=9.0)
            ax4.set_ylim(-68, 0)
            ax4.grid(True, linestyle="--", alpha=0.5)
            ax4.legend(loc="lower right", fontsize=7.8)

    def _draw_3d_spoofing_pulloff_surface(self, ax_3d, res, spoof_pwr: float) -> None:
        """Render 3D Time-Evolution Surface R(tau, t) of Pull-Off Spoofing hijacking the DLL correlation peak."""
        tau_grid, t_grid = np.meshgrid(res.pulloff_tau_chips, res.pulloff_time_s)
        ax_3d.plot_surface(
            tau_grid,
            t_grid,
            res.pulloff_surface,
            cmap="viridis",
            alpha=0.88,
            edgecolor="none",
            antialiased=True,
        )

        # Overlay 3D trajectory of hijacked dominant correlation peak over time
        dom_idx = np.argmax(res.pulloff_surface, axis=1)
        dom_tau = res.pulloff_tau_chips[dom_idx]
        dom_val = np.max(res.pulloff_surface, axis=1) + 0.04
        ax_3d.plot(
            dom_tau,
            res.pulloff_time_s,
            dom_val,
            color="#dc2626",
            linewidth=2.8,
            label="납치된 DLL 추적 피크 궤적",
        )

        ax_3d.set_xlabel("코드 지연 τ [chips]", fontsize=8.5)
        ax_3d.set_ylabel("기만 진행 시간 t [초]", fontsize=8.5)
        ax_3d.set_zlabel("합성 상관 진폭 R(τ, t)", fontsize=8.5)
        ax_3d.set_title(
            f"[3D] Pull-Off 스푸핑 상관 피크 납치 곡면 R(τ, t) (t=1.5s 이후 기만 +{spoof_pwr:.1f}dB)",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )
        ax_3d.view_init(elev=30, azim=-58)
        ax_3d.legend(loc="upper left", fontsize=7.5)

    def get_lv1_easy_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">1. 단암 항법기술연구소 핵심 기술 4가지, 비유로 30초 만에 이해하기</h3>
            <ul>
                <li><b>(1) RAIM · FDE = '9명 중 거짓말쟁이 1명 다수결로 찾아내 퇴장시키기'</b>:
                    내 위치와 시계를 구하는 데는 위성 4개면 충분합니다. 만약 하늘에 위성이 9개 떠 있다면 5개나 여유가 있죠.
                    나머지 8개 위성은 모두 <i>"너 지금 서울 시청에 있어"</i>라고 하는데, 고장 난 <code>G12</code> 위성 혼자만 <i>"너 지금 한강에 있어(+45m)"</i>라고 우기면,
                    <b>다수결(잔차 비교)로 <code>G12</code>를 범인으로 지목해 빼버리고(FDE) 나머지 8개로만 안전하게 비행</b>하는 기술입니다.</li>
                <li><b>(2) Pull-Off 스푸핑 · SQM = '내 그림자에 슬쩍 포개졌다가 나를 납치하는 가짜 신호 잡기'</b>:
                    적군 기만기(스푸퍼)는 처음부터 엉뚱한 신호를 쏘면 들키니까, 처음에는 진짜 위성 신호의 산봉우리에 자기 신호를 똑같이 포갠 뒤 힘(출력)만 살짝 키워서 내 추적기를 빼앗아 옆으로 끌고 갑니다(<b>3D 입체 시각화 탭 참고</b>).
                    이때 산봉우리 한가운데가 갈라지며 <b>삼각형의 양쪽 어깨(Early, Late)가 짝짝이로 찌그러지는 순간을 포착하는 것이 SQM 항기만</b>입니다.</li>
                <li><b>(3) 레버암(Lever-Arm) · 타임태그 지연 = '버스 앞자리와 뒷자리의 회전 차이 · 옛날 사진 비교'</b>:
                    <ul>
                        <li><b>레버암</b>: 유도무기 한가운데 있는 IMU와 껍데기에 붙은 GPS 안테나는 1.2m 떨어져 있어서, 급회전하면 안테나가 더 크게 돕니다. 이 장착 거리 차이를 빼주는 것입니다.</li>
                        <li><b>타임태그 링버퍼</b>: 초속 220m로 날아가는 비행체에서 GPS가 0.04초 늦게 도착하면 이미 9m 앞으로 간 뒤입니다. 그래서 <b>0.04초 전의 IMU 기록(링버퍼)을 꺼내 같은 시간끼리 비교</b>합니다.</li>
                    </ul>
                </li>
                <li><b>(4) STAP = '메아리(다중경로)와 넓은 주파수 재머까지 막는 입체 방음벽'</b>:
                    안테나 여러 개(공간)에 시간 지연 필터(시간 탭)까지 결합해서, 산이나 날개에 부딪혀 반사되는 광대역 재밍까지 막아내는 항재밍 기술입니다.</li>
            </ul>
        </div>
        """

    def get_lv1_easy_c_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">2. C 코드 3단계 한글 흐름도 (항법컴퓨터 EGI 소프트웨어 구조)</h3>
            <pre style="background-color:#FAFBF9; border:1px solid #D8E2DB; color:#14532d; padding:12px; border-radius:8px; font-family:'Malgun Gothic',monospace; font-size:13px;">
[1단계: 1초에 400번(400Hz) 도는 고속 IMU 적분 태스크]
   - 기체가 덜덜 떨릴 때 생기는 회전 오차(코닝/스컬링)를 외적(Cross Product)으로 먼저 닦아냄!
   - 내 위치/속도/자세를 적분하고, 나중에 GPS와 시간 맞춰 비교할 수 있게 '링버퍼(최근 256개 기록장)'에 저장!

[2단계: 1PPS 핀이 번쩍! 뛸 때 (하드웨어 캡처 인터럽트)]
   - "방금 1PPS가 뛴 순간은 링버퍼의 142번 칸이었음!" 하고 책갈피를 꽂아둠.

[3단계: 0.04초 뒤에 GPS 패킷이 도착했을 때 (10Hz 칼만필터 태스크)]
   - 위성들끼리 다수결 검사(RAIM)해서 혼자 거짓말하는 고장 위성(G12)이 있으면 먼저 빼버림(FDE)!
   - 현재 내 위치가 아니라, 아까 책갈피 꽂아둔 '142번 칸(0.04초 전)'의 IMU 위치를 꺼내고 안테나 장착 거리(레버암)를 더해서 GPS와 정확히 비교!</pre>
        </div>
        """

    def get_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72;">
            <div style="background-color:#F4F7F4; border-left:4px solid #1E3A2F; padding:12px 16px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#1E3A2F; font-size:14.5px;">[단암시스템즈 항법기술연구소 SW팀 실무 핵심]</b><br>
                방산·항공우주 항법 시스템(유도무기, 무인기, 발사체용 <b>EGI 복합항법장치</b> · <b>항재밍/항기만 위성항법장치</b>)은
                <b>(1) <a href="glossary://raim_fde" style="color:#1E3A2F; font-weight:700;">RAIM/FDE</a> · <a href="glossary://sqm_anti_spoof" style="color:#1E3A2F; font-weight:700;">SQM 항기만</a></b> 무결성 보장,
                <b>(2) <a href="glossary://lever_arm" style="color:#1E3A2F; font-weight:700;">레버암</a> · <a href="glossary://time_tag_latency" style="color:#1E3A2F; font-weight:700;">타임태그 링버퍼</a> · <a href="glossary://coning_sculling" style="color:#1E3A2F; font-weight:700;">코닝/스컬링 진동 보정</a></b>,
                <b>(3) 광대역·다중경로 억제를 위한 <a href="glossary://stap_sfap" style="color:#1E3A2F; font-weight:700;">STAP / SFAP</a></b> 기술이 SW 핵심 경쟁력입니다.
            </div>

            <h2 style="color:#1E3A2F; margin-top:0;">1. RAIM · FDE 무결성 감시 및 카이제곱 검정</h2>
            <p>
                상태 미지수가 4개<code>(X, Y, Z, b)</code>이므로 가시 위성이 <b>5기 이상</b>이면 최소자승 해의 잉여 잔차 에너지(<b>WSSE</b>)를 계산해
                <a href="glossary://raim_fde" style="color:#1E3A2F; font-weight:700;">카이제곱(χ²) 고장 검출</a>을 수행하며,
                <b>6기 이상</b>이면 정규화 잔차 <code style="color:#dc2626;">|r_i| / (σ_i √S_ii)</code>가 최대인 고장 위성을 식별해 즉시 측위에서 분리(<b>FDE</b>)합니다.
            </p>

            <h2 style="color:#1E3A2F;">2. Pull-Off 스푸핑과 SQM (Signal Quality Monitoring)</h2>
            <p>
                <a href="glossary://spoofing" style="color:#1E3A2F; font-weight:700;">기만(Spoofing)</a> 공격은 정상 PRN 피크에 동위상으로 중첩한 뒤 +3~6 dB 출력 우세로 DLL 루프를 납치(Pull-Off)합니다.
                두 상관 피크가 분리될 때 발생하는 상관 삼각형 비대칭을
                <a href="glossary://sqm_anti_spoof" style="color:#1E3A2F; font-weight:700;">SQM (M_Δ 지표)</a>과
                칼만 <a href="glossary://chi_square_gating" style="color:#1E3A2F; font-weight:700;">카이제곱 이노베이션 게이팅</a>으로 조기 탐지합니다.
            </p>

            <h2 style="color:#1E3A2F;">3. EGI 레버암 · 타임태그 링버퍼 · 광대역 STAP</h2>
            <ul>
                <li>
                    <b><a href="glossary://lever_arm" style="color:#1E3A2F; font-weight:700;">레버암(Lever-Arm) 보정</a></b>:
                    동체 IMU와 CRPA 안테나 간 이격 벡터 <code>l^b</code>에 따른 선회 가속 오차를 <code>p_GNSS = p_INS + C_b^n l^b</code>로 정밀 보상합니다.
                </li>
                <li>
                    <b><a href="glossary://time_tag_latency" style="color:#1E3A2F; font-weight:700;">타임태그 지연 · IMU 링버퍼</a></b>:
                    마하 1 비행 시 40 ms 지연은 13.6 m 편향을 유발하므로, 1PPS 에지 인덱스로 과거 INS 상태를 추출해 동일 에포크 잔차를 형성합니다.
                </li>
                <li>
                    <b><a href="glossary://stap_sfap" style="color:#1E3A2F; font-weight:700;">STAP (시공간 적응 신호처리)</a></b>:
                    N소자 뒤에 L탭 FIR 지연선을 결합(<code>N×L</code> 자유도)해 광대역 및 기체 반사 다중경로 재밍까지 전 대역 널링합니다.
                </li>
            </ul>
        </div>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. RAIM 카이제곱(χ²) 무결성 검정 통계량(WSSE) 및 FDE 정규화 잔차",
                latex_expr=(
                    r"\mathbf{S} = \mathbf{I}_N - \mathbf{H}\left(\mathbf{H}^T \mathbf{W} \mathbf{H}\right)^{-1}\mathbf{H}^T \mathbf{W}, \qquad \hat{\mathbf{r}} = \mathbf{S}\,\Delta\boldsymbol{\rho}"
                    "\n"
                    r"\mathrm{WSSE} = \hat{\mathbf{r}}^T \mathbf{W} \hat{\mathbf{r}} \ \underset{\mathcal{H}_0}{\overset{\mathcal{H}_1}{\gtrless}} \ \chi^2_{N-4, \, 1-P_{\mathrm{FA}}}, \qquad w_i = \frac{|\hat{r}_i|}{\sigma_i \sqrt{S_{ii}}}"
                ),
                intuitive_summary="최소자승 투영 여공간 행렬 S로 의사거리 잔차 벡터 r_hat을 구한 뒤, 가중 제곱합 WSSE가 자유도 (N-4) 카이제곱 임계값을 넘으면 고장으로 판정하고 정규화 잔차 w_i가 최대인 위성을 분리(FDE)합니다.",
                symbols=[
                    SymbolSpec(r"\mathbf{S}", "잔차 투영(Parity) 행렬", "R^{N×N}", "관측 벡터를 최소자승 오차(잔차) 부공간으로 투영하는 멱등 행렬", "raim_fde"),
                    SymbolSpec(r"\mathrm{WSSE}", "가중 잔차 제곱합 통계량", "-", "정상 상태에서 자유도 (N-4)인 카이제곱(χ²) 분포를 따르는 무결성 지표", "raim_fde"),
                    SymbolSpec("w_i", "i번째 위성 정규화 잔차", "σ", "각 위성의 고장 기여도를 표준정규분포(N(0,1)) 스케일로 정규화한 값", "raim_fde"),
                ],
            ),
            FormulaSpec(
                title="수식 2. Pull-Off 스푸핑 합성 상관 함수 및 SQM 비대칭 탐지 지표",
                latex_expr=(
                    r"R_{\mathrm{comp}}(\tau) = R_{\mathrm{auth}}(\tau) + \sqrt{\mathrm{S/A}} \cdot R_{\mathrm{spoof}}\left(\tau - \Delta\tau_{\mathrm{sp}}(t)\right), \qquad M_{\Delta} = \frac{\left| R_E - R_L \right|}{2 \, R_P}"
                ),
                intuitive_summary="기만 신호가 실제 상관 피크에서 Δτ_sp(t)만큼 갈라져 나올 때 Early와 Late 상관기의 대칭이 무너지며, SQM 비대칭 지표 M_Δ가 임계값을 초과하여 기만 경보를 울립니다.",
                symbols=[
                    SymbolSpec(r"\mathrm{S/A}", "기만 대 정상 신호 전력비", "배 (선형)", "수신기 추적 루프를 장악하기 위해 스푸퍼가 높이는 전력 우세비", "spoofing"),
                    SymbolSpec(r"\Delta\tau_{\mathrm{sp}}(t)", "기만 코드 오프셋 궤적", "chips", "시간에 따라 서서히 증가하며 DLL 루프를 납치하는 코드 지연량", "sqm_anti_spoof"),
                    SymbolSpec(r"M_{\Delta}", "SQM Delta 비대칭 지표", "-", "상관 삼각형의 좌우 경사 비대칭 왜곡을 정량화한 항기만 통계량", "sqm_anti_spoof"),
                ],
            ),
            FormulaSpec(
                title="수식 3. EGI 레버암(Lever-Arm) 위치·속도 변환 및 타임태그 지연 보상 방정식",
                latex_expr=(
                    r"\mathbf{p}_{\mathrm{GNSS}}(t_k) = \mathbf{p}_{\mathrm{INS}}(t_k) + \mathbf{C}_b^n(t_k)\,\mathbf{l}^b, \qquad \mathbf{v}_{\mathrm{GNSS}}(t_k) = \mathbf{v}_{\mathrm{INS}}(t_k) + \mathbf{C}_b^n(t_k)\left( \boldsymbol{\omega}_{ib}^b \times \mathbf{l}^b \right)"
                    "\n"
                    r"\mathbf{z}_{\mathrm{sync}}(t_{\mathrm{PPS}}) = \tilde{\mathbf{p}}_{\mathrm{GNSS}}(t_{\mathrm{PPS}}) - \left[ \hat{\mathbf{p}}_{\mathrm{INS\_buf}}(t_{\mathrm{PPS}}) + \hat{\mathbf{C}}_b^n(t_{\mathrm{PPS}})\,\mathbf{l}^b \right]"
                ),
                intuitive_summary="안테나 위치 측정치는 IMU 위치에 자세행렬 C_b^n으로 회전시킨 레버암 벡터 l^b를 더한 값이며, 1PPS 시점의 과거 INS 링버퍼 상태와 매칭해야 통신 지연 오차가 사라집니다.",
                symbols=[
                    SymbolSpec(r"\mathbf{l}^b", "동체 좌표계 레버암 벡터", "m", "IMU 중심으로부터 GNSS(CRPA) 안테나 위상중심까지의 물리적 장착 오프셋", "lever_arm"),
                    SymbolSpec(r"\mathbf{C}_b^n", "동체(b)→항법(n) 자세행렬", "R^{3×3}", "스트랩다운 INS가 계산한 롤·피치·요 회전 변환 행렬(DCM)", "lever_arm"),
                    SymbolSpec(r"\hat{\mathbf{p}}_{\mathrm{INS\_buf}}(t_{\mathrm{PPS}})", "1PPS 시점 링버퍼 INS 위치", "m", "GNSS 유효 측정 시각(1PPS 에지)에 정확히 대응되는 과거 INS 상태", "time_tag_latency"),
                ],
            ),
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <b style="color:#1E3A2F; font-size:14px;">단암 항법기술연구소 실무 미션 (Observation Guide)</b>
            <ol style="margin-top:8px; margin-bottom:4px; padding-left:20px;">
                <li style="margin-bottom:6px;">
                    <b>미션 1 (RAIM · FDE 고장 위성 자동 배제):</b> <b>[G12 의사거리 오차 주입]</b>을 <code style="color:#dc2626;">0 m → 60 m</code>로 올려보세요.
                    2D 그래프 1에서 <code>G12</code>의 정규화 잔차가 빨간 임계선을 넘어 FDE가 작동해 위치 오차가 <code>1.5 m</code> 수준으로 복원되는 것을 확인하세요.
                </li>
                <li style="margin-bottom:6px;">
                    <b>미션 2 (Pull-Off 스푸핑과 SQM 경보 구간):</b> <b>[기만 신호 코드 오프셋]</b>을 <code style="color:#d97706;">0.00 → 0.35 → 1.20 chip</code>으로 움직여 보세요.
                    피크가 갈라지는 <code>0.15 ~ 0.65 chip</code> 구간에서 2D 그래프 2의 합성 상관 피크 대칭이 깨지며 <b>SQM 비대칭 경보</b>가 울립니다.
                </li>
                <li>
                    <b>미션 3 (광대역 STAP 탭 수 효과):</b> <b>[STAP 시간 지연 탭 (L)]</b>을 <code style="color:#2563eb;">1 탭(SAP) → 7 탭(STAP)</code>으로 늘리고 [그래프 3 · 4 크게 보기]로 전환해 보세요.
                    2D 그래프 4에서 ±10 MHz 전 대역에 걸쳐 널 깊이가 <code>-50 dB</code> 이하로 깊어지는 것을 확인하세요.
                </li>
            </ol>
        </div>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <div style="line-height: 1.68;">
            <h2 style="color:#1E3A2F; margin-top:0;">단암시스템즈 항법기술연구소 SW팀 실전 임베디드 아키텍처 · C99 핵심 코드</h2>

            <div style="background-color:#F4F7F4; border:1px solid #D8E2DB; border-left:4px solid #1E3A2F; padding:14px 16px; border-radius:6px; margin-bottom:18px;">
                <b style="color:#1E3A2F; font-size:14px;">유도무기/무인기용 EGI + 항재밍(AJU) RTOS 멀티태스크 아키텍처 표준</b>
                <table width="100%" cellspacing="0" cellpadding="6" style="border-collapse:collapse; margin-top:8px; font-size:12.5px;">
                    <tr style="background-color:#e2e8f0; font-weight:700; color:#0f172a;">
                        <td>RTOS 태스크 / ISR</td>
                        <td>실행 주기</td>
                        <td>우선순위</td>
                        <td>핵심 담당 알고리즘 및 SW팀 구현 포인트</td>
                    </tr>
                    <tr style="border-bottom:1px solid #cbd5e1;">
                        <td><code>ISR_1PPS_Capture</code></td>
                        <td>1 Hz (에지)</td>
                        <td>최고 (HW)</td>
                        <td>32비트 타이머 하드웨어 캡처 래치, 현재 IMU 링버퍼 인덱스 스냅샷 저장</td>
                    </tr>
                    <tr style="border-bottom:1px solid #cbd5e1;">
                        <td><code>ISR_Correlator_1kHz</code></td>
                        <td>1000 Hz (1ms)</td>
                        <td>매우 높음</td>
                        <td>FPGA 상관기(E/P/L) 읽기, 3차 FLL-assisted PLL/DLL + INS 도플러 보조, SQM 감시</td>
                    </tr>
                    <tr style="border-bottom:1px solid #cbd5e1;">
                        <td><code>Task_INS_Strapdown</code></td>
                        <td>400 Hz (2.5ms)</td>
                        <td>높음</td>
                        <td>SPI IMU 샘플링, <b>Coning & Sculling 진동 보상</b>, 쿼터니언 자세·속도·위치 적분, 링버퍼 기록</td>
                    </tr>
                    <tr style="border-bottom:1px solid #cbd5e1;">
                        <td><code>Task_CRPA_STAP</code></td>
                        <td>100 Hz (10ms)</td>
                        <td>중간</td>
                        <td>공간/시공간 공분산 행렬 R_xx 읽기, 대각 로딩 + 복소 Cholesky 분해로 가중치 w 갱신</td>
                    </tr>
                    <tr>
                        <td><code>Task_EGI_Kalman_RAIM</code></td>
                        <td>10 Hz (100ms)</td>
                        <td>일반</td>
                        <td><b>RAIM/FDE 고장 위성 배제</b>, 1PPS 링버퍼 매칭 + <b>레버암 보정</b>, 카이제곱 게이팅, UD/Joseph ESKF 갱신</td>
                    </tr>
                </table>
            </div>

            <h3 style="color:#0f172a;">1. 스트랩다운 INS 2-샘플 코닝(Coning) & 스컬링(Sculling) 진동 보정 C99 구현</h3>
            <p>
                유도무기 비행 중 고주파 기체 진동에 의해 자이로/가속도계 적분이 틀어지는 것을 막기 위해
                400 Hz INS 루프에서 반드시 수행하는 외적(Cross Product) 보상 코드입니다.
            </p>
            <pre style="background-color:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#include &lt;math.h&gt;
#include &lt;stdint.h&gt;

typedef struct { float x, y, z; } Vec3f_t;

static inline Vec3f_t Vec3_Cross(Vec3f_t a, Vec3f_t b) {
    return (Vec3f_t){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

/* 400 Hz 스트랩다운 INS 루프: 이전 증분(dtheta_prev, dv_prev)과 현재 증분(dtheta_curr, dv_curr)으로
 * 회전 벡터(phi)와 속도 증분(dv_rot_scull)을 정밀 합성 */
void INS_ComputeConingAndSculling(Vec3f_t dtheta_prev, Vec3f_t dv_prev,
                                  Vec3f_t dtheta_curr, Vec3f_t dv_curr,
                                  Vec3f_t *out_rot_vec, Vec3f_t *out_vel_inc) {
    /* 1. 2-샘플 코닝(Coning) 보정: phi = dtheta_curr + (2/3) * (dtheta_prev x dtheta_curr) */
    Vec3f_t coning = Vec3_Cross(dtheta_prev, dtheta_curr);
    out_rot_vec-&gt;x = dtheta_curr.x + (2.0f / 3.0f) * coning.x;
    out_rot_vec-&gt;y = dtheta_curr.y + (2.0f / 3.0f) * coning.y;
    out_rot_vec-&gt;z = dtheta_curr.z + (2.0f / 3.0f) * coning.z;

    /* 2. 회전 보상(1/2 * dtheta x dv) + 2-샘플 스컬링(Sculling) 보정 */
    Vec3f_t rot_comp = Vec3_Cross(dtheta_curr, dv_curr);
    Vec3f_t scull_1  = Vec3_Cross(dtheta_prev, dv_curr);
    Vec3f_t scull_2  = Vec3_Cross(dv_prev, dtheta_curr);

    out_vel_inc-&gt;x = dv_curr.x + 0.5f * rot_comp.x + (2.0f / 3.0f) * (scull_1.x + scull_2.x);
    out_vel_inc-&gt;y = dv_curr.y + 0.5f * rot_comp.y + (2.0f / 3.0f) * (scull_1.y + scull_2.y);
    out_vel_inc-&gt;z = dv_curr.z + 0.5f * rot_comp.z + (2.0f / 3.0f) * (scull_1.z + scull_2.z);
}</pre>

            <h3 style="color:#0f172a;">2. 1PPS 링버퍼 타임태그 매칭 + 레버암(Lever-Arm) 보상 + 카이제곱(χ²) 게이팅</h3>
            <pre style="background-color:#0f172a; color:#38bdf8; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#define RING_BUF_SIZE 256  /* 400 Hz 기준 0.64초 이력 보관 (2의 거듭제곱 크기로 비트 마스킹) */

typedef struct {
    uint32_t tim_tick;
    double   pos_enu[3];
    float    vel_enu[3];
    float    C_bn[3][3];   /* 1PPS 시점의 동체->항법 자세변환행렬(DCM) */
    float    omega_b[3];   /* 1PPS 시점의 바이어스 보상된 동체 각속도 [rad/s] */
} InsRingNode_t;

/* 1PPS 캡처 시각에 해당하는 과거 INS 링버퍼 슬롯에서 레버암(l_body)을 보상한 칼만필터 잔차 z 계산 */
int EGI_ComputeLeverArmSyncedInnovation(const InsRingNode_t *pps_node,
                                        const float lever_body_m[3],
                                        const double gnss_pos_enu[3],
                                        float z_pos_out[3],
                                        const float S_inv_3x3[3][3],
                                        float chi2_gate_thresh) {
    /* 1. 안테나 레버암 오프셋을 항법 좌표계로 변환: l_nav = C_bn * l_body */
    float l_nav[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i &lt; 3; ++i) {
        for (int j = 0; j &lt; 3; ++j) {
            l_nav[i] += pps_node-&gt;C_bn[i][j] * lever_body_m[j];
        }
        /* 2. 동일 시각(t_PPS), 동일 위치(안테나 위상중심) 기준 측정 잔차(Innovation) 계산 */
        z_pos_out[i] = (float)(gnss_pos_enu[i] - (pps_node-&gt;pos_enu[i] + (double)l_nav[i]));
    }

    /* 3. 마할라노비스 거리(카이제곱 통계량) 계산: gamma = z^T * S^-1 * z */
    float mahalanobis_sq = 0.0f;
    for (int i = 0; i &lt; 3; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j &lt; 3; ++j) row_sum += S_inv_3x3[i][j] * z_pos_out[j];
        mahalanobis_sq += z_pos_out[i] * row_sum;
    }

    /* 카이제곱 임계값(예: 3자유도 99% = 11.34f) 초과 시 기만/다중경로 글리치로 판정하여 기각! */
    if (mahalanobis_sq &gt; chi2_gate_thresh) {
        return -1; /* Measurement Rejected (Gated Out) */
    }
    return 0; /* Valid Measurement */
}</pre>
        </div>
        """
