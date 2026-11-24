"""
Module 01: Coordinate Frames (LLH <-> ECEF <-> ENU), IEEE 754 float32 vs float64
Quantization Error Analysis, and GPS Time (Week/TOW) & 1PPS MCU Synchronization.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

from core.coords import (
    analyze_coordinate_pipeline,
    generate_wgs84_3d_globe_and_enu_triad,
    simulate_1pps_and_mcu_clock,
    simulate_float32_vs_float64_trajectory,
    utc_to_gpst,
)
from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec

# Ensure Korean fonts render cleanly on Windows Matplotlib plots
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Module01CoordsTime(BaseLabModule):
    """Module 1: Coordinate Frames, Floating-Point Precision, and GPS Time / 1PPS."""

    module_id = "m01_coords_time"
    module_title = "M01. 좌표계 변환 · float32/64 정밀도 · GPST/1PPS"
    module_subtitle = "WGS-84 LLH ↔ ECEF ↔ ENU 변환과 단정도(float32) 양자화 오차, GPS Week/TOW 및 1PPS 하드웨어 시각 동기"
    module_badge = "MOD 01"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.run_simulation()

    def build_parameter_widget(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

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

        # 1. Reference LLH Group
        g_ref = QGroupBox("1. 기준점 측지 좌표 (Reference LLH)")
        g_ref.setStyleSheet(group_style)
        f_ref = QFormLayout(g_ref)
        f_ref.setSpacing(8)

        self.spin_lat = QDoubleSpinBox()
        self.spin_lat.setRange(-89.9999, 89.9999)
        self.spin_lat.setDecimals(4)
        self.spin_lat.setSingleStep(0.05)
        self.spin_lat.setValue(37.5665)
        self.spin_lat.setSuffix(" ° (N)")
        self.spin_lat.valueChanged.connect(lambda _: self.run_simulation())
        f_ref.addRow("기준 위도 (φ_ref):", self.spin_lat)

        self.spin_lon = QDoubleSpinBox()
        self.spin_lon.setRange(-180.0, 180.0)
        self.spin_lon.setDecimals(4)
        self.spin_lon.setSingleStep(0.05)
        self.spin_lon.setValue(126.9780)
        self.spin_lon.setSuffix(" ° (E)")
        self.spin_lon.valueChanged.connect(lambda _: self.run_simulation())
        f_ref.addRow("기준 경도 (λ_ref):", self.spin_lon)

        self.spin_alt = QDoubleSpinBox()
        self.spin_alt.setRange(-500.0, 50000.0)
        self.spin_alt.setDecimals(1)
        self.spin_alt.setSingleStep(10.0)
        self.spin_alt.setValue(65.0)
        self.spin_alt.setSuffix(" m")
        self.spin_alt.valueChanged.connect(lambda _: self.run_simulation())
        f_ref.addRow("타원체 고도 (h_ref):", self.spin_alt)
        layout.addWidget(g_ref)

        # 2. Target Local ENU Offset & Float32 Test Radius
        g_enu = QGroupBox("2. 표적 ENU 변위 · float32 정밀도 실험")
        g_enu.setStyleSheet(group_style)
        f_enu = QFormLayout(g_enu)
        f_enu.setSpacing(8)

        self.spin_east = QDoubleSpinBox()
        self.spin_east.setRange(-50000.0, 50000.0)
        self.spin_east.setDecimals(2)
        self.spin_east.setSingleStep(25.0)
        self.spin_east.setValue(250.0)
        self.spin_east.setSuffix(" m (East)")
        self.spin_east.valueChanged.connect(lambda _: self.run_simulation())
        f_enu.addRow("동쪽 변위 (E):", self.spin_east)

        self.spin_north = QDoubleSpinBox()
        self.spin_north.setRange(-50000.0, 50000.0)
        self.spin_north.setDecimals(2)
        self.spin_north.setSingleStep(25.0)
        self.spin_north.setValue(400.0)
        self.spin_north.setSuffix(" m (North)")
        self.spin_north.valueChanged.connect(lambda _: self.run_simulation())
        f_enu.addRow("북쪽 변위 (N):", self.spin_north)

        self.spin_up = QDoubleSpinBox()
        self.spin_up.setRange(-10000.0, 50000.0)
        self.spin_up.setDecimals(2)
        self.spin_up.setSingleStep(10.0)
        self.spin_up.setValue(30.0)
        self.spin_up.setSuffix(" m (Up)")
        self.spin_up.valueChanged.connect(lambda _: self.run_simulation())
        f_enu.addRow("천정 변위 (U):", self.spin_up)

        self.slider_f32_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_f32_radius.setRange(5, 100)  # 0.5m to 10.0m
        self.slider_f32_radius.setValue(20)      # 2.0m
        self.lbl_f32_radius = QLabel("2.0 m")
        self.lbl_f32_radius.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_f32_radius.valueChanged.connect(self._on_radius_changed)

        rad_row = QHBoxLayout()
        rad_row.addWidget(self.slider_f32_radius, stretch=1)
        rad_row.addWidget(self.lbl_f32_radius)
        f_enu.addRow("원형 궤적 반경 (R):", rad_row)
        layout.addWidget(g_enu)

        # 3. GPS Time & 1PPS Hardware Sync Group
        g_time = QGroupBox("3. GPST 윤초 · 1PPS / MCU 클럭 동기화")
        g_time.setStyleSheet(group_style)
        f_time = QFormLayout(g_time)
        f_time.setSpacing(8)

        self.spin_leap = QSpinBox()
        self.spin_leap.setRange(0, 35)
        self.spin_leap.setValue(18)
        self.spin_leap.setSuffix(" 초 (GPST - UTC)")
        self.spin_leap.valueChanged.connect(lambda _: self.run_simulation())
        f_time.addRow("현재 윤초 (Δt_LS):", self.spin_leap)

        self.slider_jitter = QSlider(Qt.Orientation.Horizontal)
        self.slider_jitter.setRange(5, 150)  # 5 ns to 150 ns
        self.slider_jitter.setValue(25)
        self.lbl_jitter = QLabel("25 ns")
        self.lbl_jitter.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_jitter.valueChanged.connect(self._on_jitter_changed)

        j_row = QHBoxLayout()
        j_row.addWidget(self.slider_jitter, stretch=1)
        j_row.addWidget(self.lbl_jitter)
        f_time.addRow("1PPS 하드웨어 지터:", j_row)

        self.slider_drift = QSlider(Qt.Orientation.Horizontal)
        self.slider_drift.setRange(1, 100)  # 1 ppm to 100 ppm
        self.slider_drift.setValue(20)
        self.lbl_drift = QLabel("20 ppm")
        self.lbl_drift.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_drift.valueChanged.connect(self._on_drift_changed)

        d_row = QHBoxLayout()
        d_row.addWidget(self.slider_drift, stretch=1)
        d_row.addWidget(self.lbl_drift)
        f_time.addRow("MCU 크리스탈 오차:", d_row)

        self.slider_uart = QSlider(Qt.Orientation.Horizontal)
        self.slider_uart.setRange(30, 550)  # 30 ms to 550 ms
        self.slider_uart.setValue(180)
        self.lbl_uart = QLabel("180 ms")
        self.lbl_uart.setStyleSheet("font-weight:700; color:#d97706;")
        self.slider_uart.valueChanged.connect(self._on_uart_changed)

        u_row = QHBoxLayout()
        u_row.addWidget(self.slider_uart, stretch=1)
        u_row.addWidget(self.lbl_uart)
        f_time.addRow("UART NMEA 지연:", u_row)

        layout.addWidget(g_time)
        return container

    def _on_radius_changed(self, val: int) -> None:
        self.lbl_f32_radius.setText(f"{val / 10.0:.1f} m")
        self.run_simulation()

    def _on_jitter_changed(self, val: int) -> None:
        self.lbl_jitter.setText(f"{val} ns")
        self.run_simulation()

    def _on_drift_changed(self, val: int) -> None:
        self.lbl_drift.setText(f"{val} ppm")
        self.run_simulation()

    def _on_uart_changed(self, val: int) -> None:
        self.lbl_uart.setText(f"{val} ms")
        self.run_simulation()

    def build_simulation_Plot_widget(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        # Top Telemetric Readout Banner (3 Cards: ECEF/ENU Coordinates, Float32 Quantization, GPST/TOW)
        cards_frame = QFrame()
        cards_frame.setStyleSheet(
            """
            QFrame {
                background-color: #0f172a;
                border-radius: 8px;
            }
            """
        )
        grid = QGridLayout(cards_frame)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)

        self.lbl_coord_readout = QLabel()
        self.lbl_coord_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#e2e8f0; border:none;"
        )
        self.lbl_f32_readout = QLabel()
        self.lbl_f32_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#fca5a5; border:none;"
        )
        self.lbl_time_readout = QLabel()
        self.lbl_time_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#7dd3fc; border:none;"
        )

        grid.addWidget(self.lbl_coord_readout, 0, 0)
        grid.addWidget(self.lbl_f32_readout, 0, 1)
        grid.addWidget(self.lbl_time_readout, 0, 2)
        vbox.addWidget(cards_frame)

        # Matplotlib 2D Figure Canvas (Tab 0)
        self.fig = Figure(figsize=(10.2, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)

        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        self.spin_lat.blockSignals(True)
        self.spin_lon.blockSignals(True)
        self.spin_alt.blockSignals(True)
        self.spin_east.blockSignals(True)
        self.spin_north.blockSignals(True)
        self.spin_up.blockSignals(True)
        self.spin_leap.blockSignals(True)
        self.slider_jitter.blockSignals(True)
        self.slider_drift.blockSignals(True)

        self.spin_lat.setValue(float(preset_data.get("lat_deg", 37.5665)))
        self.spin_lon.setValue(float(preset_data.get("lon_deg", 126.9780)))
        self.spin_alt.setValue(float(preset_data.get("alt_m", 65.0)))
        self.spin_east.setValue(float(preset_data.get("target_east_m", 250.0)))
        self.spin_north.setValue(float(preset_data.get("target_north_m", 400.0)))
        self.spin_up.setValue(float(preset_data.get("target_up_m", 30.0)))
        self.spin_leap.setValue(int(preset_data.get("leap_seconds", 18)))

        jitter_val = int(preset_data.get("pps_jitter_ns", 25))
        drift_val = int(preset_data.get("mcu_drift_ppm", 20))
        self.slider_jitter.setValue(jitter_val)
        self.lbl_jitter.setText(f"{jitter_val} ns")
        self.slider_drift.setValue(drift_val)
        self.lbl_drift.setText(f"{drift_val} ppm")

        self.spin_lat.blockSignals(False)
        self.spin_lon.blockSignals(False)
        self.spin_alt.blockSignals(False)
        self.spin_east.blockSignals(False)
        self.spin_north.blockSignals(False)
        self.spin_up.blockSignals(False)
        self.spin_leap.blockSignals(False)
        self.slider_jitter.blockSignals(False)
        self.slider_drift.blockSignals(False)

    def run_simulation(self) -> None:
        ref_lat = self.spin_lat.value()
        ref_lon = self.spin_lon.value()
        ref_alt = self.spin_alt.value()
        east_m = self.spin_east.value()
        north_m = self.spin_north.value()
        up_m = self.spin_up.value()
        f32_radius_m = self.slider_f32_radius.value() / 10.0
        leap_sec = self.spin_leap.value()
        pps_jitter_ns = float(self.slider_jitter.value())
        mcu_drift_ppm = float(self.slider_drift.value())
        uart_delay_ms = float(self.slider_uart.value())

        # 1. Execute core coordinate conversion & float32 analysis
        res = analyze_coordinate_pipeline(ref_lat, ref_lon, ref_alt, east_m, north_m, up_m)
        traj = simulate_float32_vs_float64_trajectory(ref_lat, ref_lon, ref_alt, radius_m=f32_radius_m)

        # 2. Execute GPS Time & 1PPS simulation
        now_utc = datetime.now(timezone.utc)
        gpst_info = utc_to_gpst(now_utc, leap_seconds=leap_sec)
        pps_sim = simulate_1pps_and_mcu_clock(
            duration_sec=12,
            pps_jitter_ns=pps_jitter_ns,
            mcu_drift_ppm=mcu_drift_ppm,
            nmea_uart_delay_ms=uart_delay_ms,
        )

        # Update top telemetric banner
        rx, ry, rz = res.ref_ecef_m
        t_lat, t_lon, t_alt = res.target_llh_f64_deg_m
        self.lbl_coord_readout.setText(
            f"<div style='font-size:11.5px; color:#1E3A2F; font-weight:800;'>WGS-84 좌표 변환 (float64)</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:800; margin-top:3px;'>"
            f"방위 {res.azimuth_deg:.2f}° · 앙각 {res.elevation_deg:.2f}° · {res.slant_range_m:.1f} m</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"표적 LLH ({t_lat:.4f}°, {t_lon:.4f}°, {t_alt:.1f}m) | ECEF ({rx:,.0f}, {ry:,.0f}, {rz:,.0f})</div>"
        )

        max_3d_err = float(np.max(traj["ecef_f32_error_3d_m"]))
        mean_3d_err = float(np.mean(traj["ecef_f32_error_3d_m"]))
        self.lbl_f32_readout.setText(
            f"<div style='font-size:11.5px; color:#B45309; font-weight:800;'>float32 양자화 오차 경고</div>"
            f"<div style='font-size:13.5px; color:#DC2626; font-weight:900; margin-top:3px;'>"
            f"최대 {max_3d_err*100:.1f} cm · 평균 {mean_3d_err*100:.1f} cm</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"표적 ENU 오차 {res.f32_enu_error_norm_m*100:.1f} cm · N(φ)={res.prime_vertical_radius_n_m:,.0f} m</div>"
        )

        self.lbl_time_readout.setText(
            f"<div style='font-size:11.5px; color:#1D4ED8; font-weight:800;'>실시간 GPST · 1PPS 시각</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:800; margin-top:3px;'>"
            f"Week {gpst_info['gps_week_full']} (Mod {gpst_info['gps_week_mod1024']}) · 윤초 +{leap_sec}s</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"TOW {gpst_info['tow_seconds']:,.2f} s | GPST {gpst_info['gpst_iso']}</div>"
        )

        self.set_easy_coach_message(
            f"<b>[사수의 실시간 쉬운 해설]</b> (1) 지금 드론이 반경 <b>{f32_radius_m:.1f}m</b> 원을 돌고 있는데, "
            f"좌표를 32비트 <code>float</code>로 저장했더니 파란색 원이 <b>최대 {max_3d_err*100:.0f}cm씩 뚝뚝 끊기는 빨간 계단</b>으로 찌그러졌습니다. (64비트 <code>double</code> 필수 이유) &nbsp;|&nbsp; "
            f"(2) UART 통신은 <b>{uart_delay_ms:.0f}ms</b> 늦게 도착하지만, <b>1PPS 하드웨어 핀</b>(초록선)으로 MCU 시계 오차(<b>{mcu_drift_ppm:.0f}ppm</b>)를 잡아 시각 오차를 <b>0.05μs 이내</b>로 유지합니다."
        )

        globe_data = generate_wgs84_3d_globe_and_enu_triad(ref_lat, ref_lon, ref_alt)

        # Tab 0: Clean 2D graphs (1x2 default or 2x2)
        self.fig.clear()
        ax1, ax2, ax3, ax4 = self.allocate_2d_axes(self.fig)
        self._draw_2d_plots(
            ax1, ax2, ax3, ax4,
            traj, pps_sim, f32_radius_m, uart_delay_ms, mcu_drift_ppm, pps_jitter_ns
        )
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        # Tab 1: Dedicated 3D Earth Ellipsoid + ENU Tangent Plane
        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            self._draw_3d_globe_enu(ax_3d, globe_data, ref_lat, ref_lon)
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def _draw_2d_plots(
        self,
        ax1,
        ax2,
        ax3,
        ax4,
        traj: Dict[str, np.ndarray],
        pps_sim: Dict[str, np.ndarray],
        f32_radius_m: float,
        uart_delay_ms: float,
        mcu_drift_ppm: float,
        pps_jitter_ns: float,
    ) -> None:
        # Subplot 1: float64 True Circle vs float32 Stair-Stepped Trajectory in ENU
        if ax1 is not None:
            ax1.plot(
                traj["true_east_m"],
                traj["true_north_m"],
                color="#2563eb",
                linewidth=2.2,
                label="float64 (double) 실제 궤적",
            )
            ax1.plot(
                traj["f32_east_m"],
                traj["f32_north_m"],
                color="#dc2626",
                linestyle="-",
                marker="o",
                markersize=3.0,
                linewidth=1.3,
                alpha=0.85,
                label="float32 ECEF 계단 양자화",
            )
            ax1.scatter([0.0], [0.0], color="#0f172a", marker="+", s=80, label="기준점 (0, 0)")
            ax1.set_title(
                f"1. ECEF float32 vs float64 궤적 (반경 {f32_radius_m:.1f}m)",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax1.set_xlabel("동쪽 변위 East [m]", fontsize=9.0)
            ax1.set_ylabel("북쪽 변위 North [m]", fontsize=9.0)
            ax1.set_aspect("equal", adjustable="datalim")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8.0)

        # Subplot 2: Position Quantization Error Magnitude along trajectory angle
        if ax2 is not None:
            deg_axis = np.rad2deg(traj["angles_rad"])
            ax2.plot(
                deg_axis,
                traj["ecef_f32_error_3d_m"] * 100.0,
                color="#dc2626",
                linewidth=1.8,
                label="ECEF float32 3D 오차 [cm]",
            )
            ax2.plot(
                deg_axis,
                traj["llh_f32_error_horiz_m"] * 100.0,
                color="#d97706",
                linewidth=1.5,
                linestyle="--",
                label="위경도(deg) float32 수평 오차 [cm]",
            )
            ax2.axhline(50.0, color="#64748b", linestyle=":", linewidth=1.2, label="ECEF 1-ULP 한계 (50 cm)")
            ax2.set_title(
                "2. 단정도(float32) 양자화 위치 오차 [cm]",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax2.set_xlabel("원형 궤적 방위각 [deg]", fontsize=9.0)
            ax2.set_ylabel("위치 오차 크기 [cm]", fontsize=9.0)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right", fontsize=8.0)

        # Subplot 3: 1PPS Hardware Pulse vs Delayed UART NMEA Packet Timing Diagram
        if ax3 is not None:
            t_ms = pps_sim["waveform_t_ms"]
            ax3.fill_between(
                t_ms,
                1.15,
                1.15 + 0.75 * pps_sim["pps_wave"],
                step="post",
                color="#16a34a",
                alpha=0.75,
                label="1PPS 하드웨어 핀 (정초 0.000s, <50ns)",
            )
            ax3.fill_between(
                t_ms,
                0.15,
                0.15 + 0.75 * pps_sim["uart_wave"],
                step="post",
                color="#d97706",
                alpha=0.75,
                label=f"UART NMEA 도착 (+{uart_delay_ms:.0f}ms 지연)",
            )
            for sec_k in range(3):
                edge_x = sec_k * 1000.0
                ax3.axvline(edge_x, color="#15803d", linestyle="--", linewidth=1.2)
                ax3.annotate(
                    f"정초 {sec_k}s\n(TIM 캡처)",
                    xy=(edge_x, 1.92),
                    xytext=(edge_x + 35, 2.00),
                    fontsize=7.8,
                    fontweight="bold",
                    color="#15803d",
                )
            ax3.set_ylim(0.0, 2.38)
            ax3.set_yticks([0.52, 1.52])
            ax3.set_yticklabels(["UART RX", "1PPS GPIO"], fontsize=8.5, fontweight="bold")
            ax3.set_title(
                "3. 1PPS 인터럽트 vs UART 패킷 지연 타이밍도",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax3.set_xlabel("경과 시간 [ms]", fontsize=9.0)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="lower right", fontsize=8.0)

        # Subplot 4: MCU Free-Running Crystal Drift vs 1PPS-Disciplined Clock Error
        if ax4 is not None:
            ax4.plot(
                pps_sim["sec_axis"],
                pps_sim["free_running_err_us"],
                color="#dc2626",
                marker="s",
                markersize=3.5,
                linewidth=1.9,
                label=f"동기화 없는 MCU 표류 ({mcu_drift_ppm:.0f} ppm)",
            )
            ax4.plot(
                pps_sim["fine_t"],
                pps_sim["sawtooth_reset_only_us"],
                color="#d97706",
                linewidth=1.3,
                linestyle="--",
                label="단순 1PPS 리셋 (톱니파 오차)",
            )
            ax4.plot(
                pps_sim["fine_t"],
                pps_sim["disciplined_fll_err_us"],
                color="#16a34a",
                linewidth=2.0,
                label=f"1PPS + TIM 캡처 FLL 보정 ({pps_jitter_ns:.0f}ns급)",
            )
            ax4.set_title(
                "4. MCU 클럭 드리프트 vs 1PPS 동기화 오차 [μs]",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax4.set_xlabel("경과 시간 [초]", fontsize=9.0)
            ax4.set_ylabel("시각 동기 오차 [μs]", fontsize=9.0)
            ax4.grid(True, linestyle="--", alpha=0.5)
            ax4.legend(loc="upper left", fontsize=8.0)

    def _draw_3d_globe_enu(
        self,
        ax_3d,
        globe_data: Dict[str, np.ndarray],
        ref_lat: float,
        ref_lon: float,
    ) -> None:
        """Render 3D WGS-84 Earth ellipsoid, ECEF axes, and local ENU tangent plane + triad."""
        ax_3d.plot_wireframe(
            globe_data["globe_x_km"],
            globe_data["globe_y_km"],
            globe_data["globe_z_km"],
            color="#94a3b8",
            linewidth=0.45,
            alpha=0.35,
        )
        eq = globe_data["equator_xyz_km"]
        mer = globe_data["meridian_xyz_km"]
        ax_3d.plot(eq[0], eq[1], eq[2], color="#64748b", linestyle="--", linewidth=1.1, label="적도면 (Equator)")
        ax_3d.plot(mer[0], mer[1], mer[2], color="#64748b", linestyle=":", linewidth=1.1, label="본초자오선 (Lon 0°)")

        # Local tangent plane patch
        ax_3d.plot_surface(
            globe_data["plane_x_km"],
            globe_data["plane_y_km"],
            globe_data["plane_z_km"],
            color="#38bdf8",
            alpha=0.32,
        )

        # Reference ECEF position vector from Earth Center (0, 0, 0)
        p0 = globe_data["ref_ecef_km"]
        ax_3d.plot([0.0, p0[0]], [0.0, p0[1]], [0.0, p0[2]], color="#0f172a", linewidth=2.2, label="ECEF 위치 벡터 r_ref")
        ax_3d.scatter([p0[0]], [p0[1]], [p0[2]], color="#f59e0b", s=55, zorder=6)

        # ENU Orthogonal Basis Triad Vectors at Reference Point
        ev = globe_data["east_vec_km"]
        nv = globe_data["north_vec_km"]
        uv = globe_data["up_vec_km"]
        ax_3d.quiver(p0[0], p0[1], p0[2], ev[0], ev[1], ev[2], color="#dc2626", linewidth=2.2, label="East (동쪽 축)")
        ax_3d.quiver(p0[0], p0[1], p0[2], nv[0], nv[1], nv[2], color="#16a34a", linewidth=2.2, label="North (북쪽 축)")
        ax_3d.quiver(p0[0], p0[1], p0[2], uv[0], uv[1], uv[2], color="#2563eb", linewidth=2.2, label="Up (천정 법선 축)")

        lim = 7200.0
        ax_3d.set_xlim(-lim, lim)
        ax_3d.set_ylim(-lim, lim)
        ax_3d.set_zlim(-lim, lim)
        ax_3d.set_xlabel("ECEF X [km]", fontsize=8.5)
        ax_3d.set_ylabel("ECEF Y [km]", fontsize=8.5)
        ax_3d.set_zlabel("ECEF Z [km]", fontsize=8.5)
        ax_3d.set_title(
            f"[3D] WGS-84 지구 타원체 · 국소 ENU 접평면 (위도 {ref_lat:.2f}°, 경도 {ref_lon:.2f}°)",
            fontsize=11.0,
            fontweight="bold",
            pad=10,
        )
        ax_3d.view_init(elev=24, azim=max(min(ref_lon - 35.0, 175.0), -175.0))
        ax_3d.legend(loc="upper left", fontsize=7.5)

    def get_lv1_easy_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">1. 일상생활 비유로 30초 만에 이해하기</h3>
            <ul>
                <li><b>좌표계 3총사 (LLH vs ECEF vs ENU)</b>:
                    <ul>
                        <li><b>LLH (위도·경도·고도)</b>는 사람이 지도를 볼 때 쓰는 <b>'도로명 주소(서울시 중구...)'</b>입니다.</li>
                        <li><b>ECEF (X, Y, Z)</b>는 지구 한가운데(핵)를 <code>(0,0,0)</code>으로 잡고 자로 잰 <b>'우주 본부용 절대 거리(약 637만 미터)'</b>입니다. 위성과 내 거리를 피타고라스 정리로 쉽게 구할 때 씁니다.</li>
                        <li><b>ENU (동·북·위)</b>는 내 발밑에 돗자리를 펴고 <b>'내 기준 오른쪽(동)으로 몇 m, 앞(북)으로 몇 m'</b>인지 재는 내비게이션용 좌표입니다.</li>
                    </ul>
                </li>
                <li><b>왜 <code>float</code>(32비트)를 쓰면 망할까요?</b>:
                    <code>float</code>는 숫자 7자리까지만 기억하는 메모지입니다. 지구 중심에서 내 위치까지 거리가 <code>6,378,137 m</code>로 이미 7자리이다 보니, 소수점 아래(cm 단위)를 적을 칸이 없어서 <b>50 cm 단위로 반올림</b>해 버립니다!</li>
                <li><b>1PPS 핀이 왜 필요할까요?</b>:
                    GPS가 UART 통신선으로 <i>"지금 12시 정각이야!"</i>라고 글자를 보내면 전송하느라 0.2초가 늦습니다. 그래서 <b>정각 딱 그 순간에 전구(1PPS 핀)를 번쩍!</b> 켜서 시간부터 먼저 찍어두고, 0.2초 뒤에 도착한 글자로 몇 시였는지만 확인하는 것입니다.</li>
            </ul>

            <h3 style="color:#1E3A2F;">2. 그래프 어디를 봐야 하나요?</h3>
            <ul>
                <li><b>2D 그래프 1·2</b>: <b>파란색 매끈한 원(<code>double</code>)</b>과 <b>빨간색 각진 계단(<code>float</code>)</b>을 비교하세요. 50cm 단위로 위치가 뚝뚝 끊기는 것이 눈으로 보입니다.</li>
                <li><b>2D 그래프 3·4</b>: 초록색 <b>1PPS 펄스</b>가 1초마다 딱딱 떨어질 때 노란색 <b>UART 패킷</b>이 얼마나 늦게 도착하는지, 그리고 1PPS가 없으면 빨간선(MCU 시계 오차)이 얼마나 빨리 틀어지는지 보세요.</li>
                <li><b>3D 입체 시각화 탭</b>: 둥근 지구 중심에서 내 위치로 화살표(ECEF)가 꽂히고, 그 위에 파란 네모 돗자리(ENU 평면)가 얹혀 있는 입체 그림입니다.</li>
            </ul>
        </div>
        """

    def get_lv1_easy_c_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">3. C 코드 3단계 한글 흐름도 (수식 몰라도 이해되는 구조)</h3>
            <pre style="background-color:#FAFBF9; border:1px solid #D8E2DB; color:#14532d; padding:12px; border-radius:8px; font-family:'Malgun Gothic',monospace; font-size:13px;">
[1단계: 절대 위치는 무조건 double(64비트) 변수에 담기]
   내_위치_ECEF (double) = 위경도_변환(위도, 경도, 고도);
   기준점_ECEF  (double) = 위경도_변환(출발점_위도, 출발점_경도, 출발점_고도);

[2단계: 큰 숫자끼리 빼기(약 637만m - 637만m = 수백m)까지 반드시 double에서 수행!]
   차이_dX (double) = 내_위치_ECEF.x - 기준점_ECEF.x;

[3단계: 숫자가 작아진 상대 거리(ENU)만 float(32비트)로 바꿔서 빠르게 칼만필터 연산!]
   동쪽_거리 (float) = (float)회전변환(차이_dX, 차이_dY, 차이_dZ);</pre>
        </div>
        """

    def get_theory_html(self) -> str:
        return """
        <div style="line-height: 1.68;">
            <h2 style="color:#1E3A2F; margin-top:0;">1. 항법 소프트웨어의 3대 좌표계 및 직교 변환</h2>
            <p>
                위성 궤도(Broadcast Ephemeris)와 의사거리 방정식은 <a href="glossary://ecef" style="color:#1E3A2F; font-weight:700;">ECEF (Earth-Centered, Earth-Fixed)</a>
                직교좌표계를 기준으로 풀이되지만, 지도 투영·조종 제어·국소 센서 융합은 각각의 목적에 맞춰 아래 3가지 좌표계를 엄밀히 변환하여 사용합니다.
            </p>
            <ul>
                <li>
                    <b><a href="glossary://llh" style="color:#1E3A2F; font-weight:700;">LLH (측지 위도·경도·타원체고)</a></b>:
                    <a href="glossary://wgs84" style="color:#1E3A2F; font-weight:700;">WGS-84</a> 회전 타원체 모델(장반경 a=6,378,137 m, 편평률 f=1/298.257223563)을 따르며,
                    위도(φ)에 따라 달라지는 <a href="glossary://prime_vertical_radius" style="color:#1E3A2F; font-weight:700;">묘유선 곡률반경 N(φ)</a>을 통해 ECEF로 변환됩니다.
                </li>
                <li>
                    <b><a href="glossary://ecef" style="color:#1E3A2F; font-weight:700;">ECEF (X, Y, Z)</a></b>:
                    지심 고정 3차원 데카르트 좌표계로, 곡면 보정 없이 3차원 유클리드 거리 계산이 가능하여 GNSS PVT 연산의 표준 프레임입니다.
                </li>
                <li>
                    <b><a href="glossary://enu" style="color:#1E3A2F; font-weight:700;">ENU (동-북-천정)</a> / <a href="glossary://ned" style="color:#1E3A2F; font-weight:700;">NED (북-동-하강)</a></b>:
                    기준점(φ₀, λ₀) 접평면 위에서 정의되는 국소 항법 좌표계로, 직교 회전행렬 R<sub>ECEF→ENU</sub>를 통해 선형 변위, 앙각·방위각 및 INS 자세 방정식을 구성합니다.
                </li>
            </ul>

            <h2 style="color:#1E3A2F;">2. IEEE 754 부동소수점 정밀도 및 <code style="color:#dc2626;">float32</code> ULP 손실</h2>
            <p>
                Cortex-M4F 등 단정도 FPU 기반 임베디드 타겟에서 ECEF 좌표나 위경도를 32비트 <code style="color:#dc2626;">float</code>로 저장하면
                <b><a href="glossary://float32_quantization" style="color:#1E3A2F; font-weight:700;">단정도 양자화 오차</a></b>가 발생합니다.
                |X| ≈ 6.38×10⁶ m 범위(2²² &lt; |x| &lt; 2²³)에서 IEEE 754 float32(가수 23+1비트)의 1 ULP 간격은 <b>2⁻¹ = 0.5 m</b>에 달하며,
                기준점과의 차분(ΔX = X - X₀)에서 막대한 유효숫자 상쇄(Catastrophic Cancellation)를 유발합니다.
            </p>

            <h2 style="color:#1E3A2F;">3. GPST 연속 시간축과 1PPS 타이머 캡처 인터럽트 동기</h2>
            <p>
                UTC는 지구 자전 속도 보정을 위해 불연속적 <a href="glossary://leap_second" style="color:#1E3A2F; font-weight:700;">윤초(Leap Second)</a>를 삽입하므로
                궤도/의사거리 적분 중 시각 불연속을 방지하기 위해 1980-01-06 기준 연속 흐르는
                <a href="glossary://gpst" style="color:#1E3A2F; font-weight:700;">GPST (Week &amp; TOW)</a>를 사용합니다.
                또한 직렬 프로토콜(UART NMEA/UBX)의 비결정적 패킷 지연(수십~수백 ms)을 배제하기 위해
                <a href="glossary://pps" style="color:#1E3A2F; font-weight:700;">1PPS (Pulse Per Second)</a> 상승 에지를 MCU 하드웨어 타이머 Input Capture로 래치하여 클럭 드리프트(ppm)를 FLL/PLL로 보상해야 합니다.
            </p>
        </div>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. WGS-84 측지 좌표(LLH) → 지구중심 고정 좌표(ECEF) 변환",
                latex_expr=(
                    r"N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}, \quad e^2 = 2f - f^2"
                    "\n"
                    r"X = \left(N(\phi) + h\right)\cos\phi\cos\lambda, \quad Y = \left(N(\phi) + h\right)\cos\phi\sin\lambda, \quad Z = \left(N(\phi)(1 - e^2) + h\right)\sin\phi"
                ),
                intuitive_summary="지구가 적도 쪽이 부풀어 오른 회전 타원체이므로, 위도 φ에서의 묘유선 곡률반경 N(φ)을 먼저 구한 뒤 3차원 직교 좌표(X, Y, Z)로 투영합니다.",
                symbols=[
                    SymbolSpec("a, e^2", "장반경 및 제1이심률 제곱", "m, -", "WGS-84 적도 반지름(6,378,137 m) 및 타원체 납작한 정도(e² ≈ 0.00669438)", "wgs84"),
                    SymbolSpec(r"N(\phi)", "묘유선 곡률반경", "m", "위도 φ에서 타원체 표면 법선이 Z축과 만날 때까지의 곡률반경", "prime_vertical_radius"),
                    SymbolSpec(r"\phi, \lambda, h", "위도, 경도, 타원체고", "rad, rad, m", "사용자의 측지 위도, 경도 및 WGS-84 타원체면 기준 법선 고도", "llh"),
                    SymbolSpec("X, Y, Z", "ECEF 3차원 위치 벡터", "m", "지구 질량 중심을 원점으로 하는 직교 좌표 (크기 ~6.38×10⁶ m)", "ecef"),
                ],
            ),
            FormulaSpec(
                title="수식 2. ECEF 변위 벡터 → 지역 수평 좌표계(ENU) 직교 회전 변환",
                latex_expr=(
                    r"E = -\sin\lambda_0 \cdot \Delta X + \cos\lambda_0 \cdot \Delta Y"
                    "\n"
                    r"N = -\sin\phi_0\cos\lambda_0 \cdot \Delta X - \sin\phi_0\sin\lambda_0 \cdot \Delta Y + \cos\phi_0 \cdot \Delta Z"
                    "\n"
                    r"U = \cos\phi_0\cos\lambda_0 \cdot \Delta X + \cos\phi_0\sin\lambda_0 \cdot \Delta Y + \sin\phi_0 \cdot \Delta Z"
                ),
                intuitive_summary="표적과 기준점의 ECEF 좌표 차이 벡터(ΔX, ΔY, ΔZ)에 기준점 위경도(φ₀, λ₀)로 결정되는 3×3 직교 회전행렬을 곱해 동(E)-북(N)-천정(U) 거리로 변환합니다.",
                symbols=[
                    SymbolSpec(r"\phi_0, \lambda_0", "기준점 위도 및 경도", "rad", "국소 수평 접평면(Tangent Plane)의 원점이 되는 기준 위치", "llh"),
                    SymbolSpec(r"\Delta X, \Delta Y, \Delta Z", "ECEF 상대 변위 벡터", "m", "표적과 기준점의 ECEF 좌표 차이 (X - X₀, Y - Y₀, Z - Z₀)", "ecef"),
                    SymbolSpec("E, N, U", "동쪽, 북쪽, 천정 변위", "m", "기준점으로부터 동쪽(East), 북쪽(North), 하늘(Up) 방향 상대 거리", "enu"),
                ],
            ),
            FormulaSpec(
                title="수식 3. IEEE 754 부동소수점 최소 분해능(ULP)과 GPS 시각(GPST) 관계식",
                latex_expr=(
                    r"\mathrm{ULP}(x) = 2^{\lfloor \log_2 |x| \rfloor - p + 1}, \qquad t_{\mathrm{GPST}} = t_{\mathrm{UTC}} + \Delta t_{\mathrm{LS}} = 604800 \times \mathrm{Week} + \mathrm{TOW}"
                ),
                intuitive_summary="좌표 크기 |x| ≈ 6.38×10⁶ m (2²² < |x| < 2²³)일 때 float32(p=24)의 최소 간격(ULP)은 2⁻¹ = 0.5 m가 되며, GPST는 UTC에 윤초 Δt_LS를 더해 주(Week)와 주내 초(TOW)로 표현합니다.",
                symbols=[
                    SymbolSpec("p", "가수부 유효 비트수", "bits", "IEEE 754 float32는 p=24 (약 7자리), float64는 p=53 (약 16자리)", "float32_quantization"),
                    SymbolSpec(r"\mathrm{ULP}(x)", "최하위 비트 간격", "m", "값 x 근처에서 부동소수점이 표현할 수 있는 가장 작은 간격", "float32_quantization"),
                    SymbolSpec(r"\Delta t_{\mathrm{LS}}", "누적 윤초", "s", "1980년 1월 6일 이후 삽입된 총 윤초 (현재 +18초)", "leap_second"),
                    SymbolSpec(r"\mathrm{Week, TOW}", "GPS 주 번호 및 주내 초", "주, s", "일요일 00:00:00 GPST 기준 경과 주 및 주내 초(0 ~ 604,800 s)", "gpst"),
                ],
            ),
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <b style="color:#1E3A2F; font-size:14px;">슬라이더 조작 실습 미션 (Observation Guide)</b>
            <ol style="margin-top:8px; margin-bottom:4px; padding-left:20px;">
                <li style="margin-bottom:6px;">
                    <b>미션 1 (float32 계단 현상 관찰):</b> 좌측 <b>[원형 궤적 반경 (R)]</b> 슬라이더를 <code style="color:#dc2626;">0.5 m ~ 1.5 m</code>로 줄여보세요.
                    <b>2D 그래프 1</b>에서 파란색(<code style="color:#2563eb;">double</code>)은 매끄러운 원인데 빨간색(<code style="color:#dc2626;">float</code>)은 왜 격자 계단처럼 찌그러지는지 확인하세요.
                </li>
                <li style="margin-bottom:6px;">
                    <b>미션 2 (1PPS vs UART 지연 비교):</b> <b>[UART NMEA 지연]</b> 슬라이더를 <code style="color:#d97706;">450 ms</code>로 늘리고 상단 <b>[그래프 3 · 4 크게 보기]</b>로 전환해보세요.
                    <b>2D 그래프 3</b>에서 UART 패킷 도착 시각만으로 시간을 맞추면 왜 약 0.5초의 치명적 오차가 발생하는지 관찰하세요.
                </li>
                <li>
                    <b>미션 3 (MCU 클럭 드리프트 보정):</b> <b>[MCU 크리스탈 오차]</b>를 <code style="color:#dc2626;">80 ppm</code>으로 높여보세요.
                    <b>2D 그래프 4</b>에서 1PPS 타이머 캡처 보정(초록색)이 수백 μs의 표류를 어떻게 막아내는지 확인하세요.
                </li>
            </ol>
        </div>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <h2 style="color:#1E3A2F; margin-top:0;">실전 임베디드 C/C++ 포팅 가이드 (STM32 / ARM Cortex-M / DSP)</h2>

            <div style="background-color:#FEF2F2; border-left:4px solid #DC2626; padding:12px 14px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#991B1B;">[필수 규칙 1] 절대 위치 좌표(ECEF, LLH)와 시간(TOW)에 <code>float</code>를 쓰지 마라</b><br>
                Cortex-M4F(단정도 FPU)에서 컴파일러 옵션 <code>-fsingle-precision-constant</code>를 켜거나 습관적으로 <code>float</code>를 쓰면
                ECEF 좌표는 <b>50 cm 단위</b>, TOW(최대 604,800초)는 <b>62.5 ms 단위</b>로 양자화되어 항법 필터가 즉시 깨집니다.
                절대 좌표(ECEF, LLH)는 반드시 <code>double</code>(64-bit) 또는 <code>int32_t / int64_t</code> 고정소수점(예: <code>1e-7 deg</code> 단위 정수)으로 보관하고,
                기준점과의 <b>상대 변위(ΔENU)만 <code>float</code></b>로 변환해 칼만필터에서 연산하세요.
            </div>

            <h3 style="color:#0f172a;">1. C99 표준 WGS-84 LLH → ECEF 및 상대 ENU 변환 코드</h3>
            <pre style="background-color:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#include &lt;math.h&gt;
#include &lt;stdint.h&gt;

#define WGS84_A   6378137.0
#define WGS84_F   (1.0 / 298.257223563)
#define WGS84_E2  (2.0 * WGS84_F - WGS84_F * WGS84_F)

typedef struct {
    double x_m;   /* 반드시 64-bit double 사용! */
    double y_m;
    double z_m;
} EcefPos_t;

typedef struct {
    float east_m; /* 기준점 대비 국소 상대 변위는 수 km 이내이므로 float32 연산 가능 */
    float north_m;
    float up_m;
} EnuOffset_t;

void GNSS_LlhToEcef(double lat_rad, double lon_rad, double alt_m, EcefPos_t *out_ecef) {
    const double sin_p = sin(lat_rad);
    const double cos_p = cos(lat_rad);
    const double sin_l = sin(lon_rad);
    const double cos_l = cos(lon_rad);

    const double N = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_p * sin_p);

    out_ecef-&gt;x_m = (N + alt_m) * cos_p * cos_l;
    out_ecef-&gt;y_m = (N + alt_m) * cos_p * sin_l;
    out_ecef-&gt;z_m = (N * (1.0 - WGS84_E2) + alt_m) * sin_p;
}

/* 핵심 팁: 두 ECEF 좌표의 뺄셈(dx, dy, dz)까지 반드시 double에서 마친 뒤 float로 내립니다! */
void GNSS_EcefToLocalEnu(const EcefPos_t *target, const EcefPos_t *ref,
                         double ref_lat_rad, double ref_lon_rad, EnuOffset_t *out_enu) {
    const double dx = target-&gt;x_m - ref-&gt;x_m;
    const double dy = target-&gt;y_m - ref-&gt;y_m;
    const double dz = target-&gt;z_m - ref-&gt;z_m;

    const double sp = sin(ref_lat_rad), cp = cos(ref_lat_rad);
    const double sl = sin(ref_lon_rad), cl = cos(ref_lon_rad);

    out_enu-&gt;east_m  = (float)(-sl * dx + cl * dy);
    out_enu-&gt;north_m = (float)(-sp * cl * dx - sp * sl * dy + cp * dz);
    out_enu-&gt;up_m    = (float)( cp * cl * dx + cp * sl * dy + sp * dz);
}</pre>

            <h3 style="color:#0f172a;">2. STM32 1PPS 타이머 Input Capture + UART NMEA/UBX 시각 동기 패턴</h3>
            <p>
                UART 수신 완료 인터럽트(ISR)에서 현재 시간을 찍으면 직렬 전송 시간 때문에 수백 ms가 밀립니다.
                아래와 같이 <b>32비트 프리러닝 타이머(TIM2)의 Input Capture 채널</b>에 1PPS 핀을 연결해 하드웨어적으로 카운터 값을 래치(Latch)해야 합니다.
            </p>
            <pre style="background-color:#0f172a; color:#38bdf8; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
volatile uint32_t g_last_pps_tim_cnt = 0;
volatile uint32_t g_measured_ticks_per_sec = 100000000UL; /* 예: 100 MHz 타이머 */
volatile uint32_t g_pps_utc_sec = 0;

/* 1. 1PPS 상승 에지 하드웨어 캡처 인터럽트 (지터 &lt; 10 ns) */
void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
    if (htim-&gt;Instance == TIM2 &amp;&amp; htim-&gt;Channel == HAL_TIM_ACTIVE_CHANNEL_1) {
        uint32_t captured_cnt = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
        /* 이전 1PPS와 현재 1PPS 사이의 실제 MCU 클럭 틱 수를 측정하여 크리스탈 드리프트(ppm) 자동 보정 */
        g_measured_ticks_per_sec = captured_cnt - g_last_pps_tim_cnt;
        g_last_pps_tim_cnt = captured_cnt;
        g_pps_utc_sec++;
    }
}

/* 2. IMU 샘플링 시점에 마이크로초(us) 정밀도의 GPST 타임스탬프 획득 */
double GNSS_GetPreciseTimeSec(void) {
    uint32_t now_cnt = __HAL_TIM_GET_COUNTER(&amp;htim2);
    uint32_t elapsed_ticks = now_cnt - g_last_pps_tim_cnt;
    double frac_sec = (double)elapsed_ticks / (double)g_measured_ticks_per_sec;
    return (double)g_pps_utc_sec + frac_sec;
}</pre>
        </div>
        """
