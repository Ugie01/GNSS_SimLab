"""
Module 05: GNSS-INS Integrated Navigation using an 8-State Error-State Extended
Kalman Filter (ESKF / Indirect EKF) with Joseph-Form Covariance Update and GNSS Outage Coasting.
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
    QVBoxLayout,
    QWidget,
)

from core.gnss_ins_ekf import run_gnss_ins_ekf_simulation
from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Module05GnssInsEkf(BaseLabModule):
    """Module 5: Trajectory & IMU Generation, GNSS Outage Simulation, and Error-State EKF Fusion."""

    module_id = "m05_gnss_ins_ekf"
    module_title = "M05. GNSS-INS 통합항법 · Error-State EKF · 재밍 Outage"
    module_subtitle = "관성항법(INS) 표류와 GNSS Outage(재밍/터널) 모사, 8상태 오차 칼만필터(ESKF) 온라인 센서 바이어스 추정"
    module_badge = "MOD 05"

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

        # 1. MEMS IMU Sensor Bias & GNSS Noise Parameters
        g_imu = QGroupBox("1. MEMS IMU 센서 바이어스 · GNSS 측정 잡음")
        g_imu.setStyleSheet(group_style)
        f_imu = QFormLayout(g_imu)
        f_imu.setSpacing(7)

        self.spin_acc_bias = QDoubleSpinBox()
        self.spin_acc_bias.setRange(0.00, 1.50)
        self.spin_acc_bias.setDecimals(2)
        self.spin_acc_bias.setSingleStep(0.05)
        self.spin_acc_bias.setValue(0.25)
        self.spin_acc_bias.setSuffix(" m/s² (~25 mg)")
        self.spin_acc_bias.valueChanged.connect(lambda _: self.run_simulation())
        f_imu.addRow("가속도계 바이어스 (b_a):", self.spin_acc_bias)

        self.spin_gyro_bias = QDoubleSpinBox()
        self.spin_gyro_bias.setRange(0.00, 5.00)
        self.spin_gyro_bias.setDecimals(2)
        self.spin_gyro_bias.setSingleStep(0.20)
        self.spin_gyro_bias.setValue(1.20)
        self.spin_gyro_bias.setSuffix(" deg/s")
        self.spin_gyro_bias.valueChanged.connect(lambda _: self.run_simulation())
        f_imu.addRow("자이로 바이어스 (b_gz):", self.spin_gyro_bias)

        self.spin_gnss_sigma = QDoubleSpinBox()
        self.spin_gnss_sigma.setRange(0.20, 8.00)
        self.spin_gnss_sigma.setDecimals(1)
        self.spin_gnss_sigma.setSingleStep(0.2)
        self.spin_gnss_sigma.setValue(1.8)
        self.spin_gnss_sigma.setSuffix(" m (1σ)")
        self.spin_gnss_sigma.valueChanged.connect(lambda _: self.run_simulation())
        f_imu.addRow("GNSS 위치 잡음 (σ_p):", self.spin_gnss_sigma)
        layout.addWidget(g_imu)

        # 2. GNSS Outage (Jamming / Tunnel Blackout) Window
        g_out = QGroupBox("2. GNSS Outage (재밍 / 터널 신호 두절 구간)")
        g_out.setStyleSheet(group_style)
        f_out = QFormLayout(g_out)
        f_out.setSpacing(7)

        self.slider_out_start = QSlider(Qt.Orientation.Horizontal)
        self.slider_out_start.setRange(2, 45)
        self.slider_out_start.setValue(22)
        self.lbl_out_start = QLabel("22 초")
        self.lbl_out_start.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_out_start.valueChanged.connect(self._on_out_start_changed)

        os_row = QHBoxLayout()
        os_row.addWidget(self.slider_out_start, stretch=1)
        os_row.addWidget(self.lbl_out_start)
        f_out.addRow("재밍 시작 시각 (t_start):", os_row)

        self.slider_out_dur = QSlider(Qt.Orientation.Horizontal)
        self.slider_out_dur.setRange(0, 35)
        self.slider_out_dur.setValue(16)
        self.lbl_out_dur = QLabel("16 초 (두절)")
        self.lbl_out_dur.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_out_dur.valueChanged.connect(self._on_out_dur_changed)

        od_row = QHBoxLayout()
        od_row.addWidget(self.slider_out_dur, stretch=1)
        od_row.addWidget(self.lbl_out_dur)
        f_out.addRow("재밍 지속 시간 (Δt_out):", od_row)
        layout.addWidget(g_out)

        # 3. EKF Covariance Tuning (Q & R Multipliers)
        g_tune = QGroupBox("3. 칼만 필터 공분산 튜닝 (Q / R 가중치 조절)")
        g_tune.setStyleSheet(group_style)
        f_tune = QFormLayout(g_tune)
        f_tune.setSpacing(7)

        self.slider_q_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_q_scale.setRange(1, 50)  # 0.1x to 5.0x
        self.slider_q_scale.setValue(10)     # 1.0x
        self.lbl_q_scale = QLabel("1.0x")
        self.lbl_q_scale.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_q_scale.valueChanged.connect(self._on_q_changed)

        q_row = QHBoxLayout()
        q_row.addWidget(self.slider_q_scale, stretch=1)
        q_row.addWidget(self.lbl_q_scale)
        f_tune.addRow("시스템 잡음 배율 (Q):", q_row)

        self.slider_r_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_r_scale.setRange(1, 50)  # 0.1x to 5.0x
        self.slider_r_scale.setValue(10)     # 1.0x
        self.lbl_r_scale = QLabel("1.0x")
        self.lbl_r_scale.setStyleSheet("font-weight:700; color:#16a34a;")
        self.slider_r_scale.valueChanged.connect(self._on_r_changed)

        r_row = QHBoxLayout()
        r_row.addWidget(self.slider_r_scale, stretch=1)
        r_row.addWidget(self.lbl_r_scale)
        f_tune.addRow("측정 잡음 배율 (R):", r_row)

        layout.addWidget(g_tune)
        return container

    def _on_out_start_changed(self, val: int) -> None:
        self.lbl_out_start.setText(f"{val} 초")
        self.run_simulation()

    def _on_out_dur_changed(self, val: int) -> None:
        self.lbl_out_dur.setText(f"{val} 초 (두절)" if val > 0 else "0 초 (연속수신)")
        self.run_simulation()

    def _on_q_changed(self, val: int) -> None:
        self.lbl_q_scale.setText(f"{val / 10.0:.1f}x")
        self.run_simulation()

    def _on_r_changed(self, val: int) -> None:
        self.lbl_r_scale.setText(f"{val / 10.0:.1f}x")
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

        self.lbl_nav_readout = QLabel()
        self.lbl_nav_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#e2e8f0; border:none;"
        )
        self.lbl_outage_readout = QLabel()
        self.lbl_outage_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#7dd3fc; border:none;"
        )
        self.lbl_bias_readout = QLabel()
        self.lbl_bias_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#86efac; border:none;"
        )

        grid.addWidget(self.lbl_nav_readout, 0, 0)
        grid.addWidget(self.lbl_outage_readout, 0, 1)
        grid.addWidget(self.lbl_bias_readout, 0, 2)
        vbox.addWidget(cards_frame)

        self.fig = Figure(figsize=(10.2, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)
        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        widgets = [
            self.spin_acc_bias,
            self.spin_gyro_bias,
            self.spin_gnss_sigma,
            self.slider_out_start,
            self.slider_out_dur,
            self.slider_q_scale,
            self.slider_r_scale,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.spin_acc_bias.setValue(float(preset_data.get("accel_bias_mps2", 0.25)))
        self.spin_gyro_bias.setValue(float(preset_data.get("gyro_bias_deg_s", 1.20)))
        self.spin_gnss_sigma.setValue(float(preset_data.get("gnss_pos_sigma_m", 1.8)))

        os_val = int(preset_data.get("outage_start_sec", 22))
        self.slider_out_start.setValue(os_val)
        self.lbl_out_start.setText(f"{os_val} 초")

        od_val = int(preset_data.get("outage_duration_sec", 16))
        self.slider_out_dur.setValue(od_val)
        self.lbl_out_dur.setText(f"{od_val} 초 (두절)" if od_val > 0 else "0 초 (연속수신)")

        q_val = int(round(float(preset_data.get("ekf_q_scale", 1.0)) * 10))
        self.slider_q_scale.setValue(q_val)
        self.lbl_q_scale.setText(f"{q_val / 10.0:.1f}x")

        r_val = int(round(float(preset_data.get("ekf_r_scale", 1.0)) * 10))
        self.slider_r_scale.setValue(r_val)
        self.lbl_r_scale.setText(f"{r_val / 10.0:.1f}x")

        for w in widgets:
            w.blockSignals(False)

    def run_simulation(self) -> None:
        acc_b = self.spin_acc_bias.value()
        gyro_b = self.spin_gyro_bias.value()
        gnss_sig = self.spin_gnss_sigma.value()
        out_start = float(self.slider_out_start.value())
        out_dur = float(self.slider_out_dur.value())
        q_scale = self.slider_q_scale.value() / 10.0
        r_scale = self.slider_r_scale.value() / 10.0

        res = run_gnss_ins_ekf_simulation(
            accel_bias_mps2=acc_b,
            gyro_bias_deg_s=gyro_b,
            gnss_pos_sigma_m=gnss_sig,
            outage_start_sec=out_start,
            outage_duration_sec=out_dur,
            ekf_q_scale=q_scale,
            ekf_r_scale=r_scale,
        )

        self.lbl_nav_readout.setText(
            f"<div style='font-size:11.5px; color:#1E3A2F; font-weight:800;'>순수 INS vs EKF 통합항법</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:900; margin-top:3px;'>"
            f"정상 EKF {res.steady_ekf_rms_m:.2f} m · INS {res.max_ins_err_m:.0f} m</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"50Hz INS 적분 + 5Hz Joseph EKF</div>"
        )

        out_end = min(out_start + out_dur, 60.0)
        sup_pct = (1.0 - res.max_ekf_outage_err_m / max(res.max_ins_err_m, 1e-3)) * 100.0
        out_color = "#16A34A" if res.max_ekf_outage_err_m < 10.0 else "#DC2626"
        self.lbl_outage_readout.setText(
            f"<div style='font-size:11.5px; color:#B45309; font-weight:800;'>GNSS Outage ({out_start:.0f}s~{out_end:.0f}s, {out_dur:.0f}s)</div>"
            f"<div style='font-size:13.5px; color:{out_color}; font-weight:900; margin-top:3px;'>"
            f"최대 오차 {res.max_ekf_outage_err_m:.2f} m ({sup_pct:.1f}% 억제)</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"GNSS 복귀 시 칼만 이득 K 증대로 즉시 재수렴</div>"
        )

        final_bax = float(res.est_bias_ax_mps2[-1])
        final_bgz = float(res.est_bias_gz_deg_s[-1])
        self.lbl_bias_readout.setText(
            f"<div style='font-size:11.5px; color:#1D4ED8; font-weight:800;'>온라인 IMU 바이어스 학습 (t=60s)</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:900; margin-top:3px;'>"
            f"자이로 {final_bgz:+.2f}°/s · 가속도 {final_bax:+.2f}m/s²</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"실제 (b_gz {res.true_bias_gz_deg_s:+.2f}, b_ax {res.true_bias_ax_mps2:+.2f})</div>"
        )

        early_warn = (
            f"출발 후 <b>{out_start:.0f}초</b> 만에 재밍이 걸려 칼만필터가 IMU 영점 오차(바이어스)를 충분히 학습하기 전에 GPS가 끊겼습니다. 이로 인해 재밍 구간 오차가 <b>{res.max_ekf_outage_err_m:.1f}m</b>까지 증가했습니다."
            if out_start < 10.0 and out_dur > 0
            else f"출발 후 <b>{out_start:.0f}초</b> 동안 칼만필터가 2D 그래프 3처럼 <b>IMU 영점 오차(자이로 {final_bgz:+.2f}°/s, 가속도 {final_bax:+.2f}m/s²)를 학습해 교정</b>해 두었기 때문에, <b>{out_dur:.0f}초간 GPS가 끊겨도 오차 {res.max_ekf_outage_err_m:.2f}m 이내</b>로 버텨냈습니다."
        )
        self.set_easy_coach_message(
            f"<b>[사수의 실시간 쉬운 해설]</b> (1) 관성센서(IMU) 단독 주행은 60초 만에 <b>{res.max_ins_err_m:.0f}m 표류(빨간 점선)</b>합니다. &nbsp;|&nbsp; "
            f"(2) <b>Error-State EKF</b>는 {early_warn}"
        )

        # Tab 0: Clean 2D Graphs
        self.fig.clear()
        ax1, ax2, ax3, ax4 = self.allocate_2d_axes(self.fig)
        self._draw_2d_plots(ax1, ax2, ax3, ax4, res, out_start, out_end, out_dur)
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        # Tab 1: Dedicated 3D UAV Trajectory
        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            self._draw_3d_uav_flight_trajectory(ax_3d, res, out_start, out_end, out_dur)
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def _draw_2d_plots(self, ax1, ax2, ax3, ax4, res, out_start: float, out_end: float, out_dur: float) -> None:
        # Subplot 1: 2D ENU Trajectory Map
        if ax1 is not None:
            ax1.plot(
                res.true_pos_enu_m[:, 0],
                res.true_pos_enu_m[:, 1],
                color="#0f172a",
                linewidth=2.1,
                label="실제 궤적 (True)",
            )
            ax1.plot(
                res.ins_only_pos_enu_m[:, 0],
                res.ins_only_pos_enu_m[:, 1],
                color="#dc2626",
                linestyle="--",
                linewidth=1.3,
                alpha=0.75,
                label=f"순수 INS ({res.max_ins_err_m:.0f}m 발산)",
            )
            ax1.scatter(
                res.gnss_meas_pos_enu_m[:, 0],
                res.gnss_meas_pos_enu_m[:, 1],
                color="#f59e0b",
                s=12,
                alpha=0.7,
                label="5Hz GNSS 측정치",
            )
            ax1.plot(
                res.ekf_pos_enu_m[:, 0],
                res.ekf_pos_enu_m[:, 1],
                color="#2563eb",
                linewidth=1.9,
                label="Error-State EKF",
            )
            if np.any(res.outage_mask):
                ax1.plot(
                    res.ekf_pos_enu_m[res.outage_mask, 0],
                    res.ekf_pos_enu_m[res.outage_mask, 1],
                    color="#dc2626",
                    linewidth=3.0,
                    label="GNSS Outage 구간",
                )
            ax1.set_xlim(-230, 230)
            ax1.set_ylim(-165, 165)
            ax1.set_title("1. 2D ENU 수평 항법 궤적 비교 (Figure-8)", fontsize=10.5, fontweight="bold", pad=8)
            ax1.set_xlabel("동쪽 위치 East [m]", fontsize=9.0)
            ax1.set_ylabel("북쪽 위치 North [m]", fontsize=9.0)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=7.6)

        # Subplot 2: Position Error & EKF 3-Sigma Covariance Bound over Time
        t = res.t_sec
        if ax2 is not None:
            ax2.plot(
                t,
                res.ekf_pos_err_m,
                color="#2563eb",
                linewidth=1.7,
                label="EKF 위치 오차 [m]",
            )
            ax2.plot(
                t,
                res.ekf_pos_3sigma_m,
                color="#16a34a",
                linestyle="--",
                linewidth=1.4,
                label="EKF 3σ 불확실성 경계",
            )
            if out_dur > 0:
                ax2.axvspan(
                    out_start,
                    out_end,
                    color="#fee2e2",
                    alpha=0.7,
                    label=f"Outage ({out_start:.0f}s~{out_end:.0f}s)",
                )
            ax2.set_title("2. EKF 위치 오차 · 3σ 공분산(P) 포락선 [m]", fontsize=10.5, fontweight="bold", pad=8)
            ax2.set_xlabel("경과 시간 [초]", fontsize=9.0)
            ax2.set_ylabel("오차 및 3σ 경계 [m]", fontsize=9.0)
            ax2.set_ylim(0, max(float(np.max(res.ekf_pos_3sigma_m)) * 1.15, float(np.max(res.ekf_pos_err_m)) * 1.15, 8.0))
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper left", fontsize=7.8)

        # Subplot 3: Online IMU Sensor Bias Estimation
        if ax3 is not None:
            ax3.plot(
                t,
                res.est_bias_gz_deg_s,
                color="#2563eb",
                linewidth=1.9,
                label="추정 자이로 b̂_gz [°/s]",
            )
            ax3.axhline(
                res.true_bias_gz_deg_s,
                color="#1d4ed8",
                linestyle=":",
                linewidth=1.4,
                label=f"실제 b_gz ({res.true_bias_gz_deg_s:+.2f}°/s)",
            )
            ax3.plot(
                t,
                res.est_bias_ax_mps2,
                color="#16a34a",
                linewidth=1.9,
                label="추정 가속도 b̂_ax [m/s²]",
            )
            ax3.axhline(
                res.true_bias_ax_mps2,
                color="#15803d",
                linestyle="--",
                linewidth=1.4,
                label=f"실제 b_ax ({res.true_bias_ax_mps2:+.2f}m/s²)",
            )
            if out_dur > 0:
                ax3.axvspan(out_start, out_end, color="#fee2e2", alpha=0.55)
            ax3.set_title("3. 온라인 IMU 센서 바이어스 추정 수렴 곡선", fontsize=10.5, fontweight="bold", pad=8)
            ax3.set_xlabel("경과 시간 [초]", fontsize=9.0)
            ax3.set_ylabel("바이어스 추정치", fontsize=9.0)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="lower right", fontsize=7.6, ncol=2)

        # Subplot 4: Heading (Yaw) Error Comparison
        if ax4 is not None:
            ax4.plot(
                t,
                res.ins_yaw_err_deg,
                color="#dc2626",
                linestyle="--",
                linewidth=1.5,
                label="순수 자이로 자세 오차",
            )
            ax4.plot(
                t,
                res.ekf_yaw_err_deg,
                color="#2563eb",
                linewidth=1.8,
                label="EKF 자세(Heading) 오차",
            )
            ax4.plot(
                t,
                res.ekf_yaw_3sigma_deg,
                color="#16a34a",
                linestyle=":",
                linewidth=1.3,
                label="EKF 자세 3σ 경계",
            )
            if out_dur > 0:
                ax4.axvspan(out_start, out_end, color="#fee2e2", alpha=0.55)
            ax4.set_title("4. 기수 방위각(Heading ψ) 오차 비교 [deg]", fontsize=10.5, fontweight="bold", pad=8)
            ax4.set_xlabel("경과 시간 [초]", fontsize=9.0)
            ax4.set_ylabel("방위각 오차 [deg]", fontsize=9.0)
            ax4.grid(True, linestyle="--", alpha=0.5)
            ax4.legend(loc="upper left", fontsize=7.8)

    def _draw_3d_uav_flight_trajectory(self, ax_3d, res, out_start: float, out_end: float, out_dur: float) -> None:
        """Render 3D UAV Figure-8 flight trajectory [East, North, Altitude] with Pure INS drift vs EKF & Outage zone."""
        t = res.t_sec
        w0 = 2.0 * np.pi / max(float(t[-1]), 1.0)
        # Synthesize smooth 3D altitude profile (80m +/- 35m banking climb/descent)
        true_alt_m = 80.0 + 35.0 * np.cos(w0 * t)
        ekf_alt_m = true_alt_m + 0.35 * (res.ekf_pos_enu_m[:, 0] - res.true_pos_enu_m[:, 0])
        ins_alt_m = np.clip(true_alt_m + 0.25 * res.ins_pos_err_m, 0.0, 220.0)

        # Ground projection shadow at z = 0
        ax_3d.plot(
            res.true_pos_enu_m[:, 0],
            res.true_pos_enu_m[:, 1],
            np.zeros_like(t),
            color="#cbd5e1",
            linestyle=":",
            linewidth=1.0,
            label="지표면 투영 궤적 (h=0m)",
        )

        # True 3D trajectory
        ax_3d.plot(
            res.true_pos_enu_m[:, 0],
            res.true_pos_enu_m[:, 1],
            true_alt_m,
            color="#0f172a",
            linewidth=2.0,
            label="실제 3D 비행 궤적",
        )

        # Clipped Pure INS 3D drift
        ins_e_clip = np.clip(res.ins_only_pos_enu_m[:, 0], -230.0, 230.0)
        ins_n_clip = np.clip(res.ins_only_pos_enu_m[:, 1], -170.0, 170.0)
        ax_3d.plot(
            ins_e_clip,
            ins_n_clip,
            ins_alt_m,
            color="#dc2626",
            linestyle="--",
            linewidth=1.3,
            alpha=0.75,
            label="순수 INS 3D 표류",
        )

        # EKF 3D fused trajectory
        ax_3d.plot(
            res.ekf_pos_enu_m[:, 0],
            res.ekf_pos_enu_m[:, 1],
            ekf_alt_m,
            color="#2563eb",
            linewidth=2.2,
            label="Error-State EKF 3D 궤적",
        )

        # Highlight 3D GNSS Outage segment in thick red
        if np.any(res.outage_mask):
            ax_3d.plot(
                res.ekf_pos_enu_m[res.outage_mask, 0],
                res.ekf_pos_enu_m[res.outage_mask, 1],
                ekf_alt_m[res.outage_mask],
                color="#dc2626",
                linewidth=3.6,
                label=f"재밍 Outage ({out_dur:.0f}s 관성주행)",
            )

        ax_3d.set_xlim(-220, 220)
        ax_3d.set_ylim(-160, 160)
        ax_3d.set_zlim(0, 150)
        ax_3d.set_xlabel("East [m]", fontsize=8.5)
        ax_3d.set_ylabel("North [m]", fontsize=8.5)
        ax_3d.set_zlabel("고도 Up [m]", fontsize=8.5)
        ax_3d.set_title(
            f"[3D] UAV 입체 기동 궤적 · 재밍 Outage 관성 주행 (Outage 최대 {res.max_ekf_outage_err_m:.2f}m vs 순수 INS {res.max_ins_err_m:.0f}m)",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )
        ax_3d.view_init(elev=27, azim=-50)
        ax_3d.legend(loc="upper right", fontsize=7.3)

    def get_lv1_easy_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">1. 일상생활 비유로 30초 만에 이해하기</h3>
            <ul>
                <li><b>IMU(관성센서) vs GPS = '눈 감고 걸음 수 세기' vs '멀리 있는 등대'</b>:
                    <ul>
                        <li><b>IMU (가속도계·자이로)</b>는 눈을 감고 내 발걸음과 몸의 회전 감각만으로 걷는 것입니다. 1초에 50~400번씩 아주 빠르게 내 움직임을 알 수 있고 적군이 전파 재밍을 걸어도 절대 안 속지만, <b>내 걸음걸이에 아주 작은 버릇(센서 바이어스 = 영점 틀어짐)만 있어도 1분만 걸으면 수백 미터 엉뚱한 곳으로</b> 가버립니다.</li>
                        <li><b>GPS</b>는 1초에 몇 번 느리게 깜빡이고 터널/재밍 구역에선 안 보이지만, 절대 위치가 누적해서 틀어지지 않는 <b>등대</b>입니다.</li>
                    </ul>
                </li>
                <li><b>칼만 필터(Error-State EKF)의 진짜 핵심 역할</b>:
                    단순히 GPS와 IMU 위치를 섞는 게 아닙니다! <b>GPS가 잘 터질 때 등대를 보면서 <i>"아, 내 자이로 센서가 오른쪽으로 1초에 1.2도씩 휘는 버릇(바이어스)이 있구나!"</i>를 미리 계산해서 영점을 고쳐 놓는 것</b>입니다.
                    이렇게 영점을 고쳐 놓으면, 나중에 재밍 공격을 받아 <b>GPS가 20초 동안 완전히 끊겨도 눈 감고 똑바로 걸어갈 수 있습니다!</b></li>
            </ul>

            <h3 style="color:#1E3A2F;">2. 그래프 어디를 봐야 하나요?</h3>
            <ul>
                <li><b>2D 그래프 1 · 3D 입체 탭</b>: 검은색 8자(Figure-8)가 실제 비행 경로입니다. 빨간 점선(순수 INS)은 영점 교정을 안 해서 엉뚱한 곳으로 날아가 버리지만, <b>파란선(EKF)과 굵은 빨간 구간(GPS가 끊긴 Outage 구간)</b>은 검은색 실제 경로를 끝까지 따라갑니다.</li>
                <li><b>2D 그래프 2 (3σ 불확실성 풍선)</b>: 분홍색 배경 구간(GPS 두절 시간) 동안에는 등대가 안 보여 초록색 점선(<b>공분산 P 3σ 경계</b>)이 부풀어 오르다가, GPS가 복귀하는 순간 확 줄어듭니다.</li>
                <li><b>2D 그래프 3 (센서 바이어스 학습)</b>: 출발 시 <code>0</code>에서 시작한 파란선(추정 자이로)과 초록선(추정 가속도)이 <b>10초 만에 실제 센서 오차(점선)에 수렴</b>합니다.</li>
            </ul>
        </div>
        """

    def get_lv1_easy_c_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">3. C 코드 3단계 한글 흐름도 (고속 INS 루프 + 저속 EKF 보정)</h3>
            <pre style="background-color:#FAFBF9; border:1px solid #D8E2DB; color:#14532d; padding:12px; border-radius:8px; font-family:'Malgun Gothic',monospace; font-size:13px;">
[빠른 루프: 매 20ms(50Hz)마다 IMU 센서가 들어올 때]
1. 지금까지 학습해 둔 센서 버릇(바이어스)을 먼저 빼서 깨끗한 각속도·가속도 만들기:
   깨끗한_가속도 = 측정_가속도 - 학습된_가속도_바이어스;
   깨끗한_각속도 = 측정_각속도 - 학습된_자이로_바이어스;
2. 적분해서 내 예상 위치·속도·자세 갱신 + 불확실성(공분산 P) 조금 키우기!

[느린 루프: 매 200ms(5Hz)마다 GPS 위치가 도착했을 때 (재밍 아닐 때만!)]
3. (GPS 위치 - 내 예상 위치) 차이로부터 칼만 이득(K)을 곱해 [위치오차, 속도오차, 자세오차, 센서바이어스오차] 계산!
4. 계산된 오차를 내 위치와 센서 바이어스에 더해줘서 고치고(Injection), 불확실성(P)을 조셉 공식으로 확 줄이기!</pre>
        </div>
        """

    def get_theory_html(self) -> str:
        return """
        <div style="line-height: 1.68;">
            <h2 style="color:#1E3A2F; margin-top:0;">1. GNSS/INS 상보적 센서 특성과 관성 적분 표류</h2>
            <p>
                <b><a href="glossary://imu" style="color:#1E3A2F; font-weight:700;">IMU (가속도계·자이로)</a></b> 기반
                <a href="glossary://dead_reckoning" style="color:#1E3A2F; font-weight:700;">추측항법(Dead Reckoning)</a>은 외부 RF 재밍에 완전히 면역이며
                고율(50~400 Hz) 동적 응답을 제공하지만, 가속도 바이어스 <code>b_a</code>와 자이로 바이어스 <code>b_g</code>의 이중/삼중 적분으로
                오차가 시간의 제곱(<code>t²</code>) 및 세제곱(<code>t³</code>)으로 발산합니다.
                따라서 GNSS 가용 구간에서 관성 바이어스와 오정렬(Misalignment)을 온라인으로 추정·보정해 두는 것이 Outage 생존 성능(Coasting)을 결정합니다.
            </p>

            <h2 style="color:#1E3A2F;">2. 직접 상태(Total-State) vs 오차 상태(<a href="glossary://error_state_ekf" style="color:#1E3A2F; font-weight:700;">Error-State EKF</a>)</h2>
            <ul>
                <li>
                    <b>선형화 타당성</b>: 명목 상태(Nominal State: 위치·속도·자세)는 3차원 비선형 회전 운동을 겪지만,
                    오차 상태 <code>δx = [δp, δv, δψ, δb_a, δb_g]ᵀ</code>는 항상 0 근방의 미소량으로 유지되므로 1차 선형 섭동 동역학이 높은 정확도로 성립합니다.
                </li>
                <li>
                    <b>오차 주입(Injection) · 상태 리셋</b>: 50~400 Hz IMU 스트랩다운 적분 루프와 비동기 5~10 Hz
                    <a href="glossary://loosely_coupled" style="color:#1E3A2F; font-weight:700;">약결합(Loosely Coupled)</a>
                    <a href="glossary://kalman_gain" style="color:#1E3A2F; font-weight:700;">칼만 이득 K</a> 갱신을 분리하고,
                    갱신 후 오차 추정치 <code>δx</code>를 명목 상태에 더한 뒤 <code>δx ← 0</code>으로 리셋합니다.
                </li>
            </ul>
        </div>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. 8상태 Error-State EKF 오차 상태 벡터 및 연속시간 선형화 동역학",
                latex_expr=(
                    r"\delta \mathbf{x} = \left[ \delta p_E, \ \delta p_N, \ \delta v_E, \ \delta v_N, \ \delta \psi, \ \delta b_{ax}, \ \delta b_{ay}, \ \delta b_{gz} \right]^T"
                    "\n"
                    r"\delta \dot{\mathbf{p}} = \delta \mathbf{v}, \qquad \delta \dot{\mathbf{v}} = \left( \frac{\partial \mathbf{R}(\hat{\psi})}{\partial \psi} \hat{\mathbf{a}}^b \right) \delta\psi - \mathbf{R}(\hat{\psi})\,\delta\mathbf{b}_a, \qquad \delta \dot{\psi} = -\delta b_{gz}"
                ),
                intuitive_summary="속도 오차 변화율 δv_dot은 기수각 오차 δψ와 가속도계 측정치 a^b의 곱, 그리고 동체 좌표계 가속도 바이어스 δb_a에 의해 구동됩니다. 즉 기동(가속/회전)이 있어야 방위각과 센서 바이어스가 관찰(Observable)됩니다!",
                symbols=[
                    SymbolSpec(r"\delta \mathbf{p}, \delta \mathbf{v}, \delta \psi", "위치, 속도, 방위각 오차", "m, m/s, rad", "실제 항법 상태와 명목 INS 적분 상태 사이의 오차", "error_state_ekf"),
                    SymbolSpec(r"\delta b_{ax}, \delta b_{ay}, \delta b_{gz}", "가속도계 및 자이로 바이어스 오차", "m/s², rad/s", "아직 보상되지 않고 남아 있는 IMU 센서 영점 바이어스 잔차", "imu"),
                    SymbolSpec(r"\mathbf{R}(\hat{\psi})", "동체(Body)→항법(ENU) 회전행렬", "-", "기수 방위각 ψ에 의한 2×2 수평 좌표 회전 변환 행렬", "dead_reckoning"),
                ],
            ),
            FormulaSpec(
                title="수식 2. 공분산 시간 전파(Prediction) 및 GNSS 측정 이노베이션(Innovation)",
                latex_expr=(
                    r"\mathbf{F}_k = \mathbf{I}_8 + \mathbf{A}_k \Delta t, \qquad \mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k, \qquad \mathbf{z}_k = \mathbf{y}_{\mathrm{GNSS},k} - \hat{\mathbf{y}}_{\mathrm{INS},k}"
                ),
                intuitive_summary="IMU 샘플마다 상태천이행렬 F_k로 오차 공분산 P를 전파하며(GNSS Outage 중에는 이 단계만 반복되어 P가 팽창), GNSS가 수신되면 위치·속도 차이 z_k를 계산합니다.",
                symbols=[
                    SymbolSpec(r"\mathbf{F}_k", "이산 상태천이행렬", "R^{8×8}", "오차 동역학 야코비안 A_k를 시간 간격 Δt로 1차 이산화한 행렬", "error_state_ekf"),
                    SymbolSpec(r"\mathbf{P}_{k|k-1}, \mathbf{Q}_k", "예측 오차 공분산 및 시스템 잡음", "-", "추정 상태의 불확실성 행렬 및 IMU 열잡음/랜덤워크 공분산", "kalman_gain"),
                    SymbolSpec(r"\mathbf{z}_k", "측정 잔차 (Innovation)", "m, m/s", "GNSS 측정 위치·속도와 INS 예측 위치·속도의 차이 벡터", "loosely_coupled"),
                ],
            ),
            FormulaSpec(
                title="수식 3. 칼만 이득 K, 조셉 형태(Joseph Form) 공분산 갱신 및 오차 주입(Injection)",
                latex_expr=(
                    r"\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T \left( \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R}_k \right)^{-1}, \qquad \hat{\mathbf{x}}_{\mathrm{nom}} \leftarrow \hat{\mathbf{x}}_{\mathrm{nom}} + \mathbf{K}_k \mathbf{z}_k"
                    "\n"
                    r"\mathbf{P}_{k|k} = \left( \mathbf{I} - \mathbf{K}_k \mathbf{H} \right) \mathbf{P}_{k|k-1} \left( \mathbf{I} - \mathbf{K}_k \mathbf{H} \right)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T"
                ),
                intuitive_summary="단순 식 P=(I-KH)P 대신 두 대칭 이차형식의 합인 조셉 형태(Joseph Form)로 공분산 P를 갱신하여 부동소수점 반올림 오차에 의한 대칭성 파괴와 필터 발산을 원천 차단합니다.",
                symbols=[
                    SymbolSpec(r"\mathbf{K}_k", "최적 칼만 이득 행렬", "R^{8×4}", "측정 잔차 z_k로부터 숨겨진 자세·바이어스 오차까지 추정하는 이득", "kalman_gain"),
                    SymbolSpec(r"\mathbf{R}_k", "GNSS 측정 잡음 공분산", "R^{4×4}", "GNSS 수신기가 보고한 위치·속도 측정 불확실성(DOP 반영)", "kalman_gain"),
                    SymbolSpec(r"\mathbf{P}_{k|k}", "조셉 형태 사후 공분산", "R^{8×8}", "임베디드 부동소수점에서도 항상 양정치 대칭성을 보장하는 갱신식", "joseph_form"),
                ],
            ),
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <b style="color:#1E3A2F; font-size:14px;">슬라이더 조작 실습 미션 (Observation Guide)</b>
            <ol style="margin-top:8px; margin-bottom:4px; padding-left:20px;">
                <li style="margin-bottom:6px;">
                    <b>미션 1 (바이어스 학습 완료 후 재밍 vs 출발 직후 조기 재밍):</b>
                    좌측 상단 프리셋에서 <b>[표준 전술 기동 (22s 재밍)]</b>과 <b>[출발 직후 조기 재밍 (3s 재밍)]</b>을 번갈아 선택해 보세요.
                    출발 3초 만에 재밍이 걸리면 왜 [그래프 3 · 4]의 바이어스 추정이 끝나기 전이라 Outage 오차가 커지는지 확인하세요.
                </li>
                <li style="margin-bottom:6px;">
                    <b>미션 2 (공분산 P 팽창과 복귀 시 수축):</b> 2D 그래프 2에서 붉은색 음영(GNSS Outage) 구간 동안 초록색 점선(<code>3σ 불확실성 경계</code>)이
                    어떻게 부풀어 오르다가 GNSS 신호 복구 순간 단번에 수축하는지 관찰하세요.
                </li>
                <li>
                    <b>미션 3 (Q / R 공분산 튜닝 실습):</b> <b>[측정 잡음 배율 (R)]</b>을 <code style="color:#16a34a;">5.0x</code>로 높이면
                    필터가 GNSS를 덜 신뢰해 2D 그래프 3의 바이어스 수렴 속도가 어떻게 변하는지 확인하세요.
                </li>
            </ol>
        </div>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <h2 style="color:#1E3A2F; margin-top:0;">실전 임베디드 C 포팅 가이드: 조셉 형태(Joseph Form) 공분산 갱신 · 오차 주입 패턴</h2>

            <div style="background-color:#FEF2F2; border-left:4px solid #DC2626; padding:12px 14px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#991B1B;">[임베디드 칼만 필터 3대 폭주(NaN 발산) 원인과 해결책]</b><br>
                1. <b>비대칭 누적에 의한 음수 분산(NaN):</b> <code>P = (I - K*H)*P</code>를 <code>float32</code>로 수천 번 반복하면 반올림 오차로 대칭성이 깨지고 대각 원소가 음수가 됩니다. 반드시 <b>조셉 형태(Joseph Form)</b>로 갱신하고 마지막에 <code>P[i][j] = P[j][i] = 0.5f * (P[i][j] + P[j][i])</code> 대칭화 루프를 실행하세요!<br>
                2. <b>각도 래핑(Angle Wrapping) 누락:</b> 방위각 잔차나 갱신 후 <code>psi</code>를 <code>[-π, +π]</code>로 래핑하지 않으면 <code>+179°</code>에서 <code>-179°</code>로 넘어갈 때 <code>358°</code> 점프 오차가 주입되어 필터가 즉사합니다.<br>
                3. <b>특이치 기각(Chi-Square Gating):</b> 도심 다중경로로 GNSS 위치가 갑자기 50 m 뛸 때 그대로 받아들이면 추정해 둔 IMU 바이어스가 오염됩니다. <code>z^T S^{-1} z &gt; γ</code>이면 측정 갱신을 스킵하세요.
            </div>

            <h3 style="color:#0f172a;">1. 조셉 형태(Joseph Form) 공분산 갱신 및 대칭성 강제 C99 구현</h3>
            <pre style="background-color:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#include &lt;math.h&gt;
#include &lt;stdint.h&gt;

#define N_STATE 8
#define N_MEAS  4

typedef struct {
    double p_e, p_n;       /* 명목 위치 (절대/국소 좌표는 double 권장) */
    float  v_e, v_n;       /* 명목 속도 [m/s] */
    float  psi_rad;        /* 명목 기수 방위각 [-pi, +pi] */
    float  b_ax, b_ay;     /* 추정된 가속도계 바이어스 [m/s^2] */
    float  b_gz;           /* 추정된 자이로 바이어스 [rad/s] */
    float  P[N_STATE][N_STATE]; /* 오차 상태 공분산 행렬 */
} EskfState_t;

static inline float WrapPi(float rad) {
    return atan2f(sinf(rad), cosf(rad));
}

/* 오차 상태 dx를 명목 상태(Nominal State)에 주입(Injection)하고 대칭성 강제 */
void ESKF_InjectErrorAndSymmetrize(EskfState_t *s, const float dx[N_STATE]) {
    s-&gt;p_e     += (double)dx[0];
    s-&gt;p_n     += (double)dx[1];
    s-&gt;v_e     += dx[2];
    s-&gt;v_n     += dx[3];
    s-&gt;psi_rad  = WrapPi(s-&gt;psi_rad + dx[4]);
    s-&gt;b_ax    += dx[5];
    s-&gt;b_ay    += dx[6];
    s-&gt;b_gz    += dx[7];

    /* 공분산 행렬 P의 엄격한 대칭성(Symmetry) 및 대각 양수성 보장 */
    for (int i = 0; i &lt; N_STATE; ++i) {
        if (s-&gt;P[i][i] &lt; 1e-9f) s-&gt;P[i][i] = 1e-9f;
        for (int j = i + 1; j &lt; N_STATE; ++j) {
            float sym = 0.5f * (s-&gt;P[i][j] + s-&gt;P[j][i]);
            s-&gt;P[i][j] = sym;
            s-&gt;P[j][i] = sym;
        }
    }
}</pre>
        </div>
        """
