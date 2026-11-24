"""
Module 04: Anti-Jamming CRPA (Controlled Reception Pattern Antenna) Array Signal Processing,
Power Minimization (PI) Nulling, and MVDR (Capon) Distortionless Beamforming.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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

from core.antijam_crpa import run_crpa_simulation
from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Module04AntiJamCrpa(BaseLabModule):
    """Module 4: N-Element CRPA Array Signal Processing, Power-Min, and MVDR Adaptive Nulling."""

    module_id = "m04_antijam_crpa"
    module_title = "M04. 항재밍 CRPA 배열 안테나 · Power-Min / MVDR 널링"
    module_subtitle = "N소자 배열 매니폴드 조향 벡터와 공간 공분산 행렬(R_xx), 전력 최소화(PI) 및 MVDR(Capon) 적응형 빔포밍"
    module_badge = "MOD 04"

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

        # 1. CRPA Array Geometry & Satellite Angle
        g_arr = QGroupBox("1. CRPA 배열 안테나 형상 · 위성 입사각")
        g_arr.setStyleSheet(group_style)
        f_arr = QFormLayout(g_arr)
        f_arr.setSpacing(7)

        self.spin_num_elem = QSpinBox()
        self.spin_num_elem.setRange(2, 12)
        self.spin_num_elem.setValue(4)
        self.spin_num_elem.setSuffix(" 소자 (자유도 N-1)")
        self.spin_num_elem.valueChanged.connect(lambda _: self.run_simulation())
        f_arr.addRow("안테나 소자 수 (N):", self.spin_num_elem)

        self.spin_d_lambda = QDoubleSpinBox()
        self.spin_d_lambda.setRange(0.25, 1.00)
        self.spin_d_lambda.setDecimals(2)
        self.spin_d_lambda.setSingleStep(0.05)
        self.spin_d_lambda.setValue(0.50)
        self.spin_d_lambda.setSuffix(" λ (0.5λ ≈ 9.5cm)")
        self.spin_d_lambda.valueChanged.connect(lambda _: self.run_simulation())
        f_arr.addRow("소자 간격 (d/λ):", self.spin_d_lambda)

        self.slider_sat_ang = QSlider(Qt.Orientation.Horizontal)
        self.slider_sat_ang.setRange(-75, 75)
        self.slider_sat_ang.setValue(15)
        self.lbl_sat_ang = QLabel("+15°")
        self.lbl_sat_ang.setStyleSheet("font-weight:700; color:#16a34a;")
        self.slider_sat_ang.valueChanged.connect(self._on_sat_ang_changed)

        s_row = QHBoxLayout()
        s_row.addWidget(self.slider_sat_ang, stretch=1)
        s_row.addWidget(self.lbl_sat_ang)
        f_arr.addRow("희망 위성 입사각 (θ_s):", s_row)
        layout.addWidget(g_arr)

        # 2. Jammer 1 & Jammer 2 Threat Configuration
        g_jam = QGroupBox("2. 재머 위협 시나리오 (Jammer 1 · Jammer 2)")
        g_jam.setStyleSheet(group_style)
        f_jam = QFormLayout(g_jam)
        f_jam.setSpacing(7)

        self.slider_j1_ang = QSlider(Qt.Orientation.Horizontal)
        self.slider_j1_ang.setRange(-80, 80)
        self.slider_j1_ang.setValue(-40)
        self.lbl_j1_ang = QLabel("-40°")
        self.lbl_j1_ang.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_j1_ang.valueChanged.connect(self._on_j1_ang_changed)

        j1a_row = QHBoxLayout()
        j1a_row.addWidget(self.slider_j1_ang, stretch=1)
        j1a_row.addWidget(self.lbl_j1_ang)
        f_jam.addRow("재머 1 입사각 (θ_j1):", j1a_row)

        self.slider_j1_jsr = QSlider(Qt.Orientation.Horizontal)
        self.slider_j1_jsr.setRange(25, 85)
        self.slider_j1_jsr.setValue(60)
        self.lbl_j1_jsr = QLabel("60 dB")
        self.lbl_j1_jsr.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_j1_jsr.valueChanged.connect(self._on_j1_jsr_changed)

        j1p_row = QHBoxLayout()
        j1p_row.addWidget(self.slider_j1_jsr, stretch=1)
        j1p_row.addWidget(self.lbl_j1_jsr)
        f_jam.addRow("재머 1 세기 (J1/S):", j1p_row)

        self.chk_jam2 = QCheckBox("재머 2 (다중 방향 동시 재밍) 활성화")
        self.chk_jam2.setChecked(True)
        self.chk_jam2.setStyleSheet("font-weight:700; color:#b91c1c;")
        self.chk_jam2.toggled.connect(lambda _: self.run_simulation())
        f_jam.addRow(self.chk_jam2)

        self.slider_j2_ang = QSlider(Qt.Orientation.Horizontal)
        self.slider_j2_ang.setRange(-80, 80)
        self.slider_j2_ang.setValue(50)
        self.lbl_j2_ang = QLabel("+50°")
        self.lbl_j2_ang.setStyleSheet("font-weight:700; color:#d97706;")
        self.slider_j2_ang.valueChanged.connect(self._on_j2_ang_changed)

        j2a_row = QHBoxLayout()
        j2a_row.addWidget(self.slider_j2_ang, stretch=1)
        j2a_row.addWidget(self.lbl_j2_ang)
        f_jam.addRow("재머 2 입사각 (θ_j2):", j2a_row)

        self.slider_j2_jsr = QSlider(Qt.Orientation.Horizontal)
        self.slider_j2_jsr.setRange(25, 85)
        self.slider_j2_jsr.setValue(55)
        self.lbl_j2_jsr = QLabel("55 dB")
        self.lbl_j2_jsr.setStyleSheet("font-weight:700; color:#d97706;")
        self.slider_j2_jsr.valueChanged.connect(self._on_j2_jsr_changed)

        j2p_row = QHBoxLayout()
        j2p_row.addWidget(self.slider_j2_jsr, stretch=1)
        j2p_row.addWidget(self.lbl_j2_jsr)
        f_jam.addRow("재머 2 세기 (J2/S):", j2p_row)
        layout.addWidget(g_jam)

        # 3. Covariance Regularization (Diagonal Loading)
        g_reg = QGroupBox("3. 공분산 행렬 수치 안정화 (Diagonal Loading)")
        g_reg.setStyleSheet(group_style)
        f_reg = QFormLayout(g_reg)
        f_reg.setSpacing(7)

        self.slider_dl = QSlider(Qt.Orientation.Horizontal)
        self.slider_dl.setRange(-30, 10)  # -30 dB to +10 dB relative to thermal noise
        self.slider_dl.setValue(-10)
        self.lbl_dl = QLabel("-10 dB")
        self.lbl_dl.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_dl.valueChanged.connect(self._on_dl_changed)

        dl_row = QHBoxLayout()
        dl_row.addWidget(self.slider_dl, stretch=1)
        dl_row.addWidget(self.lbl_dl)
        f_reg.addRow("대각 로딩 (δ/σ_n²):", dl_row)

        layout.addWidget(g_reg)
        return container

    def _on_sat_ang_changed(self, val: int) -> None:
        self.lbl_sat_ang.setText(f"{val:+d}°")
        self.run_simulation()

    def _on_j1_ang_changed(self, val: int) -> None:
        self.lbl_j1_ang.setText(f"{val:+d}°")
        self.run_simulation()

    def _on_j1_jsr_changed(self, val: int) -> None:
        self.lbl_j1_jsr.setText(f"{val} dB")
        self.run_simulation()

    def _on_j2_ang_changed(self, val: int) -> None:
        self.lbl_j2_ang.setText(f"{val:+d}°")
        self.run_simulation()

    def _on_j2_jsr_changed(self, val: int) -> None:
        self.lbl_j2_jsr.setText(f"{val} dB")
        self.run_simulation()

    def _on_dl_changed(self, val: int) -> None:
        self.lbl_dl.setText(f"{val:+d} dB")
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

        self.lbl_dof_readout = QLabel()
        self.lbl_dof_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#e2e8f0; border:none;"
        )
        self.lbl_null_readout = QLabel()
        self.lbl_null_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#7dd3fc; border:none;"
        )
        self.lbl_sinr_readout = QLabel()
        self.lbl_sinr_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#86efac; border:none;"
        )

        grid.addWidget(self.lbl_dof_readout, 0, 0)
        grid.addWidget(self.lbl_null_readout, 0, 1)
        grid.addWidget(self.lbl_sinr_readout, 0, 2)
        vbox.addWidget(cards_frame)

        self.fig = Figure(figsize=(10.2, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)
        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        widgets = [
            self.spin_num_elem,
            self.spin_d_lambda,
            self.slider_sat_ang,
            self.slider_j1_ang,
            self.slider_j1_jsr,
            self.chk_jam2,
            self.slider_j2_ang,
            self.slider_j2_jsr,
            self.slider_dl,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.spin_num_elem.setValue(int(preset_data.get("num_elements", 4)))
        self.spin_d_lambda.setValue(float(preset_data.get("d_lambda", 0.5)))

        s_ang = int(preset_data.get("sat_angle_deg", 15))
        self.slider_sat_ang.setValue(s_ang)
        self.lbl_sat_ang.setText(f"{s_ang:+d}°")

        j1_ang = int(preset_data.get("jam1_angle_deg", -40))
        self.slider_j1_ang.setValue(j1_ang)
        self.lbl_j1_ang.setText(f"{j1_ang:+d}°")

        j1_jsr = int(preset_data.get("jam1_jsr_db", 60))
        self.slider_j1_jsr.setValue(j1_jsr)
        self.lbl_j1_jsr.setText(f"{j1_jsr} dB")

        self.chk_jam2.setChecked(bool(preset_data.get("jam2_enabled", True)))

        j2_ang = int(preset_data.get("jam2_angle_deg", 50))
        self.slider_j2_ang.setValue(j2_ang)
        self.lbl_j2_ang.setText(f"{j2_ang:+d}°")

        j2_jsr = int(preset_data.get("jam2_jsr_db", 55))
        self.slider_j2_jsr.setValue(j2_jsr)
        self.lbl_j2_jsr.setText(f"{j2_jsr} dB")

        dl_val = int(preset_data.get("diagonal_loading_db", -10))
        self.slider_dl.setValue(dl_val)
        self.lbl_dl.setText(f"{dl_val:+d} dB")

        for w in widgets:
            w.blockSignals(False)

    def run_simulation(self) -> None:
        n_elem = self.spin_num_elem.value()
        d_lam = self.spin_d_lambda.value()
        sat_ang = float(self.slider_sat_ang.value())
        j1_ang = float(self.slider_j1_ang.value())
        j1_jsr = float(self.slider_j1_jsr.value())
        j2_en = self.chk_jam2.isChecked()
        j2_ang = float(self.slider_j2_ang.value())
        j2_jsr = float(self.slider_j2_jsr.value())
        dl_db = float(self.slider_dl.value())

        res = run_crpa_simulation(
            num_elements=n_elem,
            d_lambda=d_lam,
            sat_angle_deg=sat_ang,
            jam1_angle_deg=j1_ang,
            jam1_jsr_db=j1_jsr,
            jam2_enabled=j2_en,
            jam2_angle_deg=j2_ang,
            jam2_jsr_db=j2_jsr,
            diagonal_loading_db=dl_db,
        )

        n_jams = len(res.jammer_angles_deg)
        dof = n_elem - 1
        dof_color = "#16A34A" if dof >= n_jams else "#DC2626"
        dof_status = f"자유도 충분 ({dof} ≥ {n_jams})" if dof >= n_jams else f"자유도 초과 ({dof} < {n_jams})"

        self.lbl_dof_readout.setText(
            f"<div style='font-size:11.5px; color:#1E3A2F; font-weight:800;'>CRPA 배열 · 공간 자유도</div>"
            f"<div style='font-size:13.5px; color:{dof_color}; font-weight:900; margin-top:3px;'>{dof_status}</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"{n_elem}소자 ULA ({d_lam:.2f}λ ≈ {d_lam*19.03:.1f}cm) · 재머 {n_jams}기 · 위성 {sat_ang:+.0f}°</div>"
        )

        j2_null_str = (
            f"J2({j2_ang:+.0f}°) PM {res.null_depths_pm_db[1]:.0f}dB / MVDR {res.null_depths_mvdr_db[1]:.0f}dB"
            if j2_en and len(res.null_depths_mvdr_db) > 1
            else "재머 2 비활성화"
        )
        self.lbl_null_readout.setText(
            f"<div style='font-size:11.5px; color:#B45309; font-weight:800;'>방향별 이득 · 널 깊이 (Null)</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:900; margin-top:3px;'>"
            f"J1({j1_ang:+.0f}°) MVDR {res.null_depths_mvdr_db[0]:.1f} dB</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"위성 {sat_ang:+.0f}°: {res.sat_gain_mvdr_db:+.1f}dB | {j2_null_str}</div>"
        )

        trk_color = "#16A34A" if res.eff_cn0_mvdr_dbhz >= 28.0 else "#DC2626"
        trk_crpa = "항법 추적 유지 (≥28)" if res.eff_cn0_mvdr_dbhz >= 28.0 else "감도 저하 (<28)"
        self.lbl_sinr_readout.setText(
            f"<div style='font-size:11.5px; color:#1D4ED8; font-weight:800;'>항재밍 유효 C/N0 복원</div>"
            f"<div style='font-size:13.5px; color:{trk_color}; font-weight:900; margin-top:3px;'>"
            f"MVDR {res.eff_cn0_mvdr_dbhz:.1f} dB-Hz ({trk_crpa})</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"단일 {res.eff_cn0_single_dbhz:.1f} · Power-Min {res.eff_cn0_pm_dbhz:.1f} dB-Hz</div>"
        )

        if dof < n_jams:
            self.set_easy_coach_message(
                f"<b>[사수의 실시간 쉬운 해설]</b> 현재 <b>{n_elem}소자</b>라서 막을 수 있는 재머 개수(<b>자유도 N-1 = {dof}개</b>)보다 "
                f"공격 중인 재머(<b>{n_jams}개</b>)가 많습니다. 유효 C/N0가 <b>{res.eff_cn0_mvdr_dbhz:.1f} dB-Hz</b>로 저하되었으니 좌측 <b>[안테나 소자 수(N)]</b>를 4소자 이상으로 늘려보세요."
            )
        else:
            self.set_easy_coach_message(
                f"<b>[사수의 실시간 쉬운 해설]</b> (1) 재머가 <b>{j1_jsr:.0f} dB</b> 세기로 공격해 단일 안테나는 무력화(<b>{res.eff_cn0_single_dbhz:.1f} dB-Hz</b>)되지만, "
                f"<b>{n_elem}소자 CRPA</b>가 재머 방향에 <b>{res.null_depths_mvdr_db[0]:.0f} dB 공간 널(Null)</b>을 형성했습니다. &nbsp;|&nbsp; "
                f"(2) 위성 방향(<b>{sat_ang:+.0f}°</b>) 신호는 <b>{res.eff_cn0_mvdr_dbhz:.1f} dB-Hz</b>로 정상 수신됩니다."
            )

        # Tab 0: Clean 2D Graphs
        self.fig.clear()
        ax1, ax2, ax3, ax4 = self.allocate_2d_axes(self.fig, projections=(None, "polar", None, None))
        self._draw_2d_plots(
            ax1, ax2, ax3, ax4,
            res, n_elem, sat_ang, j1_ang, j1_jsr, j2_en, j2_ang, j2_jsr, n_jams,
        )
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        # Tab 1: Dedicated 3D Beam/Null Surface
        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            self._draw_3d_crpa_beam_pattern(ax_3d, res, n_elem, sat_ang, j1_ang, j2_en, j2_ang)
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def _draw_2d_plots(
        self,
        ax1,
        ax2,
        ax3,
        ax4,
        res,
        n_elem: int,
        sat_ang: float,
        j1_ang: float,
        j1_jsr: float,
        j2_en: bool,
        j2_ang: float,
        j2_jsr: float,
        n_jams: int,
    ) -> None:
        # Subplot 1: Cartesian Array Gain Pattern [-90°, +90°]
        if ax1 is not None:
            ax1.plot(
                res.angles_deg,
                res.pattern_das_db,
                color="#94a3b8",
                linestyle=":",
                linewidth=1.2,
                label="일반 빔조향 (DAS)",
            )
            ax1.plot(
                res.angles_deg,
                res.pattern_pm_db,
                color="#0284c7",
                linestyle="--",
                linewidth=1.7,
                label="Power-Min (PI)",
            )
            ax1.plot(
                res.angles_deg,
                res.pattern_mvdr_db,
                color="#2563eb",
                linewidth=2.1,
                label="MVDR / Capon",
            )
            ax1.axvline(sat_ang, color="#16a34a", linewidth=1.9, linestyle="-", label=f"위성 ({sat_ang:+.0f}°)")
            ax1.axvline(j1_ang, color="#dc2626", linewidth=1.7, linestyle="--", label=f"재머1 ({j1_ang:+.0f}°)")
            if j2_en:
                ax1.axvline(j2_ang, color="#d97706", linewidth=1.7, linestyle="--", label=f"재머2 ({j2_ang:+.0f}°)")

            ax1.set_title(f"1. CRPA 직교 방사패턴 · 널 형성 ({n_elem}소자)", fontsize=10.5, fontweight="bold", pad=8)
            ax1.set_xlabel("입사각 θ [deg]", fontsize=9.0)
            ax1.set_ylabel("배열 이득 G(θ) [dB]", fontsize=9.0)
            ax1.set_xlim(-90, 90)
            ax1.set_ylim(-65, 5)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="lower left", fontsize=7.6, ncol=2)

        # Subplot 2: Upper-Hemisphere Polar Radiation Pattern
        if ax2 is not None:
            theta_rad = np.deg2rad(res.angles_deg)
            r_floor = -55.0
            r_pm_polar = np.maximum(res.pattern_pm_db, r_floor)
            r_mvdr_polar = np.maximum(res.pattern_mvdr_db, r_floor)

            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_thetamin(-90)
            ax2.set_thetamax(90)
            ax2.plot(theta_rad, r_pm_polar, color="#0284c7", linestyle="--", linewidth=1.4, label="Power-Min")
            ax2.plot(theta_rad, r_mvdr_polar, color="#2563eb", linewidth=2.0, label="MVDR")
            ax2.axvline(np.deg2rad(sat_ang), color="#16a34a", linewidth=1.9, label="위성")
            ax2.axvline(np.deg2rad(j1_ang), color="#dc2626", linestyle="--", linewidth=1.5, label="재머1")
            if j2_en:
                ax2.axvline(np.deg2rad(j2_ang), color="#d97706", linestyle="--", linewidth=1.5, label="재머2")
            ax2.set_rlim(r_floor, 5.0)
            ax2.set_rticks([-45, -30, -15, 0])
            ax2.set_yticklabels(["-45dB", "-30dB", "-15dB", "0dB"], fontsize=7.2)
            ax2.set_title("2. 상반구 극좌표 빔/널 패턴 (-90°~+90°)", fontsize=10.5, fontweight="bold", pad=12)
            ax2.legend(loc="lower right", fontsize=7.6)

        # Subplot 3: Spatial Covariance Matrix R_xx Eigenvalue Subspace Spectrum
        if ax3 is not None:
            eig_indices = [f"λ_{i+1}" for i in range(n_elem)]
            eig_colors = ["#dc2626" if val > 10.0 else "#2563eb" for val in res.eigenvalues_db]
            bars = ax3.bar(eig_indices, res.eigenvalues_db, color=eig_colors, width=0.5, alpha=0.88)
            ax3.axhline(0.0, color="#16a34a", linestyle="--", linewidth=1.2, label="열잡음 바닥 (0 dB)")
            for b, val in zip(bars, res.eigenvalues_db):
                ax3.text(
                    b.get_x() + b.get_width() / 2.0,
                    max(val, 0.0) + 1.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    fontweight="bold",
                )
            ax3.set_ylim(-5.0, max(float(np.max(res.eigenvalues_db)) * 1.18, 25.0))
            ax3.set_title(
                f"3. 공분산 R_xx 고유값 스펙트럼 (재머 부공간 {n_jams}개)",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
            )
            ax3.set_xlabel("고유값 인덱스", fontsize=9.0)
            ax3.set_ylabel("고유값 전력 [dB]", fontsize=9.0)
            ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right", fontsize=8.0)

        # Subplot 4: Post-Beamforming Effective C/N0 vs Jammer J/S Ratio Sweep
        if ax4 is not None:
            ax4.plot(
                res.jsr_sweep_db,
                res.cn0_sweep_single_dbhz,
                color="#dc2626",
                marker="x",
                markersize=3.5,
                linewidth=1.7,
                label="단일 안테나",
            )
            ax4.plot(
                res.jsr_sweep_db,
                res.cn0_sweep_pm_dbhz,
                color="#0284c7",
                linestyle="--",
                linewidth=1.9,
                label="CRPA Power-Min",
            )
            ax4.plot(
                res.jsr_sweep_db,
                res.cn0_sweep_mvdr_dbhz,
                color="#16a34a",
                linewidth=2.1,
                label=f"CRPA MVDR (+{10*np.log10(n_elem):.1f}dB)",
            )
            ax4.axhline(28.0, color="#d97706", linestyle=":", linewidth=1.4, label="추적 임계 (28 dB-Hz)")
            ax4.axvline(j1_jsr, color="#475569", linestyle="--", linewidth=1.1)
            ax4.set_title("4. 재밍 강도(J/S) 대 유효 C/N0 생존 곡선", fontsize=10.5, fontweight="bold", pad=8)
            ax4.set_xlabel("입력 재밍 대 신호비 J/S [dB]", fontsize=9.0)
            ax4.set_ylabel("유효 C/N0 [dB-Hz]", fontsize=9.0)
            ax4.set_ylim(-15, 60)
            ax4.grid(True, linestyle="--", alpha=0.5)
            ax4.legend(loc="lower left", fontsize=7.8)

    def _draw_3d_crpa_beam_pattern(
        self,
        ax_3d,
        res,
        n_elem: int,
        sat_ang: float,
        j1_ang: float,
        j2_en: bool,
        j2_ang: float,
    ) -> None:
        """Render 3D upper-hemisphere directional CRPA MVDR gain surface with 3D spatial null trenches."""
        # Subsample theta (-90..+90 deg) and cross-elevation phi (-60..+60 deg)
        stride = 8
        theta_sub_deg = res.angles_deg[::stride]
        gain_1d_db = np.maximum(res.pattern_mvdr_db[::stride], -50.0)
        # Convert [-50 dB, 0 dB] into normalized radial gain [0.05, 1.05]
        r_1d = (gain_1d_db + 52.0) / 52.0

        phi_deg = np.linspace(-65.0, 65.0, 25)
        theta_grid_rad, phi_grid_rad = np.meshgrid(np.deg2rad(theta_sub_deg), np.deg2rad(phi_deg))
        # Modulate cross-axis with smooth patch dipole factor cos(phi)
        r_grid = r_1d[None, :] * (0.25 + 0.75 * np.cos(phi_grid_rad))

        # Convert spherical (r, theta, phi) into 3D Cartesian (X: Array axis, Y: Cross axis, Z: Boresight)
        x_3d = r_grid * np.sin(theta_grid_rad) * np.cos(phi_grid_rad)
        y_3d = r_grid * np.sin(phi_grid_rad)
        z_3d = r_grid * np.cos(theta_grid_rad) * np.cos(phi_grid_rad)

        ax_3d.plot_surface(
            x_3d,
            y_3d,
            z_3d,
            cmap="viridis",
            alpha=0.84,
            edgecolor="none",
            antialiased=True,
        )

        # Draw 3D rays for Satellite (Green) and Jammers (Red/Orange)
        def plot_ray(ang_deg: float, col: str, lbl: str) -> None:
            rad = np.deg2rad(ang_deg)
            rx, rz = 1.18 * np.sin(rad), 1.18 * np.cos(rad)
            ax_3d.plot([0.0, rx], [0.0, 0.0], [0.0, rz], color=col, linewidth=2.6, label=lbl)
            ax_3d.scatter([rx], [0.0], [rz], color=col, s=55)

        plot_ray(sat_ang, "#16a34a", f"위성 주빔 ({sat_ang:+.0f}°)")
        plot_ray(j1_ang, "#dc2626", f"재머1 널 ({j1_ang:+.0f}°)")
        if j2_en:
            plot_ray(j2_ang, "#d97706", f"재머2 널 ({j2_ang:+.0f}°)")

        ax_3d.set_xlim(-1.2, 1.2)
        ax_3d.set_ylim(-1.0, 1.0)
        ax_3d.set_zlim(0.0, 1.2)
        ax_3d.set_xlabel("배열 축 sin(θ)", fontsize=8.5)
        ax_3d.set_ylabel("교차 축 sin(φ)", fontsize=8.5)
        ax_3d.set_zlabel("천정 이득 G(θ,φ)", fontsize=8.5)
        ax_3d.set_title(
            f"[3D] CRPA MVDR 입체 빔/널 방사패턴 ({n_elem}소자) · 3D 공간 널 협곡",
            fontsize=11.0,
            fontweight="bold",
            pad=10,
        )
        ax_3d.view_init(elev=26, azim=-55)
        ax_3d.legend(loc="upper right", fontsize=7.4)

    def get_lv1_easy_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">1. 일상생활 비유로 30초 만에 이해하기</h3>
            <ul>
                <li><b>CRPA 배열 안테나 = 방향을 골라 막는 '공간 노이즈 캔슬링 이어폰'</b>:
                    안테나가 1개뿐이면 옆에서 확성기(재머)를 틀었을 때 귀를 막을 방법이 없습니다.
                    하지만 안테나 여러 개(예: 4소자)를 9.5 cm 간격으로 나란히 놓으면, <b>확성기 소리가 각 안테나에 도착하는 미세한 시간차(위상차)</b>가 생깁니다.
                    이 신호들에 반대 위상을 곱해 더하면 <b>확성기가 있는 방향의 소리만 정확히 0(-50 dB 골짜기 = Null)으로 상쇄</b>시킬 수 있습니다!</li>
                <li><b>자유도(N - 1) 법칙</b>:
                    안테나가 <code>N</code>개이면 동시에 <b>최대 <code>N - 1</code>개 방향의 확성기(재머)</b>를 막을 수 있습니다. (예: 4소자 안테나 = 재머 3개까지 동시 차단!)</li>
                <li><b>Power-Min vs MVDR 차이점 한줄 요약</b>:
                    <ul>
                        <li><b>Power-Min (전력 최소화)</b>: <i>"내 친구(위성)가 어디 있는지는 모르겠지만, 일단 시끄럽게 소리 지르는 놈(재머) 방향만 무조건 귀를 막자!"</i> (자세 센서가 필요 없어 가장 널리 쓰임)</li>
                        <li><b>MVDR (Capon 빔포머)</b>: <i>"내 친구(위성)가 +15°에 있는 걸 아니까, 친구 쪽으로는 귀를 활짝 열고(0 dB 보존) 확성기 방향만 귀를 막자!"</i> (가장 성능이 좋음)</li>
                    </ul>
                </li>
            </ul>

            <h3 style="color:#1E3A2F;">2. 그래프 어디를 봐야 하나요?</h3>
            <ul>
                <li><b>2D 그래프 1·2 · 3D 입체 탭</b>: 빨간/주황 점선(재머 각도)에서 파란색 그래프가 <b>지하 -50 dB 아래로 푹 꺼진 골짜기(Null)</b>를 보세요. 그 방향 전파만 10만 분의 1로 깎아버린 것입니다. 반면 초록색 선(위성 각도)은 0 dB로 살아 있습니다.</li>
                <li><b>2D 그래프 3 (고유값 막대)</b>: 재머가 2대 켜지면 빨간색 기둥 2개만 솟고 나머지는 0 dB(잡음 바닥)에 깔립니다. 즉 <b>빨간 기둥 개수 = 현재 공격 중인 재머 개수</b>입니다.</li>
                <li><b>2D 그래프 4 (생존 곡선)</b>: 재밍 세기(가로축)가 세질 때 빨간선(단일 안테나)은 추락하지만, 파란/초록선(CRPA)은 45 dB-Hz 이상을 유지하며 살아남습니다.</li>
            </ul>
        </div>
        """

    def get_lv1_easy_c_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">3. C 코드 3단계 한글 흐름도 (항재밍 가중치 계산)</h3>
            <pre style="background-color:#FAFBF9; border:1px solid #D8E2DB; color:#14532d; padding:12px; border-radius:8px; font-family:'Malgun Gothic',monospace; font-size:13px;">
[매 10ms마다 FPGA로부터 4개 안테나 신호의 상관 행렬 R_xx (4x4 크기)를 읽어옴]
1. 대각 로딩(Diagonal Loading):
   0으로 나누기(역행렬 폭발)가 안 일어나도록 대각선 원소(R[i][i])에 아주 작은 양수(+0.1)를 더함!

2. 4x4 행렬 분해 (복소 Cholesky 분해: R = L * L^H):
   무거운 역행렬 공식을 직접 쓰지 않고 삼각형 행렬 L로 깔끔하게 쪼갬!

3. 가중치 w 구해서 FPGA에 써주기:
   1번 안테나는 1.0으로 고정하고, 전체 소음 에너지를 최소로 만드는 복소수 가중치 [w0, w1, w2, w3] 계산 완료!</pre>
        </div>
        """

    def get_theory_html(self) -> str:
        return """
        <div style="line-height: 1.68;">
            <h2 style="color:#1E3A2F; margin-top:0;">1. GNSS 링크 버짓과 단일 안테나의 J/S 한계</h2>
            <p>
                위성 수신 전력은 약 <b>-130 dBm</b>에 불과해 수 km 거리의 소출력 재머에도
                <a href="glossary://jsr" style="color:#1E3A2F; font-weight:700;">J/S (재밍 대 신호 전력비) 60~80 dB</a> 환경에 노출됩니다.
                CDMA 확산 이득(+30.1 dB)과 상관 적분만으로는 J/S 35 dB 이상에서 추적 임계값(28 dB-Hz) 아래로 붕괴합니다.
            </p>

            <h2 style="color:#1E3A2F;">2. CRPA 조향 벡터(Steering Vector)와 공간 자유도(N-1)</h2>
            <p>
                반파장(<code>λ/2 ≈ 9.5 cm</code>) 간격 <a href="glossary://crpa" style="color:#1E3A2F; font-weight:700;">CRPA</a> ULA 배열에서
                평면파 입사각 θ에 대한 위상 지연 벡터가 <a href="glossary://steering_vector" style="color:#1E3A2F; font-weight:700;">조향 벡터 a(θ)</a>입니다.
                복소 가중치 <code>w</code>의 내적 <code>wᴴ a(θ) = 0</code>을 만족시키는 방향에 -40~-60 dB 널을 형성하며,
                N소자 배열은 대수적으로 최대 <b>N - 1개의 독립 재머 부공간</b>을 제거할 수 있습니다.
            </p>

            <h2 style="color:#1E3A2F;">3. 적응형 빔포밍 양대 해법: Power Minimization vs MVDR</h2>
            <ul>
                <li>
                    <b><a href="glossary://power_minimization" style="color:#1E3A2F; font-weight:700;">전력 최소화 (Power Minimization / PI)</a></b>:
                    위성 도래각 정보 없이 기준 소자 구속(<code>wᴴ δ₁ = 1</code>) 하에
                    <a href="glossary://covariance_matrix_rxx" style="color:#1E3A2F; font-weight:700;">공분산 행렬 R_xx</a> 출력 전력을 최소화해 열잡음 이상의 재머만 선택적으로 널링합니다.
                </li>
                <li>
                    <b><a href="glossary://mvdr_capon" style="color:#1E3A2F; font-weight:700;">MVDR (Capon 빔포머)</a></b>:
                    궤도력/INS 자세로 얻은 위성 방향 구속(<code>wᴴ a(θ_s) = 1</code>)으로 신호 왜곡을 0 dB로 억제하면서
                    재머 널링과 <code>10·log10(N) dB</code> 배열 이득을 동시에 달성합니다.
                </li>
            </ul>
        </div>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. N소자 균일 선형 배열(ULA) 조향 벡터 및 공간 공분산 행렬 R_xx",
                latex_expr=(
                    r"\mathbf{a}(\theta) = \left[ 1, \ e^{j 2\pi \frac{d}{\lambda}\sin\theta}, \ \dots, \ e^{j 2\pi \frac{d}{\lambda}(N-1)\sin\theta} \right]^T"
                    "\n"
                    r"\hat{\mathbf{R}}_{xx} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{x}[k]\mathbf{x}^H[k] + \delta_{\mathrm{DL}} \mathbf{I}_N"
                ),
                intuitive_summary="입사각 θ에 따른 각 소자의 기하학적 위상 지연을 묶은 것이 조향 벡터 a(θ)이며, K개의 수신 스냅샷 외적 평균에 대각 로딩 δ_DL을 더해 공간 공분산 행렬 R_xx를 추정합니다.",
                symbols=[
                    SymbolSpec("N, d/\\lambda", "소자 수 및 파장 대비 간격", "개, -", "배열 안테나 소자 수(자유도 N-1) 및 반파장 간격(0.5λ ≈ 9.51 cm)", "crpa"),
                    SymbolSpec(r"\mathbf{a}(\theta)", "배열 조향 벡터", "C^N", "입사각 θ 방향 평면파가 각 소자에 만드는 복소 위상차 벡터", "steering_vector"),
                    SymbolSpec(r"\hat{\mathbf{R}}_{xx}", "공간 공분산 행렬", "C^{N×N}", "소자 간 상호상관 에너지를 담은 에르미트 양정치 행렬", "covariance_matrix_rxx"),
                    SymbolSpec(r"\delta_{\mathrm{DL}}", "대각 로딩 정칙화 계수", "W", "역행렬 수치 폭발을 막기 위해 대각선에 더하는 미세 잡음 전력", "diagonal_loading"),
                ],
            ),
            FormulaSpec(
                title="수식 2. 전력 최소화 (Power Minimization / PI) 적응형 널링 가중치",
                latex_expr=(
                    r"\min_{\mathbf{w}} \ \mathbf{w}^H \hat{\mathbf{R}}_{xx} \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^H \boldsymbol{\delta}_1 = 1 \quad \Longrightarrow \quad \mathbf{w}_{\mathrm{PM}} = \frac{\hat{\mathbf{R}}_{xx}^{-1} \boldsymbol{\delta}_1}{\boldsymbol{\delta}_1^H \hat{\mathbf{R}}_{xx}^{-1} \boldsymbol{\delta}_1}"
                ),
                intuitive_summary="위성 방향 정보 없이 1번 안테나 소자 가중치만 1로 구속(δ_1 = [1,0,...,0]^T)한 뒤 출력 전력을 최소화하여 열잡음보다 강한 모든 재머 방향에 널(Null)을 형성합니다.",
                symbols=[
                    SymbolSpec(r"\boldsymbol{\delta}_1", "기준 소자 구속 벡터", "R^N", "첫 번째 기준 안테나 소자만 1이고 나머지가 0인 단위 벡터 [1, 0, ..., 0]^T", "power_minimization"),
                    SymbolSpec(r"\mathbf{w}_{\mathrm{PM}}", "전력 최소화 가중치 벡터", "C^N", "공분산 역행렬의 첫 번째 열벡터를 정규화한 항재밍 가중치", "power_minimization"),
                ],
            ),
            FormulaSpec(
                title="수식 3. MVDR (Capon) 무왜곡 빔포머 가중치 및 배열 방사패턴 G(θ)",
                latex_expr=(
                    r"\mathbf{w}_{\mathrm{MVDR}} = \frac{\hat{\mathbf{R}}_{xx}^{-1} \mathbf{a}(\theta_s)}{\mathbf{a}^H(\theta_s) \hat{\mathbf{R}}_{xx}^{-1} \mathbf{a}(\theta_s)}, \qquad G_{\mathrm{dB}}(\theta) = 20 \log_{10} \left| \mathbf{w}^H \mathbf{a}(\theta) \right|"
                ),
                intuitive_summary="희망 위성 방향 조향 벡터 a(θ_s)에 대해 w^H a(θ_s) = 1 (0 dB 이득 보존) 구속조건을 걸어, 재머는 널링하고 위성 방향으로는 주빔(Main Lobe)을 형성합니다.",
                symbols=[
                    SymbolSpec(r"\mathbf{a}(\theta_s)", "희망 위성 조향 벡터", "C^N", "보호하고자 하는 목표 위성의 입사각 θ_s 조향 벡터", "steering_vector"),
                    SymbolSpec(r"\mathbf{w}_{\mathrm{MVDR}}", "MVDR 최적 가중치 벡터", "C^N", "위성 방향 무왜곡(0 dB) 및 출력 SINR을 극대화하는 가중치", "mvdr_capon"),
                    SymbolSpec(r"G_{\mathrm{dB}}(\theta)", "방향별 배열 방사패턴 이득", "dB", "각도 θ에서 들어오는 신호에 대한 배열 안테나의 공간 필터 응답", "crpa"),
                ],
            ),
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <b style="color:#1E3A2F; font-size:14px;">슬라이더 조작 실습 미션 (Observation Guide)</b>
            <ol style="margin-top:8px; margin-bottom:4px; padding-left:20px;">
                <li style="margin-bottom:6px;">
                    <b>미션 1 (공간 자유도 N-1 한계 실험):</b> <b>[재머 2 활성화]</b>가 켜진 상태(재머 2개)에서 <b>[안테나 소자 수 (N)]</b>를 <code style="color:#dc2626;">2 소자</code>로 줄여보세요.
                    자유도(<code>N-1 = 1</code>)가 재머 수(2개)보다 부족해지면 2D 그래프 1과 상단 유효 C/N0에 어떤 일이 벌어지는지 확인하세요.
                </li>
                <li style="margin-bottom:6px;">
                    <b>미션 2 (위성 인접 각도 재밍과 주빔 침식):</b> 4소자 상태에서 <b>[재머 1 입사각]</b>을 위성 입사각(<code>+15°</code>) 바로 옆인 <code style="color:#dc2626;">+18°</code>로 옮겨보세요.
                    재머를 깎기 위한 널(Null)이 위성 신호까지 함께 깎아먹는 현상을 관찰하고, 소자 수를 <code style="color:#2563eb;">8 소자</code>로 늘려 빔 폭을 좁히면 회복되는지 확인하세요.
                </li>
                <li>
                    <b>미션 3 (Spatial Aliasing / Grating Lobe 관찰):</b> <b>[소자 간격 (d/λ)]</b>을 반파장(<code>0.50 λ</code>)에서 <code style="color:#d97706;">0.95 λ</code>로 넓혀보세요.
                    엉뚱한 각도에 가짜 널(Grating Null)이 복제되어 나타나는 공간 에일리어싱 현상을 확인하세요.
                </li>
            </ol>
        </div>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <h2 style="color:#1E3A2F; margin-top:0;">실전 임베디드 C/FPGA 포팅 가이드: N×N 복소 에르미트 공분산 행렬 추정 및 역행렬 해법</h2>

            <div style="background-color:#F4F7F4; border-left:4px solid #1E3A2F; padding:12px 14px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#1E3A2F;">[실무 아키텍처] FPGA + ARM Cortex-R/M 또는 Zynq SoC</b><br>
                1. <b>FPGA 영역:</b> N개 RF 채널의 ADC 샘플 <code>x[k]</code>를 받아 K개 스냅샷(예: K=256 또는 IIR 지수 이동평균) 동안 상삼각 외적 <code>R_ij += x_i * conj(x_j)</code>를 누적하고, 계산된 가중치 <code>w</code>로 실시간 내적 <code>y[k] = w^H x[k]</code>를 수행합니다.<br>
                2. <b>MCU/DSP 영역:</b> 주기적으로(예: 1~10 ms마다) FPGA에서 4×4 공분산 행렬 <code>R_xx</code>를 읽어 <b>대각 로딩(Diagonal Loading)</b>을 적용한 뒤, <b>복소 Cholesky 분해(<code>R = L · L^H</code>)</b>로 <code>R_xx · w = δ_1</code>을 풀어 가중치 레지스터를 갱신합니다. (직접 역행렬 공식보다 Cholesky 전방/후방 대입이 수치적으로 훨씬 안정적입니다!)
            </div>

            <h3 style="color:#0f172a;">1. 4소자 CRPA 복소 Cholesky 분해 기반 Power-Min / MVDR 가중치 계산 C99 코드</h3>
            <pre style="background-color:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#include &lt;math.h&gt;
#include &lt;stdint.h&gt;

#define N_ELEM 4

typedef struct { float re, im; } cfloat_t;

static inline cfloat_t c_add(cfloat_t a, cfloat_t b) { return (cfloat_t){a.re + b.re, a.im + b.im}; }
static inline cfloat_t c_sub(cfloat_t a, cfloat_t b) { return (cfloat_t){a.re - b.re, a.im - b.im}; }
static inline cfloat_t c_conj(cfloat_t a)            { return (cfloat_t){a.re, -a.im}; }
static inline cfloat_t c_mul(cfloat_t a, cfloat_t b) {
    return (cfloat_t){a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

/* 에르미트 양정치 행렬 R_xx (4x4)의 복소 Cholesky 분해: R_xx + diag_load * I = L * L^H */
int CRPA_ComplexCholesky4x4(const cfloat_t R_in[N_ELEM][N_ELEM], float diag_load, cfloat_t L[N_ELEM][N_ELEM]) {
    for (int i = 0; i &lt; N_ELEM; ++i) {
        for (int j = 0; j &lt; N_ELEM; ++j) L[i][j] = (cfloat_t){0.0f, 0.0f};
    }

    for (int i = 0; i &lt; N_ELEM; ++i) {
        for (int j = 0; j &lt;= i; ++j) {
            cfloat_t sum = R_in[i][j];
            if (i == j) sum.re += diag_load; /* 대각 로딩(Diagonal Loading)으로 특이행렬 방지 */

            for (int k = 0; k &lt; j; ++k) {
                sum = c_sub(sum, c_mul(L[i][k], c_conj(L[j][k])));
            }
            if (i == j) {
                if (sum.re &lt;= 1e-12f) return -1; /* 양정치성 상실 에러 방어 */
                L[i][i].re = sqrtf(sum.re);
                L[i][i].im = 0.0f;
            } else {
                float inv_diag = 1.0f / L[j][j].re;
                L[i][j].re = sum.re * inv_diag;
                L[i][j].im = sum.im * inv_diag;
            }
        }
    }
    return 0;
}

/* Power-Minimization (w_PM = R^-1 * delta_1 정규화) 해법: L * y = delta_1 전방 대입 후 L^H * z = y 후방 대입 */
void CRPA_SolvePowerMinWeights(const cfloat_t L[N_ELEM][N_ELEM], cfloat_t w_out[N_ELEM]) {
    cfloat_t y[N_ELEM] = {{0}}, z[N_ELEM] = {{0}};

    /* 1. 전방 대입 (Forward Substitution): L * y = [1, 0, 0, 0]^T */
    for (int i = 0; i &lt; N_ELEM; ++i) {
        cfloat_t sum = (i == 0) ? (cfloat_t){1.0f, 0.0f} : (cfloat_t){0.0f, 0.0f};
        for (int k = 0; k &lt; i; ++k) sum = c_sub(sum, c_mul(L[i][k], y[k]));
        float inv_l = 1.0f / L[i][i].re;
        y[i] = (cfloat_t){sum.re * inv_l, sum.im * inv_l};
    }

    /* 2. 후방 대입 (Backward Substitution): L^H * z = y */
    for (int i = N_ELEM - 1; i &gt;= 0; --i) {
        cfloat_t sum = y[i];
        for (int k = i + 1; k &lt; N_ELEM; ++k) sum = c_sub(sum, c_mul(c_conj(L[k][i]), z[k]));
        float inv_l = 1.0f / L[i][i].re;
        z[i] = (cfloat_t){sum.re * inv_l, sum.im * inv_l};
    }

    /* 3. 1번 기준 안테나 가중치가 1.0 + j0.0 이 되도록 정규화 */
    float norm_re = (z[0].re &gt; 1e-12f) ? (1.0f / z[0].re) : 1.0f;
    for (int i = 0; i &lt; N_ELEM; ++i) {
        w_out[i].re = z[i].re * norm_re;
        w_out[i].im = z[i].im * norm_re;
    }
}</pre>
        </div>
        """
