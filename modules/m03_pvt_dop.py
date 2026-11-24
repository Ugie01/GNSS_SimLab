"""
Module 03: GNSS Skyplot, Dilution of Precision (DOP), Iterative Gauss-Newton
Weighted Least Squares (WLS) Positioning, and Atmospheric Error / SBAS (KASS/WAAS).
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
    QVBoxLayout,
    QWidget,
)

from core.pvt_dop import run_pvt_simulation_pipeline
from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Module03PvtDop(BaseLabModule):
    """Module 3: Skyplot, DOP, Gauss-Newton WLS Iterative PVT, and SBAS Augmentation."""

    module_id = "m03_pvt_dop"
    module_title = "M03. Skyplot · DOP · 가우스-뉴턴 WLS 측위 · SBAS"
    module_subtitle = "위성 기하 배치(Skyplot/DOP)와 가중최소자승(WLS) 반복 수렴, 전리층/대류권 오차 및 KASS/SBAS 보정 효과"
    module_badge = "MOD 03"

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

        # 1. Sky Visibility & Urban Canyon Masking
        g_sky = QGroupBox("1. 위성 차폐 조건 (Skyplot & 도심 빌딩 협곡)")
        g_sky.setStyleSheet(group_style)
        f_sky = QFormLayout(g_sky)
        f_sky.setSpacing(8)

        self.slider_mask = QSlider(Qt.Orientation.Horizontal)
        self.slider_mask.setRange(5, 55)  # 5° to 55° (allows testing <4 visible satellites)
        self.slider_mask.setValue(10)
        self.lbl_mask = QLabel("10°")
        self.lbl_mask.setStyleSheet("font-weight:700; color:#1d4ed8;")
        self.slider_mask.valueChanged.connect(self._on_mask_changed)

        m_row = QHBoxLayout()
        m_row.addWidget(self.slider_mask, stretch=1)
        m_row.addWidget(self.lbl_mask)
        f_sky.addRow("전방위 앙각 마스크:", m_row)

        self.slider_canyon = QSlider(Qt.Orientation.Horizontal)
        self.slider_canyon.setRange(0, 65)  # 0° to 65° E/W building blockage
        self.slider_canyon.setValue(0)
        self.lbl_canyon = QLabel("0° (개활지)")
        self.lbl_canyon.setStyleSheet("font-weight:700; color:#dc2626;")
        self.slider_canyon.valueChanged.connect(self._on_canyon_changed)

        c_row = QHBoxLayout()
        c_row.addWidget(self.slider_canyon, stretch=1)
        c_row.addWidget(self.lbl_canyon)
        f_sky.addRow("도심 빌딩 차폐(동/서):", c_row)
        layout.addWidget(g_sky)

        # 2. Atmospheric & Multipath Error Budget
        g_err = QGroupBox("2. 대기 지연 오차 · 의사거리 측정 잡음 (UERE)")
        g_err.setStyleSheet(group_style)
        f_err = QFormLayout(g_err)
        f_err.setSpacing(8)

        self.spin_iono = QDoubleSpinBox()
        self.spin_iono.setRange(0.5, 20.0)
        self.spin_iono.setDecimals(1)
        self.spin_iono.setSingleStep(0.5)
        self.spin_iono.setValue(4.5)
        self.spin_iono.setSuffix(" m (Zenith)")
        self.spin_iono.valueChanged.connect(lambda _: self.run_simulation())
        f_err.addRow("천정 전리층 지연 (I_z):", self.spin_iono)

        self.spin_uere = QDoubleSpinBox()
        self.spin_uere.setRange(0.2, 8.0)
        self.spin_uere.setDecimals(1)
        self.spin_uere.setSingleStep(0.2)
        self.spin_uere.setValue(1.4)
        self.spin_uere.setSuffix(" m (1σ)")
        self.spin_uere.valueChanged.connect(lambda _: self.run_simulation())
        f_err.addRow("기본 다중경로/열잡음:", self.spin_uere)
        layout.addWidget(g_err)

        # 3. WLS Weighting Policy
        g_wls = QGroupBox("3. 가우스-뉴턴 최소자승 가중치 정책")
        g_wls.setStyleSheet(group_style)
        v_wls = QVBoxLayout(g_wls)
        v_wls.setSpacing(8)

        self.chk_el_weight = QCheckBox("앙각 기반 가중치 W_ii = sin²(El_i) 적용 (WLS)")
        self.chk_el_weight.setChecked(True)
        self.chk_el_weight.setStyleSheet("font-weight:700; color:#1e293b;")
        self.chk_el_weight.toggled.connect(lambda _: self.run_simulation())
        v_wls.addWidget(self.chk_el_weight)

        wls_desc = QLabel(
            "※ 체크 해제 시 모든 위성을 동일 가중치(W=I, OLS)로 처리하여 저앙각 위성의 큰 대기·다중경로 오차가 측위 해를 오염시킵니다."
        )
        wls_desc.setWordWrap(True)
        wls_desc.setStyleSheet("font-size:11.5px; color:#475569;")
        v_wls.addWidget(wls_desc)

        layout.addWidget(g_wls)
        return container

    def _on_mask_changed(self, val: int) -> None:
        self.lbl_mask.setText(f"{val}°")
        self.run_simulation()

    def _on_canyon_changed(self, val: int) -> None:
        self.lbl_canyon.setText(f"{val}° (동/서 빌딩)" if val > 0 else "0° (개활지)")
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

        self.lbl_sky_readout = QLabel()
        self.lbl_sky_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#e2e8f0; border:none;"
        )
        self.lbl_dop_readout = QLabel()
        self.lbl_dop_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#7dd3fc; border:none;"
        )
        self.lbl_sbas_readout = QLabel()
        self.lbl_sbas_readout.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:12px; color:#86efac; border:none;"
        )

        grid.addWidget(self.lbl_sky_readout, 0, 0)
        grid.addWidget(self.lbl_dop_readout, 0, 1)
        grid.addWidget(self.lbl_sbas_readout, 0, 2)
        vbox.addWidget(cards_frame)

        self.fig = Figure(figsize=(10.2, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)
        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        widgets = [
            self.slider_mask,
            self.slider_canyon,
            self.spin_iono,
            self.spin_uere,
            self.chk_el_weight,
        ]
        for w in widgets:
            w.blockSignals(True)

        m_val = int(preset_data.get("mask_angle_deg", 10))
        self.slider_mask.setValue(m_val)
        self.lbl_mask.setText(f"{m_val}°")

        c_val = int(preset_data.get("urban_canyon_deg", 0))
        self.slider_canyon.setValue(c_val)
        self.lbl_canyon.setText(f"{c_val}° (동/서 빌딩)" if c_val > 0 else "0° (개활지)")

        self.spin_iono.setValue(float(preset_data.get("zenith_iono_m", 4.5)))
        self.spin_uere.setValue(float(preset_data.get("base_uere_noise_m", 1.4)))
        self.chk_el_weight.setChecked(bool(preset_data.get("use_elevation_weighting", True)))

        for w in widgets:
            w.blockSignals(False)

    def run_simulation(self) -> None:
        mask_deg = float(self.slider_mask.value())
        canyon_deg = float(self.slider_canyon.value())
        iono_m = self.spin_iono.value()
        uere_m = self.spin_uere.value()
        use_wls = self.chk_el_weight.isChecked()

        res = run_pvt_simulation_pipeline(
            mask_angle_deg=mask_deg,
            urban_canyon_deg=canyon_deg,
            zenith_iono_m=iono_m,
            base_uere_noise_m=uere_m,
            use_elevation_weighting=use_wls,
        )

        n_vis = len(res.active_satellites)
        n_tot = len(res.all_satellites)
        status_color = "#16A34A" if n_vis >= 4 else "#DC2626"
        status_txt = f"3D+Clock 정상 ({n_vis}/{n_tot}기)" if n_vis >= 4 else f"위성 부족 ({n_vis}기 < 4기)"

        self.lbl_sky_readout.setText(
            f"<div style='font-size:11.5px; color:#1E3A2F; font-weight:800;'>가시 위성 · WLS 측위 상태</div>"
            f"<div style='font-size:13.5px; color:{status_color}; font-weight:900; margin-top:3px;'>{status_txt}</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"{'WLS (sin²El)' if use_wls else 'OLS (균등)'} · 시계 bias {res.est_clock_bias_m:,.1f} m</div>"
        )

        d = res.dop
        dop_color = "#16A34A" if d.pdop < 2.5 else ("#B45309" if d.pdop < 5.0 else "#DC2626")
        dop_grade = "우수" if d.pdop < 2.5 else ("보통" if d.pdop < 5.0 else "기하 배치 불량")
        self.lbl_dop_readout.setText(
            f"<div style='font-size:11.5px; color:#B45309; font-weight:800;'>정밀도 저하율 (DOP · {dop_grade})</div>"
            f"<div style='font-size:13.5px; color:{dop_color}; font-weight:900; margin-top:3px;'>"
            f"PDOP {d.pdop:.2f} · HDOP {d.hdop:.2f} · VDOP {d.vdop:.2f}</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"EDOP {d.edop:.2f} · NDOP {d.ndop:.2f} · GDOP {d.gdop:.2f}</div>"
        )

        h_red = (1 - res.sbas_h_rms_m / max(res.standalone_h_rms_m, 1e-3)) * 100
        v_red = (1 - res.sbas_v_rms_m / max(res.standalone_v_rms_m, 1e-3)) * 100
        self.lbl_sbas_readout.setText(
            f"<div style='font-size:11.5px; color:#1D4ED8; font-weight:800;'>단독 vs SBAS(KASS) 100회 RMS</div>"
            f"<div style='font-size:13.5px; color:#111827; font-weight:900; margin-top:3px;'>"
            f"수평 {res.standalone_h_rms_m:.2f}m → {res.sbas_h_rms_m:.2f}m (-{h_red:.0f}%)</div>"
            f"<div style='font-size:11.5px; color:#4B5563; margin-top:2px;'>"
            f"수직 {res.standalone_v_rms_m:.2f}m → {res.sbas_v_rms_m:.2f}m (-{v_red:.0f}%)</div>"
        )

        if n_vis < 4:
            self.set_easy_coach_message(
                f"<b>[사수의 실시간 쉬운 해설]</b> 현재 보이는 위성이 <b>{n_vis}개</b>뿐입니다. "
                "내 3차원 위치(X, Y, Z)와 시계 오차(b)까지 <b>미지수가 4개</b>이므로 최소 위성 4개가 필요합니다. 좌측 앙각 마스크나 빌딩 차폐 각도를 낮춰보세요."
            )
        else:
            canyon_note = (
                f"동·서쪽 빌딩(<b>{canyon_deg:.0f}°</b>)에 위성이 가려 동서 오차 배율(<b>EDOP={d.edop:.2f}</b>)이 남북(<b>NDOP={d.ndop:.2f}</b>)보다 커지며 2D 그래프 4 산포가 좌우로 길게 퍼졌습니다."
                if canyon_deg >= 35
                else f"하늘 사방에 위성 <b>{n_vis}개</b>가 고르게 분포해 3D 위치 배율(<b>PDOP={d.pdop:.2f}</b>)이 안정적입니다."
            )
            self.set_easy_coach_message(
                f"<b>[사수의 실시간 쉬운 해설]</b> (1) {canyon_note} &nbsp;|&nbsp; "
                f"(2) 한국형 보정위성(<b>KASS/SBAS</b>, 초록 점)을 적용하면 전리층 오차가 상쇄되어 수평 오차가 <b>{res.standalone_h_rms_m:.2f}m → {res.sbas_h_rms_m:.2f}m</b>로 줄어듭니다."
            )

        # Tab 0: Clean 2D graphs
        self.fig.clear()
        ax1, ax2, ax3, ax4 = self.allocate_2d_axes(self.fig, projections=("polar", None, None, None))
        self._draw_2d_plots(ax1, ax2, ax3, ax4, res, mask_deg, n_vis, n_tot)
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        # Tab 1: Dedicated 3D Sky Dome + DOP Ellipsoid
        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            self._draw_3d_sky_dome_and_dop_ellipsoid(ax_3d, res, n_vis)
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def _draw_2d_plots(self, ax1, ax2, ax3, ax4, res, mask_deg: float, n_vis: int, n_tot: int) -> None:
        d = res.dop
        # Subplot 1: Polar Skyplot
        if ax1 is not None:
            ax1.set_theta_zero_location("N")
            ax1.set_theta_direction(-1)
            ax1.set_rlim(0, 90)
            ax1.set_rticks([30, 60, 90])
            ax1.set_yticklabels(["60°", "30°", "0°"], fontsize=7.5, color="#475569")

            theta_ring = np.linspace(0, 2 * np.pi, 180)
            r_mask = np.full_like(theta_ring, 90.0 - mask_deg)
            ax1.plot(theta_ring, r_mask, color="#dc2626", linestyle="--", linewidth=1.2)

            for sat in res.all_satellites:
                az_rad = np.deg2rad(sat.az_deg)
                r_zenith = 90.0 - sat.el_deg
                if sat.is_visible:
                    ax1.scatter([az_rad], [r_zenith], color="#2563eb", s=75, zorder=5)
                    ax1.annotate(
                        f"G{sat.prn:02d}",
                        xy=(az_rad, r_zenith),
                        xytext=(az_rad + 0.08, max(r_zenith - 6, 2)),
                        fontsize=7.8,
                        fontweight="bold",
                        color="#1e3a8a",
                    )
                else:
                    ax1.scatter([az_rad], [r_zenith], color="#94a3b8", marker="x", s=60, zorder=4)
                    ax1.annotate(
                        f"G{sat.prn:02d}",
                        xy=(az_rad, r_zenith),
                        fontsize=7.2,
                        color="#64748b",
                    )

            ax1.set_title(f"1. 위성 Skyplot (가시 {n_vis}/{n_tot}기)", fontsize=10.5, fontweight="bold", pad=12)

        # Subplot 2: DOP Breakdown Bar Chart
        if ax2 is not None:
            dop_names = ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "EDOP", "NDOP"]
            dop_vals = [d.gdop, d.pdop, d.hdop, d.vdop, d.tdop, d.edop, d.ndop]
            bar_colors = ["#1d4ed8", "#2563eb", "#16a34a", "#d97706", "#0284c7", "#0d9488", "#059669"]
            bars = ax2.bar(dop_names, dop_vals, color=bar_colors, width=0.55, alpha=0.88)
            ax2.axhline(2.5, color="#16a34a", linestyle="--", linewidth=1.2, label="우수 (<2.5)")
            ax2.axhline(5.0, color="#dc2626", linestyle=":", linewidth=1.2, label="경고 (>5.0)")
            for b, val in zip(bars, dop_vals):
                ax2.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.12,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.0,
                    fontweight="bold",
                )
            ax2.set_ylim(0, max(max(dop_vals) * 1.22, 4.0))
            ax2.set_title("2. 정밀도 저하율 (DOP) 성분별 분석", fontsize=10.5, fontweight="bold", pad=10)
            ax2.set_ylabel("DOP 배율", fontsize=9.0)
            ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right", fontsize=8.0)

        # Subplot 3: Gauss-Newton WLS Iterative Convergence
        if ax3 is not None:
            if n_vis < 4:
                ax3.text(
                    0.5,
                    0.5,
                    f"[경고] 가시 위성 부족 ({n_vis}기 < 4기)\n\n"
                    "미지수 4개 [X, Y, Z, 시계 b]를 풀기 위해\n"
                    "최소 4기 이상의 가시 위성이 필요합니다.",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#dc2626",
                    bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fef2f2", "edgecolor": "#dc2626"},
                    transform=ax3.transAxes,
                )
                ax3.set_title("3. 가우스-뉴턴 WLS 수렴 (측위 불가)", fontsize=10.5, fontweight="bold", pad=8)
            else:
                iters_step = np.arange(1, len(res.iter_step_norms_m) + 1)
                iters_pos = np.arange(0, len(res.iter_pos_errors_m))
                ax3.semilogy(
                    iters_step,
                    np.maximum(res.iter_step_norms_m, 1e-9),
                    color="#2563eb",
                    marker="o",
                    linewidth=2.0,
                    label="보정량 ||Δx_k|| [m]",
                )
                if len(res.iter_pos_errors_m) > 1:
                    ax3.semilogy(
                        iters_pos,
                        np.maximum(res.iter_pos_errors_m, 1e-9),
                        color="#dc2626",
                        marker="s",
                        linestyle="--",
                        linewidth=1.8,
                        label="3D 위치 오차 [m]",
                    )
                ax3.axhline(1e-4, color="#16a34a", linestyle=":", linewidth=1.2, label="수렴 임계 (0.1mm)")
                ax3.set_title("3. 가우스-뉴턴 WLS 반복 수렴 ([0,0,0] 출발)", fontsize=10.5, fontweight="bold", pad=8)
                ax3.set_xlabel("반복 횟수 k [Iteration]", fontsize=9.0)
                ax3.set_ylabel("크기 [m, 로그 스케일]", fontsize=9.0)
                ax3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0e}"))
                ax3.grid(True, which="both", linestyle="--", alpha=0.5)
                ax3.legend(loc="upper right", fontsize=8.0)

        # Subplot 4: Standalone GPS vs SBAS (KASS/WAAS) 100-Epoch ENU Horizontal Scatter Cloud
        if ax4 is not None:
            if n_vis < 4:
                ax4.text(
                    0.5,
                    0.5,
                    f"[경고] 가시 위성 {n_vis}기 (측위 불가)\n"
                    "행렬 H^T W H 의 Rank < 4 이어서\n"
                    "공분산이 발산(GDOP = ∞)합니다.",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#dc2626",
                    bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fef2f2", "edgecolor": "#dc2626"},
                    transform=ax4.transAxes,
                )
                ax4.set_title("4. 단독 측위 vs SBAS 오차 분포 (불가)", fontsize=10.5, fontweight="bold", pad=8)
            else:
                ax4.scatter(
                    res.mc_standalone_enu_m[:, 0],
                    res.mc_standalone_enu_m[:, 1],
                    color="#dc2626",
                    alpha=0.55,
                    s=22,
                    label=f"단독 (수평 {res.standalone_h_rms_m:.2f}m)",
                )
                ax4.scatter(
                    res.mc_sbas_enu_m[:, 0],
                    res.mc_sbas_enu_m[:, 1],
                    color="#16a34a",
                    alpha=0.75,
                    s=22,
                    label=f"SBAS (수평 {res.sbas_h_rms_m:.2f}m)",
                )
                ax4.scatter([0.0], [0.0], color="#0f172a", marker="+", s=100, linewidths=2.0, zorder=6)
                circ_t = np.linspace(0, 2 * np.pi, 120)
                ax4.plot(2.0 * np.cos(circ_t), 2.0 * np.sin(circ_t), color="#15803d", linestyle="--", linewidth=1.1, label="2.0m 원")
                ax4.set_title("4. 단독 측위 vs SBAS(KASS) 수평 오차 산포", fontsize=10.5, fontweight="bold", pad=8)
                ax4.set_xlabel("동쪽 오차 ΔEast [m]", fontsize=9.0)
                ax4.set_ylabel("북쪽 오차 ΔNorth [m]", fontsize=9.0)
                ax4.set_aspect("equal", adjustable="datalim")
                ax4.grid(True, linestyle="--", alpha=0.5)
                ax4.legend(loc="upper right", fontsize=8.0)

    def _draw_3d_sky_dome_and_dop_ellipsoid(self, ax_3d, res, n_vis: int) -> None:
        """Render 3D upper-hemisphere satellite LOS rays and 3D DOP uncertainty ellipsoid."""
        # 1. Translucent upper-hemisphere sky dome (radius = 1.0)
        az_grid, el_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0, 0.5 * np.pi, 14))
        dome_e = np.cos(el_grid) * np.sin(az_grid)
        dome_n = np.cos(el_grid) * np.cos(az_grid)
        dome_u = np.sin(el_grid)
        ax_3d.plot_wireframe(dome_e, dome_n, dome_u, color="#cbd5e1", linewidth=0.45, alpha=0.4)

        # 2. Plot 3D LOS rays from receiver (0,0,0) to each satellite on the unit sky dome
        for sat in res.all_satellites:
            az_r = np.deg2rad(sat.az_deg)
            el_r = np.deg2rad(sat.el_deg)
            sx = float(np.cos(el_r) * np.sin(az_r))
            sy = float(np.cos(el_r) * np.cos(az_r))
            sz = float(np.sin(el_r))
            if sat.is_visible:
                ax_3d.plot([0.0, sx], [0.0, sy], [0.0, sz], color="#2563eb", linewidth=1.5, alpha=0.75)
                ax_3d.scatter([sx], [sy], [sz], color="#1d4ed8", s=55, zorder=6)
                ax_3d.text(sx * 1.05, sy * 1.05, sz * 1.05, f"G{sat.prn:02d}", fontsize=7.5, fontweight="bold", color="#1e3a8a")
            else:
                ax_3d.plot([0.0, sx], [0.0, sy], [0.0, sz], color="#94a3b8", linestyle=":", linewidth=1.0, alpha=0.5)
                ax_3d.scatter([sx], [sy], [sz], color="#94a3b8", marker="x", s=45)

        # 3. 3D DOP Uncertainty Ellipsoid centered at receiver (0, 0, 0)
        if n_vis >= 4:
            d = res.dop
            u_sph, v_sph = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 14))
            # Scale ellipsoid axes by (EDOP, NDOP, VDOP) normalized for visual clarity inside unit dome
            scale = 0.14
            rx = min(d.edop * scale, 0.85)
            ry = min(d.ndop * scale, 0.85)
            rz = min(d.vdop * scale, 0.85)
            ell_x = rx * np.cos(u_sph) * np.sin(v_sph)
            ell_y = ry * np.sin(u_sph) * np.sin(v_sph)
            ell_z = np.maximum(0.0, rz * np.cos(v_sph))  # Upper half of DOP uncertainty ellipsoid
            ax_3d.plot_surface(ell_x, ell_y, ell_z, color="#ef4444", alpha=0.35, edgecolor="none")

        ax_3d.scatter([0.0], [0.0], [0.0], color="#0f172a", marker="^", s=70, label="수신기 (0,0,0)")
        ax_3d.set_xlim(-1.1, 1.1)
        ax_3d.set_ylim(-1.1, 1.1)
        ax_3d.set_zlim(0.0, 1.1)
        ax_3d.set_xlabel("East (동)", fontsize=8.5)
        ax_3d.set_ylabel("North (북)", fontsize=8.5)
        ax_3d.set_zlabel("Up (천정)", fontsize=8.5)
        ax_3d.set_title(
            f"[3D] 상반구 시선벡터(LOS) 콘 · DOP 오차 타원체 (EDOP={res.dop.edop:.2f}, NDOP={res.dop.ndop:.2f}, VDOP={res.dop.vdop:.2f})",
            fontsize=11.0,
            fontweight="bold",
            pad=10,
        )
        ax_3d.view_init(elev=26, azim=-40)

    def get_lv1_easy_theory_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">1. 일상생활 비유로 30초 만에 이해하기</h3>
            <ul>
                <li><b>왜 위성이 3개가 아니라 4개 필요할까요?</b>:
                    방 안에서 내 위치(가로, 세로, 높이 3개)를 찾으려면 원래 기준점 3개와의 거리만 알면 됩니다.
                    하지만 <b>내 손목시계(수신기 시계)가 몇 초 틀어졌는지(4번째 미지수!)</b>를 몰라서 재는 거리마다 똑같이 엉터리 길이가 더해져 있습니다(그래서 '가짜 거리 = 의사거리'라고 부릅니다).
                    미지수가 <code>(X, Y, Z, 내 시계오차)</code> 4개니까 방정식(위성)도 <b>최소 4개</b>가 필요한 것입니다!</li>
                <li><b>DOP (정밀도 저하율: 사방에서 줄다리기)</b>:
                    위성들이 사방에서 골고루 나를 끈으로 당겨주면 내 위치가 흔들리지 않습니다(<b>DOP 낮음 = 좋음</b>).
                    반면 빌딩에 가려서 위성들이 하늘 한쪽 구석에만 몰려 있으면 줄이 한쪽으로만 당겨져 작은 오차도 수십 미터로 뻥튀기됩니다(<b>DOP 높음 = 나쁨</b>).</li>
                <li><b>왜 항상 고도(VDOP)가 수평(HDOP)보다 부정확할까요?</b>:
                    동서남북 옆으로는 사방에 위성이 있지만, <b>내 발밑(땅속)에는 위성이 없고 오직 머리 위쪽 하늘에만 위성이 있어서</b> 위아래 방향 줄다리기가 헐겁기 때문입니다!</li>
            </ul>

            <h3 style="color:#1E3A2F;">2. 그래프 어디를 봐야 하나요?</h3>
            <ul>
                <li><b>2D 그래프 1 (Skyplot)</b>: 고개를 들어 하늘을 올려다본 지도입니다. 가운데가 내 머리 꼭대기(천정, 90°)이고 바깥 원이 지평선(0°)입니다. 파란 점이 보이는 위성, 회색 X가 빌딩에 가려진 위성입니다.</li>
                <li><b>2D 그래프 2 (DOP 막대)</b>: 막대가 <b>낮을수록(2.5 이하) 정밀도가 좋은 것</b>입니다. 항상 <code>VDOP(수직)</code> 막대가 <code>HDOP(수평)</code>보다 높게 솟아 있는 것을 확인하세요.</li>
                <li><b>2D 그래프 3 (WLS 수렴)</b>: 처음에 내 위치를 <b>지구 한가운데 핵(0,0,0 — 오차 6,378 km!)</b>으로 대충 찍고 시작해도, 가우스-뉴턴 공식으로 <b>5번만 고쳐 계산하면 오차가 0.1 mm 이하로 뚝 떨어지는 마법</b>을 보여줍니다.</li>
                <li><b>2D 그래프 4 (단독 vs SBAS)</b>: 빨간 점(일반 GPS)은 전리층 오차 때문에 몇 미터씩 퍼져 있지만, <b>초록 점(한국형 보정위성 KASS/SBAS)</b>은 원점(0,0) 근처 1~2 m 안에 촘촘히 모입니다.</li>
            </ul>
        </div>
        """

    def get_lv1_easy_c_html(self) -> str:
        return """
        <div style="line-height: 1.72; color: #14532d;">
            <h3 style="margin-top:0; color:#1E3A2F;">3. C 코드 3단계 한글 흐름도 (가우스-뉴턴 위치 계산 루프)</h3>
            <pre style="background-color:#FAFBF9; border:1px solid #D8E2DB; color:#14532d; padding:12px; border-radius:8px; font-family:'Malgun Gothic',monospace; font-size:13px;">
[초기값: 내 위치 = (0, 0, 0), 내 시계오차 = 0 으로 시작!]
for (반복 = 1; 반복 <= 6; 반복++) {
    1. 현재 추정 위치에서 각 위성까지의 예상 거리(r_hat) 계산 (+ 지구 자전 Sagnac 보정)
    2. 실제 측정된 의사거리와 예상 거리의 차이(잔차 = 측정치 - 예상치) 계산
    3. 고도가 낮은 위성(대기 오차가 큼)은 가중치(sin²(앙각))를 낮게 주어 4x4 행렬에 누적
    4. 4x4 연립방정식을 풀어 보정량 [ΔX, ΔY, ΔZ, Δ시계] 산출 후 내 위치에 더하기!
    5. 보정량이 1mm 보다 작아지면 "수렴 완료!" 하고 루프 탈출!
}</pre>
        </div>
        """

    def get_theory_html(self) -> str:
        return """
        <div style="line-height: 1.68;">
            <h2 style="color:#1E3A2F; margin-top:0;">1. 의사거리 관측 방정식과 4-상태 벡터 선형화</h2>
            <p>
                위성 원자시계와 달리 수신기 TCXO는 클럭 오프셋(<code>δt_u</code>)을 가지며, 빛이 <b>1 μs당 약 300 m</b>를 진행하므로
                미지수 4개 <code>[X, Y, Z, b = c·δt_u]</code>를 동시에 추정하기 위해 Rank 4 이상의 <a href="glossary://pseudorange" style="color:#1E3A2F; font-weight:700;">의사거리(Pseudorange)</a> 방정식(최소 4기 가시위성)이 필수적입니다.
            </p>

            <h2 style="color:#1E3A2F;">2. 기하 공분산 행렬 Q_enu 및 상반구 DOP 비대칭성</h2>
            <p>
                <a href="glossary://dop" style="color:#1E3A2F; font-weight:700;">DOP (Dilution of Precision)</a>는
                기하 행렬 <a href="glossary://geometry_matrix" style="color:#1E3A2F; font-weight:700;">H</a>의 정규행렬 역행렬
                <code>Q = (Hᵀ H)⁻¹</code>의 대각 성분 제곱근으로, 측정 잡음(<a href="glossary://uere" style="color:#1E3A2F; font-weight:700;">UERE</a>)이 각 축으로 전파되는 배율입니다.
                지구가 하반구 시선(El &lt; 0°)을 물리적으로 차폐하므로 수직 축 방향 시선 다양성이 제한되어 구조적으로 항상 <b>VDOP &gt; HDOP</b>(약 1.5~2.5배)가 성립합니다.
            </p>

            <h2 style="color:#1E3A2F;">3. 가우스-뉴턴 WLS 수렴과 한국형 KASS (SBAS) 오차 모델</h2>
            <p>
                앙각 기반 가중 행렬 <code>W_ii = sin²(El_i)</code>를 도입한
                <a href="glossary://wls_gauss_newton" style="color:#1E3A2F; font-weight:700;">가중최소자승(WLS)</a>은 저앙각 다중경로/대기 오차 기여를 억제하며,
                지구 중심 <code>(0,0,0)</code> 초기값에서도 5~6회 반복 내에 <code>0.1 mm</code> 이하로 수렴합니다.
                여기에 <a href="glossary://ionospheric_delay" style="color:#1E3A2F; font-weight:700;">전리층</a> ·
                <a href="glossary://tropospheric_delay" style="color:#1E3A2F; font-weight:700;">대류권 지연</a> 광역 보정정보를 제공하는
                <a href="glossary://sbas" style="color:#1E3A2F; font-weight:700;">SBAS (KASS)</a>를 적용하면 수평/수직 RMS 편향이 1~2 m 급으로 급감합니다.
            </p>
        </div>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. 의사거리(Pseudorange) 물리 관측 모델 및 시선 방향 선형화",
                latex_expr=(
                    r"\rho_i = \sqrt{(X^{(i)} - X)^2 + (Y^{(i)} - Y)^2 + (Z^{(i)} - Z)^2} + b + I_i + T_i + \varepsilon_i"
                    "\n"
                    r"H_i = \left[ -\frac{X^{(i)} - \hat{X}}{\hat{r}_i}, \quad -\frac{Y^{(i)} - \hat{Y}}{\hat{r}_i}, \quad -\frac{Z^{(i)} - \hat{Z}}{\hat{r}_i}, \quad 1 \right]"
                ),
                intuitive_summary="i번째 위성의 의사거리 ρ_i는 기하학적 거리 r_i에 수신기 시계 바이어스 b(=c·δt_u)와 대기 오차가 더해진 값이며, 이를 미지수 [X, Y, Z, b]로 편미분한 행이 기하 행렬 H_i입니다.",
                symbols=[
                    SymbolSpec(r"\rho_i", "i번째 위성 의사거리", "m", "코드 추적 루프(DLL)가 측정한 전파 도달 시간 × 광속 c", "pseudorange"),
                    SymbolSpec(r"X^{(i)}, Y^{(i)}, Z^{(i)}", "위성 ECEF 좌표", "m", "방송 궤도력(Ephemeris)으로부터 계산한 송신 시점 위성 좌표", "ecef"),
                    SymbolSpec("b", "수신기 시계 바이어스 거리", "m", "수신기 시계 오차 δt_u에 광속 c를 곱한 등가 거리 (b = c·δt_u)", "pseudorange"),
                    SymbolSpec("I_i, T_i", "전리층 및 대류권 지연", "m", "대기권 통과 시 전파 굴절 및 지연에 의해 추가된 오차", "ionospheric_delay"),
                    SymbolSpec("H_i", "기하 행렬의 i번째 행", "-", "수신기에서 위성을 바라보는 단위 시선 벡터의 음수(-e_i^T)와 시계 항(1)", "geometry_matrix"),
                ],
            ),
            FormulaSpec(
                title="수식 2. 가우스-뉴턴 가중최소자승(WLS) 반복 갱신 방정식",
                latex_expr=(
                    r"\Delta \mathbf{x}_k = \left( H_k^T W H_k \right)^{-1} H_k^T W \Delta \boldsymbol{\rho}_k, \qquad \hat{\mathbf{x}}_{k+1} = \hat{\mathbf{x}}_k + \Delta \mathbf{x}_k, \qquad W_{ii} = \frac{\sin^2(\mathrm{El}_i)}{\sigma_0^2}"
                ),
                intuitive_summary="앙각(El_i)이 낮아 대기 오차와 다중경로가 심한 위성에는 sin²(El_i)의 낮은 가중치 W를 부여하고, 보정량 Δx_k가 0에 수렴할 때까지 5~6회 반복 갱신합니다.",
                symbols=[
                    SymbolSpec(r"\Delta \mathbf{x}_k", "상태 보정 벡터", "m", "k번째 반복에서의 [ΔX, ΔY, ΔZ, Δb]^T 위치·시계 보정량", "wls_gauss_newton"),
                    SymbolSpec(r"\Delta \boldsymbol{\rho}_k", "의사거리 잔차 벡터", "m", "실제 측정 의사거리와 현재 추정 위치 기반 예측 의사거리의 차이", "wls_gauss_newton"),
                    SymbolSpec("W", "대각 가중치 행렬", "1/m²", "각 위성 측정치의 분산 역수 (앙각이 높을수록 큰 가중치 부여)", "wls_gauss_newton"),
                ],
            ),
            FormulaSpec(
                title="수식 3. 지역 수평 좌표계(ENU) 공분산 행렬 Q와 정밀도 저하율(DOP)",
                latex_expr=(
                    r"Q_{\mathrm{enu}} = \left( H_{\mathrm{enu}}^T H_{\mathrm{enu}} \right)^{-1}, \quad \mathrm{HDOP} = \sqrt{Q_{11} + Q_{22}}, \quad \mathrm{VDOP} = \sqrt{Q_{33}}, \quad \mathrm{PDOP} = \sqrt{\mathrm{HDOP}^2 + \mathrm{VDOP}^2}"
                ),
                intuitive_summary="기하 행렬 H를 ENU 좌표계로 표현했을 때 (H^T H)^{-1}의 대각 원소들이 각각 동(E), 북(N), 천정(U), 시각(T) 방향의 오차 증폭 배율 제곱이 됩니다.",
                symbols=[
                    SymbolSpec(r"Q_{\mathrm{enu}}", "기하 공분산 행렬", "-", "4×4 대칭 행렬로 대각 원소 Q_11, Q_22, Q_33, Q_44가 각 축의 분산 계수", "dop"),
                    SymbolSpec(r"\mathrm{HDOP, VDOP}", "수평 및 수직 DOP", "-", "수평 위치 오차 및 고도 오차 증폭률 (항상 VDOP > HDOP)", "dop"),
                    SymbolSpec(r"\mathrm{PDOP, GDOP}", "3차원 위치 및 기하 DOP", "-", "3D 위치 증폭률 PDOP 및 시계까지 포함한 전체 증폭률 GDOP", "dop"),
                ],
            ),
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <b style="color:#1E3A2F; font-size:14px;">슬라이더 조작 실습 미션 (Observation Guide)</b>
            <ol style="margin-top:8px; margin-bottom:4px; padding-left:20px;">
                <li style="margin-bottom:6px;">
                    <b>미션 1 (강남 도심 협곡 빌딩 차폐 실험):</b> <b>[도심 빌딩 차폐(동/서)]</b> 슬라이더를 <code style="color:#dc2626;">52°</code>로 올려보세요.
                    2D 그래프 1(Skyplot)에서 동·서쪽 위성이 차폐되면 2D 그래프 2에서 왜 <b>NDOP보다 EDOP가 급격히 치솟는지</b>, 그리고 [그래프 3 · 4 크게 보기]에서 산포도가 좌우로 늘어나는지 관찰하세요.
                </li>
                <li style="margin-bottom:6px;">
                    <b>미션 2 (앙각 가중치 WLS vs OLS 비교):</b> <b>[전방위 앙각 마스크]</b>를 <code style="color:#1d4ed8;">5°</code>로 낮춘 뒤
                    <b>[앙각 기반 가중치 W_ii = sin²(El_i)]</b> 체크박스를 껐다 켜보며 2D 그래프 4의 단독 측위 RMS 오차 변화를 확인하세요.
                </li>
                <li>
                    <b>미션 3 (지구 중심에서 출발하는 WLS 수렴):</b> [그래프 3 · 4 크게 보기]의 2D 그래프 3에서 초기 추정치를 지구 중심 <code>[0,0,0]</code>(오차 6,378 km)으로 두어도
                    가우스-뉴턴 WLS가 몇 회 반복 만에 <code>0.1 mm</code> 이하로 수렴하는지 확인하세요.
                </li>
            </ol>
        </div>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <div style="line-height: 1.65;">
            <h2 style="color:#1E3A2F; margin-top:0;">실전 임베디드 C 포팅 가이드: N×4 WLS 정규방정식과 지구 자전(Sagnac) 보정</h2>

            <div style="background-color:#FEF2F2; border-left:4px solid #DC2626; padding:12px 14px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#991B1B;">[실무 포팅 필수 체크리스트 2가지]</b><br>
                1. <b>동적 메모리 할당(malloc) 금지:</b> 가시 위성 수 <code>N</code>이 변하더라도 정규방정식 행렬 <code>A = H^T W H</code>는 항상 <b>고정 크기 4×4 대칭 양정치 행렬</b>이고 우변 벡터 <code>b = H^T W Δρ</code>는 <b>길이 4 벡터</b>입니다. <code>N×4</code> 전체 행렬을 저장할 필요 없이 위성 루프를 돌며 4×4 행렬에 바로 누적(Accumulate)하면 RAM을 거의 쓰지 않습니다!<br>
                2. <b>지구 자전(Sagnac 효과) 보정 필수:</b> 위성 전파가 날아오는 약 <code>67~80 ms</code> 동안 지구(ECEF 좌표계)가 자전하므로, 이를 보정하지 않으면 동서 방향으로 <b>약 20~30 m의 고정 측위 오차</b>가 발생합니다.
            </div>

            <h3 style="color:#0f172a;">1. RAM 절약형 4×4 정규방정식 직접 누적 & Sagnac 보정 C99 코드</h3>
            <pre style="background-color:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px; line-height:1.5;">
#include &lt;math.h&gt;
#include &lt;stdint.h&gt;

#define OMGE_EARTH  7.2921151467e-5  /* WGS-84 지구 자전 각속도 [rad/s] */
#define CLIGHT      299792458.0      /* 광속 [m/s] */

/* 전파 비행시간 동안의 지구 자전(Sagnac Effect)을 포함한 기하 거리 및 단위 시선 벡터 계산 */
double GNSS_GeometricRangeWithSagnac(const double sat_ecef[3], const double rx_ecef[3], double los_out[3]) {
    double dx = sat_ecef[0] - rx_ecef[0];
    double dy = sat_ecef[1] - rx_ecef[1];
    double dz = sat_ecef[2] - rx_ecef[2];
    double r  = sqrt(dx * dx + dy * dy + dz * dz);

    /* Sagnac 지구 자전 보정량: dR_sagnac = (omega_e / c) * (X_sat * Y_rx - Y_sat * X_rx) */
    double sagnac_m = (OMGE_EARTH / CLIGHT) * (sat_ecef[0] * rx_ecef[1] - sat_ecef[1] * rx_ecef[0]);

    los_out[0] = dx / r;
    los_out[1] = dy / r;
    los_out[2] = dz / r;
    return r + sagnac_m;
}

/* N개의 위성 데이터를 순회하며 4x4 정규방정식 (H^T W H) * dx = (H^T W d_rho) 직접 누적 */
void GNSS_AccumulateNormalEquations(int num_sats,
                                    const double sat_ecef[][3],
                                    const double pseudorange_m[],
                                    const double el_rad[],
                                    const double rx_est[4], /* [X, Y, Z, clock_m] */
                                    double A_4x4[4][4],
                                    double rhs_4[4]) {
    for (int i = 0; i &lt; 4; ++i) {
        rhs_4[i] = 0.0;
        for (int j = 0; j &lt; 4; ++j) A_4x4[i][j] = (i == j) ? 1e-9 : 0.0; /* 대각 로딩 */
    }

    for (int k = 0; k &lt; num_sats; ++k) {
        double e_los[3];
        double r_hat = GNSS_GeometricRangeWithSagnac(sat_ecef[k], rx_est, e_los);
        double res   = pseudorange_m[k] - (r_hat + rx_est[3]); /* 잔차 d_rho */
        double sin_e = sin(el_rad[k]);
        double w     = sin_e * sin_e; /* 앙각 가중치 W_kk = sin^2(El) */

        double h_row[4] = { -e_los[0], -e_los[1], -e_los[2], 1.0 };

        for (int i = 0; i &lt; 4; ++i) {
            rhs_4[i] += h_row[i] * w * res;
            for (int j = 0; j &lt; 4; ++j) {
                A_4x4[i][j] += h_row[i] * w * h_row[j];
            }
        }
    }
}</pre>
        </div>
        """
