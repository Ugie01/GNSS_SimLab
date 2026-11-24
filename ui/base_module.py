"""
Abstract Base Class `BaseLabModule` that all GNSS SimLab learning modules inherit from.
Modern Warm Alabaster & Forest Green SaaS layout with strict tab separation:
- Left Panel: Clean Scenario Presets + Parameter Sliders/Inputs (no cramped mission boxes)
- Top KPI Summary Cards (3 crisp white cards) + Real-Time Briefing Bar
- Right Panel (5 spacious dedicated tabs):
  [Tab 0: 2D 그래프] (Supports 1x2 Pair 1-2 wide view [default], 1x2 Pair 3-4 wide view, and 2x2 4-up view)
  [Tab 1: 3D 입체 뷰] (Dedicated interactive 360-deg rotatable 3D canvas)
  [Tab 2: Lv.1 쉬운 입문] (Plain analogies, graph reading guide, hands-on missions, Korean pseudocode — zero overlap with Lv.2)
  [Tab 3: Lv.2 심화 · 수식] (Engineering theory + LaTeX Formula Cards + glossary:// links)
  [Tab 4: 실전 C99 코드] (STM32 / DSP / RTOS C99 implementation guide)
"""

from __future__ import annotations

import json
import traceback
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.resource_helper import resource_path
from ui.formula_widget import FormulaCardWidget, FormulaSpec
from ui.glossary_dialog import show_glossary_popup


class BaseLabModule(QWidget):
    """Abstract base class for all pluggable GNSS SimLab educational simulation modules."""

    module_id: str = "m00_base"
    module_title: str = "기본 모듈"
    module_subtitle: str = "모듈 부제목"
    module_badge: str = "CORE"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._view_mode: str = "combined"  # Supports "combined", "pair1", "pair2", "2d", "3d"
        self._sub_2d_mode: str = "pair1"   # Default to spacious 1x2 side-by-side plots (①·②)
        self._level_mode: str = "both"     # Supports "lv1", "lv2", "both"
        self._font_zoom_steps: int = 1
        self._text_browsers: List[QTextBrowser] = []
        self._formula_cards: List[FormulaCardWidget] = []
        self._presets: Dict[str, Dict[str, Any]] = self._load_module_presets()
        self._setup_base_layout()

    @property
    def view_mode(self) -> str:
        """Return current plot visualization mode."""
        return self._view_mode

    @property
    def sub_2d_mode(self) -> str:
        """Return current 2D layout mode: 'pair1', 'pair2', or 'all4'."""
        return self._sub_2d_mode

    @property
    def level_mode(self) -> str:
        """Return current learning difficulty mode: 'lv1', 'both', or 'lv2'."""
        return self._level_mode

    def _load_module_presets(self) -> Dict[str, Dict[str, Any]]:
        preset_file = Path(resource_path("data/presets.json"))
        if not preset_file.exists():
            return {}
        try:
            with open(preset_file, "r", encoding="utf-8") as f:
                all_presets = json.load(f)
                if isinstance(all_presets, dict):
                    return all_presets.get(self.module_id, {})
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(
                self,
                "프리셋 파일 읽기 오류",
                f"시나리오 프리셋 파일을 불러오는 중 오류가 발생했습니다:\n{preset_file}\n\n{exc}",
            )
        return {}

    def _setup_base_layout(self) -> None:
        self.setStyleSheet("background-color: #F8F8F6;")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 14, 20, 14)
        root_layout.setSpacing(12)

        # ---------------- 1. MINIMAL SAAS HEADER ----------------
        header_row = QHBoxLayout()
        header_row.setContentsMargins(2, 0, 2, 0)
        header_row.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        caption_lbl = QLabel(f"DANAM NAVIGATION SW LAB  ·  {self.module_badge}")
        caption_lbl.setStyleSheet(
            "font-size: 11px; font-weight: 800; color: #5C6B62; letter-spacing: 1px; border: none;"
        )
        title_col.addWidget(caption_lbl)

        clean_title = self.module_title.replace("&", "·")
        t_lbl = QLabel(clean_title)
        t_lbl.setStyleSheet("font-size: 20px; font-weight: 900; color: #111827; border: none;")
        title_col.addWidget(t_lbl)

        clean_sub = self.module_subtitle.replace("&", "·")
        s_lbl = QLabel(clean_sub)
        s_lbl.setStyleSheet("font-size: 12.5px; color: #6B7280; border: none;")
        title_col.addWidget(s_lbl)

        header_row.addLayout(title_col, stretch=1)

        reset_btn = QPushButton("↺ 기본값 복원")
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FFFFFF;
                color: #1F2937;
                font-weight: 700;
                font-size: 12.5px;
                padding: 8px 15px;
                border-radius: 8px;
                border: 1px solid #D5DDD7;
            }
            QPushButton:hover {
                background-color: #F3F4F0;
                border-color: #94A39B;
            }
            """
        )
        reset_btn.clicked.connect(self.reset_to_default_preset)
        header_row.addWidget(reset_btn)

        run_btn = QPushButton("▶ 시뮬레이션 갱신")
        run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        run_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1E3A2F;
                color: #FFFFFF;
                font-weight: 800;
                font-size: 12.5px;
                padding: 8px 18px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #142820;
            }
            """
        )
        run_btn.clicked.connect(self._safe_run_simulation)
        header_row.addWidget(run_btn)

        root_layout.addLayout(header_row)

        # Build simulation widget early so we can extract its 3 KPI readout labels into the SaaS KPI row
        self.sim_canvas_widget = self.build_simulation_Plot_widget()
        kpi_labels = self._extract_kpi_labels(self.sim_canvas_widget)

        # ---------------- 2. TOP 3 WHITE KPI SUMMARY CARDS ----------------
        if kpi_labels:
            kpi_row = QHBoxLayout()
            kpi_row.setSpacing(10)
            accent_colors = ["#1E3A2F", "#B45309", "#1D4ED8"]
            for idx, lbl in enumerate(kpi_labels[:3]):
                card = QFrame()
                card.setStyleSheet(
                    f"""
                    QFrame {{
                        background-color: #FFFFFF;
                        border: 1px solid #E2E6E0;
                        border-top: 3px solid {accent_colors[idx % len(accent_colors)]};
                        border-radius: 10px;
                    }}
                    """
                )
                c_vbox = QVBoxLayout(card)
                c_vbox.setContentsMargins(14, 10, 14, 10)
                lbl.setParent(card)
                lbl.setWordWrap(True)
                lbl.setStyleSheet(
                    "font-family: 'Malgun Gothic', 'Consolas', sans-serif; "
                    "font-size: 12px; color: #374151; line-height: 1.45; border: none; background: transparent;"
                )
                c_vbox.addWidget(lbl)
                kpi_row.addWidget(card, stretch=1)
            root_layout.addLayout(kpi_row)

        # ---------------- 3. REAL-TIME BRIEFING BAR ----------------
        self.coach_frame = QFrame()
        self.coach_frame.setStyleSheet(
            """
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 10px;
            }
            """
        )
        coach_hbox = QHBoxLayout(self.coach_frame)
        coach_hbox.setContentsMargins(12, 8, 14, 8)
        coach_hbox.setSpacing(10)

        coach_badge = QLabel("실시간 브리핑")
        coach_badge.setStyleSheet(
            """
            background-color: #1E3A2F;
            color: #FFFFFF;
            font-size: 11.5px;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 6px;
            border: none;
            """
        )
        coach_hbox.addWidget(coach_badge)

        self.lbl_easy_coach = QLabel(
            "사수의 실시간 쉬운 해설 | 좌측 슬라이더나 프리셋을 바꾸면 현재 그래프 상태가 한눈에 이해되게 요약됩니다."
        )
        self.lbl_easy_coach.setWordWrap(True)
        self.lbl_easy_coach.setStyleSheet("font-size: 12.8px; color: #1F2937; line-height: 1.45; border: none;")
        coach_hbox.addWidget(self.lbl_easy_coach, stretch=1)

        root_layout.addWidget(self.coach_frame)

        # ---------------- 4. MAIN SPLITTER (LEFT PARAMS / RIGHT 5 CLEAN TABS) ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # LEFT PANEL (Clean White Parameter Card)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(365)
        left_scroll.setMaximumWidth(415)
        left_scroll.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #E2E6E0;
                border-radius: 12px;
                background-color: #FFFFFF;
            }
            """
        )
        left_container = QWidget()
        left_container.setStyleSheet("background-color: #FFFFFF;")
        self.left_layout = QVBoxLayout(left_container)
        self.left_layout.setContentsMargins(14, 14, 14, 14)
        self.left_layout.setSpacing(12)

        if self._presets:
            preset_box = QFrame()
            preset_box.setStyleSheet(
                """
                QFrame {
                    background-color: #F4F7F4;
                    border: 1px solid #D8E2DB;
                    border-radius: 8px;
                }
                """
            )
            p_layout = QVBoxLayout(preset_box)
            p_layout.setContentsMargins(10, 9, 10, 9)
            p_layout.setSpacing(6)
            p_lbl = QLabel("시나리오 프리셋 선택")
            p_lbl.setStyleSheet("font-size: 12.5px; font-weight: 800; color: #1E3A2F; border: none;")
            p_layout.addWidget(p_lbl)

            self.preset_combo = QComboBox()
            self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.preset_combo.setMinimumContentsLength(14)
            for key, val in self._presets.items():
                self.preset_combo.addItem(val.get("name", key).replace("&", "·"), userData=key)
            self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
            p_layout.addWidget(self.preset_combo)
            self.left_layout.addWidget(preset_box)

        param_widget = self.build_parameter_widget()
        self._polish_parameter_widget(param_widget)
        self.left_layout.addWidget(param_widget)
        self.left_layout.addStretch(1)

        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # RIGHT PANEL: 5 Spacious Dedicated Tabs
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #E2E6E0;
                border-radius: 12px;
                background-color: #FFFFFF;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #EEF0EB;
                color: #4B5563;
                font-weight: 700;
                font-size: 12px;
                padding: 8px 11px;
                margin-right: 3px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid #E2E6E0;
                border-bottom: none;
            }
            QTabBar::tab:hover {
                background-color: #E3EAE5;
                color: #1E3A2F;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #1E3A2F;
                font-weight: 800;
                border-top: 3px solid #1E3A2F;
            }
            """
        )

        # Add clean A- / A+ font zoom corner widget on the right side of the TabBar
        zoom_corner = QWidget()
        zoom_corner.setStyleSheet("background: transparent;")
        zc_layout = QHBoxLayout(zoom_corner)
        zc_layout.setContentsMargins(0, 2, 4, 4)
        zc_layout.setSpacing(4)
        for delta, z_txt in [(-1, "A-"), (1, "A+")]:
            z_btn = QPushButton(z_txt)
            z_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            z_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #FFFFFF;
                    color: #374151;
                    border: 1px solid #D5DDD7;
                    border-radius: 5px;
                    font-size: 11.5px;
                    font-weight: 800;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #E3EAE5;
                    color: #1E3A2F;
                }
                """
            )
            z_btn.clicked.connect(lambda _=False, d=delta: self.adjust_text_zoom(d))
            zc_layout.addWidget(z_btn)
        self.right_tabs.setCornerWidget(zoom_corner, Qt.Corner.TopRightCorner)

        # Tab 0: 2D Core Graphs (with Pair 1-2 / Pair 3-4 / 4-Up switcher)
        tab_2d = self._build_2d_tab()
        self.right_tabs.addTab(tab_2d, "2D 그래프")

        # Tab 1: 3D Interactive Visualization
        tab_3d = self._build_3d_tab()
        self.right_tabs.addTab(tab_3d, "3D 입체 뷰")

        # Tab 2: Lv.1 Easy Intro + Hands-on Slider Missions + Korean Pseudocode
        tab_lv1 = self._build_lv1_tab()
        self.right_tabs.addTab(tab_lv1, "Lv.1 입문")

        # Tab 3: Lv.2 Danam R&D Deep Theory + LaTeX Formulas
        tab_lv2 = self._build_lv2_tab()
        self.right_tabs.addTab(tab_lv2, "Lv.2 심화·수식")

        # Tab 4: Embedded C99 Firmware Porting Code
        tab_c = self._build_embedded_c_tab()
        self.right_tabs.addTab(tab_c, "실전 C99")

        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, stretch=1)

    def _extract_kpi_labels(self, sim_widget: QWidget) -> List[QLabel]:
        """Extract the 3 readout QLabels from subclass `build_simulation_Plot_widget` and hide the old dark frame."""
        extracted: List[QLabel] = []
        for child_frame in sim_widget.findChildren(QFrame):
            # Look for the top readout container holding QLabels inside sim_widget
            lbls = [l for l in child_frame.findChildren(QLabel) if l.parent() is child_frame]
            if len(lbls) >= 2:
                extracted = lbls
                child_frame.hide()
                break
        return extracted

    def _polish_parameter_widget(self, widget: QWidget) -> None:
        """
        Clean up all parameter group boxes, form layouts, and slider readout labels so:
        1. No '&' mnemonic underscores appear in titles or labels.
        2. Slider numeric readout labels never truncate on the right.
        """
        for gbox in widget.findChildren(QGroupBox):
            gbox.setTitle(gbox.title().replace("&", "·"))
            gbox.setStyleSheet(
                """
                QGroupBox {
                    font-weight: 800;
                    font-size: 12.8px;
                    color: #111827;
                    border: 1px solid #E2E6E0;
                    border-radius: 9px;
                    margin-top: 10px;
                    padding: 12px 10px 10px 10px;
                    background-color: #FAFBF9;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    background-color: #FFFFFF;
                    color: #1E3A2F;
                }
                """
            )

        for flay in widget.findChildren(QFormLayout):
            flay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            flay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            flay.setVerticalSpacing(9)
            flay.setHorizontalSpacing(10)

        for lbl in widget.findChildren(QLabel):
            if "&" in lbl.text():
                lbl.setText(lbl.text().replace("&", "·"))

        for combo in widget.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)

        for spin in widget.findChildren(QAbstractSpinBox):
            if isinstance(spin, (QSpinBox, QDoubleSpinBox)):
                suf = spin.suffix()
                if " (" in suf:
                    spin.setSuffix(suf.split(" (", 1)[0].rstrip())
            spin.setMinimumWidth(115)

        # Ensure every QLabel sharing an QHBoxLayout with a QSlider has ample width so values never clip
        for slider in widget.findChildren(QSlider):
            parent_lay = slider.parentWidget().layout() if slider.parentWidget() else None
            if parent_lay is None:
                continue
            for hbox in widget.findChildren(QHBoxLayout):
                has_slider = any(hbox.itemAt(i).widget() is slider for i in range(hbox.count()))
                if has_slider:
                    for i in range(hbox.count()):
                        w = hbox.itemAt(i).widget()
                        if isinstance(w, QLabel):
                            w.setMinimumWidth(78)
                            w.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                            w.setStyleSheet("font-weight: 800; color: #1E3A2F; font-size: 12px;")

    def _build_2d_tab(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: #FFFFFF;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(14, 12, 14, 10)
        vbox.setSpacing(8)

        # Sub-view pill bar (Pair 1-2 [Default] / Pair 3-4 / All 4)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar_title = QLabel("화면 배치:")
        bar_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #6B7280;")
        bar.addWidget(bar_title)

        self._sub_2d_btn_group = QButtonGroup(self)
        self._sub_2d_btn_group.setExclusive(True)
        sub_specs = [
            ("pair1", "그래프 1 · 2 크게 보기 (1x2)"),
            ("pair2", "그래프 3 · 4 크게 보기 (1x2)"),
            ("all4", "4개 한눈에 보기 (2x2)"),
        ]
        for idx, (s_key, s_lbl) in enumerate(sub_specs):
            btn = QPushButton(s_lbl)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #F3F4F0;
                    color: #4B5563;
                    font-weight: 700;
                    font-size: 12px;
                    padding: 6px 12px;
                    border-radius: 7px;
                    border: 1px solid #E2E6E0;
                }
                QPushButton:hover {
                    background-color: #E3EAE5;
                    color: #1E3A2F;
                }
                QPushButton:checked {
                    background-color: #1E3A2F;
                    color: #FFFFFF;
                    border-color: #1E3A2F;
                }
                """
            )
            if s_key == "pair1":
                btn.setChecked(True)
            btn.clicked.connect(lambda _=False, k=s_key: self.set_2d_sub_mode(k))
            self._sub_2d_btn_group.addButton(btn, idx)
            bar.addWidget(btn)

        bar.addStretch(1)
        vbox.addLayout(bar)

        # Canvas widget from subclass
        vbox.addWidget(self.sim_canvas_widget, stretch=1)

        if hasattr(self, "canvas"):
            self.mpl_toolbar = NavigationToolbar(self.canvas, container)
            self.mpl_toolbar.setStyleSheet(
                "QToolBar { background: #FAFBF9; border: 1px solid #E2E6E0; border-radius: 7px; padding: 2px; }"
            )
            vbox.addWidget(self.mpl_toolbar)

        return container

    def _build_3d_tab(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: #FFFFFF;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(14, 12, 14, 10)
        vbox.setSpacing(8)

        hint = QLabel("3D 입체 시각화 | 마우스 왼쪽 드래그로 360도 회전, 오른쪽 드래그 또는 휠로 확대/축소할 수 있습니다.")
        hint.setStyleSheet(
            "background-color: #F4F7F4; color: #1E3A2F; font-size: 12.5px; font-weight: 700; "
            "padding: 8px 12px; border-radius: 7px; border: 1px solid #D8E2DB;"
        )
        vbox.addWidget(hint)

        self.fig_3d = Figure(figsize=(8.8, 5.4), dpi=100)
        self.fig_3d.patch.set_facecolor("#FFFFFF")
        self.canvas_3d = FigureCanvas(self.fig_3d)
        vbox.addWidget(self.canvas_3d, stretch=1)

        self.mpl_toolbar_3d = NavigationToolbar(self.canvas_3d, container)
        self.mpl_toolbar_3d.setStyleSheet(
            "QToolBar { background: #FAFBF9; border: 1px solid #E2E6E0; border-radius: 7px; padding: 2px; }"
        )
        vbox.addWidget(self.mpl_toolbar_3d)
        return container

    def _build_lv1_tab(self) -> QWidget:
        """Tab 2: Pure Lv.1 beginner analogies, hands-on missions, and Korean pseudocode (zero Lv.2 duplication)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        container.setStyleSheet("background-color: #FAFBF9;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(18, 16, 18, 16)
        vbox.setSpacing(14)

        # Card 1: Everyday Analogies & Graph Reading Guide
        self.lv1_browser = QTextBrowser()
        self.lv1_browser.setOpenLinks(False)
        self.lv1_browser.anchorClicked.connect(self._handle_glossary_anchor)
        self.lv1_browser.setMinimumHeight(310)
        self.lv1_browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-left: 4px solid #16A34A;
                border-radius: 10px;
                padding: 18px;
                font-size: 14px;
                color: #1F2937;
            }
            """
        )
        self.lv1_browser.setHtml(self.get_lv1_easy_theory_html())
        self.lv1_browser.zoomIn(1)
        self._text_browsers.append(self.lv1_browser)
        vbox.addWidget(self.lv1_browser)

        # Card 2: Hands-on Slider Missions (moved out of cramped left panel!)
        self.guide_browser = QTextBrowser()
        self.guide_browser.setOpenLinks(False)
        self.guide_browser.anchorClicked.connect(self._handle_glossary_anchor)
        self.guide_browser.setMinimumHeight(185)
        self.guide_browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-left: 4px solid #1E3A2F;
                border-radius: 10px;
                padding: 16px;
                font-size: 13.5px;
                color: #1F2937;
            }
            """
        )
        self.guide_browser.setHtml(self.get_observation_guide_html())
        self.guide_browser.zoomIn(1)
        self._text_browsers.append(self.guide_browser)
        vbox.addWidget(self.guide_browser)

        # Card 3: Plain-Korean 3-Step Pseudocode Flowchart
        self.lv1_c_browser = QTextBrowser()
        self.lv1_c_browser.setOpenLinks(False)
        self.lv1_c_browser.anchorClicked.connect(self._handle_glossary_anchor)
        self.lv1_c_browser.setMinimumHeight(200)
        self.lv1_c_browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 10px;
                padding: 16px;
                font-size: 13.5px;
                color: #1F2937;
            }
            """
        )
        self.lv1_c_browser.setHtml(self.get_lv1_easy_c_html())
        self.lv1_c_browser.zoomIn(1)
        self._text_browsers.append(self.lv1_c_browser)
        vbox.addWidget(self.lv1_c_browser)

        vbox.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_lv2_tab(self) -> QWidget:
        """Tab 3: Pure Lv.2 Danam R&D deep theory + LaTeX Formula Cards (zero Lv.1 duplication)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        container.setStyleSheet("background-color: #FAFBF9;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(18, 16, 18, 16)
        vbox.setSpacing(14)

        self.theory_browser = QTextBrowser()
        self.theory_browser.setOpenLinks(False)
        self.theory_browser.anchorClicked.connect(self._handle_glossary_anchor)
        self.theory_browser.setMinimumHeight(320)
        self.theory_browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-left: 4px solid #1E3A2F;
                border-radius: 10px;
                padding: 18px;
                font-size: 14px;
                color: #1F2937;
            }
            """
        )
        self.theory_browser.setHtml(self.get_theory_html())
        self.theory_browser.zoomIn(1)
        self._text_browsers.append(self.theory_browser)
        vbox.addWidget(self.theory_browser)

        self._formula_cards.clear()
        for f_spec in self.get_formula_specs():
            card = FormulaCardWidget(spec=f_spec, on_glossary_link=self._open_glossary_term)
            self._formula_cards.append(card)
            vbox.addWidget(card)

        vbox.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_embedded_c_tab(self) -> QWidget:
        """Tab 4: Pure C99 / STM32 / DSP / RTOS implementation notes."""
        browser = QTextBrowser()
        browser.setOpenLinks(False)
        browser.anchorClicked.connect(self._handle_glossary_anchor)
        browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #FFFFFF;
                border: none;
                padding: 22px;
                font-size: 14px;
                color: #1F2937;
            }
            """
        )
        browser.setHtml(self.get_embedded_c_porting_html())
        browser.zoomIn(1)
        self._text_browsers.append(browser)
        return browser

    # ==========================================================================
    # Helpers for Clean Non-Overlapping 2D Subplot Allocation
    # ==========================================================================
    def allocate_2d_axes(
        self,
        fig: Figure,
        projections: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]] = (None, None, None, None),
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
        """
        Allocate only the visible subplots for `self._sub_2d_mode` ('pair1', 'pair2', or 'all4'),
        so 1x2 views have double the width and zero title/tick overlap.
        """
        if self._sub_2d_mode == "pair1":
            ax1 = fig.add_subplot(1, 2, 1, projection=projections[0])
            ax2 = fig.add_subplot(1, 2, 2, projection=projections[1])
            return ax1, ax2, None, None
        if self._sub_2d_mode == "pair2":
            ax3 = fig.add_subplot(1, 2, 1, projection=projections[2])
            ax4 = fig.add_subplot(1, 2, 2, projection=projections[3])
            return None, None, ax3, ax4

        ax1 = fig.add_subplot(2, 2, 1, projection=projections[0])
        ax2 = fig.add_subplot(2, 2, 2, projection=projections[1])
        ax3 = fig.add_subplot(2, 2, 3, projection=projections[2])
        ax4 = fig.add_subplot(2, 2, 4, projection=projections[3])
        return ax1, ax2, ax3, ax4

    def finalize_2d_figure(self, fig: Figure) -> None:
        """Apply generous padding so titles, axis labels, and legends never collide."""
        if self._sub_2d_mode in ("pair1", "pair2"):
            fig.tight_layout(pad=2.2, w_pad=3.0)
        else:
            fig.tight_layout(pad=1.8, w_pad=2.5, h_pad=2.8)

    def set_2d_sub_mode(self, sub_mode: str) -> None:
        if sub_mode in ("pair1", "pair2", "all4"):
            self._sub_2d_mode = sub_mode
            btn_idx = {"pair1": 0, "pair2": 1, "all4": 2}[sub_mode]
            btn = self._sub_2d_btn_group.button(btn_idx)
            if btn is not None:
                btn.setChecked(True)
            self.right_tabs.setCurrentIndex(0)
            self._safe_run_simulation()

    def set_view_mode(self, mode: str) -> None:
        """Switch view mode cleanly ('3d', '2d', 'combined', 'pair1', 'pair2')."""
        if mode not in ("combined", "2d", "3d", "pair1", "pair2"):
            return
        self._view_mode = mode
        if mode == "3d":
            self.right_tabs.setCurrentIndex(1)
        elif mode == "2d":
            self.set_2d_sub_mode("all4")
            return
        elif mode == "pair2":
            self.set_2d_sub_mode("pair2")
            return
        else:
            self.set_2d_sub_mode("pair1")
            return
        self._safe_run_simulation()

    def set_level_mode(self, level_mode: str) -> None:
        """
        Switch learning level mode ('lv1', 'lv2', 'both'):
        - 'lv1': Focuses Tab 2 (Lv.1 쉬운 입문)
        - 'lv2': Focuses Tab 3 (Lv.2 심화 · 수식)
        - 'both': Focuses Tab 0 (2D 핵심 그래프)
        """
        if level_mode not in ("lv1", "both", "lv2"):
            return
        self._level_mode = level_mode
        if level_mode == "lv1":
            self.right_tabs.setCurrentIndex(2)
        elif level_mode == "lv2":
            self.right_tabs.setCurrentIndex(3)
        else:
            self.right_tabs.setCurrentIndex(0)

    def adjust_text_zoom(self, delta: int) -> None:
        new_step = self._font_zoom_steps + delta
        if -2 <= new_step <= 6:
            self._font_zoom_steps = new_step
            for tb in self._text_browsers:
                if delta > 0:
                    tb.zoomIn(delta)
                elif delta < 0:
                    tb.zoomOut(-delta)

    def _on_preset_changed(self, idx: int) -> None:
        if idx < 0:
            return
        preset_key = self.preset_combo.itemData(idx)
        if preset_key in self._presets:
            self.apply_preset(self._presets[preset_key])
            self._safe_run_simulation()

    def reset_to_default_preset(self) -> None:
        if hasattr(self, "preset_combo") and self.preset_combo.count() > 0:
            if self.preset_combo.currentIndex() == 0:
                self._on_preset_changed(0)
            else:
                self.preset_combo.setCurrentIndex(0)
        else:
            self._safe_run_simulation()

    def _safe_run_simulation(self) -> None:
        """Execute run_simulation(), showing a clear popup alert if an error occurs."""
        try:
            self.run_simulation()
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "시뮬레이션 오류 알림",
                f"입력값 처리 또는 그래프 계산 중 문제가 발생했습니다.\n\n오류 내용: {type(exc).__name__}: {exc}",
            )

    def _handle_glossary_anchor(self, url: QUrl) -> None:
        if url.scheme() == "glossary":
            term_id = url.host() or url.path().lstrip("/")
            self._open_glossary_term(term_id)

    def _open_glossary_term(self, term_id: str) -> None:
        show_glossary_popup(term_id=term_id, parent=self)

    def set_easy_coach_message(self, html_msg: str) -> None:
        if hasattr(self, "lbl_easy_coach"):
            self.lbl_easy_coach.setText(html_msg)

    # ==========================================================================
    # Lv.1 Easy Intuition Providers
    # ==========================================================================
    def get_lv1_easy_theory_html(self) -> str:
        return "<p>이 모듈의 쉬운 일상생활 비유 해설입니다.</p>"

    def get_lv1_easy_c_html(self) -> str:
        return "<p>한글 의사코드(Pseudocode) 흐름도입니다.</p>"

    # ==========================================================================
    # Abstract Interface for Subclasses
    # ==========================================================================
    @abstractmethod
    def build_parameter_widget(self) -> QWidget:
        """Build and return the interactive parameter sliders/spinboxes widget for the left panel."""

    @abstractmethod
    def build_simulation_Plot_widget(self) -> QWidget:
        """Build and return the right-side Matplotlib canvas and numeric readouts widget."""

    @abstractmethod
    def get_theory_html(self) -> str:
        """Return Lv.2 engineering theory HTML including `glossary://<id>` links."""

    @abstractmethod
    def get_formula_specs(self) -> List[FormulaSpec]:
        """Return list of FormulaSpec objects rendered via Matplotlib mathtext cards."""

    @abstractmethod
    def get_observation_guide_html(self) -> str:
        """Return HTML containing 2-3 interactive slider mission guides."""

    @abstractmethod
    def get_embedded_c_porting_html(self) -> str:
        """Return HTML explaining embedded C99 porting pitfalls and reference code."""

    @abstractmethod
    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        """Apply a preset dictionary to the module's parameter controls."""

    @abstractmethod
    def run_simulation(self) -> None:
        """Execute core math/physics engine with current parameters and refresh 2D & 3D plots."""
