"""
Main Application Window for GNSS SimLab.
Styled after modern minimalist IT SaaS dashboards (Warm Sage Sidebar + Forest Green Accents + Clean White Cards).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core.resource_helper import resource_path
from ui.base_module import BaseLabModule
from ui.glossary_tab import GlossaryTabWidget
from ui.registry import ModuleRegistry
from ui.study_planner_tab import StudyPlannerTabWidget

# Concise, clean sidebar names that never clip horizontally
SIDEBAR_SHORT_NAMES: Dict[str, str] = {
    "m01_coords_time": "M01 · 좌표계와 1PPS 동기",
    "m02_signal_acq_track": "M02 · PRN 상관과 신호 추적",
    "m03_pvt_dop": "M03 · 위성 배치(DOP)와 측위",
    "m04_antijam_crpa": "M04 · 항재밍 배열(CRPA)",
    "m05_gnss_ins_ekf": "M05 · GNSS/INS 칼만필터",
    "m06_egi_raim_spoof": "M06 · EGI 무결성 · STAP",
}


class MainWindow(QMainWindow):
    """Primary desktop window hosting simulation modules, study planner, and glossary explorer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GNSS SimLab — 단암시스템즈 항법기술연구소 SW팀 입사 대비 시뮬레이션 플랫폼")
        self.resize(1520, 920)
        self.setMinimumSize(1220, 760)

        icon_path = Path(resource_path("assets/app.ico"))
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._nav_buttons: List[QPushButton] = []
        self._module_index_by_id: Dict[str, int] = {}
        self._module_widgets_by_id: Dict[str, BaseLabModule] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setStyleSheet("background-color: #F8F8F6;")
        root_hbox = QHBoxLayout(central)
        root_hbox.setContentsMargins(0, 0, 0, 0)
        root_hbox.setSpacing(0)

        # ---------------- LEFT SIDEBAR (WARM SAGE SAAS THEME) ----------------
        sidebar = QFrame()
        sidebar.setFixedWidth(252)
        sidebar.setStyleSheet(
            """
            QFrame#SidebarFrame {
                background-color: #F2F3EE;
                border-right: 1px solid #E2E6E0;
            }
            """
        )
        sidebar.setObjectName("SidebarFrame")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(14, 18, 14, 14)
        sb_layout.setSpacing(5)

        # Brand Header
        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)

        logo_path = Path(resource_path("assets/app_logo.png"))
        if not logo_path.exists():
            logo_path = Path(resource_path("assets/app.ico"))
        if logo_path.exists():
            logo_lbl = QLabel()
            logo_lbl.setStyleSheet("border: none; background: transparent;")
            pm = QPixmap(str(logo_path)).scaled(
                38,
                38,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl.setPixmap(pm)
            brand_row.addWidget(logo_lbl)
        else:
            badge_icon = QLabel("G")
            badge_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge_icon.setFixedSize(36, 36)
            badge_icon.setStyleSheet(
                "background-color: #1E3A2F; color: #FFFFFF; font-size: 16px; font-weight: 900; border-radius: 10px;"
            )
            brand_row.addWidget(badge_icon)

        brand_text_col = QVBoxLayout()
        brand_text_col.setSpacing(1)
        brand_title = QLabel("GNSS SimLab")
        brand_title.setStyleSheet("font-size: 16px; font-weight: 900; color: #111827; border: none;")
        brand_sub = QLabel("MY NAV LAB, MY PACE")
        brand_sub.setStyleSheet("font-size: 10px; font-weight: 700; color: #6B7280; letter-spacing: 0.6px; border: none;")
        brand_text_col.addWidget(brand_title)
        brand_text_col.addWidget(brand_sub)
        brand_row.addLayout(brand_text_col, stretch=1)
        sb_layout.addLayout(brand_row)

        sb_layout.addSpacing(10)

        sec_lbl = QLabel("나의 학습 워크스페이스")
        sec_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #9CA3AF; padding-left: 6px; border: none;")
        sb_layout.addWidget(sec_lbl)

        # Central Workspace Stack
        self.stack = QStackedWidget()
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        # Dynamically register modules
        module_classes = ModuleRegistry.get_registered_modules()
        for idx, mod_cls in enumerate(module_classes):
            mod_widget = mod_cls(self)
            self.stack.addWidget(mod_widget)
            self._module_index_by_id[mod_cls.module_id] = idx
            self._module_widgets_by_id[mod_cls.module_id] = mod_widget

            short_label = SIDEBAR_SHORT_NAMES.get(mod_cls.module_id, mod_cls.module_title)
            btn = self._create_sidebar_button(text=short_label, index=idx)
            sb_layout.addWidget(btn)

        sb_layout.addSpacing(8)

        tool_lbl = QLabel("학습 관리 및 사전")
        tool_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #9CA3AF; padding-left: 6px; border: none;")
        sb_layout.addWidget(tool_lbl)

        # Study Planner Tab Button
        self.planner_widget = StudyPlannerTabWidget(
            on_launch_module=self.navigate_to_module_with_level,
            parent=self,
        )
        planner_idx = self.stack.addWidget(self.planner_widget)
        planner_btn = self._create_sidebar_button(text="입사 전 맞춤 학습 계획표", index=planner_idx, is_accent=True)
        sb_layout.addWidget(planner_btn)

        # Glossary Tab Button
        glossary_widget = GlossaryTabWidget(self)
        glossary_idx = self.stack.addWidget(glossary_widget)
        glossary_btn = self._create_sidebar_button(text="핵심 용어 사전 (50)", index=glossary_idx)
        sb_layout.addWidget(glossary_btn)

        sb_layout.addStretch(1)

        # Bottom White Goal Card (Inspired directly by Image 2's bottom-left card)
        goal_card = QFrame()
        goal_card.setStyleSheet(
            """
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 12px;
            }
            """
        )
        gc_vbox = QVBoxLayout(goal_card)
        gc_vbox.setContentsMargins(14, 12, 14, 12)
        gc_vbox.setSpacing(5)

        gc_kicker = QLabel("지금 목표하는 곳")
        gc_kicker.setStyleSheet("font-size: 11px; font-weight: 800; color: #166534; border: none;")
        gc_vbox.addWidget(gc_kicker)

        gc_title = QLabel("단암시스템즈 항법연구소")
        gc_title.setStyleSheet("font-size: 13.5px; font-weight: 900; color: #111827; border: none;")
        gc_vbox.addWidget(gc_title)

        gc_desc = QLabel("위성항법 · 항재밍 · EGI 통합항법\nSW 역량을 차근차근 쌓아가는 중.")
        gc_desc.setWordWrap(True)
        gc_desc.setStyleSheet("font-size: 11px; color: #6B7280; line-height: 1.4; border: none;")
        gc_vbox.addWidget(gc_desc)

        tags_row = QHBoxLayout()
        tags_row.setSpacing(4)
        for tag_txt in ["GNSS/INS", "CRPA", "C/C++"]:
            t_lbl = QLabel(tag_txt)
            t_lbl.setStyleSheet(
                "background-color: #F2F3EE; color: #4B5563; font-size: 10px; font-weight: 700; "
                "padding: 3px 6px; border-radius: 4px; border: none;"
            )
            tags_row.addWidget(t_lbl)
        tags_row.addStretch(1)
        gc_vbox.addLayout(tags_row)

        sb_layout.addWidget(goal_card)

        root_hbox.addWidget(sidebar)
        root_hbox.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        # Clean Minimal Status Bar
        status = QStatusBar()
        status.setStyleSheet(
            """
            QStatusBar {
                background-color: #FFFFFF;
                color: #6B7280;
                border-top: 1px solid #E2E6E0;
                font-size: 11.5px;
                padding-left: 14px;
            }
            """
        )
        status.showMessage(
            "Ready  |  오프라인 수학·물리 엔진 가동 중  ·  상단 탭으로 [2D 그래프 / 3D 입체 뷰 / Lv.1 입문 / Lv.2 심화 / C99 코드]를 깔끔하게 전환하세요."
        )
        self.setStatusBar(status)

        if self._nav_buttons:
            self._nav_buttons[0].setChecked(True)
            self.stack.setCurrentIndex(0)

    def _create_sidebar_button(self, text: str, index: int, is_accent: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        base_color = "#166534" if is_accent else "#374151"
        btn.setStyleSheet(
            f"""
            QPushButton {{
                text-align: left;
                padding: 9px 12px;
                border-radius: 8px;
                border: none;
                background-color: transparent;
                color: {base_color};
                font-size: 12.5px;
                font-weight: {'700' if is_accent else '600'};
            }}
            QPushButton:hover {{
                background-color: #E7ECE8;
                color: #111827;
            }}
            QPushButton:checked {{
                background-color: #DDE7E0;
                color: #1E3A2F;
                font-weight: 800;
            }}
            """
        )
        btn.clicked.connect(lambda _=False, idx=index: self.stack.setCurrentIndex(idx))
        self.btn_group.addButton(btn, index)
        self._nav_buttons.append(btn)
        return btn

    def navigate_to_module_with_level(self, module_id: str, target_level: str, preset_idx: int = 0) -> None:
        """
        Switch from the Study Planner directly to `module_id`, apply `target_level`
        ('lv1', 'both', 'lv2'), and select `preset_idx`.
        """
        if module_id not in self._module_index_by_id:
            return
        idx = self._module_index_by_id[module_id]
        mod_widget = self._module_widgets_by_id[module_id]

        if 0 <= idx < len(self._nav_buttons):
            self._nav_buttons[idx].setChecked(True)
        self.stack.setCurrentIndex(idx)

        mod_widget.set_level_mode(target_level)

        if hasattr(mod_widget, "preset_combo") and 0 <= preset_idx < mod_widget.preset_combo.count():
            mod_widget.preset_combo.setCurrentIndex(preset_idx)

        if self.statusBar() is not None:
            self.statusBar().showMessage(
                f"맞춤 학습 계획 연동 완료: {mod_widget.module_title} ({target_level.upper()})",
                8000,
            )
