"""
Interactive Study Planner & Step-by-Step Curriculum Manager Tab (`ui/study_planner_tab.py`).
Allows the user to:
1. Enter any number of study days (e.g., 7, 10, 14, 21, 30 days) starting from today at 2 hours/evening
   and automatically generate a tailored step-by-step GNSS study plan.
2. Freely edit every day's title, date, target module, target level (Lv.1 / Lv.1+2 / Lv.2), 1st/2nd/3rd period plan,
   self-check question, and personal study notes, as well as add/delete/reorder rows.
3. Track daily completion with a live progress bar and persistent JSON storage.
4. Click [학습 바로 시작하기] to jump directly to the target module, preset, and Lv.1/Lv.2 tab.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.study_planner import (
    StudyDayItem,
    StudyPlanDocument,
    export_study_plan_markdown,
    generate_study_plan,
    get_user_study_plan_path,
    load_study_plan,
    save_study_plan,
)

MODULE_DISPLAY_MAP: Dict[str, str] = {
    "m01_coords_time": "M01. 좌표계 변환 · float32/64 · GPST/1PPS",
    "m02_signal_acq_track": "M02. PRN 골드코드 · 2D FFT 획득 · DLL/PLL",
    "m03_pvt_dop": "M03. Skyplot · DOP · 가우스-뉴턴 WLS · SBAS",
    "m04_antijam_crpa": "M04. 항재밍 CRPA 배열 · Power-Min / MVDR",
    "m05_gnss_ins_ekf": "M05. GNSS-INS 통합항법 · Error-State EKF",
    "m06_egi_raim_spoof": "M06. [단암 특화] EGI · RAIM/항기만 · STAP",
}

LEVEL_DISPLAY_MAP: Dict[str, str] = {
    "lv1": "Lv.1 입문 (쉬운 비유 · 직관)",
    "both": "Lv.1+2 단계별 레벨업",
    "lv2": "Lv.2 심화 (수식 · 실전 C99)",
}


class StudyPlannerTabWidget(QWidget):
    """Interactive N-day GNSS study schedule generator, editor, and step-by-step launcher."""

    def __init__(
        self,
        on_launch_module: Optional[Callable[[str, str, int], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.on_launch_module = on_launch_module
        load_err: Optional[str] = None
        try:
            self._plan: StudyPlanDocument = load_study_plan(raise_on_error=True)
        except Exception as exc:
            load_err = str(exc)
            self._plan = generate_study_plan(total_days=14, start_date=date.today(), daily_hours=2.0, focus_track="balanced")

        self._current_row: int = 0
        self._updating_ui: bool = False
        self._build_ui()
        self._populate_table_from_plan()

        if load_err:
            QMessageBox.warning(
                self,
                "학습 계획표 로드 오류",
                f"{load_err}\n\n기본 14일 학습 계획표로 초기화했습니다.",
            )

    def _build_ui(self) -> None:
        self.setStyleSheet("background-color: #F8F8F6;")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 14, 20, 14)
        root_layout.setSpacing(12)

        # ---------------- TOP GENERATOR & PROGRESS BAR BANNER ----------------
        top_card = QFrame()
        top_card.setStyleSheet(
            """
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 12px;
            }
            """
        )
        top_vbox = QVBoxLayout(top_card)
        top_vbox.setContentsMargins(16, 12, 16, 12)
        top_vbox.setSpacing(10)

        # Row 1: Title + Quick Day Presets
        row1 = QHBoxLayout()
        title_lbl = QLabel("단암시스템즈 항법연구소 입사 대비 맞춤 학습 계획표")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 900; color: #111827; border: none;")
        row1.addWidget(title_lbl)

        row1.addStretch(1)

        quick_lbl = QLabel("빠른 일수 선택:")
        quick_lbl.setStyleSheet("font-size: 12.5px; font-weight: 700; color: #5C6B62; border: none;")
        row1.addWidget(quick_lbl)

        for q_days, q_text in [
            (5, "5일 초단기"),
            (7, "7일 핵심"),
            (10, "10일 압축"),
            (14, "14일 표준(추천)"),
            (21, "21일 여유"),
            (30, "30일 마스터"),
        ]:
            q_btn = QPushButton(q_text)
            q_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            is_rec = q_days == 14
            q_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {'#E3EAE5' if is_rec else '#FAFBF9'};
                    color: {'#1E3A2F' if is_rec else '#374151'};
                    font-weight: {'800' if is_rec else '700'};
                    font-size: 12px;
                    padding: 5px 9px;
                    border: 1px solid {'#1E3A2F' if is_rec else '#D5DDD7'};
                    border-radius: 7px;
                }}
                QPushButton:hover {{
                    background-color: #E3EAE5;
                    color: #1E3A2F;
                }}
                """
            )
            q_btn.clicked.connect(lambda _=False, d=q_days: self._apply_quick_days(d))
            row1.addWidget(q_btn)

        top_vbox.addLayout(row1)

        # Row 2: Custom Generator Controls
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        lbl_start = QLabel("시작일:")
        lbl_start.setStyleSheet("font-size: 12.5px; font-weight: 700; color: #374151; border: none;")
        row2.addWidget(lbl_start)

        self.date_edit_start = QDateEdit()
        self.date_edit_start.setCalendarPopup(True)
        self.date_edit_start.setDisplayFormat("yyyy-MM-dd")
        self.date_edit_start.setMinimumWidth(118)
        try:
            dt_obj = datetime.strptime(self._plan.start_date_str, "%Y-%m-%d").date()
            self.date_edit_start.setDate(QDate(dt_obj.year, dt_obj.month, dt_obj.day))
        except ValueError:
            self.date_edit_start.setDate(QDate.currentDate())
            QMessageBox.warning(
                self,
                "날짜 형식 경고",
                f"저장된 시작 날짜('{self._plan.start_date_str}')가 YYYY-MM-DD 형식이 아니어서 오늘 날짜로 표시합니다.",
            )
        self.date_edit_start.setStyleSheet("padding: 5px 8px; border: 1px solid #D5DDD7; border-radius: 7px; font-weight: 700; background: #FAFBF9;")
        row2.addWidget(self.date_edit_start)

        lbl_days = QLabel("일수:")
        lbl_days.setStyleSheet("font-size: 12.5px; font-weight: 700; color: #374151; border: none;")
        row2.addWidget(lbl_days)

        self.spin_total_days = QSpinBox()
        self.spin_total_days.setRange(1, 90)
        self.spin_total_days.setValue(max(1, len(self._plan.days)))
        self.spin_total_days.setSuffix(" 일")
        self.spin_total_days.setMinimumWidth(88)
        self.spin_total_days.setStyleSheet("padding: 5px 8px; border: 1.5px solid #1E3A2F; border-radius: 7px; font-weight: 800; color: #1E3A2F; background: #FFFFFF;")
        row2.addWidget(self.spin_total_days)

        lbl_hours = QLabel("시간:")
        lbl_hours.setStyleSheet("font-size: 12.5px; font-weight: 700; color: #374151; border: none;")
        row2.addWidget(lbl_hours)

        self.spin_daily_hours = QDoubleSpinBox()
        self.spin_daily_hours.setRange(0.5, 10.0)
        self.spin_daily_hours.setDecimals(1)
        self.spin_daily_hours.setSingleStep(0.5)
        self.spin_daily_hours.setValue(self._plan.daily_hours)
        self.spin_daily_hours.setSuffix(" 시간")
        self.spin_daily_hours.setMinimumWidth(106)
        self.spin_daily_hours.setStyleSheet("padding: 5px 8px; border: 1px solid #D5DDD7; border-radius: 7px; font-weight: 700; background: #FAFBF9;")
        row2.addWidget(self.spin_daily_hours)

        self.combo_track = QComboBox()
        self.combo_track.addItem("표준 밸런스 (Lv.1→2 전 모듈)", userData="balanced")
        self.combo_track.addItem("기본 위성항법 집중 (M01~M03)", userData="basic_focus")
        self.combo_track.addItem("연구소 실무 심화 (Lv.2·C99)", userData="rnd_deep")
        t_idx = self.combo_track.findData(self._plan.focus_track)
        if t_idx >= 0:
            self.combo_track.setCurrentIndex(t_idx)
        self.combo_track.setMinimumWidth(210)
        self.combo_track.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_track.setMinimumContentsLength(16)
        self.combo_track.setStyleSheet("padding: 5px 10px; border: 1px solid #D5DDD7; border-radius: 7px; font-weight: 600; background: #FAFBF9;")
        row2.addWidget(self.combo_track)

        btn_gen = QPushButton("↺ 계획표 재편성")
        btn_gen.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_gen.setStyleSheet(
            """
            QPushButton {
                background-color: #1E3A2F;
                color: #FFFFFF;
                font-weight: 800;
                font-size: 12.5px;
                padding: 7px 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #142820;
            }
            """
        )
        btn_gen.clicked.connect(self.regenerate_plan_from_inputs)
        row2.addWidget(btn_gen)

        row2.addStretch(1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(175)
        self.progress_bar.setMaximumHeight(24)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #D5DDD7;
                border-radius: 7px;
                background-color: #FAFBF9;
                text-align: center;
                font-weight: 800;
                color: #111827;
            }
            QProgressBar::chunk {
                background-color: #16A34A;
                border-radius: 6px;
            }
            """
        )
        row2.addWidget(self.progress_bar)

        top_vbox.addLayout(row2)
        root_layout.addWidget(top_card)

        # ---------------- MAIN SPLITTER: LEFT TABLE / RIGHT STEP-BY-STEP EDITOR ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # ---------------- LEFT PANEL: SCHEDULE TABLE & ROW CONTROLS ----------------
        left_frame = QFrame()
        left_frame.setStyleSheet(
            """
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 12px;
            }
            """
        )
        left_vbox = QVBoxLayout(left_frame)
        left_vbox.setContentsMargins(14, 14, 14, 14)
        left_vbox.setSpacing(10)

        tbl_header_lbl = QLabel("일차별 학습 스케줄 표 (체크박스로 완료 표시 · 행 선택 시 우측에서 수정 및 모듈 실행)")
        tbl_header_lbl.setStyleSheet("font-size: 13px; font-weight: 800; color: #1E3A2F; border: none;")
        left_vbox.addWidget(tbl_header_lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["완료", "일차", "날짜", "학습 주제 (우측에서 수정)", "연동 모듈", "추천 레벨", "시간"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(
            """
            QTableWidget {
                border: 1px solid #E2E6E0;
                border-radius: 8px;
                font-size: 12.5px;
                gridline-color: #F1F3EE;
                background-color: #FFFFFF;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #E3EAE5;
                color: #1E3A2F;
                font-weight: 800;
            }
            QHeaderView::section {
                background-color: #FAFBF9;
                color: #374151;
                font-weight: 800;
                font-size: 12px;
                padding: 8px 6px;
                border: none;
                border-bottom: 1px solid #E2E6E0;
            }
            """
        )
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        self.table.currentCellChanged.connect(self._on_table_row_selected)
        self.table.itemChanged.connect(self._on_table_item_check_changed)
        left_vbox.addWidget(self.table, stretch=1)

        # Row Management Buttons Bar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(6)

        btn_style = """
            QPushButton {
                background-color: #FAFBF9;
                color: #374151;
                font-weight: 700;
                font-size: 12px;
                padding: 6px 9px;
                border-radius: 7px;
                border: 1px solid #D5DDD7;
            }
            QPushButton:hover {
                background-color: #E3EAE5;
                color: #1E3A2F;
            }
        """

        btn_add = QPushButton("+ 일정 추가")
        btn_add.setStyleSheet(btn_style)
        btn_add.clicked.connect(self.add_custom_day_row)
        btn_bar.addWidget(btn_add)

        btn_del = QPushButton("- 행 삭제")
        btn_del.setStyleSheet(btn_style)
        btn_del.clicked.connect(self.delete_selected_day_row)
        btn_bar.addWidget(btn_del)

        btn_up = QPushButton("▲ 위로")
        btn_up.setStyleSheet(btn_style)
        btn_up.clicked.connect(lambda: self.move_selected_row(-1))
        btn_bar.addWidget(btn_up)

        btn_down = QPushButton("▼ 아래로")
        btn_down.setStyleSheet(btn_style)
        btn_down.clicked.connect(lambda: self.move_selected_row(1))
        btn_bar.addWidget(btn_down)

        btn_bar.addStretch(1)

        btn_export = QPushButton("MD 내보내기")
        btn_export.setStyleSheet(btn_style)
        btn_export.clicked.connect(self.export_markdown_dialog)
        btn_bar.addWidget(btn_export)

        btn_save = QPushButton("계획표 저장")
        btn_save.setStyleSheet(
            """
            QPushButton {
                background-color: #1E3A2F;
                color: #FFFFFF;
                font-weight: 800;
                font-size: 12px;
                padding: 7px 12px;
                border-radius: 7px;
                border: none;
            }
            QPushButton:hover {
                background-color: #142820;
            }
            """
        )
        btn_save.clicked.connect(self.save_current_plan)
        btn_bar.addWidget(btn_save)

        left_vbox.addLayout(btn_bar)
        splitter.addWidget(left_frame)

        # ---------------- RIGHT PANEL: STEP-BY-STEP STUDY COACH & EDITOR ----------------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(450)
        right_scroll.setStyleSheet(
            """
            QScrollArea {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 12px;
            }
            """
        )
        right_container = QWidget()
        right_container.setStyleSheet("background-color: #FFFFFF;")
        right_vbox = QVBoxLayout(right_container)
        right_vbox.setContentsMargins(16, 14, 16, 14)
        right_vbox.setSpacing(12)

        # 1. Big Action Button: Jump directly to target module & level
        self.btn_launch = QPushButton("▶ Day 01 학습 시작  ->  [M01] 모듈 열기")
        self.btn_launch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_launch.setStyleSheet(
            """
            QPushButton {
                background-color: #166534;
                color: #FFFFFF;
                font-weight: 900;
                font-size: 13.5px;
                padding: 13px 18px;
                border-radius: 9px;
                border: none;
            }
            QPushButton:hover {
                background-color: #14532D;
            }
            """
        )
        self.btn_launch.clicked.connect(self.launch_selected_day_module)
        right_vbox.addWidget(self.btn_launch)

        # 2. Day Metadata Editor Group
        g_meta = QGroupBox("선택한 일차 계획 수정 (수정 즉시 자동 저장)")
        g_meta.setStyleSheet(
            """
            QGroupBox {
                font-weight: 800;
                font-size: 13px;
                color: #1E3A2F;
                border: 1px solid #E2E6E0;
                border-radius: 9px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #FAFBF9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background-color: #FFFFFF;
            }
            """
        )
        f_meta = QFormLayout(g_meta)
        f_meta.setSpacing(8)

        meta_top_row = QHBoxLayout()
        self.chk_day_done = QCheckBox("이 일차 학습 완료됨")
        self.chk_day_done.setStyleSheet("font-weight: 800; color: #166534; font-size: 13px;")
        self.chk_day_done.toggled.connect(self._on_detail_field_edited)
        meta_top_row.addWidget(self.chk_day_done)

        meta_top_row.addStretch(1)
        meta_top_row.addWidget(QLabel("날짜:"))
        self.edit_day_date = QLineEdit()
        self.edit_day_date.setMaximumWidth(115)
        self.edit_day_date.textChanged.connect(self._on_detail_field_edited)
        meta_top_row.addWidget(self.edit_day_date)

        meta_top_row.addWidget(QLabel("시간:"))
        self.spin_day_hours = QDoubleSpinBox()
        self.spin_day_hours.setRange(0.5, 12.0)
        self.spin_day_hours.setSingleStep(0.5)
        self.spin_day_hours.setSuffix(" h")
        self.spin_day_hours.setMinimumWidth(86)
        self.spin_day_hours.valueChanged.connect(self._on_detail_field_edited)
        meta_top_row.addWidget(self.spin_day_hours)
        f_meta.addRow(meta_top_row)

        self.edit_day_title = QLineEdit()
        self.edit_day_title.setStyleSheet("padding: 6px 8px; font-weight: 700; font-size: 13px; background: #FFFFFF; border: 1px solid #D5DDD7; border-radius: 6px;")
        self.edit_day_title.textChanged.connect(self._on_detail_field_edited)
        f_meta.addRow("학습 주제:", self.edit_day_title)

        mod_lvl_row = QHBoxLayout()
        self.combo_day_module = QComboBox()
        for m_id, m_name in MODULE_DISPLAY_MAP.items():
            self.combo_day_module.addItem(m_name, userData=m_id)
        self.combo_day_module.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_day_module.setMinimumContentsLength(14)
        self.combo_day_module.currentIndexChanged.connect(self._on_detail_field_edited)
        mod_lvl_row.addWidget(self.combo_day_module, stretch=3)

        self.combo_day_level = QComboBox()
        for l_id, l_name in LEVEL_DISPLAY_MAP.items():
            self.combo_day_level.addItem(l_name, userData=l_id)
        self.combo_day_level.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_day_level.setMinimumContentsLength(10)
        self.combo_day_level.currentIndexChanged.connect(self._on_detail_field_edited)
        mod_lvl_row.addWidget(self.combo_day_level, stretch=2)
        f_meta.addRow("모듈 / 레벨:", mod_lvl_row)

        right_vbox.addWidget(g_meta)

        # 3. 3-Period Evening Study Routine Editor (1교시 / 2교시 / 3교시)
        g_steps = QGroupBox("저녁 2시간(120분) 3교시 단계별 학습 액션 플랜")
        g_steps.setStyleSheet(
            """
            QGroupBox {
                font-weight: 800;
                font-size: 13px;
                color: #1E3A2F;
                border: 1px solid #D8E2DB;
                border-radius: 9px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #F4F7F4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background-color: #FFFFFF;
            }
            """
        )
        v_steps = QVBoxLayout(g_steps)
        v_steps.setSpacing(6)

        text_box_style = """
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #D5DDD7;
                border-radius: 7px;
                padding: 6px;
                font-size: 13px;
                color: #1F2937;
            }
        """

        lbl_s1 = QLabel("1교시 (첫 40분 — 개념 · 쉬운 비유 이해):")
        lbl_s1.setWordWrap(True)
        lbl_s1.setStyleSheet("font-weight: 800; color: #166534; font-size: 12.5px; border: none;")
        v_steps.addWidget(lbl_s1)
        self.edit_step1 = QTextEdit()
        self.edit_step1.setMaximumHeight(68)
        self.edit_step1.setStyleSheet(text_box_style)
        self.edit_step1.textChanged.connect(self._on_detail_field_edited)
        v_steps.addWidget(self.edit_step1)

        lbl_s2 = QLabel("2교시 (중간 50분 — 2D/3D 시뮬레이터 슬라이더 실습):")
        lbl_s2.setWordWrap(True)
        lbl_s2.setStyleSheet("font-weight: 800; color: #1D4ED8; font-size: 12.5px; border: none;")
        v_steps.addWidget(lbl_s2)
        self.edit_step2 = QTextEdit()
        self.edit_step2.setMaximumHeight(68)
        self.edit_step2.setStyleSheet(text_box_style)
        self.edit_step2.textChanged.connect(self._on_detail_field_edited)
        v_steps.addWidget(self.edit_step2)

        lbl_s3 = QLabel("3교시 (마지막 30분 — 핵심 용어 · 임베디드 C99 정리):")
        lbl_s3.setWordWrap(True)
        lbl_s3.setStyleSheet("font-weight: 800; color: #1E3A2F; font-size: 12.5px; border: none;")
        v_steps.addWidget(lbl_s3)
        self.edit_step3 = QTextEdit()
        self.edit_step3.setMaximumHeight(68)
        self.edit_step3.setStyleSheet(text_box_style)
        self.edit_step3.textChanged.connect(self._on_detail_field_edited)
        v_steps.addWidget(self.edit_step3)

        lbl_q = QLabel("오늘의 합격 셀프 체크 질문 (이 질문에 답할 수 있으면 달성):")
        lbl_q.setWordWrap(True)
        lbl_q.setStyleSheet("font-weight: 800; color: #B45309; font-size: 12.5px; border: none;")
        v_steps.addWidget(lbl_q)
        self.edit_check_q = QLineEdit()
        self.edit_check_q.setStyleSheet(
            "padding: 7px; font-weight: 700; font-size: 12.5px; color: #92400E; background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 6px;"
        )
        self.edit_check_q.textChanged.connect(self._on_detail_field_edited)
        v_steps.addWidget(self.edit_check_q)

        right_vbox.addWidget(g_steps)

        # 4. Personal Study Memo & Questions for Senior Engineer
        g_notes = QGroupBox("나의 공부 요약 메모 · 입사 후 질문 노트")
        g_notes.setStyleSheet(
            """
            QGroupBox {
                font-weight: 800;
                font-size: 13px;
                color: #1E3A2F;
                border: 1px solid #D5DDD7;
                border-radius: 9px;
                margin-top: 6px;
                padding-top: 12px;
                background-color: #FAFBF9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background-color: #FFFFFF;
            }
            """
        )
        v_notes = QVBoxLayout(g_notes)
        self.edit_user_notes = QTextEdit()
        self.edit_user_notes.setPlaceholderText("오늘 배운 핵심 내용을 내 말로 한 줄 요약하거나, 헷갈리는 점/메모를 자유롭게 적어두세요 (자동 저장됨)...")
        self.edit_user_notes.setMinimumHeight(75)
        self.edit_user_notes.setStyleSheet(text_box_style)
        self.edit_user_notes.textChanged.connect(self._on_detail_field_edited)
        v_notes.addWidget(self.edit_user_notes)
        right_vbox.addWidget(g_notes)

        self.lbl_save_status = QLabel(f"저장 경로: {get_user_study_plan_path()}")
        self.lbl_save_status.setWordWrap(True)
        self.lbl_save_status.setStyleSheet("font-size: 11.5px; color: #5C6B62; border: none;")
        right_vbox.addWidget(self.lbl_save_status)

        right_vbox.addStretch(1)
        right_scroll.setWidget(right_container)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)
        root_layout.addWidget(splitter, stretch=1)

    # ==========================================================================
    # Schedule Generation, Table Population & Sync Logic
    # ==========================================================================
    def _apply_quick_days(self, num_days: int) -> None:
        self.spin_total_days.setValue(num_days)
        self.regenerate_plan_from_inputs()

    def _save_plan_safe(self) -> Optional[Path]:
        try:
            return save_study_plan(self._plan)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "학습 계획표 저장 오류",
                f"학습 계획표 파일 저장 중 오류가 발생했습니다.\n\n상세 내용: {exc}",
            )
            return None

    def regenerate_plan_from_inputs(self) -> None:
        """Generate a fresh study plan for the user-entered day count, start date, daily hours, and track."""
        qdate = self.date_edit_start.date()
        start_dt = date(qdate.year(), qdate.month(), qdate.day())
        num_days = self.spin_total_days.value()
        hours = self.spin_daily_hours.value()
        track = str(self.combo_track.currentData() or "balanced")

        self._plan = generate_study_plan(
            total_days=num_days,
            start_date=start_dt,
            daily_hours=hours,
            focus_track=track,
        )
        saved_path = self._save_plan_safe()
        self._populate_table_from_plan()
        if saved_path is not None:
            self.lbl_save_status.setText(f"{num_days}일 맞춤 학습 계획표(하루 {hours:.1f}시간)가 생성 및 저장되었습니다.")

    def _populate_table_from_plan(self) -> None:
        self._updating_ui = True
        days = self._plan.days
        self.table.setRowCount(len(days))

        for r_idx, item in enumerate(days):
            self._set_table_row_items(r_idx, item)

        self._update_progress_bar()
        self._updating_ui = False

        if days:
            sel = min(max(self._current_row, 0), len(days) - 1)
            self.table.selectRow(sel)
            self._load_day_into_detail_panel(sel)

    def _set_table_row_items(self, row: int, item: StudyDayItem) -> None:
        # Column 0: Completed checkbox
        chk_item = QTableWidgetItem()
        chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        chk_item.setCheckState(Qt.CheckState.Checked if item.completed else Qt.CheckState.Unchecked)
        self.table.setItem(row, 0, chk_item)

        # Column 1: Day Number
        d_item = QTableWidgetItem(f"Day {item.day_number:02d}")
        d_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        d_item.setFont(QFont("Malgun Gothic", 9, QFont.Weight.Bold))
        self.table.setItem(row, 1, d_item)

        # Column 2: Date
        dt_item = QTableWidgetItem(item.date_str)
        dt_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 2, dt_item)

        # Column 3: Title
        t_item = QTableWidgetItem(item.title)
        if item.completed:
            t_item.setForeground(QColor("#166534"))
        self.table.setItem(row, 3, t_item)

        # Column 4: Module badge
        mod_short = item.target_module_id.split("_")[0].upper()
        m_item = QTableWidgetItem(mod_short)
        m_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 4, m_item)

        # Column 5: Target Level
        lvl_short = {"lv1": "Lv.1 입문", "both": "Lv.1+2", "lv2": "Lv.2 심화"}.get(item.target_level, item.target_level)
        l_item = QTableWidgetItem(lvl_short)
        l_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 5, l_item)

        # Column 6: Hours
        h_item = QTableWidgetItem(f"{item.hours:.1f}h")
        h_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 6, h_item)

    def _on_table_item_check_changed(self, tbl_item: QTableWidgetItem) -> None:
        if self._updating_ui or tbl_item.column() != 0:
            return
        row = tbl_item.row()
        if 0 <= row < len(self._plan.days):
            is_checked = tbl_item.checkState() == Qt.CheckState.Checked
            self._plan.days[row].completed = is_checked
            if row == self._current_row:
                self._updating_ui = True
                self.chk_day_done.setChecked(is_checked)
                self._updating_ui = False
            self._update_progress_bar()
            self._save_plan_safe()

    def _on_table_row_selected(self, current_row: int, _current_col: int, _prev_row: int, _prev_col: int) -> None:
        if self._updating_ui or current_row < 0 or current_row >= len(self._plan.days):
            return
        self._current_row = current_row
        self._load_day_into_detail_panel(current_row)

    def _load_day_into_detail_panel(self, row: int) -> None:
        if row < 0 or row >= len(self._plan.days):
            return
        self._updating_ui = True
        d = self._plan.days[row]

        self.chk_day_done.setChecked(d.completed)
        self.edit_day_date.setText(d.date_str)
        self.spin_day_hours.setValue(d.hours)
        self.edit_day_title.setText(d.title)
        self.edit_day_title.setCursorPosition(0)

        m_idx = self.combo_day_module.findData(d.target_module_id)
        if m_idx >= 0:
            self.combo_day_module.setCurrentIndex(m_idx)

        l_idx = self.combo_day_level.findData(d.target_level)
        if l_idx >= 0:
            self.combo_day_level.setCurrentIndex(l_idx)

        self.edit_step1.setPlainText(d.step1_concept)
        self.edit_step2.setPlainText(d.step2_practice)
        self.edit_step3.setPlainText(d.step3_review_c)
        self.edit_check_q.setText(d.self_check_question)
        self.edit_check_q.setCursorPosition(0)
        self.edit_user_notes.setPlainText(d.user_notes)

        mod_short = d.target_module_id.split("_")[0].upper()
        lvl_label = {"lv1": "Lv.1 쉬운 입문", "both": "Lv.1+2 레벨업", "lv2": "Lv.2 심화 · 수식"}.get(d.target_level, d.target_level)
        self.btn_launch.setText(f"▶ Day {d.day_number:02d} 학습 시작  ->  [{mod_short}] 모듈 ({lvl_label}) 열기")

        self._updating_ui = False

    def _on_detail_field_edited(self) -> None:
        if self._updating_ui or self._current_row < 0 or self._current_row >= len(self._plan.days):
            return
        row = self._current_row
        d = self._plan.days[row]

        d.completed = self.chk_day_done.isChecked()
        d.date_str = self.edit_day_date.text().strip()
        d.hours = float(self.spin_day_hours.value())
        d.title = self.edit_day_title.text().strip()
        d.target_module_id = str(self.combo_day_module.currentData() or "m01_coords_time")
        d.target_level = str(self.combo_day_level.currentData() or "both")
        d.step1_concept = self.edit_step1.toPlainText()
        d.step2_practice = self.edit_step2.toPlainText()
        d.step3_review_c = self.edit_step3.toPlainText()
        d.self_check_question = self.edit_check_q.text()
        d.user_notes = self.edit_user_notes.toPlainText()

        self._updating_ui = True
        self._set_table_row_items(row, d)
        mod_short = d.target_module_id.split("_")[0].upper()
        lvl_label = {"lv1": "Lv.1 쉬운 입문", "both": "Lv.1+2 레벨업", "lv2": "Lv.2 심화 · 수식"}.get(d.target_level, d.target_level)
        self.btn_launch.setText(f"▶ Day {d.day_number:02d} 학습 시작  ->  [{mod_short}] 모듈 ({lvl_label}) 열기")
        self._update_progress_bar()
        self._updating_ui = False

        saved_path = self._save_plan_safe()
        if saved_path is not None:
            self.lbl_save_status.setText(f"Day {d.day_number:02d} 수정 사항이 자동 저장되었습니다.")

    def _update_progress_bar(self) -> None:
        total = max(len(self._plan.days), 1)
        done = sum(1 for d in self._plan.days if d.completed)
        done_h = sum(d.hours for d in self._plan.days if d.completed)
        total_h = sum(d.hours for d in self._plan.days)
        pct = int(round((done / total) * 100))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"{done}/{len(self._plan.days)}일 완료 ({pct}%) · {done_h:.1f}h/{total_h:.0f}h")

    def add_custom_day_row(self) -> None:
        """Append a new customizable study day at the end of the schedule."""
        n_cur = len(self._plan.days)
        if n_cur > 0:
            try:
                last_dt = datetime.strptime(self._plan.days[-1].date_str, "%Y-%m-%d").date()
                next_dt_str = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            except ValueError:
                QMessageBox.warning(
                    self,
                    "날짜 형식 경고",
                    f"마지막 행의 날짜('{self._plan.days[-1].date_str}')가 YYYY-MM-DD 형식이 아니어서 오늘 날짜 기준으로 추가합니다.",
                )
                next_dt_str = date.today().strftime("%Y-%m-%d")
        else:
            next_dt_str = date.today().strftime("%Y-%m-%d")

        new_day = StudyDayItem(
            day_number=n_cur + 1,
            date_str=next_dt_str,
            phase_badge="사용자 추가 일정",
            title=f"커스텀 추가 학습 · 복습 (Day {n_cur + 1})",
            target_module_id="m01_coords_time",
            target_level="both",
            recommended_preset_idx=0,
            hours=self.spin_daily_hours.value(),
            step1_concept="[1교시 40분] 부족했던 모듈의 Lv.1 비유와 Lv.2 수식을 다시 복습합니다.",
            step2_practice="[2교시 50분] 시뮬레이터 프리셋과 슬라이더를 조작하며 2D/3D 그래프를 관찰합니다.",
            step3_review_c="[3교시 30분] 임베디드 C 포팅 코드와 핵심 용어를 노트에 정리합니다.",
            self_check_question="오늘 복습한 모듈의 핵심 원리를 내 말로 막힘없이 설명할 수 있는가?",
            user_notes="",
            completed=False,
        )
        self._plan.days.append(new_day)
        self._plan.total_days = len(self._plan.days)
        self._current_row = len(self._plan.days) - 1
        self.spin_total_days.setValue(self._plan.total_days)
        self._save_plan_safe()
        self._populate_table_from_plan()

    def delete_selected_day_row(self) -> None:
        """Delete the currently selected day row and renumber remaining days."""
        if len(self._plan.days) <= 1:
            QMessageBox.information(self, "삭제 불가", "최소 1일 이상의 학습 일정은 유지해야 합니다.")
            return
        if 0 <= self._current_row < len(self._plan.days):
            self._plan.days.pop(self._current_row)
            for idx, item in enumerate(self._plan.days):
                item.day_number = idx + 1
            self._plan.total_days = len(self._plan.days)
            self._current_row = min(self._current_row, len(self._plan.days) - 1)
            self.spin_total_days.setValue(self._plan.total_days)
            self._save_plan_safe()
            self._populate_table_from_plan()

    def move_selected_row(self, delta: int) -> None:
        """Move the currently selected day up (-1) or down (+1), keeping sequential dates/day numbers."""
        r = self._current_row
        target_r = r + delta
        if 0 <= r < len(self._plan.days) and 0 <= target_r < len(self._plan.days):
            # Swap content while preserving sequential day_number and date_str
            d1, d2 = self._plan.days[r], self._plan.days[target_r]
            saved_dates = (d1.date_str, d2.date_str)
            saved_nums = (d1.day_number, d2.day_number)

            self._plan.days[r], self._plan.days[target_r] = d2, d1
            self._plan.days[r].date_str, self._plan.days[target_r].date_str = saved_dates
            self._plan.days[r].day_number, self._plan.days[target_r].day_number = saved_nums

            self._current_row = target_r
            self._save_plan_safe()
            self._populate_table_from_plan()

    def save_current_plan(self) -> None:
        saved_path = self._save_plan_safe()
        if saved_path is not None:
            self.lbl_save_status.setText(f"전체 학습 계획표가 저장되었습니다: {saved_path}")

    def export_markdown_dialog(self) -> None:
        """Export the current study schedule to a Markdown (.md) file."""
        md_content = export_study_plan_markdown(self._plan)
        default_file = str(Path(get_user_study_plan_path()).parent / "단암시스템즈_입사전_위성항법_학습계획표.md")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "학습 계획표 마크다운(.md) 저장",
            default_file,
            "Markdown Files (*.md);;All Files (*)",
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                self.lbl_save_status.setText(f"마크다운 계획표 내보내기 완료: {file_path}")
                QMessageBox.information(self, "내보내기 완료", f"학습 계획표를 저장했습니다.\n\n{file_path}")
            except OSError as exc:
                QMessageBox.warning(self, "파일 내보내기 실패", f"마크다운 파일을 저장할 수 없습니다.\n\n{exc}")

    def launch_selected_day_module(self) -> None:
        """Switch the main window directly to the selected day's target module, difficulty level, and preset!"""
        if not (0 <= self._current_row < len(self._plan.days)):
            QMessageBox.warning(self, "선택 오류", "먼저 좌측 일정 표에서 학습할 일차를 선택해 주세요.")
            return
        d = self._plan.days[self._current_row]
        if self.on_launch_module is not None:
            self.on_launch_module(d.target_module_id, d.target_level, d.recommended_preset_idx)
