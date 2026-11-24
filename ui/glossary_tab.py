"""
Dedicated Glossary Explorer Tab supporting real-time keyword search and category filtering
across all 35+ GNSS / Signal Processing / Anti-Jamming / EKF / Embedded terms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui.glossary_dialog import GlossaryRepository


class GlossaryTabWidget(QWidget):
    """Full-screen searchable and category-filterable glossary reference tab."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._all_entries: List[Dict[str, Any]] = GlossaryRepository.load()
        self._filtered_entries: List[Dict[str, Any]] = []
        self._build_ui()
        self._apply_filter()
        err = GlossaryRepository.get_last_error()
        if err:
            QMessageBox.warning(self, "용어 사전 로드 오류", err)

    def _build_ui(self) -> None:
        self.setStyleSheet("background-color: #F8F8F6;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # Top banner & filter bar
        header_card = QFrame()
        header_card.setStyleSheet(
            """
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 10px;
            }
            """
        )
        h_layout = QHBoxLayout(header_card)
        h_layout.setContentsMargins(16, 12, 16, 12)
        h_layout.setSpacing(12)

        title_lbl = QLabel("위성항법 · 항재밍 · 통합항법 핵심 용어 사전")
        title_lbl.setStyleSheet("font-size: 16.5px; font-weight: 900; color: #111827; border: none;")
        h_layout.addWidget(title_lbl)

        h_layout.addStretch(1)

        cat_lbl = QLabel("분야 필터:")
        cat_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #4B5563; border: none;")
        h_layout.addWidget(cat_lbl)

        self.cat_combo = QComboBox()
        self.cat_combo.addItems(GlossaryRepository.categories())
        self.cat_combo.setMinimumWidth(170)
        self.cat_combo.setStyleSheet(
            """
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #D5DDD7;
                border-radius: 7px;
                background-color: #FAFBF9;
                font-size: 13px;
                color: #1F2937;
            }
            """
        )
        self.cat_combo.currentTextChanged.connect(lambda _: self._apply_filter())
        h_layout.addWidget(self.cat_combo)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("한글/영문 용어명, 약어(ECEF, DOP, CRPA, EKF 등), 키워드 검색...")
        self.search_input.setMinimumWidth(340)
        self.search_input.setStyleSheet(
            """
            QLineEdit {
                padding: 7px 12px;
                border: 1px solid #D5DDD7;
                border-radius: 7px;
                background-color: #FAFBF9;
                font-size: 13px;
                color: #1F2937;
            }
            QLineEdit:focus {
                border: 2px solid #1E3A2F;
                background-color: #FFFFFF;
            }
            """
        )
        self.search_input.textChanged.connect(lambda _: self._apply_filter())
        h_layout.addWidget(self.search_input)

        self.count_lbl = QLabel("0개 용어")
        self.count_lbl.setStyleSheet("font-size: 12.5px; font-weight: 800; color: #1E3A2F; border: none;")
        h_layout.addWidget(self.count_lbl)

        main_layout.addWidget(header_card)

        # Left list + Right detail splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        self.term_list = QListWidget()
        self.term_list.setStyleSheet(
            """
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 10px;
                padding: 6px;
                font-size: 13.5px;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-bottom: 1px solid #F1F3EE;
                border-radius: 7px;
                color: #1F2937;
            }
            QListWidget::item:hover {
                background-color: #F4F7F4;
            }
            QListWidget::item:selected {
                background-color: #E3EAE5;
                color: #1E3A2F;
                font-weight: 800;
            }
            """
        )
        self.term_list.currentRowChanged.connect(self._on_row_selected)
        splitter.addWidget(self.term_list)

        self.detail_browser = QTextBrowser()
        self.detail_browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-radius: 10px;
                padding: 20px;
                font-size: 14px;
            }
            """
        )
        splitter.addWidget(self.detail_browser)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter, stretch=1)

    def _apply_filter(self) -> None:
        selected_cat = self.cat_combo.currentText()
        query = self.search_input.text().strip().lower()

        self._filtered_entries = []
        for entry in self._all_entries:
            if selected_cat != "전체" and entry.get("category") != selected_cat:
                continue
            if query:
                searchable = " ".join(
                    [
                        entry.get("id", ""),
                        entry.get("term_ko", ""),
                        entry.get("term_en", ""),
                        entry.get("summary", ""),
                        entry.get("description", ""),
                    ]
                ).lower()
                if query not in searchable:
                    continue
            self._filtered_entries.append(entry)

        self.term_list.clear()
        for entry in self._filtered_entries:
            item_text = f"[{entry.get('category', '')}] {entry['term_ko']}\n   {entry['term_en']}"
            item = QListWidgetItem(item_text)
            self.term_list.addItem(item)

        self.count_lbl.setText(f"총 {len(self._filtered_entries)}개 용어")
        if self._filtered_entries:
            self.term_list.setCurrentRow(0)
        else:
            self.detail_browser.setHtml("<p style='color:#64748b;'>검색 조건에 맞는 용어가 없습니다.</p>")

    def _on_row_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._filtered_entries):
            return
        entry = self._filtered_entries[row]
        html = f"""
        <div style="line-height: 1.72; color: #1F2937;">
            <span style="background-color:#E3EAE5; color:#1E3A2F; padding:4px 10px; border-radius:8px; font-size:12px; font-weight:800;">
                {entry.get('category', '')}
            </span>
            <h2 style="margin-top:10px; margin-bottom:2px; color:#111827;">{entry['term_ko']}</h2>
            <p style="margin-top:0; color:#5C6B62; font-weight:600; font-size:14px;">{entry['term_en']} &nbsp;|&nbsp; <code style="color:#1E3A2F;">glossary://{entry['id']}</code></p>
            <hr style="border:none; border-top:1px solid #E2E6E0; margin:14px 0;">
            <div style="background-color:#F4F7F4; border-left:4px solid #16A34A; padding:12px 14px; border-radius:6px; margin-bottom:16px;">
                <b style="color:#166534;">[STEP 1. Lv.1 입문 한줄 직관]</b><br>
                <span style="color:#14532D; font-weight:700; font-size:14.5px;">{entry['summary']}</span>
            </div>
            <h3 style="color:#1E3A2F; margin-bottom:6px;">[STEP 2. Lv.1 → Lv.2 레벨업] 초보자 눈높이 원리 · 실무 배경</h3>
            <p style="margin-top:0; color:#374151; font-size:14.5px;">{entry['description']}</p>
            <h3 style="color:#1E3A2F; margin-bottom:6px;">[STEP 3. Lv.2 항법연구소 실무] 핵심 수식 및 수치 예시</h3>
            <pre style="background-color:#0F172A; color:#E2E8F0; padding:14px; border-radius:8px; font-family:'Consolas',monospace; font-size:13.5px;">{entry.get('formula_or_example', '-')}</pre>
            <p style="color:#5C6B62; font-size:12.5px; margin-top:16px;">
                <b>연관 학습 모듈:</b> {', '.join(entry.get('related_modules', []))}
            </p>
        </div>
        """
        self.detail_browser.setHtml(html)
