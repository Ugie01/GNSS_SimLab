"""
Glossary data repository and instant pop-up dialog triggered whenever the user clicks
a `glossary://<term_id>` hyperlink anywhere in the application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.resource_helper import resource_path


class GlossaryRepository:
    """Singleton-style loader for data/glossary.json."""

    _entries: List[Dict[str, Any]] = []
    _by_id: Dict[str, Dict[str, Any]] = {}
    _last_error: Optional[str] = None

    @classmethod
    def load(cls) -> List[Dict[str, Any]]:
        if cls._entries:
            return cls._entries
        glossary_file = Path(resource_path("data/glossary.json"))
        if not glossary_file.exists():
            cls._last_error = f"용어 사전 파일을 찾을 수 없습니다: {glossary_file}"
            return cls._entries
        try:
            with open(glossary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                cls._last_error = "glossary.json 최상위 형식이 list가 아닙니다."
                return cls._entries
            cls._entries = data
            cls._by_id = {
                str(item.get("id", "")).strip().lower(): item
                for item in cls._entries
                if isinstance(item, dict)
            }
            cls._last_error = None
        except (OSError, json.JSONDecodeError) as exc:
            cls._entries = []
            cls._by_id = {}
            cls._last_error = f"용어 사전 로드 실패: {exc}"
        return cls._entries

    @classmethod
    def get_last_error(cls) -> Optional[str]:
        return cls._last_error

    @classmethod
    def get_term(cls, term_id: str) -> Optional[Dict[str, Any]]:
        if not cls._by_id:
            cls.load()
        return cls._by_id.get(term_id.strip().lower())

    @classmethod
    def categories(cls) -> List[str]:
        entries = cls.load()
        cats = sorted({e.get("category", "기타") for e in entries})
        return ["전체"] + cats


class GlossaryPopupDialog(QDialog):
    """Modal popup window presenting detailed intuitive explanation of a glossary term."""

    def __init__(self, term_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.term_id = term_id
        self.term_data = GlossaryRepository.get_term(term_id)
        self.setWindowTitle("GNSS 전문 용어 해설")
        self.setMinimumSize(560, 380)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet("QDialog { background-color: #F8F8F6; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        if not self.term_data:
            load_err = GlossaryRepository.get_last_error()
            msg = load_err if load_err else f"용어 ID '{self.term_id}'를 사전에서 찾을 수 없습니다."
            err_lbl = QLabel(msg)
            err_lbl.setWordWrap(True)
            err_lbl.setStyleSheet("font-size: 14px; color: #DC2626; font-weight: 700;")
            layout.addWidget(err_lbl)
        else:
            cat_badge = QLabel(self.term_data.get("category", "전문 용어"))
            cat_badge.setStyleSheet(
                """
                background-color: #E3EAE5;
                color: #1E3A2F;
                font-size: 12px;
                font-weight: 800;
                padding: 4px 10px;
                border-radius: 8px;
                """
            )
            cat_row = QHBoxLayout()
            cat_row.addWidget(cat_badge)
            cat_row.addStretch(1)
            layout.addLayout(cat_row)

            title_ko = QLabel(self.term_data["term_ko"])
            title_ko.setWordWrap(True)
            title_ko.setStyleSheet("font-size: 19px; font-weight: 900; color: #111827;")
            layout.addWidget(title_ko)

            title_en = QLabel(self.term_data["term_en"])
            title_en.setWordWrap(True)
            title_en.setStyleSheet("font-size: 13px; font-weight: 600; color: #5C6B62;")
            layout.addWidget(title_en)

            browser = QTextBrowser()
            browser.setOpenLinks(False)
            browser.anchorClicked.connect(self._on_internal_anchor_clicked)
            browser.setFrameShape(QFrame.Shape.NoFrame)
            browser.setStyleSheet(
                """
                QTextBrowser {
                    background-color: #FFFFFF;
                    border: 1px solid #E2E6E0;
                    border-radius: 10px;
                    padding: 14px;
                    font-size: 13.5px;
                    color: #1F2937;
                }
                """
            )
            html = f"""
            <div style="line-height: 1.68;">
                <div style="margin-top:0; background-color:#F4F7F4; padding:11px 14px; border-left:4px solid #16A34A; border-radius:6px; margin-bottom:12px;">
                    <b style="color:#166534;">[Lv.1 입문 한줄 직관]</b><br>
                    <span style="color:#14532D; font-weight:700; font-size:14px;">{self.term_data['summary']}</span>
                </div>
                <h4 style="color:#1E3A2F; margin-bottom:4px;">[Lv.1 → Lv.2 레벨업] 쉬운 원리 · 실무 중요성</h4>
                <p style="margin-top:2px; color:#374151; font-size:13.8px;">{self.term_data['description']}</p>
                <h4 style="color:#1E3A2F; margin-bottom:4px;">[Lv.2 연구소 실무] 핵심 수식 및 수치 기준</h4>
                <pre style="background-color:#0F172A; color:#E2E8F0; padding:12px; border-radius:8px; font-family:'Consolas',monospace; font-size:12.5px;">{self.term_data.get('formula_or_example', '-')}</pre>
                <p style="color:#5C6B62; font-size:12px; margin-bottom:0;">
                    연관 모듈: <b>{', '.join(self.term_data.get('related_modules', []))}</b>
                </p>
            </div>
            """
            browser.setHtml(html)
            layout.addWidget(browser, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("닫기 (ESC)")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1E3A2F;
                color: #FFFFFF;
                font-weight: 800;
                padding: 8px 20px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #142820;
            }
            """
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_internal_anchor_clicked(self, url) -> None:
        if url.scheme() == "glossary":
            next_id = url.host() or url.path().lstrip("/")
            if next_id and next_id != self.term_id:
                show_glossary_popup(next_id, parent=self)


def show_glossary_popup(term_id: str, parent: Optional[QWidget] = None) -> None:
    """Convenience function to open the GlossaryPopupDialog for a term_id."""
    if not GlossaryRepository.get_term(term_id):
        err = GlossaryRepository.get_last_error()
        QMessageBox.warning(
            parent,
            "용어 사전 조회 실패",
            err if err else f"요청한 용어 ID '{term_id}'를 사전에서 찾을 수 없습니다.",
        )
        return
    dlg = GlossaryPopupDialog(term_id=term_id, parent=parent)
    dlg.exec()
