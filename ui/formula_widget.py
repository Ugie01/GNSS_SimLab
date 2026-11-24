"""
High-DPI offline LaTeX formula renderer using Matplotlib mathtext (no system TeX install required)
and interactive Formula Card widget with symbol-by-symbol explanation table.
Styled with the Warm Alabaster & Forest Green SaaS theme.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# Configure Matplotlib mathtext to use DejaVu Sans for crisp math rendering and complete Unicode minus glyphs
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["mathtext.default"] = "it"

_FORMULA_PIXMAP_CACHE: Dict[Tuple[str, int, str], QPixmap] = {}


@dataclass(frozen=True)
class SymbolSpec:
    """Specification of a mathematical symbol inside a formula."""

    symbol_latex: str
    name_ko: str
    unit: str
    meaning: str
    glossary_id: Optional[str] = None


@dataclass(frozen=True)
class FormulaSpec:
    """Complete educational formula specification for a module."""

    title: str
    latex_expr: str
    intuitive_summary: str
    symbols: List[SymbolSpec]


def render_latex_to_pixmap(
    latex_expr: str,
    fontsize: int = 14,
    text_color: str = "#111827",
    dpi: int = 210,
) -> QPixmap:
    """
    Render a single-line or multi-line LaTeX math expression string (using Matplotlib's
    offline mathtext engine) to a crisp, tight-cropped transparent QPixmap.
    """
    raw_lines = [line.strip() for line in latex_expr.strip().split("\n") if line.strip()]
    wrapped_lines = []
    for line in raw_lines:
        line = line.replace(r"\le ", r"\leq ").replace(r"\ge ", r"\geq ")
        if not (line.startswith("$") and line.endswith("$")):
            wrapped_lines.append(f"${line}$")
        else:
            wrapped_lines.append(line)
    clean_expr = "\n".join(wrapped_lines)

    cache_key = (clean_expr, fontsize, text_color)
    if cache_key in _FORMULA_PIXMAP_CACHE:
        return _FORMULA_PIXMAP_CACHE[cache_key]

    fig_height = max(1.2, 0.75 * len(wrapped_lines))
    fig = Figure(figsize=(9.5, fig_height), dpi=dpi)
    fig.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(fig)

    text_artist = fig.text(
        0.01,
        0.5,
        clean_expr,
        fontsize=fontsize,
        color=text_color,
        va="center",
        ha="left",
        linespacing=1.65,
    )

    try:
        canvas.draw()
    except ValueError:
        fallback_text = latex_expr.replace("$", "")
        text_artist.set_text(fallback_text)
        canvas.draw()

    renderer = canvas.get_renderer()
    bbox = text_artist.get_window_extent(renderer=renderer).padded(10)

    buf = io.BytesIO()
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches=bbox_inches, pad_inches=0.04)
    buf.seek(0)

    qimg = QImage.fromData(buf.getvalue(), "PNG")
    pixmap = QPixmap.fromImage(qimg)
    pixmap.setDevicePixelRatio(1.45)

    _FORMULA_PIXMAP_CACHE[cache_key] = pixmap
    return pixmap


class FormulaCardWidget(QFrame):
    """
    Clean SaaS visual card displaying:
    1. Formula Title & Plain-Korean intuitive translation
    2. High-DPI Matplotlib mathtext rendered formula image (auto-fit to tab width)
    3. Structured HTML table explaining every symbol, unit, physical meaning, and glossary link
    """

    def __init__(
        self,
        spec: FormulaSpec,
        on_glossary_link: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.on_glossary_link = on_glossary_link
        self.symbol_browser: Optional[QTextBrowser] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("FormulaCard")
        self.setStyleSheet(
            """
            QFrame#FormulaCard {
                background-color: #FFFFFF;
                border: 1px solid #E2E6E0;
                border-left: 4px solid #1E3A2F;
                border-radius: 10px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Title
        title_lbl = QLabel(self.spec.title.replace("&", "·"))
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet("font-size: 14.5px; font-weight: 800; color: #111827; border: none;")
        layout.addWidget(title_lbl)

        # Plain-Korean Summary Box
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            """
            QFrame {
                background-color: #F4F7F4;
                border: 1px solid #D8E2DB;
                border-radius: 7px;
            }
            """
        )
        sf_layout = QVBoxLayout(self.summary_frame)
        sf_layout.setContentsMargins(12, 8, 12, 8)
        summary_lbl = QLabel(f"해설  |  {self.spec.intuitive_summary}")
        summary_lbl.setWordWrap(True)
        summary_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #1E3A2F; border: none;")
        sf_layout.addWidget(summary_lbl)
        layout.addWidget(self.summary_frame)

        # Rendered LaTeX Formula box
        self.formula_box = QFrame()
        self.formula_box.setStyleSheet(
            """
            QFrame {
                background-color: #FAFBF9;
                border: 1px solid #E2E6E0;
                border-radius: 8px;
            }
            """
        )
        fb_layout = QHBoxLayout(self.formula_box)
        fb_layout.setContentsMargins(14, 10, 14, 10)

        formula_img_lbl = QLabel()
        formula_img_lbl.setStyleSheet("border: none; background: transparent;")
        pixmap = render_latex_to_pixmap(self.spec.latex_expr, fontsize=14, text_color="#111827")
        logical_w = pixmap.width() / pixmap.devicePixelRatio()
        if logical_w > 760:
            target_px_w = int(760 * pixmap.devicePixelRatio())
            pixmap = pixmap.scaledToWidth(target_px_w, Qt.TransformationMode.SmoothTransformation)
        formula_img_lbl.setPixmap(pixmap)
        formula_img_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        fb_layout.addWidget(formula_img_lbl)
        fb_layout.addStretch(1)
        layout.addWidget(self.formula_box)

        # Symbol table rendered in QTextBrowser to support glossary:// links cleanly
        if self.spec.symbols:
            browser = QTextBrowser()
            self.symbol_browser = browser
            browser.setOpenLinks(False)
            browser.setFrameShape(QFrame.Shape.NoFrame)
            browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            browser.setStyleSheet("background: transparent; border: none;")
            browser.anchorClicked.connect(self._handle_anchor_clicked)

            rows_html = []
            for sym in self.spec.symbols:
                term_link = ""
                if sym.glossary_id:
                    term_link = (
                        f' <a href="glossary://{sym.glossary_id}" '
                        f'style="color:#166534; text-decoration:none; font-weight:700;">[용어보기]</a>'
                    )
                rows_html.append(
                    f"""
                    <tr>
                        <td style="padding:6px 8px; border-bottom:1px solid #E5E7E1; font-family:'Consolas',monospace; font-weight:700; color:#1E3A2F; white-space:nowrap;">{sym.symbol_latex}</td>
                        <td style="padding:6px 8px; border-bottom:1px solid #E5E7E1; font-weight:600; color:#111827; white-space:nowrap;">{sym.name_ko}</td>
                        <td style="padding:6px 8px; border-bottom:1px solid #E5E7E1; color:#6B7280; font-family:'Consolas',monospace; white-space:nowrap;">{sym.unit}</td>
                        <td style="padding:6px 8px; border-bottom:1px solid #E5E7E1; color:#374151;">{sym.meaning}{term_link}</td>
                    </tr>
                    """
                )

            table_html = f"""
            <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse; font-size:12.5px;">
                <thead>
                    <tr style="background-color:#F3F4F0;">
                        <th align="left" style="padding:7px 8px; border-bottom:2px solid #D5DDD7; color:#374151;">기호</th>
                        <th align="left" style="padding:7px 8px; border-bottom:2px solid #D5DDD7; color:#374151;">명칭</th>
                        <th align="left" style="padding:7px 8px; border-bottom:2px solid #D5DDD7; color:#374151;">단위</th>
                        <th align="left" style="padding:7px 8px; border-bottom:2px solid #D5DDD7; color:#374151;">물리적 의미 및 역할</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
            """
            browser.setHtml(table_html)
            browser.document().setTextWidth(660)
            doc_height = int(browser.document().size().height()) + 16
            browser.setMinimumHeight(max(doc_height, 40 + len(self.spec.symbols) * 34))
            browser.setMaximumHeight(max(doc_height + 24, 48 + len(self.spec.symbols) * 40))
            layout.addWidget(browser)

    def set_level_mode(self, level_mode: str) -> None:
        if self.symbol_browser is not None:
            self.symbol_browser.setVisible(level_mode != "lv1")

    def _handle_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() == "glossary" and self.on_glossary_link:
            term_id = url.host() or url.path().lstrip("/")
            self.on_glossary_link(term_id)
