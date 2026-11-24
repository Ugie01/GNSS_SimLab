"""
GNSS SimLab — Main Entrypoint
Modern IT / SaaS Desktop Theme (Warm Alabaster & Forest Green Palette),
Windows Taskbar AppUserModelID registration, and explicit QMessageBox alert dialogs on error.
"""

from __future__ import annotations

import ctypes
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import matplotlib
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from core.resource_helper import resource_path
from ui.main_window import MainWindow

# Configure Matplotlib for crisp, non-overlapping scientific plots matching the warm SaaS theme
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.facecolor"] = "#FAFBF9"
matplotlib.rcParams["axes.edgecolor"] = "#D5DDD7"
matplotlib.rcParams["axes.linewidth"] = 0.9
matplotlib.rcParams["grid.color"] = "#E2E8E4"
matplotlib.rcParams["grid.alpha"] = 0.8
matplotlib.rcParams["axes.titlesize"] = 10.5
matplotlib.rcParams["axes.titlepad"] = 8.0
matplotlib.rcParams["axes.labelsize"] = 9.2
matplotlib.rcParams["xtick.labelsize"] = 8.4
matplotlib.rcParams["ytick.labelsize"] = 8.4
matplotlib.rcParams["legend.fontsize"] = 8.2
matplotlib.rcParams["figure.facecolor"] = "#FFFFFF"

GLOBAL_STYLESHEET = """
QWidget {
    font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
    color: #1E2420;
}
QToolTip {
    background-color: #1E3A2F;
    color: #FFFFFF;
    border: 1px solid #2D5A47;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #C5CEC8;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #94A39B;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #C5CEC8;
    min-width: 28px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #94A39B;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QLineEdit {
    padding: 5px 9px;
    border: 1px solid #D5DDD7;
    border-radius: 7px;
    background-color: #FFFFFF;
    color: #111827;
    font-weight: 600;
    font-size: 12.5px;
    min-height: 20px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus, QLineEdit:focus {
    border: 1.5px solid #1E3A2F;
    background-color: #FAFBF9;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #E2E8E4;
    margin: 4px 0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #1E3A2F;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2.5px solid #1E3A2F;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #E3EAE5;
    border: 2.5px solid #142820;
}
QCheckBox {
    spacing: 8px;
    font-size: 12.5px;
    font-weight: 600;
    color: #1E2420;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1.5px solid #94A39B;
    background-color: #FFFFFF;
}
QCheckBox::indicator:checked {
    background-color: #1E3A2F;
    border: 1.5px solid #1E3A2F;
}
"""


def _set_windows_taskbar_appid() -> None:
    """Register an explicit AppUserModelID on Windows so the taskbar uses our custom app.ico."""
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Danam.NavigationLab.GNSSSimLab.3.0")


def _global_exception_hook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    """Display an explicit popup alert dialog whenever an unhandled error occurs."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    sys.stderr.write(f"[GNSS SimLab Error]\n{err_msg}\n")

    app = QApplication.instance()
    if app is not None:
        QMessageBox.critical(
            app.activeWindow(),
            "오류 알림 — GNSS SimLab",
            f"프로그램 실행 중 문제가 발생했습니다.\n\n오류 유형: {exc_type.__name__}\n상세 내용: {exc_value}",
        )


def main() -> int:
    _set_windows_taskbar_appid()
    sys.excepthook = _global_exception_hook

    app = QApplication(sys.argv)
    app.setApplicationName("GNSS SimLab")
    app.setOrganizationName("DanamNavigationSWLab")
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_STYLESHEET)

    icon_path = Path(resource_path("assets/app.ico"))
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    default_font = QFont("Malgun Gothic", 10)
    app.setFont(default_font)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
