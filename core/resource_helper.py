"""Resource path helper for both development environment and PyInstaller frozen bundle."""

import os
import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """
    Return absolute path to a resource file, working transparently in development
    and inside a PyInstaller bundle (sys._MEIPASS).
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(getattr(sys, "_MEIPASS"))
    else:
        base_path = Path(__file__).resolve().parent.parent
    return str(base_path / relative_path)


def ensure_asset_dir() -> Path:
    """Return path to the assets directory, creating it if needed in dev mode."""
    p = Path(resource_path("assets"))
    if not hasattr(sys, "_MEIPASS"):
        p.mkdir(parents=True, exist_ok=True)
    return p
