"""
End-to-end integration tests for GNSS SimLab UI architecture, plugin registry,
glossary hyperlink integrity, presets loading, and pluggable template module.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.resource_helper import resource_path
from modules._template_new_module import TemplateNewLabModule
from ui.glossary_dialog import GlossaryRepository
from ui.main_window import MainWindow
from ui.registry import ModuleRegistry


@pytest.fixture(scope="session")
def qapp():
    """Provide a shared offscreen QApplication instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_resource_paths_and_data_files_exist():
    """Verify resource_path resolves glossary.json, presets.json, and app.ico."""
    glossary_path = Path(resource_path("data/glossary.json"))
    presets_path = Path(resource_path("data/presets.json"))
    icon_path = Path(resource_path("assets/app.ico"))

    assert glossary_path.exists()
    assert presets_path.exists()
    assert icon_path.exists()


def test_glossary_repository_minimum_35_terms():
    """Verify glossary.json contains at least 35 comprehensive terms with all required schema keys."""
    entries = GlossaryRepository.load()
    assert len(entries) >= 35

    required_keys = {"id", "term_ko", "term_en", "category", "summary", "description", "formula_or_example", "related_modules"}
    for item in entries:
        assert required_keys.issubset(item.keys()), f"Term {item.get('id')} missing required keys"
        assert GlossaryRepository.get_term(item["id"]) is not None


def test_all_registered_modules_and_hyperlinks(qapp):
    """
    Verify all 6 registered modules (M01..M06) instantiate cleanly, render formulas,
    have valid glossary:// links, and support all scenario presets in presets.json.
    """
    all_glossary_ids = {e["id"] for e in GlossaryRepository.load()}
    registered = ModuleRegistry.get_registered_modules()
    assert len(registered) == 6

    with open(resource_path("data/presets.json"), "r", encoding="utf-8") as f:
        presets_data = json.load(f)

    for mod_cls in registered:
        assert mod_cls.module_id in presets_data, f"Missing presets for {mod_cls.module_id}"
        assert len(presets_data[mod_cls.module_id]) >= 3

        widget = mod_cls()
        assert len(widget.get_formula_specs()) >= 3

        # Check all glossary:// hyperlinks in HTML sections
        for section_html in [
            widget.get_theory_html(),
            widget.get_observation_guide_html(),
            widget.get_embedded_c_porting_html(),
        ]:
            for tid in re.findall(r"glossary://([a-zA-Z0-9_]+)", section_html):
                assert tid in all_glossary_ids, f"Broken glossary link glossary://{tid} in {mod_cls.module_id}"

        # Check formula symbol glossary links
        for fspec in widget.get_formula_specs():
            for sym in fspec.symbols:
                if sym.glossary_id:
                    assert sym.glossary_id in all_glossary_ids

        # Cycle through all presets, test 2D/3D/Combined view modes, test Lv.1/Lv.2/Both level modes, and test reset
        for idx in range(widget.preset_combo.count()):
            widget.preset_combo.setCurrentIndex(idx)
        for v_mode in ("3d", "2d", "combined"):
            widget.set_view_mode(v_mode)
            assert widget.view_mode == v_mode
        for l_mode in ("lv1", "lv2", "both"):
            widget.set_level_mode(l_mode)
            assert widget.level_mode == l_mode
        assert "사수의 실시간 쉬운 해설" in widget.lbl_easy_coach.text()
        widget.reset_to_default_preset()


def test_template_module_and_main_window_navigation(qapp):
    """Verify the extension template module instantiates cleanly, MainWindow switches across all pages, and StudyPlanner launches modules."""
    tmpl = TemplateNewLabModule()
    tmpl.run_simulation()
    assert len(tmpl.get_formula_specs()) >= 1

    win = MainWindow()
    # 6 learning modules + 1 study planner tab + 1 glossary tab = 8 pages
    assert win.stack.count() == 8
    for i in range(win.stack.count()):
        win.stack.setCurrentIndex(i)
        assert win.stack.currentIndex() == i

    # Test StudyPlannerTabWidget dynamic regeneration for 10 days and 14 days + module launch
    planner = win.planner_widget
    planner.spin_total_days.setValue(10)
    planner.regenerate_plan_from_inputs()
    assert planner.table.rowCount() == 10

    planner.spin_total_days.setValue(14)
    planner.regenerate_plan_from_inputs()
    assert planner.table.rowCount() == 14

    # Launch Day 1 module from planner and verify MainWindow switches to M01 in Lv.1 mode
    planner.table.selectRow(0)
    planner.launch_selected_day_module()
    assert win.stack.currentIndex() == 0
    assert win._module_widgets_by_id["m01_coords_time"].level_mode == "lv1"

