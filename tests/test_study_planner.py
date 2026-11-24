"""Unit tests for core/study_planner.py validating dynamic N-day schedule generation, editing, persistence, and Markdown export."""

from datetime import date
from pathlib import Path

import pytest

from core.study_planner import (
    export_study_plan_markdown,
    generate_study_plan,
    load_study_plan,
    save_study_plan,
)


@pytest.mark.parametrize("num_days", [3, 5, 7, 10, 14, 21, 30])
def test_dynamic_study_plan_generation_any_days(num_days: int):
    """Verify generate_study_plan produces exact day count, sequential dates from 2026-11-24, and valid module links."""
    start = date(2026, 11, 24)
    plan = generate_study_plan(total_days=num_days, start_date=start, daily_hours=2.0, focus_track="balanced")

    assert len(plan.days) == num_days
    assert plan.days[0].date_str == "2026-11-24"
    assert plan.days[-1].day_number == num_days

    valid_modules = {
        "m01_coords_time",
        "m02_signal_acq_track",
        "m03_pvt_dop",
        "m04_antijam_crpa",
        "m05_gnss_ins_ekf",
        "m06_egi_raim_spoof",
    }
    for d in plan.days:
        assert d.target_module_id in valid_modules
        assert d.target_level in ("lv1", "both", "lv2")
        assert d.hours == 2.0
        assert len(d.step1_concept) > 10
        assert len(d.self_check_question) > 10


def test_study_plan_save_load_and_markdown_export(tmp_path: Path):
    """Verify saving user edits (notes, title, completion status) to JSON and reloading them."""
    start = date(2026, 11, 24)
    plan = generate_study_plan(total_days=7, start_date=start, daily_hours=2.0)
    plan.days[0].completed = True
    plan.days[0].title = "사용자 커스텀 수정 제목"
    plan.days[0].user_notes = "ECEF는 반드시 double로 선언할 것!"

    json_file = tmp_path / "custom_plan.json"
    save_study_plan(plan, custom_path=json_file)
    assert json_file.exists()

    loaded = load_study_plan(custom_path=json_file)
    assert len(loaded.days) == 7
    assert loaded.days[0].completed is True
    assert loaded.days[0].title == "사용자 커스텀 수정 제목"
    assert "double" in loaded.days[0].user_notes

    md = export_study_plan_markdown(loaded)
    assert "2026-11-24" in md
    assert "사용자 커스텀 수정 제목" in md
