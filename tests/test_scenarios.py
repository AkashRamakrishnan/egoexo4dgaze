"""Scenario matching, config coverage, and keep-all-labels prune."""

from __future__ import annotations

from pathlib import Path

from egoexo4d.export_csv import prune_rare_labels
from egoexo4d.paths import REPO_ROOT
from egoexo4d.scenarios import SCENARIOS, SLUGS, is_scenario_take, scenario_spec


def test_official_parent_task_ids():
    assert scenario_spec("cooking")["parent_task_id"] == 1000
    assert scenario_spec("health")["parent_task_id"] == 2000
    assert scenario_spec("bike_repair")["parent_task_id"] == 4000
    assert scenario_spec("music")["parent_task_id"] == 5000
    assert scenario_spec("basketball")["parent_task_id"] == 6000
    assert scenario_spec("soccer")["parent_task_id"] == 8000
    assert SLUGS == (
        "cooking",
        "health",
        "bike_repair",
        "music",
        "basketball",
        "soccer",
    )


def test_scenario_configs_exist():
    for slug in SLUGS:
        path = REPO_ROOT / SCENARIOS[slug]["config"]
        assert path.is_file(), path


def test_scenario_match_does_not_cross_subsets():
    cooking = {"scenario": {"slug": "cooking"}}
    health = {"scenario": {"slug": "health"}}
    bike = {"scenario": {"slug": "bike_repair"}}
    soccer = {"scenario": {"slug": "soccer"}}

    assert is_scenario_take({"parent_task_id": "1000", "take_name": "x"}, cooking)
    assert is_scenario_take({"take_name": "iiith_cooking_12_1"}, cooking)
    assert not is_scenario_take({"take_name": "cmu_bike01_2"}, cooking)
    assert not is_scenario_take({"take_name": "sfu_covid_001_2"}, cooking)

    assert is_scenario_take({"take_name": "sfu_covid_001_2"}, health)
    assert is_scenario_take({"parent_task_id": 2000}, health)
    assert not is_scenario_take({"take_name": "iiith_cooking_12_1"}, health)

    assert is_scenario_take({"take_name": "cmu_bike01_2"}, bike)
    assert is_scenario_take({"parent_task_id": 4000}, bike)
    assert not is_scenario_take({"take_name": "iiith_cooking_12_1"}, bike)

    assert is_scenario_take({"parent_task_id": 8000, "take_name": "x"}, soccer)
    assert is_scenario_take({"take_name": "uniandes_soccer_001_2"}, soccer)


def test_min_label_count_one_keeps_rare_labels():
    rows = [
        {"label_text": "rare spice"},
        {"label_text": "bowl"},
        {"label_text": "bowl"},
    ]
    kept, stats = prune_rare_labels(rows, min_count=1)
    assert kept == rows
    assert stats["dropped_rows"] == 0
    assert stats["dropped_labels"] == 0

    pruned, prune_stats = prune_rare_labels(rows, min_count=5)
    assert [r["label_text"] for r in pruned] == []
    assert prune_stats["dropped_labels"] == 2
    assert prune_stats["dropped_rows"] == 3


def test_cooking_config_keeps_all_labels():
    text = Path(REPO_ROOT / "configs/cooking_mvp.yaml").read_text()
    assert "min_label_count: 1" in text
    for slug in SLUGS:
        if slug == "cooking":
            continue
        other = Path(REPO_ROOT / SCENARIOS[slug]["config"]).read_text()
        assert "min_label_count: 1" in other
