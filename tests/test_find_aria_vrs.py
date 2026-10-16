"""VRS discovery for Aria CPF calibration."""

from __future__ import annotations

from pathlib import Path

from egoexo4d.gaze_project import find_aria_vrs


def test_prefers_noimagestreams(tmp_path: Path):
    take = tmp_path / "takes" / "sfu_cooking_013_3"
    take.mkdir(parents=True)
    (take / "aria01.vrs").write_bytes(b"")
    (take / "aria01_noimagestreams.vrs").write_bytes(b"")
    found = find_aria_vrs(take)
    assert found is not None
    assert found.name == "aria01_noimagestreams.vrs"


def test_missing_vrs(tmp_path: Path):
    take = tmp_path / "takes" / "nus_cooking_16_2"
    take.mkdir(parents=True)
    assert find_aria_vrs(take) is None
