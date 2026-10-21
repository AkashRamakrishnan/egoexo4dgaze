#!/usr/bin/env python3
"""Print take UIDs for the config scenario (default: Cooking)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("list_scenario_uids.py")), run_name="__main__")
