"""Path helpers and config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cooking_mvp.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    root = base if base is not None else REPO_ROOT
    return (root / p).resolve()


def egoexo_root(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg["egoexo_root"])


def output_root(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg["output_root"])


def manifests_dir(cfg: Dict[str, Any]) -> Path:
    return output_root(cfg) / "manifests"


def images_dir(cfg: Dict[str, Any]) -> Path:
    return output_root(cfg) / "images"


def annotations_dir(cfg: Dict[str, Any]) -> Path:
    return output_root(cfg) / "annotations"


def take_dir(cfg: Dict[str, Any], take_name: str) -> Path:
    return egoexo_root(cfg) / "takes" / take_name


def relations_path(cfg: Dict[str, Any], split: str) -> Path:
    """Resolve Relations JSON (config path or common egoexo layout fallbacks)."""
    root = egoexo_root(cfg)
    rel = cfg["relations"][split]
    direct = root / rel
    if direct.exists():
        return direct
    fallbacks = [
        root / f"annotations/relations_{split}.json",
        root / f"annotations/egoexo_relations_{split}.json",
        root / f"annotations/benchmarks/relations/{split}.json",
        root / f"annotations/benchmarks/relations/relations_{split}.json",
    ]
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return direct


def takes_json_path(cfg: Dict[str, Any]) -> Path:
    return egoexo_root(cfg) / "takes.json"
