"""Ego-Exo4D Relations scenarios (not Bouldering / Dance).

Official ``parent_task_id`` values from the Ego-Exo4D takes catalog
(Ego4D ``TASK_ID_CAT``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Relations benchmark covers these six parent tasks.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "cooking": {
        "slug": "cooking",
        "parent_task_id": 1000,
        "parent_task_name": "Cooking",
        "name_tokens": ("cooking", "cook", "omelet", "salad", "sushi", "recipe"),
        "exclude_tokens": ("covid",),
        "config": "configs/cooking_mvp.yaml",
        "output_root": "./data",
    },
    "health": {
        "slug": "health",
        "parent_task_id": 2000,
        "parent_task_name": "Health",
        "name_tokens": ("health", "covid", "cpr", "firstaid", "first_aid"),
        "exclude_tokens": (),
        "config": "configs/health.yaml",
        "output_root": "./data_health",
    },
    "bike_repair": {
        "slug": "bike_repair",
        "parent_task_id": 4000,
        "parent_task_name": "Bike Repair",
        "name_tokens": ("bike",),
        "exclude_tokens": ("covid",),
        "config": "configs/bike_repair.yaml",
        "output_root": "./data_bike_repair",
    },
    "music": {
        "slug": "music",
        "parent_task_id": 5000,
        "parent_task_name": "Music",
        "name_tokens": ("music", "guitar", "piano", "violin", "drum"),
        "exclude_tokens": ("covid",),
        "config": "configs/music.yaml",
        "output_root": "./data_music",
    },
    "basketball": {
        "slug": "basketball",
        "parent_task_id": 6000,
        "parent_task_name": "Basketball",
        "name_tokens": ("basketball",),
        "exclude_tokens": ("covid",),
        "config": "configs/basketball.yaml",
        "output_root": "./data_basketball",
    },
    "soccer": {
        "slug": "soccer",
        "parent_task_id": 8000,
        "parent_task_name": "Soccer",
        "name_tokens": ("soccer",),
        "exclude_tokens": ("covid",),
        "config": "configs/soccer.yaml",
        "output_root": "./data_soccer",
    },
}

SLUGS: Tuple[str, ...] = (
    "cooking",
    "health",
    "bike_repair",
    "music",
    "basketball",
    "soccer",
)


def scenario_spec(cfg_or_slug: Mapping[str, Any] | str) -> Dict[str, Any]:
    if isinstance(cfg_or_slug, str):
        slug = cfg_or_slug
        extra: Dict[str, Any] = {}
    else:
        block = dict(cfg_or_slug.get("scenario") or {})
        slug = str(block.get("slug") or _slug_from_block(block) or "cooking")
        extra = block
    if slug not in SCENARIOS:
        raise ValueError(f"Unknown scenario slug {slug!r}; expected one of {SLUGS}")
    out = dict(SCENARIOS[slug])
    out.update({k: v for k, v in extra.items() if v is not None})
    out["slug"] = slug
    return out


def _slug_from_block(block: Mapping[str, Any]) -> Optional[str]:
    raw_id = block.get("parent_task_id")
    if raw_id is not None:
        for slug, spec in SCENARIOS.items():
            if str(spec.get("parent_task_id")) == str(raw_id):
                return slug
    name = str(block.get("parent_task_name") or "").strip().lower()
    if name:
        for slug, spec in SCENARIOS.items():
            if str(spec["parent_task_name"]).lower() == name:
                return slug
    return None


def scenario_name(cfg: Mapping[str, Any]) -> str:
    return str(scenario_spec(cfg)["parent_task_name"])


def scenario_slug(cfg: Mapping[str, Any]) -> str:
    return str(scenario_spec(cfg)["slug"])


def _blob(*parts: Any) -> str:
    return " ".join(str(p) for p in parts if p).lower()


def text_matches_scenario(spec: Mapping[str, Any], *parts: Any) -> bool:
    text = _blob(*parts)
    if not text:
        return False
    for token in spec.get("exclude_tokens") or ():
        if token and token in text:
            return False
    tokens: Sequence[str] = tuple(spec.get("name_tokens") or ())
    return any(token in text for token in tokens)


def is_scenario_take(take: dict, cfg: Mapping[str, Any], extra_parts: Iterable[Any] = ()) -> bool:
    spec = scenario_spec(cfg)
    task_id = spec.get("parent_task_id")
    raw_id = take.get("parent_task_id", take.get("task_id"))
    if task_id is not None and raw_id is not None and str(raw_id) == str(task_id):
        return True
    task_name = str(spec.get("parent_task_name") or "")
    name = str(take.get("parent_task_name") or take.get("task_name") or "")
    if task_name and name.lower().startswith(task_name.lower()):
        return True
    return text_matches_scenario(
        spec,
        name,
        take.get("take_name"),
        take.get("parent_task_name"),
        *extra_parts,
    )
