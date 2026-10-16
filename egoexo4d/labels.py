"""Normalize Relations track names to GazeDETR label_text."""

from __future__ import annotations

import re

_TRACK_SUFFIX = re.compile(r"_\d+$")
_HAND_PATTERN = re.compile(r"hand", re.IGNORECASE)


def normalize_label(track_name: str) -> str:
    """Strip trailing ``_N`` instance suffix from a Relations track name."""
    return _TRACK_SUFFIX.sub("", track_name.strip())


def is_hand_track(track_name: str) -> bool:
    return bool(_HAND_PATTERN.search(track_name))


def should_keep_track(track_name: str, *, drop_hand_tracks: bool) -> bool:
    if drop_hand_tracks and is_hand_track(track_name):
        return False
    label = normalize_label(track_name)
    return bool(label)
