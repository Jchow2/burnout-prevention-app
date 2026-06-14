"""
src/analysis/personas.py

Loads personas.yaml and tracks.yaml from config/ and exposes typed
accessor functions used by the Streamlit front-end.

All data is read-only. No ML or external calls.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@lru_cache(maxsize=1)
def load_tracks() -> list[dict[str, Any]]:
    with open(_CONFIG_DIR / "tracks.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["tracks"]


@lru_cache(maxsize=1)
def load_personas() -> list[dict[str, Any]]:
    with open(_CONFIG_DIR / "personas.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["personas"]


def get_track(track_id: str) -> dict[str, Any] | None:
    return next((t for t in load_tracks() if t["track_id"] == track_id), None)


def get_persona(persona_id: str) -> dict[str, Any] | None:
    return next((p for p in load_personas() if p["id"] == persona_id), None)


def cluster_to_track_id(cluster_id: int) -> str | None:
    """Return the first track that covers the given cluster_id, or None."""
    for track in load_tracks():
        if cluster_id in track.get("cluster_ids", []):
            return track["track_id"]
    return None


def cluster_to_track_ids(cluster_id: int) -> list[str]:
    """Return all track_ids that cover the given cluster_id."""
    return [
        t["track_id"]
        for t in load_tracks()
        if cluster_id in t.get("cluster_ids", [])
    ]