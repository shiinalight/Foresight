"""Load and validate a dataset written in the INTERFACE.md format."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["episode", "t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius", "regime"]
NUMERIC = ["episode", "t", "agent_id", "is_ego", "x", "y", "vx", "vy", "radius"]
MAX_SPEED = {"person": 5.0, "robot": 5.0, "car": 60.0}


def _read(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    return pd.DataFrame()


def load(dataset_dir: str | Path):
    """Return (tracks, obstacles, meta). Raises ValueError on a broken dataset."""
    d = Path(dataset_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tracks = _read(d / "tracks")
    obstacles = _read(d / "obstacles")
    validate(tracks, meta)
    tracks = tracks.sort_values(["episode", "t", "agent_id"]).reset_index(drop=True)
    return tracks, obstacles, meta


def validate(tracks: pd.DataFrame, meta: dict) -> None:
    problems = []
    missing = [c for c in REQUIRED if c not in tracks.columns]
    if missing:
        raise ValueError(f"tracks is missing columns: {missing}")
    for c in NUMERIC:
        if not pd.api.types.is_numeric_dtype(tracks[c]):
            problems.append(f"column {c} is not numeric")
    dt = float(meta["dt"])
    off_grid = np.abs(tracks["t"] / dt - np.round(tracks["t"] / dt)) > 1e-3
    if off_grid.any():
        problems.append(f"{int(off_grid.sum())} rows have t off the dt={dt} grid")
    dup = tracks.duplicated(["episode", "t", "agent_id"])
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate (episode, t, agent_id) rows")
    egos = tracks[tracks["is_ego"] == 1].groupby("episode")["agent_id"].nunique()
    if (egos > 1).any():
        problems.append(f"episodes with more than one ego: {list(egos[egos > 1].index)[:5]}")
    speed = np.hypot(tracks["vx"], tracks["vy"])
    limit = tracks["kind"].map(MAX_SPEED).fillna(60.0)
    if (speed > limit).any():
        problems.append(f"{int((speed > limit).sum())} rows faster than the sanity bound")
    if problems:
        raise ValueError("dataset invalid:\n  " + "\n  ".join(problems))
