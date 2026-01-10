from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


@dataclass(frozen=True)
class PickLogConfig:
    log_dir: Path

    def season_log_path(self, season: int) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"picks_{season}.csv"

    def test_runs_path(self) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / "test_runs.csv"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def deny_if_duplicate_pick(
    log_df: pd.DataFrame,
    *,
    season: int,
    league_key: str,
    username: str,
    event_id: int,
) -> bool:
    """Return True if a pick already exists for this key."""
    if log_df.empty:
        return False
    need = {"season", "league_key", "username", "event_id"}
    if not need.issubset(set(log_df.columns)):
        return False  # old log schema; don't hard-block

    mask = (
        (pd.to_numeric(log_df["season"], errors="coerce") == int(season))
        & (log_df["league_key"].astype(str) == str(league_key))
        & (log_df["username"].astype(str) == str(username))
        & (pd.to_numeric(log_df["event_id"], errors="coerce") == int(event_id))
    )
    return bool(mask.any())


def append_pick_row(
    *,
    cfg: PickLogConfig,
    season: int,
    row: Dict[str, Any],
    deny_duplicates: bool = True,
) -> tuple[bool, str]:
    """
    Append a single pick row to picks_{season}.csv.
    Returns (success, message).
    """
    path = cfg.season_log_path(season)
    _ensure_parent(path)

    log_df = read_log(path)

    # required keys for dedupe
    league_key = str(row.get("league_key", ""))
    username = str(row.get("username", ""))
    event_id = int(row.get("event_id"))

    if deny_duplicates and deny_if_duplicate_pick(
        log_df, season=season, league_key=league_key, username=username, event_id=event_id
    ):
        return False, f"Denied: pick already exists in {path.name} for event_id={event_id}."

    out_row = pd.DataFrame([row])

    # If file doesn't exist, write with header. Otherwise append.
    if not path.exists() or log_df.empty:
        out_row.to_csv(path, index=False)
    else:
        out_row.to_csv(path, mode="a", header=False, index=False)

    return True, f"Saved to {path.name}."


def append_test_run(
    *,
    cfg: PickLogConfig,
    run_rows: pd.DataFrame,
) -> tuple[bool, str]:
    """
    Append an entire test run (multiple picks) to test_runs.csv
    (append-only).
    """
    path = cfg.test_runs_path()
    _ensure_parent(path)

    if run_rows is None or run_rows.empty:
        return False, "No rows to save."

    exists = path.exists()
    run_rows.to_csv(path, mode="a", header=not exists, index=False)
    return True, f"Saved test run rows to {path.name}."
