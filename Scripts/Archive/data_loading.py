from __future__ import annotations

from typing import Optional

import pandas as pd
import sys
sys.path.append("/")

from Scripts.Archive.config import (
    COMBINED_ROUNDS_ALL_PATH,
    EVENT_SKILL_PATH,
    SEASON_FILES,
    SeasonFiles,
    assert_file_exists,
)

# ---------------------------------------------------------------------
# Internal caches so we only hit disk once per process
# ---------------------------------------------------------------------
_COMBINED_CACHE: Optional[pd.DataFrame] = None
_EVENT_SKILL_CACHE: Optional[pd.DataFrame] = None
_ODDS_CACHE: dict[int, pd.DataFrame] = {}
_MASTER_CACHE: dict[int, pd.DataFrame] = {}
_OAD_SCHED_CACHE: dict[int, pd.DataFrame] = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_season_files(season: int) -> SeasonFiles:
    """
    Return configured file paths for a given season.

    Will raise KeyError if the season hasn't been wired into SEASON_FILES
    in config.py yet.
    """
    try:
        return SEASON_FILES[season]
    except KeyError as exc:
        raise KeyError(
            f"Season {season} is not configured in config.SEASON_FILES. "
            f"Add a SeasonFiles entry in config.py."
        ) from exc


# ---------------------------------------------------------------------
# Combined rounds (2017–2025)
# ---------------------------------------------------------------------
def load_combined_rounds(copy: bool = True) -> pd.DataFrame:
    """
    Load the master combined rounds file (2017–2025).

    Returns a DataFrame with:
      - year, season as int
      - event_completed, round_date as Timestamp (where available)
      - dg_id as Int64
    """
    global _COMBINED_CACHE

    if _COMBINED_CACHE is None:
        assert_file_exists(COMBINED_ROUNDS_ALL_PATH, "combined_rounds_all")
        df = pd.read_csv(COMBINED_ROUNDS_ALL_PATH, low_memory=False)

        # Basic type cleanup
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

        if "dg_id" in df.columns:
            df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

        if "event_id" in df.columns:
            df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")

        if "course_num" in df.columns:
            df["course_num"] = pd.to_numeric(df["course_num"], errors="coerce").astype("Int64")

        # Dates: event_completed is full-tournament; round_date may be partial
        if "event_completed" in df.columns:
            df["event_completed"] = pd.to_datetime(
                df["event_completed"], errors="coerce"
            )

        if "round_date" in df.columns:
            df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

        _COMBINED_CACHE = df

    return _COMBINED_CACHE.copy() if copy else _COMBINED_CACHE


def load_rounds_for_season(
    season: int,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Convenience helper: subset combined rounds to a single season.
    """
    df_all = load_combined_rounds(copy=False)
    mask = df_all["season"] == season
    df = df_all.loc[mask].copy()

    return df if copy else df


# ---------------------------------------------------------------------
# Event skill (avg field strength etc.)
# ---------------------------------------------------------------------
def load_event_skill(copy: bool = True) -> pd.DataFrame:
    """
    Load event_skill.xlsx and normalize core columns:
      - year (int)
      - event_id (Int64)
      - event_date (Timestamp, if present)
      - avg_skill as float
    """
    global _EVENT_SKILL_CACHE

    if _EVENT_SKILL_CACHE is None:
        assert_file_exists(EVENT_SKILL_PATH, "event_skill")
        esk = pd.read_excel(EVENT_SKILL_PATH)

        # Normalize obvious columns if they exist
        for col in ("year", "event_year"):
            if col in esk.columns:
                esk[col] = pd.to_numeric(esk[col], errors="coerce").astype("Int64")

        for col in ("event_id", "event_id_fixed"):
            if col in esk.columns:
                esk[col] = pd.to_numeric(esk[col], errors="coerce").astype("Int64")

        if "event_date" in esk.columns:
            esk["event_date"] = pd.to_datetime(esk["event_date"], errors="coerce")

        if "avg_skill" in esk.columns:
            esk["avg_skill"] = pd.to_numeric(esk["avg_skill"], errors="coerce")

        _EVENT_SKILL_CACHE = esk

    return _EVENT_SKILL_CACHE.copy() if copy else _EVENT_SKILL_CACHE


# ---------------------------------------------------------------------
# Season-specific: odds, master results, OAD schedule
# (stubs – we will wire in Odds_and_Results.xlsx / season CSVs later)
# ---------------------------------------------------------------------
def load_odds_for_season(season: int, copy: bool = True) -> pd.DataFrame:
    """
    Load odds for a given season.

    Currently assumes a CSV path in config.SEASON_FILES[season].odds_path.
    We'll later support reading from Odds_and_Results.xlsx via a preprocessing step.
    """
    global _ODDS_CACHE

    if season in _ODDS_CACHE:
        return _ODDS_CACHE[season].copy() if copy else _ODDS_CACHE[season]

    files = _get_season_files(season)
    assert_file_exists(files.odds_path, f"odds_{season}")

    df = pd.read_csv(files.odds_path, low_memory=False)

    # We'll formalize schema when we plug in the Excel source.
    _ODDS_CACHE[season] = df
    return df.copy() if copy else df


def load_master_results_for_season(season: int, copy: bool = True) -> pd.DataFrame:
    """
    Load master results for a given season.

    Currently assumes a CSV path in config.SEASON_FILES[season].master_results_path.
    Later we can consolidate this with Odds_and_Results.xlsx.
    """
    global _MASTER_CACHE

    if season in _MASTER_CACHE:
        return _MASTER_CACHE[season].copy() if copy else _MASTER_CACHE[season]

    files = _get_season_files(season)
    assert_file_exists(files.master_results_path, f"master_results_{season}")

    df = pd.read_csv(files.master_results_path, low_memory=False)

    # Normalize a couple obvious columns
    for col in ("dg_id", "event_id", "event_id_fixed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")

    _MASTER_CACHE[season] = df
    return df.copy() if copy else df


def load_oad_schedule_for_season(season: int, copy: bool = True) -> pd.DataFrame:
    """
    Load your OAD schedule (event list + rank + purse) for a given season.
    """
    global _OAD_SCHED_CACHE

    if season in _OAD_SCHED_CACHE:
        return _OAD_SCHED_CACHE[season].copy() if copy else _OAD_SCHED_CACHE[season]

    files = _get_season_files(season)
    assert_file_exists(files.oad_schedule_path, f"oad_schedule_{season}")

    df = pd.read_csv(files.oad_schedule_path, low_memory=False)

    # Standardize obvious columns
    df.columns = [c.strip() for c in df.columns]
    for col in ("event_id_fixed", "event_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    _OAD_SCHED_CACHE[season] = df
    return df.copy() if copy else df