from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import warnings

from Scripts.config import (
    COMBINED_ROUNDS_PATH,
    COURSE_FIT_TEMPLATE,
    EVENT_SKILL_PATH,
    OAD_TEMPLATE,
    ODDS_AND_RESULTS_PATH,
    PRESEASON_TEMPLATE,
    PLAYER_SKILL_TEMPLATE,
)

warnings.filterwarnings(
    "ignore",
    message="Columns \\(36\\) have mixed types. Specify dtype option on import or set low_memory=False.",
    category=pd.errors.DtypeWarning,
)


def _ensure_path(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}")
    return p


def load_rounds() -> pd.DataFrame:
    """
    Load all rounds 2017–2025.

    Returns
    -------
    DataFrame with at least:
    ['tour','year','season','event_completed','event_name','event_id',
     'player_name','dg_id','fin_text','round_num','course_name','course_num',
     'course_par','start_hole','teetime','round_score','sg_putt','sg_arg',
     'sg_app','sg_ott','sg_t2g','sg_total','driving_dist','driving_acc',
     'gir','scrambling','prox_rgh','prox_fw','great_shots','poor_shots',
     'eagles_or_better','birdies','pars','bogies','doubles_or_worse',
     'finish_num','round_date']
    """
    path = _ensure_path(COMBINED_ROUNDS_PATH)
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={"round_date": "string"},
    )

    # Standardize dtypes
    date_cols = ["round_date", "event_completed"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Make sure key IDs are numeric where possible
    for col in ["year", "season", "event_id", "dg_id", "course_num"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_course_fit(season: int) -> pd.DataFrame:
    """
    Load course importance profile for a given season.
    """
    path = _ensure_path(Path(str(COURSE_FIT_TEMPLATE).format(season=season)))
    df = pd.read_csv(path, low_memory=False)
    return df


def load_event_skill() -> pd.DataFrame:
    """
    Load event_skill.xlsx with event-level avg_skill and x_score.
    """
    path = _ensure_path(EVENT_SKILL_PATH)
    df = pd.read_excel(path)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    for col in ["year", "event_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_oad_schedule(season: int) -> pd.DataFrame:
    """
    Load OAD season schedule for a given year (OAD_YYYY.xlsx).
    """
    path = _ensure_path(Path(str(OAD_TEMPLATE).format(season=season)))
    df = pd.read_excel(path)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    for col in ["year", "event_id", "course_num"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_odds_and_results() -> pd.DataFrame:
    """
    Load Odds_and_Results.xlsx with fields, odds, and outcomes.
    """
    path = _ensure_path(ODDS_AND_RESULTS_PATH)
    df = pd.read_excel(path)
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
    for col in ["year", "event_id", "dg_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_preseason(season: int) -> pd.DataFrame:
    """
    Load preseason shortlist-style file for a given season.
    """
    path = _ensure_path(Path(str(PRESEASON_TEMPLATE).format(season=season)))
    df = pd.read_csv(path, low_memory=False)
    for col in ["dg_id", "target_season"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_player_skills(season: int) -> pd.DataFrame:
    """
    Load player 5-attribute skills (if/when you start saving them).
    """
    path = Path(str(PLAYER_SKILL_TEMPLATE).format(season=season))
    if not path.exists():
        raise FileNotFoundError(
            f"Player skills file not found for season {season}: {path}"
        )
    df = pd.read_csv(path, low_memory=False)
    for col in ["dg_id", "n_rounds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df