# Scripts/schedule.py
from __future__ import annotations

from typing import Optional

import pandas as pd

from Scripts.data_io import (
    load_oad_schedule,
    load_event_skill,
    load_odds_and_results,
)


def build_season_schedule(season_year: int) -> pd.DataFrame:
    """
    Build the season schedule for a given year by combining:
      - OAD_YYYY.xlsx (OAD schedule)
      - event_skill.xlsx (avg_skill, x_score)
      - Odds_and_Results.xlsx (Event_Tier at event-level)

    Returns
    -------
    DataFrame with at least:
      ['year', 'event_id', 'event_name', 'event_date',
       'start_date', 'course_num', 'course_name',
       'purse', 'winner_share', 'rank',
       'avg_skill', 'x_score', 'Event_Tier']
    """
    yr = int(season_year)

    oad = load_oad_schedule(yr).copy()
    ev_skill = load_event_skill().copy()
    odds = load_odds_and_results().copy()

    # --- normalize types ---
    for col in ["year", "event_id", "course_num"]:
        if col in oad.columns:
            oad[col] = pd.to_numeric(oad[col], errors="coerce")
        if col in ev_skill.columns:
            ev_skill[col] = pd.to_numeric(ev_skill[col], errors="coerce")
        if col in odds.columns:
            odds[col] = pd.to_numeric(odds[col], errors="coerce")

    if "start_date" in oad.columns:
        oad["start_date"] = pd.to_datetime(oad["start_date"], errors="coerce")
    if "event_date" in ev_skill.columns:
        ev_skill["event_date"] = pd.to_datetime(
            ev_skill["event_date"], errors="coerce"
        )
    if "event_completed" in odds.columns:
        odds["event_completed"] = pd.to_datetime(
            odds["event_completed"], errors="coerce"
        )

    # --- event-level Event_Tier from odds ---
    tier_cols = ["year", "event_id", "Event_Tier"]
    if "Event_Tier" in odds.columns:
        event_tier = (
            odds.dropna(subset=["Event_Tier"])
            .drop_duplicates(subset=["year", "event_id"])[tier_cols]
        )
    else:
        # if Event_Tier isn't present, just create a placeholder
        event_tier = oad[["year", "event_id"]].drop_duplicates().copy()
        event_tier["Event_Tier"] = "UNKNOWN"

    # --- restrict to target year ---
    oad_yr = oad[oad["year"] == yr].copy()
    ev_skill_yr = ev_skill[ev_skill["year"] == yr].copy()
    event_tier_yr = event_tier[event_tier["year"] == yr].copy()

    # --- merge OAD + event_skill on (year, event_id) ---
    sched = oad_yr.merge(
        ev_skill_yr[["year", "event_id", "event_date", "avg_skill", "x_score"]],
        on=["year", "event_id"],
        how="left",
    )

    # if event_date missing, fall back to start_date
    if "event_date" not in sched.columns:
        sched["event_date"] = sched.get("start_date")
    else:
        sched["event_date"] = sched["event_date"].fillna(sched.get("start_date"))

    # --- attach Event_Tier ---
    sched = sched.merge(
        event_tier_yr[["year", "event_id", "Event_Tier"]],
        on=["year", "event_id"],
        how="left",
    )

    # sort chronologically
    sched = sched.sort_values("event_date").reset_index(drop=True)

    return sched