# Scripts/manual_sim.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Scripts.schedule import build_season_schedule
from Scripts.data_io import load_odds_and_results
from Scripts.weekly_view import build_weekly_view


def init_manual_season(season_year: int) -> Dict:
    """
    Initialize a manual OAD season session.

    Loads:
      - schedule for the year (from OAD_YYYY + event_skill + Event_Tier)
      - odds/results table (for backtesting Winnings)
      - empty log
      - empty used_dg_ids set

    Returns a state dict you pass back into other functions.
    """
    yr = int(season_year)

    schedule_df = build_season_schedule(yr).copy()
    schedule_df = schedule_df.sort_values("event_date").reset_index(drop=True)

    odds_df = load_odds_and_results().copy()
    for col in ["year", "event_id", "dg_id"]:
        if col in odds_df.columns:
            odds_df[col] = pd.to_numeric(odds_df[col], errors="coerce")

    state = {
        "season_year": yr,
        "schedule": schedule_df,
        "odds": odds_df,
        "log": [],            # list of dicts, one per event once picked
        "used_dg_ids": set(), # players you've already used this season
    }
    return state


def list_weeks(state: Dict) -> pd.DataFrame:
    """
    Show the event order for the season in OAD terms.

    Returns DataFrame with:
      week_index, event_id, event_name, event_date
    """
    sched = state["schedule"]
    out = sched[["event_id", "event_name", "event_date"]].copy()
    out.insert(0, "week_index", out.index)
    return out


def build_weekly_context(
    state: Dict,
    week_index: int,
) -> Dict[str, pd.DataFrame]:
    """
    Build the full weekly view (all helper tables) for a single event
    in the current season state.

    This is what you look at to make your pick.

    Parameters
    ----------
    state : dict
        Season state from init_manual_season.
    week_index : int
        Row index into state["schedule"] (0-based, chronological).

    Returns
    -------
    weekly : dict of DataFrames (same structure as build_weekly_view)
    """
    yr = state["season_year"]
    sched = state["schedule"]

    if week_index < 0 or week_index >= len(sched):
        raise IndexError(f"week_index {week_index} out of range 0..{len(sched)-1}")

    ev = sched.iloc[week_index]
    eid = int(ev["event_id"])

    weekly = build_weekly_view(yr, eid)
    return weekly


def apply_manual_pick(
    state: Dict,
    week_index: int,
    dg_id: int,
    weekly: Dict[str, pd.DataFrame],
) -> Tuple[Dict, pd.DataFrame]:
    """
    Apply a manually chosen pick for a given week using the already-built
    weekly context.

    Steps:
      - mark dg_id as used
      - pull EV/current info from weekly's performance/summary tables
      - look up actual Winnings for (year, event_id, dg_id) in Odds_and_Results
      - append a log row with cumulative total

    Returns
    -------
    updated_state, single-row DataFrame of this week's log entry
    """
    yr = state["season_year"]
    sched = state["schedule"]
    odds_df = state["odds"]
    used_dg_ids = state["used_dg_ids"]
    log_list = state["log"]

    if week_index < 0 or week_index >= len(sched):
        raise IndexError(f"week_index {week_index} out of range 0..{len(sched)-1}")

    dg_id = int(dg_id)
    used_dg_ids.add(dg_id)

    ev = sched.iloc[week_index]
    eid = int(ev["event_id"])
    ename = str(ev["event_name"])
    edate = ev["event_date"]

    # We rely on the weekly dict passed in (no re-compute)
    perf = weekly.get("table_performance")
    summary = weekly.get("summary")

    if perf is not None and not perf.empty and "dg_id" in perf.columns:
        row_df = perf[perf["dg_id"] == dg_id]
    else:
        row_df = pd.DataFrame()

    if row_df.empty and summary is not None and not summary.empty and "dg_id" in summary.columns:
        row_df = summary[summary["dg_id"] == dg_id]

    if row_df.empty:
        player_name = None
        ev_current = np.nan
        ev_future_total = np.nan
        decimal_odds = np.nan
    else:
        row = row_df.iloc[0]
        player_name = row.get("player_name", None)
        ev_current = row.get("ev_current", np.nan)
        ev_future_total = row.get("ev_future_total", np.nan)
        decimal_odds = row.get("decimal_odds", np.nan)

    # Actual winnings from Odds_and_Results
    mask = (odds_df["year"] == yr) & (odds_df["event_id"] == eid) & (odds_df["dg_id"] == dg_id)
    od_row = odds_df[mask]
    if od_row.empty:
        winnings = 0.0
    else:
        winnings = float(od_row.iloc[0].get("Winnings", 0.0) or 0.0)

    prev_cum = log_list[-1]["Cumulative"] if log_list else 0.0
    cumulative = prev_cum + winnings

    log_entry = {
        "year": yr,
        "week_index": week_index,
        "event_id": eid,
        "event_name": ename,
        "event_date": edate,
        "dg_id": dg_id,
        "player_name": player_name,
        "decimal_odds": decimal_odds,
        "ev_current": ev_current,
        "ev_future_total": ev_future_total,
        "Winnings": winnings,
        "Cumulative": cumulative,
    }

    log_list.append(log_entry)
    state["used_dg_ids"] = used_dg_ids
    state["log"] = log_list

    return state, pd.DataFrame([log_entry])


def get_log(state: Dict) -> pd.DataFrame:
    """
    Convert state["log"] (list of dicts) into a DataFrame.
    """
    if not state["log"]:
        return pd.DataFrame(
            columns=[
                "year",
                "week_index",
                "event_id",
                "event_name",
                "event_date",
                "dg_id",
                "player_name",
                "decimal_odds",
                "ev_current",
                "ev_future_total",
                "Winnings",
                "Cumulative",
            ]
        )
    return pd.DataFrame(state["log"])