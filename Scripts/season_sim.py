# Scripts/season_sim.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from Scripts.schedule import build_season_schedule
from Scripts.data_io import load_odds_and_results
from Scripts.weekly_view import build_weekly_view


def simple_ev_max_strategy(
    weekly: Dict[str, pd.DataFrame],
    used_dg_ids: set[int],
) -> Optional[int]:
    """
    Simple baseline strategy:
      - Look at table_performance
      - Remove already-used players
      - Pick the player with the highest ev_current

    Returns
    -------
    dg_id or None if no valid candidate.
    """
    perf = weekly.get("table_performance")
    if perf is None or perf.empty or "dg_id" not in perf.columns:
        return None

    df = perf.copy()
    df = df[~df["dg_id"].isin(used_dg_ids)]
    df = df.dropna(subset=["ev_current"])

    if df.empty:
        return None

    df = df.sort_values("ev_current", ascending=False)
    return int(df.iloc[0]["dg_id"])


StrategyFn = Callable[[Dict[str, pd.DataFrame], set[int]], Optional[int]]


def simulate_season(
    season_year: int,
    strategy_fn: StrategyFn = simple_ev_max_strategy,
) -> Dict[str, pd.DataFrame]:
    """
    Auto-simulate an OAD season:

      For each event:
        1) Build weekly view (all helper tables)
        2) Use strategy_fn to pick a dg_id not yet used
        3) Look up actual Winnings for that player at that event
        4) Track cumulative earnings

    Returns
    -------
    dict with:
      - 'log': per-event pick log
      - 'schedule': schedule used
      - 'odds': odds_and_results df (for reference)
    """
    yr = int(season_year)

    schedule_df = build_season_schedule(yr).copy()
    schedule_df = schedule_df.sort_values("event_date")

    odds_df = load_odds_and_results().copy()
    for col in ["year", "event_id", "dg_id"]:
        if col in odds_df.columns:
            odds_df[col] = pd.to_numeric(odds_df[col], errors="coerce")

    used_dg_ids: set[int] = set()
    log_rows: List[dict] = []

    for _, ev in schedule_df.iterrows():
        eid = int(ev["event_id"])
        ename = str(ev["event_name"])
        edate = ev["event_date"]

        # 1) weekly context
        weekly = build_weekly_view(yr, eid)

        # 2) auto pick
        pick_dg_id = strategy_fn(weekly, used_dg_ids)
        if pick_dg_id is None:
            log_rows.append(
                {
                    "year": yr,
                    "event_id": eid,
                    "event_name": ename,
                    "event_date": edate,
                    "dg_id": np.nan,
                    "player_name": None,
                    "decimal_odds": np.nan,
                    "ev_current": np.nan,
                    "ev_future_total": np.nan,
                    "Winnings": 0.0,
                    "Cumulative": log_rows[-1]["Cumulative"] if log_rows else 0.0,
                    "note": "no_pick",
                }
            )
            continue

        used_dg_ids.add(pick_dg_id)

        perf = weekly.get("table_performance")
        summary = weekly.get("summary")

        if perf is not None and not perf.empty and "dg_id" in perf.columns:
            row_df = perf[perf["dg_id"] == pick_dg_id]
        else:
            row_df = pd.DataFrame()

        if row_df.empty and summary is not None and not summary.empty and "dg_id" in summary.columns:
            row_df = summary[summary["dg_id"] == pick_dg_id]

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

        mask = (
            (odds_df["year"] == yr)
            & (odds_df["event_id"] == eid)
            & (odds_df["dg_id"] == pick_dg_id)
        )
        od_row = odds_df[mask]
        if od_row.empty:
            winnings = 0.0
        else:
            winnings = float(od_row.iloc[0].get("Winnings", 0.0) or 0.0)

        prev_cum = log_rows[-1]["Cumulative"] if log_rows else 0.0
        cumulative = prev_cum + winnings

        log_rows.append(
            {
                "year": yr,
                "event_id": eid,
                "event_name": ename,
                "event_date": edate,
                "dg_id": pick_dg_id,
                "player_name": player_name,
                "decimal_odds": decimal_odds,
                "ev_current": ev_current,
                "ev_future_total": ev_future_total,
                "Winnings": winnings,
                "Cumulative": cumulative,
                "note": "",
            }
        )

    log_df = pd.DataFrame(log_rows)
    return {
        "log": log_df,
        "schedule": schedule_df,
        "odds": odds_df,
    }