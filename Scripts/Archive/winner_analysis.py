# Scripts/winner_analysis.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import pandas as pd

try:
    from .data_loading import load_combined_rounds
    from .schedule_utils import load_event_schedule
except ImportError:
    from Scripts.Archive.data_loading import load_combined_rounds
    from Scripts.Archive.schedule_utils import load_event_schedule


@dataclass
class WinnerPatternConfig:
    lookback_seasons: int = 5
    min_events_for_pattern: int = 5
    min_top5s_for_stat_pattern: int = 3


def load_winner_analysis_env() -> Dict[str, pd.DataFrame]:
    combined = load_combined_rounds(copy=False)
    sched = load_event_schedule()
    return {"combined": combined, "sched": sched}


def shortlist_players_matching_patterns(
    patterns_df: pd.DataFrame,
    week_full: pd.DataFrame,
    event_id: int,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    PLACEHOLDER.

    Eventually: given
      - patterns_df: per-event winner/top-5 pattern summary
      - week_full: detailed EV table for this event
    return a shortlist of players whose profile matches the
    historical winner/top-5 patterns.

    For now just returns the top_n by ev_total so the call
    site doesn't break.
    """
    df = week_full.copy()
    if "ev_total" in df.columns:
        df = df.sort_values("ev_total", ascending=False)
    if top_n is not None and top_n > 0:
        df = df.head(top_n)
    return df.reset_index(drop=True)