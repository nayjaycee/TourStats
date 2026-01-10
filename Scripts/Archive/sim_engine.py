# sim_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work as package AND when run directly
# ------------------------------------------------------------
try:
    from .config import LIVE_OAD_SEASON
    from .ev_utils import build_week_ev_table
except ImportError:
    from Scripts.Archive.config import LIVE_OAD_SEASON
    from Scripts.Archive.ev_utils import build_week_ev_table

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WeekPick:
    year: int
    week_index: int
    event_id: int
    event_name: str
    tour: str
    start_date: pd.Timestamp
    dg_id: int
    player_name: Optional[str]
    pick_reason: str
    pick_metric: str
    pick_value: float
    already_used: bool = False


# ============================================================
# PICK LOGIC
# ============================================================

def choose_pick_from_week(
    week_full: pd.DataFrame,
    week_index: int,
    pick_col: str = "ev_total",
    shortlist_ids: Optional[Sequence[int]] = None,
    used_ids: Optional[Iterable[int]] = None,
    allow_reuse: bool = False,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> Tuple[WeekPick, pd.DataFrame]:
    """
    Choose a single player for a week from the detailed EV table.

    week_full must contain at least:
      ['year', 'week_index', 'event_id', 'event_name', 'tour',
       'start_date', 'dg_id', pick_col]
    and optionally 'player_name' and 'rank' / 'model_rank'.
    """
    df = week_full.copy()

    if df.empty:
        raise ValueError(f"Week {week_index}: week_full is empty; cannot choose a pick.")

    # ---- determine which rank column to use (if any) ----
    rank_col = None
    for candidate in ("model_rank", "rank"):
        if candidate in df.columns:
            rank_col = candidate
            break

    # ---- candidate filter: shortlist ----
    if shortlist_ids:
        shortlist_ids = [int(x) for x in shortlist_ids]
        df = df[df["dg_id"].isin(shortlist_ids)]

    # ---- candidate filter: used IDs ----
    if not allow_reuse and used_ids is not None:
        used_set = {int(x) for x in used_ids}
        df = df[~df["dg_id"].isin(used_set)]

    # ---- candidate filter: rank window ----
    if rank_col is not None:
        if min_rank is not None:
            df = df[df[rank_col] >= min_rank]
        if max_rank is not None:
            df = df[df[rank_col] <= max_rank]

    if df.empty:
        raise ValueError(f"Week {week_index}: no candidates left after filters.")

    # ---- determine pick metric column ----
    pick_metric = pick_col
    if pick_metric not in df.columns:
        for fallback in ("ev_money", "ev", "ev_win"):
            if fallback in df.columns:
                pick_metric = fallback
                break
        else:
            raise ValueError(
                f"Week {week_index}: pick_col '{pick_col}' not found and no fallback "
                "EV column ('ev_money', 'ev', 'ev_win') present."
            )

    # ---- choose best row by pick metric ----
    df = df.sort_values(by=pick_metric, ascending=False).reset_index(drop=True)
    best = df.iloc[0]

    pick_year = int(best["year"]) if "year" in best.index and pd.notna(best["year"]) else int(LIVE_OAD_SEASON)

    pick = WeekPick(
        year=pick_year,
        week_index=int(best.get("week_index", week_index)),
        event_id=int(best["event_id"]),
        event_name=str(best["event_name"]),
        tour=str(best.get("tour", "")),
        start_date=pd.to_datetime(best.get("start_date")),
        dg_id=int(best["dg_id"]),
        player_name=str(best["player_name"]) if "player_name" in best.index else None,
        pick_reason=f"max_{pick_metric}",
        pick_metric=pick_metric,
        pick_value=float(best[pick_metric]),
        already_used=bool(
            used_ids is not None and int(best["dg_id"]) in {int(x) for x in used_ids}
        ),
    )

    return pick, df


# ============================================================
# RESULT LOOKUP
# ============================================================

def _lookup_realized_result(
    master: pd.DataFrame,
    year: int,
    event_id: int,
    dg_id: int,
) -> Dict[str, object]:
    """
    Join realized results from the master results table for a given player-event.

    Only fills fields that actually exist in master, everything else -> NaN.
    """
    cols_of_interest = [
        "finish_position",
        "finish_rank",
        "money_earned",
        "official_money",
        "fedex_points",
        "owgr_points",
    ]

    # 'year' is optional; if missing, we only filter on event_id + dg_id
    if "year" in master.columns:
        mask = (
            (master["year"] == year)
            & (master["event_id"] == event_id)
            & (master["dg_id"] == dg_id)
        )
    else:
        mask = (
            (master["event_id"] == event_id)
            & (master["dg_id"] == dg_id)
        )

    subset = master.loc[mask]

    if subset.empty:
        return {c: np.nan for c in cols_of_interest}

    row = subset.iloc[0]

    out: Dict[str, object] = {}
    for c in cols_of_interest:
        out[c] = row[c] if c in subset.columns else np.nan

    return out


# ============================================================
# SEASON SIMULATION
# ============================================================

def simulate_season(
    env: Dict[str, pd.DataFrame],
    odds: pd.DataFrame,
    master: pd.DataFrame,
    week_indices: Optional[Sequence[int]] = None,
    shortlist_ids: Optional[Sequence[int]] = None,
    allow_reuse: bool = False,
    pick_col: str = "ev_total",
    top_k: int = 30,
    include_future_ev: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a full-season One-and-Done simulation using the EV engine.

    env must contain at least:
      - 'sched': schedule with week_index, event_id, event_name, start_date, tour
      plus whatever build_week_ev_table requires (event_skill, rounds, etc.).
    """
    sched = env.get("sched")
    if sched is None:
        raise ValueError("env must contain a 'sched' DataFrame.")

    if week_indices is None:
        week_indices = sorted(sched["week_index"].unique())

    used_ids: List[int] = []
    picks: List[Dict[str, object]] = []

    for wk in week_indices:
        if verbose:
            print(f"\n=== Week {wk} ===")

        # Build EV table for this week
        week_full, week_view = build_week_ev_table(
            env=env,
            odds=odds,
            master=master,
            week_index=wk,
            shortlist_ids=shortlist_ids,
            exclude_used=not allow_reuse,
            top_k=top_k,
            include_future_ev=include_future_ev,
        )

        # Choose pick
        pick, filtered_week = choose_pick_from_week(
            week_full=week_full,
            week_index=wk,
            pick_col=pick_col,
            shortlist_ids=shortlist_ids,
            used_ids=used_ids,
            allow_reuse=allow_reuse,
        )

        used_ids.append(pick.dg_id)

        # Pull realized result if available in master
        realized = _lookup_realized_result(
            master=master,
            year=pick.year,
            event_id=pick.event_id,
            dg_id=pick.dg_id,
        )

        row: Dict[str, object] = {
            "year": pick.year,
            "week_index": pick.week_index,
            "event_id": pick.event_id,
            "event_name": pick.event_name,
            "tour": pick.tour,
            "start_date": pick.start_date,
            "dg_id": pick.dg_id,
            "player_name": pick.player_name,
            "pick_metric": pick.pick_metric,
            "pick_value": pick.pick_value,
            "pick_reason": pick.pick_reason,
            "already_used": pick.already_used,
        }
        row.update(realized)

        picks.append(row)

        if verbose:
            money = realized.get("money_earned", np.nan)
            print(
                f"Pick: {pick.player_name} (dg_id={pick.dg_id}) | "
                f"{pick.pick_metric}={pick.pick_value:.1f} | "
                f"money_earned={money}"
            )

    picks_df = (
        pd.DataFrame(picks)
        .sort_values(["year", "week_index"])
        .reset_index(drop=True)
    )

    # For now log_df == picks_df; can expand later
    log_df = picks_df.copy()

    return picks_df, log_df


# ============================================================
# CONVENIENCE WRAPPER
# ============================================================

def run_default_sim(
    env: Dict[str, pd.DataFrame],
    odds: pd.DataFrame,
    master: pd.DataFrame,
    week_start: int = 1,
    week_end: Optional[int] = None,
    shortlist_ids: Optional[Sequence[int]] = None,
    allow_reuse: bool = False,
    pick_col: str = "ev_total",
    top_k: int = 30,
    include_future_ev: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for simulate_season that infers week_indices
    from env['sched'] and returns only the picks DataFrame.
    """
    sched = env.get("sched")
    if sched is None:
        raise ValueError("env must contain a 'sched' DataFrame.")

    all_weeks = sorted(sched["week_index"].unique())
    if week_end is None:
        week_end = max(all_weeks)

    week_indices = [w for w in all_weeks if week_start <= w <= week_end]

    picks_df, _ = simulate_season(
        env=env,
        odds=odds,
        master=master,
        week_indices=week_indices,
        shortlist_ids=shortlist_ids,
        allow_reuse=allow_reuse,
        pick_col=pick_col,
        top_k=top_k,
        include_future_ev=include_future_ev,
        verbose=verbose,
    )

    return picks_df