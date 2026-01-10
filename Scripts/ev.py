# Scripts/ev.py
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from Scripts.data_io import load_odds_and_results
from Scripts.schedule import build_season_schedule

from typing import Iterable, Optional


def expected_value_from_odds(decimal_odds: float, payout: float) -> float:
    """
    Simple EV: implied win probability (1 / decimal_odds) times payout.

    payout can be the full purse, winner_share, or any other $-per-win figure.
    """
    if decimal_odds is None or np.isnan(decimal_odds) or decimal_odds <= 0:
        return 0.0
    p = 1.0 / decimal_odds
    return float(p * payout)


def get_latest_same_tier_odds(
    odds_df: pd.DataFrame,
    dg_id: int,
    tier: Optional[str],
    as_of_date,
    use_pre_odds: bool = True,   # kept for compatibility, but ignored
    fallback_odds: float = 1000.0,
) -> float:
    """
    For a given player and event tier, as of a date, return the most recent
    decimal odds from a past event of the same tier.

    NOTE: This version ALWAYS uses close_odds and ignores Pre_Odds.
    """
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = odds_df.copy()
    df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
    df = df[(df["dg_id"] == int(dg_id)) & (df["event_completed"] < ts)].copy()

    if tier is not None and "Event_Tier" in df.columns:
        df = df[df["Event_Tier"].astype(str).str.upper() == str(tier).upper()]

    if df.empty or "close_odds" not in df.columns:
        return float(fallback_odds)

    df["close_odds"] = pd.to_numeric(df["close_odds"], errors="coerce")
    df = df.dropna(subset=["close_odds"])
    if df.empty:
        return float(fallback_odds)

    latest = df.sort_values("event_completed").iloc[-1]
    odds_val = latest["close_odds"]
    if odds_val is None or np.isnan(odds_val) or odds_val <= 0:
        return float(fallback_odds)
    return float(odds_val)


def compute_current_event_ev(
    odds_df: pd.DataFrame,
    event_id: int,
    purse: float,
    use_pre_odds: bool = False,
) -> pd.DataFrame:
    """
    Compute EV for the current event using decimal odds and TOTAL PURSE.

    EV_current = (1 / decimal_odds) * purse
    """
    eid = int(event_id)

    sub = odds_df[odds_df["event_id"] == eid].copy()
    if sub.empty:
        return pd.DataFrame(columns=["dg_id", "decimal_odds", "ev_current"])

    # choose which odds column to use
    odds_col = "Pre_Odds" if use_pre_odds and "Pre_Odds" in sub.columns else "close_odds"
    sub[odds_col] = pd.to_numeric(sub[odds_col], errors="coerce")

    # treat the odds as decimal directly
    sub["decimal_odds"] = sub[odds_col]

    # guard against bad odds
    sub.loc[sub["decimal_odds"] <= 0, "decimal_odds"] = pd.NA

    # implied probability
    sub["p_win"] = 1.0 / sub["decimal_odds"]
    sub["ev_current"] = sub["p_win"] * float(purse)

    out = sub[["dg_id", "decimal_odds", "ev_current"]].copy()
    out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce").astype("Int64")

    return out


# simple convenience wrapper if you just want EV for one event from disk
def compute_current_event_ev_from_disk(
    event_id: int,
    winner_share: float,
    use_pre_odds: bool = True,
) -> pd.DataFrame:
    odds_df = load_odds_and_results()
    return compute_current_event_ev(odds_df, event_id, winner_share, use_pre_odds)

from Scripts.schedule import build_season_schedule


from Scripts.data_io import load_odds_and_results
from Scripts.schedule import build_season_schedule


def compute_future_ev_for_players(
    odds_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    season_year: int,
    as_of_date,
    dg_ids: Iterable[int],
    use_pre_odds: bool = False,   # kept for API compatibility, but ignored
    fallback_odds: float = 1000.0,
) -> pd.DataFrame:
    """
    Advanced-ish future EV:

      - For each future event in the schedule (same season, event_date > as_of_date):
          * Determine its Event_Tier (REGULAR / SIGNATURE / MAJOR / PLAYOFF, etc.)
          * For each player in dg_ids:
              - Use the most recent same-tier decimal odds from odds_df
                (based on close_odds, as-of as_of_date).
              - If no same-tier history exists, fall back to the player's most
                recent odds from ANY tier.
              - If still nothing, use fallback_odds (e.g. 1000).

      - EV_future = (1 / decimal_odds_proxy) * purse_future_event

    Returns one row per (dg_id, event_id) with:
        ['dg_id', 'event_id', 'event_date',
         'Event_Tier', 'decimal_odds_proxy', 'ev_future']
    """
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    yr = int(season_year)
    dg_ids_arr = pd.Series(list(dg_ids)).dropna()
    if dg_ids_arr.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "event_id",
                "event_date",
                "Event_Tier",
                "decimal_odds_proxy",
                "ev_future",
            ]
        )

    dg_ids_arr = dg_ids_arr.astype("Int64").unique()

    # ------------------------------------------------------------------
    # Clean odds_df: use close_odds as decimal odds, filter to history
    # ------------------------------------------------------------------
    odds = odds_df.copy()

    # enforce basic numeric types
    for col in ("year", "event_id", "dg_id"):
        if col not in odds.columns:
            raise ValueError(f"odds_df must contain '{col}' column.")
        odds[col] = pd.to_numeric(odds[col], errors="coerce").astype("Int64")

    odds["event_completed"] = pd.to_datetime(
        odds.get("event_completed"), errors="coerce"
    )

    # Event_Tier: trust what's in odds if present, else default REGULAR
    if "Event_Tier" in odds.columns:
        odds["Event_Tier"] = (
            odds["Event_Tier"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        odds["Event_Tier"] = "REGULAR"

    # Always use close_odds as the decimal odds source
    if "close_odds" not in odds.columns:
        raise ValueError("odds_df must contain 'close_odds' for future EV.")
    odds["close_odds"] = pd.to_numeric(odds["close_odds"], errors="coerce")

    odds = odds.dropna(subset=["close_odds", "event_completed"])
    odds = odds[(odds["year"] <= yr) & (odds["event_completed"] < ts)].copy()

    if odds.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "event_id",
                "event_date",
                "Event_Tier",
                "decimal_odds_proxy",
                "ev_future",
            ]
        )

    odds["decimal_odds"] = odds["close_odds"]

    # ------------------------------------------------------------------
    # Historical odds:
    #   - last odds per (dg_id, Event_Tier)
    #   - last odds per dg_id across ALL tiers (fallback)
    # ------------------------------------------------------------------
    odds = odds.sort_values("event_completed")

    hist_by_tier = (
        odds.groupby(["dg_id", "Event_Tier"], as_index=False)
        .tail(1)[["dg_id", "Event_Tier", "decimal_odds"]]
    )

    hist_any = (
        odds.groupby("dg_id", as_index=False)
        .tail(1)[["dg_id", "decimal_odds"]]
        .rename(columns={"decimal_odds": "decimal_odds_any_tier"})
    )

    # ------------------------------------------------------------------
    # Future schedule: events after as_of_date in this season
    # ------------------------------------------------------------------
    sched = schedule_df.copy()

    if "event_date" not in sched.columns:
        raise ValueError("schedule_df must contain 'event_date' column.")
    sched["event_date"] = pd.to_datetime(sched["event_date"], errors="coerce")

    if "year" not in sched.columns:
        if "season" in sched.columns:
            sched["year"] = pd.to_numeric(sched["season"], errors="coerce")
        else:
            raise ValueError("schedule_df must contain 'year' or 'season' column.")

    sched["year"] = pd.to_numeric(sched["year"], errors="coerce")
    sched["event_id"] = pd.to_numeric(sched["event_id"], errors="coerce").astype("Int64")

    future_events = sched[(sched["year"] == yr) & (sched["event_date"] > ts)].copy()
    if future_events.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "event_id",
                "event_date",
                "Event_Tier",
                "decimal_odds_proxy",
                "ev_future",
            ]
        )

    # Event tier in schedule
    if "Event_Tier" in future_events.columns:
        future_events["Event_Tier"] = (
            future_events["Event_Tier"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    elif "event_tier" in future_events.columns:
        future_events["Event_Tier"] = (
            future_events["event_tier"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        future_events["Event_Tier"] = "REGULAR"

    # Purse as numeric
    if "purse" not in future_events.columns:
        raise ValueError("schedule_df must contain 'purse' column for future EV.")

    purse_raw = future_events["purse"].astype(str)
    purse_clean = (
        purse_raw.str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    future_events["purse_num"] = pd.to_numeric(purse_clean, errors="coerce")

    # ------------------------------------------------------------------
    # Cartesian product: players × future events
    # ------------------------------------------------------------------
    players_df = pd.DataFrame({"dg_id": dg_ids_arr})
    fev = future_events[
        ["event_id", "event_date", "Event_Tier", "purse_num"]
    ].copy()

    cart = (
        players_df.assign(key=1)
        .merge(fev.assign(key=1), on="key", how="left")
        .drop(columns="key")
    )

    # ------------------------------------------------------------------
    # Attach historical odds:
    #   - same-tier first
    #   - any-tier fallback
    # ------------------------------------------------------------------
    cart = cart.merge(
        hist_by_tier,
        on=["dg_id", "Event_Tier"],
        how="left",
    )
    cart = cart.merge(
        hist_any,
        on="dg_id",
        how="left",
    )

    # Choose decimal_odds_proxy with fallbacks
    cart["decimal_odds_proxy"] = cart["decimal_odds"]

    missing_mask = cart["decimal_odds_proxy"].isna()
    if missing_mask.any():
        cart.loc[missing_mask, "decimal_odds_proxy"] = cart.loc[
            missing_mask, "decimal_odds_any_tier"
        ]

    cart["decimal_odds_proxy"] = cart["decimal_odds_proxy"].fillna(float(fallback_odds))

    # ------------------------------------------------------------------
    # EV_future = implied win prob × purse
    # ------------------------------------------------------------------
    cart["p_win_future"] = 1.0 / cart["decimal_odds_proxy"]
    cart["ev_future"] = cart["p_win_future"] * cart["purse_num"].astype(float)

    return cart[
        ["dg_id", "event_id", "event_date", "Event_Tier", "decimal_odds_proxy", "ev_future"]
    ].copy()


def compute_future_ev_for_players_from_disk(
    season_year: int,
    as_of_date,
    dg_ids: Iterable[int],
    use_pre_odds: bool = False,
    fallback_odds: float = 1000.0,
) -> pd.DataFrame:
    """
    Convenience wrapper: load odds and schedule from disk and compute future EV
    using close_odds-only logic and purse-based EV.

    Parameters
    ----------
    season_year : int
        Season to consider (e.g. 2024).
    as_of_date : str or Timestamp
        Date cutoff (events strictly after this date are "future").
    dg_ids : iterable of int
        Players to compute future EV for.
    use_pre_odds : bool
        Kept only for compatibility; ignored (we always use close_odds).
    fallback_odds : float
        Odds to use if a player has no usable historical odds.

    Returns
    -------
    DataFrame as from compute_future_ev_for_players().
    """
    odds_df = load_odds_and_results()
    schedule_df = build_season_schedule(season_year)
    return compute_future_ev_for_players(
        odds_df=odds_df,
        schedule_df=schedule_df,
        season_year=season_year,
        as_of_date=as_of_date,
        dg_ids=dg_ids,
        use_pre_odds=use_pre_odds,
        fallback_odds=fallback_odds,
    )
