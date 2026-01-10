# Scripts/field_sim.py
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from Scripts.data_io import load_odds_and_results
from Scripts.schedule import build_season_schedule


def get_actual_field(
    odds_df: pd.DataFrame,
    season_year: int,
    event_id: int,
) -> pd.DataFrame:
    """
    For backtesting 2024/2025:
      Return the actual field for (year, event_id) based on Odds_and_Results.

    Parameters
    ----------
    odds_df : DataFrame
        Loaded via load_odds_and_results().
    season_year : int
        Year (e.g., 2024).
    event_id : int
        Event identifier.

    Returns
    -------
    DataFrame with at least:
      ['year', 'event_id', 'dg_id', 'player_name', 'Event_Tier', 'close_odds']
    """
    yr = int(season_year)
    eid = int(event_id)

    df = odds_df.copy()
    for col in ["year", "event_id", "dg_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = (df["year"] == yr) & (df["event_id"] == eid)
    field = df[mask].copy()

    if field.empty:
        return pd.DataFrame(
            columns=["year", "event_id", "dg_id", "player_name", "Event_Tier", "close_odds"]
        )

    keep_cols = []
    for col in [
        "year",
        "event_id",
        "dg_id",
        "player_name",
        "Event_Tier",
        "close_odds",
    ]:
        if col in field.columns:
            keep_cols.append(col)

    field = field[keep_cols].drop_duplicates(subset=["dg_id"])

    return field


def _odds_up_to(odds_df: pd.DataFrame, as_of_date) -> pd.DataFrame:
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = odds_df.copy()
    df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
    return df[df["event_completed"] < ts].copy()


def simulate_signature_field(
    odds_up_to: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    """
    Signature event field:
      1) If played THIS event since 2021 -> include.
      2) Else, if played ANY signature event -> include.
    """
    eid = int(event_id)

    df = odds_up_to.copy()
    for col in ["year", "event_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sig_mask = df["Event_Tier"].astype(str).str.upper() == "SIGNATURE"
    df_sig = df[sig_mask].copy()

    # 1) Played this event since 2021
    hist_this = df_sig[(df_sig["event_id"] == eid) & (df_sig["year"] >= 2021)]

    # 2) Played any signature event
    hist_any_sig = df_sig[df_sig["year"] >= 2021]

    dg_ids_this = set(hist_this["dg_id"].unique().tolist())
    dg_ids_any = set(hist_any_sig["dg_id"].unique().tolist())

    pool_ids = dg_ids_this | dg_ids_any
    if not pool_ids:
        return pd.DataFrame(columns=["dg_id", "player_name", "source"])

    names = (
        df[df["dg_id"].isin(pool_ids)]
        .sort_values("event_completed")
        .groupby("dg_id", as_index=False)
        .agg(player_name=("player_name", "last"))
    )

    names["source"] = np.where(
        names["dg_id"].isin(dg_ids_this),
        "this_sig_or_prior",
        "other_sig_only",
    )
    return names


def simulate_major_field(
    odds_up_to: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    """
    Major event field:
      1) If played THIS major -> include.
      2) Else, if played >1 signature event -> include.
    """
    eid = int(event_id)

    df = odds_up_to.copy()
    for col in ["year", "event_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) Played this major
    hist_this = df[(df["event_id"] == eid)].copy()

    # 2) >1 signature event
    sig_mask = df["Event_Tier"].astype(str).str.upper() == "SIGNATURE"
    df_sig = df[sig_mask].copy()
    sig_counts = (
        df_sig.groupby("dg_id", as_index=False)["event_id"]
        .nunique()
        .rename(columns={"event_id": "n_sig_events"})
    )
    big_sig = sig_counts[sig_counts["n_sig_events"] > 1]

    dg_ids_this = set(hist_this["dg_id"].unique().tolist())
    dg_ids_sig = set(big_sig["dg_id"].unique().tolist())
    pool_ids = dg_ids_this | dg_ids_sig

    if not pool_ids:
        return pd.DataFrame(columns=["dg_id", "player_name", "source"])

    names = (
        df[df["dg_id"].isin(pool_ids)]
        .sort_values("event_completed")
        .groupby("dg_id", as_index=False)
        .agg(player_name=("player_name", "last"))
    )

    names["source"] = np.where(
        names["dg_id"].isin(dg_ids_this),
        "this_major_or_prior",
        "sig_qual_major",
    )
    return names


def simulate_regular_field(
    odds_up_to: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    """
    Regular event field:
      1) If played THIS event before -> include.
    """
    eid = int(event_id)
    df = odds_up_to.copy()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")

    hist_this = df[df["event_id"] == eid].copy()
    if hist_this.empty:
        return pd.DataFrame(columns=["dg_id", "player_name", "source"])

    names = (
        hist_this.sort_values("event_completed")
        .groupby("dg_id", as_index=False)
        .agg(player_name=("player_name", "last"))
    )
    names["source"] = "this_regular_prior"
    return names


def simulate_field_for_event(
    season_year: int,
    event_id: int,
    as_of_date,
    odds_df: Optional[pd.DataFrame] = None,
    schedule_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Simulated field for an event as of as_of_date, using your rules.

    Parameters
    ----------
    season_year : int
        Season year (e.g., 2024).
    event_id : int
        Event id.
    as_of_date : datetime-like
        Only historical data with event_completed < as_of_date are used.
    odds_df : DataFrame, optional
        If None, will load from disk via load_odds_and_results().
    schedule_df : DataFrame, optional
        If None, will build via build_season_schedule(season_year).

    Returns
    -------
    DataFrame with ['dg_id', 'player_name', 'source', 'Event_Tier']
    """
    yr = int(season_year)
    eid = int(event_id)

    if odds_df is None:
        odds_df = load_odds_and_results()
    if schedule_df is None:
        schedule_df = build_season_schedule(yr)

    # Identify tier from schedule (preferred) or from odds
    sched_row = schedule_df[schedule_df["event_id"] == eid]
    if sched_row.empty:
        raise ValueError(f"Event_id {eid} not found in schedule for year {yr}.")

    tier = str(sched_row.iloc[0]["Event_Tier"]).upper() if "Event_Tier" in sched_row.columns else None

    odds_season = odds_df.copy()
    for col in ["year", "event_id"]:
        odds_season[col] = pd.to_numeric(odds_season[col], errors="coerce")

    odds_season = odds_season[odds_season["year"] <= yr].copy()
    up_to = _odds_up_to(odds_season, as_of_date)

    tier = tier or "REGULAR"  # default if missing
    if tier == "SIGNATURE":
        sim_field = simulate_signature_field(up_to, eid)
    elif tier == "MAJOR":
        sim_field = simulate_major_field(up_to, eid)
    else:
        sim_field = simulate_regular_field(up_to, eid)

    if sim_field.empty:
        sim_field["dg_id"] = sim_field.get("dg_id", pd.Series(dtype="Int64"))
        sim_field["player_name"] = sim_field.get("player_name", pd.Series(dtype="string"))
        sim_field["source"] = sim_field.get("source", pd.Series(dtype="string"))

    sim_field["Event_Tier"] = tier
    return sim_field
