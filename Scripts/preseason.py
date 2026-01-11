# Scripts/preseason.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List, Dict

import numpy as np
import pandas as pd

from Scripts.data_io import (
    load_preseason,
    load_odds_and_results,
    load_rounds,
    load_oad_schedule,
)
from Scripts.config import PRESEASON_TEMPLATE, PRESEASON_SHORTLIST_TEMPLATE


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _choose_decayed_sg_col(df: pd.DataFrame) -> str:
    """
    Decide which column to treat as 'weighted strokes gained decayed'
    in preseason_YYYY.

    Preference order based on what you've described / shown:
      1) dec_sg_total_field
      2) dec_wgh_sg
      3) dec_sg_total
    """
    candidates = [
        "dec_sg_total_field",
        "dec_wgh_sg",
        "dec_sg_total",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a decayed SG column in preseason df. "
        f"Columns: {df.columns.tolist()}"
    )


def _get_liv_players_for_season(season: int) -> set[int]:
    """
    Return set of dg_id that should be treated as 'LIV' players
    for a given target season.

    Rule: a player is LIV in season Y if they have at least one start
    where tour == 'LIV' in any year < Y.
    """
    rounds = load_rounds()

    rounds["tour"] = rounds["tour"].astype(str).str.upper()
    rounds["year"] = pd.to_numeric(rounds["year"], errors="coerce")
    rounds["dg_id"] = pd.to_numeric(rounds["dg_id"], errors="coerce")

    hist = rounds[(rounds["year"] < season) & (rounds["tour"] == "LIV")].copy()
    if hist.empty:
        return set()

    liv_ids = (
        hist["dg_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return set(liv_ids)


def _fmt_int(val, default="NA") -> str:
    """Safely format a value as int-like string, or default if NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return str(int(val))
    except (TypeError, ValueError):
        return default


def _choose_star_power_column(longlist: pd.DataFrame) -> str:
    """
    Try to find the best 'star power' column in the preseason longlist.
    Preference order: dec_wgh_sg, dec_sg_total_field, dec_sg_total, dec_raw_sg, raw_sg.
    """
    candidates = [
        "dec_wgh_sg",
        "dec_sg_total_field",
        "dec_sg_total",
        "dec_raw_sg",
        "wgh_sg",
        "raw_sg",
    ]
    for col in candidates:
        if col in longlist.columns:
            return col
    raise ValueError(
        f"Could not find any star power column among: {candidates} "
        f"in longlist columns {list(longlist.columns)}"
    )


# ---------------------------------------------------------------------
# 1) Preseason longlist utilities
# ---------------------------------------------------------------------
def build_preseason_longlist_topN(
    season: int,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    For a given season, load preseason_YYYY.csv and return the
    top N players by 'weighted strokes gained decayed'.

    Output columns (no names):
      ['dg_id', 'hist_rounds', decayed_sg_col, 'is_liv', 'target_season']
    """
    season = int(season)
    df = load_preseason(season).copy()

    # Normalize rounds column name
    if "hist_rounds" not in df.columns and "total_rounds_hist" in df.columns:
        df = df.rename(columns={"total_rounds_hist": "hist_rounds"})

    dec_col = _choose_decayed_sg_col(df)

    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df.dropna(subset=["dg_id"]).copy()
    df["dg_id"] = df["dg_id"].astype(int)

    df = df.sort_values(dec_col, ascending=False).head(top_n).copy()

    # Attach LIV flag
    liv_set = _get_liv_players_for_season(season)
    df["is_liv"] = df["dg_id"].astype(int).isin(liv_set)

    keep_cols = [
        col
        for col in [
            "dg_id",
            "hist_rounds",
            dec_col,
            "is_liv",
            "target_season",
        ]
        if col in df.columns
    ]
    longlist = df[keep_cols].reset_index(drop=True)

    return longlist


# ---------------------------------------------------------------------
# 2) Preseason baseline windows for longlist
# ---------------------------------------------------------------------
def compute_preseason_baseline_windows_for_longlist(
    season: int,
    longlist: pd.DataFrame,
    windows: Iterable[int] = (40, 24, 12),
) -> pd.DataFrame:
    """
    For each player in the preseason longlist, compute pre-season baseline
    L40/L24/L12 SG components, using rounds BEFORE the first OAD event
    start_date in that season.

    Returns a wide DataFrame with one row per dg_id and columns like:
      sg_ott_L40, sg_app_L40, sg_arg_L40, sg_putt_L40, sg_total_L40, n_L40,
      sg_ott_L24, ..., sg_total_L24, n_L24, ...
      rank_total_L40, rank_total_L24, rank_total_L12 (within longlist)
    """
    season = int(season)
    windows = list(windows)

    rounds_df = load_rounds()
    sched = load_oad_schedule(season)
    cutoff_date = sched["start_date"].min()

    dg_ids = longlist["dg_id"].dropna().astype(int).unique().tolist()

    df = rounds_df.copy()
    if "round_date" not in df.columns:
        raise ValueError("round_date column missing in combined rounds.")
    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    df = df[(df["round_date"] < cutoff_date) & (df["dg_id"].isin(dg_ids))]

    for col in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dg_id"])

    df = df.sort_values(
        ["dg_id", "round_date", "event_id", "round_num"],
        ascending=[True, True, True, True],
    )

    records: List[Dict] = []
    for dg_id, g in df.groupby("dg_id"):
        g = g.sort_values("round_date", ascending=False)
        rec = {"dg_id": int(dg_id)}

        for w in windows:
            g_w = g.head(w).copy()
            n = len(g_w)
            rec[f"n_L{w}"] = n
            if n == 0:
                for stat in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]:
                    rec[f"{stat}_L{w}"] = np.nan
                continue

            for stat in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]:
                if stat in g_w.columns:
                    rec[f"{stat}_L{w}"] = g_w[stat].mean()
                else:
                    rec[f"{stat}_L{w}"] = np.nan

        records.append(rec)

    base_df = pd.DataFrame(records)
    if base_df.empty:
        return pd.DataFrame(columns=["dg_id"])

    n_players = len(base_df)
    for w in windows:
        col = f"sg_total_L{w}"
        if col in base_df.columns:
            base_df[f"rank_total_L{w}"] = base_df[col].rank(
                ascending=False, method="min"
            )
            base_df[f"rank_total_L{w}_denom"] = n_players

    return base_df


# ---------------------------------------------------------------------
# 3) Big-event history from rounds (MAJOR + SIGNATURE, no playoffs, grouped)
# ---------------------------------------------------------------------
def compute_big_event_history_from_rounds(
    season: int,
    dg_ids: Iterable[int],
    min_year: int = 2017,
) -> pd.DataFrame:
    """
    Build multi-year big-event history (MAJOR + SIGNATURE) using only rounds
    for performance + finishes, and Odds only to determine which event_ids
    belong to MAJOR or SIGNATURE tiers.

    Aggregates STRICTLY by (dg_id, event_id) and assigns the most recent
    event_name available before the target season.

    Excludes Sentry TOC (16), FedEx St. Jude / playoff 27, BMW (28).
    """
    season = int(season)
    dg_ids = list({int(x) for x in dg_ids})
    if not dg_ids:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "event_id",
                "event_name",
                "Event_Tier",
                "starts",
                "cuts_made",
                "top5",
                "top10",
                "top25",
                "best_finish",
                "worst_finish",
                "avg_finish",
                "avg_sg_event",
            ]
        )

    # --- 1) Identify MAJOR/SIGNATURE event_ids from Odds (tier info only) ---
    odds = load_odds_and_results()
    odds["event_id"] = pd.to_numeric(odds["event_id"], errors="coerce")

    tier_df = odds[["event_id", "Event_Tier"]].dropna()
    tier_df["Event_Tier"] = tier_df["Event_Tier"].astype(str).str.upper()
    tier_df = tier_df[tier_df["Event_Tier"].isin(["MAJOR", "SIGNATURE"])]
    tier_df = tier_df.drop_duplicates(subset=["event_id"])

    big_event_ids = tier_df["event_id"].dropna().astype(int).unique().tolist()

    # hard drop specific events
    drop_ids = {16, 27, 28, 60}
    big_event_ids = [e for e in big_event_ids if e not in drop_ids]

    # --- 2) Load rounds and filter to these players + events + years ---
    rounds = load_rounds()
    for col in ["year", "event_id", "dg_id", "finish_num"]:
        if col in rounds.columns:
            rounds[col] = pd.to_numeric(rounds[col], errors="coerce")

    hist = rounds[
        (rounds["dg_id"].isin(dg_ids))
        & (rounds["event_id"].isin(big_event_ids))
        & (rounds["year"] >= min_year)
        & (rounds["year"] < season)
    ].copy()

    if hist.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "event_id",
                "event_name",
                "Event_Tier",
                "starts",
                "cuts_made",
                "top5",
                "top10",
                "top25",
                "best_finish",
                "worst_finish",
                "avg_finish",
                "avg_sg_event",
            ]
        )

    hist["sg_total"] = pd.to_numeric(hist["sg_total"], errors="coerce")

    # --- 3) event-year stats ---
    event_year = (
        hist.groupby(["year", "event_id", "dg_id"], as_index=False)
        .agg(
            sg_event=("sg_total", "sum"),
            finish_num=("finish_num", "min"),
            fin_text=("fin_text", "first"),
            event_name=("event_name", "first"),
        )
    )

    fin_text = event_year["fin_text"].astype(str).str.upper()
    event_year["made_cut_year"] = np.where(fin_text.str.contains("CUT"), 0, 1)

    # --- 4) Determine canonical event_name per event_id before target season ---
    recent_names = (
        event_year.sort_values("year")
        .groupby(["event_id"])["event_name"]
        .last()
        .rename("canonical_event_name")
        .reset_index()
    )

    # --- 5) Aggregate STRICTLY by (dg_id, event_id) ---
    def _cnt_top(x, k):
        x = pd.to_numeric(x, errors="coerce")
        return int((x <= k).sum())

    agg = (
        event_year.groupby(["dg_id", "event_id"], as_index=False)
        .agg(
            starts=("year", "nunique"),
            cuts_made=("made_cut_year", "sum"),
            top5=("finish_num", lambda x: _cnt_top(x, 5)),
            top10=("finish_num", lambda x: _cnt_top(x, 10)),
            top25=("finish_num", lambda x: _cnt_top(x, 25)),
            best_finish=("finish_num", "min"),
            worst_finish=("finish_num", "max"),
            avg_finish=("finish_num", "mean"),
            avg_sg_event=("sg_event", "mean"),
        )
    )

    # --- 6) add canonical event_name + tier ---
    agg = agg.merge(recent_names, on="event_id", how="left")
    agg = agg.merge(
        tier_df[["event_id", "Event_Tier"]],
        on="event_id",
        how="left",
    )

    out = agg[
        [
            "dg_id",
            "event_id",
            "canonical_event_name",
            "Event_Tier",
            "starts",
            "cuts_made",
            "top5",
            "top10",
            "top25",
            "best_finish",
            "worst_finish",
            "avg_finish",
            "avg_sg_event",
        ]
    ].rename(columns={"canonical_event_name": "event_name"})

    # sort for readability: majors first, then signatures, then by name
    out["Event_Tier"] = out["Event_Tier"].astype(str)
    out = out.sort_values(["dg_id", "Event_Tier", "event_name"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# 4) Preseason package + shortlist IO
# ---------------------------------------------------------------------
def build_preseason_package(season: int, top_n: int = 30) -> dict:
    """
    Returns:
      - 'longlist': top-N preseason players from preseason_{season}.csv,
                    sorted by decayed weighted SG (field-adjusted if available),
                    no name columns, plus is_liv.
      - 'major_sig_history': big-event history (majors + signatures, no playoffs)
                             aggregated by (dg_id, event_id) across all years
                             < season using combined rounds.
    """
    season = int(season)

    # 1) load preseason_YYYY via helper
    longlist = build_preseason_longlist_topN(season, top_n=top_n)

    # 2) big-event history from rounds
    big_hist = compute_big_event_history_from_rounds(
        season=season,
        dg_ids=longlist["dg_id"].tolist(),
        min_year=2017,
    )

    return {
        "longlist": longlist.reset_index(drop=True),
        "major_sig_history": big_hist,
    }


def get_shortlist_path(season: int) -> Path:
    """
    Resolve PRESEASON_SHORTLIST_TEMPLATE for a given season.
    """
    return Path(str(PRESEASON_SHORTLIST_TEMPLATE).format(season=season))


def init_shortlist_template(season: int, top_n: int = 30) -> Path:
    """
    Create preseason_shortlist_{season}.csv if it doesn't already exist.

    Columns:
      dg_id, is_liv, tag_event_1..4, notes
    """
    out_path = get_shortlist_path(season)
    if out_path.exists():
        return out_path

    pkg = build_preseason_package(season=season, top_n=top_n)
    longlist = pkg["longlist"]

    liv_set = _get_liv_players_for_season(season)

    template = longlist[["dg_id"]].copy()
    template["dg_id"] = pd.to_numeric(template["dg_id"], errors="coerce").astype(
        "Int64"
    )
    template["is_liv"] = template["dg_id"].astype(int).isin(liv_set)

    for col in ["tag_event_1", "tag_event_2", "tag_event_3", "tag_event_4"]:
        template[col] = pd.NA
    template["notes"] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_path, index=False)
    return out_path


def load_shortlist(season: int) -> pd.DataFrame:
    """
    Load your manually edited preseason_shortlist_YYYY.csv.

    Columns expected:
      dg_id,is_liv,tag_event_1..4,notes

    tag_event_* are event_id ints (or blank).
    """
    path = get_shortlist_path(season)
    df = pd.read_csv(path)

    int_cols = ["dg_id", "player_name", "tag_event_1", "tag_event_2", "tag_event_3", "tag_event_4"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "is_liv" in df.columns:
        df["is_liv"] = df["is_liv"].astype(bool)

    return df


# ---------------------------------------------------------------------
# 5) Text report for a single preseason player
# ---------------------------------------------------------------------
def render_preseason_player_report(
    season: int,
    dg_id: int,
    longlist: pd.DataFrame,
    baseline_df: pd.DataFrame,
    big_hist_df: pd.DataFrame,
) -> str:
    """
    Render a text block summary for a single player in the preseason longlist:

      Player dg_id XXXX
      Star power (dec_wgh_sg or similar): X.XXX

      Baseline strokes gained (L40/L24/L12, pre season):
        ...

      Big-event history since 2017 (MAJOR + SIGNATURE, no playoffs):
        Event Name (event_id X)
          Starts: ..., Cuts made: ..., Top10: ..., Top25: ...
          Best finish: ..., Worst: ..., Avg: ..., Avg SG event: ...

    Returns a multi-line string.
    """
    dg_id = int(dg_id)

    # --- star power ---
    row_long = longlist[longlist["dg_id"] == dg_id]
    if row_long.empty:
        raise ValueError(f"dg_id {dg_id} not found in longlist.")
    row_long = row_long.iloc[0]

    star_col = _choose_star_power_column(longlist)
    star_val = row_long.get(star_col, np.nan)

    # --- baseline windows ---
    row_base = baseline_df[baseline_df["dg_id"] == dg_id]
    if row_base.empty:
        row_base = None
    else:
        row_base = row_base.iloc[0]

    windows = [40, 24, 12]
    lines: List[str] = []

    lines.append(f"Player dg_id {dg_id}")

    if pd.isna(star_val):
        lines.append(f"Star power ({star_col}): NA\n")
    else:
        lines.append(f"Star power ({star_col}): {float(star_val):.3f}\n")

    # Baseline SG
    lines.append("Baseline strokes gained (L40/L24/L12, pre season):")
    if row_base is None:
        lines.append("  No pre-season rounds available before first OAD event.")
    else:
        for w in windows:
            n_rounds = row_base.get(f"n_L{w}", 0)
            if n_rounds is None or (isinstance(n_rounds, float) and np.isnan(n_rounds)):
                n_rounds = 0

            if n_rounds > 0:
                sg_ott = row_base.get(f"sg_ott_L{w}", np.nan)
                sg_app = row_base.get(f"sg_app_L{w}", np.nan)
                sg_arg = row_base.get(f"sg_arg_L{w}", np.nan)
                sg_putt = row_base.get(f"sg_putt_L{w}", np.nan)
                sg_tot = row_base.get(f"sg_total_L{w}", np.nan)
                rank = row_base.get(f"rank_total_L{w}", np.nan)
                denom = row_base.get(f"rank_total_L{w}_denom", np.nan)

                rank_str = ""
                if not pd.isna(rank) and not pd.isna(denom):
                    rank_str = f"  (rank {int(rank)}/{int(denom)} in shortlist)"

                line = (
                    f"  L{w}: OTT {sg_ott:+.3f}, APP {sg_app:+.3f}, "
                    f"ARG {sg_arg:+.3f}, PUTT {sg_putt:+.3f}, "
                    f"TOTAL {sg_tot:+.3f} (n={int(n_rounds)}){rank_str}"
                )
            else:
                line = f"  L{w}: no rounds available pre-season."
            lines.append(line)

    # --- big event history ---
    lines.append("\nBig-event history since 2017 (MAJOR + SIGNATURE, no playoffs):")

    if big_hist_df is None or big_hist_df.empty:
        lines.append("  No major/signature history in this period.")
    else:
        sub = big_hist_df[big_hist_df["dg_id"] == dg_id].copy()
        if sub.empty:
            lines.append("  No major/signature history in this period.")
        else:
            sub = sub.sort_values(["Event_Tier", "event_name"])
            for _, row in sub.iterrows():
                ename = row.get("event_name", "")
                eid = row.get("event_id", np.nan)
                starts = row.get("starts", 0)
                cuts = row.get("cuts_made", 0)
                top10 = row.get("top10", 0)
                top25 = row.get("top25", 0)
                best = row.get("best_finish", np.nan)
                worst = row.get("worst_finish", np.nan)
                avg = row.get("avg_finish", np.nan)
                avg_sg = row.get("avg_sg_event", np.nan)

                lines.append(f"  {ename} (event_id {_fmt_int(eid)})")
                lines.append(
                    "    Starts: {s}, Cuts made: {c}, Top10: {t10}, Top25: {t25}".format(
                        s=_fmt_int(starts, "0"),
                        c=_fmt_int(cuts, "0"),
                        t10=_fmt_int(top10, "0"),
                        t25=_fmt_int(top25, "0"),
                    )
                )

                best_str = _fmt_int(best)
                worst_str = _fmt_int(worst)
                if not pd.isna(avg):
                    avg_str = f"{float(avg):.1f}"
                else:
                    avg_str = "NA"

                if not pd.isna(avg_sg):
                    avg_sg_str = f"{float(avg_sg):+.2f}"
                    lines.append(
                        f"    Best finish: {best_str}, Worst: {worst_str}, "
                        f"Avg: {avg_str}, Avg SG event: {avg_sg_str}"
                    )
                else:
                    lines.append(
                        f"    Best finish: {best_str}, Worst: {worst_str}, "
                        f"Avg: {avg_str}"
                    )

    return "\n".join(lines)