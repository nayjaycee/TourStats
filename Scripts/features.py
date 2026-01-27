# Scripts/features.py
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1) Rolling stats: L40 / L24 / L12 for SG + driving + score
# ---------------------------------------------------------------------
def _player_rolling_for_windows(
    g: pd.DataFrame,
    windows: Sequence[int],
) -> pd.Series:
    """
    Helper: for a single player's rounds (sorted descending by date),
    compute windowed means for each stat.
    """
    out = {}

    stats = [
        "sg_total",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "driving_dist",
        "driving_acc",
        "round_score",
    ]

    for w in windows:
        sub = g.head(w)
        for stat in stats:
            col_name = f"{stat}_L{w}"
            if stat in sub.columns:
                out[col_name] = sub[stat].mean()
            else:
                out[col_name] = np.nan

    return pd.Series(out)


def compute_rolling_stats(
    rounds_df: pd.DataFrame,
    as_of_date,
    dg_ids: Iterable[int],
    windows: Sequence[int] = (40, 24, 12),
) -> pd.DataFrame:
    """
    Compute rolling window stats for each player, as of as_of_date.

    Uses round_date if available, otherwise event_completed.

    Parameters
    ----------
    rounds_df : DataFrame
        Combined rounds 2017-2025.
    as_of_date : str or datetime
        Only rounds strictly before this date are used.
    dg_ids : iterable[int]
        Players to include.
    windows : sequence[int]
        Window sizes in rounds (e.g., (40,24,12)).

    Returns
    -------
    DataFrame with index dg_id and columns like:
      sg_total_L40, sg_total_L24, ..., driving_acc_L12, round_score_L40, etc.
    """
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = rounds_df.copy()
    for col in ["dg_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["dg_id"].isin([int(x) for x in dg_ids])].copy()

    # prefer round_date, fallback to event_completed
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
        date_col = "round_date"
    else:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
        date_col = "event_completed"

    df = df[df[date_col] < ts].copy()
    if df.empty:
        # no history before this date
        return pd.DataFrame(columns=["dg_id"])

    # sort descending for recency
    df = df.sort_values([ "dg_id", date_col, "event_id", "round_num" ], ascending=[True, False, False, False])

    grouped = (
        df.groupby("dg_id", group_keys=False)
          .apply(_player_rolling_for_windows, windows=windows)
          .reset_index()
    )

    return grouped


def compute_event_history(
    rounds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    event_id: int,
    as_of_date,
    min_year: int = 2017,
) -> pd.DataFrame:
    """
    Per-player history at THIS event across all years up to as_of_date.

    One *start* = one tournament year at this event for that player
    (NOT number of rounds).

    Returns one row per dg_id with:

        dg_id
        starts_event           # number of years played at this event
        made_cuts_event        # number of years made the cut
        made_cut_pct_event     # made_cuts_event / starts_event
        top25_event            # years with top-25 finish
        top10_event            # years with top-10 finish
        top5_event             # years with top-5 finish
        wins_event             # years with a win
        prev_finish_num_event  # most recent numeric finish
        prev_finish_text_event # most recent finish as string (if available)
        avg_score_event        # mean round_score across all years
        avg_sg_total_event     # mean sg_total across all years
    """
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"compute_event_history: invalid as_of_date={as_of_date}")

    df = rounds_df.copy()

    # --- filter to this event ---
    if "event_id" not in df.columns:
        raise ValueError("compute_event_history: rounds_df must contain 'event_id'.")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    df = df[df["event_id"] == int(event_id)].copy()

    if "dg_id" not in df.columns:
        raise ValueError("compute_event_history: rounds_df must contain 'dg_id'.")
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    # --- date cutoff: only history *before* as_of_date ---
    date_col = None
    for cand in ("event_completed", "event_date", "start_date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(
            "compute_event_history: rounds_df must have event_completed/event_date/start_date."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col] < ts].copy()

    # --- define event-year column: one "start" per (dg_id, event_year) ---
    year_col = None
    for cand in ("event_year", "season", "year"):
        if cand in df.columns:
            year_col = cand
            break

    if year_col is None:
        # derive from date if not present
        df["event_year"] = df[date_col].dt.year
        year_col = "event_year"

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[df[year_col] >= min_year].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "starts_event",
                "made_cuts_event",
                "made_cut_pct_event",
                "top25_event",
                "top10_event",
                "top5_event",
                "wins_event",
                "prev_finish_num_event",
                "prev_finish_text_event",
                "avg_score_event",
                "avg_sg_total_event",
            ]
        )

    # --- normalize column names / flags ---
    col_map = {c.lower(): c for c in df.columns}

    def get(col_lower: str) -> Optional[str]:
        return col_map.get(col_lower)

    fin_col = (
        get("finish_num")
        or get("finish_position")
        or get("finish_rank")
    )
    if fin_col:
        df[fin_col] = pd.to_numeric(df[fin_col], errors="coerce")

    sg_col = get("sg_total")
    if sg_col:
        df[sg_col] = pd.to_numeric(df[sg_col], errors="coerce")

    score_col = get("round_score")
    if score_col:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    def coerce_flag(col_lower: str) -> Optional[str]:
        c = get(col_lower)
        if c is None:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        return c

    made_flag_col = coerce_flag("made_cut")
    top25_flag_col = coerce_flag("top_25") or coerce_flag("top25")
    top10_flag_col = coerce_flag("top_10") or coerce_flag("top10")
    top5_flag_col = coerce_flag("top_5") or coerce_flag("top5")
    win_flag_col = coerce_flag("win")

    # fall back from finish number if needed, at *event* level later
    if fin_col and made_flag_col is None:
        df["made_cut_flag_round"] = df[fin_col].notna().astype(int)
        made_flag_col = "made_cut_flag_round"
    if fin_col and top25_flag_col is None:
        df["top25_flag_round"] = df[fin_col].between(1, 25, inclusive="both").astype(int)
        top25_flag_col = "top25_flag_round"
    if fin_col and top10_flag_col is None:
        df["top10_flag_round"] = df[fin_col].between(1, 10, inclusive="both").astype(int)
        top10_flag_col = "top10_flag_round"
    if fin_col and top5_flag_col is None:
        df["top5_flag_round"] = df[fin_col].between(1, 5, inclusive="both").astype(int)
        top5_flag_col = "top5_flag_round"
    if fin_col and win_flag_col is None:
        df["win_flag_round"] = (df[fin_col] == 1).astype(int)
        win_flag_col = "win_flag_round"

    # ------------------------------------------------------------------
    # 1) Collapse to one row per (dg_id, event_year) = one tournament start
    # ------------------------------------------------------------------
    df = df.sort_values([ "dg_id", year_col, date_col ])

    def _agg_event(g: pd.DataFrame) -> pd.Series:
        # event-level flags: if *any* round has the flag, treat event as having that outcome
        def ev_flag(col: Optional[str]) -> int:
            return int(g[col].max()) if col and col in g.columns else 0

        made_ev = ev_flag(made_flag_col)
        top25_ev = ev_flag(top25_flag_col)
        top10_ev = ev_flag(top10_flag_col)
        top5_ev  = ev_flag(top5_flag_col)
        win_ev   = ev_flag(win_flag_col)

        fin_val = g[fin_col].iloc[-1] if fin_col and fin_col in g.columns else np.nan
        avg_score = g[score_col].mean(skipna=True) if score_col else np.nan
        avg_sg = g[sg_col].mean(skipna=True) if sg_col else np.nan

        return pd.Series(
            {
                "dg_id": g["dg_id"].iloc[0],
                "event_year": g[year_col].iloc[0],
                "made_ev": made_ev,
                "top25_ev": top25_ev,
                "top10_ev": top10_ev,
                "top5_ev": top5_ev,
                "win_ev": win_ev,
                "finish_num_event": fin_val,
                "avg_score_event_year": avg_score,
                "avg_sg_total_event_year": avg_sg,
                "last_date_event_year": g[date_col].max(),
            }
        )

    ev_level = (
        df.groupby(["dg_id", year_col], group_keys=False)
        .apply(_agg_event)
        .reset_index(drop=True)
    )

    if ev_level.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "starts_event",
                "made_cuts_event",
                "made_cut_pct_event",
                "top25_event",
                "top10_event",
                "top5_event",
                "wins_event",
                "prev_finish_num_event",
                "prev_finish_text_event",
                "avg_score_event",
                "avg_sg_total_event",
            ]
        )

    # ------------------------------------------------------------------
    # 2) Aggregate per player across all years
    # ------------------------------------------------------------------
    ev_level = ev_level.sort_values(["dg_id", "last_date_event_year"])

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        starts = len(g)  # one row per (dg_id, event_year) → true event starts
        made = int(g["made_ev"].sum())
        top25 = int(g["top25_ev"].sum())
        top10 = int(g["top10_ev"].sum())
        top5 = int(g["top5_ev"].sum())
        wins = int(g["win_ev"].sum())

        last = g.iloc[-1]
        prev_finish_num = last["finish_num_event"]

        if not pd.isna(prev_finish_num):
            prev_finish_text = str(int(prev_finish_num))
        else:
            prev_finish_text = np.nan

        return pd.Series(
            {
                "dg_id": g["dg_id"].iloc[0],
                "starts_event": starts,
                "made_cuts_event": made,
                "made_cut_pct_event": (made / starts) if starts > 0 else np.nan,
                "top25_event": top25,
                "top10_event": top10,
                "top5_event": top5,
                "wins_event": wins,
                "prev_finish_num_event": prev_finish_num,
                "prev_finish_text_event": prev_finish_text,
                "avg_score_event": g["avg_score_event_year"].mean(skipna=True),
                "avg_sg_total_event": g["avg_sg_total_event_year"].mean(skipna=True),
            }
        )

    hist = (
        ev_level.groupby("dg_id", group_keys=False)
        .apply(_agg_player)
        .reset_index(drop=True)
    )

    return hist

# ---------------------------------------------------------------------
# 3) YTD stats (current season, as-of date)
# ---------------------------------------------------------------------
def compute_ytd_stats(
    rounds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    season_year: int,
    as_of_date,
    ytd_tracker: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Year-to-date stats up to as_of_date for a given season.

    Returns per-player:
      ytd_starts, ytd_made_cuts, ytd_made_cut_pct,
      ytd_top25, ytd_top10, ytd_top5, ytd_wins,
      ytd_avg_score, ytd_avg_sg_total.

    NOTE:
    - If ytd_tracker is provided, ytd_starts is computed from it (all tours).
    - Otherwise, ytd_starts falls back to odds_df behavior.
    """
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    yr = int(season_year)
    use_tracker = ytd_tracker is not None and not ytd_tracker.empty

    # ------------------------------------------------------------
    # STARTS: prefer ytd_tracker if provided (all tours)
    # ------------------------------------------------------------
    starts_df = None
    if ytd_tracker is not None and not ytd_tracker.empty:
        yt = ytd_tracker.copy()
        for col in ["year", "event_id", "dg_id"]:
            if col in yt.columns:
                yt[col] = pd.to_numeric(yt[col], errors="coerce")

        if "event_completed" in yt.columns:
            yt["event_completed"] = pd.to_datetime(yt["event_completed"], errors="coerce")
            yt_win = yt[(yt["year"] == yr) & (yt["event_completed"] < ts)].copy()

            # Optional: if your tracker includes multiple tours, keep PGA only
            if "tour" in yt_win.columns:
                yt_win = yt_win[yt_win["tour"].astype(str).str.upper().str.contains("PGA", na=False)].copy()

            # Critical: ensure 1 row per (dg_id, event_id)
            yt_win = (
                yt_win.dropna(subset=["dg_id", "event_id"])
                .sort_values("event_completed")
                .drop_duplicates(subset=["dg_id", "event_id"], keep="last")
            )

            starts_df = (
                yt_win.groupby("dg_id", as_index=False)
                .agg(ytd_starts=("event_id", "count"))
            )

    tracker_agg = None

    if use_tracker:
        yt = ytd_tracker.copy()

        for col in ["year", "event_id", "dg_id"]:
            if col in yt.columns:
                yt[col] = pd.to_numeric(yt[col], errors="coerce")

        if "event_completed" in yt.columns:
            yt["event_completed"] = pd.to_datetime(yt["event_completed"], errors="coerce")
            yt = yt[(yt["year"] == yr) & (yt["event_completed"] < ts)]

        # Optional: if tracker includes multiple tours, keep PGA only
        if "tour" in yt.columns:
            yt = yt[yt["tour"].astype(str).str.upper().str.contains("PGA", na=False)].copy()

        # Critical: ensure 1 row per (dg_id, event_id) before summing flags
        yt = (
            yt.dropna(subset=["dg_id", "event_id"])
            .sort_values("event_completed")
            .drop_duplicates(subset=["dg_id", "event_id"], keep="last")
        )

        for col in ["Top_25", "Top_10", "Top_5", "Win"]:
            if col in yt.columns:
                yt[col] = pd.to_numeric(yt[col], errors="coerce").fillna(0).clip(0, 1).astype(int)
            else:
                yt[col] = 0

        # made-cut logic
        if "tour" in yt.columns:
            is_liv = yt["tour"].astype(str).str.lower().str.contains("liv", na=False)
        else:
            is_liv = pd.Series(False, index=yt.index)

        if "Made_Cut" in yt.columns:
            mc = pd.to_numeric(yt["Made_Cut"], errors="coerce")
        else:
            mc = np.nan

        mc = mc.where(~is_liv, 1)
        yt["Made_Cut"] = mc.fillna(0).clip(0, 1).astype(int)

        tracker_agg = (
            yt.groupby("dg_id", as_index=False)
            .agg(
                ytd_starts=("event_id", "count"),
                ytd_made_cuts=("Made_Cut", "sum"),
                ytd_top25=("Top_25", "sum"),
                ytd_top10=("Top_10", "sum"),
                ytd_top5=("Top_5", "sum"),
                ytd_wins=("Win", "sum"),
            )
        )

    # ------------------------------------------------------------
    # FLAGS: keep your existing odds_df aggregation (for now)
    # ------------------------------------------------------------
    o = odds_df.copy()
    for col in ["year", "event_id", "dg_id"]:
        if col in o.columns:
            o[col] = pd.to_numeric(o[col], errors="coerce")

    if "event_completed" in o.columns:
        o["event_completed"] = pd.to_datetime(o["event_completed"], errors="coerce")
        o_ytd = o[(o["year"] == yr) & (o["event_completed"] < ts)].copy()
    else:
        o_ytd = o[o["year"] == yr].copy()

    if o_ytd.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "ytd_starts",
                "ytd_made_cuts",
                "ytd_made_cut_pct",
                "ytd_top25",
                "ytd_top10",
                "ytd_top5",
                "ytd_wins",
                "ytd_avg_score",
                "ytd_avg_sg_total",
            ]
        )

    for col in ["Made_Cut", "Top_25", "Top_10", "Top_5", "Win"]:
        if col in o_ytd.columns:
            o_ytd[col] = pd.to_numeric(o_ytd[col], errors="coerce").fillna(0).astype(int)
        else:
            o_ytd[col] = 0

    if tracker_agg is not None:
        agg_flags = tracker_agg.copy()
    else:
        # fallback to old odds_df behavior
        agg_flags = (
            o_ytd.groupby(["dg_id", "player_name"], as_index=False)
            .agg(
                ytd_starts=("dg_id", "count"),
                ytd_made_cuts=("Made_Cut", "sum"),
                ytd_top25=("Top_25", "sum"),
                ytd_top10=("Top_10", "sum"),
                ytd_top5=("Top_5", "sum"),
                ytd_wins=("Win", "sum"),
            )
        )

    # overwrite starts if we computed all-tour starts from ytd_tracker
    if starts_df is not None:
        agg_flags = agg_flags.drop(columns=["ytd_starts"], errors="ignore")
        agg_flags = agg_flags.merge(starts_df, on="dg_id", how="left")
        agg_flags["ytd_starts"] = agg_flags["ytd_starts"].fillna(0).astype(int)

    # made cut pct (safe)
    agg_flags["ytd_made_cut_pct"] = np.where(
        agg_flags["ytd_starts"] > 0,
        agg_flags["ytd_made_cuts"] / agg_flags["ytd_starts"],
        np.nan,
    )

    # ------------------------------------------------------------
    # AVG SCORE / SG: from rounds_df (existing behavior)
    # ------------------------------------------------------------
    r = rounds_df.copy()
    for col in ["year", "event_id", "dg_id"]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")

    if "event_completed" in r.columns:
        r["event_completed"] = pd.to_datetime(r["event_completed"], errors="coerce")

    r_ytd = r[r["year"] == yr].copy()
    if "event_completed" in r_ytd.columns:
        r_ytd = r_ytd[r_ytd["event_completed"] < ts]

    for col in ["round_score", "sg_total"]:
        if col in r_ytd.columns:
            r_ytd[col] = pd.to_numeric(r_ytd[col], errors="coerce")

    r_agg = (
        r_ytd.groupby("dg_id", as_index=False)
        .agg(
            ytd_avg_score=("round_score", "mean"),
            ytd_avg_sg_total=("sg_total", "mean"),
        )
    )

    out = agg_flags.merge(r_agg, on="dg_id", how="left")
    return out

# ---------------------------------------------------------------------
# 4) Course-fit score (player 5-attr skill × course_importances)
# ---------------------------------------------------------------------
def compute_course_fit_score(
    player_skills_df: pd.DataFrame,
    course_fit_df: pd.DataFrame,
    course_num: int,
    dg_ids: Iterable[int],
) -> pd.DataFrame:
    """
    Compute a simple course-fit score per player:
      score_raw = sum_j skill_j * imp_j
    where j in {dist, acc, app, arg, putt}.

    Then z-score across the field for comparability.

    Parameters
    ----------
    player_skills_df : DataFrame
        e.g. player_skill_5attr_hist_to_YYYY.csv
        with columns: ['dg_id', 'player_name', 'n_rounds',
                       'skill_dist', 'skill_acc', 'skill_app',
                       'skill_arg', 'skill_putt', ...]
    course_fit_df : DataFrame
        course_fit_YYYY_dg_style_5attr.csv
        with columns: ['course_num', 'imp_dist', 'imp_acc', 'imp_app',
                       'imp_arg', 'imp_putt', ...]
    course_num : int
        Main course for this event.
    dg_ids : iterable[int]
        Players to include.

    Returns
    -------
    DataFrame with:
      ['dg_id', 'course_fit_score']
    """
    cnum = int(course_num)
    df_course = course_fit_df.copy()
    df_course["course_num"] = pd.to_numeric(df_course["course_num"], errors="coerce")

    row = df_course[df_course["course_num"] == cnum]
    if row.empty:
        # no profile for this course
        return pd.DataFrame(columns=["dg_id", "course_fit_score"])

    row = row.iloc[0]
    imp = {
        "dist": float(row.get("imp_dist", 0.0) or 0.0),
        "acc": float(row.get("imp_acc", 0.0) or 0.0),
        "app": float(row.get("imp_app", 0.0) or 0.0),
        "arg": float(row.get("imp_arg", 0.0) or 0.0),
        "putt": float(row.get("imp_putt", 0.0) or 0.0),
    }

    skills = player_skills_df.copy()
    skills["dg_id"] = pd.to_numeric(skills["dg_id"], errors="coerce")
    skills = skills[skills["dg_id"].isin([int(x) for x in dg_ids])].copy()

    if skills.empty:
        return pd.DataFrame(columns=["dg_id", "course_fit_score"])

    # compute raw dot-product
    scores = []
    for _, s in skills.iterrows():
        raw = (
            (float(s.get("skill_dist", 0.0) or 0.0) * imp["dist"])
            + (float(s.get("skill_acc", 0.0) or 0.0) * imp["acc"])
            + (float(s.get("skill_app", 0.0) or 0.0) * imp["app"])
            + (float(s.get("skill_arg", 0.0) or 0.0) * imp["arg"])
            + (float(s.get("skill_putt", 0.0) or 0.0) * imp["putt"])
        )
        scores.append((int(s["dg_id"]), raw))

    scores_df = pd.DataFrame(scores, columns=["dg_id", "course_fit_raw"])

    # z-score across the field
    mu = scores_df["course_fit_raw"].mean()
    sd = scores_df["course_fit_raw"].std(ddof=0)
    if sd == 0 or np.isnan(sd):
        scores_df["course_fit_score"] = 0.0
    else:
        scores_df["course_fit_score"] = (scores_df["course_fit_raw"] - mu) / sd

    return scores_df[["dg_id", "course_fit_score"]]

import numpy as np
import pandas as pd


def build_event_history_for_event(
    master_df: pd.DataFrame,
    event_id_fixed: int,
    limit_dg_ids: Optional[Iterable[int]] = None,
    min_year: int = 2017,
) -> pd.DataFrame:
    """
    Build per-player *counts* of history at a single event.

    Returns one row per dg_id with:
        starts_event   - number of starts at this event
        top25_event    - # of top-25 finishes
        top10_event    - # of top-10 finishes
        top5_event     - # of top-5 finishes
        wins_event     - # of wins
        best_finish    - best finishing position
        avg_finish     - average finishing position

    Optional:
        - limit_dg_ids: if provided, only return rows for these dg_ids
        - min_year: only use results from this year forward (e.g. 2019)
    """
    df = master_df.copy()

    # normalize types
    df["dg_id"]          = pd.to_numeric(df.get("dg_id"), errors="coerce").astype("Int64")
    df["event_id_fixed"] = pd.to_numeric(df.get("event_id_fixed"), errors="coerce").astype("Int64")
    df["finish_position"] = pd.to_numeric(df.get("finish_position"), errors="coerce")

    # year filter if you have a year/season column
    year_col = None
    for c in ("event_year", "season", "year"):
        if c in df.columns:
            year_col = c
            break

    if year_col is not None:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df[df[year_col] >= min_year].copy()

    # keep only this event
    df = df[df["event_id_fixed"] == int(event_id_fixed)].copy()
    df = df.dropna(subset=["dg_id", "finish_position"])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "starts_event",
                "top25_event",
                "top10_event",
                "top5_event",
                "wins_event",
                "best_finish",
                "avg_finish",
            ]
        )

    # basic flags
    df["is_top25"] = df["finish_position"].le(25)
    df["is_top10"] = df["finish_position"].le(10)
    df["is_top5"]  = df["finish_position"].le(5)
    df["is_win"]   = df["finish_position"].eq(1)

    hist = (
        df.groupby("dg_id")
        .agg(
            starts_event=("dg_id", "size"),
            top25_event=("is_top25", "sum"),
            top10_event=("is_top10", "sum"),
            top5_event=("is_top5", "sum"),
            wins_event=("is_win", "sum"),
            best_finish=("finish_position", "min"),
            avg_finish=("finish_position", "mean"),
        )
        .reset_index()
    )

    # optionally restrict to the dg_ids in your top-30 performance view
    if limit_dg_ids is not None:
        limit = pd.Series(list(limit_dg_ids)).dropna().astype(int).unique().tolist()
        hist = hist[hist["dg_id"].astype(int).isin(limit)].copy()

    # nice types
    for c in ["starts_event", "top25_event", "top10_event", "top5_event", "wins_event"]:
        if c in hist.columns:
            hist[c] = hist[c].astype(int)

    if "avg_finish" in hist.columns:
        hist["avg_finish"] = hist["avg_finish"].round(2)

    return hist

def compute_course_history_stats(
    rounds_df: pd.DataFrame,
    dg_ids: list[int],
    course_num: int,
    as_of_date,
) -> pd.DataFrame:
    ts = pd.to_datetime(as_of_date, errors="coerce")
    r = rounds_df.copy()

    r["dg_id"] = pd.to_numeric(r["dg_id"], errors="coerce")
    r["course_num"] = pd.to_numeric(r["course_num"], errors="coerce")
    r["event_completed"] = pd.to_datetime(r["event_completed"], errors="coerce")
    r["sg_total"] = pd.to_numeric(r["sg_total"], errors="coerce")
    r["round_score"] = pd.to_numeric(r["round_score"], errors="coerce")

    r = r[
        (r["dg_id"].isin(dg_ids)) &
        (r["course_num"] == int(course_num)) &
        (r["event_completed"] < ts)
    ].copy()

    if r.empty:
        return pd.DataFrame({"dg_id": dg_ids})

    out = (
        r.groupby("dg_id", as_index=False)
         .agg(
             course_starts=("event_completed", "nunique"),
             course_avg_sg_total=("sg_total", "mean"),
             course_avg_score=("round_score", "mean"),
         )
    )

    return out