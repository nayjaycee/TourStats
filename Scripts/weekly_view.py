# Scripts/weekly_view.py
from __future__ import annotations

from typing import Dict

import pandas as pd
import numpy as np
from pathlib import Path

from Scripts.schedule import build_season_schedule
from Scripts.data_io import (
    load_rounds,
    load_odds_and_results,
    load_course_fit,
    load_player_skills,
)
from Scripts.field_sim import get_actual_field, simulate_field_for_event
from Scripts.features import (
    compute_rolling_stats,
    compute_event_history,
    compute_ytd_stats,
    compute_course_fit_score,
    compute_course_history_stats,  # NEW
)
from Scripts.ev import (
    compute_current_event_ev,
    compute_future_ev_for_players,
)
from Scripts.preseason import load_shortlist
from Scripts.patterns import (
    compute_event_patterns,
    score_current_field_against_patterns,
    build_event_patterns_text_from_rounds,  # <- NEW import
)

YTD_TRACKER_PATH = Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/ytd_tracker.csv")
ytd_tracker = pd.read_csv(YTD_TRACKER_PATH)

COURSE_SENSITIVITY_PATH = Path("/Users/joshmacbook/python_projects/OAD/Data/Clean/Processed/course_sensitivity_table.csv")

print("COURSE_SENSITIVITY_PATH:", COURSE_SENSITIVITY_PATH)
print("  exists?:", COURSE_SENSITIVITY_PATH.exists())

try:
    course_sens_df = pd.read_csv(COURSE_SENSITIVITY_PATH)
    course_sens_df["course_num"] = pd.to_numeric(course_sens_df["course_num"], errors="coerce").astype("Int64")
    course_sens_df["course_type"] = course_sens_df["course_type"].astype(str).str.lower()
except Exception as e:
    print("FAILED loading course sensitivity:", repr(e))
    course_sens_df = pd.DataFrame(columns=["course_num", "course_type"])


def _coerce_dg_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "dg_id" not in out.columns:
        return out
    out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
    out = out.dropna(subset=["dg_id"]).copy()
    out["dg_id"] = out["dg_id"].astype(int)
    return out

def _attach_shortlist_flags(
    summary: pd.DataFrame,
    season_year: int,
    event_id: int,
) -> pd.DataFrame:
    """
    Attach shortlist info from the preseason shortlist CSV.

    Rules:
    - If a flag column exists (e.g. 'in_shortlist', 'keep', 'keep_preseason'):
        * If it has any True-like values -> only those rows are in the shortlist.
        * If it has zero True-like values -> any dg_id present in the file
          is treated as in the shortlist.
    - If NO flag column exists:
        * Any dg_id present in the file is treated as in the shortlist.

    Tags:
    - Any columns named 'tag_event_*' are merged through.
    - 'tagged_here' = True if any tag_event_* == current event_id.
    """
    yr = int(season_year)
    eid = int(event_id)

    try:
        sl = load_shortlist(yr)
    except FileNotFoundError:
        summary["is_shortlist"] = False
        summary["tagged_here"] = False
        return summary

    if sl.empty or "dg_id" not in sl.columns:
        summary["is_shortlist"] = False
        summary["tagged_here"] = False
        return summary

    sl = sl.copy()
    sl["dg_id"] = pd.to_numeric(sl["dg_id"], errors="coerce")
    sl = sl[sl["dg_id"].notna()]
    if sl.empty:
        summary["is_shortlist"] = False
        summary["tagged_here"] = False
        return summary

    # --- determine flag column (if any) ---
    flag_candidates = [
        "in_shortlist",
        "keep",
        "keep_preseason",
        "keep_flag",
    ]
    flag_col = None
    for c in flag_candidates:
        if c in sl.columns:
            flag_col = c
            break

    if flag_col is None:
        # no explicit flag → being present in the file = shortlist
        shortlist_ids = set(sl["dg_id"].astype(int).tolist())
    else:
        vals = sl[flag_col].astype(str).str.lower()
        mask_true = vals.isin(["1", "true", "t", "yes", "y"])

        if mask_true.any():
            # only explicitly-kept rows are shortlist
            shortlist_ids = set(sl.loc[mask_true, "dg_id"].astype(int).tolist())
        else:
            # flag column exists but has no True → fall back to "present in file"
            shortlist_ids = set(sl["dg_id"].astype(int).tolist())

    # --- tags / extras to merge in ---
    tag_cols = [c for c in sl.columns if c.startswith("tag_event")]
    extra_cols = ["is_liv"] if "is_liv" in sl.columns else []
    merge_cols = ["dg_id"] + extra_cols + tag_cols
    merge_cols = [c for c in merge_cols if c in sl.columns]

    if merge_cols:
        sl_unique = sl[merge_cols].drop_duplicates(subset=["dg_id"])
        summary = summary.merge(sl_unique, on="dg_id", how="left")
    else:
        # ensure these columns exist downstream
        summary = summary.copy()

    # is_shortlist = membership in shortlist_ids
    summary["dg_id"] = pd.to_numeric(summary["dg_id"], errors="coerce")
    summary["is_shortlist"] = summary["dg_id"].astype("Int64").isin(shortlist_ids)

    # tagged_here = any tag_event_* equals current event_id
    tagged_mask = False
    for col in tag_cols:
        if col in summary.columns:
            tagged_mask = tagged_mask | (summary[col] == eid)
    summary["tagged_here"] = tagged_mask.fillna(False) if isinstance(tagged_mask, pd.Series) else False

    return summary

def _pick_date_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_ytd_stats_rolling_90d(
    rounds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    dg_ids: list[int],
    as_of_date: pd.Timestamp,
    days: int = 90,
    ytd_tracker: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Jan–Mar YTD proxy: use last `days` of *events* ending at as_of_date.
    HARD GUARANTEE:
      - starts/outcomes are computed at the EVENT LEVEL (1 row per dg_id-year-event_id)
      - ytd_made_cut_pct is always in [0, 1] when starts>0
    Prefers ytd_tracker if provided (since it's already event-level).
    """
    end_dt = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(end_dt):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    start_dt = end_dt - pd.Timedelta(days=days)

    dg_ids = [int(x) for x in dg_ids if pd.notna(x)]
    base = pd.DataFrame({"dg_id": dg_ids}).drop_duplicates().copy()
    base["dg_id"] = pd.to_numeric(base["dg_id"], errors="coerce").astype(int)

    # -------------------------
    # 1) EVENT-LEVEL outcomes (starts, cuts, topX, wins)
    # Prefer ytd_tracker (already event-level); fallback to odds_df
    # -------------------------
    src = None
    if ytd_tracker is not None and not ytd_tracker.empty:
        src = ytd_tracker.copy()
        # expected columns in your ytd_tracker: year, event_id, dg_id, event_completed, Made_Cut, Top_25, Top_10, Top_5, Win, tour (optional)
        for c in ["year", "event_id", "dg_id"]:
            if c in src.columns:
                src[c] = pd.to_numeric(src[c], errors="coerce")

        if "event_completed" in src.columns:
            src["event_completed"] = pd.to_datetime(src["event_completed"], errors="coerce")
            src = src[(src["event_completed"] >= start_dt) & (src["event_completed"] <= end_dt)].copy()

        src = src[pd.to_numeric(src.get("dg_id"), errors="coerce").isin(dg_ids)].copy()

        # LIV made-cut logic (if tour exists)
        if "tour" in src.columns:
            is_liv = src["tour"].astype(str).str.lower().str.contains("liv", na=False)
        else:
            is_liv = pd.Series(False, index=src.index)

        # normalize flags
        def _flag(col: str) -> pd.Series:
            if col in src.columns:
                return pd.to_numeric(src[col], errors="coerce").fillna(0).astype(int)
            return pd.Series(0, index=src.index, dtype=int)

        src["Made_Cut"] = _flag("Made_Cut")
        src["Top_25"] = _flag("Top_25")
        src["Top_10"] = _flag("Top_10")
        src["Top_5"]  = _flag("Top_5")
        src["Win"]    = _flag("Win")

        # LIV → treat as made cut
        src.loc[is_liv, "Made_Cut"] = 1

        # critical: one row per player-event
        key_cols = [c for c in ["dg_id", "year", "event_id"] if c in src.columns]
        if len(key_cols) < 2:
            # if something is missing, still dedupe on dg_id+event_id at minimum
            key_cols = [c for c in ["dg_id", "event_id"] if c in src.columns]
        src = src.dropna(subset=["dg_id", "event_id"]).drop_duplicates(subset=key_cols).copy()

        outcome_agg = (
            src.groupby("dg_id", as_index=False)
               .agg(
                   ytd_starts=("event_id", "count"),
                   ytd_made_cuts=("Made_Cut", "sum"),
                   ytd_top25=("Top_25", "sum"),
                   ytd_top10=("Top_10", "sum"),
                   ytd_top5=("Top_5", "sum"),
                   ytd_wins=("Win", "sum"),
               )
        )

    else:
        # fallback: use odds_df but still force event-level dedupe
        o = odds_df.copy()
        for c in ["year", "event_id", "dg_id"]:
            if c in o.columns:
                o[c] = pd.to_numeric(o[c], errors="coerce")

        o = o[pd.to_numeric(o.get("dg_id"), errors="coerce").isin(dg_ids)].copy()

        o_date_col = _pick_date_col(o, ["event_completed", "event_date", "date", "tournament_date", "end_date", "start_date"])
        if o_date_col is not None:
            o[o_date_col] = pd.to_datetime(o[o_date_col], errors="coerce")
            o = o[(o[o_date_col] >= start_dt) & (o[o_date_col] <= end_dt)].copy()

        # normalize flags
        def _flag_o(col: str) -> pd.Series:
            if col in o.columns:
                return pd.to_numeric(o[col], errors="coerce").fillna(0).astype(int)
            return pd.Series(0, index=o.index, dtype=int)

        made_cut_col = None
        for c in ["Made_Cut", "made_cut", "made_cut_flag"]:
            if c in o.columns:
                made_cut_col = c
                break

        if made_cut_col is not None:
            o["Made_Cut"] = _flag_o(made_cut_col)
        else:
            # infer from finish_num if present
            if "finish_num" in o.columns:
                o["Made_Cut"] = pd.to_numeric(o["finish_num"], errors="coerce").notna().astype(int)
            else:
                o["Made_Cut"] = 0

        # topX/win from existing columns if present; else infer from finish_num
        if "Top_25" in o.columns:
            o["Top_25"] = _flag_o("Top_25")
            o["Top_10"] = _flag_o("Top_10") if "Top_10" in o.columns else 0
            o["Top_5"]  = _flag_o("Top_5")  if "Top_5"  in o.columns else 0
            o["Win"]    = _flag_o("Win")    if "Win"    in o.columns else 0
        else:
            fin = pd.to_numeric(o["finish_num"], errors="coerce") if "finish_num" in o.columns else pd.Series(np.nan, index=o.index)
            o["Top_25"] = (fin <= 25).fillna(False).astype(int)
            o["Top_10"] = (fin <= 10).fillna(False).astype(int)
            o["Top_5"]  = (fin <= 5).fillna(False).astype(int)
            o["Win"]    = (fin == 1).fillna(False).astype(int)

        # critical: one row per player-event (prevents inflated starts/cuts)
        key_cols = [c for c in ["dg_id", "year", "event_id"] if c in o.columns]
        if len(key_cols) < 2:
            key_cols = [c for c in ["dg_id", "event_id"] if c in o.columns]
        o = o.dropna(subset=["dg_id", "event_id"]).drop_duplicates(subset=key_cols).copy()

        outcome_agg = (
            o.groupby("dg_id", as_index=False)
             .agg(
                 ytd_starts=("event_id", "count"),
                 ytd_made_cuts=("Made_Cut", "sum"),
                 ytd_top25=("Top_25", "sum"),
                 ytd_top10=("Top_10", "sum"),
                 ytd_top5=("Top_5", "sum"),
                 ytd_wins=("Win", "sum"),
             )
        )

    out = base.merge(outcome_agg, on="dg_id", how="left")
    for c in ["ytd_starts", "ytd_made_cuts", "ytd_top25", "ytd_top10", "ytd_top5", "ytd_wins"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)

    # made cut pct: hard-safe
    out["ytd_made_cut_pct"] = np.where(out["ytd_starts"] > 0, out["ytd_made_cuts"] / out["ytd_starts"], 0.0)
    out["ytd_made_cut_pct"] = out["ytd_made_cut_pct"].clip(lower=0.0, upper=1.0)

    # -------------------------
    # 2) FORM-ish averages from rounds_df over the same window
    # -------------------------
    r = rounds_df.copy()
    r = r[pd.to_numeric(r.get("dg_id"), errors="coerce").isin(dg_ids)].copy()

    r_date_col = _pick_date_col(r, ["round_date", "event_completed", "date", "dt", "timestamp"])
    if r_date_col is not None:
        r[r_date_col] = pd.to_datetime(r[r_date_col], errors="coerce")
        r = r[(r[r_date_col] >= start_dt) & (r[r_date_col] <= end_dt)].copy()

    if "round_score" in r.columns:
        r["round_score"] = pd.to_numeric(r["round_score"], errors="coerce")
        avg_score = r.groupby("dg_id")["round_score"].mean()
        out["ytd_avg_score"] = avg_score.reindex(out["dg_id"]).fillna(0.0).values
    else:
        out["ytd_avg_score"] = 0.0

    if "sg_total" in r.columns:
        r["sg_total"] = pd.to_numeric(r["sg_total"], errors="coerce")
        avg_sg = r.groupby("dg_id")["sg_total"].mean()
        out["ytd_avg_sg_total"] = avg_sg.reindex(out["dg_id"]).fillna(0.0).values
    else:
        out["ytd_avg_sg_total"] = 0.0

    return out

def _round_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round float columns for readability:
      - ev_current, ev_future_total: 0 decimals (whole dollars)
      - everything else: 2 decimals
    """
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    for col in float_cols:
        if col in ("ev_current", "ev_future_total", "ev_current_adj", "ev_future_max", "ev_future_max_adj"):
            out[col] = out[col].round(0)
        else:
            out[col] = out[col].round(2)
    return out

def _pct_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank in [0,1] (higher=better). NaNs stay NaN.
    """
    s = pd.to_numeric(series, errors="coerce")
    n = s.notna().sum()
    if n <= 1:
        return pd.Series([np.nan] * len(s), index=s.index)
    r = s.rank(method="average", ascending=True)
    return (r - 1) / (n - 1)

def _normalize_tier_from_schedule_row(sched_row: pd.Series) -> str:
    """
    Returns: 'major' | 'signature' | 'regular'
    Prefers schedule tier columns; falls back to name heuristics for majors.
    """
    # 1) schedule-provided tier if present
    tier = ""
    for k in ["event_tier", "Event_Tier", "tier"]:
        if k in sched_row.index:
            tier = str(sched_row.get(k, "")).strip().lower()
            break

    # normalize common variants
    if tier in ["majors", "major"]:
        return "major"
    if tier in ["signature", "signature event", "sig"]:
        return "signature"
    if tier in ["regular", "standard", "full", ""]:
        tier = ""  # keep going to fallback

    # 2) fallback: major detection by name
    ev_name = str(sched_row.get("event_name", "")).lower()
    if any(key in ev_name for key in ["masters", "u.s. open", "us open", "pga championship", "open championship"]):
        return "major"

    # 3) if we can't tell, treat as regular
    return "regular"

def build_weekly_view(
    season_year: int,
    event_id: int,
) -> Dict[str, pd.DataFrame]:
    """
    Build the full weekly helper set for a given event:
      - schedule_row
      - field
      - rolling / event_history / ytd / course_fit / course_history
      - ev_current / ev_future
      - summary (master)
      - shortlist tags
      - pattern scores/flags
      - table_performance / table_event_history / table_ytd (+ top50 versions)
      - pattern_candidates (short list of pattern matches)
      - patterns_summary_text
    """
    yr = int(season_year)
    eid = int(event_id)

    # --- schedule / context ---
    schedule_df = build_season_schedule(yr)
    sched_row = schedule_df[schedule_df["event_id"] == eid]
    if sched_row.empty:
        raise ValueError(f"Event_id {eid} not found in season {yr} schedule.")
    sched_row = sched_row.iloc[0]

    event_date = pd.to_datetime(sched_row["event_date"], errors="coerce")
    if pd.isna(event_date):
        raise ValueError(f"Schedule event_date is missing/invalid for yr={yr}, event_id={eid}")
    course_num = int(sched_row["course_num"])
    purse = float(sched_row["purse"])
    ws = pd.to_numeric(sched_row.get("winner_share", np.nan), errors="coerce")
    winner_share = float(ws) if np.isfinite(ws) else None

    cutoff_date = event_date - pd.Timedelta(days=1)

    # Hard-code Players Championship 2025 only (schedule date is Monday, but completion is Sunday)
    if yr == 2025 and eid == 11:
        cutoff_date = pd.Timestamp("2025-03-16")

    # --- core data ---
    rounds_df = load_rounds()
    odds_all = load_odds_and_results()
    odds_df = odds_all
    course_fit_df = load_course_fit(yr)
    player_skills_df = load_player_skills(yr)

    # keep a season slice ONLY for "actual field" lookups
    odds_season = odds_all.copy()
    if "year" in odds_season.columns:
        odds_season["year"] = pd.to_numeric(odds_season["year"], errors="coerce")
        odds_season = odds_season[odds_season["year"] == yr].copy()

    # --- field ---
    field_df = get_actual_field(odds_season, yr, eid)

    # if 2026 (or any season) has no actual field yet, simulate it from historical odds
    if field_df is None or field_df.empty:
        field_df = simulate_field_for_event(
            season_year=yr,
            event_id=eid,
            as_of_date=cutoff_date,
            odds_df=odds_all,  # <-- IMPORTANT: pass ALL years, not year==yr
            schedule_df=schedule_df,
        )

    field_df = _coerce_dg_id(field_df)
    if field_df is None or field_df.empty:
        raise ValueError(...)
    dg_ids = field_df["dg_id"].dropna().astype(int).unique().tolist()

    if field_df is None or field_df.empty:
        raise ValueError(f"Field is empty for yr={yr}, event_id={eid} (after coercing dg_id).")

    # --- rolling stats ---
    rolling_df = compute_rolling_stats(
        rounds_df,
        as_of_date=cutoff_date,
        dg_ids=dg_ids,
        windows=(40, 24, 12),
    )

    # --- event history (this event only, prior years) ---
    event_hist_df = compute_event_history(
        rounds_df,
        odds_df,
        event_id=eid,
        as_of_date=cutoff_date,
    )

    # --- YTD stats: Jan–Mar use last 90 days ending at cutoff_date; April+ use true season YTD ---
    if pd.to_datetime(cutoff_date).month <= 3:
        ytd_df = compute_ytd_stats_rolling_90d(
            rounds_df=rounds_df,
            odds_df=odds_df,
            dg_ids=dg_ids,
            as_of_date=cutoff_date,
            days=90,
            ytd_tracker=ytd_tracker,
        )
    else:
        ytd_df = compute_ytd_stats(
            rounds_df=rounds_df,
            odds_df=odds_df,
            season_year=yr,
            as_of_date=cutoff_date,
            ytd_tracker=ytd_tracker,
        )

    # --- course history (this course only, prior years / prior rounds) ---
    course_hist_df = compute_course_history_stats(
        rounds_df=rounds_df,
        dg_ids=dg_ids,
        course_num=course_num,
        as_of_date=cutoff_date,
    )

    # --- course fit (player skills × course profile) ---
    course_fit_scores = compute_course_fit_score(
        player_skills_df,
        course_fit_df,
        course_num=course_num,
        dg_ids=dg_ids,
    )

    # --- EV for current event ---
    ev_current_df = compute_current_event_ev(
        odds_df=odds_df,
        event_id=eid,
        purse=purse,
        winner_share=winner_share,  # <-- add
        use_pre_odds=False,
    )

    # --- future EV over remaining events this season ---
    ev_future_df = compute_future_ev_for_players(
        odds_df=odds_df,
        schedule_df=schedule_df,
        season_year=yr,
        as_of_date=cutoff_date,
        dg_ids=dg_ids,
        use_pre_odds=False,
        fallback_odds=1000.0,
    )

    # Aggregate future EV per player: total and max single-event EV
    if ev_future_df is not None and not ev_future_df.empty and "dg_id" in ev_future_df.columns:
        future_agg = (
            ev_future_df.groupby("dg_id", as_index=False)
            .agg(
                ev_future_total=("ev_future", "sum"),
                ev_future_max=("ev_future", "max"),
            )
        )
    else:
        future_agg = pd.DataFrame(columns=["dg_id", "ev_future_total", "ev_future_max"])

    # --- master summary join ---
    summary = (
        field_df.merge(rolling_df, on="dg_id", how="left")
        .merge(event_hist_df, on="dg_id", how="left")
        .merge(ytd_df, on="dg_id", how="left")
        .merge(course_fit_scores, on="dg_id", how="left")
        .merge(ev_current_df[["dg_id", "decimal_odds", "ev_current"]], on="dg_id", how="left")
        .merge(future_agg, on="dg_id", how="left")
    )

    # --- FORCE dg_id dtype match before course history merge ---
    summary["dg_id"] = pd.to_numeric(summary["dg_id"], errors="coerce").astype("Int64")
    course_hist_df["dg_id"] = pd.to_numeric(course_hist_df["dg_id"], errors="coerce").astype("Int64")

    summary = summary.merge(course_hist_df, on="dg_id", how="left")

    # --- course sensitivity label (course_type) ---

    # ensure course_num is a column (Series) on summary
    if "course_num" in summary.columns:
        summary["course_num"] = pd.to_numeric(summary["course_num"], errors="coerce").astype("Int64")
    else:
        summary["course_num"] = pd.Series([course_num] * len(summary), index=summary.index, dtype="Int64")

    # merge course sensitivity lookup if available
    if course_sens_df is not None and not course_sens_df.empty and "course_num" in course_sens_df.columns:
        tmp = course_sens_df[["course_num", "course_type"]].copy()
        tmp["course_num"] = pd.to_numeric(tmp["course_num"], errors="coerce").astype("Int64")
        tmp["course_type"] = tmp["course_type"].astype(str).str.lower()

        summary = summary.merge(tmp, on="course_num", how="left")
        summary["course_type"] = summary["course_type"].fillna("mixed")
    else:
        summary["course_type"] = "mixed"

    # --- basic EV ratios (inspection only) ---
    if "ev_current" in summary.columns and "ev_future_max" in summary.columns:
        summary["ev_current_vs_future_max_pct"] = np.where(
            pd.to_numeric(summary["ev_future_max"], errors="coerce") > 0,
            pd.to_numeric(summary["ev_current"], errors="coerce") / pd.to_numeric(summary["ev_future_max"], errors="coerce"),
            np.nan,
        )

    if "ev_current" in summary.columns and "ev_future_total" in summary.columns:
        summary["ev_current_pct_of_future"] = np.where(
            pd.to_numeric(summary["ev_future_total"], errors="coerce") > 0,
            pd.to_numeric(summary["ev_current"], errors="coerce") / pd.to_numeric(summary["ev_future_total"], errors="coerce"),
            np.nan,
        )
    else:
        summary["ev_current_pct_of_future"] = pd.NA

    # one row per player in the field
    summary = summary.drop_duplicates(subset="dg_id", keep="first")

    # shortlist flags/tags
    summary = _attach_shortlist_flags(summary, season_year=yr, event_id=eid)

    # ---------------------------------------------------------------
    # Helper for min-max scaling
    # ---------------------------------------------------------------
    def _minmax_series(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if len(s) == 0:
            return pd.Series(0.0, index=s.index)
        mn = s.min()
        mx = s.max()
        if pd.isna(mn) or pd.isna(mx) or mx <= mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    # ---------------------------------------------------------------
    # Field-adjusted EV (current / future), but ONLY current EV in oad_score
    # ---------------------------------------------------------------
    field_size = len(field_df)
    field_factor = np.sqrt(156.0 / max(field_size, 1))

    summary["ev_current_adj"] = pd.to_numeric(summary.get("ev_current"), errors="coerce") * field_factor
    summary["ev_future_total_adj"] = pd.to_numeric(summary.get("ev_future_total"), errors="coerce") * field_factor
    summary["ev_future_max_adj"] = pd.to_numeric(summary.get("ev_future_max"), errors="coerce") * field_factor

    summary["ev_current_to_future_max_ratio"] = np.where(
        pd.to_numeric(summary["ev_future_max_adj"], errors="coerce") > 0,
        pd.to_numeric(summary["ev_current_adj"], errors="coerce") / pd.to_numeric(summary["ev_future_max_adj"], errors="coerce"),
        np.nan,
    )

    # ---------------------------------------------------------------
    # Recent form: z_sg_recent (L12 / L24 / L40)
    # ---------------------------------------------------------------
    idx = summary.index
    sg_L12 = _minmax_series(summary["sg_total_L12"]) if "sg_total_L12" in summary.columns else pd.Series(0.0, index=idx)
    sg_L24 = _minmax_series(summary["sg_total_L24"]) if "sg_total_L24" in summary.columns else pd.Series(0.0, index=idx)
    sg_L40 = _minmax_series(summary["sg_total_L40"]) if "sg_total_L40" in summary.columns else pd.Series(0.0, index=idx)

    summary["z_sg_recent"] = 0.25 * sg_L12 + 0.35 * sg_L24 + 0.40 * sg_L40

    # ---------------------------------------------------------------
    # YTD form: z_ytd (made_cut% + top10 count)
    # ---------------------------------------------------------------
    z_ytd_mc = _minmax_series(summary["ytd_made_cut_pct"]) if "ytd_made_cut_pct" in summary.columns else pd.Series(0.0, index=idx)
    z_ytd_t10 = _minmax_series(summary["ytd_top10"]) if "ytd_top10" in summary.columns else pd.Series(0.0, index=idx)
    summary["z_ytd"] = 0.55 * z_ytd_mc + 0.45 * z_ytd_t10

    # ---------------------------------------------------------------
    # Event history metric + majors / course-change logic
    # ---------------------------------------------------------------
    z_starts = _minmax_series(summary["starts_event"]) if "starts_event" in summary.columns else pd.Series(0.0, index=idx)
    z_mc = _minmax_series(summary["made_cut_pct_event"]) if "made_cut_pct_event" in summary.columns else pd.Series(0.0, index=idx)

    high_fin = pd.Series(0.0, index=idx)
    if "top10_event" in summary.columns:
        high_fin += pd.to_numeric(summary["top10_event"], errors="coerce").fillna(0.0)
    if "wins_event" in summary.columns:
        high_fin += pd.to_numeric(summary["wins_event"], errors="coerce").fillna(0.0)
    z_high_fin = _minmax_series(high_fin)

    z_sg_evt = _minmax_series(summary["avg_sg_total_event"]) if "avg_sg_total_event" in summary.columns else pd.Series(0.0, index=idx)

    summary["event_hist_raw"] = 0.25 * z_starts + 0.25 * z_mc + 0.25 * z_high_fin + 0.25 * z_sg_evt
    summary["event_hist_z"] = _minmax_series(summary["event_hist_raw"])

    # determine majors vs non-majors
    ev_name = str(sched_row["event_name"]).lower()
    is_major = any(
        key in ev_name
        for key in ["masters tournament", "u.s. open", "us open", "pga championship", "the open championship", "open championship"]
    )

    # detect course change (non-majors only): compare current course_num to last prior course_num for this event_id
    course_changed = False
    if not is_major and "course_num" in rounds_df.columns:
        hist_rounds = rounds_df[
            (pd.to_numeric(rounds_df["event_id"], errors="coerce") == eid) &
            (pd.to_numeric(rounds_df["year"], errors="coerce") < yr)
        ].copy()
        if not hist_rounds.empty:
            hist_rounds["course_num"] = pd.to_numeric(hist_rounds["course_num"], errors="coerce")
            hist_rounds = hist_rounds.dropna(subset=["course_num"])
            if not hist_rounds.empty:
                last_course = hist_rounds.sort_values("year")["course_num"].iloc[-1]
                course_changed = (float(last_course) != float(course_num))

    if is_major:
        summary["history_metric"] = summary["event_hist_z"]
    elif course_changed:
        summary["history_metric"] = 0.25 * summary["event_hist_z"]
    else:
        summary["history_metric"] = summary["event_hist_z"]

    summary["history_metric"] = pd.to_numeric(summary["history_metric"], errors="coerce").fillna(0.0)
    summary["z_history"] = _minmax_series(summary["history_metric"])

    # ---------------------------------------------------------------
    # EV component (current EV only, field-adjusted)
    # ---------------------------------------------------------------
    summary["z_ev_current"] = _minmax_series(summary["ev_current_adj"])

    # ---------------------------------------------------------------
    # Final OAD score (keep as-is)
    # ---------------------------------------------------------------
    summary["oad_score"] = (
        0.36 * summary["z_sg_recent"] +
        0.08 * summary["z_ev_current"] +
        0.28 * summary["z_ytd"] +
        0.28 * summary["z_history"]
    )

    # ---------------------------------------------------------------
    # Final rank score (your tier blend; keep as-is)
    # ---------------------------------------------------------------
    tier = _normalize_tier_from_schedule_row(sched_row)

    summary["pct_sg_total_L12"] = _pct_rank(summary["sg_total_L12"]) if "sg_total_L12" in summary.columns else np.nan
    summary["pct_ev_current_adj"] = _pct_rank(summary["ev_current_adj"]) if "ev_current_adj" in summary.columns else np.nan
    summary["pct_oad_score"] = _pct_rank(summary["oad_score"]) if "oad_score" in summary.columns else np.nan
    summary["pct_form"] = summary["pct_sg_total_L12"]

    if tier == "major":
        w_sg12, w_ev, w_oad = 1.00, 0.00, 0.00
    elif tier == "signature":
        w_sg12, w_ev, w_oad = 0.65, 0.15, 0.20
    else:
        w_sg12, w_ev, w_oad = 0.60, 0.30, 0.10

    summary["final_rank_score"] = (
        w_sg12 * pd.to_numeric(summary["pct_sg_total_L12"], errors="coerce").fillna(0.0) +
        w_ev   * pd.to_numeric(summary["pct_ev_current_adj"], errors="coerce").fillna(0.0) +
        w_oad  * pd.to_numeric(summary["pct_oad_score"], errors="coerce").fillna(0.0)
    )

    # ---------------------------------------------------------------
    # Decision score (FIXED ordering: define percentiles BEFORE using)
    # ---------------------------------------------------------------
    summary["pct_ytd_avg_sg_total"] = _pct_rank(summary["ytd_avg_sg_total"]) if "ytd_avg_sg_total" in summary.columns else np.nan
    summary["pct_ytd_made_cut_pct"] = _pct_rank(summary["ytd_made_cut_pct"]) if "ytd_made_cut_pct" in summary.columns else np.nan
    summary["pct_event_hist_sg"] = _pct_rank(summary["avg_sg_total_event"]) if "avg_sg_total_event" in summary.columns else np.nan
    summary["pct_course_hist_sg"] = _pct_rank(summary["course_avg_sg_total"]) if "course_avg_sg_total" in summary.columns else np.nan

    course_type = (
        str(summary["course_type"].iloc[0]).lower()
        if ("course_type" in summary.columns and not summary.empty)
        else "mixed"
    )

    if course_type == "form_only":
        if tier == "major":
            w_oad, w_form, w_ytd, w_evt, w_course = 0.55, 0.30, 0.15, 0.00, 0.00
        elif tier == "signature":
            w_oad, w_form, w_ytd, w_evt, w_course = 0.55, 0.30, 0.10, 0.05, 0.00
        else:
            w_oad, w_form, w_ytd, w_evt, w_course = 0.60, 0.30, 0.10, 0.00, 0.00

    elif course_type == "history_sensitive":
        if tier == "major":
            w_oad, w_form, w_ytd, w_evt, w_course = 0.45, 0.25, 0.15, 0.05, 0.10
        elif tier == "signature":
            w_oad, w_form, w_ytd, w_evt, w_course = 0.50, 0.25, 0.10, 0.05, 0.10
        else:
            w_oad, w_form, w_ytd, w_evt, w_course = 0.50, 0.25, 0.10, 0.05, 0.10

    else:  # mixed
        if tier == "major":
            w_oad, w_form, w_ytd, w_evt, w_course = 0.50, 0.25, 0.15, 0.05, 0.05
        elif tier == "signature":
            w_oad, w_form, w_ytd, w_evt, w_course = 0.55, 0.25, 0.10, 0.05, 0.05
        else:
            w_oad, w_form, w_ytd, w_evt, w_course = 0.55, 0.25, 0.10, 0.05, 0.05

    course_gate = (pd.to_numeric(summary["pct_form"], errors="coerce").fillna(0.0) >= 0.60).astype(float)

    summary["decision_score"] = (
        w_oad    * pd.to_numeric(summary["pct_oad_score"], errors="coerce").fillna(0.0) +
        w_form   * pd.to_numeric(summary["pct_form"], errors="coerce").fillna(0.0) +
        w_ytd    * pd.to_numeric(summary["pct_ytd_avg_sg_total"], errors="coerce").fillna(0.0) +
        w_evt    * pd.to_numeric(summary["pct_event_hist_sg"], errors="coerce").fillna(0.0) +
        w_course * pd.to_numeric(summary["pct_course_hist_sg"], errors="coerce").fillna(0.0) * course_gate
    )

    summary["decision_context"] = f"{tier}|{course_type}"

    # ------------------------------------------------------------------
    # Pattern analysis: numeric pattern scores + multi-year text summary
    # ------------------------------------------------------------------
    patterns = compute_event_patterns(yr, eid)
    pattern_scores = score_current_field_against_patterns(summary, patterns)

    if not pattern_scores.empty:
        summary = summary.merge(pattern_scores, on="dg_id", how="left")
    else:
        summary["pattern_score_winner"] = 0
        summary["pattern_flag_winner"] = False
        summary["pattern_score_top5"] = 0
        summary["pattern_flag_top5"] = False

    patterns_summary_text = build_event_patterns_text_from_rounds(
        season_year=yr,
        event_id=eid,
        min_year=2017,
    )

    # ------------------------------------------------------------------
    # PERFORMANCE TABLE
    # ------------------------------------------------------------------
    perf_cols_base = [
        "dg_id",
        "oad_score",
        "pct_sg_total_L12",
        "pct_ev_current_adj",
        "pct_oad_score",
        "final_rank_score",
        "decision_score",
        "decision_context",
        "decimal_odds",
        "ev_current",
        "ev_current_adj",
        "ev_future_total",
        "ev_future_max",
        "ev_future_total_adj",
        "ev_future_max_adj",
        "ev_current_to_future_max_ratio",
        "z_sg_recent",
        "z_ytd",
        "z_history",
        "z_ev_current",
        "course_fit_score",
        "is_shortlist",
        "tagged_here",
        "pattern_score_winner",
        "pattern_score_top5",
    ]

    rolling_cols = []
    for w in (40, 24, 12):
        for stat in ["sg_total", "sg_app", "sg_putt", "round_score"]:
            col = f"{stat}_L{w}"
            if col in summary.columns:
                rolling_cols.append(col)

    perf_cols = [c for c in perf_cols_base if c in summary.columns] + rolling_cols
    table_performance = summary[perf_cols].copy()

    sort_col = "decision_score" if "decision_score" in table_performance.columns else (
        "final_rank_score" if "final_rank_score" in table_performance.columns else (
            "oad_score" if "oad_score" in table_performance.columns else "ev_current"
        )
    )
    table_performance = table_performance.sort_values(sort_col, ascending=False)

    table_performance = _round_float_columns(table_performance)

    # ------------------------------------------------------------------
    # EVENT HISTORY TABLE
    # ------------------------------------------------------------------
    event_hist_cols = [
        "dg_id",
        "starts_event",
        "made_cuts_event",
        "made_cut_pct_event",
        "top25_event",
        "top10_event",
        "top5_event",
        "wins_event",
        "prev_finish_num_event",
        "avg_score_event",
        "avg_sg_total_event",
    ]
    event_hist_cols = [c for c in event_hist_cols if c in summary.columns]
    table_event_history = summary[event_hist_cols].copy()

    # ------------------------------------------------------------------
    # YTD TABLE
    # ------------------------------------------------------------------
    ytd_cols = [
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
    ytd_cols = [c for c in ytd_cols if c in summary.columns]
    table_ytd = summary[ytd_cols].copy()

    # --- top views ---
    table_performance_top50 = table_performance.head(25).copy()

    top_dg_ids = table_performance_top50["dg_id"].unique().tolist()
    order_map = {dg: i for i, dg in enumerate(top_dg_ids)}

    table_event_history_top50 = table_event_history[table_event_history["dg_id"].isin(top_dg_ids)].copy()
    if not table_event_history_top50.empty:
        table_event_history_top50["__order"] = table_event_history_top50["dg_id"].map(order_map)
        table_event_history_top50 = table_event_history_top50.sort_values("__order").drop(columns="__order")

    table_ytd_top50 = table_ytd[table_ytd["dg_id"].isin(top_dg_ids)].copy()
    if not table_ytd_top50.empty:
        table_ytd_top50["__order"] = table_ytd_top50["dg_id"].map(order_map)
        table_ytd_top50 = table_ytd_top50.sort_values("__order").drop(columns="__order")

    table_performance_top50 = _round_float_columns(table_performance_top50)
    table_event_history_top50 = _round_float_columns(table_event_history_top50)
    table_ytd_top50 = _round_float_columns(table_ytd_top50)

    # ------------------------------------------------------------------
    # Pattern candidates
    # ------------------------------------------------------------------
    MAX_PATTERN_CANDIDATES = 20
    valid_field_dg = set(field_df["dg_id"].dropna().astype(int))

    pattern_candidates = summary[
        ((summary["pattern_flag_winner"] == True) | (summary["pattern_flag_top5"] == True))
        & (summary["dg_id"].isin(valid_field_dg))
    ].copy()

    if "ev_current" in pattern_candidates.columns:
        pattern_candidates = pattern_candidates.sort_values(
            ["pattern_flag_winner", "pattern_score_winner", "ev_current"],
            ascending=[False, False, False],
        )

    name_cols = [c for c in pattern_candidates.columns if "player_name" in c]
    if name_cols:
        pattern_candidates = pattern_candidates.drop(columns=name_cols)

    pattern_candidates = pattern_candidates.head(MAX_PATTERN_CANDIDATES).copy()
    pattern_candidates = _round_float_columns(pattern_candidates)

    return {
        "schedule_row": sched_row.to_frame().T,
        "field": field_df,
        "rolling": rolling_df,
        "event_history": event_hist_df,
        "ytd": ytd_df,
        "course_fit": course_fit_scores,
        "course_history": course_hist_df,
        "ev_current": ev_current_df,
        "ev_future": ev_future_df,
        "summary": summary,
        "table_performance": table_performance,
        "table_event_history": table_event_history,
        "table_ytd": table_ytd,
        "table_performance_top50": table_performance_top50,
        "table_event_history_top50": table_event_history_top50,
        "table_ytd_top50": table_ytd_top50,
        "pattern_candidates": pattern_candidates,
        "patterns_summary_text": patterns_summary_text,
        "patterns_winners_features": patterns.winners_features,
        "patterns_top5_features": patterns.top5_features,
    }