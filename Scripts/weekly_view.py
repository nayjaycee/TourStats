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
from Scripts.field_sim import get_actual_field
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

COURSE_SENSITIVITY_PATH = Path("/Data/Clean/Processed/course_sensitivity_table.csv")
try:
    course_sens_df = pd.read_csv(COURSE_SENSITIVITY_PATH)
    course_sens_df["course_num"] = pd.to_numeric(course_sens_df["course_num"], errors="coerce").astype("Int64")
    # expect column name: course_type
    course_sens_df["course_type"] = course_sens_df["course_type"].astype(str).str.lower()
except Exception:
    course_sens_df = pd.DataFrame(columns=["course_num", "course_type"])


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

def _round_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round float columns for readability:
      - ev_current, ev_future_total: 0 decimals (whole dollars)
      - everything else: 2 decimals
    """
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    for col in float_cols:
        if col in ("ev_current", "ev_future_total"):
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

    cutoff_date = event_date - pd.Timedelta(days=1)

    # Hard-code Players Championship 2025 only (schedule date is Monday, but completion is Sunday)
    if yr == 2025 and eid == 11:
        cutoff_date = pd.Timestamp("2025-03-16")

    # --- core data ---
    rounds_df = load_rounds()
    odds_df = load_odds_and_results()
    if "year" in odds_df.columns:
        odds_df = odds_df[pd.to_numeric(odds_df["year"], errors="coerce") == yr].copy()
    course_fit_df = load_course_fit(yr)
    player_skills_df = load_player_skills(yr)

    # --- actual field (for backtests) ---
    field_df = get_actual_field(odds_df, yr, eid)
    dg_ids = field_df["dg_id"].dropna().astype(int).unique().tolist()

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

    # --- YTD stats for current season ---
    ytd_df = compute_ytd_stats(
        rounds_df,
        odds_df,
        season_year=yr,
        as_of_date=cutoff_date,
        ytd_tracker=ytd_tracker,  # module-level
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
        .merge(course_hist_df, on="dg_id", how="left")
    )

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

    summary["z_sg_recent"] = 0.40 * sg_L12 + 0.35 * sg_L24 + 0.25 * sg_L40

    # ---------------------------------------------------------------
    # YTD form: z_ytd (made_cut% + top10 count)
    # ---------------------------------------------------------------
    z_ytd_mc = _minmax_series(summary["ytd_made_cut_pct"]) if "ytd_made_cut_pct" in summary.columns else pd.Series(0.0, index=idx)
    z_ytd_t10 = _minmax_series(summary["ytd_top10"]) if "ytd_top10" in summary.columns else pd.Series(0.0, index=idx)
    summary["z_ytd"] = 0.60 * z_ytd_mc + 0.40 * z_ytd_t10

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
        0.40 * summary["z_sg_recent"] +
        0.15 * summary["z_ev_current"] +
        0.20 * summary["z_ytd"] +
        0.20 * summary["z_history"]
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

    course_type = str(summary["course_type"].iloc[0]).lower() if "course_type" in summary.columns else "mixed"

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