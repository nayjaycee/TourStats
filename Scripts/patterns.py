# Scripts/patterns.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from Scripts.data_io import load_rounds, load_odds_and_results, load_event_skill, load_course_fit
from Scripts.schedule import build_season_schedule
from Scripts.features import compute_event_history, compute_ytd_stats, compute_rolling_stats
# and your EventPatterns, PatternThresholds definitions, etc.


@dataclass
class PatternThresholds:
    # event history
    min_starts_event: Optional[float] = None
    min_top25_event: Optional[float] = None
    min_top10_event: Optional[float] = None
    min_top5_event: Optional[float] = None

    # YTD volume/form
    min_ytd_starts: Optional[float] = None
    min_ytd_made_cut_pct: Optional[float] = None
    min_ytd_top25: Optional[float] = None
    min_ytd_top10: Optional[float] = None
    min_ytd_top5: Optional[float] = None

    # rolling SG totals
    min_sg_total_L24: Optional[float] = None
    min_sg_total_L12: Optional[float] = None

    # rolling SG components (L24)
    min_sg_ott_L24: Optional[float] = None
    min_sg_app_L24: Optional[float] = None
    min_sg_arg_L24: Optional[float] = None
    min_sg_putt_L24: Optional[float] = None



@dataclass
class EventPatterns:
    winners_features: pd.DataFrame
    top5_features: pd.DataFrame
    winner_thresholds: PatternThresholds
    top5_thresholds: PatternThresholds
    summary_text: str


def _get_event_date_for_year(
    event_id: int,
    year: int,
    event_skill_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> Optional[pd.Timestamp]:
    """
    Get the event_date for a given (event_id, year).
    Prefer event_skill.xlsx, fall back to the max event_completed
    in Odds_and_Results for that year/event.
    """
    eid = int(event_id)
    yr = int(year)

    es = event_skill_df[
        (event_skill_df["event_id"] == eid) & (event_skill_df["year"] == yr)
    ]
    if not es.empty and "event_date" in es.columns:
        dt = pd.to_datetime(es.iloc[0]["event_date"], errors="coerce")
        if not pd.isna(dt):
            return dt

    # fallback: max event_completed in odds_df
    df = odds_df[
        (odds_df["event_id"] == eid) & (odds_df["year"] == yr)
    ].copy()
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
        dt = df["event_completed"].max()
        if not pd.isna(dt):
            return dt

    return None

def _compute_thresholds_from_features(feat: pd.DataFrame) -> PatternThresholds:
    """
    Given a features DataFrame for winners or top-5, compute
    simple thresholds used to flag pattern matches in the current field.

    Heuristics:
      - If >=50% had >=1 prior start/top-25/top-10/top-5 here, require at least 1.
      - YTD thresholds and SG thresholds are medians.
      - SG component thresholds are medians, but only if the median is > 0
        (i.e. we only enforce components that were typically positive).
    """
    if feat.empty:
        return PatternThresholds()

    out = PatternThresholds()

    # -------- Event history: starts / top finishes at this event --------
    if "starts_event" in feat.columns:
        mask = feat["starts_event"].fillna(0) >= 1
        pct = mask.mean()
        out.min_starts_event = 1.0 if pct >= 0.5 else 0.0

    for col, attr in [
        ("top25_event", "min_top25_event"),
        ("top10_event", "min_top10_event"),
        ("top5_event",  "min_top5_event"),
    ]:
        if col in feat.columns:
            mask = feat[col].fillna(0) >= 1
            pct = mask.mean()
            setattr(out, attr, 1.0 if pct >= 0.5 else 0.0)

    # -------- YTD volume / form --------
    if "ytd_starts" in feat.columns:
        out.min_ytd_starts = float(np.nanmedian(feat["ytd_starts"]))

    if "ytd_made_cut_pct" in feat.columns:
        val = np.nanmedian(feat["ytd_made_cut_pct"])
        out.min_ytd_made_cut_pct = None if np.isnan(val) else float(val)

    for col, attr in [
        ("ytd_top25", "min_ytd_top25"),
        ("ytd_top10", "min_ytd_top10"),
        ("ytd_top5",  "min_ytd_top5"),
    ]:
        if col in feat.columns:
            val = np.nanmedian(feat[col])
            setattr(out, attr, None if np.isnan(val) else float(val))

    # -------- Rolling SG totals --------
    if "sg_total_L24" in feat.columns:
        out.min_sg_total_L24 = float(np.nanmedian(feat["sg_total_L24"]))
    if "sg_total_L12" in feat.columns:
        out.min_sg_total_L12 = float(np.nanmedian(feat["sg_total_L12"]))

    # -------- Rolling SG components (L24) --------
    comp_map = {
        "sg_ott_L24":  "min_sg_ott_L24",
        "sg_app_L24":  "min_sg_app_L24",
        "sg_arg_L24":  "min_sg_arg_L24",
        "sg_putt_L24": "min_sg_putt_L24",
    }
    for col, attr in comp_map.items():
        if col in feat.columns:
            med_val = np.nanmedian(feat[col])
            # Only enforce if winners/top-5 were typically *positive* here
            if not np.isnan(med_val) and med_val > 0.0:
                setattr(out, attr, float(med_val))

    return out


def compute_event_patterns(
    season_year: int,
    event_id: int,
) -> EventPatterns:
    """
    Compute historical patterns for winners and top-5 finishers
    at a given event (event_id), using data from years < season_year.

    For each past year where this event was played:
      - Find winners (Win == 1) and top-5 (Top_5 == 1).
      - For that year's event date, compute:
          * event_history
          * ytd_stats
          * rolling stats (L40/L24/L12) for those players
      - Attach these features to winners/top-5.

    Returns an EventPatterns object containing:
      - winners_features: one row per (year, dg_id) winner
      - top5_features: one row per (year, dg_id) top-5
      - winner_thresholds: pattern thresholds from winners
      - top5_thresholds: pattern thresholds from top-5
      - summary_text: human-readable summary of patterns
    """
    # stricter thresholds: only call a % pattern out if >= 70%
    MIN_PCT_PATTERN_WINNER = 0.70
    MIN_PCT_PATTERN_TOP5   = 0.70

    yr_target = int(season_year)
    eid = int(event_id)

    rounds_df = load_rounds()
    odds_df = load_odds_and_results()
    event_skill_df = load_event_skill()

    for col in ["year", "event_id", "dg_id"]:
        if col in odds_df.columns:
            odds_df[col] = pd.to_numeric(odds_df[col], errors="coerce")

    hist = odds_df[
        (odds_df["event_id"] == eid) & (odds_df["year"] < yr_target)
    ].copy()
    if hist.empty:
        return EventPatterns(
            winners_features=pd.DataFrame(),
            top5_features=pd.DataFrame(),
            winner_thresholds=PatternThresholds(),
            top5_thresholds=PatternThresholds(),
            summary_text="No historical data for this event before this season.",
        )

    # Ensure flags exist
    for col in ["Win", "Top_5"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0).astype(int)
        else:
            hist[col] = 0

    winners_feats_list: List[pd.DataFrame] = []
    top5_feats_list: List[pd.DataFrame] = []

    years = sorted(hist["year"].dropna().unique().tolist())
    for y in years:
        sub = hist[hist["year"] == y].copy()
        if sub.empty:
            continue

        event_date = _get_event_date_for_year(eid, y, event_skill_df, odds_df)
        if event_date is None:
            continue

        # winners / top5 for this year
        winners_y = sub[sub["Win"] == 1].copy()
        top5_y = sub[sub["Top_5"] == 1].copy()

        if winners_y.empty and top5_y.empty:
            continue

        # players we care about in this year
        dg_ids_y = pd.concat(
            [winners_y["dg_id"], top5_y["dg_id"]], ignore_index=True
        ).dropna().unique().astype(int).tolist()

        # as-of snapshot features for that event-date
        ev_hist_y = compute_event_history(
            rounds_df=rounds_df,
            odds_df=odds_df,
            event_id=eid,
            as_of_date=event_date,
        )
        ytd_y = compute_ytd_stats(
            rounds_df=rounds_df,
            odds_df=odds_df,
            season_year=int(y),
            as_of_date=event_date,
        )
        rolling_y = compute_rolling_stats(
            rounds_df=rounds_df,
            as_of_date=event_date,
            dg_ids=dg_ids_y,
            windows=(40, 24, 12),
        )

        # Merge features for winners
        if not winners_y.empty:
            wf = (
                winners_y[["year", "dg_id"]]
                .drop_duplicates()
                .merge(ev_hist_y, on="dg_id", how="left")
                .merge(ytd_y, on="dg_id", how="left")
                .merge(rolling_y, on="dg_id", how="left")
            )
            winners_feats_list.append(wf)

        # Merge features for top-5
        if not top5_y.empty:
            tf = (
                top5_y[["year", "dg_id"]]
                .drop_duplicates()
                .merge(ev_hist_y, on="dg_id", how="left")
                .merge(ytd_y, on="dg_id", how="left")
                .merge(rolling_y, on="dg_id", how="left")
            )
            top5_feats_list.append(tf)

    if winners_feats_list:
        winners_features = pd.concat(winners_feats_list, ignore_index=True)
    else:
        winners_features = pd.DataFrame()

    if top5_feats_list:
        top5_features = pd.concat(top5_feats_list, ignore_index=True)
    else:
        top5_features = pd.DataFrame()

    winner_thresholds = _compute_thresholds_from_features(winners_features)
    top5_thresholds = _compute_thresholds_from_features(top5_features)

    summary_lines: List[str] = []

    # ---------------- Winners summary -----------------
    if not winners_features.empty:
        n_w = len(winners_features)
        summary_lines.append(f"Winners sample size: {n_w}")

        if "starts_event" in winners_features.columns:
            pct_start = (winners_features["starts_event"].fillna(0) >= 1).mean()
            if pct_start >= MIN_PCT_PATTERN_WINNER:
                summary_lines.append(
                    f"- {pct_start:.0%} of winners had at least 1 prior start at this event."
                )

        # event-history patterns
        for col, label in [
            ("top25_event", "prior top-25 here"),
            ("top10_event", "prior top-10 here"),
            ("top5_event",  "prior top-5 here"),
        ]:
            if col in winners_features.columns:
                pct = (winners_features[col].fillna(0) >= 1).mean()
                if pct >= MIN_PCT_PATTERN_WINNER:
                    summary_lines.append(
                        f"- {pct:.0%} of winners had at least one {label}."
                    )

        # YTD volume / form
        if "ytd_made_cut_pct" in winners_features.columns:
            med_ytd_cut = np.nanmedian(winners_features["ytd_made_cut_pct"])
            if not np.isnan(med_ytd_cut):
                summary_lines.append(
                    f"- Median YTD made-cut% for winners: {med_ytd_cut:.0%}"
                )

        for col, label in [
            ("ytd_top25", "YTD top-25s"),
            ("ytd_top10", "YTD top-10s"),
            ("ytd_top5",  "YTD top-5s"),
        ]:
            if col in winners_features.columns:
                med_val = np.nanmedian(winners_features[col])
                if not np.isnan(med_val) and med_val > 0:
                    summary_lines.append(
                        f"- Median {label} for winners: {med_val:.1f}"
                    )

        # Rolling SG totals
        if "sg_total_L24" in winners_features.columns:
            med_sg24 = np.nanmedian(winners_features["sg_total_L24"])
            if not np.isnan(med_sg24):
                summary_lines.append(
                    f"- Median SG_total L24 for winners before this event: {med_sg24:+.2f}"
                )

        # SG-category patterns (L24)
        sg_map = {
            "sg_ott_L24": "OTT",
            "sg_app_L24": "APP",
            "sg_arg_L24": "ARG",
            "sg_putt_L24": "PUTT",
        }
        pos_parts = []
        for col, label in sg_map.items():
            if col in winners_features.columns:
                med_val = np.nanmedian(winners_features[col])
                if not np.isnan(med_val) and med_val > 0.0:
                    pos_parts.append(f"{label} (+{med_val:.2f})")
        if pos_parts:
            summary_lines.append(
                "- For winners, positive L24 median SG in: " + ", ".join(pos_parts)
            )

    else:
        summary_lines.append("No historical winners data for this event.")

    # ---------------- Top-5 summary -------------------
    if not top5_features.empty:
        n_t = len(top5_features)
        summary_lines.append(f"Top-5 sample size: {n_t}")

        if "ytd_made_cut_pct" in top5_features.columns:
            med_ytd_cut_t = np.nanmedian(top5_features["ytd_made_cut_pct"])
            if not np.isnan(med_ytd_cut_t):
                summary_lines.append(
                    f"- Median YTD made-cut% for top-5: {med_ytd_cut_t:.0%}"
                )

        if "sg_total_L24" in top5_features.columns:
            med_sg24_t = np.nanmedian(top5_features["sg_total_L24"])
            if not np.isnan(med_sg24_t):
                summary_lines.append(
                    f"- Median SG_total L24 for top-5 finishers before this event: {med_sg24_t:.2f}"
                )

        # SG-category patterns (L24) for top-5
        sg_map = {
            "sg_ott_L24": "OTT",
            "sg_app_L24": "APP",
            "sg_arg_L24": "ARG",
            "sg_putt_L24": "PUTT",
        }
        pos_parts_t = []
        for col, label in sg_map.items():
            if col in top5_features.columns:
                med_val = np.nanmedian(top5_features[col])
                if not np.isnan(med_val) and med_val > 0.0:
                    pos_parts_t.append(f"{label} (+{med_val:.2f})")
        if pos_parts_t:
            summary_lines.append(
                "- For top-5, positive L24 median SG in: " + ", ".join(pos_parts_t)
            )

    else:
        summary_lines.append("No historical top-5 data for this event.")

    # ---------------- Course fit summary -------------------
    try:
        sched_df = build_season_schedule(yr_target)
        row = sched_df[sched_df["event_id"] == eid].iloc[0]
        course_num = int(row["course_num"])
        course_fit_df = load_course_fit(yr_target)
        cf_row = course_fit_df[course_fit_df["course_num"] == course_num].iloc[0]

        imp_vals = {
            "DIST": cf_row.get("imp_dist", 0.0),
            "ACC": cf_row.get("imp_acc", 0.0),
            "APP": cf_row.get("imp_app", 0.0),
            "ARG": cf_row.get("imp_arg", 0.0),
            "PUTT": cf_row.get("imp_putt", 0.0),
        }
        # sort by importance desc
        sorted_imp = sorted(imp_vals.items(), key=lambda x: x[1], reverse=True)
        top_parts = [f"{k} ({v:.2f})" for k, v in sorted_imp]

        summary_lines.append(
            "Course fit profile: most important stats are " + ", ".join(top_parts[:3]) +
            "; least important are " + ", ".join(top_parts[-2:])
        )
    except Exception:
        # if anything fails (no schedule/course_fit), just skip this line
        pass

    summary_text = "\n".join(summary_lines)

    return EventPatterns(
        winners_features=winners_features,
        top5_features=top5_features,
        winner_thresholds=winner_thresholds,
        top5_thresholds=top5_thresholds,
        summary_text=summary_text,
    )

def _build_condition_list(
    df: pd.DataFrame,
    t: PatternThresholds,
) -> List[pd.Series]:
    """
    Build a list of boolean Series, one per criterion, for a given threshold set.
    Only adds conditions where both the threshold and the column exist.
    """
    conds: List[pd.Series] = []

    # ---- Event history ----
    if t.min_starts_event is not None and "starts_event" in df.columns:
        conds.append(df["starts_event"].fillna(0) >= t.min_starts_event)

    if t.min_top25_event is not None and "top25_event" in df.columns:
        conds.append(df["top25_event"].fillna(0) >= t.min_top25_event)

    if t.min_top10_event is not None and "top10_event" in df.columns:
        conds.append(df["top10_event"].fillna(0) >= t.min_top10_event)

    if t.min_top5_event is not None and "top5_event" in df.columns:
        conds.append(df["top5_event"].fillna(0) >= t.min_top5_event)

    # ---- YTD volume / form ----
    if t.min_ytd_starts is not None and "ytd_starts" in df.columns:
        conds.append(df["ytd_starts"].fillna(0) >= t.min_ytd_starts)

    if t.min_ytd_made_cut_pct is not None and "ytd_made_cut_pct" in df.columns:
        conds.append(df["ytd_made_cut_pct"].fillna(0) >= t.min_ytd_made_cut_pct)

    if t.min_ytd_top25 is not None and "ytd_top25" in df.columns:
        conds.append(df["ytd_top25"].fillna(0) >= t.min_ytd_top25)

    if t.min_ytd_top10 is not None and "ytd_top10" in df.columns:
        conds.append(df["ytd_top10"].fillna(0) >= t.min_ytd_top10)

    if t.min_ytd_top5 is not None and "ytd_top5" in df.columns:
        conds.append(df["ytd_top5"].fillna(0) >= t.min_ytd_top5)

    # ---- Rolling SG totals ----
    if t.min_sg_total_L24 is not None and "sg_total_L24" in df.columns:
        conds.append(df["sg_total_L24"].fillna(-999) >= t.min_sg_total_L24)

    if t.min_sg_total_L12 is not None and "sg_total_L12" in df.columns:
        conds.append(df["sg_total_L12"].fillna(-999) >= t.min_sg_total_L12)

    # ---- Rolling SG components (L24) ----
    comp_thresh = [
        ("sg_ott_L24",  t.min_sg_ott_L24),
        ("sg_app_L24",  t.min_sg_app_L24),
        ("sg_arg_L24",  t.min_sg_arg_L24),
        ("sg_putt_L24", t.min_sg_putt_L24),
    ]
    for col, thr in comp_thresh:
        if thr is not None and col in df.columns:
            conds.append(df[col].fillna(-999) >= thr)

    return conds


def _score_and_flag(conds: List[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a list of boolean Series into:
      - integer scores (# of criteria met)
      - boolean flags (strong pattern match)

    Heuristic for flag:
      - if 0 conditions: all zeros / all False
      - if 1 condition: flag if score >= 1
      - else: flag if score >= max(2, ceil(0.6 * n_conds))
    """
    n = len(conds[0]) if conds else 0
    if not conds:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=bool)

    cond_matrix = np.column_stack([c.astype(int).to_numpy() for c in conds])
    scores = cond_matrix.sum(axis=1)

    k = cond_matrix.shape[1]
    if k == 1:
        flags = scores >= 1
    else:
        thresh = max(2, int(np.ceil(0.6 * k)))
        flags = scores >= thresh

    return scores, flags


def score_current_field_against_patterns(
    summary_df: pd.DataFrame,
    patterns: EventPatterns,
) -> pd.DataFrame:
    """
    Given the weekly 'summary' table for the current field and the
    historical EventPatterns for this event, assign:

      - pattern_score_winner: how many winner-pattern criteria are satisfied
      - pattern_flag_winner: True if strong winner-pattern match
      - pattern_score_top5:   how many top-5 criteria are satisfied
      - pattern_flag_top5

    The exact criteria come from PatternThresholds, which are derived
    from historical feature medians / frequencies.
    """
    if summary_df.empty or "dg_id" not in summary_df.columns:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "pattern_score_winner",
                "pattern_flag_winner",
                "pattern_score_top5",
                "pattern_flag_top5",
            ]
        )

    df = summary_df.copy()
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")

    # winner side
    conds_w = _build_condition_list(df, patterns.winner_thresholds)
    score_w, flag_w = _score_and_flag(conds_w) if conds_w else (np.zeros(len(df), int), np.zeros(len(df), bool))

    # top-5 side
    conds_t = _build_condition_list(df, patterns.top5_thresholds)
    score_t, flag_t = _score_and_flag(conds_t) if conds_t else (np.zeros(len(df), int), np.zeros(len(df), bool))

    out = pd.DataFrame(
        {
            "dg_id": df["dg_id"].to_numpy(),
            "pattern_score_winner": score_w,
            "pattern_flag_winner": flag_w,
            "pattern_score_top5": score_t,
            "pattern_flag_top5": flag_t,
        }
    )

    return out


def build_event_patterns_text_from_rounds(
    season_year: int,
    event_id: int,
    min_year: int = 2017,
) -> str:
    """
    Build a natural-language pattern summary for a given event_id, using
    ALL historical data in combined_rounds (years >= min_year and < season_year).

    Looks at:
      - Winners at this event across all years
      - Top-5 finishers at this event across all years
      - For each winner/top-5 instance, checks:
          * Did they have a prior start at this event?
          * Did they have a prior top-25 / top-10 / top-5 at this event?
          * What was their sg_total L24 leading into the event (last 24 rounds)?

    Returns:
      A multi-line string with *only positive statements*:
        - No "0% of winners..." bullets.
        - Skips bullets when the relevant count is zero or there's no data.
    """
    season_year = int(season_year)
    event_id = int(event_id)

    rounds = load_rounds()

    # ensure dtypes
    for col in ["year", "event_id", "dg_id", "finish_num"]:
        if col in rounds.columns:
            rounds[col] = pd.to_numeric(rounds[col], errors="coerce")

    # restrict to this event and historical years
    hist = rounds[
        (rounds["event_id"] == event_id)
        & (rounds["year"] >= min_year)
        & (rounds["year"] < season_year)
    ].copy()

    if hist.empty:
        return (
            f"No historical winners/top-5 data for this event before {season_year} "
            f"(years >= {min_year})."
        )

    # use finish_num per event-year-player; sg_total for performance
    hist["sg_total"] = pd.to_numeric(hist["sg_total"], errors="coerce")

    # aggregate per event-year-player
    ev_year = (
        hist.groupby(["year", "event_id", "dg_id"], as_index=False)
        .agg(
            sg_event=("sg_total", "sum"),
            finish_num=("finish_num", "min"),
            fin_text=("fin_text", "first"),
            event_name=("event_name", "first"),
            first_round_date=("round_date", "min"),
        )
    )

    # use most recent event_name before season_year for display
    latest_name_row = ev_year.sort_values("year").iloc[-1]
    event_name_latest = str(latest_name_row.get("event_name", f"Event {event_id}"))

    # determine winners and top-5
    ev_year["finish_num"] = pd.to_numeric(ev_year["finish_num"], errors="coerce")
    winners = ev_year[ev_year["finish_num"] == 1].copy()
    top5 = ev_year[ev_year["finish_num"] <= 5].copy()

    # helper to compute prior starts / prior top finishes and L24 sg_total
    def attach_prior_and_L24(df_players: pd.DataFrame) -> pd.DataFrame:
        if df_players.empty:
            df_players["prior_start_here"] = pd.Series([], dtype=bool)
            df_players["prior_top25_here"] = pd.Series([], dtype=bool)
            df_players["prior_top10_here"] = pd.Series([], dtype=bool)
            df_players["prior_top5_here"] = pd.Series([], dtype=bool)
            df_players["sg_L24_pre"] = pd.Series([], dtype=float)
            return df_players

        df_players = df_players.copy()
        rounds_all = rounds.copy()

        # pre-compute for speed: sort rounds
        rounds_all = rounds_all.sort_values(
            ["dg_id", "round_date", "event_id", "round_num"],
            ascending=[True, True, True, True],
        )

        prior_start_flags = []
        prior_top25_flags = []
        prior_top10_flags = []
        prior_top5_flags = []
        sg_L24_vals = []

        for _, row in df_players.iterrows():
            yr = int(row["year"])
            did = int(row["dg_id"])

            # prior history at this same event (any earlier year)
            prior_ev = ev_year[
                (ev_year["event_id"] == event_id)
                & (ev_year["dg_id"] == did)
                & (ev_year["year"] < yr)
            ].copy()

            prior_start_here = not prior_ev.empty
            prior_top25_here = False
            prior_top10_here = False
            prior_top5_here = False

            if not prior_ev.empty:
                fn = pd.to_numeric(prior_ev["finish_num"], errors="coerce")
                prior_top25_here = bool((fn <= 25).any())
                prior_top10_here = bool((fn <= 10).any())
                prior_top5_here = bool((fn <= 5).any())

            prior_start_flags.append(prior_start_here)
            prior_top25_flags.append(prior_top25_here)
            prior_top10_flags.append(prior_top10_here)
            prior_top5_flags.append(prior_top5_here)

            # L24 SG leading into this event: use all rounds before first_round_date
            cutoff = row.get("first_round_date", pd.NaT)
            if pd.isna(cutoff):
                sg_L24_vals.append(np.nan)
                continue

            pl_rounds = rounds_all[
                (rounds_all["dg_id"] == did)
                & (rounds_all["round_date"] < cutoff)
            ].copy()
            if pl_rounds.empty:
                sg_L24_vals.append(np.nan)
                continue

            pl_rounds = pl_rounds.sort_values("round_date", ascending=False)
            pl_rounds = pl_rounds.head(24)
            sg_vals = pd.to_numeric(pl_rounds["sg_total"], errors="coerce").dropna()
            if sg_vals.empty:
                sg_L24_vals.append(np.nan)
            else:
                sg_L24_vals.append(float(sg_vals.mean()))

        df_players["prior_start_here"] = prior_start_flags
        df_players["prior_top25_here"] = prior_top25_flags
        df_players["prior_top10_here"] = prior_top10_flags
        df_players["prior_top5_here"]  = prior_top5_flags
        df_players["sg_L24_pre"] = sg_L24_vals

        return df_players

    winners = attach_prior_and_L24(winners)
    top5 = attach_prior_and_L24(top5)

    # summary stats
    lines = []
    lines.append(f"Patterns for {event_name_latest} (event_id {event_id})")
    lines.append(f"History window: {min_year}–{season_year-1}")

    # Winners block
    if not winners.empty:
        n_w = len(winners)
        lines.append(f"Winners sample size: {n_w}")

        def _pct_line(flag_col: str, label: str) -> None:
            if flag_col in winners.columns:
                share = float(winners[flag_col].mean())
                if share > 0:  # only positive statements
                    pct = int(round(share * 100))
                    lines.append(f"- {pct}% of winners had {label}.")

        _pct_line("prior_start_here",  "at least 1 prior start at this event")
        _pct_line("prior_top25_here",  "at least one prior top-25 here")
        _pct_line("prior_top10_here",  "at least one prior top-10 here")
        _pct_line("prior_top5_here",   "at least one prior top-5 here")

        # L24 sg_total
        w_L24 = winners["sg_L24_pre"].dropna()
        if len(w_L24) > 0:
            med_L24 = float(w_L24.median())
            lines.append(
                f"- Median SG_total L24 for winners before this event: {med_L24:+.2f}"
            )
    else:
        lines.append("No historical winners found for this event in the given window.")

    # Top-5 block
    if not top5.empty:
        n_t5 = len(top5)
        lines.append(f"Top-5 sample size: {n_t5}")

        t5_L24 = top5["sg_L24_pre"].dropna()
        if len(t5_L24) > 0:
            med_L24_t5 = float(t5_L24.median())
            lines.append(
                f"- Median SG_total L24 for top-5 finishers before this event: {med_L24_t5:+.2f}"
            )

        # optional positive bullets
        if "prior_start_here" in top5.columns:
            share_t5_prior_start = float(top5["prior_start_here"].mean())
            if share_t5_prior_start > 0:
                pct = int(round(share_t5_prior_start * 100))
                lines.append(
                    f"- {pct}% of top-5 finishers had at least 1 prior start here."
                )

        if "prior_top25_here" in top5.columns:
            share_t5_prior_t25 = float(top5["prior_top25_here"].mean())
            if share_t5_prior_t25 > 0:
                pct = int(round(share_t5_prior_t25 * 100))
                lines.append(
                    f"- {pct}% of top-5 finishers had at least one prior top-25 here."
                )

    else:
        lines.append("No historical top-5 data for this event in the given window.")

    return "\n".join(lines)