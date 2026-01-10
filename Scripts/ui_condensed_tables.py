# Streamlit/ui_condensed_tables.py

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Column groupings (your spec)
# -----------------------------
CONDENSED_GROUPS: Dict[str, List[str]] = {
    "Identifiers / Context": [
        "dg_id",
        "is_shortlist",
        "decision_context",
        "course_type",
    ],
    "Decision scores": [
        "decision_score",
        "final_rank_score",
        "oad_score",
    ],
    "Percentile drivers": [
        "pct_ytd_avg_sg_total",
        "pct_ytd_made_cut_pct",
        "pct_event_hist_sg",
        "pct_course_hist_sg",
        "pct_sg_total_L12",
        "pct_ev_current_adj",
        "pct_oad_score",
    ],
    "Odds / EV": [
        "decimal_odds",
        "ev_current_adj",
        "ev_future_max",
        "ev_current_to_future_max_ratio",
    ],
    "Recent form (SG)": [
        "sg_total_L40",
        "sg_total_L24",
        "sg_total_L12",
    ],
    "Event history": [
        "starts_event",
        "made_cut_pct_event",
        "top25_event",
        "top10_event",
        "top5_event",
        "wins_event",
        "prev_finish_num_event",
    ],
    "YTD": [
        "ytd_starts",
        "ytd_made_cut_pct",
        "ytd_top25",
        "ytd_top10",
        "ytd_top5",
        "ytd_wins",
        "ytd_avg_sg_total",
    ],
}

def _format_for_ui(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # build a per-column format map
    fmt = {}

    # Currency-like EV columns
    for c in ["ev_current_adj", "ev_future_max", "ev_future_max_adj", "ev_current"]:
        if c in df.columns:
            fmt[c] = "{:,.0f}"

    # Ratios / probabilities
    for c in ["ev_current_to_future_max_ratio", "win_prob_est", "made_cut_pct_event", "ytd_made_cut_pct"]:
        if c in df.columns:
            fmt[c] = "{:.2f}"

    # Scores / SG / course-fit
    for c in df.columns:
        if c.startswith("sg_") or c.startswith("pct_") or c in ["oad_score", "decision_score", "final_rank_score", "course_fit_score"]:
            fmt[c] = "{:.2f}"

    # Odds
    for c in ["close_odds", "decimal_odds"]:
        if c in df.columns:
            fmt[c] = "{:.1f}"

    # apply formatting (only for columns present)
    return df.style.format(fmt, na_rep="")


# -----------------------------
# Internal helpers
# -----------------------------
def _coerce_numeric_only(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Coerce ONLY truly numeric columns. Do not touch context strings like
    decision_context / course_type.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            # already numeric: keep
            continue
        # try coercion, but only keep it if it produces at least one non-null
        coerced = pd.to_numeric(out[c], errors="coerce")
        if coerced.notna().any():
            out[c] = coerced
    return out


def _stable_test_label(dg_id: int, season: int) -> str:
    """
    Stable anonymized label per (season, dg_id). This allows consistent testing without names.
    """
    raw = f"{int(season)}:{int(dg_id)}".encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:6].upper()
    return f"P_{h}"


def add_display_player_col(
    df: pd.DataFrame,
    season: int,
    show_names: bool,
    name_col_candidates: tuple[str, ...] = ("player_name", "name", "player", "golfer", "full_name"),
) -> pd.DataFrame:
    """
    Adds `player_display` column.

    - show_names=True  => "Name (dg_id)" if a known name column exists
    - show_names=False => "P_ABC123" stable pseudonym per (season, dg_id)

    Safe even if no name column exists.
    """
    out = df.copy()

    if "dg_id" not in out.columns:
        raise ValueError("add_display_player_col expected df to contain 'dg_id' column.")

    # Choose the first existing name column if any
    name_col = next((c for c in name_col_candidates if c in out.columns), None)

    if show_names and name_col is not None:
        out["player_display"] = (
            out[name_col].astype(str).fillna("")
            + " ("
            + out["dg_id"].astype("Int64").astype(str)
            + ")"
        )
    else:
        out["player_display"] = out["dg_id"].apply(
            lambda x: _stable_test_label(int(x), season) if pd.notna(x) else "P_??????"
        )

    return out


def _infer_pct_scale(series: pd.Series) -> str:
    """
    Guess whether percent-like columns are 0-1 or 0-100.
    Returns: "0_1" or "0_100"
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "0_100"
    # If max <= 1.5, treat as 0-1
    return "0_1" if float(s.max()) <= 1.5 else "0_100"


def _style_group_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Apply numeric formatting + color scales.
    Designed to be readable (avoid over-coloring).
    """
    sty = df.style

    # Build per-column format map
    fmt: dict[str, str] = {}
    for c in df.columns:
        if c == "dg_id":
            continue

        if c.startswith("pct_"):
            # format percentiles as 0-100 integers if it looks like 0-100, else show 0-1 as %.
            scale = _infer_pct_scale(df[c])
            fmt[c] = "{:.0f}" if scale == "0_100" else "{:.0%}"
        elif "made_cut_pct" in c:
            # these are usually 0-1
            fmt[c] = "{:.0%}" if df[c].dropna().max() <= 1.5 else "{:.2f}"
        elif c in ("decimal_odds",):
            fmt[c] = "{:.1f}"
        elif c == "ev_current_to_future_max_ratio":
            fmt[c] = "{:.2f}"
        elif c.startswith("ev_"):
            fmt[c] = "{:,.0f}"
        elif c.startswith("ev_"):
            # EV should read like dollars (no decimals, commas)
            fmt[c] = "{:,.0f}"
        elif c in ("oad_score", "decision_score", "final_rank_score"):
            fmt[c] = "{:.2f}"
        elif c.startswith("sg_") or c.endswith("_avg_sg_total") or c.endswith("_sg_total"):
            fmt[c] = "{:.1f}"
        elif c in (
            "starts_event",
            "wins_event",
            "ytd_starts",
            "ytd_wins",
            "top25_event",
            "top10_event",
            "top5_event",
            "ytd_top25",
            "ytd_top10",
            "ytd_top5",
            "prev_finish_num_event",
        ):
            fmt[c] = "{:.0f}"

    sty = sty.format(fmt, na_rep="")

    # Helper: apply gradients
    def apply_gradient(cols: List[str], *, reverse: bool = False, cmap: str = "RdYlGn") -> None:
        nonlocal sty
        use = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not use:
            return
        _cmap = f"{cmap}_r" if reverse else cmap
        sty = sty.background_gradient(subset=use, cmap=_cmap)

    # Percentiles: higher is better
    apply_gradient([c for c in df.columns if c.startswith("pct_")], reverse=False)

    # EV + scores: higher is better
    apply_gradient(
        [c for c in ["ev_current_adj", "ev_future_max", "ev_current_to_future_max_ratio",
                     "oad_score", "decision_score", "final_rank_score"] if c in df.columns],
        reverse=False,
    )

    # Odds: lower is better
    apply_gradient(["decimal_odds"], reverse=True)

    # SG: higher better (simple gradient; we can switch to diverging later if you want)
    apply_gradient([c for c in df.columns if c.startswith("sg_") or c.endswith("_avg_sg_total")], reverse=False)

    # Made cut pct: higher better
    apply_gradient([c for c in df.columns if "made_cut_pct" in c], reverse=False)

    return sty


# -----------------------------
# Public API: render grouped top-N
# -----------------------------
def render_grouped_condensed_tables(
    summary: pd.DataFrame,
    season: int,
    groups: Dict[str, List[str]] = CONDENSED_GROUPS,
    *,
    sort_col: str = "oad_score",
    top_n: int = 20,
    show_names: bool = True,
    available_only: bool = True,
) -> pd.DataFrame:
    """
    Renders grouped condensed tables in Streamlit, and returns the top-N dataframe used.

    - summary: weekly["summary"] dataframe
    - season: used for stable test-mode pseudonyms
    - groups: column group definitions
    - sort_col: default 'oad_score'
    - top_n: default 20
    - show_names: True (live) / False (test)
    - available_only: if summary has 'is_available', filter to it; otherwise no filter is applied
    """
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("render_grouped_condensed_tables expected `summary` to be a pandas DataFrame.")

    s = summary.copy()

    if "dg_id" not in s.columns:
        raise ValueError("weekly summary is missing required column: 'dg_id'")

    # Add display label column
    s = add_display_player_col(s, season=season, show_names=show_names)

    # Filter to available if column exists
    if available_only and "is_available" in s.columns:
        s = s[s["is_available"] == True].copy()

    # Coerce numeric for all group columns (so style gradients work reliably)
    all_group_cols = sorted({c for cols in groups.values() for c in cols if c in s.columns})

    # only coerce numeric-like columns; preserve decision_context/course_type strings
    s = _coerce_numeric_only(s, [c for c in all_group_cols if c != "dg_id"])

    # Sort + top N
    if sort_col in s.columns:
        s = s.sort_values(sort_col, ascending=False, na_position="last")
    s_top = s.head(int(top_n)).copy()

    # Render each group
    for group_name, cols in groups.items():
        cols_present = [c for c in cols if c in s_top.columns]
        if not cols_present:
            continue

        # Ensure a usable identifier is first
        # If you want ONLY dg_id always first, change this to ["dg_id", ...]
        front = ["player_display", "dg_id"]
        ordered = [c for c in front if c in s_top.columns] + [c for c in cols_present if c not in front]

        view = s_top[ordered].copy()

        with st.expander(group_name, expanded=True):
            st.dataframe(_style_group_table(view), use_container_width=True, hide_index=True)

    return s_top
