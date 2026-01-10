# Scripts/app.py
from __future__ import annotations
from league import build_league_standings_through_prior, build_event_order_map

import sys
import os
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px

COMBINED_ROUNDS_PATH = Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/combined_rounds_all_2017_2025.csv")

# ============================================================
# PATH + IMPORT FIX (match notebook: from Scripts.weekly_view ...)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../OAD
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "Data"


# =========================
# CONFIG (YOUR PATHS)
# =========================
LEAGUE_CSV = {
    2024: Path("/Users/joshmacbook/python_projects/OAD/Data/Clean/Leagues/2024_small_normalized.csv"),
    2025: Path("/Users/joshmacbook/python_projects/OAD/Data/Clean/Leagues/2025_small_normalized.csv"),
}

SCHEDULE_XLSX = {
    2024: Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/OAD_2024.xlsx"),
    2025: Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/OAD_2025.xlsx"),
}

ODDS_AND_RESULTS_XLSX = Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/Odds_and_Results.xlsx")

PICKS_LOG_DIR = Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/Picks Log")
PICKS_LOG_DIR.mkdir(parents=True, exist_ok=True)

# NEW: per-event hide shortlist file
SHORTLIST_EXCLUSIONS_PATH = lambda season: (PICKS_LOG_DIR / f"shortlist_exclusions_{season}.csv")

YOU_USERNAME = "You"
TOP_N_DEFAULT = 30


# =========================
# OPTIONAL IMPORTS (LOCAL)
# =========================
def get_pre_event_cutoff_date(sched: pd.DataFrame, event_id: int) -> Optional[pd.Timestamp]:
    if sched is None or sched.empty:
        return None

    s = sched.copy()
    s["event_id"] = pd.to_numeric(s.get("event_id"), errors="coerce")
    row = s.loc[s["event_id"] == int(event_id)].head(1)
    if row.empty:
        return None

    # Use your schedule's actual date column name here
    for c in ["event_date", "start_date"]:
        if c in row.columns:
            dt = pd.to_datetime(row.iloc[0][c], errors="coerce")
            if pd.notna(dt):
                return dt - pd.Timedelta(days=1)

    # fallback: first *date-ish* column if needed
    date_cols = [c for c in row.columns if "date" in c.lower()]
    if date_cols:
        dt = pd.to_datetime(row.iloc[0][date_cols[0]], errors="coerce")
        if pd.notna(dt):
            return dt - pd.Timedelta(days=1)

    return None

def _safe_import_load_rounds():
    try:
        from Scripts.data_io import load_rounds  # type: ignore
        return load_rounds, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

LOAD_ROUNDS, LOAD_ROUNDS_ERR = _safe_import_load_rounds()


def _import_build_weekly_view():
    """
    IMPORTANT: Match notebook import.
    If this fails, you will see the error in the UI and odds-only fallback will be used.
    """
    from Scripts.weekly_view import build_weekly_view  # match notebook
    return build_weekly_view


def _safe_import_build_weekly_view():
    try:
        fn = _import_build_weekly_view()
        return fn, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _safe_import_data_io():
    # Keep if you later use it; unused currently.
    try:
        from Scripts import data_io  # type: ignore
        return data_io, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _safe_import_course_tables():
    try:
        from Scripts.data_io import load_course_fit, load_player_skills  # type: ignore
        return load_course_fit, load_player_skills, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


LOAD_COURSE_FIT, LOAD_PLAYER_SKILLS, COURSE_TABLES_ERR = _safe_import_course_tables()


@st.cache_data(show_spinner=False)
def load_course_fit_df(season: int) -> pd.DataFrame:
    if LOAD_COURSE_FIT is None:
        return pd.DataFrame()
    return LOAD_COURSE_FIT(int(season))

@st.cache_data(show_spinner=False)
def load_player_skills_df(season: int) -> pd.DataFrame:
    if LOAD_PLAYER_SKILLS is None:
        return pd.DataFrame()
    return LOAD_PLAYER_SKILLS(int(season))


BUILD_WEEKLY_VIEW, WEEKLY_VIEW_IMPORT_ERR = _safe_import_build_weekly_view()
DATA_IO, DATA_IO_IMPORT_ERR = _safe_import_data_io()

from Scripts.ui_condensed_tables import (
    CONDENSED_GROUPS,
    render_grouped_condensed_tables,
)


# =========================
# STREAMLIT SETTINGS
# =========================
st.set_page_config(
    page_title="One and Done",
    layout="wide",
)

st.title("One and Done")


# =========================
# UTIL: generic helpers
# =========================
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================
# UTIL: Loading
# =========================
@st.cache_data(show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    path = SCHEDULE_XLSX[season]
    df = pd.read_excel(path)

    for c in ["event_id", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        dcol = date_cols[0]
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.sort_values(dcol).reset_index(drop=True)

    df["event_order"] = np.arange(1, len(df) + 1)
    return df


@st.cache_data(show_spinner=False)
def load_league(season: int) -> pd.DataFrame:
    path = LEAGUE_CSV[int(season)]
    df = pd.read_csv(path)

    # normalize types a bit
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["raw_winnings"] = pd.to_numeric(df["raw_winnings"], errors="coerce")
    df["league_id"] = df["league_id"].astype(str)
    df["entry_id"] = df["entry_id"].astype(str)
    df["username"] = df["username"].astype(str)

    return df


@st.cache_data(show_spinner=False)
def load_odds(season: int) -> pd.DataFrame:
    df = pd.read_excel(ODDS_AND_RESULTS_XLSX)

    for c in ["year", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "win_prob_est" not in df.columns:
        df["win_prob_est"] = np.nan

    # CRITICAL: filter to the selected season/year
    if "year" in df.columns:
        df = df[df["year"] == int(season)].copy()

    return df

@st.cache_data(show_spinner=False)
def load_odds_results_full(season: int) -> pd.DataFrame:
    df = pd.read_excel(ODDS_AND_RESULTS_XLSX)
    for c in ["year", "event_id", "dg_id", "finish_num", "Winnings"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "year" in df.columns:
        df = df[df["year"] == int(season)].copy()
    return df

def log_path(season: int, mode: str) -> Path:
    mode_tag = "live" if mode.lower().startswith("live") else "test"
    return PICKS_LOG_DIR / f"picks_{mode_tag}_{season}.csv"


def load_picks_log(season: int, mode: str) -> pd.DataFrame:
    p = log_path(season, mode)
    if not p.exists():
        return pd.DataFrame(columns=[
            "season", "event_id", "event_name", "username",
            "dg_id", "player_name", "ts",
            "finish_num", "finish_text", "winnings",
        ])
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=[
            "season", "event_id", "event_name", "username",
            "dg_id", "player_name", "ts",
            "finish_num", "finish_text", "winnings",
        ])

    for c in ["season", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def deny_if_already_picked(picks_df: pd.DataFrame, season: int, event_id: int, username: str) -> bool:
    if picks_df.empty:
        return False
    sub = picks_df[
        (picks_df["season"] == season)
        & (picks_df["event_id"] == event_id)
        & (picks_df["username"] == username)
    ]
    return not sub.empty


def append_pick_log(
    season: int,
    mode: str,
    event_id: int,
    event_name: str,
    username: str,
    dg_id: int,
    player_name: str
) -> None:
    df = load_picks_log(season, mode)

    # --- lookup actual result (if available) ---
    odds_full = load_odds_results_full(season)
    mask = (
        (odds_full["year"] == int(season)) &
        (odds_full["event_id"] == int(event_id)) &
        (odds_full["dg_id"] == int(dg_id))
    )
    res = odds_full.loc[mask]

    if res.empty:
        finish_num = np.nan
        finish_text = None
        winnings = 0.0
    else:
        r0 = res.iloc[0]
        finish_num = r0.get("finish_num", np.nan)
        finish_text = r0.get("finish_text", None)  # ok if missing
        winnings = r0.get("Winnings", 0.0)

    row = {
        "season": int(season),
        "event_id": int(event_id),
        "event_name": str(event_name),
        "username": str(username),
        "dg_id": int(dg_id),
        "player_name": str(player_name),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "finish_num": finish_num,
        "finish_text": finish_text,
        "winnings": float(pd.to_numeric(winnings, errors="coerce") or 0.0),
    }

    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out = log_path(season, mode)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df2.to_csv(tmp, index=False)
    tmp.replace(out)


# =========================
# NEW: Per-event shortlist exclusions
# =========================
def _empty_exclusions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["season", "event_id", "dg_id", "reason", "ts"])


def load_exclusions(season: int) -> pd.DataFrame:
    p = SHORTLIST_EXCLUSIONS_PATH(season)
    if not p.exists():
        return _empty_exclusions_df()
    try:
        df = pd.read_csv(p)
    except Exception:
        return _empty_exclusions_df()

    for c in ["season", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_exclusions(season: int, df: pd.DataFrame) -> None:
    p = SHORTLIST_EXCLUSIONS_PATH(season)
    df.to_csv(p, index=False)


def excluded_ids_for_event(excl: pd.DataFrame, season: int, event_id: int) -> set[int]:
    if excl is None or excl.empty:
        return set()
    sub = excl[(excl["season"] == season) & (excl["event_id"] == event_id)]
    if sub.empty or "dg_id" not in sub.columns:
        return set()
    dg = pd.to_numeric(sub["dg_id"], errors="coerce").dropna().astype(int).tolist()
    return set(dg)

# =========================
# UTIL: Used-player filtering (global, based on picks log)
# =========================
def get_used_dg_ids_from_picks_log(season: int, mode: str, username: Optional[str] = None) -> set[int]:
    """
    Returns dg_ids already used, based on this app's picks log file for the given season/mode.
    If username is provided, only uses that user's picks (recommended).
    """
    picks = load_picks_log(season, mode)
    if picks is None or picks.empty:
        return set()

    # normalize
    picks["season"] = pd.to_numeric(picks.get("season"), errors="coerce")
    picks["dg_id"] = pd.to_numeric(picks.get("dg_id"), errors="coerce")
    picks["username"] = picks.get("username", "").astype(str)

    picks = picks[picks["season"] == int(season)].copy()
    if username is not None:
        picks = picks[picks["username"] == str(username)].copy()

    picks = picks.dropna(subset=["dg_id"]).copy()
    return set(picks["dg_id"].astype(int).tolist())


def apply_used_filter_to_summary_inplace(summary: pd.DataFrame, used: set[int]) -> pd.DataFrame:
    """Drop used dg_ids from the main candidate summary."""
    if summary is None or summary.empty or not used or "dg_id" not in summary.columns:
        return summary
    out = summary.copy()
    out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
    out = out.dropna(subset=["dg_id"]).copy()
    out["dg_id"] = out["dg_id"].astype(int)
    return out.loc[~out["dg_id"].isin(used)].copy()


def apply_used_filter_to_weekly_dict_inplace(weekly: dict, used: set[int]) -> dict:
    """
    If you use any weekly tables directly (table_performance etc),
    this mirrors your notebook's approach.
    """
    if not isinstance(weekly, dict):
        return weekly
    weekly["used_dg_ids"] = used
    if not used:
        return weekly

    keys = [
        "field",
        "summary",
        "table_performance",
        "table_performance_top50",
        "table_event_history",
        "table_event_history_top50",
        "table_ytd",
        "table_ytd_top50",
        "pattern_candidates",
    ]
    for k in keys:
        df = weekly.get(k)
        if isinstance(df, pd.DataFrame) and "dg_id" in df.columns:
            dg = pd.to_numeric(df["dg_id"], errors="coerce")
            mask = ~dg.fillna(-1).astype(int).isin(used)
            weekly[k] = df.loc[mask].copy()

    return weekly

# =========================
# UTIL: Styling
# =========================
def pick_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def style_table(df: pd.DataFrame, gradients=None, formats=None):
    """
    gradients: List[Tuple[col_name, cmap_name]]
    formats: Dict[col_name, format_str]
    """
    sty = df.style

    # 1) gradients first (needs numeric dtype)
    if gradients:
        for col, cmap in gradients:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                sty = sty.background_gradient(subset=[col], cmap=cmap)

    # 2) formatting LAST (so it doesn't get clobbered)
    if formats:
        use = {k: v for k, v in formats.items() if k in df.columns}
        if use:
            sty = sty.format(use, na_rep="")

    return sty

def coerce_numeric_for_formatting(df: pd.DataFrame, skip: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert all non-skip columns to numeric where possible.
    This stabilizes Streamlit Styler formatting + gradients.
    """
    out = df.copy()
    skip_set = set(skip or [])
    for c in out.columns:
        if c in skip_set:
            continue
        # try convert; if it's truly non-numeric it will become NaN
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def preround_panel(df: pd.DataFrame, round_map: Dict[str, int]) -> pd.DataFrame:
    """
    Actually round the underlying numeric values BEFORE styling.
    round_map = {"colA": 1, "colB": 0, ...}
    """
    out = df.copy()
    for c, d in round_map.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(int(d))
    return out


# =========================
# SCORING ENFORCEMENT
# =========================
def ensure_oad_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a numeric oad_score column (so UI logic is stable).
    Does NOT overwrite existing oad_score (only coerces it).
    """
    if df is None or df.empty:
        return df

    if "oad_score" in df.columns:
        df["oad_score"] = pd.to_numeric(df["oad_score"], errors="coerce")
        return df

    proxies = [
        "final_rank_score",
        "decision_score",
        "ev_current_adj",
        "ev_future_max",
        "win_prob_est",
    ]
    for c in proxies:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            df["oad_score"] = pd.to_numeric(df[c], errors="coerce")
            return df

    if "close_odds" in df.columns:
        odds = pd.to_numeric(df["close_odds"], errors="coerce")
        df["oad_score"] = np.where(odds > 0, 1.0 / odds, np.nan)
        return df

    df["oad_score"] = np.nan
    return df


# =========================
# UTIL: Weekly candidate table
# =========================
def get_weekly_summary(season: int, event_id: int) -> pd.DataFrame:
    if BUILD_WEEKLY_VIEW is not None:
        weekly = BUILD_WEEKLY_VIEW(season, int(event_id))

        if isinstance(weekly, dict):
            if "summary" in weekly and isinstance(weekly["summary"], pd.DataFrame):
                out = weekly["summary"].copy()
                return ensure_oad_score(out)

    # fallback to odds-only
    odds = load_odds(season)
    sub = odds[odds["event_id"] == int(event_id)].copy()
    if sub.empty:
        return pd.DataFrame()

    if "player_name" not in sub.columns:
        sub["player_name"] = sub.get("name", np.nan)

    if "oad_score" not in sub.columns:
        sub["oad_score"] = pd.to_numeric(sub.get("win_prob_est", np.nan), errors="coerce")

    return ensure_oad_score(sub)


def usable_sort_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            out.append(c)
    return out


def pick_best_rank_col(df: pd.DataFrame) -> str:
    candidates = ["oad_score", "final_rank_score", "decision_score", "ev_current_adj", "ev_future_max"]
    good = usable_sort_cols(df, candidates)
    return good[0] if good else "_rank"


# =========================
# VISUALS
# =========================
def plot_radar_course_vs_players(
    course_vec: Dict[str, float],
    player_vecs: Dict[str, Dict[str, float]],
    title: str,
):
    axes = ["DIST", "ACC", "APP", "ARG", "PUTT"]

    def close(vals):
        return vals + [vals[0]]

    theta = axes + [axes[0]]
    fig = go.Figure()

    # --- DataGolf-ish palette (explicitly requested) ---
    COURSE_LINE = "#59C98C"   # green
    COURSE_FILL = "rgba(89, 201, 140, 0.20)"

    # Higher-contrast player lines (kept readable on dark bg)
    PLAYER_ALPHA_FILL = 0.12

    # course
    cvals = [float(course_vec.get(a, np.nan)) for a in axes]
    fig.add_trace(go.Scatterpolar(
        r=close(cvals),
        theta=theta,
        mode="lines+markers",
        name="Course",
        line=dict(color=COURSE_LINE, width=3),
        marker=dict(size=6, color=COURSE_LINE),
        fill="toself",
        fillcolor=COURSE_FILL,
        opacity=1.0,
    ))

    # players (let Plotly pick contrasting colors; keep fill subtle)
    for label, vec in player_vecs.items():
        vals = [float(vec.get(a, np.nan)) for a in axes]
        fig.add_trace(go.Scatterpolar(
            r=close(vals),
            theta=theta,
            mode="lines+markers",
            name=label,
            line=dict(width=2),
            marker=dict(size=5),
            fill="toself",
            opacity=1.0,
            fillcolor=f"rgba(255,255,255,{PLAYER_ALPHA_FILL})",
        ))

    fig.update_traces(hovertemplate="%{theta}: %{r:.2f}<extra></extra>")

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".2f"),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(font=dict(size=12)),
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_movement_lines(standings: pd.DataFrame, title: str):
    if standings.empty:
        st.info("No standings data available for movement chart.")
        return

    fig = px.line(
        standings,
        x="event_order",
        y="cum_winnings",
        color="username",
        title=title,
    )
    fig.update_xaxes(
        dtick=1,
        tick0=1,
        tickformat="d",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _radar_course_vs_players_global(
    weekly: dict,
    course_fit_df: pd.DataFrame,
    player_skills_df: pd.DataFrame,
    dg_ids: List[int],
    label_map: Dict[int, str],
) -> tuple[Optional[Dict[str, float]], Dict[str, Dict[str, float]], Optional[str]]:
    """
    Build course + player radar vectors on a global 0-1 scale:
      - course vector: imp_* divided by GLOBAL max imp over all courses+attrs
      - player vectors: min-max per skill_* over full skills table
    Returns: (course_vec, player_vecs, error_message)
    """
    if not isinstance(weekly, dict) or "schedule_row" not in weekly:
        return None, {}, "weekly missing schedule_row"

    sr = weekly["schedule_row"]
    if isinstance(sr, pd.DataFrame) and len(sr) > 0:
        sr = sr.iloc[0]
    if not isinstance(sr, pd.Series):
        return None, {}, "schedule_row not a Series/row"

    course_num = pd.to_numeric(sr.get("course_num", np.nan), errors="coerce")
    if not np.isfinite(course_num):
        return None, {}, "schedule_row missing course_num"
    course_num = int(course_num)

    # ---- course vector ----
    if course_fit_df is None or course_fit_df.empty:
        return None, {}, "course_fit_df is empty"

    cf = course_fit_df.copy()
    if "course_num" not in cf.columns:
        return None, {}, "course_fit_df missing course_num"

    cf["course_num"] = pd.to_numeric(cf["course_num"], errors="coerce").astype("Int64")

    imp_cols = ["imp_dist", "imp_acc", "imp_app", "imp_arg", "imp_putt"]
    missing = [c for c in imp_cols if c not in cf.columns]
    if missing:
        return None, {}, f"course_fit_df missing columns: {missing}"

    row = cf.loc[cf["course_num"] == course_num]
    if row.empty:
        return None, {}, f"no course_fit row for course_num={course_num}"
    row = row.iloc[0]

    global_max_imp = float(np.nanmax(cf[imp_cols].to_numpy()))
    if not np.isfinite(global_max_imp) or global_max_imp <= 0:
        global_max_imp = 1.0

    course_vec = {
        "DIST": float(row.get("imp_dist", 0.0) or 0.0) / global_max_imp,
        "ACC":  float(row.get("imp_acc",  0.0) or 0.0) / global_max_imp,
        "APP":  float(row.get("imp_app",  0.0) or 0.0) / global_max_imp,
        "ARG":  float(row.get("imp_arg",  0.0) or 0.0) / global_max_imp,
        "PUTT": float(row.get("imp_putt", 0.0) or 0.0) / global_max_imp,
    }
    course_vec = {k: float(np.clip(v, 0.0, 1.0)) for k, v in course_vec.items()}

    # ---- player vectors ----
    if player_skills_df is None or player_skills_df.empty:
        return None, {}, "player_skills_df is empty"

    skills = player_skills_df.copy()
    if "dg_id" not in skills.columns:
        return None, {}, "player_skills_df missing dg_id"

    skills["dg_id"] = pd.to_numeric(skills["dg_id"], errors="coerce").astype("Int64")

    attr_map = {
        "DIST": "skill_dist",
        "ACC":  "skill_acc",
        "APP":  "skill_app",
        "ARG":  "skill_arg",
        "PUTT": "skill_putt",
    }
    missing = [c for c in attr_map.values() if c not in skills.columns]
    if missing:
        return None, {}, f"player_skills_df missing columns: {missing}"

    mins = {k: float(pd.to_numeric(skills[col], errors="coerce").min()) for k, col in attr_map.items()}
    maxs = {k: float(pd.to_numeric(skills[col], errors="coerce").max()) for k, col in attr_map.items()}

    player_vecs: Dict[str, Dict[str, float]] = {}
    for dg in dg_ids:
        r = skills.loc[skills["dg_id"] == int(dg)]
        if r.empty:
            continue
        r = r.iloc[0]

        vec = {}
        for k, col in attr_map.items():
            v = pd.to_numeric(r.get(col, np.nan), errors="coerce")
            lo, hi = mins[k], maxs[k]
            if np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                vec[k] = float((float(v) - lo) / (hi - lo))
            else:
                vec[k] = np.nan

        label = label_map.get(int(dg), f"dg_{int(dg)}")
        player_vecs[label] = vec

    if not player_vecs:
        return None, {}, "no matching dg_ids found in player_skills_df"

    return course_vec, player_vecs, None

def plot_skill_radar_h2h(
    player_skills_df: pd.DataFrame,
    dg_ids: List[int],
    labels: Dict[int, str],
    title: str = "Skills (global 0–1 scale)",
):
    if player_skills_df is None or player_skills_df.empty:
        st.warning("player_skills_df is empty; cannot plot radar.")
        return

    skills = player_skills_df.copy()
    if "dg_id" not in skills.columns:
        st.warning("player_skills_df missing dg_id; cannot plot radar.")
        return

    skills["dg_id"] = pd.to_numeric(skills["dg_id"], errors="coerce").astype("Int64")

    attr_map = {"DIST":"skill_dist","ACC":"skill_acc","APP":"skill_app","ARG":"skill_arg","PUTT":"skill_putt"}
    missing = [c for c in attr_map.values() if c not in skills.columns]
    if missing:
        st.warning(f"player_skills_df missing columns: {missing}")
        return

    # global min/max for normalization
    mins = {k: float(pd.to_numeric(skills[col], errors="coerce").min()) for k, col in attr_map.items()}
    maxs = {k: float(pd.to_numeric(skills[col], errors="coerce").max()) for k, col in attr_map.items()}

    axes = ["DIST","ACC","APP","ARG","PUTT"]
    theta = axes + [axes[0]]

    fig = go.Figure()

    for dg in dg_ids:
        row = skills.loc[skills["dg_id"] == int(dg)]
        if row.empty:
            continue
        row = row.iloc[0]

        vals = []
        for a in axes:
            col = attr_map[a]
            v = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            lo, hi = mins[a], maxs[a]
            if np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                vals.append(float((float(v) - lo) / (hi - lo)))
            else:
                vals.append(np.nan)

        r = vals + [vals[0]]

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            mode="lines+markers",
            name=labels.get(int(dg), f"dg_id {dg}"),
            line=dict(width=3),
            marker=dict(size=6),
            fill="toself",
            opacity=0.9,
        ))

    fig.update_traces(hovertemplate="%{theta}: %{r:.2f}<extra></extra>")
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".2f"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def build_yearly_sg(
    rounds_all: pd.DataFrame,
    dg_id: int,
    date_max: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    dfp = rounds_all.loc[rounds_all["dg_id"] == int(dg_id)].copy()
    if dfp.empty:
        return pd.DataFrame(columns=["year", "sg_total"])

    # tournament end dates (field-wide)
    ends = _build_event_end_table(rounds_all)

    t = (
        dfp.groupby(["year", "event_id"], as_index=False)
           .agg(sg_total=("sg_total", "sum"))
    )
    t = t.merge(ends, on=["year", "event_id"], how="left")
    t["event_end"] = pd.to_datetime(t["event_end"], errors="coerce")

    if date_max is not None:
        dm = pd.to_datetime(date_max, errors="coerce")
        if pd.notna(dm):
            t = t.loc[t["event_end"].notna() & (t["event_end"] <= dm)].copy()

    y = t.groupby("year", as_index=False)["sg_total"].mean()
    y["year"] = pd.to_numeric(y["year"], errors="coerce")
    y = y.dropna(subset=["year"]).copy()
    y["year"] = y["year"].astype(int)
    return y.sort_values("year")

# =========================
# SIDEBAR: Inputs + DEBUG
# =========================
with st.sidebar:
    st.header("Inputs")

    season = st.selectbox("Season", [2024, 2025], index=1, key="sb_season")

    mode = st.radio("Mode", ["Live", "Test"], horizontal=True, key="sb_mode")
    test_mode = (mode == "Test")

    if test_mode:
        hide_names = st.checkbox("Hide player names (test mode)", value=True, key="sb_hide_names_test")
        show_names = not hide_names
    else:
        hide_names = False
        show_names = True
        st.caption("Names are shown in Live mode.")

    top_n = st.slider("Show top N candidates", 10, 60, TOP_N_DEFAULT, key="sb_top_n")

    st.divider()
    st.caption("Filters")
    hide_shortlist = st.checkbox("Hide shortlist players (this event)", value=False, key="sb_hide_shortlist")

    # Event selection
    sched = load_schedule(season).copy()
    if "event_id" not in sched.columns:
        raise ValueError("Schedule is missing required column: event_id")

    event_name_col = "event_name" if "event_name" in sched.columns else None

    sched["event_id"] = pd.to_numeric(sched["event_id"], errors="coerce")
    if "event_order" in sched.columns:
        sched["event_order"] = pd.to_numeric(sched["event_order"], errors="coerce")

    sched = sched.dropna(subset=["event_id"]).copy()
    sched["event_id"] = sched["event_id"].astype(int)

    if event_name_col:
        sched[event_name_col] = sched[event_name_col].astype(str)
        if "event_order" in sched.columns and sched["event_order"].notna().any():
            label_map_events = {
                int(r["event_id"]): f"{int(r['event_order']):02d} — {r[event_name_col]}"
                for _, r in sched.iterrows()
            }
        else:
            label_map_events = {
                int(r["event_id"]): f"{int(r['event_id'])} — {r[event_name_col]}"
                for _, r in sched.iterrows()
            }

        event_id = st.selectbox(
            "Event",
            options=list(label_map_events.keys()),
            format_func=lambda x: label_map_events.get(int(x), str(x)),
            key="sb_event_id",
        )
        event_name = label_map_events.get(int(event_id), str(event_id))
    else:
        event_ids = sorted(sched["event_id"].unique().tolist())
        event_id = st.selectbox("Event (event_id)", options=event_ids, key="sb_event_id_fallback")
        event_name = str(event_id)

    username = st.text_input("Username", value=YOU_USERNAME, key="sb_username")

    st.divider()
    st.caption("Pick logging")

    # single source of truth for pick-logging toggle
    do_log = st.checkbox("Enable pick logging", value=True, key="sb_enable_pick_logging")

    # show active log file (critical for sanity)
    active_log = log_path(season, mode)
    st.caption(f"Active log file:\n{active_log}")

    # manual backup (COPY, not move)
    if st.button("Backup picks log", key="sb_backup_picks_log"):
        if not active_log.exists():
            st.warning("No picks log exists yet for this season/mode.")
        else:
            import shutil
            backup_path = active_log.with_name(
                active_log.stem + f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            shutil.copy2(active_log, backup_path)
            st.success(f"Backed up to:\n{backup_path}")

    st.divider()
    debug = st.checkbox("Debug", value=False, key="sb_debug")
    st.session_state["debug"] = debug
    if debug:
        st.caption("Import status")
        st.write("BUILD_WEEKLY_VIEW loaded:", BUILD_WEEKLY_VIEW is not None)
        if BUILD_WEEKLY_VIEW is not None:
            try:
                st.write("build_weekly_view file:", inspect.getfile(BUILD_WEEKLY_VIEW))
            except Exception:
                st.write("build_weekly_view file: (could not inspect)")
        if WEEKLY_VIEW_IMPORT_ERR:
            st.error("weekly_view import error")
            st.code(WEEKLY_VIEW_IMPORT_ERR)

# =========================
# LOAD WEEKLY + PREP (ONCE)
# =========================
summary = get_weekly_summary(season, int(event_id))

weekly = None
if BUILD_WEEKLY_VIEW is not None:
    weekly = BUILD_WEEKLY_VIEW(season, int(event_id))

# If weekly_view didn't populate event-level fields into summary, copy them down from schedule_row
if isinstance(weekly, dict):
    sr = weekly.get("schedule_row", None)
    if isinstance(sr, pd.DataFrame) and len(sr) > 0:
        sr = sr.iloc[0]
    if isinstance(sr, pd.Series):
        for k in ["decision_context", "course_type", "course_num", "course_name", "event_name"]:
            if k in sr.index:
                v = sr.get(k)
                if (k not in summary.columns) or summary[k].isna().all():
                    summary[k] = v

if summary.empty:
    st.error("No data returned for this week/event. Check odds + schedule + build_weekly_view.")
    st.stop()

if "dg_id" not in summary.columns:
    st.error("Weekly summary missing dg_id; cannot continue.")
    st.stop()

summary["dg_id"] = pd.to_numeric(summary["dg_id"], errors="coerce")
summary = summary.dropna(subset=["dg_id"]).copy()
summary["dg_id"] = summary["dg_id"].astype(int)

# Ensure player_name exists
if "player_name" not in summary.columns:
    summary["player_name"] = summary["dg_id"].apply(lambda x: f"dg_{int(x)}")

# Label used in your older panels/picker
summary["player_label"] = summary["player_name"].astype(str) if show_names else summary["dg_id"].apply(lambda x: f"dg_{int(x)}")

# Make sure oad_score exists for downstream (even if odds-only)
summary = ensure_oad_score(summary)

# Fill decision_context/course_type nulls with any non-null value (prevents the grouped tables showing None)
for k in ["decision_context", "course_type"]:
    if k in summary.columns and summary[k].notna().any():
        fill_val = summary.loc[summary[k].notna(), k].iloc[0]
        summary[k] = summary[k].fillna(fill_val)

# -------------------------
# NEW: Apply per-event exclusions
# -------------------------
excl_df = load_exclusions(season)
excluded_ids = excluded_ids_for_event(excl_df, season, int(event_id))
if excluded_ids:
    summary = summary.loc[~summary["dg_id"].isin(excluded_ids)].copy()

# -------------------------
# NEW: Remove already-used players (from picks log)
# -------------------------
used_ids = get_used_dg_ids_from_picks_log(season=season, mode=mode, username=username)

# filter the UI driver table
summary = apply_used_filter_to_summary_inplace(summary, used_ids)

# also filter weekly dict if present (defensive; helps if you use weekly tables anywhere)
if isinstance(weekly, dict):
    weekly = apply_used_filter_to_weekly_dict_inplace(weekly, used_ids)

if debug:
    st.write("Used dg_ids (from picks log):", len(used_ids))


# Optional filter: hide shortlist players
if "is_shortlist" in summary.columns:
    is_sl = summary["is_shortlist"].fillna(False)
    if is_sl.dtype != bool:
        is_sl = is_sl.astype(bool)
    if hide_shortlist:
        summary = summary.loc[~is_sl].copy()

used_ids = get_used_dg_ids_from_picks_log(season=season, mode=mode, username=username)
if used_ids:
    summary = summary.loc[~summary["dg_id"].isin(used_ids)].copy()

if debug:
    st.caption(f"Used filtered (from {log_path(season, mode)}): {len(used_ids)}")

rank_col = pick_best_rank_col(summary)
if rank_col not in summary.columns:
    summary["_rank"] = np.arange(len(summary), 0, -1)
    rank_col = "_rank"

summary[rank_col] = pd.to_numeric(summary[rank_col], errors="coerce")
summary = summary.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)

summary_top = summary.head(int(top_n)).copy()

if debug:
    with st.expander("Debug: summary schema", expanded=False):
        st.write("Rows:", len(summary))
        st.write("Rank col:", rank_col)
        st.write("Has oad_score:", "oad_score" in summary.columns)
        st.write("Excluded IDs (this event):", sorted(list(excluded_ids))[:30])
        st.write(sorted(summary.columns.tolist()))


# Tabs
tab1, tab2, tab3 = st.tabs([
    "Pick",
    "Player Profile",
    "League",
    # "Head to Head",
])


# =========================
# TAB 1: Pick
# =========================
with tab1:
    st.header("Weekly decision view")

    sort_candidates = ["oad_score", "final_rank_score", "decision_score", "ev_current_adj", "ev_future_max"]
    sort_options = usable_sort_cols(summary, sort_candidates)
    if not sort_options:
        sort_options = [rank_col]

    sort_col = st.selectbox("Sort by", options=sort_options, index=0)

    _ = render_grouped_condensed_tables(
        summary=summary,
        season=season,
        groups=CONDENSED_GROUPS,
        sort_col=sort_col,
        top_n=20,
        show_names=show_names,
        available_only=True,
    )

    st.divider()

    # =========================
    # Panel A
    # =========================
    st.subheader("A) Shortlist / Final Decision")

    cols_A = pick_cols(summary_top, [
        "player_label", "dg_id",
        "oad_score", "decision_score", "decision_context", "final_rank_score",
        "close_odds", "win_prob_est",
        "course_fit_score",
        "pick_total_aggr", "new_cum_total_aggr",
    ])

    panelA = summary_top[cols_A].copy()
    if "player_label" in panelA.columns:
        front = ["player_label"]
        rest = [c for c in panelA.columns if c not in front]
        panelA = panelA[front + rest]

    grad_A = []
    for c in ["final_rank_score", "decision_score", "oad_score", "course_fit_score", "win_prob_est"]:
        if c in panelA.columns:
            grad_A.append((c, "Greens"))
    if "close_odds" in panelA.columns:
        grad_A.append(("close_odds", "Reds"))

    fmt_A = {}
    for c in ["oad_score", "decision_score", "final_rank_score", "course_fit_score", "win_prob_est"]:
        if c in panelA.columns:
            fmt_A[c] = "{:.1f}"
    for c in ["pick_total_aggr", "new_cum_total_aggr"]:
        if c in panelA.columns:
            fmt_A[c] = "{:,.0f}"
    if "close_odds" in panelA.columns:
        fmt_A["close_odds"] = "{:.1f}"

    st.dataframe(style_table(panelA, gradients=grad_A, formats=fmt_A), use_container_width=True, height=420)

    st.divider()

    # =========================
    # NEW: Per-event shortlist exclusion toggles
    # =========================
    st.subheader("Filter shortlist for this event")

    pick_choices_all = summary_top[["dg_id", "player_label"]].dropna().copy()
    pick_choices_all["dg_id"] = pick_choices_all["dg_id"].astype(int)
    label_map_players = dict(zip(pick_choices_all["dg_id"], pick_choices_all["player_label"]))

    # show only current shortlist players if available, else top-N
    candidate_ids = pick_choices_all["dg_id"].tolist()
    if "is_shortlist" in summary_top.columns:
        tmp = summary_top.copy()
        is_sl = tmp["is_shortlist"].fillna(False).astype(bool)
        shortlist_ids = tmp.loc[is_sl, "dg_id"].astype(int).tolist()
        if shortlist_ids:
            candidate_ids = shortlist_ids

    default_hidden = sorted(list(set(candidate_ids).intersection(excluded_ids)))
    hidden_selected = st.multiselect(
        "Hide these players (this event only)",
        options=candidate_ids,
        default=default_hidden,
        format_func=lambda dg: f"{label_map_players.get(int(dg), f'dg_id {dg}')} ({int(dg)})",
    )

    col_save, col_hint = st.columns([1, 3])
    with col_save:
        if st.button("Save event filters"):
            keep = set(int(x) for x in hidden_selected)
            before = excl_df.copy()

            # drop existing rows for this season/event then re-add
            excl_df2 = excl_df.copy()
            if not excl_df2.empty:
                excl_df2 = excl_df2[~((excl_df2["season"] == season) & (excl_df2["event_id"] == int(event_id)))].copy()

            new_rows = []
            for dg in sorted(list(keep)):
                new_rows.append({
                    "season": int(season),
                    "event_id": int(event_id),
                    "dg_id": int(dg),
                    "reason": "manual_hide_shortlist",
                    "ts": datetime.now().isoformat(timespec="seconds"),
                })

            excl_df2 = pd.concat([excl_df2, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else excl_df2
            save_exclusions(season, excl_df2)
            st.success(f"Saved. {len(keep)} hidden for this event. File: {SHORTLIST_EXCLUSIONS_PATH(season)}")
            st.rerun()

    with col_hint:
        st.caption("This is separate from the global shortlist. It only hides players for the currently selected event.")

    st.divider()

    # =========================
    # Drill-down above visuals
    # =========================
    st.subheader("Drill-down (select 1–3 players)")

    pick_map = dict(zip(pick_choices_all["player_label"], pick_choices_all["dg_id"]))

    default_labels = pick_choices_all["player_label"].head(3).tolist()
    selected_labels = st.multiselect(
        "Players",
        options=list(pick_map.keys()),
        default=default_labels,
        max_selections=3
    )
    selected_ids = [int(pick_map[l]) for l in selected_labels]

    st.divider()

    # =========================
    # Panels B/C/D/E
    # =========================
    st.subheader("B) Form & Skill Right Now")

    cols_B = pick_cols(summary_top, [
        "player_label", "dg_id",
        "sg_total_L12", "sg_total_L24", "sg_total_L40",
        "sg_ott_L24", "sg_app_L24", "sg_arg_L24", "sg_putt_L24",
    ])
    panelB = summary_top[cols_B].copy()
    panelB = coerce_numeric_for_formatting(panelB, skip=["player_label"])
    grad_B = []
    for c in ["sg_total_L12", "sg_total_L24", "sg_total_L40", "sg_app_L24", "sg_putt_L24", "sg_ott_L24"]:
        if c in panelB.columns:
            grad_B.append((c, "RdYlGn"))
    fmt_B = {c: "{:.1f}" for c in panelB.columns if c.startswith("sg_") or c.startswith("sg_total")}
    st.dataframe(style_table(panelB, gradients=grad_B, formats=fmt_B), use_container_width=True, height=360)

    st.subheader("C) Season Reliability (YTD)")

    cols_C = pick_cols(summary_top, [
        "player_label", "dg_id",
        "ytd_starts", "ytd_made_cut_pct",
        "ytd_avg_sg_total",
        "ytd_top25", "ytd_top10", "ytd_top5", "ytd_wins",
    ])
    panelC = summary_top[cols_C].copy()
    panelC = coerce_numeric_for_formatting(panelC, skip=["player_label"])
    grad_C = []
    for c in ["ytd_made_cut_pct", "ytd_avg_sg_total"]:
        if c in panelC.columns:
            grad_C.append((c, "Greens"))
    for c in ["ytd_starts", "ytd_top25", "ytd_top10", "ytd_top5", "ytd_wins"]:
        if c in panelC.columns:
            grad_C.append((c, "Blues"))
    fmt_C = {}
    for c in ["ytd_starts", "ytd_top25", "ytd_top10", "ytd_top5", "ytd_wins"]:
        if c in panelC.columns:
            fmt_C[c] = "{:.0f}"

    # pct + SG
    if "ytd_made_cut_pct" in panelC.columns:
        fmt_C["ytd_made_cut_pct"] = "{:.1f}"  # or "{:.0%}" if you want 100% style
    if "ytd_avg_sg_total" in panelC.columns:
        fmt_C["ytd_avg_sg_total"] = "{:.1f}"

    st.dataframe(style_table(panelC, gradients=grad_C, formats=fmt_C), use_container_width=True, height=320)

    st.subheader("D) Event & Course History")

    cols_D = pick_cols(summary_top, [
        "player_label", "dg_id",
        "starts_event", "made_cut_pct_event", "top25_event", "top10_event", "top5_event", "wins_event",
        "prev_finish_num_event",
        "course_num", "course_name",
        "course_fit_score",
    ])
    panelD = summary_top[cols_D].copy()
    panelD = coerce_numeric_for_formatting(
        panelD,
        skip=["player_label", "course_name"]
    )
    grad_D = []
    for c in ["course_fit_score", "made_cut_pct_event"]:
        if c in panelD.columns:
            grad_D.append((c, "Greens"))
    if "prev_finish_num_event" in panelD.columns:
        grad_D.append(("prev_finish_num_event", "Reds"))
    fmt_D = {}
    # counts should be integers
    for c in ["starts_event", "top25_event", "top10_event", "top5_event", "wins_event"]:
        if c in panelD.columns:
            fmt_D[c] = "{:.0f}"

    # pct + course fit + finish num
    if "made_cut_pct_event" in panelD.columns:
        fmt_D["made_cut_pct_event"] = "{:.1f}"  # or "{:.0%}" if you want percent display
    if "course_fit_score" in panelD.columns:
        fmt_D["course_fit_score"] = "{:.1f}"
    if "prev_finish_num_event" in panelD.columns:
        fmt_D["prev_finish_num_event"] = "{:.0f}"

    st.dataframe(style_table(panelD, gradients=grad_D, formats=fmt_D), use_container_width=True, height=320)

    st.subheader("E) One-and-Done Economics")

    cols_E = pick_cols(summary_top, [
        "player_label", "dg_id",
        "ev_current_adj", "ev_future_max_adj", "ev_current_to_future_max_ratio",
        "is_shortlist",
    ])
    panelE = summary_top[cols_E].copy()
    panelE = coerce_numeric_for_formatting(panelE, skip=["player_label"])
    grad_E = []
    for c in ["ev_current_adj", "ev_future_max_adj", "ev_current_to_future_max_ratio"]:
        if c in panelE.columns:
            grad_E.append((c, "Greens"))
    fmt_E = {}
    for c in ["ev_current_adj", "ev_future_max_adj"]:
        if c in panelE.columns:
            fmt_E[c] = "{:,.0f}"
    if "ev_current_to_future_max_ratio" in panelE.columns:
        fmt_E["ev_current_to_future_max_ratio"] = "{:.2f}"
    st.dataframe(style_table(panelE, gradients=grad_E, formats=fmt_E), use_container_width=True, height=280)

    st.divider()

    # =========================
    # VISUALS
    # =========================
    st.subheader("Selected-player visuals")

    sel_df = summary_top[summary_top["dg_id"].isin(selected_ids)].copy()
    if sel_df.empty or not selected_ids:
        st.info("Select 1–3 players above to see visuals.")
    else:
        # ------------------------------------------------------------
        # Recent form: per-round SG line
        # ------------------------------------------------------------
        st.markdown("#### Recent form: strokes gained per round")

        if LOAD_ROUNDS is None:
            st.info("Rounds loader not available (Scripts.data_io.load_rounds import failed).")
            if debug and LOAD_ROUNDS_ERR:
                st.code(LOAD_ROUNDS_ERR)
        else:
            rounds = LOAD_ROUNDS()
            if not isinstance(rounds, pd.DataFrame) or rounds.empty:
                st.info("Rounds table is empty/unavailable.")
            else:
                for c in ["dg_id", "year", "event_id", "round_num"]:
                    if c in rounds.columns:
                        rounds[c] = pd.to_numeric(rounds[c], errors="coerce")

                sg_col = pick_first_existing(
                    rounds,
                    candidates=["sg_total", "sg_total_round", "sg_total_field", "sg_total_raw"],
                )

                if sg_col is None:
                    st.info("Could not find an SG-per-round column in rounds table.")
                else:
                    rsub = rounds.loc[rounds["dg_id"].isin(selected_ids)].copy()
                    rsub = rsub.dropna(subset=["dg_id", sg_col]).copy()

                    sort_cols = [c for c in ["year", "event_id", "round_num"] if c in rsub.columns]
                    if sort_cols:
                        rsub = rsub.sort_values(sort_cols)
                    else:
                        rsub = rsub.sort_values(["dg_id"])

                    N_ROUNDS = 24
                    rsub = rsub.groupby("dg_id", as_index=False).tail(N_ROUNDS)
                    rsub["round_index"] = rsub.groupby("dg_id").cumcount() + 1

                    id_to_label = dict(zip(sel_df["dg_id"], sel_df["player_label"]))
                    rsub["player_label"] = rsub["dg_id"].map(id_to_label).fillna(rsub["dg_id"].astype(str))

                    fig = px.line(
                        rsub,
                        x="round_index",
                        y=sg_col,
                        color="player_label",
                        markers=True,
                        title="Recent form: SG per round (last ~30 rounds per player)",
                    )
                    fig.update_yaxes(
                        tickmode="linear",
                        dtick=1,  # use 0.5 if you want tighter grid
                        zeroline=True
                    )

                    fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
                    fig.update_traces(hovertemplate="Round %{x}<br>SG %{y:.1f}<extra></extra>")
                    st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------
        # SG components focused on what matters this week (using course_fit importance if possible)
        # ------------------------------------------------------------
        st.markdown("#### SG components (focused on this event)")

        comp_candidates = [
            ("OTT", "sg_ott_L24"),
            ("APP", "sg_app_L24"),
            ("ARG", "sg_arg_L24"),
            ("PUTT", "sg_putt_L24"),
        ]
        present = [(lab, col) for lab, col in comp_candidates if col in sel_df.columns]

        if not present:
            st.info("SG component columns (L24) not present for selected players.")
        else:
            # try to pick top 3 components by course importance using course_fit_df + course_num
            chosen = present
            course_fit_df = load_course_fit_df(season)
            if isinstance(weekly, dict) and not course_fit_df.empty:
                sr = weekly.get("schedule_row", None)
                if isinstance(sr, pd.DataFrame) and len(sr) > 0:
                    sr = sr.iloc[0]
                if isinstance(sr, pd.Series) and "course_num" in sr.index:
                    course_num = pd.to_numeric(sr.get("course_num"), errors="coerce")
                    if np.isfinite(course_num):
                        course_num = int(course_num)
                        cf = course_fit_df.copy()
                        cf["course_num"] = pd.to_numeric(cf.get("course_num", np.nan), errors="coerce").astype("Int64")
                        row = cf[cf["course_num"] == course_num]
                        if not row.empty:
                            row = row.iloc[0]
                            imp_map = {"OTT": "imp_ott", "APP": "imp_app", "ARG": "imp_arg", "PUTT": "imp_putt"}
                            if all(v in cf.columns for v in imp_map.values()):
                                weights = []
                                for lab, _ in present:
                                    w = pd.to_numeric(row.get(imp_map[lab], np.nan), errors="coerce")
                                    weights.append((lab, float(w) if np.isfinite(w) else -np.inf))
                                weights = sorted(weights, key=lambda x: x[1], reverse=True)
                                K = min(3, len(weights))
                                keep = set([lab for lab, _ in weights[:K]])
                                chosen = [(lab, col) for lab, col in present if lab in keep]
                                st.caption(f"Showing top {K} components by course importance: {', '.join([lab for lab, _ in weights[:K]])}")

            comp = sel_df[["player_label"] + [col for _, col in chosen]].copy()
            rename = {col: lab for lab, col in chosen}
            comp = comp.rename(columns=rename)

            comp_m = comp.melt(id_vars="player_label", var_name="component", value_name="sg")
            fig = px.bar(
                comp_m,
                x="component",
                y="sg",
                color="player_label",
                barmode="group",
                title="SG component profile (L24) — focused on what matters this week",
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
            fig.update_traces(hovertemplate="%{x}<br>SG %{y:.1f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------
        # Radar (wired using course_fit + player_skills, global scaling)
        # ------------------------------------------------------------
        st.markdown("#### Course-fit radar (overlay)")

        # 1) pull the season-scoped tables
        course_fit_df = load_course_fit_df(season)
        player_skills_df = load_player_skills_df(season)

        # 2) label map for selected players
        label_map = {int(r["dg_id"]): str(r["player_label"]) for _, r in sel_df.iterrows()}

        # 3) build vectors (global 0–1 scale like your visuals.py)
        course_vec, player_vecs, err = _radar_course_vs_players_global(
            weekly=weekly,
            course_fit_df=course_fit_df,
            player_skills_df=player_skills_df,
            dg_ids=selected_ids,
            label_map=label_map,
        )

        if err:
            st.warning(f"Radar unavailable: {err}")
        else:
            plot_radar_course_vs_players(
                course_vec=course_vec,
                player_vecs=player_vecs,
                title="Course vs Player skill (global 0–1 scale)",
            )

    st.divider()

    # =========================
    # Make a pick (bottom)
    # =========================
    st.markdown("### Make a pick")

    pick_label = st.selectbox("Select player", options=list(pick_map.keys()))
    pick_dg_id = int(pick_map[pick_label])

    real_name = summary.loc[summary["dg_id"] == pick_dg_id, "player_name"]
    real_name = str(real_name.iloc[0]) if not real_name.empty else pick_label

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        confirm = st.button("Log pick", type="primary", disabled=not do_log)
    with col2:
        st.write(f"DG ID: **{pick_dg_id}**")
    with col3:
        st.write(f"Player: **{real_name if show_names else pick_label}**")

    if confirm and do_log:
        existing = load_picks_log(season, mode)
        if deny_if_already_picked(existing, season, int(event_id), username):
            st.error("Denied: you already logged a pick for this event. Delete the row manually in the picks log file to change it.")
        else:
            append_pick_log(season, mode, int(event_id), event_name, username, pick_dg_id, real_name)
            st.success(f"Logged. Saved to: {log_path(season, mode)}")


# =========================
# TAB 2: PLAYER PROFILE  (clean rewrite)
# =========================
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

COMBINED_ROUNDS_PATH = Path("/Users/joshmacbook/python_projects/OAD/Data/in Use/combined_rounds_all_2017_2025.csv")


# -------------------------
# Data load
# -------------------------
@st.cache_data(show_spinner=False)
def load_combined_rounds() -> pd.DataFrame:
    if not COMBINED_ROUNDS_PATH.exists():
        raise FileNotFoundError(f"Missing combined rounds file: {COMBINED_ROUNDS_PATH}")
    df = pd.read_csv(COMBINED_ROUNDS_PATH)

    # types
    for c in ["dg_id", "event_id", "year", "round_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    # numeric cols of interest
    num_cols = [
        "sg_putt","sg_arg","sg_app","sg_ott","sg_t2g","sg_total",
        "driving_dist","driving_acc","gir","scrambling","prox_rgh","prox_fw",
        "great_shots","poor_shots",
        "eagles_or_better","birdies","pars","bogies","doubles_or_worse",
        "round_score",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -------------------------
# Helpers
# -------------------------
def _get_event_date_from_schedule(sched: pd.DataFrame, event_id: int) -> Optional[pd.Timestamp]:
    """Return event date for the selected event_id from schedule (first column containing 'date')."""
    if sched is None or sched.empty or "event_id" not in sched.columns:
        return None

    s = sched.copy()
    s["event_id"] = pd.to_numeric(s["event_id"], errors="coerce")
    s = s.dropna(subset=["event_id"]).copy()
    s["event_id"] = s["event_id"].astype(int)

    row = s.loc[s["event_id"] == int(event_id)]
    if row.empty:
        return None

    date_cols = [c for c in s.columns if "date" in c.lower()]
    for c in date_cols:
        dt = pd.to_datetime(row.iloc[0].get(c), errors="coerce")
        if pd.notna(dt):
            return dt
    return None


def _global_tournament_end_dates(rounds_all: pd.DataFrame) -> pd.DataFrame:
    """Field-wide tournament end date = max round_date across ALL players (stable)."""
    if "round_date" in rounds_all.columns and rounds_all["round_date"].notna().any():
        base = rounds_all.dropna(subset=["round_date"]).copy()
        base["round_date"] = pd.to_datetime(base["round_date"], errors="coerce")
        ends = (
            base.groupby(["year", "event_id", "event_name"], as_index=False)["round_date"]
                .max()
                .rename(columns={"round_date": "event_end"})
        )
        ends["event_end"] = pd.to_datetime(ends["event_end"], errors="coerce")
        return ends

    ends = rounds_all[["year", "event_id", "event_name"]].drop_duplicates().copy()
    ends["event_end"] = pd.to_datetime(ends["year"].astype(str) + "-12-31", errors="coerce")
    return ends

def _build_event_end_table(rounds_all: pd.DataFrame) -> pd.DataFrame:
    """
    Field-wide tournament end date table:
    returns columns: year, event_id, event_end
    event_end = max(round_date) across ALL players in the event.
    Fallback: year-12-31 if round_date missing.
    """
    if rounds_all is None or rounds_all.empty:
        return pd.DataFrame(columns=["year", "event_id", "event_end"])

    ra = rounds_all.copy()

    # normalize types
    ra["year"] = pd.to_numeric(ra.get("year"), errors="coerce")
    ra["event_id"] = pd.to_numeric(ra.get("event_id"), errors="coerce")

    if "round_date" in ra.columns and ra["round_date"].notna().any():
        ra = ra.dropna(subset=["round_date", "year", "event_id"]).copy()
        ra["round_date"] = pd.to_datetime(ra["round_date"], errors="coerce")
        ra = ra.dropna(subset=["round_date"]).copy()

        ends = (
            ra.groupby(["year", "event_id"], as_index=False)["round_date"]
              .max()
              .rename(columns={"round_date": "event_end"})
        )
    else:
        ends = ra[["year", "event_id"]].drop_duplicates().copy()
        ends["event_end"] = pd.to_datetime(ends["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")

    # final cleanup
    ends["year"] = pd.to_numeric(ends["year"], errors="coerce").astype("Int64")
    ends["event_id"] = pd.to_numeric(ends["event_id"], errors="coerce").astype("Int64")
    ends = ends.dropna(subset=["year", "event_id"]).copy()

    ends["year"] = ends["year"].astype(int)
    ends["event_id"] = ends["event_id"].astype(int)
    ends["event_end"] = pd.to_datetime(ends["event_end"], errors="coerce")

    return ends[["year", "event_id", "event_end"]]

def build_last_n_events_table(
    rounds_all: pd.DataFrame,
    dg_id: int,
    n: int = 25,
    date_max: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Tournament-level: Event, Finish, SG Total, Year. Ordered by field-wide event_end desc."""
    dfp = rounds_all.loc[rounds_all["dg_id"] == int(dg_id)].copy()
    if dfp.empty:
        return pd.DataFrame(columns=["Event", "Finish", "SG Total", "Year", "event_id"])

    ends_all = _global_tournament_end_dates(rounds_all)

    if date_max is not None:
        dm = pd.to_datetime(date_max, errors="coerce")
        if pd.notna(dm):
            ends_all = ends_all.loc[ends_all["event_end"].notna() & (ends_all["event_end"] <= dm)].copy()

    def _first_non_null_str(s: pd.Series) -> str:
        s2 = s.dropna().astype(str)
        return s2.iloc[0] if len(s2) else ""

    fin = (
        dfp.groupby(["year", "event_id", "event_name"], as_index=False)
           .agg(
               Finish=("fin_text", _first_non_null_str),
               SG_Total=("sg_total", "sum"),
           )
    )

    out = fin.merge(ends_all, on=["year", "event_id", "event_name"], how="inner")
    out = out.sort_values(["event_end", "year", "event_id"], ascending=[False, False, False]).head(int(n)).copy()

    out["SG Total"] = out["SG_Total"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    out = out.rename(columns={"event_name": "Event", "year": "Year"})
    out = out[["Event", "Finish", "SG Total", "Year", "event_id"]]
    return out.reset_index(drop=True)

def build_course_history_table(
    rounds_all: pd.DataFrame,
    dg_id: int,
    course_num: int,
    date_max: Optional[pd.Timestamp] = None,
    n: int = 25,
    sched_all: Optional[pd.DataFrame] = None,  # only needed if rounds_all lacks course_num
) -> pd.DataFrame:
    if rounds_all is None or rounds_all.empty:
        return pd.DataFrame()

    ra = rounds_all.copy()
    ra["dg_id"] = pd.to_numeric(ra["dg_id"], errors="coerce").astype("Int64")
    ra["year"] = pd.to_numeric(ra["year"], errors="coerce").astype("Int64")
    ra["event_id"] = pd.to_numeric(ra["event_id"], errors="coerce").astype("Int64")

    # ---------- get course_num per (year,event_id) ----------
    if "course_num" in ra.columns and ra["course_num"].notna().any():
        ra["course_num"] = pd.to_numeric(ra["course_num"], errors="coerce")
        course_map = ra[["year", "event_id", "course_num"]].dropna().drop_duplicates()
    else:
        if sched_all is None or sched_all.empty:
            return pd.DataFrame()
        s = sched_all.copy()
        if "course_num" not in s.columns:
            return pd.DataFrame()

        s["year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
        s["event_id"] = pd.to_numeric(s["event_id"], errors="coerce").astype("Int64")
        s["course_num"] = pd.to_numeric(s["course_num"], errors="coerce")
        course_map = s[["year", "event_id", "course_num"]].dropna().drop_duplicates()

    # ---------- allowed events at this course ----------
    allowed = course_map.loc[course_map["course_num"] == int(course_num), ["year", "event_id"]].dropna().copy()
    allowed["year"] = allowed["year"].astype(int)
    allowed["event_id"] = allowed["event_id"].astype(int)
    allowed = allowed.drop_duplicates()
    if allowed.empty:
        return pd.DataFrame()

    # ---------- player rows at those events ----------
    dfp = ra.loc[ra["dg_id"] == int(dg_id)].copy()
    if dfp.empty:
        return pd.DataFrame()

    dfp = dfp.dropna(subset=["year", "event_id"]).copy()
    dfp["year"] = dfp["year"].astype(int)
    dfp["event_id"] = dfp["event_id"].astype(int)

    # keep only player rounds from events that were played at this course_num
    dfp = dfp.merge(allowed.assign(_ok=1), on=["year", "event_id"], how="inner")
    dfp = dfp.drop(columns=["_ok"])

    if dfp.empty:
        return pd.DataFrame()

    # ---------- tournament-level aggregate ----------
    def _first_finish(s: pd.Series) -> str:
        s2 = s.dropna().astype(str)
        return s2.iloc[0] if len(s2) else ""

    agg = {
        "Finish": ("fin_text", _first_finish),
        "SG PUTT": ("sg_putt", "sum"),
        "SG ARG": ("sg_arg", "sum"),
        "SG APP": ("sg_app", "sum"),
        "SG OTT": ("sg_ott", "sum"),
        "SG T2G": ("sg_t2g", "sum"),
        "SG Total": ("sg_total", "sum"),
    }
    present = {k: v for k, v in agg.items() if v[0] in dfp.columns}
    if "fin_text" not in dfp.columns:
        present["Finish"] = ("fin_text", _first_finish)

    t = (
        dfp.groupby(["year", "event_id", "event_name"], as_index=False)
           .agg(**present)
    )

    # field-wide end date
    ends_all = _build_event_end_table(ra)
    ends_all["year"] = pd.to_numeric(ends_all["year"], errors="coerce").astype("Int64")
    ends_all["event_id"] = pd.to_numeric(ends_all["event_id"], errors="coerce").astype("Int64")
    ends_all = ends_all.dropna(subset=["year", "event_id"]).copy()
    ends_all["year"] = ends_all["year"].astype(int)
    ends_all["event_id"] = ends_all["event_id"].astype(int)

    t = t.merge(ends_all[["year", "event_id", "event_end"]], on=["year", "event_id"], how="left")
    t["event_end"] = pd.to_datetime(t["event_end"], errors="coerce")

    # cutoff (anything before this date)
    if date_max is not None:
        dm = pd.to_datetime(date_max, errors="coerce")
        if pd.notna(dm):
            # year is still lowercase here
            fallback_end = pd.to_datetime(t["year"].astype(str) + "-12-31", errors="coerce")
            effective_end = t["event_end"].fillna(fallback_end)
            t = t.loc[effective_end <= dm].copy()

    if t.empty:
        return pd.DataFrame()

    # sort + format
    t = t.sort_values(["event_end", "year", "event_id"], ascending=[False, False, False]).head(int(n)).copy()
    t = t.rename(columns={"year": "Year", "event_name": "Event"})

    for c in ["SG PUTT", "SG ARG", "SG APP", "SG OTT", "SG T2G", "SG Total"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")

    cols = ["Year", "Event", "Finish", "SG Total",
            "SG PUTT", "SG ARG", "SG APP", "SG OTT", "SG T2G" ]
    cols = [c for c in cols if c in t.columns]
    return t[cols].reset_index(drop=True)

def plot_player_skill_radar(player_skills_df: pd.DataFrame, dg_id: int, title: str = "Skill Profile"):
    """Player-only radar; min-max normalized across full skills table."""
    if player_skills_df is None or player_skills_df.empty:
        st.warning("player_skills_df is empty; cannot plot radar.")
        return
    skills = player_skills_df.copy()
    if "dg_id" not in skills.columns:
        st.warning("player_skills_df missing dg_id; cannot plot radar.")
        return

    skills["dg_id"] = pd.to_numeric(skills["dg_id"], errors="coerce").astype("Int64")

    attr_map = {"DIST":"skill_dist","ACC":"skill_acc","APP":"skill_app","ARG":"skill_arg","PUTT":"skill_putt"}
    missing = [c for c in attr_map.values() if c not in skills.columns]
    if missing:
        st.warning(f"player_skills_df missing columns: {missing}")
        return

    row = skills.loc[skills["dg_id"] == int(dg_id)]
    if row.empty:
        st.warning(f"No skills row for dg_id={dg_id}")
        return
    row = row.iloc[0]

    mins = {k: float(pd.to_numeric(skills[col], errors="coerce").min()) for k, col in attr_map.items()}
    maxs = {k: float(pd.to_numeric(skills[col], errors="coerce").max()) for k, col in attr_map.items()}

    vec = {}
    for k, col in attr_map.items():
        v = pd.to_numeric(row.get(col, np.nan), errors="coerce")
        lo, hi = mins[k], maxs[k]
        vec[k] = float((float(v) - lo) / (hi - lo)) if np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi) and hi > lo else np.nan

    axes = ["DIST","ACC","APP","ARG","PUTT"]
    theta = axes + [axes[0]]
    r = [vec.get(a, np.nan) for a in axes]
    r = r + [r[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta, mode="lines+markers", name="Player",
        line=dict(width=3), marker=dict(size=6), fill="toself", opacity=1.0,
    ))
    fig.update_traces(hovertemplate="%{theta}: %{r:.2f}<extra></extra>")
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".2f"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_round_series(
    rounds_df: pd.DataFrame,
    dg_id: int,
    metric: str,
    date_min: Optional[pd.Timestamp],
    date_max: Optional[pd.Timestamp],
    ma_window: int = 50,
    show_trend: bool = True,
    title: str = "",
):
    df = rounds_df.loc[rounds_df["dg_id"] == int(dg_id)].copy()
    if df.empty:
        st.info("No round data for this player.")
        return

    use_dates = ("round_date" in df.columns) and df["round_date"].notna().any()
    if use_dates:
        df = df.dropna(subset=["round_date"]).copy()
        if date_min is not None:
            df = df[df["round_date"] >= pd.to_datetime(date_min)]
        if date_max is not None:
            df = df[df["round_date"] <= pd.to_datetime(date_max)]
        sort_cols = [c for c in ["round_date", "event_id", "round_num"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        x = df["round_date"]
        x_label = "Date"
    else:
        sort_cols = [c for c in ["year", "event_id", "round_num"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        df["_x"] = np.arange(len(df), dtype=int)
        x = df["_x"]
        x_label = "Round index (round_date missing)"

    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found in combined rounds.")
        return

    y = pd.to_numeric(df[metric], errors="coerce")
    df = df.loc[y.notna()].copy()
    if df.empty:
        st.info("No non-null values for this metric in the selected range.")
        return

    x = df["round_date"] if use_dates else df["_x"]
    y = pd.to_numeric(df[metric], errors="coerce")

    ma_window = max(1, int(ma_window))
    ma = y.rolling(ma_window, min_periods=max(5, ma_window // 3)).mean()

    hover_cols = [c for c in ["event_name", "year", "event_id", "round_num"] if c in df.columns]
    customdata = df[hover_cols].to_numpy() if hover_cols else None
    idx = {c: i for i, c in enumerate(hover_cols)}

    def _hover_line(prefix: str) -> str:
        ht = prefix
        if "event_name" in idx:
            ht += f"<br>Event: %{{customdata[{idx['event_name']}]}}"
        if "year" in idx:
            ht += f"<br>Year: %{{customdata[{idx['year']}]}}"
        if "event_id" in idx:
            ht += f"<br>event_id: %{{customdata[{idx['event_id']}]}}"
        if "round_num" in idx:
            ht += f"<br>Round: %{{customdata[{idx['round_num']}]}}"
        ht += "<extra></extra>"
        return ht

    fig = go.Figure()

    colors = np.where(y >= 0, "rgba(89, 201, 140, 0.65)", "rgba(231, 76, 60, 0.65)")
    fig.add_trace(go.Bar(
        x=x, y=y, name=metric,
        marker=dict(color=colors),
        customdata=customdata,
        hovertemplate=_hover_line(f"{metric}: %{{y:.2f}}"),
    ))

    fig.add_trace(go.Scatter(
        x=x, y=ma, mode="lines",
        name=f"{ma_window}-round MA",
        line=dict(width=3, color="rgba(240,240,240,0.9)"),
        customdata=customdata,
        hovertemplate=_hover_line(f"{ma_window}-round MA: %{{y:.2f}}"),
    ))

    if show_trend and len(df) >= 10:
        xn = (df["round_date"].astype("int64") / 1e9).to_numpy() if use_dates else df["_x"].to_numpy()
        yn = y.to_numpy()
        m, b = np.polyfit(xn, yn, 1)
        yhat = m * xn + b
        fig.add_trace(go.Scatter(
            x=x, y=yhat, mode="lines",
            name="Trend",
            line=dict(width=2, dash="dash", color="rgba(200,200,200,0.9)"),
            customdata=customdata,
            hovertemplate=_hover_line("Trend: %{y:.2f}"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=22), pad=dict(b=20)),
        height=560,
        margin=dict(l=60, r=40, t=90, b=105),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(size=13)),
    )

    fig.update_yaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.25)",
        zerolinewidth=1,
    )
    fig.update_xaxes(title=x_label)

    st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB 2 UI
# =========================
with tab2:
    st.header("Player Profile")

    rounds_all = load_combined_rounds()
    sched_here = load_schedule(season)  # assumes you already have load_schedule(season) in your app

    # --- pre-event cutoff: (event date - 1 day). This is the ONLY cutoff used everywhere pre-event matters.
    event_dt = _get_event_date_from_schedule(sched_here, int(event_id))
    course_cutoff = pd.to_datetime(event_dt, errors="coerce") if event_dt is not None else None
    pre_event_cutoff = None
    if event_dt is not None and pd.notna(event_dt):
        pre_event_cutoff = pd.to_datetime(event_dt, errors="coerce") - pd.Timedelta(days=1)

    # -------------------------
    # Player selector
    # -------------------------
    # Prefer current-season list using YEAR (do NOT use "season" column)
    if "year" in rounds_all.columns:
        candidates = (
            rounds_all.loc[rounds_all["year"] == int(season), ["player_name", "dg_id"]]
            .dropna()
            .drop_duplicates()
            .sort_values("player_name")
        )
    else:
        candidates = pd.DataFrame(columns=["player_name", "dg_id"])

    if candidates.empty:
        candidates = rounds_all[["player_name", "dg_id"]].dropna().drop_duplicates().sort_values("player_name")

    candidates["dg_id"] = pd.to_numeric(candidates["dg_id"], errors="coerce").astype("Int64")
    candidates = candidates.dropna(subset=["dg_id"]).copy()
    candidates["dg_id"] = candidates["dg_id"].astype(int)
    candidates["player_name"] = candidates["player_name"].astype(str)

    # show_names must already exist (your sidebar control)
    if show_names:
        candidates["label"] = candidates["player_name"] + " — " + candidates["dg_id"].astype(str)
    else:
        candidates["label"] = candidates["dg_id"].astype(str)

    labels = candidates["label"].tolist()
    label_to_id = dict(zip(labels, candidates["dg_id"]))

    # default = top player from tab1 if available
    default_index = 0
    if "summary_top" in globals() and isinstance(summary_top, pd.DataFrame) and not summary_top.empty and "dg_id" in summary_top.columns:
        top_dg_id = int(summary_top.iloc[0]["dg_id"])
        for i, lab in enumerate(labels):
            if int(label_to_id.get(lab)) == top_dg_id:
                default_index = i
                break

    label_sel = st.selectbox("Player", labels, index=default_index)
    dg_id_sel = int(label_to_id[label_sel])

    if show_names:
        player_name_sel = candidates.loc[candidates["dg_id"] == dg_id_sel, "player_name"].iloc[0]
    else:
        player_name_sel = f"dg_id {dg_id_sel}"

    # -------------------------
    # Top row: recent finishes + radar
    # -------------------------
    colL, colR = st.columns([1.25, 1], gap="large")

    with colL:
        st.subheader("Recent finishes (last 25 events)")
        last25 = build_last_n_events_table(rounds_all, dg_id_sel, n=25, date_max=pre_event_cutoff)
        if last25.empty:
            st.info("No recent events found for this player.")
        else:
            st.dataframe(last25[["Event", "Finish", "SG Total", "Year"]], use_container_width=True, hide_index=True)

    with colR:
        st.subheader("Skill profile")
        try:
            plot_player_skill_radar(player_skills_df, dg_id_sel, title=f"{player_name_sel} — Skill Profile")
        except Exception as e:
            st.warning(f"Radar error: {e}")

    # -------------------------
    # Course history (this week’s course)
    # -------------------------
    st.divider()
    st.markdown("### Course history (this week’s course)")

    # schedule for the currently selected season (the one you’re picking)
    sched_here = load_schedule(season)

    # cutoff = day before the selected event (so we don't include the event itself)
    pre_event_cutoff = get_pre_event_cutoff_date(sched_here, int(event_id))  # event_date - 1 day

    # --- find course_num for the selected event_id from the schedule ---
    course_num_here = None
    if "course_num" in sched_here.columns:
        r = sched_here.loc[pd.to_numeric(sched_here["event_id"], errors="coerce") == int(event_id)]
        if not r.empty:
            course_num_here = pd.to_numeric(r.iloc[0]["course_num"], errors="coerce")

    if course_num_here is None or not np.isfinite(course_num_here):
        st.info("No course_num found for this event in schedule; cannot build course history.")
    else:
        st.write("rounds_all has course_num:", "course_num" in rounds_all.columns)
        course_hist = build_course_history_table(
            rounds_all=rounds_all,  # combined_rounds_all_2017_2025.csv
            dg_id=int(dg_id_sel),
            course_num=int(course_num_here),
            date_max=pre_event_cutoff,  # anything before event start (event_date - 1)
            n=25,
            sched_all=sched_here,  # ONLY used if rounds_all lacks course_num
        )

        if course_hist.empty:
            st.info("No prior starts for this player at this course (before the event).")
        else:
            st.dataframe(course_hist, use_container_width=True, hide_index=True)

    # -------------------------
    # Rolling / per-round chart
    # -------------------------
    st.divider()
    st.subheader("Rolling stats (per round)")

    metric_options = [c for c in [
        "sg_total","sg_t2g","sg_putt","sg_arg","sg_app","sg_ott",
        "driving_dist","driving_acc","gir","scrambling","prox_rgh","prox_fw",
        "great_shots","poor_shots",
        "round_score",
        "birdies","bogies","doubles_or_worse",
    ] if c in rounds_all.columns]

    metric = st.selectbox(
        "Metric",
        metric_options,
        index=metric_options.index("sg_total") if "sg_total" in metric_options else 0,
    )

    show_trend = st.checkbox("Show trendline", value=True)
    ma_window = st.slider("Moving average window (rounds)", min_value=5, max_value=200, value=50, step=5)

    # Date range (only if round_date exists for this player)
    date_min = None
    date_max = None

    df_player_dates = rounds_all.loc[rounds_all["dg_id"] == dg_id_sel, ["round_date"]].copy()
    has_dates = ("round_date" in df_player_dates.columns) and df_player_dates["round_date"].notna().any()

    if has_dates:
        min_d = pd.to_datetime(df_player_dates["round_date"].min(), errors="coerce")
        max_d = pd.to_datetime(df_player_dates["round_date"].max(), errors="coerce")

        # Right bound defaults to pre-event cutoff, else latest available round
        right = pd.to_datetime(pre_event_cutoff, errors="coerce") if pre_event_cutoff is not None else pd.to_datetime(max_d, errors="coerce")
        if pd.isna(right):
            right = pd.to_datetime(max_d, errors="coerce")

        # Left bound defaults to ~2 years before right
        left = right - pd.Timedelta(days=365 * 2)

        # Clamp to available
        if pd.notna(min_d):
            left = max(left, min_d)
        if pd.notna(max_d):
            right = min(right, max_d)

        if pd.notna(left) and pd.notna(right) and left > right:
            left, right = min_d, max_d

        d0, d1 = st.date_input(
            "Date range",
            value=(left.date(), right.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
        date_min = pd.to_datetime(d0)
        date_max = pd.to_datetime(d1)

    plot_round_series(
        rounds_df=rounds_all,
        dg_id=dg_id_sel,
        metric=metric,
        date_min=date_min,
        date_max=date_max,
        ma_window=ma_window,
        show_trend=show_trend,
        title=f"{player_name_sel} — {metric} per round",
    )

    # -------------------------
    # Tournament breakdown table
    # -------------------------
    with st.expander("Tournament breakdown (last 25 starts)", expanded=True):
        dfp = rounds_all.loc[rounds_all["dg_id"] == dg_id_sel].copy()
        if dfp.empty:
            st.info("No tournament data.")
        else:
            ends_all = _global_tournament_end_dates(rounds_all)

            t = (
                dfp.groupby(["year", "event_id", "event_name"], as_index=False)
                   .agg(
                       Finish=("fin_text", lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else ""),
                       SG_PUTT=("sg_putt", "sum"),
                       SG_ARG=("sg_arg", "sum"),
                       SG_APP=("sg_app", "sum"),
                       SG_OTT=("sg_ott", "sum"),
                       SG_TOTAL=("sg_total", "sum"),
                       DIST=("driving_dist", "mean"),
                       ACC=("driving_acc", "mean"),
                   )
            )

            t = t.merge(ends_all, on=["year", "event_id", "event_name"], how="left")
            t["event_end"] = pd.to_datetime(t["event_end"], errors="coerce")

            # Apply the SAME date window as the chart (if user picked one)
            if date_min is not None:
                t = t[t["event_end"] >= pd.to_datetime(date_min)]
            if date_max is not None:
                t = t[t["event_end"] <= pd.to_datetime(date_max)]

            if t.empty:
                st.info("No tournaments in the selected date range.")
            else:
                t["Event date"] = t["event_end"].dt.date.astype(str)
                t = t.sort_values(["event_end", "year", "event_id"], ascending=[False, False, False]).head(25).copy()

                for c in ["SG_PUTT", "SG_ARG", "SG_APP", "SG_OTT", "SG_TOTAL"]:
                    if c in t.columns:
                        t[c] = t[c].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
                if "DIST" in t.columns:
                    t["DIST"] = t["DIST"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
                if "ACC" in t.columns:
                    t["ACC"] = t["ACC"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")

                t = t.rename(columns={"event_name": "Event", "year": "Year"})
                show_cols = ["Event", "Event date", "Finish", "SG_PUTT", "SG_ARG", "SG_APP", "SG_OTT", "SG_TOTAL", "DIST", "ACC", "Year"]
                show_cols = [c for c in show_cols if c in t.columns]
                st.dataframe(t[show_cols], use_container_width=True, hide_index=True)

# =========================
# TAB 3: SEASON / LEAGUE
# =========================
with tab3:
    st.header("Season / League")
    st.subheader("League movement (cumulative winnings through the prior event)")

    # -------------------------
    # Load inputs
    # -------------------------
    league_all = load_league(season)
    sched2 = load_schedule(season)

    # -------------------------
    # Basic validation
    # -------------------------
    need_league_cols = {"league_id", "entry_id", "username", "event_id", "raw_winnings"}
    missing_league = need_league_cols - set(league_all.columns)
    if missing_league:
        st.error(f"League file missing required columns: {sorted(missing_league)}")
        st.stop()

    if "event_id" not in sched2.columns:
        st.error("Schedule is missing required column: event_id")
        st.stop()

    if "event_order" not in sched2.columns:
        st.error("Schedule is missing required column: event_order")
        st.stop()

    # -------------------------
    # League selection
    # -------------------------
    league_ids = sorted(league_all["league_id"].dropna().astype(str).unique().tolist())
    if not league_ids:
        st.info("No leagues found in league file.")
        st.stop()

    selected_league_id = st.selectbox("League", league_ids, index=0, key="tab3_league")
    league = league_all[league_all["league_id"].astype(str) == str(selected_league_id)].copy()

    # -------------------------
    # Raw winnings meaning (advanced)
    # -------------------------
    raw_override = None
    with st.expander("Advanced (raw_winnings meaning)", expanded=False):
        raw_mode = st.selectbox(
            "raw_winnings meaning",
            ["Auto-detect", "Per-event payout", "Already cumulative (season-to-date)"],
            index=0,
            help=(
                "Auto-detect tries to infer whether raw_winnings is already season-to-date or per-event. "
                "Use overrides if a league scrape behaves differently."
            ),
            key="tab3_rawmode",
        )
        if raw_mode == "Per-event payout":
            raw_override = False
        elif raw_mode == "Already cumulative (season-to-date)":
            raw_override = True

    # -------------------------
    # Current event id sanity
    # -------------------------
    try:
        cur_event_id_int = int(event_id)
    except Exception:
        st.error(f"event_id in app state is not an int: {event_id!r}")
        st.stop()

    # schedule cleanup
    sched_clean = sched2.copy()
    sched_clean["event_id"] = pd.to_numeric(sched_clean["event_id"], errors="coerce")
    sched_clean["event_order"] = pd.to_numeric(sched_clean["event_order"], errors="coerce")
    sched_clean = sched_clean.dropna(subset=["event_id", "event_order"]).copy()
    sched_clean["event_id"] = sched_clean["event_id"].astype(int)
    sched_clean["event_order"] = sched_clean["event_order"].astype(int)

    max_week = int(sched_clean["event_order"].max())

    # auto week = current event's order - 1 (clamped)
    cur_order_series = sched_clean.loc[sched_clean["event_id"] == cur_event_id_int, "event_order"]
    if len(cur_order_series) == 0:
        auto_week = min(max_week, 1)
        st.warning(
            f"Current event_id={cur_event_id_int} not found in schedule for season={season}. "
            "Using week=1 as default until schedule/event_ids are aligned."
        )
    else:
        auto_week = int(cur_order_series.iloc[0]) - 1
        auto_week = max(0, min(max_week, auto_week))

    # -------------------------
    # ONE control for week: follow selected event unless end-of-season is checked
    # -------------------------
    end_season = st.checkbox(
        "End of season",
        value=False,
        help="When checked, forces standings to the last scheduled week.",
        key="tab3_end_season",
    )

    cut_week = max_week if end_season else auto_week
    cut_week = max(1, int(cut_week))  # clamp (week 0 = none)

    # st.caption(
    #     f"Viewing: {'end of season' if end_season else 'auto (selected event - 1)'} | "
    #     f"week={cut_week}"
    # )
    #
    # # Optional: show what it's doing (no control)
    # st.caption(
    #     f"Viewing: {'end of season' if end_season else 'auto (selected event - 1)'} | "
    #     f"week={cut_week}"
    # )

    # -------------------------
    # Compute FULL standings through prior event (entire league)
    # NOTE: we still ask your builder for ALL rows; we will slice to cut_week ourselves.
    # -------------------------
    standings_all, cutoff, raw_used = build_league_standings_through_prior(
        league_df=league,
        schedule_df=sched2,
        current_event_id=cur_event_id_int,
        username_you=username,
        top_n=10000,  # effectively "all entries"
        raw_is_cumulative=raw_override,
    )

    if standings_all is None or standings_all.empty:
        st.info("No league rows found (after schedule mapping).")
        with st.expander("Debug: league rows (first 50)"):
            st.dataframe(league.head(50), use_container_width=True, hide_index=True)
        st.stop()

    # normalize standings_all
    snap_all = standings_all.copy()
    snap_all["event_order"] = pd.to_numeric(snap_all.get("event_order"), errors="coerce")
    snap_all["cum_winnings"] = pd.to_numeric(snap_all.get("cum_winnings"), errors="coerce").fillna(0.0)
    snap_all = snap_all.dropna(subset=["label", "event_order"]).copy()
    snap_all["event_order"] = snap_all["event_order"].astype(int)

    # -------------------------
    # Helper: "latest <= week" snapshot per label
    # (THIS is the fix for your week31 missing/empty tables)
    # -------------------------
    def latest_snapshot_upto(df: pd.DataFrame, week: int) -> pd.DataFrame:
        d = df[df["event_order"] <= int(week)].copy()
        if d.empty:
            return d
        d = d.sort_values(["label", "event_order"])
        return d.groupby("label", as_index=False).tail(1)[["label", "cum_winnings", "event_order"]]

    # totals at cut_week for EVERYONE
    league_totals_all = latest_snapshot_upto(snap_all, cut_week)
    n_entries = int(len(league_totals_all))

    # paid cutoff at cut_week (7th place)
    paid_cut = None
    if n_entries >= 7:
        tmp = league_totals_all.sort_values("cum_winnings", ascending=False).reset_index(drop=True)
        paid_cut = float(tmp.loc[6, "cum_winnings"])

    # -------------------------
    # Prior event label (nice caption)
    # -------------------------
    prior_event_label = None
    pr = sched_clean[sched_clean["event_order"] == int(cut_week)].head(1)
    if not pr.empty:
        if "event_name" in pr.columns and pd.notna(pr.iloc[0].get("event_name")):
            prior_event_label = str(pr.iloc[0]["event_name"])
        else:
            prior_event_label = f"event_id={int(pr.iloc[0]['event_id'])}"

    # st.caption(
    #     f"Cutoff: through week {cut_week} (prior event). raw_is_cumulative used: {raw_used}"
    #     + (f" | prior event: {prior_event_label}" if prior_event_label else "")
    # )

    # -------------------------
    # Helper: build your tracked schedule-aligned series + table
    # -------------------------
    def build_you_tracked_series(
        season: int,
        mode: str,
        username: str,
        sched2: pd.DataFrame,
        cut_week: int,
    ) -> pd.DataFrame:
        s = sched2.copy()
        s["event_id"] = pd.to_numeric(s.get("event_id"), errors="coerce")
        s["event_order"] = pd.to_numeric(s.get("event_order"), errors="coerce")
        s = s.dropna(subset=["event_id", "event_order"]).copy()
        s["event_id"] = s["event_id"].astype(int)
        s["event_order"] = s["event_order"].astype(int)

        name_col = "event_name" if "event_name" in s.columns else None
        if name_col is None:
            s["event_name"] = s["event_id"].astype(str)
            name_col = "event_name"
        else:
            s[name_col] = s[name_col].astype(str)

        s = s[s["event_order"] <= int(cut_week)].sort_values("event_order")
        base = s[["event_order", "event_id", name_col]].rename(columns={name_col: "event_name"}).copy()

        picks = load_picks_log(season, mode)
        if picks is None or picks.empty:
            base["player_picked"] = ""
            base["finish"] = ""
            base["winnings"] = 0.0
            base["cum_winnings"] = 0.0
            base["label"] = f"{username} (tracked)"
            base["username"] = username
            base["entry_id"] = "YOU_TRACKED"
            return base[[
                "label","username","entry_id",
                "event_order","event_id","event_name",
                "player_picked","finish","winnings","cum_winnings"
            ]]

        picks = picks.copy()
        picks["season"] = pd.to_numeric(picks.get("season"), errors="coerce")
        picks["event_id"] = pd.to_numeric(picks.get("event_id"), errors="coerce")
        picks["dg_id"] = pd.to_numeric(picks.get("dg_id"), errors="coerce")
        picks["username"] = picks.get("username", "").astype(str)

        picks = picks[(picks["season"] == int(season)) & (picks["username"] == str(username))].copy()
        picks = picks.dropna(subset=["event_id"]).copy()
        if picks.empty:
            base["player_picked"] = ""
            base["finish"] = ""
            base["winnings"] = 0.0
            base["cum_winnings"] = 0.0
            base["label"] = f"{username} (tracked)"
            base["username"] = username
            base["entry_id"] = "YOU_TRACKED"
            return base[[
                "label","username","entry_id",
                "event_order","event_id","event_name",
                "player_picked","finish","winnings","cum_winnings"
            ]]

        picks["event_id"] = picks["event_id"].astype(int)
        if "ts" in picks.columns:
            picks["ts"] = pd.to_datetime(picks["ts"], errors="coerce")
            picks = picks.sort_values("ts")

        # --- finish comes from picks log (authoritative) ---
        picks["finish_text"] = picks.get("finish_text", "").astype(str).replace({"nan": "", "None": ""}).str.strip()

        fn = pd.to_numeric(picks.get("finish_num"), errors="coerce")
        picks["finish_num_str"] = fn.map(lambda x: f"{int(x)}" if pd.notna(x) else "")

        # Prefer finish_text; fallback to finish_num
        picks["finish"] = picks["finish_text"]
        mask = picks["finish"].eq("") | picks["finish"].isna()
        picks.loc[mask, "finish"] = picks.loc[mask, "finish_num_str"]

        if "player_name" in picks.columns:
            picks["player_picked"] = picks["player_name"].astype(str)
        else:
            picks["player_picked"] = picks.get("dg_id", "").astype(str)

        odds = load_odds(season)
        odds = odds.copy() if odds is not None else pd.DataFrame()

        # normalize merge keys
        if "event_id" in odds.columns:
            odds["event_id"] = pd.to_numeric(odds["event_id"], errors="coerce").astype("Int64")
        if "dg_id" in odds.columns:
            odds["dg_id"] = pd.to_numeric(odds["dg_id"], errors="coerce").astype("Int64")

        # winnings + finish columns (prefer finish_text)
        winnings_col = "Winnings" if "Winnings" in odds.columns else (
            "winnings" if "winnings" in odds.columns else None)
        finish_text_col = "finish_text" if "finish_text" in odds.columns else None
        finish_num_col = "finish_num" if "finish_num" in odds.columns else None

        m = picks.copy()
        m["finish"] = m.get("finish", "").astype(str)
        m["dg_id"] = pd.to_numeric(m.get("dg_id"), errors="coerce")
        m = m.dropna(subset=["dg_id"]).copy()
        m["dg_id"] = m["dg_id"].astype(int)

        m["event_id"] = pd.to_numeric(m["event_id"], errors="coerce").astype("Int64")
        m["dg_id"] = pd.to_numeric(m["dg_id"], errors="coerce").astype("Int64")

        join_cols = ["event_id", "dg_id"]
        odds_keep = join_cols.copy()
        if winnings_col: odds_keep.append(winnings_col)
        if finish_text_col: odds_keep.append(finish_text_col)
        if finish_num_col: odds_keep.append(finish_num_col)

        if all(c in odds.columns for c in join_cols):
            m = m.merge(odds[odds_keep], on=join_cols, how="left")

        m["winnings_from_log"] = pd.to_numeric(m.get("winnings"), errors="coerce") if "winnings" in m.columns else np.nan
        m["winnings_from_odds"] = pd.to_numeric(m.get(winnings_col), errors="coerce") if winnings_col else np.nan
        m["winnings"] = m["winnings_from_odds"].where(m["winnings_from_odds"].notna(), m["winnings_from_log"])
        m["winnings"] = m["winnings"].fillna(0.0)

        m_small = m[["event_id", "player_picked", "finish", "winnings"]].copy()
        out = base.merge(m_small, on="event_id", how="left")

        out["player_picked"] = out["player_picked"].fillna("")
        out["finish"] = out["finish"].fillna("")
        out["winnings"] = pd.to_numeric(out["winnings"], errors="coerce").fillna(0.0)

        out = out.sort_values("event_order")
        out["cum_winnings"] = out["winnings"].cumsum()

        out["label"] = f"{username} (tracked)"
        out["username"] = username
        out["entry_id"] = "YOU_TRACKED"

        return out[[
            "label","username","entry_id",
            "event_order","event_id","event_name",
            "player_picked","finish","winnings","cum_winnings"
        ]]

    you_tbl = build_you_tracked_series(season, mode, username, sched2, cut_week)
    you_label = f"{username} (tracked)"

    you_total = 0.0
    if you_tbl is not None and not you_tbl.empty:
        you_total = float(pd.to_numeric(you_tbl["cum_winnings"], errors="coerce").fillna(0.0).iloc[-1])

    # rank vs full league totals at cut_week
    your_rank = None
    if n_entries > 0:
        your_rank = int((league_totals_all["cum_winnings"] > you_total).sum()) + 1

    # -------------------------
    # Chart + snapshot standings
    # -------------------------
    col_left, col_right = st.columns([2, 1], gap="large")

    plot_mode = st.selectbox(
        "Lines to plot",
        ["Top 10 + You", "Top 25 + You", "All players (slow)"],
        index=0,
        key="tab3_plot_mode",
    )

    if plot_mode.startswith("All"):
        labels_to_plot = set(league_totals_all["label"].tolist())
    else:
        k = 10 if plot_mode.startswith("Top 10") else 25
        topk = (
            league_totals_all.sort_values("cum_winnings", ascending=False)
            .head(k)["label"]
            .tolist()
        )
        labels_to_plot = set(topk)

    labels_to_plot.add(you_label)

    # plot data: use all points <= cut_week for those labels
    standings_plot = snap_all[(snap_all["event_order"] <= cut_week) & (snap_all["label"].isin(labels_to_plot))].copy()

    # remove any league "you_label" if it exists (avoid duplicate line)
    standings_plot = standings_plot[standings_plot["label"] != you_label].copy()

    # append tracked line
    if you_tbl is not None and not you_tbl.empty:
        you_line = you_tbl[["label", "event_order", "cum_winnings"]].copy()
        you_line["event_order"] = pd.to_numeric(you_line["event_order"], errors="coerce").fillna(0).astype(int)
        you_line["cum_winnings"] = pd.to_numeric(you_line["cum_winnings"], errors="coerce").fillna(0.0)
        standings_plot = pd.concat([standings_plot, you_line], ignore_index=True)

    standings_plot = standings_plot.sort_values(["label", "event_order"]).reset_index(drop=True)

    with col_left:
        fig = px.line(
            standings_plot,
            x="event_order",
            y="cum_winnings",
            color="label",
            title=f"{season} cumulative winnings ({plot_mode}) — through week {cut_week}",
        )
        fig.update_xaxes(dtick=1, tick0=1)

        if paid_cut is not None:
            fig.add_hline(
                y=paid_cut,
                line_dash="dash",
                line_color="#00FF00",
                annotation_text="Paid cutoff (Top 7)",
                annotation_position="top left",
            )

        fig.update_layout(height=550, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Standings at cutoff (Top shown; rank computed on full league)**")

        prev_week = max(cut_week - 1, 1)

        # snapshot at cut_week = latest <= cut_week
        cur_full = latest_snapshot_upto(snap_all, cut_week).copy()
        prev_full = latest_snapshot_upto(snap_all, prev_week).copy()

        cur_full = cur_full.sort_values("cum_winnings", ascending=False).reset_index(drop=True)
        cur_full["rank"] = np.arange(1, len(cur_full) + 1, dtype=int)

        prev_full = prev_full.sort_values("cum_winnings", ascending=False).reset_index(drop=True)
        prev_full["prev_rank"] = np.arange(1, len(prev_full) + 1, dtype=int)

        cur_full = cur_full.merge(prev_full[["label", "prev_rank"]], on="label", how="left")

        # fewer controls: use a selectbox instead of a slider
        rows_to_show = st.selectbox(
            "Rows to show (standings)",
            options=[10, 20, 30, 49, 100],
            index=0,
            key="tab3_rows_to_show",
        )
        rows_to_show = min(int(rows_to_show), len(cur_full))

        show = cur_full.head(rows_to_show).copy()

        # ensure tracked shows even if outside
        if you_label not in set(show["label"]):
            show = pd.concat([show, pd.DataFrame([{
                "label": you_label,
                "cum_winnings": you_total,
                "rank": your_rank,
                "prev_rank": np.nan,
            }])], ignore_index=True)

        show["Total ($)"] = show["cum_winnings"].map(lambda x: f"${float(x):,.0f}")
        show["Prev rank"] = show["prev_rank"].astype("Int64")
        show["Paid"] = show["rank"].map(
            lambda r: (
                "🥇 $15,000" if pd.notna(r) and int(r) == 1 else
                "🥈 $10,000" if pd.notna(r) and int(r) == 2 else
                "🥉 $7,500" if pd.notna(r) and int(r) == 3 else
                "💰 $5,000" if pd.notna(r) and int(r) == 4 else
                "💰 $3,000" if pd.notna(r) and int(r) == 5 else
                "💰 $2,500" if pd.notna(r) and int(r) == 6 else
                "💰 $2,000" if pd.notna(r) and int(r) == 7 else
                ""
            )
        )
        show = show.rename(columns={"label": "User"})[["User", "Paid", "Total ($)", "Prev rank"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

    # -------------------------
    # Your tracked position
    # -------------------------
    st.markdown("### Your tracked position (from picks log)")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tracked total", f"${you_total:,.0f}")
    c2.metric("Projected rank", f"{your_rank}/{n_entries}" if your_rank is not None else "—")
    c3.metric("Paid cutoff (Top 7)", f"${paid_cut:,.0f}" if paid_cut is not None else "—")

    gap_to_paid = max(0.0, paid_cut - you_total) if paid_cut is not None else None
    c4.metric("Gap to paid", f"${gap_to_paid:,.0f}" if gap_to_paid is not None else "—")

    st.markdown("### Your tracked picks")

    if you_tbl is None or you_tbl.empty:
        st.info("No tracked picks yet.")
    else:
        show = you_tbl.copy().rename(columns={
            "event_name": "Event",
            "player_picked": "Player",
            "finish": "Finish",
            "winnings": "Winnings",
            "cum_winnings": "Cum winnings",
        })

        show["Winnings"] = pd.to_numeric(show.get("Winnings"), errors="coerce").fillna(0.0)
        show["Cum winnings"] = pd.to_numeric(show.get("Cum winnings"), errors="coerce").fillna(0.0)

        show["Winnings"] = show["Winnings"].map(lambda x: f"${x:,.0f}")
        show["Cum winnings"] = show["Cum winnings"].map(lambda x: f"${x:,.0f}")

        cols = ["Event", "Player"]
        if "Finish" in show.columns:
            cols.append("Finish")
        cols += ["Winnings", "Cum winnings"]

        st.dataframe(show[cols], use_container_width=True, hide_index=True)
    # -------------------------
    # Pick popularity through cutoff (league-wide)
    # -------------------------
    st.markdown("### Pick popularity (through prior event)")

    order_map = build_event_order_map(sched2)

    picks = league.copy()
    picks["entry_id"] = picks["entry_id"].astype(str)
    picks["player_name"] = picks["player_name"].astype(str) if "player_name" in picks.columns else ""

    picks["event_id"] = pd.to_numeric(picks["event_id"], errors="coerce")
    picks = picks.dropna(subset=["event_id"]).copy()
    picks["event_id"] = picks["event_id"].astype(int)

    picks["event_order"] = picks["event_id"].map(order_map)
    picks = picks.dropna(subset=["event_order"]).copy()
    picks["event_order"] = picks["event_order"].astype(int)

    picks = picks[picks["event_order"] <= int(cut_week)].copy()

    picks["player_name_clean"] = picks["player_name"].str.strip()
    picks = picks[picks["player_name_clean"].ne("")].copy()
    picks = picks[~picks["player_name_clean"].str.lower().isin({"none", "nan", "no pick", "null"})].copy()

    total_entries = picks["entry_id"].nunique()
    if total_entries == 0:
        st.info("No picks found through the cutoff.")
    else:
        uniq = picks.drop_duplicates(subset=["entry_id", "player_name_clean"]).copy()
        pop = (
            uniq.groupby("player_name_clean")["entry_id"]
            .nunique()
            .reset_index(name="entries_picked")
        )
        pop["pick_pct"] = pop["entries_picked"] / float(total_entries)
        pop = pop.sort_values(["pick_pct", "entries_picked", "player_name_clean"], ascending=[False, False, True])

        pop["Pick %"] = pop["pick_pct"].map(lambda x: f"{x:.1%}")
        pop = pop.rename(columns={"player_name_clean": "Player", "entries_picked": "Entries picked"})

        # keep this small; change to selectbox if you want zero sliders
        show_n = st.selectbox(
            "Rows to show",
            options=[10, 20, 30, 50, 100],
            index=2,
            key="tab3_pop_rows",
        )
        st.dataframe(pop[["Player", "Entries picked", "Pick %"]].head(int(show_n)),
                     use_container_width=True, hide_index=True)

        st.caption(f"Denominator: {total_entries} distinct entries (through week {cut_week}).")

# # =========================
# # TAB 4: COMPARE
# # =========================
# with tab4:
#     st.header("Compare (Top 2 by oad_score)")
#
#     # leakage-free cutoff: start_date - 1 day (your helper already does this)
#     cutoff_dt = get_pre_event_cutoff_date(sched2, event_id)
#     if cutoff_dt is None or pd.isna(cutoff_dt):
#         raise ValueError("[tab4] could not compute cutoff_dt from schedule (need start_date/event_date for this event_id).")
#
#     # explicit columns you told me to use (NO autodetect, NO global name)
#     COMPARE_COLS = [
#         "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total",
#         "driving_dist", "driving_acc",
#     ]
#
#     from Scripts.compare import render_compare_tab
#
#     render_compare_tab(
#         summary=summary,
#         rounds_all=rounds_all,
#         cutoff_dt=cutoff_dt,
#         last_n_events_default=30,
#         player_skills_df=player_skills_df,
#         plot_player_skill_radar=plot_player_skill_radar,
#         compare_cols=COMPARE_COLS,
#         score_col="oad_score",
#         dg_id_col="dg_id",
#     )
