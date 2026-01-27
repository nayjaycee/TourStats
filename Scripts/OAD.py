# Scripts/app.py
from __future__ import annotations

import sys
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re

# ============================================================
# PATH + IMPORT FIX
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../OAD
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "Data"
DATA_IN_USE = DATA_ROOT / "in Use"

COMBINED_ROUNDS_PATH = DATA_IN_USE / "combined_rounds_all_2017_2026.csv"
ODDS_AND_RESULTS_XLSX = DATA_IN_USE / "Odds_and_Results.xlsx"
PICKS_LOG_DIR = DATA_IN_USE / "Picks Log"

YOU_USERNAME = "You"
TOP_N_DEFAULT = 30
SHORTLIST_PATH = DATA_IN_USE / "preseason_shortlist_2026.csv"

# ============================================================
# OPTIONAL IMPORTS (LOCAL)
# ============================================================
from Scripts.league import build_league_standings_through_prior, build_event_order_map
from Scripts.ui_condensed_tables import CONDENSED_GROUPS, render_grouped_condensed_tables
from Scripts.data_io import load_rounds


def _safe_import_load_rounds():
    try:
        from Scripts.data_io import load_rounds  # type: ignore
        return load_rounds, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _safe_import_build_weekly_view():
    try:
        from Scripts.weekly_view import build_weekly_view  # type: ignore
        return build_weekly_view, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _safe_import_course_tables():
    try:
        from Scripts.data_io import load_course_fit, load_player_skills  # type: ignore
        return load_course_fit, load_player_skills, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


LOAD_ROUNDS, LOAD_ROUNDS_ERR = _safe_import_load_rounds()
BUILD_WEEKLY_VIEW, WEEKLY_VIEW_IMPORT_ERR = _safe_import_build_weekly_view()
LOAD_COURSE_FIT, LOAD_PLAYER_SKILLS, COURSE_TABLES_ERR = _safe_import_course_tables()

# ============================================================
# STREAMLIT SETTINGS
# ============================================================
st.set_page_config(page_title="One and Done", layout="wide")

# ============================================================
# LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def _load_rounds_all():
    df = load_rounds()
    # safety: make sure dates are parsed
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_combined_rounds() -> pd.DataFrame:
    if not COMBINED_ROUNDS_PATH.exists():
        raise FileNotFoundError(f"Missing combined rounds file: {COMBINED_ROUNDS_PATH}")
    df = pd.read_csv(COMBINED_ROUNDS_PATH, low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    path = DATA_IN_USE / f"OAD_{int(season)}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing schedule file: {path}")

    df = pd.read_excel(path)

    # normalize numeric
    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # find a date-ish column for ordering
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        dcol = date_cols[0]
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.sort_values(dcol).reset_index(drop=True)

    # create event_order if missing
    if "event_order" not in df.columns:
        df["event_order"] = np.arange(1, len(df) + 1)

    # make event_order numeric
    df["event_order"] = pd.to_numeric(df["event_order"], errors="coerce")

    # clean
    df = df.dropna(subset=["event_id"]).copy()
    df["event_id"] = df["event_id"].astype(int)

    # ensure event_name exists
    if "event_name" not in df.columns:
        df["event_name"] = df["event_id"].astype(str)

    df["event_name"] = df["event_name"].astype(str)
    return df

def _norm_name(x: str) -> str:
    x = str(x or "").strip().lower()
    x = x.replace("&", "and")
    x = "".join(ch for ch in x if ch.isalnum() or ch.isspace())
    x = " ".join(x.split())
    return x

def normalize_league_df(df: pd.DataFrame, sched: pd.DataFrame, season: int, league_id_value: str = "main") -> pd.DataFrame:
    df = df.copy()

    if "username" not in df.columns and "user" in df.columns:
        df["username"] = df["user"]

    if "entry_id" not in df.columns:
        if "entryId" in df.columns:
            df["entry_id"] = df["entryId"].astype(str)
        elif "entryid" in df.columns:
            df["entry_id"] = df["entryid"].astype(str)

    if "event_name" not in df.columns:
        if "eventName" in df.columns:
            df["event_name"] = df["eventName"].astype(str)
        elif "event" in df.columns:
            df["event_name"] = df["event"].astype(str)

    if "raw_winnings" not in df.columns:
        if "winnings" in df.columns:
            df["raw_winnings"] = pd.to_numeric(df["winnings"], errors="coerce").fillna(0.0)
        else:
            df["raw_winnings"] = 0.0

    if "league_id" not in df.columns or df["league_id"].isna().all():
        df["league_id"] = league_id_value

    if "year" not in df.columns or df["year"].isna().all():
        if "eventDate" in df.columns:
            dt = pd.to_datetime(df["eventDate"], errors="coerce")
            df["year"] = dt.dt.year
        else:
            df["year"] = int(season)

    if "event_id" not in df.columns or df["event_id"].isna().all():
        s = sched.copy()
        if "event_name" not in s.columns and "eventName" in s.columns:
            s["event_name"] = s["eventName"]

        s = s.dropna(subset=["event_id", "event_name"]).copy()
        s["event_id"] = pd.to_numeric(s["event_id"], errors="coerce")
        s = s.dropna(subset=["event_id"]).copy()
        s["event_id"] = s["event_id"].astype(int)

        s["_event_key"] = s["event_name"].astype(str).map(_norm_name)
        df["_event_key"] = df["event_name"].astype(str).map(_norm_name)

        df = df.merge(s[["_event_key", "event_id"]], on="_event_key", how="left")
        df = df.drop(columns=["_event_key"])

    if "week_num" in sched.columns:
        m = sched[["event_id", "week_num"]].copy()
        m["event_id"] = pd.to_numeric(m["event_id"], errors="coerce")
        m["week_num"] = pd.to_numeric(m["week_num"], errors="coerce")
        m = m.dropna(subset=["event_id", "week_num"]).copy()
        m["event_id"] = m["event_id"].astype(int)
        m["week_num"] = m["week_num"].astype(int)
        df = df.merge(m, on="event_id", how="left")
    elif "event_order" in sched.columns:
        m = sched[["event_id", "event_order"]].copy()
        m["event_id"] = pd.to_numeric(m["event_id"], errors="coerce")
        m["event_order"] = pd.to_numeric(m["event_order"], errors="coerce")
        m = m.dropna(subset=["event_id", "event_order"]).copy()
        m["event_id"] = m["event_id"].astype(int)
        m["event_order"] = m["event_order"].astype(int)
        df = df.merge(m, on="event_id", how="left")
        df["week_num"] = df["event_order"]

    return df

@st.cache_data(show_spinner=False)
def load_league(season: int) -> pd.DataFrame:
    path = DATA_ROOT / "Clean" / "Leagues" / f"{int(season)}_small_normalized.csv"

    expected_cols = [
        "league_id", "entry_id", "username",
        "event_id", "event_name",
        "selection",  # <-- add this
        "dg_id", "player_name",
        "raw_winnings",
        "week_num", "year",
    ]

    if not path.exists():
        return pd.DataFrame(columns=expected_cols)

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame(columns=expected_cols)
    except OSError:
        return pd.DataFrame(columns=expected_cols)

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=expected_cols)

    # -----------------------------
    # NEW: normalize raw export -> expected schema
    # -----------------------------
    try:
        sched = load_schedule(int(season))  # uses your OAD_YYYY.xlsx loader
        df = normalize_league_df(df, sched=sched, season=int(season), league_id_value="main")
    except Exception:
        # if schedule load/normalize fails, keep df as-is (Tab4 debug will show why)
        pass

    # ensure expected columns exist
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # keep only expected cols (prevents raw export columns from leaking through)
    df = df[expected_cols].copy()

    return df


@st.cache_data(show_spinner=False)
def load_odds(season: int) -> pd.DataFrame:
    if not ODDS_AND_RESULTS_XLSX.exists():
        return pd.DataFrame()

    df = pd.read_excel(ODDS_AND_RESULTS_XLSX)

    for c in ["year", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "year" in df.columns:
        df = df[df["year"] == int(season)].copy()

    if "win_prob_est" not in df.columns:
        df["win_prob_est"] = np.nan

    return df


@st.cache_data(show_spinner=False)
def load_odds_results_full(season: int) -> pd.DataFrame:
    if not ODDS_AND_RESULTS_XLSX.exists():
        return pd.DataFrame()

    df = pd.read_excel(ODDS_AND_RESULTS_XLSX)
    for c in ["year", "event_id", "dg_id", "finish_num", "Winnings"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "year" in df.columns:
        df = df[df["year"] == int(season)].copy()
    return df


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

# --- Shortlist source of truth ---

@st.cache_data(show_spinner=False)
def load_shortlist_df_2026() -> pd.DataFrame:
    p = SHORTLIST_PATH
    if not p.exists():
        return pd.DataFrame(columns=["dg_id"])

    df = pd.read_csv(p, low_memory=False)

    if "dg_id" not in df.columns:
        return pd.DataFrame(columns=["dg_id"])

    df = df.copy()
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df.dropna(subset=["dg_id"]).copy()
    df["dg_id"] = df["dg_id"].astype(int)

    tag_cols = [c for c in df.columns if c.lower().startswith("tag_event")]
    for c in tag_cols:
        df[c] = (
            df[c].astype(str).str.strip().str.lower()
              .replace({"nan": "", "none": ""})
        )

    return df


def shortlist_ids_all_2026() -> set[int]:
    df = load_shortlist_df_2026()
    if df.empty:
        return set()
    return set(df["dg_id"].astype(int).tolist())


def shortlist_ids_for_event_2026(event_id: int, event_name: str) -> set[int]:
    df = load_shortlist_df_2026()
    if df.empty:
        return set()

    tag_cols = [c for c in df.columns if c.lower().startswith("tag_event")]
    if not tag_cols:
        return set()

    eid = str(int(event_id)).strip().lower()
    en = str(event_name or "").strip().lower()

    tags = df[tag_cols]
    mask = tags.eq(eid).any(axis=1)

    if en:
        mask = mask | tags.apply(lambda col: col.str.contains(en, na=False)).any(axis=1)

    return set(df.loc[mask, "dg_id"].astype(int).tolist())


def ensure_is_shortlist(summary: pd.DataFrame, event_id: int, event_name: str) -> pd.DataFrame:
    if summary is None or summary.empty or "dg_id" not in summary.columns:
        return summary

    out = summary.copy()
    out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
    out = out.dropna(subset=["dg_id"]).copy()
    out["dg_id"] = out["dg_id"].astype(int)

    sl_all = shortlist_ids_all_2026()
    sl_evt = shortlist_ids_for_event_2026(int(event_id), str(event_name))

    out["is_shortlist"] = out["dg_id"].isin(sl_all)
    out["is_shortlist_event"] = out["dg_id"].isin(sl_evt)
    return out

# ============================================================
# DATE HELPERS
# ============================================================
def get_pre_event_cutoff_date(sched: pd.DataFrame, event_id: int) -> Optional[pd.Timestamp]:
    """
    Returns (event_date/start_date - 1 day) for leakage-free filtering.
    """
    if sched is None or sched.empty:
        return None

    s = sched.copy()
    s["event_id"] = pd.to_numeric(s.get("event_id"), errors="coerce")
    row = s.loc[s["event_id"] == int(event_id)].head(1)
    if row.empty:
        return None

    for c in ["event_date", "start_date"]:
        if c in row.columns:
            dt = pd.to_datetime(row.iloc[0][c], errors="coerce")
            if pd.notna(dt):
                return dt - pd.Timedelta(days=1)

    date_cols = [c for c in row.columns if "date" in c.lower()]
    if date_cols:
        dt = pd.to_datetime(row.iloc[0][date_cols[0]], errors="coerce")
        if pd.notna(dt):
            return dt - pd.Timedelta(days=1)

    return None


def _build_event_end_table(rounds_all: pd.DataFrame) -> pd.DataFrame:
    """
    Field-wide tournament end date:
    year,event_id,event_end = max(round_date) across ALL players in event.
    Fallback: year-12-31 if round_date missing.
    """
    if rounds_all is None or rounds_all.empty:
        return pd.DataFrame(columns=["year", "event_id", "event_end"])

    ra = rounds_all.copy()
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
        ends["event_end"] = pd.to_datetime(
            ends["year"].astype("Int64").astype(str) + "-12-31",
            errors="coerce",
        )

    ends["year"] = pd.to_numeric(ends["year"], errors="coerce").astype("Int64")
    ends["event_id"] = pd.to_numeric(ends["event_id"], errors="coerce").astype("Int64")
    ends = ends.dropna(subset=["year", "event_id"]).copy()

    ends["year"] = ends["year"].astype(int)
    ends["event_id"] = ends["event_id"].astype(int)
    ends["event_end"] = pd.to_datetime(ends["event_end"], errors="coerce")

    return ends[["year", "event_id", "event_end"]]

# ============================================================
# PICKS LOG (KEEP FOR USED-PLAYER FILTERING; NO SIDEBAR UI)
# ============================================================
def log_path(season: int, mode: str) -> Path:
    mode_tag = "live" if mode.lower().startswith("live") else "test"
    PICKS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return PICKS_LOG_DIR / f"picks_{mode_tag}_{int(season)}.csv"


def load_picks_log(season: int, mode: str) -> pd.DataFrame:
    p = log_path(season, mode)
    cols = [
        "season", "event_id", "event_name", "username",
        "dg_id", "player_name", "ts",
        "finish_num", "finish_text", "winnings",
    ]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=cols)

    for c in ["season", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["username"] = df.get("username", "").astype(str)
    return df


def deny_if_already_picked(picks_df: pd.DataFrame, season: int, event_id: int, username: str) -> bool:
    if picks_df is None or picks_df.empty:
        return False
    sub = picks_df[
        (picks_df["season"] == int(season)) &
        (picks_df["event_id"] == int(event_id)) &
        (picks_df["username"] == str(username))
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

    odds_full = load_odds_results_full(season)
    finish_num = np.nan
    finish_text = ""
    winnings = 0.0

    if not odds_full.empty and {"event_id", "dg_id"}.issubset(odds_full.columns):
        mask = (
                (odds_full["year"] == int(season)) &
                (odds_full["event_id"] == int(event_id)) &
                (odds_full["dg_id"] == int(dg_id))
        )
        res = odds_full.loc[mask]
        if not res.empty:
            r0 = res.iloc[0]
            finish_num = r0.get("finish_num", np.nan)
            finish_text = str(r0.get("finish_text", "") or "")
            winnings = float(pd.to_numeric(r0.get("Winnings", 0.0), errors="coerce") or 0.0)

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
        "winnings": winnings,
    }

    out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out_file = log_path(season, mode)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(out_file)


def get_used_dg_ids_from_picks_log(season: int, mode: str, username: str) -> set[int]:
    picks = load_picks_log(season, mode)
    if picks is None or picks.empty:
        return set()
    picks = picks.copy()
    picks["season"] = pd.to_numeric(picks.get("season"), errors="coerce")
    picks["dg_id"] = pd.to_numeric(picks.get("dg_id"), errors="coerce")
    picks["username"] = picks.get("username", "").astype(str)

    picks = picks[(picks["season"] == int(season)) & (picks["username"] == str(username))].copy()
    picks = picks.dropna(subset=["dg_id"]).copy()
    return set(picks["dg_id"].astype(int).tolist())


def apply_used_filter_to_summary(summary: pd.DataFrame, used: set[int]) -> pd.DataFrame:
    if summary is None or summary.empty or not used or "dg_id" not in summary.columns:
        return summary
    out = summary.copy()
    out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
    out = out.dropna(subset=["dg_id"]).copy()
    out["dg_id"] = out["dg_id"].astype(int)
    return out.loc[~out["dg_id"].isin(used)].copy()

# ============================================================
# MISC HELPERS
# ============================================================
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_oad_score(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if "oad_score" in df.columns:
        df["oad_score"] = pd.to_numeric(df["oad_score"], errors="coerce")
        return df

    proxies = ["final_rank_score", "decision_score", "ev_current_adj", "ev_future_max", "win_prob_est"]
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


def style_table(df: pd.DataFrame, gradients=None, formats=None):
    sty = df.style
    if gradients:
        for col, cmap in gradients:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                sty = sty.background_gradient(subset=[col], cmap=cmap)
    if formats:
        use = {k: v for k, v in formats.items() if k in df.columns}
        if use:
            sty = sty.format(use, na_rep="")
    return sty


def coerce_numeric_for_formatting(df: pd.DataFrame, skip: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    skip_set = set(skip or [])
    for c in out.columns:
        if c in skip_set:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_last_n_events_table(
    rounds_all: pd.DataFrame,
    dg_id: int,
    n: int = 25,
    date_max: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    dfp = rounds_all.loc[pd.to_numeric(rounds_all.get("dg_id"), errors="coerce") == int(dg_id)].copy()
    if dfp.empty:
        return pd.DataFrame(columns=["Event", "Finish", "SG Total", "Year", "event_id"])

    ends_all = _build_event_end_table(rounds_all)
    if date_max is not None and pd.notna(date_max):
        ends_all = ends_all.loc[ends_all["event_end"].notna() & (ends_all["event_end"] <= pd.to_datetime(date_max))].copy()

    def _first_non_null_str(s: pd.Series) -> str:
        s2 = s.dropna().astype(str)
        return s2.iloc[0] if len(s2) else ""

    # choose finish label column
    fin_col = "fin_text" if "fin_text" in dfp.columns else ("finish_text" if "finish_text" in dfp.columns else None)
    if fin_col is None:
        dfp["fin_text"] = ""
        fin_col = "fin_text"

    t = (
        dfp.groupby(["year", "event_id", "event_name"], as_index=False)
           .agg(
               Finish=(fin_col, _first_non_null_str),
               SG_Total=("sg_total", "sum") if "sg_total" in dfp.columns else ("dg_id", "size"),
           )
    )

    out = t.merge(ends_all, on=["year", "event_id"], how="left")
    out = out.sort_values(["event_end", "year", "event_id"], ascending=[False, False, False]).head(int(n)).copy()

    out["SG Total"] = pd.to_numeric(out["SG_Total"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    out = out.rename(columns={"event_name": "Event", "year": "Year"})
    out = out[["Event", "Finish", "SG Total", "Year", "event_id"]]
    return out.reset_index(drop=True)

DG_GREEN = "#2ECC71"   # DataGolf-ish green
DG_BLUE  = "#2F80ED"   # DataGolf-ish blue

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_player_skill_radar(
    player_skills_df: pd.DataFrame,
    dg_id: int,
    title: str = "Skill Profile",
    line_color: str = DG_BLUE,
):
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

    fill_color = _hex_to_rgba(line_color, 0.28)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode="lines+markers",
        line=dict(width=3, color=line_color),
        marker=dict(size=7, color=line_color),
        fill="toself",
        fillcolor=fill_color,
        opacity=1.0,
        name=title,
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".2f",
                gridcolor="rgba(255,255,255,0.12)",
                tickcolor="rgba(255,255,255,0.55)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.12)",
                tickcolor="rgba(255,255,255,0.75)",
            ),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

def pick_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# ============================================================
# WEEKLY SUMMARY
# ============================================================
def get_weekly_summary(season: int, event_id: int) -> pd.DataFrame:
    if BUILD_WEEKLY_VIEW is not None:
        weekly = BUILD_WEEKLY_VIEW(season, int(event_id))
        if isinstance(weekly, dict) and isinstance(weekly.get("summary"), pd.DataFrame):
            return ensure_oad_score(weekly["summary"].copy())

    odds = load_odds(season)
    sub = odds[odds.get("event_id") == int(event_id)].copy() if not odds.empty else pd.DataFrame()
    if sub.empty:
        return pd.DataFrame()

    if "player_name" not in sub.columns:
        sub["player_name"] = sub.get("name", np.nan)

    if "oad_score" not in sub.columns:
        sub["oad_score"] = pd.to_numeric(sub.get("win_prob_est", np.nan), errors="coerce")

    return ensure_oad_score(sub)

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

# ============================================================
# SIDEBAR (NO SEASON PICKER, NO USERNAME PICKER, NO LOGGING UI)
# ============================================================
season = 2026
username = YOU_USERNAME  # fixed username backbone (for used-player filtering + picks log)

with st.sidebar:
    st.header("One and Done 2026")

    mode = st.radio("Mode", ["Live", "Test"], horizontal=True, key="sb_mode")
    test_mode = (mode == "Test")

    st.divider()
    st.caption("Filters")
    hide_shortlist = st.checkbox("Hide shortlist players (this event)", value=False, key="sb_hide_shortlist")

    if test_mode:
        hide_names = st.checkbox("Hide player names (test mode)", value=True, key="sb_hide_names_test")
        show_names = not hide_names
    else:
        hide_names = False
        show_names = True

    top_n = st.slider("Show top N candidates", 10, 60, TOP_N_DEFAULT, key="sb_top_n")

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
        if LOAD_ROUNDS_ERR:
            st.error("data_io.load_rounds import error")
            st.code(LOAD_ROUNDS_ERR)
        if COURSE_TABLES_ERR:
            st.error("course tables import error")
            st.code(COURSE_TABLES_ERR)

    do_log = st.checkbox("Enable pick logging", value=True, key="sb_enable_pick_logging")


# ============================================================
# EVENT SELECTION (STILL ALLOWED)
# ============================================================
sched = load_schedule(season).copy()

label_map_events = {
    int(r["event_id"]): f"{int(r['event_order']):02d} — {r['event_name']}"
    for _, r in sched.iterrows()
    if pd.notna(r.get("event_id")) and pd.notna(r.get("event_order"))
}

event_id = st.sidebar.selectbox(
    "Event",
    options=list(label_map_events.keys()),
    format_func=lambda x: label_map_events.get(int(x), str(x)),
    key="sb_event_id",
)

row_now = sched.loc[sched["event_id"] == int(event_id)].head(1)
wk = int(pd.to_numeric(row_now.iloc[0]["event_order"], errors="coerce")) if not row_now.empty else None
event_name = str(row_now.iloc[0]["event_name"]) if (not row_now.empty and "event_name" in row_now.columns) else label_map_events.get(int(event_id), str(event_id))

st.title(f"One and Done 2026 — Week {wk}" if wk else "One and Done 2026")

# ============================================================
# LOAD WEEKLY DATA ONCE
# ============================================================
summary = get_weekly_summary(season, int(event_id))
weekly = BUILD_WEEKLY_VIEW(season, int(event_id)) if BUILD_WEEKLY_VIEW is not None else None

if summary.empty:
    st.error("No data returned for this week/event. Check odds + schedule + build_weekly_view.")
    st.stop()

if "dg_id" not in summary.columns:
    st.error("Weekly summary missing dg_id; cannot continue.")
    st.stop()

# --- normalize dg_id robustly ---
summary = summary.copy()

summary["dg_id"] = (
    pd.to_numeric(summary.get("dg_id"), errors="coerce")
      .astype("Int64")  # nullable int
)

summary = summary.dropna(subset=["dg_id"]).copy()
summary["dg_id"] = summary["dg_id"].astype(int)

# --- ensure player_name exists ---
if "player_name" not in summary.columns:
    summary["player_name"] = summary["dg_id"].map(lambda x: f"dg_{x}")

# --- label for UI (no risky int(...) apply) ---
summary["player_label"] = np.where(
    show_names,
    summary["player_name"].astype(str),
    summary["dg_id"].astype(str).radd("dg_"),
)

# --- ensure scoring column exists ---
summary = ensure_oad_score(summary)

# ============================================================
# CRITICAL: REMOVE ALREADY-USED PLAYERS (YOU STILL NEED THIS)
# ============================================================
used_ids = get_used_dg_ids_from_picks_log(season=season, mode=mode, username=username)
summary = apply_used_filter_to_summary(summary, used_ids)

# shortlist flags (global + this-event)
# shortlist flags (global + this-event)
summary = ensure_is_shortlist(summary, int(event_id), event_name)

# Hide shortlist players (this event) with sensible fallback:
if hide_shortlist:
    # Prefer event-tagged removal if it exists AND actually matches anyone
    if "is_shortlist_event" in summary.columns and summary["is_shortlist_event"].fillna(False).any():
        summary = summary.loc[~summary["is_shortlist_event"].fillna(False)].copy()
    # Otherwise fall back to global shortlist removal
    elif "is_shortlist" in summary.columns:
        summary = summary.loc[~summary["is_shortlist"].fillna(False)].copy()

# hide only the ones tagged for THIS event
if hide_shortlist and "is_shortlist_event" in summary.columns:
    summary = summary.loc[~summary["is_shortlist_event"]].copy()

rank_col = pick_best_rank_col(summary)
if rank_col not in summary.columns:
    summary["_rank"] = np.arange(len(summary), 0, -1)
    rank_col = "_rank"

summary[rank_col] = pd.to_numeric(summary[rank_col], errors="coerce")
summary = summary.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)

summary_top = summary.head(int(top_n)).copy()

if st.session_state.get("debug"):
    st.caption(f"Used players removed: {len(used_ids)} (from {log_path(season, mode)})")

# ============================================================
# TABS (Tab1 unchanged; old Tab2 becomes Tab3; new Tab2 compare)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Weekly",
    "H2H",
    "Deep Dive",
    "League",
    "Trending",   # NEW
])

with tab1:
    st.header("Weekly decision view")

    sort_candidates = ["oad_score", "final_rank_score", "decision_score", "ev_current_adj", "ev_future_max"]
    sort_options = usable_sort_cols(summary, sort_candidates)
    if not sort_options:
        sort_options = [rank_col]

    sort_col = st.selectbox("Sort by", options=sort_options, index=0)

    # =========================
    # Panel A
    # =========================
    st.subheader("Shortlist / Final Decision")

    cols_A = pick_cols(summary_top, [
        "player_label", "dg_id",
        "oad_score", "decision_score", "final_rank_score",
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
    st.subheader("Stats")

    _ = render_grouped_condensed_tables(
        summary=summary,
        season=season,
        groups=CONDENSED_GROUPS,
        sort_col=sort_col,
        top_n=50,
        show_names=show_names,
        available_only=True,
    )

    st.divider()



    # =========================
    # Panels B/C/D/E
    # =========================
    st.subheader("Form & Skill Right Now")

    cols_B = pick_cols(summary_top, [
        "player_label", "dg_id",
        "sg_total_L12", "sg_total_L24", "sg_total_L40",
        "sg_app_L12", "sg_app_L24", "sg_app_L40",
        "sg_putt_L12", "sg_putt_L24", "sg_putt_L40",
    ])
    panelB = summary_top[cols_B].copy()
    panelB = coerce_numeric_for_formatting(panelB, skip=["player_label"])
    grad_B = []
    for c in ["sg_total_L12", "sg_total_L24", "sg_total_L40", "sg_app_L12", "sg_app_L24", "sg_app_L40", "sg_putt_L12", "sg_putt_L24", "sg_putt_L40"]:
        if c in panelB.columns:
            grad_B.append((c, "RdYlGn"))
    fmt_B = {c: "{:.1f}" for c in panelB.columns if c.startswith("sg_") or c.startswith("sg_total")}
    st.dataframe(style_table(panelB, gradients=grad_B, formats=fmt_B), use_container_width=True, height=360)

    st.subheader("Season Reliability (YTD)")

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

    st.subheader("Event & Course History")

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

    st.subheader("One-and-Done Economics")

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
    # NEW: Per-event shortlist exclusion toggles
    # =========================

    pick_choices_all = summary_top[["dg_id", "player_label"]].dropna().copy()
    pick_choices_all["dg_id"] = pick_choices_all["dg_id"].astype(int)
    label_map_players = dict(zip(pick_choices_all["dg_id"], pick_choices_all["player_label"]))

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
            st.error(
                "Denied: you already logged a pick for this event. Delete the row manually in the picks log file to change it.")
        else:
            append_pick_log(season, mode, int(event_id), event_name, username, pick_dg_id, real_name)
            st.success(f"Logged. Saved to: {log_path(season, mode)}")



# ============================================================
# TAB 2: Compare (CLEAN)
#   - ONE set of selectors
#   - NO "A vs B" headline spam
#   - Uses summary_top (already filtered for used players)
#   - Never assumes player_label exists
# ============================================================
with tab2:
    st.header("Head-to-head comparison")

    rounds_all = load_combined_rounds()
    cutoff_dt = get_pre_event_cutoff_date(sched, int(event_id))

    # --- pool: only available players (summary_top should already reflect used-player filtering) ---
    pool = summary_top[["dg_id", "player_name"]].dropna().drop_duplicates().copy()
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    if len(pool) < 2:
        st.info("Need at least two available players to compare (after used-player filtering).")
        st.stop()

    # label shown in selector
    pool["label"] = pool["player_name"] if show_names else pool["dg_id"].astype(str)
    id_to_label = dict(zip(pool["dg_id"], pool["label"]))
    id_to_name = dict(zip(pool["dg_id"], pool["player_name"]))

    # defaults: top 2 in pool (pool already reflects your sort in summary_top)
    opts = pool["dg_id"].tolist()
    default_a = opts[0]
    default_b = opts[1] if len(opts) > 1 else opts[0]

    colA, colB = st.columns(2, gap="large")
    with colA:
        dg_a = st.selectbox(
            "Player A",
            options=opts,
            index=0,
            format_func=lambda x: id_to_label.get(int(x), str(x)),
            key="cmp_a",
        )
    with colB:
        dg_b = st.selectbox(
            "Player B",
            options=opts,
            index=1 if len(opts) > 1 else 0,
            format_func=lambda x: id_to_label.get(int(x), str(x)),
            key="cmp_b",
        )

    dg_a = int(dg_a)
    dg_b = int(dg_b)
    if dg_a == dg_b:
        st.warning("Pick two different players.")
        st.stop()

    name_a = (str(id_to_name.get(dg_a)) if show_names else f"dg_{dg_a}")
    name_b = (str(id_to_name.get(dg_b)) if show_names else f"dg_{dg_b}")

    # ---- Recent finishes (last 25 events) side-by-side ----
    st.subheader("Recent finishes (last 25 events)")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        last25_a = build_last_n_events_table(rounds_all, dg_a, n=25, date_max=cutoff_dt)
        st.dataframe(
            last25_a[["Event", "Finish", "SG Total", "Year"]] if not last25_a.empty else last25_a,
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        last25_b = build_last_n_events_table(rounds_all, dg_b, n=25, date_max=cutoff_dt)
        st.dataframe(
            last25_b[["Event", "Finish", "SG Total", "Year"]] if not last25_b.empty else last25_b,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ---- SG per round line chart (raw sg_total), last 40 rounds pre-event ----
    st.subheader("Strokes gained per round (last 40 rounds pre-event)")

    if "sg_total" not in rounds_all.columns:
        st.warning("combined rounds is missing sg_total; cannot plot.")
    else:
        ends = _build_event_end_table(rounds_all)

        def _last_n_rounds_pre_event(dg_id: int, n: int = 40) -> pd.DataFrame:
            df = rounds_all.loc[pd.to_numeric(rounds_all.get("dg_id"), errors="coerce") == int(dg_id)].copy()
            if df.empty:
                return df

            df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
            df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")
            df = df.dropna(subset=["year", "event_id"]).copy()
            df["year"] = df["year"].astype(int)
            df["event_id"] = df["event_id"].astype(int)

            df = df.merge(ends, on=["year", "event_id"], how="left")
            df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")

            if cutoff_dt is not None and pd.notna(cutoff_dt):
                df = df.loc[df["event_end"].notna() & (df["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

            if "round_date" in df.columns and df["round_date"].notna().any():
                df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
                df = df.sort_values(["round_date", "event_id", "round_num"], ascending=True)
            else:
                if "round_num" in df.columns:
                    df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce")
                df = df.sort_values(["year", "event_id", "round_num"], ascending=True)

            return df.tail(int(n)).reset_index(drop=True)

        ra = _last_n_rounds_pre_event(dg_a, 40)
        rb = _last_n_rounds_pre_event(dg_b, 40)

        if ra.empty or rb.empty:
            st.info("Not enough round data to plot both players.")
        else:
            ra["sg"] = pd.to_numeric(ra["sg_total"], errors="coerce")
            rb["sg"] = pd.to_numeric(rb["sg_total"], errors="coerce")
            ra = ra.dropna(subset=["sg"]).copy()
            rb = rb.dropna(subset=["sg"]).copy()

            ra["round_index"] = range(1, len(ra) + 1)
            rb["round_index"] = range(1, len(rb) + 1)
            ra["player"] = name_a
            rb["player"] = name_b

            plot_df = pd.concat(
                [ra[["round_index", "sg", "player"]], rb[["round_index", "sg", "player"]]],
                ignore_index=True,
            )

            fig = px.line(plot_df, x="round_index", y="sg", color="player", markers=True)
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            fig.update_yaxes(zeroline=True)
            st.plotly_chart(fig, use_container_width=True,
                            key=f"radar_compare_{mode}_{season}_{event_id}_{dg_a}_{dg_b}")

    st.divider()

    # ---- 2017–2026 performance: starts, wins, avg SG ----
    st.subheader("Performance by season (2017–2026, pre-event)")

    def _perf_by_season(dg_id: int) -> pd.DataFrame:
        df = rounds_all.loc[pd.to_numeric(rounds_all.get("dg_id"), errors="coerce") == int(dg_id)].copy()
        if df.empty:
            return pd.DataFrame(columns=["year", "starts", "wins", "avg_sg_per_round"])

        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
        df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")
        df = df.dropna(subset=["year", "event_id"]).copy()
        df["year"] = df["year"].astype(int)
        df["event_id"] = df["event_id"].astype(int)

        ends_tbl = _build_event_end_table(rounds_all)
        df = df.merge(ends_tbl, on=["year", "event_id"], how="left")
        df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")

        if cutoff_dt is not None and pd.notna(cutoff_dt):
            df = df.loc[df["event_end"].notna() & (df["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

        df = df[(df["year"] >= 2017) & (df["year"] <= 2026)].copy()

        starts = df.groupby("year")["event_id"].nunique().rename("starts").reset_index()

        # wins: compute from finish_num if available, else 0
        wins = starts[["year"]].copy()
        wins["wins"] = 0
        if "finish_num" in df.columns:
            tmp = df.copy()
            tmp["finish_num_n"] = pd.to_numeric(tmp["finish_num"], errors="coerce")

            ev_best = (
                tmp.groupby(["year", "event_id"], as_index=False)["finish_num_n"]
                .min()
            )

            w = (
                ev_best.loc[ev_best["finish_num_n"] == 1]
                .groupby("year")["event_id"]
                .nunique()
                .rename("wins")
                .reset_index()
            )

            wins = wins.drop(columns=["wins"]).merge(w, on="year", how="left")
            wins["wins"] = wins["wins"].fillna(0).astype(int)

        df["sg_total_n"] = pd.to_numeric(df.get("sg_total"), errors="coerce")
        sg = df.groupby("year")["sg_total_n"].mean().rename("avg_sg_per_round").reset_index()

        out = (
            starts.merge(wins[["year", "wins"]], on="year", how="left")
            .merge(sg, on="year", how="left")
            .sort_values("year", ascending=False)
            .reset_index(drop=True)
        )
        out["wins"] = out["wins"].fillna(0).astype(int)
        return out

    left, right = st.columns(2, gap="large")
    with left:
        st.dataframe(_perf_by_season(dg_a), use_container_width=True, hide_index=True)
    with right:
        st.dataframe(_perf_by_season(dg_b), use_container_width=True, hide_index=True)

    st.divider()

    # ---- radar ----
    st.subheader("Skill profiles (radar)")
    skills_df = load_player_skills_df(season)
    r1, r2 = st.columns(2, gap="large")
    with r1:
        plot_player_skill_radar(skills_df, dg_a, title=name_a, line_color=DG_GREEN)
    with r2:
        plot_player_skill_radar(skills_df, dg_b, title=name_b, line_color=DG_BLUE)

    st.divider()

    # ---- stat comparison table ----
    st.subheader("Stat comparison (last 40 rounds pre-event)")

    stats = [
        "sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt",
        "driving_dist", "driving_acc",
        "gir", "scrambling", "prox_rgh", "prox_fw",
        "great_shots", "poor_shots",
        "birdies", "bogies", "doubles_or_worse",
        "round_score",
    ]
    stats = [c for c in stats if c in rounds_all.columns]
    lower_better = {"round_score", "bogies", "doubles_or_worse", "poor_shots", "prox_rgh", "prox_fw"}

    ends_tbl = _build_event_end_table(rounds_all)

    def _last_n_rounds_pre_event_for_means(dg_id: int, n: int = 40) -> pd.DataFrame:
        df = rounds_all.loc[pd.to_numeric(rounds_all.get("dg_id"), errors="coerce") == int(dg_id)].copy()
        if df.empty:
            return df

        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
        df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")
        df = df.dropna(subset=["year", "event_id"]).copy()
        df["year"] = df["year"].astype(int)
        df["event_id"] = df["event_id"].astype(int)

        df = df.merge(ends_tbl, on=["year", "event_id"], how="left")
        df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")

        if cutoff_dt is not None and pd.notna(cutoff_dt):
            df = df.loc[df["event_end"].notna() & (df["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

        if "round_date" in df.columns and df["round_date"].notna().any():
            df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
            df = df.sort_values(["round_date", "event_id", "round_num"], ascending=True)
        else:
            if "round_num" in df.columns:
                df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce")
            df = df.sort_values(["year", "event_id", "round_num"], ascending=True)

        return df.tail(int(n)).reset_index(drop=True)

    def _means(df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in stats:
            out[c] = float(pd.to_numeric(df.get(c), errors="coerce").mean()) if df is not None and not df.empty else np.nan
        return out

    ra40 = _last_n_rounds_pre_event_for_means(dg_a, 40)
    rb40 = _last_n_rounds_pre_event_for_means(dg_b, 40)
    ma = _means(ra40)
    mb = _means(rb40)

    rows = []
    for c in stats:
        va, vb = ma.get(c, np.nan), mb.get(c, np.nan)
        win = ""
        if np.isfinite(va) and np.isfinite(vb) and va != vb:
            if c in lower_better:
                win = "◀" if va < vb else "▶"
            else:
                win = "◀" if va > vb else "▶"
        rows.append([c, va, win, vb])

    comp = pd.DataFrame(rows, columns=["stat", name_a, "winner", name_b])
    st.dataframe(comp, use_container_width=True, hide_index=True)

# =========================
# TAB 2: PLAYER PROFILE  (clean rewrite)
# =========================
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
# TAB 3 UI
# =========================
with tab3:
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

        # LOAD skills table for radar
        player_skills_df = load_player_skills_df(season)

        try:
            plot_player_skill_radar(
                player_skills_df,
                dg_id_sel,
                title=f"{player_name_sel} — Skill Profile"
            )
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
            rounds_all=rounds_all,  # combined_rounds_all_2017_2026.csv
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

# ============================================================
# TAB 4: League (clean)
# ============================================================
with tab4:

    league_all = load_league(season)
    sched2 = load_schedule(season)

    # -------------------------
    # validation
    # -------------------------
    ok = True
    need_league_cols = {"league_id", "entry_id", "username", "event_id", "raw_winnings"}
    missing_league = need_league_cols - set(league_all.columns)
    if missing_league:
        st.error(f"League file missing required columns: {sorted(missing_league)}")
        ok = False
    if "event_id" not in sched2.columns:
        st.error("Schedule is missing required column: event_id")
        ok = False
    if not (("event_order" in sched2.columns) or ("week_num" in sched2.columns) or ("start_date" in sched2.columns)):
        st.error("Schedule needs one of: event_order, week_num, start_date")
        ok = False

    if not ok:
        st.info("Fix inputs above to enable League visuals.")
    else:
        # -------------------------
        # single league only (no dropdown)
        # -------------------------
        league_ids = sorted(league_all["league_id"].dropna().astype(str).unique().tolist())
        selected_league_id = league_ids[0] if league_ids else "main"
        league = league_all[league_all["league_id"].astype(str) == str(selected_league_id)].copy()

        # -------------------------
        # pick which league username is "you"
        # -------------------------
        usernames_in_league = sorted(league["username"].dropna().astype(str).unique().tolist())
        default_idx = usernames_in_league.index("You") if "You" in usernames_in_league else 0

        tracked_username = st.selectbox(
            "Your league username",
            usernames_in_league,
            index=default_idx,
            key="tab4_tracked_username",
        )

        # -------------------------
        # cutoff week logic
        # -------------------------
        raw_override = False  # your league file is per-event payout

        cur_event_id_int = int(event_id)

        # schedule cleanup: we need event_id + event_order
        # (if your schedule uses week_num, convert it here)
        sched_clean = sched2.copy()
        sched_clean["event_id"] = pd.to_numeric(sched_clean.get("event_id"), errors="coerce")

        if "event_order" in sched_clean.columns:
            sched_clean["event_order"] = pd.to_numeric(sched_clean.get("event_order"), errors="coerce")
        elif "week_num" in sched_clean.columns:
            sched_clean["event_order"] = pd.to_numeric(sched_clean.get("week_num"), errors="coerce")
        else:
            # fallback: derive event_order from start_date
            sd = pd.to_datetime(sched_clean.get("start_date"), errors="coerce")
            sched_clean["event_order"] = sd.rank(method="dense").astype("Int64")

        sched_clean = sched_clean.dropna(subset=["event_id", "event_order"]).copy()
        sched_clean["event_id"] = sched_clean["event_id"].astype(int)
        sched_clean["event_order"] = sched_clean["event_order"].astype(int)

        max_week = int(sched_clean["event_order"].max())

        cur_order_series = sched_clean.loc[sched_clean["event_id"] == cur_event_id_int, "event_order"]
        if len(cur_order_series) == 0:
            auto_week = 1
            st.warning(f"Current event_id={cur_event_id_int} not found in schedule; defaulting cutoff week=1.")
        else:
            auto_week = int(cur_order_series.iloc[0]) - 1
            auto_week = max(0, min(max_week, auto_week))

        end_season = st.checkbox("End of season", value=False, key="tab4_end_season")
        cut_week = max_week if end_season else auto_week
        cut_week = max(1, int(cut_week))

        # -------------------------
        # build standings through prior event
        # -------------------------
        standings_all, cutoff, raw_used = build_league_standings_through_prior(
            league_df=league,
            schedule_df=sched2,
            current_event_id=cur_event_id_int,
            username_you=tracked_username,   # ensures your entries are retained
            top_n=10000,
            raw_is_cumulative=raw_override,
        )

        if standings_all is None or standings_all.empty:
            st.info("No league rows found (after schedule mapping).")
        else:
            snap_all = standings_all.copy()
            snap_all["event_order"] = pd.to_numeric(snap_all.get("event_order"), errors="coerce")
            snap_all["cum_winnings"] = pd.to_numeric(snap_all.get("cum_winnings"), errors="coerce").fillna(0.0)
            snap_all = snap_all.dropna(subset=["label", "event_order"]).copy()
            snap_all["event_order"] = snap_all["event_order"].astype(int)

            def latest_snapshot_upto(df: pd.DataFrame, week: int) -> pd.DataFrame:
                d = df[df["event_order"] <= int(week)].copy()
                if d.empty:
                    return d
                d = d.sort_values(["label", "event_order"])
                return d.groupby("label", as_index=False).tail(1)[["label", "username", "cum_winnings", "event_order"]]

            league_totals_all = latest_snapshot_upto(snap_all, cut_week)
            n_entries = int(len(league_totals_all))

            paid_cut = None
            if n_entries >= 7:
                tmp = league_totals_all.sort_values("cum_winnings", ascending=False).reset_index(drop=True)
                paid_cut = float(tmp.loc[6, "cum_winnings"])

            # -------------------------
            # ONE plot selector (unique key)
            # -------------------------
            plot_mode = st.selectbox(
                "Lines to plot",
                ["Top 10 + Me", "Top 25 + Me", "All players (slow)"],
                index=0,
                key="tab4_plot_mode_v2",
            )

            if plot_mode.startswith("All"):
                labels_to_plot = set(league_totals_all["label"].tolist())
            else:
                k = 10 if plot_mode.startswith("Top 10") else 25
                topk_labels = (
                    league_totals_all.sort_values("cum_winnings", ascending=False)
                    .head(k)["label"].tolist()
                )
                labels_to_plot = set(topk_labels)

            # always include your username (all entries under that username)
            your_labels = set(
                snap_all.loc[snap_all["username"].astype(str) == str(tracked_username), "label"]
                .dropna()
                .tolist()
            )
            labels_to_plot |= your_labels

            standings_plot = snap_all[
                (snap_all["event_order"] <= cut_week) & (snap_all["label"].isin(labels_to_plot))
            ].copy()
            standings_plot = standings_plot.sort_values(["label", "event_order"]).reset_index(drop=True)

            col_left, col_right = st.columns([2, 1], gap="large")

            with col_left:
                fig = px.line(
                    standings_plot,
                    x="event_order",
                    y="cum_winnings",
                    color="label",
                    title=f"{season} cumulative winnings — through week {cut_week}",
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
                st.markdown("**Standings at cutoff**")
                cur_full = latest_snapshot_upto(snap_all, cut_week).copy()
                cur_full = cur_full.sort_values("cum_winnings", ascending=False).reset_index(drop=True)
                cur_full["rank"] = np.arange(1, len(cur_full) + 1, dtype=int)

                rows_to_show = st.selectbox(
                    "Rows to show",
                    options=[10, 20, 30, 49, 100],
                    index=0,
                    key="tab4_rows",
                )
                show = cur_full.head(int(rows_to_show)).copy()

                # ensure your username appears even if outside the top N rows
                you_rows = cur_full[cur_full["username"].astype(str) == str(tracked_username)].copy()
                if not you_rows.empty:
                    show = (
                        pd.concat([show, you_rows], ignore_index=True)
                        .drop_duplicates(subset=["label"], keep="first")
                    )

                show["Total ($)"] = show["cum_winnings"].map(lambda x: f"${float(x):,.0f}")
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
                show = show.rename(columns={"label": "User"})[["User", "Paid", "Total ($)"]]
                st.dataframe(show, use_container_width=True, hide_index=True)
    # =========================
    # BELOW: Your position summary
    # =========================
    st.divider()

    # =========================
    # BELOW: Your position summary (from LEAGUE standings, not rounds)
    # =========================

    # cur_full already exists above as the standings-at-cutoff table.
    # If you want it isolated/explicit, rebuild it from snap_all:
    cur_full2 = latest_snapshot_upto(snap_all, cut_week).copy()

    if cur_full2 is None or cur_full2.empty:
        st.info("No standings available at this cutoff.")
    else:
        # ensure numeric
        cur_full2["cum_winnings"] = pd.to_numeric(cur_full2["cum_winnings"], errors="coerce").fillna(0.0)
        cur_full2 = cur_full2.sort_values("cum_winnings", ascending=False).reset_index(drop=True)
        cur_full2["rank"] = np.arange(1, len(cur_full2) + 1, dtype=int)

        # rows for your username
        you_rows2 = cur_full2[cur_full2["username"].astype(str) == str(tracked_username)].copy()

        if you_rows2.empty:
            st.info(f"No standings row found for username '{tracked_username}' at cutoff week {cut_week}.")
        else:
            # if you have multiple entries under the same username, use their best entry
            you_best = you_rows2.sort_values("cum_winnings", ascending=False).head(1)
            you_label_best = str(you_best["label"].iloc[0])
            you_total = float(you_best["cum_winnings"].iloc[0])

            # rank (label should be unique per entry)
            you_rank = int(cur_full2.loc[cur_full2["label"] == you_label_best, "rank"].iloc[0])

            leader_total = float(cur_full2.loc[0, "cum_winnings"]) if len(cur_full2) else 0.0

            money_line = None
            behind_money = None
            if len(cur_full2) >= 7:
                money_line = float(cur_full2.loc[6, "cum_winnings"])
                behind_money = max(0.0, money_line - you_total)

            behind_leader = max(0.0, leader_total - you_total)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Your place", f"{you_rank}")
            c2.metric("Your total", f"${you_total:,.0f}")
            c3.metric("Behind leader", f"${behind_leader:,.0f}")
            c4.metric("Behind the money (Top 7)", f"${behind_money:,.0f}" if money_line is not None else "N/A")

        if money_line is not None:
            c4.metric("Behind the money (Top 7)", f"${behind_money:,.0f}")
        else:
            c4.metric("Behind the money (Top 7)", "N/A")

    # =========================
    # BELOW: Selection share table (through cutoff week)
    # =========================
    st.subheader("Selection share (through cutoff)")

    if "selection" not in league.columns:
        st.info(
            "Column 'selection' not found in league data. Add it to load_league() expected_cols to enable this table.")
    else:
        # Map league rows to event_order so we can filter “thus far”
        order_map = build_event_order_map(sched2)

        sel = league.copy()
        sel["event_id"] = pd.to_numeric(sel.get("event_id"), errors="coerce")
        sel = sel.dropna(subset=["event_id"]).copy()
        sel["event_id"] = sel["event_id"].astype(int)
        sel["event_order"] = sel["event_id"].map(order_map)

        # through cutoff week (same week used by chart)
        sel = sel.dropna(subset=["event_order"]).copy()
        sel["event_order"] = sel["event_order"].astype(int)
        sel = sel[sel["event_order"] <= int(cut_week)].copy()

        # normalize selection strings
        sel["selection"] = sel["selection"].astype(str).str.strip()
        sel = sel[sel["selection"].ne("") & sel["selection"].ne("None") & sel["selection"].ne("nan")].copy()

        total_entries = int(sel["entry_id"].nunique()) if "entry_id" in sel.columns else 0
        total_picks = int(len(sel))

        if total_entries == 0 or total_picks == 0:
            st.info("No selections found to summarize yet (through cutoff).")
        else:
            # % of entries that have used the player at least once so far
            entries_used = (
                sel.groupby("selection")["entry_id"]
                .nunique()
                .reset_index(name="entries_used")
            )
            entries_used["pct_entries_used"] = entries_used["entries_used"] / total_entries

            # also: share of all picks (if you want it)
            picks_used = (
                sel.groupby("selection")
                .size()
                .reset_index(name="picks")
            )
            picks_used["pct_picks"] = picks_used["picks"] / total_picks

            share = entries_used.merge(picks_used, on="selection", how="outer").fillna(0)
            share = share.sort_values(["pct_entries_used", "pct_picks", "selection"], ascending=[False, False, True])

            share["% of entries"] = (share["pct_entries_used"] * 100).map(lambda x: f"{x:.1f}%")
            share["% of picks"] = (share["pct_picks"] * 100).map(lambda x: f"{x:.1f}%")

            out = share.rename(columns={"selection": "Player", "entries_used": "Entries", "picks": "Picks"})[
                ["Player", "Entries", "% of entries", "Picks", "% of picks"]
            ].copy()

            st.dataframe(out, use_container_width=True, hide_index=True)



with tab5:
    st.header("Baseline vs current form")

    rounds_all = _load_rounds_all()
    today = pd.Timestamp.today().normalize()

    # Universe: anyone who played in a 2025 PGA Tour field
    pga_2025_ids = (
        rounds_all[
            (rounds_all["tour"].astype(str) == "PGA")
            & (pd.to_numeric(rounds_all["year"], errors="coerce") == 2025)
            & (rounds_all["dg_id"].notna())
        ]["dg_id"]
        .astype(int)
        .unique()
        .tolist()
    )

    # Restrict to PGA rounds before today for those players
    r = rounds_all[
        (rounds_all["tour"].astype(str) == "PGA")
        & (rounds_all["round_date"] < today)
        & (rounds_all["dg_id"].isin(pga_2025_ids))
    ].copy()

    # Minimum rounds overall: 50
    counts = r.groupby("dg_id").size()
    keep_ids = counts[counts >= 50].index.astype(int)
    r = r[r["dg_id"].isin(keep_ids)].copy()

    # birdies + eagles_or_better per round
    r["be_rate"] = (
        pd.to_numeric(r["birdies"], errors="coerce").fillna(0)
        + pd.to_numeric(r["eagles_or_better"], errors="coerce").fillna(0)
    )

    stats_sg = ["sg_putt","sg_arg","sg_app","sg_ott","sg_t2g","sg_total"]
    score_stat = "round_score"
    cols = stats_sg + ["be_rate", score_stat]

    def _last_n_means(g: pd.DataFrame, n: int) -> pd.Series:
        gg = g.sort_values("round_date").tail(n)
        out = {"n": len(gg)}
        for c in cols:
            out[c] = pd.to_numeric(gg[c], errors="coerce").mean()
        return pd.Series(out)

    def _trail_days(g: pd.DataFrame, days: int) -> pd.Series:
        start = today - pd.Timedelta(days=days)
        gg = g[(g["round_date"] >= start) & (g["round_date"] < today)].copy()
        out = {"n": len(gg)}
        for c in cols:
            s = pd.to_numeric(gg[c], errors="coerce")
            out[c+"_mean"] = s.mean()
            out[c+"_std"]  = s.std(ddof=0)
        return pd.Series(out)

    # Current windows
    cur12 = r.groupby("dg_id", group_keys=False).apply(lambda g: _last_n_means(g, 12)).add_prefix("L12_")
    cur24 = r.groupby("dg_id", group_keys=False).apply(lambda g: _last_n_means(g, 24)).add_prefix("L24_")
    cur40 = r.groupby("dg_id", group_keys=False).apply(lambda g: _last_n_means(g, 40)).add_prefix("L40_")

    # Baselines (1y, 2y)
    b1y = r.groupby("dg_id", group_keys=False).apply(lambda g: _trail_days(g, 365)).add_prefix("B1Y_")
    b2y = r.groupby("dg_id", group_keys=False).apply(lambda g: _trail_days(g, 730)).add_prefix("B2Y_")

    # Names
    names = (
        r.sort_values("round_date")
         .groupby("dg_id")["player_name"]
         .last()
         .to_frame()
    )

    out = names.join([cur12, cur24, cur40, b1y, b2y], how="inner").reset_index()

    # Min 12 rounds in the current window (your rule)
    out = out[out["L12_n"] >= 12].copy()

    # Changes: SG + BE higher is better, Score lower is better
    def _add_change(df, cur_prefix, base_prefix, label):
        for c in stats_sg + ["be_rate"]:
            df[f"{label}_{c}_delta"] = df[f"{cur_prefix}{c}"] - df[f"{base_prefix}{c}_mean"]
            denom = df[f"{base_prefix}{c}_std"].replace(0, np.nan)
            df[f"{label}_{c}_zchg"] = df[f"{label}_{c}_delta"] / denom

        # score: improvement = baseline - current (positive is better)
        c = score_stat
        df[f"{label}_{c}_improve"] = df[f"{base_prefix}{c}_mean"] - df[f"{cur_prefix}{c}"]
        denom = df[f"{base_prefix}{c}_std"].replace(0, np.nan)
        df[f"{label}_{c}_zchg"] = df[f"{label}_{c}_improve"] / denom
        return df


    out = _add_change(out, "L12_", "B1Y_", "L12_vs_1Y")
    out = _add_change(out, "L24_", "B1Y_", "L24_vs_1Y")
    out = _add_change(out, "L40_", "B1Y_", "L40_vs_1Y")

    out = _add_change(out, "L12_", "B2Y_", "L12_vs_2Y")
    out = _add_change(out, "L24_", "B2Y_", "L24_vs_2Y")
    out = _add_change(out, "L40_", "B2Y_", "L40_vs_2Y")

    # Optional: join current field tag so you can filter to this week’s field
    # (summary must exist in oad.py already)
    if "summary" in locals() and summary is not None and "dg_id" in summary.columns:
        field_ids = pd.to_numeric(summary["dg_id"], errors="coerce").dropna().astype(int).unique().tolist()
        out["in_field"] = out["dg_id"].astype(int).isin(field_ids)
    else:
        out["in_field"] = False

    # UI controls
    only_field = st.toggle("Only show this week’s field", value=True)
    if only_field:
        out_view = out[out["in_field"]].copy()
    else:
        out_view = out.copy()

    # -------------------------
    # Sort controls (3 dropdowns)
    # -------------------------
    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        win_choice = st.selectbox("Window", ["L12", "L24", "L40"], index=0)

    with colB:
        base_choice = st.selectbox("Baseline", ["1Y", "2Y"], index=1)

    metric_menu = [
        ("SG Total (z-change)", "sg_total_zchg"),
        ("SG Total (delta)", "sg_total_delta"),
        ("Birdies+Eagles (z-change)", "be_rate_zchg"),
        ("Round score (z-change; + is better)", "round_score_zchg"),
        ("Round score (improvement; + is better)", "round_score_improve"),
    ]

    with colC:
        metric_label = st.selectbox("Metric", [m[0] for m in metric_menu], index=0)

    suffix = dict(metric_menu)[metric_label]
    sort_col = f"{win_choice}_vs_{base_choice}_{suffix}"

    # guard if user picks something you didn't compute (or missing data)
    if sort_col not in out_view.columns:
        fallback_cols = [
            "L12_vs_2Y_sg_total_zchg",
            "L12_vs_1Y_sg_total_zchg",
            "L24_vs_2Y_sg_total_zchg",
            "L40_vs_2Y_sg_total_zchg",
        ]
        sort_col = next((c for c in fallback_cols if c in out_view.columns), out_view.columns[0])

    out_view = out_view.sort_values(sort_col, ascending=False)

    # ------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------
    tmp = out_view[["player_name", "dg_id", sort_col]].copy()
    tmp[sort_col] = pd.to_numeric(tmp[sort_col], errors="coerce")
    tmp = tmp.dropna(subset=[sort_col])

    top_n = st.slider("Show top N movers", 10, 50, 20)

    top = tmp.sort_values(sort_col, ascending=False).head(top_n)
    bot = tmp.sort_values(sort_col, ascending=True).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top improvers")
        st.plotly_chart(
            px.bar(top, x=sort_col, y="player_name", orientation="h"),
            use_container_width=True,
        )
    with c2:
        st.subheader("Biggest declines")
        st.plotly_chart(
            px.bar(bot, x=sort_col, y="player_name", orientation="h"),
            use_container_width=True,
        )

    # Scatter: baseline vs current for SG Total (only if present)
    if "B2Y_sg_total_mean" in out_view.columns and "L12_sg_total" in out_view.columns:
        sc = out_view[["player_name", "B2Y_sg_total_mean", "L12_sg_total"]].copy()
        sc["B2Y_sg_total_mean"] = pd.to_numeric(sc["B2Y_sg_total_mean"], errors="coerce")
        sc["L12_sg_total"] = pd.to_numeric(sc["L12_sg_total"], errors="coerce")
        sc = sc.dropna(subset=["B2Y_sg_total_mean", "L12_sg_total"])

        st.subheader("SG Total: baseline vs current")
        st.plotly_chart(
            px.scatter(sc, x="B2Y_sg_total_mean", y="L12_sg_total", hover_name="player_name"),
            use_container_width=True,
        )


    # ------------------------------------------------------------
    # Display: make it readable + optional "views"
    # ------------------------------------------------------------

    def _pretty_col(c: str) -> str:
        c = c.replace("player_name", "Player")
        c = c.replace("dg_id", "DG ID")
        c = c.replace("in_field", "In field")
        c = c.replace("_n", " (N)")

        c = c.replace("L12", "Last 12")
        c = c.replace("L24", "Last 24")
        c = c.replace("L40", "Last 40")
        c = c.replace("B1Y", "1Y baseline")
        c = c.replace("B2Y", "2Y baseline")

        c = c.replace("sg_total", "SG Total")
        c = c.replace("sg_t2g", "SG T2G")
        c = c.replace("sg_ott", "SG OTT")
        c = c.replace("sg_app", "SG APP")
        c = c.replace("sg_arg", "SG ARG")
        c = c.replace("sg_putt", "SG PUTT")

        c = c.replace("be_rate", "Birdies+Eagles / Rd")
        c = c.replace("round_score", "Round score")

        c = c.replace("_mean", " (mean)")
        c = c.replace("_std", " (std)")
        c = c.replace("_delta", " (Δ)")
        c = c.replace("_improve", " (improvement)")
        c = c.replace("_zchg", " (z-change)")

        c = c.replace("_", " ")
        c = re.sub(r"\s+", " ", c).strip()
        return c


    view = st.radio(
        "View",
        ["Overview", "SG details", "Scoring + Birdies/Eagles", "All columns"],
        horizontal=True,
    )

    base_cols = ["player_name", "dg_id", "in_field", "L12_n"]

    overview_cols = base_cols + [
        "L12_sg_total",
        "B2Y_sg_total_mean",
        "L12_vs_2Y_sg_total_delta",
        "L12_vs_2Y_sg_total_zchg",
        "L12_be_rate",
        "B2Y_be_rate_mean",
        "L12_vs_2Y_be_rate_zchg",
        "L12_round_score",
        "B2Y_round_score_mean",
        "L12_vs_2Y_round_score_improve",
        "L12_vs_2Y_round_score_zchg",
    ]

    sg_cols = base_cols + [c for c in out_view.columns if "sg_" in c]
    score_cols = base_cols + [c for c in out_view.columns if ("be_rate" in c or "round_score" in c)]

    if view == "Overview":
        show_cols = [c for c in overview_cols if c in out_view.columns]
    elif view == "SG details":
        show_cols = [c for c in sg_cols if c in out_view.columns]
    elif view == "Scoring + Birdies/Eagles":
        show_cols = [c for c in score_cols if c in out_view.columns]
    else:
        show_cols = list(out_view.columns)

    display_df = out_view[show_cols].copy()
    display_df = display_df.rename(columns={c: _pretty_col(c) for c in display_df.columns})

    # --- color scales using pandas Styler (requires st.data_editor) ---
    num_cols = display_df.select_dtypes(include=["number"]).columns.tolist()

    # pick “important” columns to color if present (keeps it readable)
    prefer = [
        "Last 12 SG Total",
        "2Y baseline SG Total (mean)",
        "Last 12 vs 2Y SG Total (Δ)",
        "Last 12 vs 2Y SG Total (z-change)",
        "Last 12 Birdies+Eagles / Rd",
        "2Y baseline Birdies+Eagles / Rd (mean)",
        "Last 12 vs 2Y Birdies+Eagles / Rd (z-change)",
        "Last 12 Round score",
        "2Y baseline Round score (mean)",
        "Last 12 vs 2Y Round score (improvement)",
        "Last 12 vs 2Y Round score (z-change)",
    ]
    cols_to_color = [c for c in prefer if c in display_df.columns]
    if not cols_to_color:
        cols_to_color = num_cols  # fallback

    styled = display_df.style.format(precision=3).background_gradient(subset=cols_to_color)

    st.data_editor(
        styled,
        use_container_width=True,
        hide_index=True,
        disabled=True,
    )



