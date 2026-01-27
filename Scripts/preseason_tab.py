from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Helpers: safe loaders
# -------------------------
def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _coerce_int_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _usable_cols(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


# -------------------------
# Default locations (match your Data/in Use convention)
# You can override by passing explicit paths from app.py
# -------------------------
def default_preseason_paths(project_root: Path, season: int) -> dict[str, Path]:
    base = project_root / "Data" / "in Use"
    return {
        "longlist": base / f"preseason_longlist_{season}.csv",
        "shortlist": base / f"preseason_shortlist_{season}.csv",
        # optional if you decide to write it out:
        "rolling": base / f"preseason_rolling_{season}.csv",
        "course_fit": base / f"course_fit_{season}_dg_style_5attr.csv",
        "player_skills": base / f"player_skills_{season}_dg_style_5attr.csv",
    }


@st.cache_data(show_spinner=False)
def load_preseason_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    out = {}
    for k, p in paths.items():
        out[k] = _read_csv_if_exists(p)
    return out


# -------------------------
# Public renderer
# -------------------------
def render_preseason_tab(
    season: int,
    project_root: Path,
    *,
    hide_names: bool = False,
    paths_override: Optional[dict[str, Path]] = None,
) -> None:
    """
    Preseason tab: reads preseason artifacts from Data/in Use and renders them.

    Expected (but flexible) artifacts:
      - preseason_longlist_{season}.csv
      - preseason_shortlist_{season}.csv
      - preseason_rolling_{season}.csv (optional)
      - course_fit_{season}_dg_style_5attr.csv
      - player_skills_{season}_dg_style_5attr.csv
    """
    st.write("DEBUG: render_preseason_tab called")

    st.header("Preseason")

    paths = default_preseason_paths(project_root, season)
    if paths_override:
        paths.update(paths_override)

    tables = load_preseason_tables(paths)

    longlist = tables.get("longlist", pd.DataFrame())
    shortlist = tables.get("shortlist", pd.DataFrame())
    rolling = tables.get("rolling", pd.DataFrame())
    course_fit = tables.get("course_fit", pd.DataFrame())
    player_skills = tables.get("player_skills", pd.DataFrame())

    # -------------------------
    # Status / missing files
    # -------------------------
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        st.info(
            "Some preseason files are missing. That's fine if you haven't generated them yet.\n\n"
            + "\n".join([f"- {k}: {paths[k]}" for k in missing])
        )

    # -------------------------
    # Controls
    # -------------------------
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        view = st.selectbox(
            "View",
            options=["Shortlist", "Longlist", "Rolling (optional)", "Course Fit", "Player Skills"],
            index=0,
        )
    with c2:
        max_rows = st.number_input("Max rows", min_value=10, max_value=5000, value=200, step=10)
    with c3:
        search = st.text_input("Search (name or dg_id)", value="").strip()

    # Utility: apply search + hide names
    def _apply_common(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        out = df.copy()

        # normalize dg_id
        if "dg_id" in out.columns:
            out["dg_id"] = _coerce_int_nullable(out["dg_id"])

        # search
        if search:
            # if digits, treat as dg_id search
            if search.isdigit() and "dg_id" in out.columns:
                out = out[out["dg_id"] == int(search)]
            else:
                # name-ish search if player_name exists
                if "player_name" in out.columns:
                    out = out[out["player_name"].astype(str).str.contains(search, case=False, na=False)]
                elif "name" in out.columns:
                    out = out[out["name"].astype(str).str.contains(search, case=False, na=False)]

        # hide names in test mode
        if hide_names:
            if "player_name" in out.columns:
                out = out.drop(columns=["player_name"])
            if "name" in out.columns:
                out = out.drop(columns=["name"])

        return out

    # -------------------------
    # Render each view
    # -------------------------
    if view == "Shortlist":
        st.subheader("Preseason shortlist")
        df_show = _apply_common(shortlist)

        if df_show is None or df_show.empty:
            st.warning("Shortlist is empty / not found.")
            return

        # put dg_id first if present
        cols = list(df_show.columns)
        if "dg_id" in cols:
            cols = ["dg_id"] + [c for c in cols if c != "dg_id"]
        df_show = df_show[cols]

        st.dataframe(df_show.head(int(max_rows)), use_container_width=True)

    elif view == "Longlist":
        st.subheader("Preseason longlist")
        df_show = _apply_common(longlist)

        if df_show is None or df_show.empty:
            st.warning("Longlist is empty / not found.")
            return

        # common sort candidates (only if present)
        sort_candidates = [
            "sg_total_L40", "sg_total_L24", "sg_total_L12",
            "skill_score", "baseline_score", "rank", "preseason_rank"
        ]
        sort_cols = _usable_cols(df_show, sort_candidates)
        sort_col = st.selectbox("Sort by", options=(sort_cols or list(df_show.columns)), index=0)
        asc = st.checkbox("Ascending", value=False)

        df_show = df_show.sort_values(sort_col, ascending=asc, kind="mergesort")
        st.dataframe(df_show.head(int(max_rows)), use_container_width=True)

    elif view == "Rolling (optional)":
        st.subheader("Rolling stats (optional)")
        df_show = _apply_common(rolling)

        if df_show is None or df_show.empty:
            st.warning("Rolling file is empty / not found.")
            st.caption("If you haven't generated preseason_rolling_{season}.csv yet, this is expected.")
            return

        # quick column grouping if you have L40/L24/L12 fields
        st.dataframe(df_show.head(int(max_rows)), use_container_width=True)

    elif view == "Course Fit":
        st.subheader("Course fit profiles")
        df_show = course_fit.copy()
        if df_show is None or df_show.empty:
            st.warning("Course fit is empty / not found.")
            return

        # search on course
        course_search = st.text_input("Course search", value="").strip()
        if course_search:
            if "course_name" in df_show.columns:
                df_show = df_show[df_show["course_name"].astype(str).str.contains(course_search, case=False, na=False)]
            elif "course_num" in df_show.columns and course_search.isdigit():
                df_show = df_show[pd.to_numeric(df_show["course_num"], errors="coerce") == int(course_search)]

        # typical columns first
        pref = [c for c in ["course_num", "course_name", "imp_dist", "imp_acc", "imp_app", "imp_arg", "imp_putt", "n_obs", "n_events", "n_players"] if c in df_show.columns]
        rest = [c for c in df_show.columns if c not in pref]
        df_show = df_show[pref + rest]

        st.dataframe(df_show.head(int(max_rows)), use_container_width=True)

    elif view == "Player Skills":
        st.subheader("Player skills (5-attr)")
        df_show = _apply_common(player_skills)

        if df_show is None or df_show.empty:
            st.warning("Player skills is empty / not found.")
            return

        pref = [c for c in ["dg_id", "player_name", "n_rounds", "skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt"] if c in df_show.columns]
        rest = [c for c in df_show.columns if c not in pref]
        df_show = df_show[pref + rest]

        sort_candidates = ["n_rounds", "skill_app", "skill_putt", "skill_dist", "skill_acc", "skill_arg"]
        sort_cols = _usable_cols(df_show, sort_candidates)
        sort_col = st.selectbox("Sort by", options=(sort_cols or list(df_show.columns)), index=0)
        asc = st.checkbox("Ascending ", value=False)

        df_show = df_show.sort_values(sort_col, ascending=asc, kind="mergesort")
        st.dataframe(df_show.head(int(max_rows)), use_container_width=True)
