# Scripts/compare.py
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ------------------------------------------------------------
# column helpers (no guessing on your stat cols; only for date keys)
# ------------------------------------------------------------
_DATE_COL_CANDIDATES = [
    "round_date", "date", "dt", "event_completed", "event_date", "start_date", "end_date"
]
_YEAR_COL_CANDIDATES = ["year", "season", "event_year"]
_NAME_COL_CANDIDATES = ["player_name", "name"]


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _need_cols(df: pd.DataFrame, need: Sequence[str], where: str) -> None:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[compare] {where} missing required columns: {missing}")


def _coerce_int(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _prep_rounds(rounds_all: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns: (clean_df, date_col, year_col)
    """
    if rounds_all is None or rounds_all.empty:
        raise ValueError("[compare] rounds_all is empty")

    df = rounds_all.copy()

    # required backbone columns
    _need_cols(df, ["dg_id", "event_id"], "rounds_all")

    df["dg_id"] = _coerce_int(df["dg_id"])
    df["event_id"] = _coerce_int(df["event_id"])

    date_col = _first_existing(df, _DATE_COL_CANDIDATES)
    if date_col is None:
        raise ValueError(
            "[compare] rounds_all needs a date column to enforce cutoff_dt. "
            f"Add one of: {_DATE_COL_CANDIDATES}"
        )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    year_col = _first_existing(df, _YEAR_COL_CANDIDATES)
    if year_col is None:
        raise ValueError(
            "[compare] rounds_all needs a year/season column to uniquely identify events across years. "
            f"Add one of: {_YEAR_COL_CANDIDATES}"
        )
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    return df, date_col, year_col


def _top2_from_summary(summary: pd.DataFrame, score_col: str, dg_id_col: str) -> List[int]:
    if summary is None or summary.empty:
        raise ValueError("[compare] summary is empty")

    if dg_id_col not in summary.columns:
        raise ValueError(f"[compare] summary missing '{dg_id_col}'")
    if score_col not in summary.columns:
        raise ValueError(f"[compare] summary missing '{score_col}'")

    s = summary.copy()
    s[dg_id_col] = _coerce_int(s[dg_id_col])
    s[score_col] = pd.to_numeric(s[score_col], errors="coerce")

    s = s.dropna(subset=[dg_id_col, score_col]).sort_values(score_col, ascending=False)
    top = s[dg_id_col].astype(int).head(2).tolist()
    if len(top) < 2:
        raise ValueError("[compare] could not determine top 2 players by score_col")
    return top


def _label_for_player(summary: pd.DataFrame, dg_id: int, name_col: Optional[str]) -> str:
    if name_col and name_col in summary.columns:
        row = summary.loc[_coerce_int(summary["dg_id"]) == int(dg_id)]
        if not row.empty:
            v = row.iloc[0].get(name_col)
            if pd.notna(v) and str(v).strip():
                return str(v)
    return f"dg_id {int(dg_id)}"


def _wins_before_cutoff(rounds: pd.DataFrame, dg_id: int, cutoff_dt: pd.Timestamp, date_col: str) -> int:
    df = rounds.loc[(rounds["dg_id"] == int(dg_id)) & (rounds[date_col] <= cutoff_dt)].copy()
    if df.empty:
        return 0
    if "finish_num" not in df.columns:
        raise ValueError("[compare] rounds_all missing 'finish_num' for win counting (finish_num==1)")

    df["finish_num"] = pd.to_numeric(df["finish_num"], errors="coerce")
    # event-level win: any row in that event-year has finish_num==1
    grp = df.groupby(["year", "event_id"], dropna=True)["finish_num"].apply(lambda x: (x == 1).any())
    return int(grp.sum())


def _event_sg_timeseries(
    rounds: pd.DataFrame,
    dg_id: int,
    cutoff_dt: pd.Timestamp,
    date_col: str,
    year_col: str,
    last_n_events: int,
) -> pd.DataFrame:
    """
    Event-level sg_total (sum across rounds inside event) ordered chronologically by event end date (max round date).
    """
    need = ["sg_total", "event_id", "dg_id", year_col, date_col]
    _need_cols(rounds, need, "rounds_all for event_sg_timeseries")

    df = rounds.loc[(rounds["dg_id"] == int(dg_id)) & (rounds[date_col] <= cutoff_dt)].copy()
    if df.empty:
        return pd.DataFrame(columns=["event_dt", "year", "event_id", "sg_total_event"])

    df["sg_total"] = pd.to_numeric(df["sg_total"], errors="coerce")
    # event date proxy = max date seen in that event
    g = (
        df.groupby([year_col, "event_id"], dropna=True)
          .agg(event_dt=(date_col, "max"), sg_total_event=("sg_total", "sum"))
          .reset_index()
          .sort_values("event_dt", ascending=False)
          .head(int(last_n_events))
          .sort_values("event_dt", ascending=True)
          .rename(columns={year_col: "year"})
    )
    return g


def _season_perf(
    rounds: pd.DataFrame,
    dg_id: int,
    cutoff_dt: pd.Timestamp,
    date_col: str,
    year_col: str,
) -> pd.DataFrame:
    """
    Before cutoff_dt: per season -> starts (n events), wins (finish_num==1), sg_total sum
    """
    df = rounds.loc[(rounds["dg_id"] == int(dg_id)) & (rounds[date_col] <= cutoff_dt)].copy()
    if df.empty:
        return pd.DataFrame(columns=["season", "starts", "wins", "sg_total"])

    if "finish_num" not in df.columns:
        raise ValueError("[compare] rounds_all missing 'finish_num' for season wins (finish_num==1)")
    if "sg_total" not in df.columns:
        raise ValueError("[compare] rounds_all missing 'sg_total'")

    df["finish_num"] = pd.to_numeric(df["finish_num"], errors="coerce")
    df["sg_total"] = pd.to_numeric(df["sg_total"], errors="coerce")

    # event-level wins and event-level sg_total
    ev = (
        df.groupby([year_col, "event_id"], dropna=True)
          .agg(
              win=("finish_num", lambda x: (x == 1).any()),
              sg_total_event=("sg_total", "sum"),
          )
          .reset_index()
    )

    out = (
        ev.groupby(year_col, dropna=True)
          .agg(
              starts=("event_id", "nunique"),
              wins=("win", "sum"),
              sg_total=("sg_total_event", "sum"),
          )
          .reset_index()
          .rename(columns={year_col: "season"})
          .sort_values("season", ascending=False)
    )
    out["wins"] = out["wins"].astype(int)
    out["starts"] = out["starts"].astype(int)
    return out


def _compare_stats_table(
    rounds: pd.DataFrame,
    dg_ids: List[int],
    cutoff_dt: pd.Timestamp,
    date_col: str,
    cols: Sequence[str],
) -> pd.DataFrame:
    """
    Comparison “like your image”: show round-level means for selected columns before cutoff_dt.
    (You can change to event-level if you want later; this is stable and non-leaky.)
    """
    missing = [c for c in cols if c not in rounds.columns]
    if missing:
        raise ValueError(f"[compare] rounds_all missing requested compare columns: {missing}")

    df = rounds.loc[(rounds["dg_id"].isin([int(x) for x in dg_ids])) & (rounds[date_col] <= cutoff_dt)].copy()
    if df.empty:
        return pd.DataFrame()

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby("dg_id")[list(cols)]
          .mean(numeric_only=True)
          .reset_index()
    )
    # pretty formatting happens in Streamlit display
    return agg


def render_compare_tab(
    *,
    summary: pd.DataFrame,
    rounds_all: pd.DataFrame,
    cutoff_dt: pd.Timestamp,
    last_n_events_default: int,
    player_skills_df: pd.DataFrame,
    plot_player_skill_radar: Callable[[pd.DataFrame, int, str], None],
    compare_cols: Sequence[str],
    score_col: str = "oad_score",
    dg_id_col: str = "dg_id",
) -> None:
    rounds, date_col, year_col = _prep_rounds(rounds_all)

    # top 2 by score
    top2 = _top2_from_summary(summary, score_col=score_col, dg_id_col=dg_id_col)
    a, b = int(top2[0]), int(top2[1])

    name_col = _first_existing(summary, _NAME_COL_CANDIDATES)
    label_a = _label_for_player(summary, a, name_col)
    label_b = _label_for_player(summary, b, name_col)

    st.subheader("Top 2 by oad_score")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{label_a}**")
        st.caption(f"dg_id = {a}")
    with c2:
        st.markdown(f"**{label_b}**")
        st.caption(f"dg_id = {b}")

    st.divider()

    # wins before cutoff
    wins_a = _wins_before_cutoff(rounds, a, cutoff_dt, date_col)
    wins_b = _wins_before_cutoff(rounds, b, cutoff_dt, date_col)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Wins (pre-event)", wins_a)
    with c2:
        st.metric("Wins (pre-event)", wins_b)

    st.caption(f"Cutoff (leakage-free): {cutoff_dt.date()} (schedule start_date - 1 day)")

    st.divider()

    # last N events sg_total
    last_n = st.slider("Last N events (event-level sg_total)", min_value=5, max_value=80, value=int(last_n_events_default), step=5)

    ts_a = _event_sg_timeseries(rounds, a, cutoff_dt, date_col, year_col, last_n)
    ts_b = _event_sg_timeseries(rounds, b, cutoff_dt, date_col, year_col, last_n)

    if ts_a.empty or ts_b.empty:
        st.warning("Not enough pre-event history to plot last-N event sg_total for one or both players.")
    else:
        plot_df = pd.concat(
            [
                ts_a.assign(player=label_a),
                ts_b.assign(player=label_b),
            ],
            ignore_index=True,
        )
        fig = px.line(
            plot_df,
            x="event_dt",
            y="sg_total_event",
            color="player",
            markers=True,
            title=f"Event-level sg_total over last {last_n} events (no MA)",
        )
        fig.update_layout(xaxis_title="Event (date)", yaxis_title="sg_total (sum across rounds)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # season performance
    st.subheader("Performance by season (pre-event)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**{label_a}**")
        st.dataframe(_season_perf(rounds, a, cutoff_dt, date_col, year_col), use_container_width=True)
    with s2:
        st.markdown(f"**{label_b}**")
        st.dataframe(_season_perf(rounds, b, cutoff_dt, date_col, year_col), use_container_width=True)

    st.divider()

    # skill profiles (two radars)
    st.subheader("Skill profiles")
    r1, r2 = st.columns(2)
    with r1:
        plot_player_skill_radar(player_skills_df, a, title=f"{label_a} — Skill Profile")
    with r2:
        plot_player_skill_radar(player_skills_df, b, title=f"{label_b} — Skill Profile")

    st.divider()

    # comparison stats table
    st.subheader("Stat comparison (pre-event, round-level means)")

    comp = _compare_stats_table(rounds, [a, b], cutoff_dt, date_col, compare_cols)
    if comp.empty:
        st.warning("No rows available for stat comparison before cutoff.")
    else:
        # attach readable labels
        comp["player"] = comp["dg_id"].map({a: label_a, b: label_b})
        comp = comp[["player", "dg_id"] + list(compare_cols)]
        st.dataframe(comp, use_container_width=True)
