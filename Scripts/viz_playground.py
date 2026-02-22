from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# Config (repo-relative)
# =========================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
DATA_ROOT = REPO_ROOT / "Data"
INUSE_DIR = DATA_ROOT / "in Use"

SCHED_PATH  = INUSE_DIR / "OAD_2026_Schedule.xlsx"
FIELDS_PATH = INUSE_DIR / "Fields.xlsx"
ROUNDS_PATH = INUSE_DIR / "combined_rounds_all_2017_2026.csv"

SEASON_YEAR = 2026

# =========================
# Small helpers
# =========================
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


@st.cache_data(show_spinner=False)
def load_schedule() -> pd.DataFrame:
    df = pd.read_excel(SCHED_PATH)
    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "event_name" in df.columns:
        df["event_name"] = df["event_name"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_fields() -> pd.DataFrame:
    df = pd.read_excel(FIELDS_PATH)
    for c in ["year", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_rounds_minimal() -> pd.DataFrame:
    """
    Load only the columns we need for rolling SG.
    This keeps memory and compute sane.
    """
    usecols = [
        "tour", "dg_id", "event_id", "year",
        "round_date", "event_completed",
        "sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg",
    ]

    # figure out which are actually in the csv header
    with open(ROUNDS_PATH, "r") as f:
        header = f.readline().strip().split(",")
    cols = [c for c in usecols if c in header]

    df = pd.read_csv(ROUNDS_PATH, usecols=cols)

    if "tour" in df.columns:
        df["tour"] = df["tour"].astype(str).str.lower().str.strip()
    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")

    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")

    for c in ["sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg", "year", "event_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _player_rolling_for_windows(g: pd.DataFrame, windows: Sequence[int]) -> pd.Series:
    out = {}
    stats = ["sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg"]
    for w in windows:
        sub = g.head(w)
        for stat in stats:
            out[f"{stat}_L{w}"] = float(sub[stat].mean()) if stat in sub.columns else np.nan
        out[f"n_rounds_L{w}"] = int(sub.shape[0])
    return pd.Series(out)


@st.cache_data(show_spinner=False)
def compute_rolling_stats(
    rounds_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    dg_ids: Iterable[int],
    windows: Sequence[int] = (12,),
) -> pd.DataFrame:
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = rounds_df.copy()

    if "tour" in df.columns:
        df = df[df["tour"] == "pga"].copy()

    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce")
    df = df[df["dg_id"].isin([int(x) for x in dg_ids])].copy()

    if "round_date" in df.columns and df["round_date"].notna().any():
        date_col = "round_date"
    elif "event_completed" in df.columns:
        date_col = "event_completed"
    else:
        raise ValueError("Rounds data must include round_date or event_completed.")

    df = df[df[date_col] < ts].copy()
    if df.empty:
        return pd.DataFrame({"dg_id": list(dg_ids)})

    df = df.sort_values(["dg_id", date_col], ascending=[True, False])

    rolled = (
        df.groupby("dg_id", group_keys=False)
          .apply(_player_rolling_for_windows, windows=windows)
          .reset_index()
    )
    return rolled


def build_field_scatter(
    df: pd.DataFrame,
    *,
    window: int = 12,
    top_ring_n: int = 12,
    ax_cap: float = 3.0,
    course_name: str = "",
    label_top_n: int = 8,
) -> go.Figure:
    req = [f"sg_ott_L{window}", f"sg_app_L{window}", f"sg_arg_L{window}", f"sg_putt_L{window}", "player_name"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")

    d = df.copy()

    d["z_OTT"]  = zscore(d[f"sg_ott_L{window}"])
    d["z_APP"]  = zscore(d[f"sg_app_L{window}"])
    d["z_ARG"]  = zscore(d[f"sg_arg_L{window}"])
    d["z_PUTT"] = zscore(d[f"sg_putt_L{window}"])

    d["ball_striking"] = d["z_OTT"] + d["z_APP"]
    d["short_game"]    = d["z_ARG"] + d["z_PUTT"]

    d = d.dropna(subset=["ball_striking", "short_game"]).copy()

    # "top" should be based on a sensible combined score; otherwise you’re just ringing best ball-strikers
    d["fit_score_simple"] = d["ball_striking"] + d["short_game"]

    d = d.sort_values("fit_score_simple", ascending=False).reset_index(drop=True)
    d["is_top"] = False
    d.loc[: max(top_ring_n - 1, 0), "is_top"] = True

    d["label"] = ""
    d.loc[: max(label_top_n - 1, 0), "label"] = d.loc[: max(label_top_n - 1, 0), "player_name"].astype(str)

    fig = go.Figure()

    base = d[~d["is_top"]]
    fig.add_trace(
        go.Scatter(
            x=base["ball_striking"],
            y=base["short_game"],
            mode="markers",
            marker=dict(size=7, color="rgba(210,210,210,0.28)", line=dict(width=0)),
            customdata=np.stack([base["player_name"], base["fit_score_simple"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{customdata[1]:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    top = d[d["is_top"]]
    fig.add_trace(
        go.Scatter(
            x=top["ball_striking"],
            y=top["short_game"],
            mode="markers+text",
            text=top["label"],
            textposition="top center",
            textfont=dict(size=12, color="rgba(235,235,235,0.95)"),
            marker=dict(
                size=11,
                color="rgba(235,235,235,0.35)",
                line=dict(width=2.5, color="rgba(235,235,235,0.95)"),
            ),
            customdata=np.stack([top["player_name"], top["fit_score_simple"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{customdata[1]:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    # crosshair
    fig.add_shape(type="line", x0=-ax_cap, x1=ax_cap, y0=0, y1=0, line=dict(width=1, color="rgba(255,255,255,0.14)"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-ax_cap, y1=ax_cap, line=dict(width=1, color="rgba(255,255,255,0.14)"))

    fig.update_layout(
        height=650,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"L{window} Field Map — {course_name}", x=0.02, xanchor="left"),
        plot_bgcolor="rgba(10,12,16,1)",
        paper_bgcolor="rgba(10,12,16,1)",
        font=dict(color="rgba(235,235,235,0.9)"),
        showlegend=False,
    )

    fig.update_xaxes(
        title="Ball Striking (z_OTT + z_APP)",
        range=[-ax_cap, ax_cap],
        zeroline=False,
        showline=False,
        gridcolor="rgba(255,255,255,0.06)",
    )

    fig.update_yaxes(
        title="Short Game (z_ARG + z_PUTT)",
        range=[-ax_cap, ax_cap],
        zeroline=False,
        showline=False,
        gridcolor="rgba(255,255,255,0.06)",
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# =========================
# Streamlit app
# =========================
st.set_page_config(layout="wide")
st.title("Visual Playground")

missing = [p for p in [SCHED_PATH, FIELDS_PATH, ROUNDS_PATH] if not p.exists()]
if missing:
    st.error("Missing required file(s):")
    for p in missing:
        st.code(str(p))
    st.stop()

schedule = load_schedule()
fields = load_fields()
rounds = load_rounds_minimal()

# Event selector with names when available
sched = schedule.copy()
sched["event_id"] = pd.to_numeric(sched.get("event_id"), errors="coerce")
sched = sched.dropna(subset=["event_id"]).copy()
sched["event_id"] = sched["event_id"].astype(int)

label = sched.get("event_name", sched["event_id"].astype(str)).astype(str)
label_by_id = dict(zip(sched["event_id"].tolist(), label.tolist()))

event_ids = sched["event_id"].drop_duplicates().tolist()
event_id = st.selectbox("Event", event_ids, format_func=lambda eid: label_by_id.get(int(eid), str(eid)))

# Field for selected event/year
f = fields.copy()
f["event_id"] = pd.to_numeric(f.get("event_id"), errors="coerce")
f["year"] = pd.to_numeric(f.get("year"), errors="coerce")
f["dg_id"] = pd.to_numeric(f.get("dg_id"), errors="coerce")

field_ev = f[(f["event_id"] == int(event_id)) & (f["year"] == int(SEASON_YEAR))].dropna(subset=["dg_id"]).copy()
field_ev["dg_id"] = field_ev["dg_id"].astype(int)
field_ev = field_ev.drop_duplicates(subset=["dg_id"], keep="last")

if field_ev.empty:
    st.warning("No field found for this event/year in Fields.xlsx.")
    st.stop()

field_ids = field_ev["dg_id"].tolist()

with st.sidebar:
    st.subheader("Controls")
    as_of = st.date_input("As-of (pre-event cutoff)", value=pd.Timestamp.today().date())
    window = st.slider("Window", 12, 40, 12, step=1)
    top_n = st.slider("Ring Top N", 5, 25, 12, step=1)
    label_n = st.slider("Label Top N", 0, 20, 8, step=1)
    ax_cap = st.slider("Axis Range", 2.0, 6.0, 3.0, step=0.25)
    min_rounds = st.slider("Min rounds required", 0, 20, 6, step=1)

rolling = compute_rolling_stats(
    rounds_df=rounds,
    as_of_date=pd.Timestamp(as_of),
    dg_ids=field_ids,
    windows=(window,),
)

df = rolling.merge(
    field_ev[["dg_id", "player_name"]].drop_duplicates(subset=["dg_id"]),
    on="dg_id",
    how="left",
)

# basic filter for sample size
ncol = f"n_rounds_L{window}"
if ncol in df.columns:
    df[ncol] = pd.to_numeric(df[ncol], errors="coerce").fillna(0).astype(int)
    df = df[df[ncol] >= int(min_rounds)].copy()

fig = build_field_scatter(
    df,
    window=window,
    top_ring_n=top_n,
    label_top_n=label_n,
    ax_cap=ax_cap,
    course_name=label_by_id.get(int(event_id), f"event_id={event_id}"),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption(f"Players plotted: {len(df)}")
