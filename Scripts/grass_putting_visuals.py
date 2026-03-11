"""
grass_putting_visuals.py  [v6 — uses course_greens_reference.csv]
─────────────────────────────────────────────────────────────────
SG: Putting on this week's greens grass type.

Grass lookup priority:
  1. course_greens_reference.csv  (full 2025+ coverage)
  2. OAD_2026_Schedule.xlsx       (fallback)

Windows: L12 / L24 / L36 selectable. L60 = baseline.

Primary:  Top 10 + Bottom 10 bar chart (matches Recent Form Trends style).
Secondary (expander): Bump chart L60→L36→L24→L12, top 10 only.

Usage in sg_production_tab.py after Player Strengths Map:

    from grass_putting_visuals import render_bermuda_putting_visuals
    render_bermuda_putting_visuals(
        rounds_df=rounds_df,
        schedule_df=schedule_df,
        field_df=_field_df,
        event_id=event_id,
        cutoff_dt=cutoff,
        greens_ref_path="Data/in Use/course_greens_reference.csv",  # optional
    )
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOWS         = [12, 24, 36, 60]
MIN_ROUNDS      = 4
MOVER_THRESHOLD = 3

SURFACE_ORANGE  = "#f97316"
GREEN_STRONG    = "#00CC96"
RED_STRONG      = "#EF553B"
STABLE_GREY     = "#475569"

TIER_HIGH       = 20
TIER_MED        = 10


# ── Grass lookup ──────────────────────────────────────────────────────────────

def _build_grass_lookup(schedule_df=None, greens_ref_path=None):
    """
    Returns dict: course_num (int) → greens_type (str)
    Merges reference file (priority) over schedule fallback.
    """
    lk = {}

    # 1. OAD schedule as base
    if schedule_df is not None:
        sched = schedule_df.copy()
        sched.columns = [c.strip().lower() for c in sched.columns]
        if "greens_type" in sched.columns and "course_num" in sched.columns:
            sched["course_num"]  = pd.to_numeric(sched["course_num"], errors="coerce")
            sched["greens_type"] = sched["greens_type"].astype(str).str.strip().str.title()
            sched.loc[sched["greens_type"].isin(["Nan","None",""]), "greens_type"] = np.nan
            for _, row in sched.dropna(subset=["course_num","greens_type"]).iterrows():
                lk[int(row["course_num"])] = row["greens_type"]

    # 2. Reference file overrides (higher priority — more complete)
    if greens_ref_path is not None:
        ref_path = Path(greens_ref_path)
        if ref_path.exists():
            ref = pd.read_csv(ref_path, low_memory=False)
            ref.columns = [c.strip().lower() for c in ref.columns]
            ref["course_num"]  = pd.to_numeric(ref["course_num"], errors="coerce")
            ref["greens_type"] = ref["greens_type"].astype(str).str.strip().str.title()
            ref.loc[ref["greens_type"].isin(["Nan","None",""]), "greens_type"] = np.nan
            for _, row in ref.dropna(subset=["course_num","greens_type"]).iterrows():
                lk[int(row["course_num"])] = row["greens_type"]

    return lk


def _get_this_week_greens(schedule_df, event_id, greens_ref_path=None):
    """Return (greens_type, event_name) for the selected event."""
    # try schedule first
    if schedule_df is not None:
        sched = schedule_df.copy()
        sched.columns = [c.strip().lower() for c in sched.columns]
        sched["event_id"] = pd.to_numeric(sched["event_id"], errors="coerce")
        row = sched[sched["event_id"] == float(event_id)]
        if not row.empty:
            r = row.iloc[0]
            gt         = str(r.get("greens_type","")).strip().title()
            event_name = str(r.get("event_name",""))
            course_num = pd.to_numeric(r.get("course_num"), errors="coerce")

            # override from reference if available
            if greens_ref_path and pd.notna(course_num):
                lk = _build_grass_lookup(greens_ref_path=greens_ref_path)
                ref_gt = lk.get(int(course_num))
                if ref_gt:
                    gt = ref_gt

            if gt and gt not in ("Nan","None",""):
                return gt, event_name

    return None, ""


# ── Date col helper ───────────────────────────────────────────────────────────

def _get_date_col(df):
    if "round_date" in df.columns:
        return "round_date"
    if "event_completed" in df.columns:
        return "event_completed"
    raise ValueError("rounds DataFrame needs round_date or event_completed")


# ── Compute stats ─────────────────────────────────────────────────────────────

def _compute_putting_stats(rounds_df, schedule_df, field_df, event_id,
                           cutoff_dt, greens_ref_path):
    greens_type, event_name = _get_this_week_greens(
        schedule_df, event_id, greens_ref_path
    )
    if not greens_type:
        return pd.DataFrame(), None, None

    grass_lk = _build_grass_lookup(schedule_df, greens_ref_path)

    fdf = field_df.copy()
    fdf.columns = [c.strip().lower() for c in fdf.columns]
    fdf["dg_id"] = pd.to_numeric(fdf["dg_id"], errors="coerce")
    fdf = fdf.dropna(subset=["dg_id"])
    fdf["dg_id"] = fdf["dg_id"].astype(int)
    field_ids  = set(fdf["dg_id"].tolist())
    id_to_name = fdf.drop_duplicates("dg_id").set_index("dg_id")["player_name"].to_dict() \
        if "player_name" in fdf.columns else {}

    date_col = _get_date_col(rounds_df)
    df = rounds_df.copy()
    df["dg_id"]      = pd.to_numeric(df["dg_id"],       errors="coerce")
    df["course_num"] = pd.to_numeric(df["course_num"],  errors="coerce")
    df[date_col]     = pd.to_datetime(df[date_col],     errors="coerce")
    df["sg_putt"]    = pd.to_numeric(df.get("sg_putt"), errors="coerce")

    if "tour" in df.columns:
        df = df[df["tour"] == "PGA"]

    cutoff_ts = pd.to_datetime(cutoff_dt) if cutoff_dt is not None else pd.Timestamp.now()
    df = df[df[date_col] < cutoff_ts]
    df = df[df["dg_id"].isin(field_ids)]
    df = df.dropna(subset=["dg_id", "course_num", "sg_putt"]).copy()
    df["dg_id"]      = df["dg_id"].astype(int)
    df["course_num"] = df["course_num"].astype(int)

    df["surface"] = df["course_num"].map(grass_lk)
    df = df[df["surface"] == greens_type].copy()
    df = df.sort_values(["dg_id", date_col], ascending=[True, False])

    records = []
    for dg_id, grp in df.groupby("dg_id"):
        vals = grp["sg_putt"].values
        if len(vals) < MIN_ROUNDS:
            continue
        row = {
            "dg_id":       int(dg_id),
            "player_name": id_to_name.get(int(dg_id), str(int(dg_id))),
            "n_total":     len(vals),
        }
        for w in WINDOWS:
            s = vals[:w]
            row[f"L{w}_avg"] = float(np.mean(s))
            row[f"L{w}_n"]   = len(s)
        records.append(row)

    if not records:
        return pd.DataFrame(), greens_type, event_name

    out = pd.DataFrame(records)
    for w in WINDOWS:
        out[f"rank_L{w}"] = out[f"L{w}_avg"].rank(ascending=False, method="min").astype(int)

    return out, greens_type, event_name


# ── Visual 1: Bar chart ───────────────────────────────────────────────────────

def _render_bar_chart(df, greens, window):
    sel_col = f"L{window}_avg"

    top10 = df.nlargest(10,  sel_col).copy()
    bot10 = df.nsmallest(10, sel_col).copy()

    combined = pd.concat([
        bot10.sort_values(sel_col, ascending=True),   # worst at bottom
        top10.sort_values(sel_col, ascending=True),   # best at top
    ])

    names  = combined["player_name"].tolist()
    vals   = combined[sel_col].tolist()
    n_tots = combined["n_total"].tolist()

    def _opacity(n):
        if n >= TIER_HIGH: return 1.0
        if n >= TIER_MED:  return 0.60
        return 0.32

    def _rgba(hex_c, alpha):
        h = hex_c.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha:.2f})"

    colors = [
        _rgba(GREEN_STRONG if v >= 0 else RED_STRONG, _opacity(n))
        for v, n in zip(vals, n_tots)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=vals,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10),
        customdata=n_tots,
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"L{window} avg: %{{x:+.3f}}<br>"
            "Rounds on surface: %{customdata}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.add_vline(x=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.add_hline(y=9.5,
                  line=dict(color="rgba(255,255,255,0.07)", width=1, dash="dot"))

    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160, r=80, t=44, b=50),
        title=dict(
            text=f"SG: Putt on {greens} Greens — L{window} Average  (top 10 / bottom 10)",
            font=dict(size=13, color="rgba(200,200,200,0.85)"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=f"SG: Putt  (L{window} avg on {greens})",
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=False, tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.1)",
            tickfont=dict(size=10, color="rgba(210,210,210,0.88)"),
            fixedrange=True,
        ),
        bargap=0.28,
    )

    conf_legend = (
        "<div style='display:flex;gap:20px;font-size:11px;margin-bottom:8px;"
        "color:rgba(150,150,150,0.75)'>"
        "<span>Opacity = rounds on this surface:</span>"
        f"<span style='color:{GREEN_STRONG}'>&#9646; High (≥{TIER_HIGH})</span>"
        f"<span style='color:rgba(0,204,150,0.6)'>&#9646; Medium (≥{TIER_MED})</span>"
        f"<span style='color:rgba(0,204,150,0.32)'>&#9646; Low (&lt;{TIER_MED})</span>"
        "</div>"
    )
    st.markdown(conf_legend, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Visual 2: Bump chart ──────────────────────────────────────────────────────

def _render_bump_chart(df, greens):
    sub = df.nsmallest(10, "rank_L12").copy()
    sub["rank_change"] = sub["rank_L60"] - sub["rank_L12"]
    sub["is_mover"]    = sub["rank_change"].abs() >= MOVER_THRESHOLD

    rank_cols = ["rank_L60", "rank_L36", "rank_L24", "rank_L12"]
    xlabels   = ["L60", "L36", "L24", "L12"]

    fig = go.Figure()

    for _, row in sub[~sub["is_mover"]].iterrows():
        fig.add_trace(go.Scatter(
            x=xlabels, y=[row[rc] for rc in rank_cols],
            mode="lines+markers",
            line=dict(color=STABLE_GREY, width=1.5),
            marker=dict(size=5, color=STABLE_GREY),
            showlegend=False, opacity=0.28,
            hovertemplate=(
                f"<b>{row['player_name']}</b><br>"
                + "".join([f"L{w}: {row[f'L{w}_avg']:+.3f} (#{row[f'rank_L{w}']})<br>"
                           for w in [60,36,24,12]])
                + "<extra></extra>"
            ),
        ))
    for _, row in sub[~sub["is_mover"]].iterrows():
        fig.add_annotation(
            x="L12", y=row["rank_L12"],
            text=f"  {row['player_name'].split(',')[0]}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color="rgba(110,110,110,0.55)"),
        )

    for _, row in sub[sub["is_mover"]].iterrows():
        rising = row["rank_change"] > 0
        color  = GREEN_STRONG if rising else RED_STRONG
        arrow  = "▲" if rising else "▼"
        name   = row["player_name"]
        last   = name.split(",")[0]

        fig.add_trace(go.Scatter(
            x=xlabels, y=[row[rc] for rc in rank_cols],
            mode="lines+markers",
            line=dict(color=color, width=3.5),
            marker=dict(size=10, color=color, line=dict(width=1.5, color="rgba(0,0,0,0.4)")),
            showlegend=False,
            hovertemplate=(
                f"<b>{name}</b> {arrow}<br>"
                + "".join([f"L{w}: {row[f'L{w}_avg']:+.3f} (#{row[f'rank_L{w}']})<br>"
                           for w in [60,36,24,12]])
                + f"Rank shift L60→L12: {row['rank_change']:+.0f} spots"
                + "<extra></extra>"
            ),
        ))
        fig.add_annotation(x="L12", y=row["rank_L12"],
            text=f"  <b>{last}</b>", showarrow=False, xanchor="left",
            font=dict(size=11, color=color))
        fig.add_annotation(x="L60", y=row["rank_L60"],
            text=f"<b>{last}</b>  ", showarrow=False, xanchor="right",
            font=dict(size=11, color=color))

    max_rank = int(sub[[f"rank_L{w}" for w in [60,36,24,12]]].max().max()) + 2

    fig.update_layout(
        height=420,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=120, t=40, b=30),
        title=dict(
            text=f"Rank Movement on {greens} Greens — L60→L36→L24→L12  (top 10 by L12)",
            font=dict(size=12, color="rgba(160,160,160,0.65)"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(tickfont=dict(size=13, color="rgba(200,200,200,0.85)"),
                   gridcolor="rgba(255,255,255,0.04)", fixedrange=True),
        yaxis=dict(autorange="reversed", range=[0.5, max_rank],
                   tickfont=dict(size=9, color="rgba(100,100,100,0.28)"),
                   gridcolor="rgba(255,255,255,0.03)",
                   title=dict(text="Rank", font=dict(size=9, color="rgba(100,100,100,0.22)")),
                   fixedrange=True),
    )

    st.markdown(
        f"<div style='display:flex;gap:22px;font-size:11px;margin-bottom:6px;"
        f"color:rgba(140,140,140,0.7)'>"
        f"<span style='color:{GREEN_STRONG}'>▲ Climbing ({MOVER_THRESHOLD}+ spots)</span>"
        f"<span style='color:{RED_STRONG}'>▼ Falling ({MOVER_THRESHOLD}+ spots)</span>"
        f"<span style='color:{STABLE_GREY}'>— Stable</span></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Public entry point ────────────────────────────────────────────────────────

def render_bermuda_putting_visuals(
    rounds_df,
    schedule_df,
    field_df,
    event_id,
    cutoff_dt=None,
    greens_ref_path=None,
):
    df, greens, event_name = _compute_putting_stats(
        rounds_df, schedule_df, field_df, event_id, cutoff_dt, greens_ref_path
    )

    if df is None or df.empty:
        st.caption(f"Not enough putting data on {greens or 'this surface'} to render visuals.")
        return

    st.markdown(
        f"<div style='font-size:16px;font-weight:800;color:rgba(220,220,220,0.9);"
        f"margin-bottom:2px'>SG: Putting on "
        f"<span style='color:{SURFACE_ORANGE}'>{greens}</span> Greens</div>"
        f"<div style='font-size:11px;color:rgba(120,120,120,0.5);margin-bottom:12px'>"
        f"{event_name} · {len(df)} field players with ≥{MIN_ROUNDS} rounds on {greens} · "
        f"windows = most recent N rounds on this surface only · L60 = baseline</div>",
        unsafe_allow_html=True,
    )

    col_sel, _ = st.columns([2, 5])
    with col_sel:
        window = st.radio(
            "Window", options=[12, 24, 36],
            format_func=lambda x: f"L{x}",
            index=2,
            horizontal=True, key="grass_putt_window",
        )

    _render_bar_chart(df, greens, window)

    with st.expander("Rank movement trend  (L60→L36→L24→L12, top 10)", expanded=False):
        _render_bump_chart(df, greens)
