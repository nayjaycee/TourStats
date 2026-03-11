"""
grass_putting_deepdive.py  [v3]
────────────────────────────────
Surface putting profile for Player Deep Dive tab.

Bar chart: player's SG: Putt avg on each surface MINUS their career avg
           across all surfaces. Shows surface specialization clearly.
           Positive = putts better than usual on that surface.
           Negative = putts worse than usual.

Percentile dot: where they rank vs the field on each surface.
                This week's surface highlighted with orange border + star.

Grass lookup: course_greens_reference.csv (priority) → OAD schedule (fallback)

Integration in player_deep_dive_tab.py:
  1. Signature: add schedule_df=None, event_id=None, greens_ref_path=None
  2. Import:    from grass_putting_deepdive import render_surface_putting_deepdive
  3. Call after Section 2:
        if schedule_df is not None:
            st.divider()
            st.subheader("Putting by Surface")
            render_surface_putting_deepdive(
                rounds_df=rounds_df,
                schedule_df=schedule_df,
                event_id=event_id,
                dg_id=dg_id,
                player_name=player_name,
                cutoff_dt=cutoff_dt,
                field_ids=field_ids,
                greens_ref_path=greens_ref_path,
            )
  4. Stats.py: add schedule_df=schedule_df, event_id=event_id,
               greens_ref_path=str(INUSE_DIR / "course_greens_reference.csv")
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOWS       = [12, 24, 36]
MIN_ROUNDS    = 4

COL_PLAYER    = "rgba(255,165,0,1)"
COL_FIELD_TK  = "rgba(200,200,200,0.6)"
COL_FIELD     = "rgba(150,150,150,0.5)"
GREEN_STRONG  = "#00CC96"
RED_STRONG    = "#EF553B"
SURFACE_ORDER = ["Bermuda", "Bentgrass", "Poa Annua", "Fescue", "Ryegrass",
                 "Paspalum", "Zoysia"]


# ── Grass lookup ───────────────────────────────────────────────────────────────

def _build_grass_lookup(schedule_df=None, greens_ref_path=None):
    lk = {}
    if schedule_df is not None:
        s = schedule_df.copy()
        s.columns = [c.strip().lower() for c in s.columns]
        if "greens_type" in s.columns and "course_num" in s.columns:
            s["course_num"]  = pd.to_numeric(s["course_num"], errors="coerce")
            s["greens_type"] = s["greens_type"].astype(str).str.strip().str.title()
            s.loc[s["greens_type"].isin(["Nan","None",""]), "greens_type"] = np.nan
            for _, row in s.dropna(subset=["course_num","greens_type"]).iterrows():
                lk[int(row["course_num"])] = row["greens_type"]
    if greens_ref_path:
        p = Path(greens_ref_path)
        if p.exists():
            ref = pd.read_csv(p, low_memory=False)
            ref.columns = [c.strip().lower() for c in ref.columns]
            ref["course_num"]  = pd.to_numeric(ref["course_num"], errors="coerce")
            ref["greens_type"] = ref["greens_type"].astype(str).str.strip().str.title()
            ref.loc[ref["greens_type"].isin(["Nan","None",""]), "greens_type"] = np.nan
            for _, row in ref.dropna(subset=["course_num","greens_type"]).iterrows():
                lk[int(row["course_num"])] = row["greens_type"]
    return lk


def _this_week_surface(schedule_df, event_id, greens_ref_path=None):
    """Return greens_type string for this week's event."""
    if schedule_df is None or event_id is None:
        return None
    s = schedule_df.copy()
    s.columns = [c.strip().lower() for c in s.columns]
    s["event_id"] = pd.to_numeric(s["event_id"], errors="coerce")
    row = s[s["event_id"] == float(event_id)]
    if row.empty:
        return None
    r          = row.iloc[0]
    course_num = pd.to_numeric(r.get("course_num"), errors="coerce")
    gt         = str(r.get("greens_type","")).strip().title()
    # override from reference
    if greens_ref_path and pd.notna(course_num):
        lk = _build_grass_lookup(greens_ref_path=greens_ref_path)
        gt = lk.get(int(course_num), gt)
    return gt if gt and gt not in ("Nan","None","") else None


def _get_date_col(df):
    for c in ["round_date", "event_completed"]:
        if c in df.columns:
            return c
    raise ValueError("rounds DataFrame needs round_date or event_completed")


# ── Compute stats ──────────────────────────────────────────────────────────────

def _compute_surface_stats(rounds_df, schedule_df, dg_id,
                           field_ids, cutoff_dt, greens_ref_path):
    """
    Returns:
      player_df  — one row per surface with window avgs + career_avg_all_surfaces
      field_df   — one row per (dg_id, surface) for percentile calcs
    """
    grass_lk = _build_grass_lookup(schedule_df, greens_ref_path)
    if not grass_lk:
        return pd.DataFrame(), pd.DataFrame()

    date_col = _get_date_col(rounds_df)
    df = rounds_df.copy()
    for col in ["dg_id", "course_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[date_col]  = pd.to_datetime(df[date_col],    errors="coerce")
    df["sg_putt"] = pd.to_numeric(df.get("sg_putt"), errors="coerce")

    if "tour" in df.columns:
        df = df[df["tour"] == "PGA"]

    cutoff_ts = pd.to_datetime(cutoff_dt) if cutoff_dt is not None else pd.Timestamp.now()
    df = df[df[date_col] < cutoff_ts]
    df = df.dropna(subset=["dg_id", "course_num", "sg_putt"]).copy()
    df["dg_id"]      = df["dg_id"].astype(int)
    df["course_num"] = df["course_num"].astype(int)
    df["surface"]    = df["course_num"].map(grass_lk)
    df = df.sort_values(["dg_id", date_col], ascending=[True, False])

    def _windows(vals):
        out = {"n_total": len(vals)}
        for w in WINDOWS:
            s = vals[:w]
            out[f"L{w}_avg"] = float(np.mean(s)) if len(s) >= MIN_ROUNDS else np.nan
        return out

    # ── player: career avg across ALL rounds (any surface) ────────────────────
    p_all = df[df["dg_id"] == dg_id]["sg_putt"].values
    career_avg = float(np.mean(p_all)) if len(p_all) >= MIN_ROUNDS else np.nan

    # ── player: per surface ───────────────────────────────────────────────────
    p_df = df[df["dg_id"] == dg_id].dropna(subset=["surface"])
    p_rows = []
    for surface, grp in p_df.groupby("surface"):
        vals = grp["sg_putt"].values
        if len(vals) < MIN_ROUNDS:
            continue
        row = {"surface": surface, "career_avg": career_avg, **_windows(vals)}
        # vs_career for each window
        for w in WINDOWS:
            raw = row.get(f"L{w}_avg")
            row[f"L{w}_vs_career"] = (raw - career_avg) if (
                raw is not None and not np.isnan(raw) and not np.isnan(career_avg)
            ) else np.nan
        p_rows.append(row)

    if not p_rows:
        return pd.DataFrame(), pd.DataFrame()

    player_out = pd.DataFrame(p_rows)

    # ── field: per (dg_id, surface) ───────────────────────────────────────────
    fids  = set(int(x) for x in field_ids if pd.notna(x)) if field_ids else None
    f_df  = df[df["dg_id"].isin(fids)].dropna(subset=["surface"]) if fids \
            else df.dropna(subset=["surface"])

    f_rows = []
    for (fid, surface), grp in f_df.groupby(["dg_id", "surface"]):
        vals = grp["sg_putt"].values
        if len(vals) < MIN_ROUNDS:
            continue
        # career avg for this field player
        fall = df[df["dg_id"] == fid]["sg_putt"].values
        f_career = float(np.mean(fall)) if len(fall) >= MIN_ROUNDS else np.nan
        row = {"dg_id": fid, "surface": surface, "career_avg": f_career,
               **_windows(vals)}
        for w in WINDOWS:
            raw = row.get(f"L{w}_avg")
            row[f"L{w}_vs_career"] = (raw - f_career) if (
                raw is not None and not np.isnan(raw) and not np.isnan(f_career)
            ) else np.nan
        f_rows.append(row)

    field_out = pd.DataFrame(f_rows) if f_rows else pd.DataFrame()
    return player_out, field_out


# ── Visual 1: Percentile dot plot ─────────────────────────────────────────────

def _render_percentile_plot(player_df, field_df, player_name,
                            window, tw_surface):
    vc = f"L{window}_vs_career"
    sub = player_df.dropna(subset=[vc]).copy()
    if sub.empty:
        st.caption(f"No L{window} vs-career data.")
        return

    def _sk(s):
        try:    return SURFACE_ORDER.index(s)
        except: return len(SURFACE_ORDER)

    sub = sub.copy()
    sub["_o"] = sub["surface"].apply(_sk)
    sub = sub.sort_values("_o", ascending=False)   # reversed = top of chart first

    fig = go.Figure()

    for x0, x1, col in [
        (0, 25,  "rgba(220,50,50,0.04)"),
        (25,50,  "rgba(220,150,50,0.04)"),
        (50,75,  "rgba(100,180,100,0.04)"),
        (75,100, "rgba(50,200,100,0.06)"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col, layer="below", line_width=0)
    for xv, txt in [(12.5,"Bottom 25%"),(37.5,"25–50th"),(62.5,"50–75th"),(87.5,"Top 25%")]:
        fig.add_annotation(x=xv, y=1.01, yref="paper",
            text=f"<span style='font-size:9px;color:rgba(180,180,180,0.28)'>{txt}</span>",
            showarrow=False, xanchor="center")

    first = True
    for _, row in sub.iterrows():
        surface  = row["surface"]
        p_val    = row[vc]
        is_tw    = bool(tw_surface and surface.lower() == tw_surface.lower())

        f_vals = (field_df[field_df["surface"] == surface][vc].dropna().values
                  if not field_df.empty and vc in field_df.columns
                  else np.array([]))

        if len(f_vals) >= 4:
            pct           = float((f_vals < p_val).sum() / len(f_vals) * 100)
            fmed_pct      = float((f_vals < np.median(f_vals)).sum() / len(f_vals) * 100)
            n_field       = len(f_vals)
        else:
            pct = fmed_pct = np.nan
            n_field = 0

        if not np.isnan(pct):
            fig.add_trace(go.Scatter(
                x=[fmed_pct, pct], y=[surface, surface], mode="lines",
                line=dict(color="rgba(255,165,0,0.28)", width=2),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[fmed_pct], y=[surface], mode="markers",
                marker=dict(size=12, color=COL_FIELD, symbol="line-ns",
                            line=dict(color=COL_FIELD_TK, width=2)),
                name="Field median" if first else "",
                legendgroup="field", showlegend=first,
                hovertemplate=(f"<b>{surface}</b><br>Field median: "
                               f"{fmed_pct:.0f}th pct  (n={n_field})<extra></extra>"),
            ))

        fig.add_trace(go.Scatter(
            x=[pct if not np.isnan(pct) else 50], y=[surface], mode="markers",
            marker=dict(
                size=18 if is_tw else 14,
                color=COL_PLAYER,
                line=dict(color="white" if is_tw else "rgba(255,165,0,0.5)",
                          width=2.5 if is_tw else 1.5),
            ),
            name=player_name if first else "",
            legendgroup="player", showlegend=first,
            hovertemplate=(
                f"<b>{surface}</b>"
                + (" ★ THIS WEEK" if is_tw else "")
                + f"<br>L{window} vs career: {p_val:+.3f}<br>"
                + (f"Field pct: {pct:.0f}th  (n={n_field})"
                   if not np.isnan(pct) else "Insufficient field data")
                + "<extra></extra>"
            ),
        ))

        if is_tw:
            fig.add_annotation(
                x=pct if not np.isnan(pct) else 50, y=surface,
                text="★ this week", showarrow=False, xanchor="left",
                xshift=18,   # clear the dot (dot radius ~14px + breathing room)
                yshift=0,
                font=dict(size=11, color="rgba(255,165,0,1.0)"),
                bgcolor="rgba(255,165,0,0.12)",
                borderpad=4,
            )
        first = False

    fig.update_layout(
        height=max(320, len(sub) * 90),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=30, t=30, b=90),
        xaxis=dict(range=[-2,102], title="Field Percentile (vs-career)",
                   tickvals=[0,25,50,75,100],
                   ticktext=["0th","25th","50th","75th","100th"],
                   gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=12), fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.28,
                    xanchor="center", x=0.5,
                    font=dict(size=13), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Visual 2: vs-career bar per surface ───────────────────────────────────────

def _render_vs_career_bar(player_df, player_name, window, tw_surface):
    vc  = f"L{window}_vs_career"
    raw = f"L{window}_avg"
    sub = player_df.dropna(subset=[vc]).copy()
    if sub.empty:
        return

    def _sk(s):
        try:    return SURFACE_ORDER.index(s)
        except: return len(SURFACE_ORDER)

    sub["_o"] = sub["surface"].apply(_sk)
    sub = sub.sort_values("_o", ascending=False)

    surfaces   = sub["surface"].tolist()
    vs_vals    = sub[vc].tolist()
    raw_vals   = sub[raw].tolist()
    n_tots     = sub["n_total"].tolist()
    career_avg = float(sub["career_avg"].iloc[0]) if "career_avg" in sub.columns else np.nan

    def _rgba(hex_c, alpha):
        h = hex_c.lstrip("#")
        r2,g2,b2 = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r2},{g2},{b2},{alpha})"

    bar_colors  = []
    line_colors = []
    line_widths = []
    for s, v in zip(surfaces, vs_vals):
        is_tw = bool(tw_surface and s.lower() == tw_surface.lower())
        base  = GREEN_STRONG if v >= 0 else RED_STRONG
        alpha = 1.0 if is_tw else 0.55
        bar_colors.append(_rgba(base, alpha))
        line_colors.append("rgba(255,165,0,0.9)" if is_tw else "rgba(0,0,0,0)")
        line_widths.append(2.5 if is_tw else 0)

    hover_text = [
        f"<b>{s}</b>{'  ★ this week' if tw_surface and s.lower()==tw_surface.lower() else ''}"
        f"<br>L{window} avg: {rv:+.3f}"
        f"<br>Career avg: {career_avg:+.3f}"
        f"<br>vs career: {vv:+.3f}"
        f"<br>n={n}"
        for s, rv, vv, n in zip(surfaces, raw_vals, vs_vals, n_tots)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=surfaces, x=vs_vals,
        orientation="h",
        marker=dict(color=bar_colors,
                    line=dict(color=line_colors, width=line_widths)),
        text=[f"{v:+.3f}" for v in vs_vals],
        textposition="auto",
        textfont=dict(size=11, color="rgba(255,255,255,0.9)"),
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))

    # zero = career avg baseline
    fig.add_vline(x=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(
        height=max(320, len(surfaces) * 90),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=110, r=40, t=50, b=50),
        title=dict(
            text=f"L{window} vs career baseline ({career_avg:+.3f})",
            font=dict(size=11, color="rgba(180,180,180,0.7)"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="SG: Putt above / below career avg",
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=False, tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="rgba(210,210,210,0.88)"),
            gridcolor="rgba(128,128,128,0.1)", fixedrange=True,
        ),
        bargap=0.45,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Public entry point ─────────────────────────────────────────────────────────

def render_surface_putting_deepdive(
    rounds_df,
    schedule_df,
    dg_id:       int,
    player_name: str,
    event_id=None,
    cutoff_dt=None,
    field_ids=None,
    greens_ref_path=None,
):
    if schedule_df is None and greens_ref_path is None:
        st.caption("No surface data available.")
        return

    tw_surface = _this_week_surface(schedule_df, event_id, greens_ref_path)

    player_df, field_df = _compute_surface_stats(
        rounds_df, schedule_df, dg_id, field_ids, cutoff_dt, greens_ref_path
    )

    if player_df.empty:
        st.caption(f"Not enough surface putting data for {player_name}.")
        return

    # this week surface callout
    if tw_surface:
        st.markdown(
            f"<div style='font-size:11px;color:rgba(255,165,0,0.7);"
            f"margin-bottom:10px'>★ This week: <b>{tw_surface}</b> greens</div>",
            unsafe_allow_html=True,
        )

    col_sel, _ = st.columns([2, 5])
    with col_sel:
        window = st.radio(
            "Window", options=[12, 24, 36],
            format_func=lambda x: f"L{x}",
            index=2,
            horizontal=True,
            key=f"surface_putt_window_{dg_id}",
        )

    vc = f"L{window}_vs_career"
    if player_df.dropna(subset=[vc]).empty:
        st.caption(f"No L{window} surface data for {player_name}.")
        return

    col_pct, col_bar = st.columns([3, 2], gap="large")

    with col_pct:
        st.markdown(
            "<div style='font-size:12px;color:rgba(150,150,150,0.6);"
            "margin-bottom:4px'>Field percentile  (vs-career on each surface)</div>",
            unsafe_allow_html=True,
        )
        _render_percentile_plot(player_df, field_df, player_name, window, tw_surface)

    with col_bar:
        st.markdown(
            "<div style='font-size:12px;color:rgba(150,150,150,0.6);"
            "margin-bottom:4px'>Above / below career avg by surface</div>",
            unsafe_allow_html=True,
        )
        _render_vs_career_bar(player_df, player_name, window, tw_surface)
