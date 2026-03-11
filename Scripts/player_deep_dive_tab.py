from __future__ import annotations
from grass_putting_deepdive import render_surface_putting_deepdive
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _info_expander(label: str, content: str):
    with st.expander(f"ⓘ  {label}", expanded=False):
        st.markdown(content, unsafe_allow_html=True)


def _safe_mean(df, col):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _fmt_delta(v, ref, lower_better=False):
    if not np.isfinite(v) or not np.isfinite(ref):
        return ""
    diff = v - ref
    better = diff < 0 if lower_better else diff > 0
    color = "#22c55e" if better else "#ef4444"
    arrow = "▼" if diff < 0 else "▲"
    sign  = "+" if diff > 0 else ""
    return f"<span style='color:{color};font-size:11px'>{arrow}{sign}{diff:.1f}</span>"


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_player_deep_dive_tab(
    *,
    summary_top: pd.DataFrame,
    rounds_df: pd.DataFrame,
    cutoff_dt,
    all_players: pd.DataFrame,
    ID_TO_IMG: dict,
    NAME_TO_IMG: dict,
    render_player_hero,
    build_last_n_events_table,
    _last_n_rounds_pre_event,
    _event_end_table_roundlevel,
    get_headshot_url,
    show_headshot_cropped_card,
    heat_table,
    ytd: pd.DataFrame,
    course_fit_df: "pd.DataFrame | None" = None,
    course_num: "int | None" = None,
    approach_skill_df: "pd.DataFrame | None" = None,
    field_ids: "list | None" = None,
    season_year: int = 2026,
    schedule_df: "pd.DataFrame | None" = None,
    event_id: "int | None" = None,
    greens_ref_path: "str | None" = None,
):
    COL = "rgba(255, 165, 0, 1)"
    COL_FILL = "rgba(255, 165, 0, 0.18)"
    COL_FIELD = "rgba(150,150,150,0.5)"

    # ── Pool ─────────────────────────────────────────────────────────────────
    pool = (
        summary_top[["dg_id", "player_name", "close_odds"]]
        .dropna(subset=["dg_id", "player_name"])
        .drop_duplicates(subset=["dg_id"])
        .copy()
    )
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    if field_ids:
        pool = pool[pool["dg_id"].isin([int(x) for x in field_ids if pd.notna(x)])].copy()

    if pool.empty:
        st.info("No players available.")
        return

    dg_options  = pool["dg_id"].tolist()
    name_by_id  = dict(zip(pool["dg_id"], pool["player_name"]))
    odds_by_id  = dict(zip(pool["dg_id"], pool.get("close_odds", pd.Series())))

    weekly_top1 = st.session_state.get("weekly_top1_dg_id")
    default_dg  = int(weekly_top1) if weekly_top1 and int(weekly_top1) in dg_options else dg_options[0]

    if "dd_dg_id" not in st.session_state or st.session_state["dd_dg_id"] not in dg_options:
        st.session_state["dd_dg_id"] = default_dg

    # ── Player selector ───────────────────────────────────────────────────────
    dg_id = st.selectbox(
        "Player",
        options=dg_options,
        index=dg_options.index(st.session_state["dd_dg_id"]),
        format_func=lambda x: name_by_id.get(int(x), str(x)),
        key="dd_dg_id",
    )
    dg_id       = int(dg_id)
    player_name = name_by_id.get(dg_id, "Unknown")
    odds_val    = odds_by_id.get(dg_id)
    try:
        odds_val = float(odds_val) if odds_val is not None else None
    except Exception:
        odds_val = None

    # ── L40 rounds ────────────────────────────────────────────────────────────
    r40 = _last_n_rounds_pre_event(rounds_df, dg_id, cutoff_dt, n=60)

    ALL_STATS = [
        "sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt",
        "driving_dist", "driving_acc", "gir", "scrambling",
        "birdies", "bogies", "round_score",
    ]
    ALL_STATS = [c for c in ALL_STATS if c in rounds_df.columns]
    m = {c: _safe_mean(r40, c) for c in ALL_STATS}

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — HERO + KPIs
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    hero_l, hero_r = st.columns([7, 3], gap="large")

    with hero_l:
        render_player_hero(
            dg_id=dg_id, player_name=player_name,
            all_players=all_players,
            ID_TO_IMG=ID_TO_IMG, NAME_TO_IMG=NAME_TO_IMG,
            odds=odds_val, headshot_width=110,
        )

        # YTD KPIs
        yrow = ytd.loc[ytd["dg_id"] == dg_id].iloc[0] if not ytd.loc[ytd["dg_id"] == dg_id].empty else None

        def _fv(key, fmt="{:.0f}"):
            if yrow is None: return "—"
            v = yrow.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
            try: return fmt.format(float(v))
            except: return str(v)

        kpi_items = [
            ("Odds",    f"{odds_val:.1f}" if odds_val else "—"),
            ("Starts",  _fv("ytd_starts")),
            ("Cut %",   _fv("ytd_made_cut_pct", "{:.0%}")),
            ("Top 10",  _fv("ytd_top10")),
            ("Top 25",  _fv("ytd_top25")),
            ("Wins",    _fv("ytd_wins")),
        ]
        kpi_html = "".join(
            f"<div style='text-align:center;padding:10px 16px;"
            f"background:rgba(255,255,255,0.04);border-radius:8px;"
            f"border:1px solid rgba(255,255,255,0.07)'>"
            f"<div style='font-size:10px;color:rgba(180,180,180,0.6);"
            f"text-transform:uppercase;letter-spacing:0.08em'>{lbl}</div>"
            f"<div style='font-size:20px;font-weight:700;color:rgba(255,255,255,0.9)'>{val}</div>"
            f"</div>"
            for lbl, val in kpi_items
        )
        st.markdown(
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:12px'>{kpi_html}</div>",
            unsafe_allow_html=True,
        )

    with hero_r:
        # Implied probability gauge
        if odds_val and np.isfinite(odds_val) and odds_val > 0:
            prob = round(1 / odds_val * 100, 1)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": "Win Probability", "font": {"size": 12}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 9}},
                    "bar": {"color": COL, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 5],   "color": "rgba(255,255,255,0.03)"},
                        {"range": [5, 15],  "color": "rgba(255,255,255,0.05)"},
                        {"range": [15, 100],"color": "rgba(255,255,255,0.02)"},
                    ],
                    "threshold": {"line": {"color": COL, "width": 3}, "value": prob},
                },
            ))
            fig_gauge.update_layout(
                height=200, margin=dict(l=20, r=20, t=30, b=10),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1b — STACKED SG ROUNDS CHART (Last 60 Rounds)
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Form Trend - Last 60 Rounds")
    st.caption("Rounds without bars indicate non-PGA Tour events (SG breakdown not available).")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — FORM TREND
    # ══════════════════════════════════════════════════════════════════════════
    # st.divider()
    # st.subheader("Form Trend — Last 40 Rounds")
    _info_expander("How to read this",
        "SG Total per round (grey dots) with a smoothed moving average (orange line) and ±1 std deviation band. "
        "Rising line = improving form. Wide band = inconsistent."
    )

    smooth_w = st.slider("Smoothing window", 1, 15, 5, key="dd_smooth")

    if not r40.empty and "sg_total" in rounds_df.columns:
        td = r40.copy()
        td["sg"] = pd.to_numeric(td["sg_total"], errors="coerce")
        td = td.dropna(subset=["sg"]).reset_index(drop=True)
        td["idx"] = range(1, len(td) + 1)
        s = td["sg"].rolling(window=smooth_w, min_periods=1)
        td["smooth"] = s.mean()
        td["std"]    = s.std().fillna(0)
        td["upper"]  = td["smooth"] + td["std"]
        td["lower"]  = td["smooth"] - td["std"]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(td["idx"]) + list(td["idx"])[::-1],
            y=list(td["upper"]) + list(td["lower"])[::-1],
            fill="toself", fillcolor=COL_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig_trend.add_trace(go.Scatter(
            x=td["idx"], y=td["sg"], mode="markers",
            marker=dict(color="rgba(200,200,200,0.4)", size=5),
            showlegend=False,
            hovertemplate="Round %{x}<br>SG: %{y:.2f}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=td["idx"], y=td["smooth"], mode="lines",
            line=dict(color=COL, width=3), name="Smoothed SG",
            hovertemplate="Round %{x}<br>Avg: %{y:.2f}<extra></extra>",
        ))
        fig_trend.update_layout(
            height=380, template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=50),
            yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", title="SG Total"),
            xaxis=dict(title="Round (oldest → most recent)"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Not enough round data for form trend.")

    df60 = _last_n_rounds_pre_event(rounds_df, dg_id, cutoff_dt, n=60)
    if df60.empty:
        st.info("Not enough round data.")
    else:
        df60 = df60.copy()
        df60["round_index"] = range(1, len(df60) + 1)
        for c in ["sg_total", "sg_app", "sg_ott", "sg_arg", "sg_putt"]:
            if c in df60.columns:
                df60[c] = pd.to_numeric(df60[c], errors="coerce")

        if "sg_total" not in df60.columns:
            st.info("sg_total not available.")
        else:
            comps = [("SG OTT", "sg_ott"), ("SG APP", "sg_app"), ("SG ARG", "sg_arg"), ("SG PUTT", "sg_putt")]
            base = df60.copy()

            bar_df = base.copy()
            for _, c in comps:
                bar_df[c] = pd.to_numeric(bar_df[c], errors="coerce")
            bar_df["sg_total"] = pd.to_numeric(bar_df["sg_total"], errors="coerce")
            need = [c for _, c in comps] + ["sg_total"]
            bar_df = bar_df.dropna(subset=need).copy()

            abs_sum = np.zeros(len(bar_df), dtype=float)
            for _, c in comps:
                abs_sum += np.abs(bar_df[c].to_numpy(dtype=float))
            bar_df = bar_df.loc[abs_sum > 0].copy()
            abs_sum = abs_sum[abs_sum > 0]

            for label, c in comps:
                w = np.abs(bar_df[c].to_numpy(dtype=float)) / abs_sum
                bar_df[f"w_{c}"] = w
                bar_df[f"seg_{c}"] = bar_df["sg_total"].to_numpy(dtype=float) * w

            x_bar = bar_df["round_index"].astype(int).tolist()
            COLOR_MAP = {
                "SG OTT":  "rgba(120,180,255,0.55)",
                "SG APP":  "rgba(255,170,190,0.55)",
                "SG ARG":  "rgba(190,150,255,0.55)",
                "SG PUTT": "rgba(255,190,120,0.55)",
            }

            fig_sg60 = go.Figure()
            for label, c in comps:
                fig_sg60.add_trace(go.Bar(
                    x=x_bar,
                    y=bar_df[f"seg_{c}"],
                    name=label,
                    marker=dict(color=COLOR_MAP[label]),
                    customdata=(bar_df[f"w_{c}"] * 100.0),
                    hovertemplate="%{fullData.name}: %{customdata:.0f}%<br>Seg: %{y:.2f}<extra></extra>",
                ))

            line_df = base.copy()
            line_df["sg_total"] = pd.to_numeric(line_df["sg_total"], errors="coerce")
            line_df = line_df.dropna(subset=["sg_total"]).copy()
            fig_sg60.add_trace(go.Scatter(
                x=line_df["round_index"].astype(int),
                y=line_df["sg_total"],
                mode="lines+markers",
                name="SG Total",
                line=dict(color="rgba(100,255,180,0.9)", width=2),
                marker=dict(size=5),
            ))

            fig_sg60.update_layout(
                barmode="relative",
                height=380,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=10, b=80),
                legend_title_text="",
                legend=dict(
                    orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_sg60.update_xaxes(type="linear", tickmode="auto", nticks=12)
            fig_sg60.update_yaxes(zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
            st.plotly_chart(fig_sg60, use_container_width=True)



    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — FIELD PERCENTILE BARS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Field Percentile — Last 60 Rounds")
    _info_expander("How to read this",
        "Each bar shows where this player ranks among everyone in this week's field (0 = last, 100 = best). "
        "Stats marked with † are lower-is-better (bogeys, round score) and are inverted so right is always good."
    )

    PCTILE_STATS = [
        ("sg_total",     "SG: Total",          False),
        ("sg_ott",       "SG: Off the Tee",    False),
        ("sg_app",       "SG: Approach",       False),
        ("sg_arg",       "SG: Around Green",   False),
        ("sg_putt",      "SG: Putting",        False),
        ("driving_dist", "Driving Distance",   False),
        ("driving_acc",  "Driving Accuracy",   False),
        ("gir",          "GIR %",              False),
        ("scrambling",   "Scrambling %",       False),
        ("birdies",      "Birdies / Round",    False),
        ("bogies",       "Bogeys / Round †",   True),
        ("round_score",  "Avg Round Score †",  True),
    ]

    _pctile_field_ids = field_ids or (
        pd.to_numeric(summary_top["dg_id"], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    _field_means_cache: dict = {}

    def _get_field_means(col):
        if col in _field_means_cache:
            return _field_means_cache[col]
        if col not in rounds_df.columns:
            _field_means_cache[col] = pd.Series(dtype=float)
            return _field_means_cache[col]
        fr = rounds_df[rounds_df["dg_id"].isin(_pctile_field_ids)] if _pctile_field_ids else rounds_df
        vals = (
            fr.groupby("dg_id")[col]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
            .dropna()
        )
        _field_means_cache[col] = vals
        return vals

    def _field_pctile(col, val, lower_better=False):
        if not np.isfinite(val):
            return np.nan
        fv = _get_field_means(col)
        if len(fv) < 2:
            return np.nan
        pct = float((fv < val).sum() / len(fv) * 100)
        return float(100 - pct) if lower_better else pct

    valid_pctile = []
    for col, label, lb in PCTILE_STATS:
        val = m.get(col, np.nan)
        pct = _field_pctile(col, val, lb)
        if np.isfinite(pct):
            valid_pctile.append((col, label, lb, val, pct))

    if valid_pctile:
        fig_pct = go.Figure()

        for x0, x1, color in [
            (0, 25,  "rgba(220,50,50,0.04)"),
            (25, 50, "rgba(220,150,50,0.04)"),
            (50, 75, "rgba(100,180,100,0.04)"),
            (75, 100,"rgba(50,200,100,0.06)"),
        ]:
            fig_pct.add_vrect(x0=x0, x1=x1, fillcolor=color, layer="below", line_width=0)

        for xv, txt in [(12.5,"Bottom 25%"),(37.5,"25–50th"),(62.5,"50–75th"),(87.5,"Top 25%")]:
            fig_pct.add_annotation(
                x=xv, y=1.01, yref="paper",
                text=f"<span style='font-size:9px;color:rgba(180,180,180,0.3)'>{txt}</span>",
                showarrow=False, xanchor="center",
            )

        for col, label, lb, val, pct in valid_pctile:
            fv = _get_field_means(col)
            field_med_pct = np.nan
            if len(fv) >= 2:
                field_med_pct = _field_pctile(col, float(fv.median()), lb)

            # Connector line between field avg and player dot
            if np.isfinite(field_med_pct):
                fig_pct.add_trace(go.Scatter(
                    x=[field_med_pct, pct], y=[label, label],
                    mode="lines",
                    line=dict(color="rgba(255,165,0,0.35)", width=2),
                    showlegend=False, hoverinfo="skip",
                ))

            # Field avg tick
            if np.isfinite(field_med_pct):
                fig_pct.add_trace(go.Scatter(
                    x=[field_med_pct], y=[label],
                    mode="markers",
                    marker=dict(size=10, color="rgba(150,150,150,0.7)",
                                symbol="line-ns", line=dict(color="rgba(200,200,200,0.6)", width=2)),
                    showlegend=(col == valid_pctile[0][0]),
                    name="Field avg",
                    legendgroup="field_avg",
                    hovertemplate=f"Field avg: <b>{field_med_pct:.0f}th pct</b><extra></extra>",
                ))

            # Player dot
            fig_pct.add_trace(go.Scatter(
                x=[pct], y=[label],
                mode="markers",
                marker=dict(size=14, color=COL, line=dict(color="white", width=1.5)),
                showlegend=(col == valid_pctile[0][0]),
                name=player_name,
                legendgroup="player",
                hovertemplate=f"<b>{label}</b>: <b>{pct:.0f}th pct</b><extra></extra>",
            ))

        fig_pct.update_layout(
            height=max(380, len(valid_pctile) * 48),
            template="plotly_dark",
            margin=dict(l=160, r=20, t=30, b=50),
            xaxis=dict(
                range=[-2, 102],
                title="Field Percentile",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0th", "25th", "50th", "75th", "100th"],
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                autorange="reversed",
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(size=12),
            ),
            barmode="overlay",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="top", y=-0.12,
                xanchor="center", x=0.5, font=dict(size=12),
                bgcolor="rgba(0,0,0,0)",
            ),
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pct, use_container_width=True)
    else:
        st.info("Not enough field data to compute percentiles.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2b — PUTTING BY SURFACE
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — SKILL PROFILE (ridge plot, single player vs field)
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Skill Distribution Profile — Last 60 Rounds")
    _info_expander("How to read this",
        "Each row is a kernel density curve showing the <b>full spread of round-by-round performance</b> "
        "for one skill over the last 60 rounds. A tall narrow peak = consistent. Wide flat curve = volatile. "
        "The dotted line and dot mark the player's mean. The grey shaded area shows the field distribution. "
        "The <span style='color:rgba(80,200,120,0.9)'>green band</span> shows what this course rewards — "
        "wider = more important skill."
    )

    RIDGE_CATS = ["driving_dist", "driving_acc", "sg_app", "sg_arg", "sg_putt"]
    RIDGE_FULL = ["Distance", "Accuracy", "Approach", "Around Green", "Putting"]
    IMP_KEYS   = ["imp_dist", "imp_acc", "imp_app", "imp_arg", "imp_putt"]
    RIDGE_NORMS = {
        "driving_dist": (295.0, 12.0),
        "driving_acc":  (0.60,  0.08),
        "sg_app":       (0.0,   0.80),
        "sg_arg":       (0.0,   0.45),
        "sg_putt":      (0.0,   0.55),
    }

    def _to_z(series, col):
        mu, sd = RIDGE_NORMS.get(col, (0.0, 1.0))
        return (series - mu) / sd if sd > 0 else series

    def _sg_series(df, col):
        if df is None or df.empty or col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna()

    def _kde(series, x_grid, bw=0.40):
        if len(series) < 3:
            return np.zeros_like(x_grid, dtype=float)
        diffs = (x_grid[:, None] - series.values[None, :]) / bw
        raw = np.mean(np.exp(-0.5 * diffs**2) / (bw * np.sqrt(2 * np.pi)), axis=1)
        pk = raw.max()
        return raw / pk if pk > 0 else raw

    sg_series = {c: _to_z(_sg_series(r40, c), c) for c in RIDGE_CATS}

    # Field distributions for context
    field_rounds = rounds_df[rounds_df["dg_id"].isin(_pctile_field_ids)] if _pctile_field_ids else rounds_df
    sg_field = {c: _to_z(_sg_series(field_rounds, c), c) for c in RIDGE_CATS}

    # Course importance
    course_imp = {}
    if course_fit_df is not None and course_num is not None:
        row = course_fit_df[pd.to_numeric(course_fit_df.get("course_num", pd.Series()), errors="coerce") == int(course_num)]
        if not row.empty:
            r0 = row.iloc[0]
            for k, key in zip(IMP_KEYS, RIDGE_CATS):
                v = pd.to_numeric(r0.get(k, np.nan), errors="coerce")
                course_imp[key] = float(v) if np.isfinite(v) else np.nan

    x_min, x_max = -4.0, 4.0
    x_grid = np.linspace(x_min, x_max, 300)
    OFFSET_STEP = 3.8
    CURVE_HEIGHT = 2.2

    all_z = pd.concat(list(sg_series.values()))
    if all_z.empty:
        st.info("Not enough round data for distribution chart.")
    else:
        fig_ridge = go.Figure()
        legend_added = {"player": False, "field": False, "course": False}

        for i, (cat, full_name) in enumerate(zip(RIDGE_CATS, RIDGE_FULL)):
            offset = i * OFFSET_STEP
            sp = sg_series[cat]
            sf = sg_field[cat]

            # Course band
            imp = course_imp.get(cat, np.nan)
            if np.isfinite(imp) and imp > 0:
                hw  = imp * 3.5
                bx0 = max(x_min, -hw)
                bx1 = min(x_max,  hw)
                top = offset + CURVE_HEIGHT * 1.05
                fig_ridge.add_trace(go.Scatter(
                    x=[bx0, bx0, bx1, bx1, bx0],
                    y=[offset, top, top, offset, offset],
                    fill="toself",
                    fillcolor="rgba(80,200,120,0.07)",
                    line=dict(color="rgba(80,200,120,0.28)", width=1, dash="dot"),
                    showlegend=not legend_added["course"],
                    name="Course sweet spot", legendgroup="course",
                    hoverinfo="skip",
                ))
                legend_added["course"] = True
                fig_ridge.add_annotation(
                    x=(bx0 + bx1) / 2, y=top + 0.12,
                    text=f"{imp:.0%}", showarrow=False, xanchor="center",
                    font=dict(size=10, color="rgba(80,200,120,0.75)"),
                )

            # Field distribution (grey)
            if len(sf) >= 3:
                kde_f = _kde(sf, x_grid) * CURVE_HEIGHT * 0.7 + offset
                fig_ridge.add_trace(go.Scatter(
                    x=list(x_grid) + list(x_grid[::-1]),
                    y=list(kde_f) + [offset] * len(x_grid),
                    fill="toself",
                    fillcolor="rgba(150,150,150,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=not legend_added["field"],
                    name="Field", legendgroup="field",
                    hoverinfo="skip",
                ))
                fig_ridge.add_trace(go.Scatter(
                    x=x_grid, y=kde_f, mode="lines",
                    line=dict(color="rgba(150,150,150,0.35)", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                legend_added["field"] = True

            # Player curve
            if len(sp) >= 3:
                kde_p = _kde(sp, x_grid) * CURVE_HEIGHT + offset
                mean_z = float(sp.mean())
                std_z  = float(sp.std())

                fig_ridge.add_trace(go.Scatter(
                    x=list(x_grid) + list(x_grid[::-1]),
                    y=list(kde_p) + [offset] * len(x_grid),
                    fill="toself", fillcolor=COL_FILL,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_ridge.add_trace(go.Scatter(
                    x=x_grid, y=kde_p, mode="lines",
                    line=dict(color=COL, width=2.5),
                    name=player_name, legendgroup="player",
                    showlegend=not legend_added["player"],
                    hoverinfo="skip",
                ))
                legend_added["player"] = True

                peak = float(_kde(sp, np.array([mean_z]), bw=0.40)[0]) * CURVE_HEIGHT + offset
                fig_ridge.add_trace(go.Scatter(
                    x=[mean_z, mean_z], y=[offset, peak], mode="lines",
                    line=dict(color=COL, width=2, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_ridge.add_trace(go.Scatter(
                    x=[mean_z], y=[peak], mode="markers",
                    marker=dict(size=13, color=COL, line=dict(color="white", width=2)),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{player_name}</b> — {full_name}<br>"
                        f"Mean: <b>{mean_z:+.2f}σ</b>  ·  Spread: {std_z:.2f}σ"
                        "<extra></extra>"
                    ),
                ))
                fig_ridge.add_annotation(
                    x=mean_z + (0.12 if mean_z >= 0 else -0.12),
                    y=peak + 0.08,
                    text=f"<b>{mean_z:+.2f}σ</b>",
                    showarrow=False,
                    xanchor="left" if mean_z >= 0 else "right",
                    font=dict(size=10, color=COL),
                )

            # Row label
            fig_ridge.add_annotation(
                xref="paper", x=0.0,
                yref="y",     y=offset + OFFSET_STEP * 0.42,
                text=f"<b>{full_name}</b>",
                showarrow=False, xanchor="right",
                font=dict(size=12, color="rgba(210,210,210,0.9)"),
            )
            fig_ridge.add_shape(
                type="line", x0=x_min, x1=x_max, y0=offset, y1=offset,
                line=dict(color="rgba(255,255,255,0.07)", width=1),
            )

        tick_vals = [-3, -2, -1, 0, 1, 2, 3]
        tick_text = [f"{v}σ" for v in tick_vals]
        tick_text[0] = "-3σ\nwell below avg"
        tick_text[3] = "0\nfield avg"
        tick_text[6] = "+3σ\nelite"

        fig_ridge.update_layout(
            height=900,
            template="plotly_dark",
            margin=dict(l=90, r=30, t=20, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                range=[x_min, x_max],
                tickvals=tick_vals, ticktext=tick_text,
                title="Standard Deviations from Tour Average",
                zeroline=True, zerolinecolor="rgba(255,255,255,0.25)",
                zerolinewidth=1.5,
                gridcolor="rgba(255,255,255,0.06)",
            ),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            legend=dict(
                orientation="h", yanchor="top", y=-0.12,
                xanchor="center", x=0.5, font=dict(size=12),
            ),
            showlegend=True,
            hovermode="x",
        )
        st.plotly_chart(fig_ridge, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — COURSE FIT
    # ══════════════════════════════════════════════════════════════════════════
    if course_fit_df is not None and course_num is not None:
        st.divider()
        st.subheader("Course Fit")
        _info_expander("How to read this",
            "Course fit is computed as the sum of <b>course beta × player z-score</b> for each skill. "
            "A positive score means the player's strengths align with what this course historically rewards. "
            "The bar chart shows the per-skill contribution."
        )

        IMP_LABELS  = ["Distance", "Accuracy", "Approach", "Around Green", "Putting"]
        SKILL_COLS  = ["driving_dist", "driving_acc", "sg_app", "sg_arg", "sg_putt"]
        BETA_KEYS   = ["beta_dist_shrunk", "beta_acc_shrunk", "beta_app_shrunk", "beta_arg_shrunk", "beta_putt_shrunk"]
        SKILL_COLORS = ["#60a5fa", "#34d399", "#f97316", "#a78bfa", "#fb7185"]

        fit_row = course_fit_df[
            pd.to_numeric(course_fit_df.get("course_num", pd.Series()), errors="coerce") == int(course_num)
        ]
        if fit_row.empty:
            st.info("No course fit data for this course.")
        else:
            r0 = fit_row.iloc[0]

            # Player z-scores
            player_z = {}
            for col in SKILL_COLS:
                mu, sd = RIDGE_NORMS.get(col, (0.0, 1.0))
                val = m.get(col, np.nan)
                player_z[col] = (val - mu) / sd if np.isfinite(val) and sd > 0 else np.nan

            # Fit score + breakdown
            total_fit = 0.0
            breakdown = {}
            for lbl, col, bk in zip(IMP_LABELS, SKILL_COLS, BETA_KEYS):
                beta = pd.to_numeric(r0.get(bk, np.nan), errors="coerce")
                z    = player_z.get(col, np.nan)
                if np.isfinite(beta) and np.isfinite(z):
                    contrib = float(beta * z)
                    breakdown[lbl] = contrib
                    total_fit += contrib

            randomness_pct = pd.to_numeric(r0.get("randomness_pct", np.nan), errors="coerce")
            course_name = str(r0.get("course", r0.get("course_name", f"Course {course_num}")))

            # Score card
            color_fit = "#22c55e" if total_fit > 0 else "#ef4444"
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.04);border-radius:10px;"
                f"border:1px solid rgba(255,255,255,0.08);padding:16px 20px;display:flex;"
                f"align-items:center;justify-content:space-between;margin-bottom:14px'>"
                f"<div>"
                f"<div style='font-size:11px;color:rgba(180,180,180,0.6);text-transform:uppercase;"
                f"letter-spacing:0.08em'>Course Fit Score</div>"
                f"<div style='font-size:32px;font-weight:800;color:{color_fit}'>{total_fit:+.3f}</div>"
                f"<div style='font-size:12px;color:rgba(180,180,180,0.7)'>{course_name}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if breakdown:
                labels = list(breakdown.keys())
                values = list(breakdown.values())
                colors = [SKILL_COLORS[IMP_LABELS.index(l)] for l in labels]

                fig_fit = go.Figure()
                fig_fit.add_trace(go.Bar(
                    y=labels, x=values, orientation="h",
                    marker=dict(color=colors),
                    hovertemplate="%{y}: <b>%{x:+.3f}</b><extra></extra>",
                ))
                fig_fit.update_layout(
                    height=280, template="plotly_dark",
                    margin=dict(l=20, r=40, t=10, b=40),
                    xaxis=dict(
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.35)",
                        title="Fit contribution",
                        gridcolor="rgba(255,255,255,0.06)",
                    ),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_fit, use_container_width=True)

            if np.isfinite(randomness_pct) and randomness_pct > 0.70:
                st.markdown(
                    f"<div style='background:rgba(255,165,0,0.08);border-radius:6px;"
                    f"border-left:3px solid rgba(255,165,0,0.6);padding:8px 12px;"
                    f"font-size:12px;color:rgba(220,200,150,0.9)'>"
                    f"High randomness course — {course_name} ranks in the "
                    f"{int(round(randomness_pct*100))}th percentile for randomness. "
                    f"Course fit is a weaker signal here."
                    f"</div>",
                    unsafe_allow_html=True,
                )



    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — APPROACH PROXIMITY (two combined greens)
    # ══════════════════════════════════════════════════════════════════════════
    if approach_skill_df is not None and not approach_skill_df.empty:
        st.divider()
        st.subheader("Approach Proximity")
        _info_expander("How to read this",
            "Top-down view of a green. Each ring is a distance bucket — smaller ring = closer to the hole = better. "
            "Left green = from fairway (4 buckets). Right green = from rough (2 buckets). "
            "The radius spoke shows the exact distance. Tour avg shown below."
        )

        PROX_COL  = "proximity_per_shot"
        FIXED_MAX = 60.0
        N_PTS     = 120

        FW_BUCKETS_P = [
            ("50–100 yds",  "50_100_fw",   "#f97316"),   # orange
            ("100–150 yds", "100_150_fw",  "#38bdf8"),   # sky blue
            ("150–200 yds", "150_200_fw",  "#a78bfa"),   # purple
            ("200+ yds",    "over_200_fw", "#34d399"),   # green
        ]
        RGH_BUCKETS_P = [
            ("Under 150 yds", "under_150_rgh", "#fb7185"),  # pink
            ("Over 150 yds",  "over_150_rgh",  "#fbbf24"),  # amber
        ]

        def _pv(prefix):
            col = f"{prefix}_{PROX_COL}"
            if col not in approach_skill_df.columns:
                return np.nan
            row = approach_skill_df[approach_skill_df["dg_id"] == dg_id]
            if row.empty:
                return np.nan
            return float(pd.to_numeric(row.iloc[0].get(col, np.nan), errors="coerce"))

        def _pt(prefix):
            col = f"{prefix}_{PROX_COL}"
            if col not in approach_skill_df.columns:
                return np.nan
            vals = pd.to_numeric(approach_skill_df[col], errors="coerce").dropna()
            return float(vals.median()) if len(vals) else np.nan

        def _ring_xy(r):
            a = np.linspace(0, 2 * np.pi, N_PTS)
            return list(np.cos(a) * r) + [None], list(np.sin(a) * r) + [None]

        def _build_green(buckets, title):
            axis_range = [-FIXED_MAX * 1.15, FIXED_MAX * 1.15]
            fig = go.Figure()

            # Green surface
            gx, gy = _ring_xy(FIXED_MAX)
            fig.add_trace(go.Scatter(
                x=gx, y=gy, fill="toself",
                fillcolor="rgba(34,85,34,0.35)",
                line=dict(color="rgba(80,140,80,0.5)", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(
                x=0, y=FIXED_MAX * 0.90,
                text="<span style='font-size:9px;color:rgba(160,200,160,0.55)'>60 ft</span>",
                showarrow=False, xanchor="center",
            )

            # Rings — draw largest first so smaller ones sit on top
            sorted_buckets = sorted(buckets, key=lambda b: _pv(b[1]) if np.isfinite(_pv(b[1])) else 999, reverse=True)

            # Angle offsets so radius spokes don't collide (spread around clock)
            n = len(buckets)
            angles = [i * (2 * np.pi / n) - np.pi / 2 for i in range(n)]  # start at top, spread evenly

            for idx, (blabel, bprefix, color) in enumerate(sorted_buckets):
                pval = _pv(bprefix)
                if not np.isfinite(pval):
                    continue
                pval = min(pval, FIXED_MAX * 0.97)
                # Convert hex to rgba for fill
                def _hex_to_rgba(h, a=0.15):
                    h = h.lstrip("#")
                    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                    return f"rgba({r},{g},{b},{a})"

                fill = _hex_to_rgba(color)

                rx, ry = _ring_xy(pval)
                fig.add_trace(go.Scatter(
                    x=rx, y=ry, fill="toself",
                    fillcolor=fill,
                    line=dict(color=color, width=2.5),
                    name=blabel,
                    showlegend=True,
                    hovertemplate=f"<b>{blabel}</b>: <b>{pval:.1f} ft</b><extra></extra>",
                ))

                # Radius spoke at assigned angle
                orig_idx = next(i for i, b in enumerate(buckets) if b[1] == bprefix)
                angle = angles[orig_idx]
                ex = np.cos(angle) * pval
                ey = np.sin(angle) * pval
                fig.add_shape(type="line", x0=0, y0=0, x1=ex, y1=ey,
                    line=dict(color=color, width=1.5, dash="dot"))
                label_x = np.cos(angle) * (pval + FIXED_MAX * 0.06)
                label_y = np.sin(angle) * (pval + FIXED_MAX * 0.06)
                fig.add_annotation(
                    x=label_x, y=label_y,
                    text=f"<b>{pval:.1f}</b>",
                    showarrow=False, xanchor="center",
                    font=dict(size=10, color=color),
                )

            # Hole dot + flag
            fig.add_trace(go.Scatter(
                x=[0], y=[0], mode="markers",
                marker=dict(size=10, color="white"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=FIXED_MAX * 0.20,
                line=dict(color="white", width=1.5))

            fig.update_layout(
                xaxis=dict(range=axis_range, showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="y"),
                yaxis=dict(range=axis_range, showgrid=False, zeroline=False,
                           showticklabels=False),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.04,
                    xanchor="center", x=0.5, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=True,
                height=420,
                margin=dict(l=5, r=5, t=30, b=60),
                title=dict(text=title, font=dict(size=13), x=0.5, xanchor="center"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        col_fw, col_rgh = st.columns(2, gap="large")

        with col_fw:
            fig_fw = _build_green(FW_BUCKETS_P, "From Fairway")
            st.plotly_chart(fig_fw, use_container_width=True)
            # Tour avg row
            avg_html = "".join(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<span style='color:{color};font-size:11px'>{lbl}</span>"
                f"<span style='font-size:11px;color:rgba(180,180,180,0.6)'>Tour avg "
                f"<b style='color:rgba(220,220,220,0.8)'>{_pt(pre):.1f} ft</b>"
                + (
                    f"  <span style='color:{'#22c55e' if (_pv(pre)-_pt(pre))<0 else '#ef4444'}'>"
                    f"{'+'if(_pv(pre)-_pt(pre))>0 else ''}{_pv(pre)-_pt(pre):.1f}</span>"
                    if np.isfinite(_pv(pre)) and np.isfinite(_pt(pre)) else ""
                ) +
                f"</span></div>"
                for lbl, pre, color in FW_BUCKETS_P
                if np.isfinite(_pt(pre))
            )
            if avg_html:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.03);border-radius:8px;"
                    f"padding:8px 12px;margin-top:-8px'>{avg_html}</div>",
                    unsafe_allow_html=True,
                )

        with col_rgh:
            fig_rgh = _build_green(RGH_BUCKETS_P, "From Rough")
            st.plotly_chart(fig_rgh, use_container_width=True)
            avg_html_r = "".join(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<span style='color:{color};font-size:11px'>{lbl}</span>"
                f"<span style='font-size:11px;color:rgba(180,180,180,0.6)'>Tour avg "
                f"<b style='color:rgba(220,220,220,0.8)'>{_pt(pre):.1f} ft</b>"
                + (
                    f"  <span style='color:{'#22c55e' if (_pv(pre)-_pt(pre))<0 else '#ef4444'}'>"
                    f"{'+'if(_pv(pre)-_pt(pre))>0 else ''}{_pv(pre)-_pt(pre):.1f}</span>"
                    if np.isfinite(_pv(pre)) and np.isfinite(_pt(pre)) else ""
                ) +
                f"</span></div>"
                for lbl, pre, color in RGH_BUCKETS_P
                if np.isfinite(_pt(pre))
            )
            if avg_html_r:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.03);border-radius:8px;"
                    f"padding:8px 12px;margin-top:-8px'>{avg_html_r}</div>",
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — COURSE HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Course History")
    _info_expander("How to read this",
        "Past results at this specific venue going back to 2017. SG columns are per-round averages. "
        "Green = above average, red = below average."
    )

    if course_num is None:
        st.info("No course selected.")
    else:
        out_parts = []

        # 2024+ roundlevel
        r = rounds_df.copy()
        r["dg_id"] = pd.to_numeric(r.get("dg_id"), errors="coerce")
        r = r.loc[r["dg_id"] == dg_id].copy()
        if not r.empty and "course_num" in r.columns:
            r["course_num"] = pd.to_numeric(r["course_num"], errors="coerce")
            r = r.loc[r["course_num"] == int(course_num)].copy()
            if not r.empty:
                ends_r = _event_end_table_roundlevel(rounds_df)
                r["year"] = pd.to_numeric(r.get("year"), errors="coerce")
                r["event_id"] = pd.to_numeric(r.get("event_id"), errors="coerce")
                r = r.dropna(subset=["year", "event_id"])
                r["year"] = r["year"].astype(int)
                r["event_id"] = r["event_id"].astype(int)
                r = r.merge(ends_r, on=["year", "event_id"], how="left")
                r["event_end"] = pd.to_datetime(r["event_end"], errors="coerce")
                if cutoff_dt is not None:
                    r = r.loc[r["event_end"].isna() | (r["event_end"] <= pd.to_datetime(cutoff_dt))]

                fin_col = "fin_text" if "fin_text" in r.columns else "finish_text" if "finish_text" in r.columns else None
                if fin_col is None:
                    r["fin_text"] = ""
                    fin_col = "fin_text"

                def _fns(s): v = s.dropna().astype(str); return v.iloc[0] if len(v) else ""
                agg = {"Finish": (fin_col, _fns)}
                for sg, lbl in [("sg_total","SG Total"),("sg_app","SG APP"),("sg_ott","SG OTT"),("sg_arg","SG ARG"),("sg_putt","SG PUTT")]:
                    if sg in r.columns:
                        agg[lbl] = (sg, "mean")

                t = r.groupby(["year", "event_id", "event_name"], as_index=False).agg(**agg).rename(
                    columns={"year": "Year", "event_name": "Event"})
                t["event_end"] = r.groupby(["year", "event_id"])["event_end"].max().values
                t["Event"] = t.apply(
                    lambda row: row["Event"] + " ⚠ - Cancelled after R1 "
                    if int(row["Year"]) == 2020 and int(row["event_id"]) == course_num
                    else row["Event"],
                    axis=1,
                )
                out_parts.append(t)

        if not out_parts:
            st.info("No course history found for this player at this venue.")
        else:
            ch = pd.concat(out_parts, ignore_index=True)
            ch["event_end"] = pd.to_datetime(ch.get("event_end"), errors="coerce")
            ch = ch.sort_values("event_end", ascending=False)
            sg_cols = [c for c in ["SG Total","SG APP","SG OTT","SG ARG","SG PUTT"] if c in ch.columns]
            for c in sg_cols:
                ch[c] = pd.to_numeric(ch[c], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
            show = [c for c in ["Year","Event","Finish","SG Total","SG APP","SG OTT","SG ARG","SG PUTT"] if c in ch.columns]
            st.dataframe(heat_table(ch[show], sg_cols=sg_cols, precision=2), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — RECENT TOURNAMENTS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Recent Tournaments")

    # Pull last 12 months of round-level data
    r_recent = rounds_df.copy()
    r_recent["dg_id"] = pd.to_numeric(r_recent.get("dg_id"), errors="coerce")
    r_recent = r_recent.loc[r_recent["dg_id"] == dg_id].copy()

    if not r_recent.empty:
        ends_r = _event_end_table_roundlevel(rounds_df)
        r_recent["year"] = pd.to_numeric(r_recent.get("year"), errors="coerce")
        r_recent["event_id"] = pd.to_numeric(r_recent.get("event_id"), errors="coerce")
        r_recent = r_recent.dropna(subset=["year", "event_id"]).copy()
        r_recent["year"] = r_recent["year"].astype(int)
        r_recent["event_id"] = r_recent["event_id"].astype(int)
        r_recent = r_recent.merge(ends_r, on=["year", "event_id"], how="left")
        r_recent["event_end"] = pd.to_datetime(r_recent["event_end"], errors="coerce")

        # Filter to last 12 months before cutoff
        cutoff_ts = pd.to_datetime(cutoff_dt) if cutoff_dt is not None else pd.Timestamp.now()
        one_year_ago = cutoff_ts - pd.DateOffset(months=12)
        r_recent = r_recent.loc[
            r_recent["event_end"].notna() &
            (r_recent["event_end"] <= cutoff_ts) &
            (r_recent["event_end"] >= one_year_ago)
            ].copy()

    if r_recent.empty:
        st.info("No round-level data in the last 12 months.")
    else:
        # Build round number within each event
        r_recent = r_recent.sort_values(["event_end", "event_id", "round_num"]
                                        if "round_num" in r_recent.columns
                                        else ["event_end", "event_id"])

        sg_round_cols = [c for c in ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt", "round_score"]
                         if c in r_recent.columns]
        for c in sg_round_cols:
            r_recent[c] = pd.to_numeric(r_recent[c], errors="coerce")

        name_col = "event_name" if "event_name" in r_recent.columns else "event_id"
        rnd_col = "round_num" if "round_num" in r_recent.columns else None

        date_col = "round_date" if "round_date" in r_recent.columns else None
        fin_col = "fin_text" if "fin_text" in r_recent.columns else "finish_text" if "finish_text" in r_recent.columns else None

        display_cols = {}
        display_cols["Event"] = r_recent[name_col].astype(str)
        display_cols["Year"] = r_recent["year"]
        if fin_col:
            # extract numeric finish for heat scaling
            display_cols["_fin_num"] = pd.to_numeric(
                r_recent[fin_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
        if rnd_col:
            display_cols["Rnd"] = pd.to_numeric(r_recent[rnd_col], errors="coerce").astype("Int64")
        if date_col:
            display_cols["Date"] = pd.to_datetime(r_recent[date_col], errors="coerce").dt.strftime("%b %d")
        if fin_col:
            display_cols["Finish"] = r_recent[fin_col].astype(str).replace("nan", "")
        if "round_position_text" in r_recent.columns:
            display_cols["Pos"] = r_recent["round_position_text"].astype(str).replace("nan", "")
        if "round_position" in r_recent.columns:
            display_cols["_pos_num"] = pd.to_numeric(r_recent["round_position"], errors="coerce")
        for c in sg_round_cols:
            lbl = c.replace("sg_", "SG ").replace("_", " ").upper().replace("ROUND SCORE", "Score")
            display_cols[lbl] = r_recent[c]

        rnd_df = pd.DataFrame(display_cols)
        if date_col:
            rnd_df["_sort_date"] = pd.to_datetime(r_recent[date_col].values, errors="coerce")
            rnd_df = rnd_df.sort_values("_sort_date", ascending=False).drop(columns=["_sort_date"])
        else:
            rnd_df = rnd_df.sort_values(
                ["Year", "Event", "Rnd"] if rnd_col else ["Year", "Event"],
                ascending=[False, True, True] if rnd_col else [False, True],
            )

        sg_show = [c for c in rnd_df.columns if c.startswith("SG") or c == "Score"]
        sg_only = [c for c in sg_show if c != "Score"]
        for c in sg_only:
            rnd_df[c] = pd.to_numeric(rnd_df[c], errors="coerce").map(
                lambda x: f"{x:+.2f}" if pd.notna(x) else ""
            )
        if "Score" in rnd_df.columns:
            rnd_df["Score"] = pd.to_numeric(rnd_df["Score"], errors="coerce").map(
                lambda x: f"{int(x)}" if pd.notna(x) else ""
            )

            # Heat scale finish and position — lower number = better (invert colormap)
            heat_num_cols = []
            sty = heat_table(rnd_df.drop(columns=[c for c in ["_pos_num", "_fin_num"] if c in rnd_df.columns]),
                             sg_cols=sg_only, precision=2)

            # Apply inverted gradient to numeric position/finish cols
            import re
            if "_pos_num" in rnd_df.columns and rnd_df["_pos_num"].notna().any():
                pos_num = pd.to_numeric(rnd_df["_pos_num"], errors="coerce")
                sty = sty.background_gradient(
                    gmap=-pos_num.fillna(pos_num.max()), subset=["Pos"],
                    cmap="RdYlGn", axis=0,
                )
            if "_fin_num" in rnd_df.columns and rnd_df["_fin_num"].notna().any():
                fin_num = pd.to_numeric(rnd_df["_fin_num"], errors="coerce")
                sty = sty.background_gradient(
                    gmap=-fin_num.fillna(fin_num.max()), subset=["Finish"],
                    cmap="RdYlGn", axis=0,
                )
            if "Score" in rnd_df.columns and rnd_df["Score"].replace("", np.nan).notna().any():
                score_num = pd.to_numeric(rnd_df["Score"].replace("", np.nan), errors="coerce")
                sty = sty.background_gradient(
                    gmap=-score_num.fillna(score_num.max()), subset=["Score"],
                    cmap="RdYlGn", axis=0,
                )

            st.dataframe(sty, use_container_width=True, hide_index=True)