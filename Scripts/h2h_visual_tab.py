from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def _info_expander(label: str, content: str):
    with st.expander(f"ⓘ  {label}", expanded=False):
        st.markdown(content, unsafe_allow_html=True)

def render_h2h_visual_tab(
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
    course_fit_df: "pd.DataFrame | None" = None,
    course_num: "int | None" = None,
    approach_skill_df: "pd.DataFrame | None" = None,
    field_ids: "list | None" = None,
):
    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    COL_A      = "rgba(255, 165, 0, 1)"
    COL_A_FILL = "rgba(255, 165, 0, 0.18)"
    COL_B      = "rgba(0, 191, 255, 1)"
    COL_B_FILL = "rgba(0, 191, 255, 0.18)"

    # ------------------------------------------------------------------
    # Pool / player selection
    # ------------------------------------------------------------------
    pool = (
        summary_top[["dg_id", "player_name", "close_odds"]]
        .dropna(subset=["dg_id", "player_name"])
        .drop_duplicates()
        .copy()
    )
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    # Filter to this week's field if provided — matches the only_in_field behaviour
    if field_ids:
        pool = pool[pool["dg_id"].isin([int(x) for x in field_ids if pd.notna(x)])].copy()

    if len(pool) < 2:
        st.info("Need at least two players available to compare.")
        return

    odds_by_id  = dict(zip(pool["dg_id"], pool["close_odds"]))
    name_to_id  = dict(zip(pool["player_name"], pool["dg_id"]))
    id_to_name  = dict(zip(pool["dg_id"], pool["player_name"]))
    player_options = sorted(pool["player_name"].unique().tolist())

    weekly_top1 = st.session_state.get("weekly_top1_dg_id")
    weekly_top2 = st.session_state.get("weekly_top2_dg_id")
    top1_name = id_to_name.get(int(weekly_top1)) if weekly_top1 is not None else None
    top2_name = id_to_name.get(int(weekly_top2)) if weekly_top2 is not None else None
    if not top1_name or top1_name not in player_options:
        top1_name = player_options[0]
    if not top2_name or top2_name not in player_options:
        top2_name = player_options[1] if len(player_options) > 1 else top1_name
    if top1_name == top2_name and len(player_options) > 1:
        top2_name = next((n for n in player_options if n != top1_name), top2_name)

    for key, default in [("h2h_vis_a", top1_name), ("h2h_vis_b", top2_name)]:
        if key not in st.session_state or st.session_state[key] not in player_options:
            st.session_state[key] = default
    if st.session_state["h2h_vis_a"] == st.session_state["h2h_vis_b"] and len(player_options) > 1:
        st.session_state["h2h_vis_b"] = next(
            (n for n in player_options if n != st.session_state["h2h_vis_a"]),
            st.session_state["h2h_vis_b"],
        )

    # ------------------------------------------------------------------
    # Header + selectors
    # ------------------------------------------------------------------
    st.header("Head-to-Head  —  Visual")

    selA, selB = st.columns(2, gap="large")
    with selA:
        st.markdown("### Player A")
        name_a = st.selectbox(" ", player_options, key="h2h_vis_a", label_visibility="collapsed")
    with selB:
        st.markdown("### Player B")
        name_b = st.selectbox(" ", player_options, key="h2h_vis_b", label_visibility="collapsed")

    dg_a = int(name_to_id[name_a])
    dg_b = int(name_to_id[name_b])
    if dg_a == dg_b:
        st.warning("Pick two different players.")
        return

    odds_a = odds_by_id.get(dg_a)
    odds_b = odds_by_id.get(dg_b)

    heroA, heroB = st.columns(2, gap="large")
    with heroA:
        render_player_hero(
            dg_id=dg_a, player_name=name_a, all_players=all_players,
            ID_TO_IMG=ID_TO_IMG, NAME_TO_IMG=NAME_TO_IMG,
            odds=odds_a, headshot_width=110,
        )
    with heroB:
        render_player_hero(
            dg_id=dg_b, player_name=name_b, all_players=all_players,
            ID_TO_IMG=ID_TO_IMG, NAME_TO_IMG=NAME_TO_IMG,
            odds=odds_b, headshot_width=110,
        )

    # ------------------------------------------------------------------
    # Shared data
    # ------------------------------------------------------------------
    SG_CATS   = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    SG_SHORT  = ["OTT", "APP", "ARG", "PUTT"]
    LOWER_BETTER = {"round_score", "bogies", "doubles_or_worse", "poor_shots", "prox_rgh", "prox_fw"}

    ALL_STATS = [
        "sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt",
        "driving_dist", "driving_acc", "gir", "scrambling",
        "prox_rgh", "prox_fw", "great_shots", "poor_shots",
        "birdies", "bogies", "doubles_or_worse", "round_score",
    ]
    ALL_STATS = [c for c in ALL_STATS if c in rounds_df.columns]

    ra40 = _last_n_rounds_pre_event(rounds_df, dg_a, cutoff_dt, n=40)
    rb40 = _last_n_rounds_pre_event(rounds_df, dg_b, cutoff_dt, n=40)

    def _safe_mean(df, col):
        if df is None or df.empty or col not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    ma = {c: _safe_mean(ra40, c) for c in ALL_STATS}
    mb = {c: _safe_mean(rb40, c) for c in ALL_STATS}

    # ==================================================================
    # 1. SKILL PROFILE — ridge plot with course fit bands
    # ==================================================================
    st.divider()
    st.subheader("Strokes Gained Profile (Last 40 Rounds)")
    _info_expander(" How to read this", """
    Each row shows the <b>full distribution of round-by-round performance</b> for one skill —
    not just an average, but every round played across the last 40.
    A tall, narrow peak means the player is <b>consistent</b>; a wide, flat curve means <b>volatile</b>.
    The dotted vertical line and labeled dot mark each player's <b>mean</b>.
    All skills are converted to tour z-scores so rows are directly comparable.
    The <span style='color:rgba(80,200,120,0.9)'>green band</span> shows the SG range
    this course rewards — a player whose peak sits inside the band fits this track.
    """)

    def _sg_series(df, col):
        if df is None or df.empty or col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna()

    # All 5 skills — SG cats + driving stats
    RIDGE_CATS  = ["driving_dist", "driving_acc", "sg_app", "sg_arg", "sg_putt"]
    RIDGE_FULL  = ["Distance", "Accuracy", "Approach", "Around Green", "Putting"]
    IMP_KEYS    = ["imp_dist",   "imp_acc",   "imp_app", "imp_arg",  "imp_putt"]

    # Field-level norms for z-scoring so all rows share one axis
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

    sg_series_a = {c: _to_z(_sg_series(ra40, c), c) for c in RIDGE_CATS}
    sg_series_b = {c: _to_z(_sg_series(rb40, c), c) for c in RIDGE_CATS}

    def _kde(series, x_grid, bw=0.40):
        if len(series) < 3:
            return np.zeros_like(x_grid)
        diffs = (x_grid[:, None] - series.values[None, :]) / bw
        raw = np.mean(np.exp(-0.5 * diffs**2) / (bw * np.sqrt(2 * np.pi)), axis=1)
        # Normalize to fixed peak height of 1.0 so all rows are equally tall
        peak = raw.max()
        return raw / peak if peak > 0 else raw

    all_z = pd.concat(list(sg_series_a.values()) + list(sg_series_b.values()))

    if all_z.empty:
        st.info("Not enough round data for distribution chart.")
    else:

        x_min = max(float(all_z.quantile(0.01)) - 0.5, -4.0)
        x_max = min(float(all_z.quantile(0.99)) + 0.5,  4.0)
        x_grid = np.linspace(x_min, x_max, 300)

        # Course importance weights
        course_imp = {}
        course_name_ridge = None
        if course_fit_df is not None and course_num is not None:
            cf_row = course_fit_df[
                pd.to_numeric(course_fit_df["course_num"], errors="coerce") == int(course_num)
            ]
            if not cf_row.empty:
                cf = cf_row.iloc[0]
                course_imp = {
                    cat: float(cf.get(imp_key, np.nan))
                    for cat, imp_key in zip(RIDGE_CATS, IMP_KEYS)
                }
                course_name_ridge = cf.get("course_name", "")

        OFFSET_STEP = 3.8
        CURVE_HEIGHT = 2.2  # how tall each normalized curve can grow within its row

        fig_ridge = go.Figure()
        legend_added = {"a": False, "b": False, "course": False}

        for i, (cat, full_name) in enumerate(zip(RIDGE_CATS, RIDGE_FULL)):
            offset = i * OFFSET_STEP
            sa = sg_series_a[cat]
            sb = sg_series_b[cat]

            # ── Course fit band ─────────────────────────────────────────
            imp = course_imp.get(cat, np.nan)
            if np.isfinite(imp) and imp > 0:
                hw = imp * 3.5
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
                    name="Course sweet spot",
                    legendgroup="course",
                    hoverinfo="skip",
                ))
                legend_added["course"] = True
                fig_ridge.add_annotation(
                    x=(bx0 + bx1) / 2,
                    y=top + 0.12,
                    text=f"{imp:.0%}",
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=10, color="rgba(80,200,120,0.75)"),
                )

            # ── KDE curves ──────────────────────────────────────────────
            means_this_row = {}
            for series, name_t, col_t, fill_t, dash, leg_key in [
                (sa, name_a, COL_A, COL_A_FILL, "solid", "a"),
                (sb, name_b, COL_B, COL_B_FILL, "dash",  "b"),
            ]:
                if len(series) < 3:
                    continue
                kde_y = _kde(series, x_grid) * CURVE_HEIGHT + offset
                mean_z = float(series.mean())
                std_z  = float(series.std())
                means_this_row[name_t] = (mean_z, std_z, col_t)

                # Fill
                fig_ridge.add_trace(go.Scatter(
                    x=list(x_grid) + list(x_grid[::-1]),
                    y=list(kde_y) + [offset] * len(x_grid),
                    fill="toself",
                    fillcolor=fill_t,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Curve
                fig_ridge.add_trace(go.Scatter(
                    x=x_grid,
                    y=kde_y,
                    mode="lines",
                    line=dict(color=col_t, width=2.5, dash=dash),
                    name=name_t,
                    legendgroup=name_t,
                    showlegend=not legend_added[leg_key],
                    hoverinfo="skip",
                ))
                legend_added[leg_key] = True

                # Mean tick — vertical line + prominent dot at top
                peak = float(_kde(series, np.array([mean_z]), bw=0.40)[0]) * CURVE_HEIGHT + offset
                fig_ridge.add_trace(go.Scatter(
                    x=[mean_z, mean_z],
                    y=[offset, peak],
                    mode="lines",
                    line=dict(color=col_t, width=2, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Large dot at the peak
                fig_ridge.add_trace(go.Scatter(
                    x=[mean_z],
                    y=[peak],
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=col_t,
                        line=dict(color="white", width=2),
                        symbol="circle",
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name_t}</b>  —  {full_name}<br>"
                        f"Mean: <b>{mean_z:+.2f}σ</b>  ·  Spread: {std_z:.2f}σ"
                        "<extra></extra>"
                    ),
                ))

            # Mean value labels — stagger vertically if players are close
            items = list(means_this_row.items())
            label_positions = []
            for name_t, (mean_z, std_z, col_t) in items:
                peak = float(_kde(
                    sg_series_a[cat] if name_t == name_a else sg_series_b[cat],
                    np.array([mean_z]), bw=0.40
                )[0]) * CURVE_HEIGHT + offset
                label_positions.append((name_t, mean_z, std_z, col_t, peak))

            # Sort by mean_z so we can detect overlap
            label_positions.sort(key=lambda t: t[1])
            y_offsets = [0.0] * len(label_positions)
            if len(label_positions) == 2:
                gap = abs(label_positions[1][1] - label_positions[0][1])
                if gap < 0.5:  # too close — stagger vertically
                    y_offsets = [0.25, -0.25]

            for (name_t, mean_z, std_z, col_t, peak), y_off in zip(label_positions, y_offsets):
                anchor = "left" if mean_z >= 0 else "right"
                x_off = mean_z + (0.12 if mean_z >= 0 else -0.12)
                fig_ridge.add_annotation(
                    x=x_off,
                    y=peak + 0.08 + y_off,
                    text=f"<b>{mean_z:+.2f}σ</b>",
                    showarrow=False,
                    xanchor=anchor,
                    font=dict(size=10, color=col_t),
                )

            # Row label
            fig_ridge.add_annotation(
                xref="paper", x=0.0,
                yref="y",     y=offset + OFFSET_STEP * 0.42,
                text=f"<b>{full_name}</b>",
                showarrow=False,
                xanchor="right",
                font=dict(size=12, color="rgba(210,210,210,0.9)"),
            )
            # Baseline
            fig_ridge.add_shape(
                type="line",
                x0=x_min, x1=x_max,
                y0=offset, y1=offset,
                line=dict(color="rgba(255,255,255,0.07)", width=1),
            )

        # x-axis tick labels: show z-score + intuitive meaning
        tick_vals = [-3, -2, -1, 0, 1, 2, 3]
        tick_text = [
            "−3σ<br><sub>well below avg</sub>",
            "−2σ", "−1σ",
            "0<br><sub>field avg</sub>",
            "+1σ", "+2σ",
            "+3σ<br><sub>elite</sub>",
        ]

        fig_ridge.update_layout(
            height=900,
            template="plotly_dark",
            margin=dict(l=90, r=30, t=20, b=80),
            xaxis=dict(
                title="Standard Deviations from Tour Average",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.25)",
                zerolinewidth=1.5,
                range=[x_min, x_max],
                tickvals=tick_vals,
                ticktext=tick_text,
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10),
            ),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            legend=dict(
                orientation="h", yanchor="top", y=-0.12,
                xanchor="center", x=0.5,
                font=dict(size=12),
            ),
            showlegend=True,
            hovermode="x",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ridge, use_container_width=True)



    # ==================================================================
    # 2. PERCENTILE BARS — stat comparison with field context
    # ==================================================================
    st.divider()
    st.subheader("Stat Comparison — Field Percentile")
    _info_expander(" How to read this", """
        Each bar shows where a player ranks in the current field (0 = last, 100 = best).
        Dots are colored by who wins that stat. The gap between dots shows how meaningful
        the edge is relative to everyone they're competing against this week.
        </span></div>
    """)

    PCTILE_STATS = [
        ("sg_total",    "SG: Total",           False),
        ("sg_ott",      "SG: Off the Tee",      False),
        ("sg_app",      "SG: Approach",         False),
        ("sg_arg",      "SG: Around Green",     False),
        ("sg_putt",     "SG: Putting",          False),
        ("driving_dist","Driving Distance",      False),
        ("driving_acc", "Driving Accuracy",      False),
        ("gir",         "GIR %",                False),
        ("scrambling",  "Scrambling %",         False),
        ("birdies",     "Birdies / Round",      False),
        ("bogies",      "Bogeys / Round",       True),   # lower is better
        ("round_score", "Avg Round Score",      True),   # lower is better
    ]

    # Build field percentiles from summary_top (all players in the field this week)
    # Pre-compute field means from L40 rounds for all field players
    _field_means_cache = {}
    def _get_field_means(col):
        if col in _field_means_cache:
            return _field_means_cache[col]
        if col not in rounds_df.columns:
            _field_means_cache[col] = pd.Series(dtype=float)
            return _field_means_cache[col]
        field_rounds = rounds_df[rounds_df["dg_id"].isin(field_ids)] if field_ids else rounds_df
        vals = (
            field_rounds.groupby("dg_id")[col]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
            .dropna()
        )
        _field_means_cache[col] = vals
        return vals

    def _field_pctile(col, val, lower_better=False):
        if not np.isfinite(val):
            return np.nan
        field_vals = _get_field_means(col)
        if len(field_vals) < 2:
            return np.nan
        n_below = (field_vals < val).sum()
        pct = float(n_below / len(field_vals) * 100)
        return float(100 - pct) if lower_better else pct

    # Compute means from L40 rounds
    valid_rows = []
    for col, label, lower_better in PCTILE_STATS:
        if col not in rounds_df.columns:
            continue
        va = ma.get(col, np.nan)
        vb = mb.get(col, np.nan)
        pct_a = _field_pctile(col, va, lower_better)
        pct_b = _field_pctile(col, vb, lower_better)
        if not np.isfinite(pct_a) and not np.isfinite(pct_b):
            continue
        valid_rows.append((col, label, lower_better, va, vb, pct_a, pct_b))

    if valid_rows:
        fig_pct = go.Figure()
        legend_a = legend_b = False

        for col, label, lower_better, va, vb, pct_a, pct_b in valid_rows:
            # Who wins this stat
            if np.isfinite(pct_a) and np.isfinite(pct_b):
                a_wins = pct_a > pct_b
            else:
                a_wins = None

            ca = "#f97316" if a_wins else "#64748b" if a_wins is None else "#64748b"
            cb = "#38bdf8" if not a_wins else "#64748b" if a_wins is None else "#f97316"
            if a_wins is True:
                ca, cb = "#f97316", "#94a3b8"
            elif a_wins is False:
                ca, cb = "#94a3b8", "#38bdf8"
            else:
                ca = cb = "#94a3b8"

            # Connector line
            if np.isfinite(pct_a) and np.isfinite(pct_b):
                fig_pct.add_trace(go.Scatter(
                    x=[pct_a, pct_b], y=[label, label],
                    mode="lines",
                    line=dict(color="rgba(150,150,150,0.25)", width=2),
                    showlegend=False, hoverinfo="skip",
                ))

            # Shaded bar background
            fig_pct.add_shape(
                type="rect",
                x0=0, x1=100,
                y0=label, y1=label,
                xref="x", yref="y",
                fillcolor="rgba(255,255,255,0.03)",
                line_width=0,
            )

            # Player A dot
            if np.isfinite(pct_a):
                fmt_va = f"{va:.1f}" if abs(va) > 0.1 else f"{va:.2f}"
                fig_pct.add_trace(go.Scatter(
                    x=[pct_a], y=[label],
                    mode="markers",
                    marker=dict(size=16, color=ca,
                                line=dict(color="white", width=1.5)),
                    name=name_a, legendgroup="A",
                    showlegend=not legend_a,
                    hovertemplate=(
                        f"<b>{name_a}</b><br>"
                        f"{label}: {fmt_va}<br>"
                        f"Field percentile: <b>{pct_a:.0f}th</b>"
                        "<extra></extra>"
                    ),
                ))
                legend_a = True

            # Player B dot
            if np.isfinite(pct_b):
                fmt_vb = f"{vb:.1f}" if abs(vb) > 0.1 else f"{vb:.2f}"
                fig_pct.add_trace(go.Scatter(
                    x=[pct_b], y=[label],
                    mode="markers",
                    marker=dict(size=16, color=cb, symbol="diamond",
                                line=dict(color="white", width=1.5)),
                    name=name_b, legendgroup="B",
                    showlegend=not legend_b,
                    hovertemplate=(
                        f"<b>{name_b}</b><br>"
                        f"{label}: {fmt_vb}<br>"
                        f"Field percentile: <b>{pct_b:.0f}th</b>"
                        "<extra></extra>"
                    ),
                ))
                legend_b = True

        # Percentile zone backgrounds
        for x0, x1, color, zone_label in [
            (0,  25,  "rgba(220,50,50,0.04)",   ""),
            (25, 50,  "rgba(220,150,50,0.04)",  ""),
            (50, 75,  "rgba(100,180,100,0.04)", ""),
            (75, 100, "rgba(50,200,100,0.06)",  ""),
        ]:
            fig_pct.add_vrect(
                x0=x0, x1=x1,
                fillcolor=color, layer="below", line_width=0,
            )

        # Zone tick labels
        for xv, txt in [(12.5, "Bottom 25%"), (37.5, "25–50th"), (62.5, "50–75th"), (87.5, "Top 25%")]:
            fig_pct.add_annotation(
                x=xv, y=1.01, yref="paper",
                text=f"<span style='font-size:9px;color:rgba(180,180,180,0.35)'>{txt}</span>",
                showarrow=False, xanchor="center",
            )

        fig_pct.update_layout(
            height=max(400, len(valid_rows) * 52),
            template="plotly_dark",
            margin=dict(l=160, r=20, t=30, b=60),
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
            legend=dict(
                orientation="h", yanchor="top", y=-0.1,
                xanchor="center", x=0.5, font=dict(size=12),
            ),
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pct, use_container_width=True)
    else:
        st.info("Not enough field data to compute percentiles.")

    # ==================================================================
    # 3. FORM TREND with confidence band
    # ==================================================================
    st.divider()
    st.subheader("Form Trend — Last 40 Rounds")

    smooth_window = st.slider("Smoothing window", 1, 15, 5, key="h2h_vis_smooth")

    if not ra40.empty and not rb40.empty and "sg_total" in rounds_df.columns:
        def _prep_trend(df, w):
            d = df.copy()
            d["sg"] = pd.to_numeric(d["sg_total"], errors="coerce")
            d = d.dropna(subset=["sg"]).reset_index(drop=True)
            d["idx"] = range(1, len(d) + 1)
            s = d["sg"].rolling(window=w, min_periods=1)
            d["smooth"] = s.mean()
            d["std"]    = s.std().fillna(0)
            d["upper"]  = d["smooth"] + d["std"]
            d["lower"]  = d["smooth"] - d["std"]
            return d

        ra_t = _prep_trend(ra40, smooth_window)
        rb_t = _prep_trend(rb40, smooth_window)

        fig_trend = go.Figure()
        for df_t, name_t, col_line, col_fill in [
            (ra_t, name_a, COL_A, COL_A_FILL),
            (rb_t, name_b, COL_B, COL_B_FILL),
        ]:
            fig_trend.add_trace(go.Scatter(
                x=list(df_t["idx"]) + list(df_t["idx"])[::-1],
                y=list(df_t["upper"]) + list(df_t["lower"])[::-1],
                fill="toself", fillcolor=col_fill,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_t["idx"], y=df_t["sg"], mode="markers",
                marker=dict(color=col_fill.replace("0.18", "0.55"), size=5),
                showlegend=False,
                hovertemplate="Round %{x}<br>SG: %{y:.2f}<extra></extra>",
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_t["idx"], y=df_t["smooth"], mode="lines",
                line=dict(color=col_line, width=3), name=name_t,
                hovertemplate="Round %{x}<br>Avg: %{y:.2f}<extra></extra>",
            ))

        fig_trend.update_layout(
            height=420, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=60),
            yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", title="SG Total"),
            xaxis=dict(title="Round (oldest → most recent)"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Not enough 2024+ round data to plot both players.")

    # ==================================================================
    # 4. VIOLIN — SG total distribution
    # ==================================================================
    st.divider()
    st.subheader("SG Total Distribution (Last 40 Rounds)")

    if not ra40.empty and not rb40.empty and "sg_total" in rounds_df.columns:
        fig_violin = go.Figure()
        for df_t, name_t, col in [(ra40, name_a, COL_A), (rb40, name_b, COL_B)]:
            vals = pd.to_numeric(df_t["sg_total"], errors="coerce").dropna().tolist()
            fig_violin.add_trace(go.Violin(
                y=vals, name=name_t,
                box_visible=True, meanline_visible=True,
                points="all", jitter=0.35, pointpos=0,
                line_color=col,
                fillcolor=col.replace("1)", "0.25)"),
                marker=dict(size=4, opacity=0.55),
                hovertemplate="%{y:.2f}<extra></extra>",
            ))
        fig_violin.update_layout(
            height=380, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", title="SG Total"),
            violinmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.info("Not enough data for distribution chart.")

    # ==================================================================
    # 5. IMPLIED WIN PROBABILITY bar
    # ==================================================================
    st.divider()
    st.subheader("Implied Win Probability")

    def _odds_to_prob(odds):
        try:
            o = float(odds)
            return round(1 / o * 100, 1) if o > 0 else np.nan
        except Exception:
            return np.nan

    prob_a = _odds_to_prob(odds_a)
    prob_b = _odds_to_prob(odds_b)

    if np.isfinite(prob_a) and np.isfinite(prob_b):
        total  = prob_a + prob_b
        norm_a = round(prob_a / total * 100, 1)
        norm_b = round(prob_b / total * 100, 1)

        fig_gauge = go.Figure()
        fig_gauge.add_trace(go.Bar(
            x=[norm_a], y=["H2H"], orientation="h",
            marker_color=COL_A, name=name_a,
            text=f"<b>{name_a}</b>  {norm_a}%",
            textposition="inside", insidetextanchor="start",
            hovertemplate=f"{name_a}: {norm_a}% (raw {prob_a}%)<extra></extra>",
            width=0.35,
        ))
        fig_gauge.add_trace(go.Bar(
            x=[norm_b], y=["H2H"], orientation="h",
            marker_color=COL_B, name=name_b,
            text=f"<b>{name_b}</b>  {norm_b}%",
            textposition="inside", insidetextanchor="end",
            hovertemplate=f"{name_b}: {norm_b}% (raw {prob_b}%)<extra></extra>",
            width=0.35,
        ))
        fig_gauge.update_layout(
            barmode="stack", height=140, template="plotly_dark",
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False),
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(
            f"Probabilities normalised to this matchup only.  "
            f"Raw implied: {name_a} {prob_a}%  |  {name_b} {prob_b}%"
        )
    else:
        st.info("Odds not available for one or both players.")

    # ==================================================================
    # 6. COURSE FIT
    # ==================================================================
    st.divider()
    st.subheader("Course Fit")

    # Skill labels / colors / column mappings
    IMP_COLS   = ["imp_dist", "imp_acc", "imp_app", "imp_arg", "imp_putt"]
    IMP_LABELS = ["Distance", "Accuracy", "Approach", "Around Green", "Putting"]
    SKILL_COLORS = [
        "#63B3ED",  # Distance      — blue
        "#9ACD32",  # Accuracy      — yellow-green
        "#FFA500",  # Approach      — orange
        "#DA70D6",  # Around Green  — orchid
        "#40E0D0",  # Putting       — turquoise
    ]

    # The betas were fit on z-scored inputs, so we must z-score player stats
    # before multiplying. Field-level normalization constants (tour-wide estimates):
    FIELD_NORMS = {
        # stat_col: (field_mean, field_sd)
        "driving_dist": (295.0, 12.0),
        "driving_acc":  (0.60,  0.08),
        "sg_app":       (0.0,   0.80),
        "sg_arg":       (0.0,   0.45),
        "sg_putt":      (0.0,   0.55),
    }
    # Maps beta column → (player stat column, display label)
    SG_MAP = {
        "beta_dist_shrunk": ("driving_dist", "Distance"),
        "beta_acc_shrunk":  ("driving_acc",  "Accuracy"),
        "beta_app_shrunk":  ("sg_app",       "Approach"),
        "beta_arg_shrunk":  ("sg_arg",       "Around Green"),
        "beta_putt_shrunk": ("sg_putt",      "Putting"),
    }

    if course_fit_df is None or course_fit_df.empty or course_num is None:
        st.info("Course fit data not available for this event.")
    else:
        cfit = course_fit_df[
            pd.to_numeric(course_fit_df["course_num"], errors="coerce") == int(course_num)
        ]
        if cfit.empty:
            st.info(f"No course fit data found for course_num {course_num}.")
        else:
            cfit = cfit.iloc[0]
            course_name_display = cfit.get("course_name", f"Course {course_num}")
            predictability_pct  = float(cfit.get("predictability_pct", np.nan))
            randomness_pct      = float(cfit.get("randomness_pct", np.nan))

            imp_vals = [float(cfit.get(c, np.nan)) for c in IMP_COLS]

            # ── Extend player means to include driving stats ─────────────
            for col in ["driving_dist", "driving_acc"]:
                if col not in ma:
                    ma[col] = _safe_mean(ra40, col)
                if col not in mb:
                    mb[col] = _safe_mean(rb40, col)

            # ── Fit score: beta * z_score(player_stat) ───────────────────
            def _fit_score(player_means):
                total = 0.0
                breakdown = {}
                for beta_col, (stat_col, label) in SG_MAP.items():
                    beta = float(cfit.get(beta_col, np.nan))
                    raw  = player_means.get(stat_col, np.nan)
                    if not np.isfinite(beta) or not np.isfinite(raw):
                        breakdown[label] = np.nan
                        continue
                    mu, sd = FIELD_NORMS.get(stat_col, (0.0, 1.0))
                    z = (raw - mu) / sd if sd > 0 else 0.0
                    contrib = beta * z
                    total  += contrib
                    breakdown[label] = contrib
                return total, breakdown

            score_a, breakdown_a = _fit_score(ma)
            score_b, breakdown_b = _fit_score(mb)
            scores_valid = np.isfinite(score_a) and np.isfinite(score_b)

            # ── Layout: DNA bar left, fit scores right ───────────────────
            col_dna, col_scores = st.columns([3, 2], gap="large")

            with col_dna:
                st.markdown(f"**{course_name_display}**")
                st.caption("Importance of each skill in predicting performance here")

                if any(np.isfinite(v) for v in imp_vals):
                    fig_dna = go.Figure()
                    for label, val, color in zip(IMP_LABELS, imp_vals, SKILL_COLORS):
                        if not np.isfinite(val):
                            continue
                        fig_dna.add_trace(go.Bar(
                            x=[val * 100], y=[""],
                            orientation="h",
                            marker_color=color,
                            name=label,
                            text=f"<b>{label}</b>  {val*100:.0f}%",
                            textposition="inside",
                            insidetextanchor="middle",
                            hovertemplate=f"{label}: {val*100:.1f}%<extra></extra>",
                            width=0.5,
                        ))
                    fig_dna.update_layout(
                        barmode="stack", height=80,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(range=[0, 100], showticklabels=False,
                                   showgrid=False, zeroline=False),
                        yaxis=dict(showticklabels=False, showgrid=False),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_dna, use_container_width=True)

                if np.isfinite(predictability_pct):
                    pct = int(round(predictability_pct * 100))
                    bar_color = "#40E0D0" if pct >= 50 else "#FFA07A"
                    st.markdown(
                        f"<div style='margin-top:6px'>"
                        f"<span style='color:rgba(180,180,180,0.8);font-size:12px'>Skill predictability</span><br>"
                        f"<div style='background:rgba(255,255,255,0.08);border-radius:4px;"
                        f"height:8px;width:100%;margin:4px 0'>"
                        f"<div style='background:{bar_color};border-radius:4px;"
                        f"height:8px;width:{pct}%'></div></div>"
                        f"<span style='color:rgba(220,220,220,0.9);font-size:13px'>"
                        f"{pct}th percentile</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with col_scores:
                if scores_valid:
                    diff = score_a - score_b
                    leader = name_a if diff > 0 else name_b
                    edge   = abs(diff)

                    # Score cards
                    def _score_card(name, score, color):
                        sign = "+" if score >= 0 else ""
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;"
                            f"border-left:3px solid {color};padding:10px 14px;margin-bottom:8px'>"
                            f"<div style='color:rgba(180,180,180,0.8);font-size:12px'>{name}</div>"
                            f"<div style='color:{color};font-size:28px;font-weight:700;line-height:1.2'>"
                            f"{sign}{score:.2f}</div>"
                            f"<div style='color:rgba(150,150,150,0.7);font-size:11px'>fit score</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    _score_card(name_a, score_a, COL_A)
                    _score_card(name_b, score_b, COL_B)

                    edge_color = COL_A if diff > 0 else COL_B
                    st.markdown(
                        f"<div style='color:rgba(180,180,180,0.7);font-size:12px;margin-top:4px'>"
                        f"<span style='color:{edge_color};font-weight:600'>{leader}</span>"
                        f" has the course fit edge by "
                        f"<span style='color:{edge_color};font-weight:600'>{edge:.2f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Insufficient data for fit scores.")

            # ── Skill breakdown — one row per skill, both players side by side ──
            if scores_valid:

                # Methodology explainer
                _info_expander("How course fit is calculated", """
                    Each course has regression-derived betas that measure how much each skill
                    predicted actual scoring outcomes at that venue across thousands of rounds.
                    A higher score means the player's strengths align with what this course rewards.
                    <b> Course Fit Score = (course <span style='color:rgba(80,200,120,0.9)'>beta</span> × player <span style='color:rgba(80,200,120,0.9)'>z-score</span>)</b>
                    <br><br>
                    <span style='color:rgba(80,200,120,0.9)'>
                    beta:</span> coefficient measuring the strength of relationship between a skill
                    stat and scoring at that course.
                    <br>

                    <span style='color:rgba(80,200,120,0.9)'>
                    z-score:</span> how many standard deviations a player's stat sits above or below
                    the tour average.
                    </span>
                    </div>
                """)

                st.markdown("**Skill-by-skill breakdown**")

                # Build one Plotly row per skill with both players — clean & readable
                valid_skills = [
                    (lbl, color, breakdown_a.get(lbl, np.nan), breakdown_b.get(lbl, np.nan))
                    for lbl, color in zip(IMP_LABELS, SKILL_COLORS)
                    if np.isfinite(breakdown_a.get(lbl, np.nan))
                    and np.isfinite(breakdown_b.get(lbl, np.nan))
                ]

                # Get player raw stat values for richer hover
                STAT_COLS = {
                    "Distance":     ("driving_dist", "{:.0f} yds"),
                    "Accuracy":     ("driving_acc",  "{:.1%}"),
                    "Approach":     ("sg_app",       "{:+.2f} SG"),
                    "Around Green": ("sg_arg",       "{:+.2f} SG"),
                    "Putting":      ("sg_putt",      "{:+.2f} SG"),
                }

                fig_bd = go.Figure()

                # Add traces in reverse order so Distance appears at top
                for lbl, color, va, vb in reversed(valid_skills):
                    stat_col, fmt = STAT_COLS.get(lbl, (None, "{}"))
                    raw_a = ma.get(stat_col, np.nan) if stat_col else np.nan
                    raw_b = mb.get(stat_col, np.nan) if stat_col else np.nan

                    hover_a = (
                        f"<b>{name_a}</b> — {lbl}<br>"
                        f"Fit contribution: <b>{va:+.3f}</b><br>"
                        + (f"L40 avg: {fmt.format(raw_a)}" if np.isfinite(raw_a) else "")
                        + "<extra></extra>"
                    )
                    hover_b = (
                        f"<b>{name_b}</b> — {lbl}<br>"
                        f"Fit contribution: <b>{vb:+.3f}</b><br>"
                        + (f"L40 avg: {fmt.format(raw_b)}" if np.isfinite(raw_b) else "")
                        + "<extra></extra>"
                    )

                    fig_bd.add_trace(go.Bar(
                        y=[lbl, lbl],
                        x=[va, vb],
                        orientation="h",
                        marker_color=[COL_A, COL_B],
                        marker_line_width=0,
                        name=lbl,
                        legendgroup=lbl,
                        showlegend=False,
                        customdata=[[name_a, va, raw_a], [name_b, vb, raw_b]],
                        hovertemplate=[hover_a, hover_b],
                        width=0.6,
                    ))

                # Add a zero reference line annotation per skill
                x_all = [v for _, _, va, vb in valid_skills for v in [va, vb] if np.isfinite(v)]
                x_min = min(x_all) * 1.25 if x_all else -0.5
                x_max = max(x_all) * 1.25 if x_all else 1.5

                # Legend proxy traces
                fig_bd.add_trace(go.Bar(
                    y=[None], x=[None], orientation="h",
                    marker_color=COL_A, name=name_a, showlegend=True,
                ))
                fig_bd.add_trace(go.Bar(
                    y=[None], x=[None], orientation="h",
                    marker_color=COL_B, name=name_b, showlegend=True,
                ))

                fig_bd.update_layout(
                    barmode="overlay",
                    height=max(300, len(valid_skills) * 68),
                    template="plotly_dark",
                    margin=dict(l=20, r=40, t=10, b=80),
                    xaxis=dict(
                        zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.35)",
                        zerolinewidth=1.5,
                        range=[x_min, x_max],
                        title="Fit contribution  (course beta × player z-score)",
                        gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(size=11),
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(size=13, color="rgba(220,220,220,0.9)"),
                        ticklabelposition="outside",
                    ),
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.22,
                        xanchor="center", x=0.5,
                        font=dict(size=12),
                    ),
                    showlegend=True,
                    hovermode="closest",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    bargap=0.3,
                    bargroupgap=0.05,
                )
                st.plotly_chart(fig_bd, use_container_width=True)

                if np.isfinite(randomness_pct) and randomness_pct > 0.70:
                    st.markdown(
                        f"<div style='background:rgba(255,165,0,0.08);border-radius:6px;"
                        f"border-left:3px solid rgba(255,165,0,0.6);padding:8px 12px;"
                        f"font-size:12px;color:rgba(220,200,150,0.9)'>"
                        f"High randomness course — {course_name_display} ranks in the "
                        f"{int(round(randomness_pct*100))}th percentile for randomness. "
                        f"Course fit is a weaker signal here; variance will play a larger role."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ==================================================================
    # 7. APPROACH PROXIMITY — top-down green view
    # ==================================================================
    if approach_skill_df is not None and not approach_skill_df.empty:
        st.divider()
        st.subheader("Approach Proximity from Fairway")
        _info_expander(" How to read this","""
            <span style='color:rgba(220,220,220,0.85);font-size:13px;line-height:1.7'>
            Top-down view looking at the green. The center dot is the hole.
            <b>Smaller ring = closer to the hole = better.</b>
            White dashed ring = tour avg.
            </span></div>
        """)

        FW_BUCKETS_PROX = [
            ("50–100 yds",  "50_100_fw",  60.0),
            ("100–150 yds", "100_150_fw", 60.0),
            ("150–200 yds", "150_200_fw", 60.0),
            ("200+ yds",    "over_200_fw", 60.0),
        ]
        PROX_COL = "proximity_per_shot"
        N_PTS    = 120

        def _get_prox(df, dg_id, prefix):
            col = f"{prefix}_{PROX_COL}"
            if col not in df.columns:
                return np.nan
            row = df[df["dg_id"] == dg_id]
            if row.empty:
                return np.nan
            return pd.to_numeric(row.iloc[0].get(col, np.nan), errors="coerce")

        def _tour_avg_prox(df, prefix):
            col = f"{prefix}_{PROX_COL}"
            if col not in df.columns:
                return np.nan
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(vals.median()) if len(vals) > 0 else np.nan

        def _ring_xy(r, n=N_PTS):
            """Return x, y arrays for a circle of radius r."""
            a = np.linspace(0, 2 * np.pi, n)
            return list(np.cos(a) * r) + [None], list(np.sin(a) * r) + [None]

        prox_cols = st.columns(4, gap="medium")

        # Shared legend above charts
        st.markdown(
            f"<div style='display:flex;justify-content:center;gap:28px;margin:6px 0 10px'>"
            f"<span style='display:flex;align-items:center;gap:6px'>"
            f"<span style='display:inline-block;width:28px;height:3px;background:{COL_A}'></span>"
            f"<span style='font-size:12px;color:{COL_A};font-weight:600'>{name_a.split(',')[0]}</span></span>"
            f"<span style='display:flex;align-items:center;gap:6px'>"
            f"<span style='display:inline-block;width:28px;height:3px;background:{COL_B};border-top:2px dashed {COL_B}'></span>"
            f"<span style='font-size:12px;color:{COL_B};font-weight:600'>{name_b.split(',')[0]}</span></span>"
            f"<span style='display:flex;align-items:center;gap:6px'>"
            f"<span style='display:inline-block;width:28px;height:3px;background:rgba(255,255,255,0.8)'></span>"
            f"<span style='font-size:12px;color:rgba(200,200,200,0.7)'>Tour avg</span></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        prox_chart_cols = st.columns(4, gap="medium")

        for ci, (blabel, bprefix, fixed_max) in enumerate(FW_BUCKETS_PROX):
            pa = _get_prox(approach_skill_df, dg_a, bprefix)
            pb = _get_prox(approach_skill_df, dg_b, bprefix)
            pt = _tour_avg_prox(approach_skill_df, bprefix)

            # Cap any value that exceeds our fixed ceiling so ring stays inside chart
            def _cap(v): return min(v, fixed_max * 0.97) if np.isfinite(v) else v

            pa, pb, pt = _cap(pa), _cap(pb), _cap(pt)

            with prox_chart_cols[ci]:
                st.markdown(
                    f"<div style='text-align:center;font-size:13px;font-weight:700;"
                    f"color:rgba(220,220,220,0.85);margin-bottom:4px'>{blabel}</div>",
                    unsafe_allow_html=True,
                )

                all_vals = [v for v in [pa, pb, pt] if np.isfinite(v)]
                if not all_vals:
                    st.caption("No data")
                    continue

                fig_prox = go.Figure()
                axis_range = [-fixed_max * 1.08, fixed_max * 1.08]

                # ── Green surface fill ─────────────────────────────────
                gx, gy = _ring_xy(fixed_max * 1.0)
                fig_prox.add_trace(go.Scatter(
                    x=gx, y=gy,
                    fill="toself",
                    fillcolor="rgba(34,85,34,0.35)",
                    line=dict(color="rgba(80,140,80,0.5)", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))

                # ── Reference distance label (ft) ──────────────────────
                fig_prox.add_annotation(
                    x=0, y=fixed_max * 0.88,
                    text=f"<span style='font-size:9px;color:rgba(160,200,160,0.6)'>60 ft</span>",
                    showarrow=False, xanchor="center",
                )

                # ── Tour average ring ──────────────────────────────────
                if np.isfinite(pt):
                    tx, ty = _ring_xy(pt)
                    fig_prox.add_trace(go.Scatter(
                        x=tx, y=ty,
                        fill="toself",
                        fillcolor="rgba(255,255,255,0.08)",
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.85)", width=3),
                        name="Tour avg",
                        showlegend=(ci == 0),
                        hovertemplate=f"Tour median: <b>{pt:.1f} ft</b><extra></extra>",
                    ))

                # ── Player B ring ──────────────────────────────────────
                if np.isfinite(pb):
                    bx, by = _ring_xy(pb)
                    fig_prox.add_trace(go.Scatter(
                        x=bx, y=by,
                        fill="toself",
                        fillcolor=COL_B_FILL,
                        line=dict(color=COL_B, width=2.5, dash="dash"),
                        name=name_b,
                        showlegend=(ci == 0),
                        hovertemplate=f"<b>{name_b}</b>: <b>{pb:.1f} ft</b><extra></extra>",
                    ))

                # ── Player A ring ──────────────────────────────────────
                if np.isfinite(pa):
                    ax, ay = _ring_xy(pa)
                    fig_prox.add_trace(go.Scatter(
                        x=ax, y=ay,
                        fill="toself",
                        fillcolor=COL_A_FILL,
                        line=dict(color=COL_A, width=2.5),
                        name=name_a,
                        showlegend=(ci == 0),
                        hovertemplate=f"<b>{name_a}</b>: <b>{pa:.1f} ft</b><extra></extra>",
                    ))

                # ── Hole dot ───────────────────────────────────────────
                fig_prox.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode="markers",
                    marker=dict(size=10, color="white", symbol="circle"),
                    showlegend=False, hoverinfo="skip",
                ))

                # ── Radius lines with labels ───────────────────────────
                # Player A radius line — points right (positive x axis)
                if np.isfinite(pa):
                    fig_prox.add_shape(
                        type="line", x0=0, y0=0, x1=pa, y1=0,
                        line=dict(color=COL_A, width=1.5, dash="dot"),
                    )
                    fig_prox.add_annotation(
                        x=pa + fixed_max * 0.04, y=0,
                        text=f"<b>{pa:.1f} ft</b>",
                        showarrow=False, xanchor="left",
                        font=dict(size=10, color=COL_A),
                    )

                # Player B radius line — points left (negative x axis)
                if np.isfinite(pb):
                    fig_prox.add_shape(
                        type="line", x0=0, y0=0, x1=-pb, y1=0,
                        line=dict(color=COL_B, width=1.5, dash="dot"),
                    )
                    fig_prox.add_annotation(
                        x=-pb - fixed_max * 0.04, y=0,
                        text=f"<b>{pb:.1f} ft</b>",
                        showarrow=False, xanchor="right",
                        font=dict(size=10, color=COL_B),
                    )

                # Tour avg radius line — points down
                if np.isfinite(pt):
                    fig_prox.add_shape(
                        type="line", x0=0, y0=0, x1=0, y1=-pt,
                        line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dot"),
                    )
                    fig_prox.add_annotation(
                        x=0, y=-pt - fixed_max * 0.06,
                        text=f"<b>{pt:.1f} ft</b>",
                        showarrow=False, xanchor="center",
                        font=dict(size=10, color="rgba(255,255,255,0.7)"),
                    )

                fig_prox.update_layout(
                    xaxis=dict(
                        range=axis_range, showgrid=False, zeroline=False,
                        showticklabels=False, scaleanchor="y",
                    ),
                    yaxis=dict(
                        range=axis_range, showgrid=False, zeroline=False,
                        showticklabels=False,
                    ),
                    showlegend=False,
                    height=300,
                    margin=dict(l=5, r=5, t=5, b=5),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_prox, use_container_width=True)

                # ── Stats below chart ──────────────────────────────────
                def _prox_txt(v, ref):
                    if not np.isfinite(v):
                        return "—"
                    diff = v - ref
                    arrow = "▲" if diff > 0 else "▼"
                    color = "#ef4444" if diff > 0 else "#22c55e"
                    sign  = "+" if diff > 0 else ""
                    return (
                        f"<span style='font-size:13px;font-weight:700'>{v:.1f} ft</span> "
                        f"<span style='font-size:11px;color:{color}'>{arrow}{sign}{diff:.1f}</span>"
                    )

                rows_html = ""
                for name_t, val, col_t in [(name_a, pa, COL_A), (name_b, pb, COL_B)]:
                    label = name_t.split(",")[0]
                    val_html = _prox_txt(val, pt) if np.isfinite(pt) else (f"{val:.1f} ft" if np.isfinite(val) else "—")
                    rows_html += (
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;margin:2px 0'>"
                        f"<span style='font-size:11px;color:{col_t};font-weight:600'>{label}</span>"
                        f"<span>{val_html}</span></div>"
                    )
                if np.isfinite(pt):
                    rows_html += (
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;margin:3px 0;border-top:1px solid "
                        f"rgba(255,255,255,0.08);padding-top:4px'>"
                        f"<span style='font-size:11px;color:rgba(200,200,200,0.5)'>Tour avg</span>"
                        f"<span style='font-size:12px;color:rgba(200,200,200,0.5)'>{pt:.1f} ft</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.03);border-radius:6px;"
                    f"padding:8px 10px'>{rows_html}</div>",
                    unsafe_allow_html=True,
                )


    # ==================================================================
    # 8. RECENT RESULTS (compact)
    # ==================================================================
    st.divider()
    st.subheader("Recent Tournaments")

    tA = build_last_n_events_table(rounds_df, dg_a, n=10, date_max=cutoff_dt)
    tB = build_last_n_events_table(rounds_df, dg_b, n=10, date_max=cutoff_dt)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"#### {name_a}")
        st.dataframe(
            tA[["Event", "Finish", "SG Total", "Year"]] if not tA.empty else tA,
            use_container_width=True, hide_index=True,
        )
    with right:
        st.markdown(f"#### {name_b}")
        st.dataframe(
            tB[["Event", "Finish", "SG Total", "Year"]] if not tB.empty else tB,
            use_container_width=True, hide_index=True,
        )
