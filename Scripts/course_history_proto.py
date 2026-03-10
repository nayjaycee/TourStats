"""
Course History Tab - Redesigned (DEMO VERSION)
More visual, less spreadsheet-heavy, easier to digest
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


def render_course_history_demo(
    course_num,
    rounds_df,
    all_players,
    field_ids,
    cutoff_dt,
    season_year,
    build_course_history_func,  # Pass the function as parameter
    ev_2017_2023=None,          # Kept for signature compatibility but not used
    id_to_img: Optional[dict] = None,
    name_to_img: Optional[dict] = None
):
    """
    Demo Course History tab - visual and digestible.
    Uses rounds_df (combined_rounds_all_2017_2026.csv) as the single data source for all years.
    """

    st.title("Course History")

    if course_num is None:
        st.info("No course_num found for this schedule row.")
        return

    # ── helpers ──────────────────────────────────────────────────────────────
    def _course_rounds(df, cn):
        """All rows for this course from the combined rounds table."""
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df["course_num"] == cn].copy()

    def _one_row_per_player_event(df):
        """
        Deduplicate to one row per (dg_id, event_id / year) by taking round_num == 1.
        Falls back to first row if round_num not present.
        """
        if "round_num" in df.columns:
            out = df[pd.to_numeric(df["round_num"], errors="coerce") == 1].copy()
            if len(out) == 0:
                out = df.groupby(["dg_id", "year"], as_index=False).first()
        else:
            out = df.groupby(["dg_id", "year"], as_index=False).first()
        return out

    # ── get effective player IDs ──────────────────────────────────────────────
    effective_base_ids = []
    if field_ids and len(field_ids) > 0:
        effective_base_ids = [int(x) for x in field_ids if pd.notna(x)]
    else:
        if all_players is not None and "dg_id" in all_players.columns:
            effective_base_ids = (
                pd.to_numeric(all_players["dg_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

    if not effective_base_ids:
        st.error("No players found")
        return

    # ── build course history via the passed function ──────────────────────────
    with st.spinner("Loading course history..."):
        course_hist, horses = build_course_history_func(
            course_num=int(course_num),
            base_ids=effective_base_ids,
            rounds_df=rounds_df,
            cutoff_dt=cutoff_dt,
            season_year=season_year,
            years_back=9,
        )

    # ── pre-compute wins directly from rounds_df (single source of truth) ─────
    cr_all = _course_rounds(rounds_df, int(course_num))
    # Cancelled/incomplete events where finish_num == 1 should not count as wins
    CANCELLED_EVENT_EXCLUSIONS = {
        # (player_name_substring, year) — The Players 2020 was cancelled mid-round (COVID)
        ("Matsuyama", 2020),
    }

    wins_by_player = {}
    if not cr_all.empty and "finish_num" in cr_all.columns and "player_name" in cr_all.columns:
        one_row = _one_row_per_player_event(cr_all)
        one_row["finish_num"] = pd.to_numeric(one_row["finish_num"], errors="coerce")
        wins_df = one_row[one_row["finish_num"] == 1].copy()

        # Remove cancelled event entries
        for name_substr, excl_year in CANCELLED_EVENT_EXCLUSIONS:
            wins_df = wins_df[
                ~(
                    wins_df["player_name"].str.contains(name_substr, na=False)
                    & (wins_df["year"] == excl_year)
                )
            ]

        wins_by_player = wins_df.groupby("player_name").size().to_dict()

    # =========================================================================
    # SECTION 1: TOP 5 COURSE HORSES
    # =========================================================================
    if horses is not None and not horses.empty:
        horses_show = horses.copy()
        horses_show["SG"] = pd.to_numeric(horses_show.get("SG", np.nan), errors="coerce")
        horses_show["ROUNDS"] = pd.to_numeric(horses_show.get("ROUNDS", 0), errors="coerce").fillna(0)

        top5 = (
            horses_show
            .sort_values(["SG", "ROUNDS"], ascending=[False, False], na_position="last")
            .head(5)
            .copy()
        )

        st.markdown("### Course Horses")
        st.markdown("")

        cols = st.columns(5, gap="large")
        for i, (_, r) in enumerate(top5.iterrows()):
            with cols[i]:
                dg_id = None
                if "dg_id" in top5.columns and pd.notna(r.get("dg_id")):
                    try:
                        dg_id = int(r.get("dg_id"))
                    except Exception:
                        dg_id = None

                name = str(r.get("PLAYER", "")) if pd.notna(r.get("PLAYER", "")) else ""

                img = None
                if id_to_img and dg_id:
                    img = id_to_img.get(dg_id)
                if img is None and name_to_img and name:
                    img = name_to_img.get(name)
                if img:
                    st.image(img, use_container_width=True)

                sg_val = r.get("SG")
                sg_txt = f"{float(sg_val):+.1f}" if pd.notna(sg_val) else ""
                rounds_val = r.get("ROUNDS", 0)

                st.markdown(
                    f"<div style='text-align: center; margin-top: 8px;'>"
                    f"<div style='font-weight: 700; font-size: 14px;'>{name}</div>"
                    f"<div style='font-size: 20px; font-weight: 800; color: #00CC96; margin: 4px 0;'>{sg_txt}</div>"
                    f"<div style='font-size: 11px; color: #888;'>{int(rounds_val)} rounds</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # =========================================================================
    # SECTION 2: COURSE PERFORMANCE RANKINGS
    # =========================================================================
    st.markdown("### Course Performance Rankings")
    st.caption("How players in this week's field have performed at this course")

    if course_hist is not None and not course_hist.empty:
        ch_stats = course_hist.copy()
        ch_stats["SG"] = pd.to_numeric(ch_stats.get("SG", 0), errors="coerce").fillna(0)
        ch_stats["ROUNDS"] = pd.to_numeric(ch_stats.get("ROUNDS", 0), errors="coerce").fillna(0)

        # Attach win counts from rounds_df directly
        ch_stats["wins"] = ch_stats["PLAYER"].map(wins_by_player).fillna(0).astype(int)

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("#### Best Average SG")
            st.caption("Highest strokes gained at this course")
            qualified = ch_stats[ch_stats["ROUNDS"] >= 4].copy()
            top_sg = qualified.nlargest(10, "SG")[["PLAYER", "SG", "ROUNDS"]]

            for idx, (_, player) in enumerate(top_sg.iterrows(), 1):
                st.markdown(
                    f"""
                    <div style='padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,0.1);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: #888; margin-right: 8px;'>{idx}.</span>
                                <span style='font-weight: 600;'>{player['PLAYER']}</span>
                            </div>
                            <div>
                                <span style='color: #00CC96; font-weight: 700; margin-right: 8px;'>{player['SG']:+.1f}</span>
                                <span style='color: #888; font-size: 11px;'>({int(player['ROUNDS'])}R)</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown("#### Most Experienced")
            st.caption("Most rounds played at this course")
            most_rounds = ch_stats.nlargest(10, "ROUNDS")[["PLAYER", "ROUNDS", "SG"]]

            for idx, (_, player) in enumerate(most_rounds.iterrows(), 1):
                sg_val = player["SG"]
                sg_color = "#00CC96" if sg_val > 0 else "#EF553B" if sg_val < 0 else "#888"

                st.markdown(
                    f"""
                    <div style='padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,0.1);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: #888; margin-right: 8px;'>{idx}.</span>
                                <span style='font-weight: 600;'>{player['PLAYER']}</span>
                            </div>
                            <div>
                                <span style='color: #636EFA; font-weight: 700; margin-right: 8px;'>{int(player['ROUNDS'])} rounds</span>
                                <span style='color: {sg_color}; font-size: 11px;'>({sg_val:+.1f})</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col3:
            st.markdown("#### Course Winners")
            st.caption("Players with wins at this course")

            winners = ch_stats[ch_stats["wins"] > 0].nlargest(10, "wins")[["PLAYER", "wins", "SG"]]

            if len(winners) > 0:
                for idx, (_, player) in enumerate(winners.iterrows(), 1):
                    win_text = "win" if player["wins"] == 1 else "wins"
                    st.markdown(
                        f"""
                        <div style='padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,0.1);'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <span style='color: #888; margin-right: 8px;'>{idx}.</span>
                                    <span style='font-weight: 600;'>{player['PLAYER']}</span>
                                </div>
                                <div>
                                    <span style='color: #FFD700; font-weight: 700; margin-right: 8px;'>{int(player['wins'])} {win_text}</span>
                                    <span style='color: #00CC96; font-size: 11px;'>({player['SG']:+.1f})</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No players in field have won here")

    st.divider()

    # =========================================================================
    # SECTION 3: PAST PERFORMANCE ANALYSIS
    # Uses rounds_df for ALL years — no ev_2017_2023 branch
    # =========================================================================
    st.markdown("### Past Performance Analysis")
    st.caption("Player positioning vs actual finish at this course")

    col_sel1, col_sel2, col_spacer = st.columns([1, 1, 2])

    with col_sel1:
        perf_window = st.selectbox(
            "Stats Window",
            options=["L12", "L24", "L36", "L60"],
            index=1,
            key="perf_window",
        )

    with col_sel2:
        year_cols = []
        if course_hist is not None and not course_hist.empty:
            year_cols = sorted(
                [c for c in course_hist.columns if str(c).isdigit()], reverse=True
            )

        perf_year = None
        if year_cols:
            perf_year = st.selectbox(
                "Tournament Year",
                options=year_cols[:5],   # show up to 5 most recent years
                index=0,
                key="perf_year",
            )

    if perf_year is not None and not cr_all.empty:
        window_num = int(perf_window[1:])
        event_year = int(perf_year)

        # Approximate cutoff — day before the event year ends
        event_cutoff_date = pd.Timestamp(f"{event_year}-01-01")

        # Get finishes for this course + year (one row per player)
        cr_year = cr_all[cr_all["year"] == event_year].copy()
        if not cr_year.empty:
            cr_year_one = _one_row_per_player_event(cr_year)
            cr_year_one["finish_num"] = pd.to_numeric(cr_year_one["finish_num"], errors="coerce")

            positioning_data = []

            for _, player_row in cr_year_one.iterrows():
                player_name = player_row.get("player_name", "Unknown")
                dg_id = player_row.get("dg_id")
                finish_num = player_row.get("finish_num")
                finish_text = player_row.get("fin_text", "")

                if pd.isna(finish_num) or pd.isna(dg_id):
                    continue

                fin_num = int(finish_num) if finish_text != "CUT" else 999
                dg_id = int(dg_id)

                # Rolling stats from rounds_df BEFORE this event
                rounds_df["round_date"] = pd.to_datetime(rounds_df["round_date"], errors="coerce")
                prior = (
                    rounds_df[
                        (rounds_df["dg_id"] == dg_id)
                        & (rounds_df["round_date"] < event_cutoff_date)
                    ]
                    .sort_values("round_date", ascending=False)
                    .head(window_num)
                )

                if prior.empty:
                    continue

                sg_ott  = pd.to_numeric(prior["sg_ott"],  errors="coerce").mean()
                sg_app  = pd.to_numeric(prior["sg_app"],  errors="coerce").mean()
                sg_arg  = pd.to_numeric(prior["sg_arg"],  errors="coerce").mean()
                sg_putt = pd.to_numeric(prior["sg_putt"], errors="coerce").mean()

                if not (pd.notna(sg_ott) or pd.notna(sg_app) or pd.notna(sg_arg) or pd.notna(sg_putt)):
                    continue

                t2g        = (sg_ott  if pd.notna(sg_ott)  else 0) + (sg_app  if pd.notna(sg_app)  else 0)
                short_game = (sg_arg  if pd.notna(sg_arg)  else 0) + (sg_putt if pd.notna(sg_putt) else 0)

                positioning_data.append({
                    "player":       player_name,
                    "t2g":          t2g,
                    "short_game":   short_game,
                    "finish_text":  finish_text,
                    "finish_num":   fin_num,
                })

            if positioning_data:
                pos_df = pd.DataFrame(positioning_data)

                def get_finish_color(fn):
                    if fn == 1:    return "#FFD700"
                    if fn <= 3:    return "#C0C0C0"
                    if fn <= 5:    return "#CD7F32"
                    if fn <= 10:   return "#00CC96"
                    if fn <= 25:   return "#66D9A6"
                    if fn == 999:  return "#EF553B"
                    return "#888888"

                pos_df["color"] = pos_df["finish_num"].apply(get_finish_color)
                top5_df  = pos_df[pos_df["finish_num"] <= 5].copy()
                rest_df  = pos_df[pos_df["finish_num"] > 5].copy()

                fig_pos = go.Figure()

                fig_pos.add_trace(go.Scatter(
                    x=rest_df["t2g"], y=rest_df["short_game"],
                    mode="markers",
                    marker=dict(size=10, color=rest_df["color"], line=dict(width=1, color="white")),
                    text=rest_df["player"],
                    customdata=rest_df[["finish_text", "t2g", "short_game"]],
                    hovertemplate="<b>%{text}</b><br>Finish: %{customdata[0]}<br>Tee-to-Green: %{customdata[1]:.2f}<br>Short Game: %{customdata[2]:.2f}<extra></extra>",
                    showlegend=False,
                ))

                if len(top5_df) > 0:
                    def short_name(full_name):
                        return str(full_name).strip().split()[0].rstrip(",")

                    top5_df["label"] = top5_df.apply(
                        lambda r: f"{r['finish_text']} {short_name(r['player'])}", axis=1
                    )

                    fig_pos.add_trace(go.Scatter(
                        x=top5_df["t2g"], y=top5_df["short_game"],
                        mode="markers+text",
                        marker=dict(size=14, color=top5_df["color"], line=dict(width=2, color="white")),
                        text=top5_df["label"],
                        textposition="top center",
                        textfont=dict(size=11, color="white"),
                        customdata=top5_df[["finish_text", "t2g", "short_game", "player"]],
                        hovertemplate="<b>%{customdata[3]}</b><br>Finish: %{customdata[0]}<br>Tee-to-Green: %{customdata[1]:.2f}<br>Short Game: %{customdata[2]:.2f}<extra></extra>",
                        showlegend=False,
                    ))

                fig_pos.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
                fig_pos.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

                max_x = pos_df["t2g"].max()
                max_y = pos_df["short_game"].max()
                min_x = pos_df["t2g"].min()
                min_y = pos_df["short_game"].min()

                for txt, ax, ay, align in [
                    ("<b>Elite All-Around</b>",  max_x * 0.85,  max_y * 0.9,  "right"),
                    ("<b>Short Game Gods</b>",   min_x * 0.85,  max_y * 0.9,  "left"),
                    ("<b>Ball Strikers</b>",     max_x * 0.85,  min_y * 0.9,  "right"),
                ]:
                    fig_pos.add_annotation(
                        x=ax, y=ay, text=txt, showarrow=False,
                        font=dict(size=10, color="rgba(255,255,255,0.4)"), align=align,
                    )

                fig_pos.update_layout(
                    xaxis_title="Tee-to-Green (OTT + APP)",
                    yaxis_title="Short Game (ARG + PUTT)",
                    height=600, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False),
                    yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False),
                    hovermode="closest", margin=dict(t=40, b=40, l=40, r=40),
                )

                st.plotly_chart(fig_pos, use_container_width=True)

                col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
                with col_leg1: st.markdown("**Gold** = Winner")
                with col_leg2: st.markdown("**Silver** = Top 3")
                with col_leg3: st.markdown("**Green** = Top 10")
                with col_leg4: st.markdown("**Red** = Missed Cut")
            else:
                st.info(f"Insufficient SG data for {perf_year} with {perf_window} window")
        else:
            st.info(f"No rounds found for {perf_year} at this course")

    st.divider()

    # =========================================================================
    # SECTION 4: RECENT TOURNAMENT RESULTS
    # Uses rounds_df for ALL years — single source of truth
    # =========================================================================
    st.markdown("### Recent Tournament Results")
    st.caption("Actual top 10 finishers from the last 3 years")

    years_to_show = [season_year - 1, season_year - 2, season_year - 3]
    year_columns  = st.columns(len(years_to_show), gap="large")

    for idx, year in enumerate(years_to_show):
        with year_columns[idx]:
            st.markdown(f"#### {year}")

            top10_data = []

            if not cr_all.empty and "year" in cr_all.columns:
                cr_year = cr_all[cr_all["year"] == year].copy()

                if not cr_year.empty:
                    # One row per player (deduplicate rounds)
                    one_row = _one_row_per_player_event(cr_year)
                    one_row["finish_num"] = pd.to_numeric(one_row["finish_num"], errors="coerce")

                    # Drop missed cuts and WDs for top-10 display
                    one_row = one_row[one_row["finish_num"].notna()]
                    top10 = one_row.nsmallest(10, "finish_num")

                    for _, player in top10.iterrows():
                        top10_data.append({
                            "name":         player.get("player_name", "Unknown"),
                            "finish_text":  player.get("fin_text", ""),
                            "finish_num":   player.get("finish_num", 999),
                        })

            if top10_data:
                for player_data in top10_data:
                    fn = player_data["finish_num"]
                    if fn == 1:      color = "#FFD700"
                    elif fn <= 3:    color = "#C0C0C0"
                    elif fn <= 10:   color = "#00CC96"
                    else:            color = "#888"

                    st.markdown(
                        f"""
                        <div style='padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,0.08);'>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='font-weight: 600; font-size: 13px;'>{player_data['name']}</span>
                                <span style='color: {color}; font-weight: 700; font-size: 13px;'>{player_data['finish_text']}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"No data for {year}")

    st.divider()

    # =========================================================================
    # SECTION 5: COMPLETE COURSE HISTORY TABLE
    # =========================================================================
    st.markdown("### Complete Course History")
    st.caption("Full finish history for all players in this week's field")

    if course_hist is not None and not course_hist.empty:
        ch = course_hist.copy()
        year_cols_all = sorted([c for c in ch.columns if str(c).isdigit()], reverse=True)
        cols = ["PLAYER"] + year_cols_all + ["ROUNDS", "SG"]
        cols = [c for c in cols if c in ch.columns]

        for col in year_cols_all:
            if col in ch.columns:
                ch[col] = ch[col].fillna("DNP")

        def style_history_table(df):
            def color_finish(val):
                if pd.isna(val) or val == "":
                    return "background-color: #111111; color: #444444;"
                if val in ["DNP", "WD"]:
                    return "background-color: #111111; color: #444444;"
                if val == "CUT":
                    return "background-color: rgba(239,85,59,0.3); color: #EF553B;"
                try:
                    fin_num = int(str(val).replace("T", ""))
                except Exception:
                    return ""
                if fin_num == 1:
                    return "background-color: rgba(255,215,0,0.3); color: #FFD700; font-weight: 700;"
                elif fin_num <= 3:
                    return "background-color: rgba(192,192,192,0.3); color: #C0C0C0; font-weight: 600;"
                elif fin_num <= 5:
                    return "background-color: rgba(205,127,50,0.3); color: #CD7F32; font-weight: 500;"
                elif fin_num <= 10:
                    return "background-color: rgba(0,204,150,0.3); color: #00CC96;"
                elif fin_num <= 25:
                    return "background-color: rgba(0,204,150,0.15); color: #00CC96;"
                return ""

            styled = df.style
            for col in df.columns:
                if str(col).isdigit():
                    styled = styled.applymap(color_finish, subset=[col])
            if "SG" in df.columns:
                styled = styled.background_gradient(subset=["SG"], cmap="RdYlGn", vmin=-2, vmax=2)
            return styled

        st.dataframe(
            style_history_table(ch[cols]),
            use_container_width=True,
            hide_index=True,
            height=700,
        )