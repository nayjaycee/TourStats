"""
Strokes Gained Tab - Production Version
Customizable stat analysis with category breakdowns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


def calculate_rolling_stats(rounds_df, dg_id, stat_cols, windows=[12, 24, 36, 60]):
    """Calculate rolling averages for a player across multiple windows"""
    date_col = 'round_date' if 'round_date' in rounds_df.columns else 'event_completed'

    player_rounds = rounds_df[rounds_df['dg_id'] == dg_id].copy()
    player_rounds = player_rounds.sort_values(date_col, ascending=False)

    results = {}

    for window in windows:
        window_rounds = player_rounds.head(window)

        for stat in stat_cols:
            if stat in window_rounds.columns:
                vals = pd.to_numeric(window_rounds[stat], errors='coerce').dropna()
                if len(vals) >= 3:
                    results[f"{stat}_L{window}"] = vals.mean()
                else:
                    results[f"{stat}_L{window}"] = None
            else:
                results[f"{stat}_L{window}"] = None

    return results


def get_form_indicator(trend_value):
    """Return clean form indicator"""
    if trend_value > 0.3:
        return "▲ Trending Up", "#00CC96"
    elif trend_value < -0.3:
        return "▼ Trending Down", "#EF553B"
    else:
        return "■ Steady", "#636EFA"


def render_production_sg_tab(rounds_df, field_ids: Optional[List[int]] = None, all_players=None,
                             id_to_img: Optional[dict] = None, name_to_img: Optional[dict] = None):
    """
    Production Strokes Gained tab with customizable stat analysis.
    """

    try:
        st.title("Strokes Gained Analysis")

        # ===== PRIMARY STAT SELECTOR =====
        col_stat, col_window, col_spacer = st.columns([2, 1, 2])

        with col_stat:
            primary_stat = st.selectbox(
                "Primary Metric",
                options=[
                    'Total SG',
                    'Elite Finish Score (L36)',  # Special volatility penalty
                    'Driving (OTT)',
                    'Approach (APP)',
                    'Short Game (ARG)',
                    'Putting'
                ],
                index=0
            )

        with col_window:
            # Disable window selector if Elite Finish Score is selected
            if primary_stat == 'Elite Finish Score (L36)':
                primary_window = 'L36'
                st.selectbox(
                    "Window",
                    options=['L36'],
                    index=0,
                    disabled=True
                )
            else:
                primary_window = st.selectbox(
                    "Window",
                    options=['L12', 'L24', 'L36', 'L60'],
                    index=0
                )

        # Map selection to stat column
        stat_map = {
            'Total SG': 'sg_total',
            'Elite Finish Score (L36)': 'elite_finish',
            'Driving (OTT)': 'sg_ott',
            'Approach (APP)': 'sg_app',
            'Short Game (ARG)': 'sg_arg',
            'Putting': 'sg_putt'
        }

        primary_stat_col = stat_map[primary_stat]
        window_num = int(primary_window[1:])  # Extract number from L12, L24, etc.

        # Filter to field
        if field_ids and len(field_ids) > 0:
            players_list = field_ids
        else:
            if all_players is not None and 'dg_id' in all_players.columns:
                players_list = pd.to_numeric(all_players['dg_id'], errors='coerce').dropna().astype(int).tolist()
            else:
                players_list = rounds_df['dg_id'].unique().tolist()

        # Calculate stats for all players
        with st.spinner("Calculating player statistics..."):
            stat_cols = ['sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt']

            all_player_stats = []

            for dg_id in players_list:
                if pd.isna(dg_id):
                    continue

                dg_id = int(dg_id)
                stats = calculate_rolling_stats(rounds_df, dg_id, stat_cols, windows=[12, 24, 36, 60])

                # For Elite Finish Score, we need sg_total_L36 to exist
                # But don't filter here - we'll calculate it separately later
                should_include = False

                if primary_stat == 'Elite Finish Score (L36)':
                    # Just need ANY data for this player
                    if any(v is not None for v in stats.values()):
                        should_include = True
                else:
                    # For regular stats, check if the specific stat exists
                    if stats.get(f'{primary_stat_col}_{primary_window}') is not None:
                        should_include = True

                if not should_include:
                    continue

                # Get player name
                player_name = None
                if all_players is not None and 'player_name' in all_players.columns:
                    player_row = all_players[all_players['dg_id'] == dg_id]
                    if len(player_row) > 0:
                        player_name = player_row.iloc[0]['player_name']

                if player_name is None:
                    player_rounds = rounds_df[rounds_df['dg_id'] == dg_id]
                    if len(player_rounds) > 0 and 'player_name' in player_rounds.columns:
                        player_name = player_rounds.iloc[0]['player_name']
                    else:
                        player_name = f"Player {dg_id}"

                all_player_stats.append({
                    'player_name': player_name,
                    'dg_id': dg_id,
                    **stats
                })

            if len(all_player_stats) == 0:
                st.warning("Insufficient data to calculate stats")
                return

            stats_df = pd.DataFrame(all_player_stats)

            # Calculate derived stats
            for window in [12, 24, 36, 60]:
                if f'sg_ott_L{window}' in stats_df.columns and f'sg_app_L{window}' in stats_df.columns:
                    stats_df[f'ball_striking_L{window}'] = (
                            stats_df[f'sg_ott_L{window}'].fillna(0) +
                            stats_df[f'sg_app_L{window}'].fillna(0)
                    )

            # Calculate Elite Finish Score (L36 volatility penalty)
            # Check what SG total column exists
            sg_total_col = None
            for possible_col in ['sg_total', 'SG_total', 'sg_total_raw', 'strokes_gained_total']:
                if possible_col in rounds_df.columns:
                    sg_total_col = possible_col
                    break

            if sg_total_col is None:
                # If no raw sg_total, we can't calculate Elite Finish Score
                # Just use the L36 average we already calculated
                if 'sg_total_L36' in stats_df.columns:
                    stats_df['elite_finish_L36'] = stats_df['sg_total_L36']
                else:
                    stats_df['elite_finish_L36'] = None
            else:
                # Calculate with volatility penalty
                elite_scores = []
                for _, player in stats_df.iterrows():
                    dg_id = int(player['dg_id'])
                    player_rounds = rounds_df[rounds_df['dg_id'] == dg_id].copy()

                    if len(player_rounds) == 0:
                        elite_scores.append(None)
                        continue

                    date_col = 'round_date' if 'round_date' in player_rounds.columns else 'event_completed'
                    player_rounds = player_rounds.sort_values(date_col, ascending=False).head(36)

                    vals = pd.to_numeric(player_rounds[sg_total_col], errors='coerce').dropna()
                    if len(vals) >= 5:
                        mean_val = vals.mean()
                        std_val = vals.std()
                        elite_score = mean_val - (0.3 * std_val)
                        elite_scores.append(elite_score)
                    else:
                        elite_scores.append(None)

                stats_df['elite_finish_L36'] = elite_scores

            # Debug: count how many valid scores
            valid_count = sum(1 for x in stats_df['elite_finish_L36'] if pd.notna(x))

            # Sort by selected stat
            if primary_stat == 'Elite Finish Score (L36)':
                sort_col = 'elite_finish_L36'
                # Only filter out None values AFTER we have the full stats_df
                valid_players = stats_df[stats_df['elite_finish_L36'].notna()].copy()

                if len(valid_players) == 0:
                    st.error(f"Cannot calculate Elite Finish Score (L36).")
                    st.info(
                        f"Checked for sg_total column in rounds data: {sg_total_col if sg_total_col else 'NOT FOUND'}")
                    st.info(
                        f"Available columns in rounds_df: {', '.join([c for c in rounds_df.columns if 'sg' in c.lower()][:10])}")
                    st.info("Try selecting 'Total SG' with window 'L36' instead.")
                    return

                stats_df = valid_players
            else:
                sort_col = f'{primary_stat_col}_{primary_window}'

            stats_df = stats_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        # Write top 1 and top 2 to session state so H2H and Player Deep Dive use them as defaults
        if len(stats_df) >= 1:
            st.session_state["weekly_top1_dg_id"] = int(stats_df.iloc[0]["dg_id"])
        if len(stats_df) >= 2:
            st.session_state["weekly_top2_dg_id"] = int(stats_df.iloc[1]["dg_id"])

        # ===== TOP 10 PERFORMERS WITH CATEGORY BREAKDOWN =====
        st.markdown("### Top Performers")
        st.caption(f"Based on {primary_stat} ({primary_window}) • {len(stats_df)} players")
        st.markdown("")

        top10 = stats_df.head(10)

        # Create 10 columns (2 rows of 5)
        for row_idx in range(2):
            cols = st.columns(5, gap="medium")
            start_idx = row_idx * 5
            end_idx = start_idx + 5

            for col_idx, (idx, player) in enumerate(top10.iloc[start_idx:end_idx].iterrows()):
                with cols[col_idx]:
                    rank = start_idx + col_idx + 1

                    # Get headshot
                    img_url = None
                    dg_id = int(player['dg_id'])
                    player_name = player['player_name']

                    if id_to_img and dg_id in id_to_img:
                        img_url = id_to_img[dg_id]
                    elif name_to_img and player_name in name_to_img:
                        img_url = name_to_img[player_name]

                    if img_url:
                        st.image(img_url, use_container_width=True)

                    # Rank and name
                    st.markdown(
                        f"<div style='text-align: center; font-size: 11px; color: #888; margin-top: 8px;'>#{rank}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='text-align: center; font-weight: 700; font-size: 13px; margin-bottom: 6px;'>{player_name}</div>",
                        unsafe_allow_html=True
                    )

                    # Primary stat - big number
                    primary_val = player[sort_col]
                    color = "#00CC96" if primary_val > 0 else "#EF553B"
                    st.markdown(
                        f"<div style='text-align: center; font-size: 22px; font-weight: 800; color: {color}; margin-bottom: 8px;'>{primary_val:+.2f}</div>",
                        unsafe_allow_html=True
                    )

                    # Category breakdown - color-coded mini bars (CENTERED AT ZERO)
                    categories = ['OTT', 'APP', 'ARG', 'PUTT']
                    cat_stats = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt']

                    for cat, stat in zip(categories, cat_stats):
                        val = player.get(f'{stat}_{primary_window}', 0) or 0

                        # Color based on value
                        if val > 0.5:
                            bar_color = "#00CC96"  # Strong green
                        elif val > 0:
                            bar_color = "#66D9A6"  # Light green
                        elif val > -0.5:
                            bar_color = "#FFA07A"  # Light red
                        else:
                            bar_color = "#EF553B"  # Strong red

                        # Bar fills FROM CENTER (50%)
                        # Positive: 50% + (value * scale), Negative: 50% - (abs(value) * scale)
                        scale = 25  # Each +1.0 = 25% of bar width
                        if val >= 0:
                            bar_start = 50
                            bar_width = min(val * scale, 50)  # Cap at 50% (full right)
                            bar_position = f"margin-left: {bar_start}%;"
                        else:
                            bar_width = min(abs(val) * scale, 50)  # Cap at 50% (full left)
                            bar_start = 50 - bar_width
                            bar_position = f"margin-left: {bar_start}%;"

                        st.markdown(
                            f"""
                            <div style='display: flex; align-items: center; margin: 2px 0; font-size: 10px;'>
                                <div style='width: 30px; text-align: right; margin-right: 4px; opacity: 0.7;'>{cat}</div>
                                <div style='flex: 1; background: rgba(128,128,128,0.2); height: 8px; border-radius: 4px; position: relative;'>
                                    <div style='position: absolute; left: 50%; width: 1px; height: 100%; background: rgba(255,255,255,0.3);'></div>
                                    <div style='background: {bar_color}; height: 100%; width: {bar_width:.1f}%; {bar_position}'></div>
                                </div>
                                <div style='width: 35px; text-align: left; margin-left: 4px; font-weight: 600; color: {bar_color};'>{val:+.1f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ===== KEY METRICS (4 COLUMNS - RIGHT AFTER TOP PERFORMERS) =====
        st.markdown("### Key Metrics")

        col_metric_selector, col_spacer2 = st.columns([2, 3])

        with col_metric_selector:
            metrics_stat = st.selectbox(
                "Analyze",
                options=[
                    'Total SG',
                    'Driving (OTT)',
                    'Approach (APP)',
                    'Short Game (ARG)',
                    'Putting'
                ],
                index=0,
                key='metrics_stat'
            )

        # Map without Elite Finish for metrics
        metrics_map = {
            'Total SG': 'sg_total',
            'Driving (OTT)': 'sg_ott',
            'Approach (APP)': 'sg_app',
            'Short Game (ARG)': 'sg_arg',
            'Putting': 'sg_putt'
        }

        metrics_stat_col = metrics_map[metrics_stat]
        metrics_sort_col = f'{metrics_stat_col}_{primary_window}'

        st.caption(f"Rankings based on {metrics_stat} ({primary_window})")
        st.markdown("")

        # 4 columns side by side - ALL ON ONE LINE
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("#### Top")
            st.caption("Highest averages")

            top_players = stats_df.nlargest(5, metrics_sort_col)
            for rank, (_, p) in enumerate(top_players.iterrows(), 1):
                val = p[metrics_sort_col]
                st.markdown(
                    f"""
                    <div style='padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.12);'>
                        <div style='font-weight: 700; font-size: 13px;'>{rank}. {p['player_name']}</div>
                        <div style='font-size: 12px; color: #00CC96; margin-top: 3px; font-weight: 600;'>{val:+.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            st.markdown("#### Bottom")
            st.caption("Needs work")

            bottom_players = stats_df.nsmallest(5, metrics_sort_col)
            for rank, (_, p) in enumerate(bottom_players.iterrows(), 1):
                val = p[metrics_sort_col]
                st.markdown(
                    f"""
                    <div style='padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.12);'>
                        <div style='font-weight: 700; font-size: 13px;'>{rank}. {p['player_name']}</div>
                        <div style='font-size: 12px; color: #EF553B; margin-top: 3px; font-weight: 600;'>{val:+.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col3:
            st.markdown("#### Consistent")
            st.caption("Low volatility")

            # Calculate std dev for selected stat
            temp_stats = []
            for dg_id in stats_df['dg_id'].head(30):
                player_rounds = rounds_df[rounds_df['dg_id'] == dg_id].copy()
                date_col = 'round_date' if 'round_date' in player_rounds.columns else 'event_completed'
                player_rounds = player_rounds.sort_values(date_col, ascending=False).head(window_num)

                if metrics_stat_col in player_rounds.columns:
                    vals = pd.to_numeric(player_rounds[metrics_stat_col], errors='coerce').dropna()
                    if len(vals) >= 5:
                        temp_stats.append({
                            'dg_id': dg_id,
                            'std': vals.std()
                        })

            if temp_stats:
                std_df = pd.DataFrame(temp_stats)
                std_df = std_df.sort_values('std')

                for rank, (_, row) in enumerate(std_df.head(5).iterrows(), 1):
                    player_row = stats_df[stats_df['dg_id'] == row['dg_id']].iloc[0]
                    st.markdown(
                        f"""
                        <div style='padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.12);'>
                            <div style='font-weight: 700; font-size: 13px;'>{rank}. {player_row['player_name']}</div>
                            <div style='font-size: 12px; color: #636EFA; margin-top: 3px; font-weight: 600;'>σ = {row['std']:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with col4:
            st.markdown("#### Volatile")
            st.caption("Boom or bust")

            if temp_stats:
                for rank, (_, row) in enumerate(std_df.tail(5).sort_values('std', ascending=False).iterrows(), 1):
                    player_row = stats_df[stats_df['dg_id'] == row['dg_id']].iloc[0]
                    st.markdown(
                        f"""
                        <div style='padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.12);'>
                            <div style='font-weight: 700; font-size: 13px;'>{rank}. {player_row['player_name']}</div>
                            <div style='font-size: 12px; color: #FFA500; margin-top: 3px; font-weight: 600;'>σ = {row['std']:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ===== RECENT FORM COMPARISON (CUSTOMIZABLE) =====
        st.markdown("### Recent Form Trends")

        col_form1, col_form2, col_form3 = st.columns(3)

        with col_form1:
            form_stat = st.selectbox(
                "Stat to Compare",
                options=[
                    'Total SG',
                    'Driving (OTT)',
                    'Approach (APP)',
                    'Short Game (ARG)',
                    'Putting'
                ],
                index=0,
                key='form_stat'
            )

        with col_form2:
            form_window1 = st.selectbox(
                "Recent Window",
                options=['L12', 'L24', 'L36'],
                index=0,
                key='form_w1'
            )

        with col_form3:
            form_window2 = st.selectbox(
                "Compare to",
                options=['L24', 'L36', 'L60'],
                index=0,
                key='form_w2'
            )

        form_stat_col = stat_map[form_stat]

        # Calculate form trend
        col1_name = f'{form_stat_col}_{form_window1}'
        col2_name = f'{form_stat_col}_{form_window2}'

        if col1_name in stats_df.columns and col2_name in stats_df.columns:
            stats_df['form_trend'] = stats_df[col1_name] - stats_df[col2_name]

            # Top 10 gainers + top 10 decliners
            movers = pd.concat([
                stats_df.nlargest(10, 'form_trend'),
                stats_df.nsmallest(10, 'form_trend')
            ])
            movers = movers.sort_values('form_trend', ascending=True)

            fig_trend = go.Figure()

            colors = ['#00CC96' if x > 0 else '#EF553B' for x in movers['form_trend']]

            fig_trend.add_trace(go.Bar(
                y=movers['player_name'],
                x=movers['form_trend'],
                orientation='h',
                marker_color=colors,
                text=movers['form_trend'].apply(lambda x: f"{x:+.2f}"),
                textposition='outside',
                textfont=dict(size=10)
            ))

            fig_trend.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

            fig_trend.update_layout(
                xaxis_title=f"{form_stat} Change ({form_window1} vs {form_window2})",
                yaxis_title="",
                height=600,  # Increased for 20 players
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Selected comparison windows not available")

        st.divider()

        # ===== PLAYER POSITIONING: TEE-TO-GREEN VS SHORT GAME =====
        st.markdown("### Player Strengths Map")
        st.caption(f"Tee-to-Green (OTT + APP) vs Short Game (ARG + PUTT) • Based on {primary_window}")

        # Calculate composite scores
        t2g_col = f'sg_ott_{primary_window}'
        app_col = f'sg_app_{primary_window}'
        arg_col = f'sg_arg_{primary_window}'
        putt_col = f'sg_putt_{primary_window}'

        if all(col in stats_df.columns for col in [t2g_col, app_col, arg_col, putt_col]):
            stats_df['tee_to_green'] = (
                    stats_df[t2g_col].fillna(0) + stats_df[app_col].fillna(0)
            )
            stats_df['short_game'] = (
                    stats_df[arg_col].fillna(0) + stats_df[putt_col].fillna(0)
            )

            # Create scatter plot
            fig_scatter = go.Figure()

            # Get top 10 performers (from top10 dataframe we already have)
            top10_ids = set(top10['dg_id'].tolist())

            # Split into two traces: top 10 with labels, everyone else without
            top10_data = stats_df[stats_df['dg_id'].isin(top10_ids)]
            others_data = stats_df[~stats_df['dg_id'].isin(top10_ids)]

            # Add everyone else first (no text labels)
            fig_scatter.add_trace(go.Scatter(
                x=others_data['tee_to_green'],
                y=others_data['short_game'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=others_data[sort_col],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title=primary_stat),
                    line=dict(width=0.5, color='white')
                ),
                text=others_data['player_name'],
                hovertemplate='<b>%{text}</b><br>' +
                              'Tee-to-Green: %{x:.2f}<br>' +
                              'Short Game: %{y:.2f}<br>' +
                              '<extra></extra>',
                showlegend=False
            ))

            # Add top 10 with text labels
            fig_scatter.add_trace(go.Scatter(
                x=top10_data['tee_to_green'],
                y=top10_data['short_game'],
                mode='markers+text',
                marker=dict(
                    size=10,  # Slightly larger
                    color=top10_data[sort_col],
                    colorscale='RdYlGn',
                    showscale=False,
                    line=dict(width=1, color='white')
                ),
                text=top10_data['player_name'],
                textposition='top center',
                textfont=dict(size=8, color='white'),
                hovertemplate='<b>%{text}</b><br>' +
                              'Tee-to-Green: %{x:.2f}<br>' +
                              'Short Game: %{y:.2f}<br>' +
                              '<extra></extra>',
                showlegend=False,
                cliponaxis=False  # Allow text to extend beyond plot area
            ))

            # Add quadrant lines at 0
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
            fig_scatter.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

            # Add quadrant labels in corners (away from data clustering)
            max_x = stats_df['tee_to_green'].max()
            max_y = stats_df['short_game'].max()
            min_x = stats_df['tee_to_green'].min()
            min_y = stats_df['short_game'].min()

            # Position labels at edges
            fig_scatter.add_annotation(
                x=max_x * 0.85, y=max_y * 0.9,
                text="<b>Elite All-Around</b>",
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.4)"),
                align="right"
            )

            fig_scatter.add_annotation(
                x=min_x * 0.85, y=max_y * 0.9,
                text="<b>Short Game God</b>",
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.4)"),
                align="left"
            )

            fig_scatter.add_annotation(
                x=max_x * 0.85, y=min_y * 0.9,
                text="<b>Ball Strikers</b>",
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.4)"),
                align="right"
            )

            fig_scatter.add_annotation(
                x=min_x * 0.85, y=min_y * 0.9,
                text="<b>Needs Work</b>",
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.4)"),
                align="left"
            )

            fig_scatter.update_layout(
                xaxis_title="Tee-to-Green (OTT + APP)",
                yaxis_title="Short Game (ARG + PUTT)",
                height=700,  # Increased from 600 for more space
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=False
                ),
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=False
                ),
                hovermode='closest',
                margin=dict(t=40, b=40, l=40, r=40)  # Add margins for text
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

            # Quadrant breakdown
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)

            elite = stats_df[(stats_df['tee_to_green'] > 0) & (stats_df['short_game'] > 0)]
            short_game_artists = stats_df[(stats_df['tee_to_green'] < 0) & (stats_df['short_game'] > 0)]
            ball_strikers = stats_df[(stats_df['tee_to_green'] > 0) & (stats_df['short_game'] < 0)]
            needs_work = stats_df[(stats_df['tee_to_green'] < 0) & (stats_df['short_game'] < 0)]

            with col_q1:
                st.markdown(f"**Elite All-Around**")
                st.caption(f"{len(elite)} players (+T2G, +Short)")

            with col_q2:
                st.markdown(f"**Short Game Artists**")
                st.caption(f"{len(short_game_artists)} players (-T2G, +Short)")

            with col_q3:
                st.markdown(f"**Ball Strikers**")
                st.caption(f"{len(ball_strikers)} players (+T2G, -Short)")

            with col_q4:
                st.markdown(f"**Needs Work**")
                st.caption(f"{len(needs_work)} players (-T2G, -Short)")

        else:
            st.warning("Insufficient data for player positioning map")

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ===== FULL FIELD TABLE WITH HEATMAP =====
        with st.expander("View Complete Field Statistics", expanded=False):
            st.markdown("#### Full Field Breakdown")
            st.caption("Heat-mapped by column - green = good, red = bad")

            # Create display dataframe
            display_df = stats_df[[
                'player_name',
                f'sg_total_{primary_window}',
                f'sg_ott_{primary_window}',
                f'sg_app_{primary_window}',
                f'sg_arg_{primary_window}',
                f'sg_putt_{primary_window}'
            ]].copy()

            display_df.columns = [
                'Player',
                'Total',
                'Driving',
                'Approach',
                'Short Game',
                'Putting'
            ]

            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

            # Apply heatmap styling per column
            def color_negative_positive(val):
                """Color cells based on value - per column"""
                try:
                    num_val = float(val)
                    if num_val > 0.5:
                        return 'background-color: rgba(0, 204, 150, 0.3)'  # Strong green
                    elif num_val > 0:
                        return 'background-color: rgba(0, 204, 150, 0.15)'  # Light green
                    elif num_val > -0.5:
                        return 'background-color: rgba(239, 85, 59, 0.15)'  # Light red
                    else:
                        return 'background-color: rgba(239, 85, 59, 0.3)'  # Strong red
                except:
                    return ''

            # Format and style
            styled_df = display_df.style.applymap(
                color_negative_positive,
                subset=['Total', 'Driving', 'Approach', 'Short Game', 'Putting']
            ).format(
                {
                    'Total': '{:+.2f}',
                    'Driving': '{:+.2f}',
                    'Approach': '{:+.2f}',
                    'Short Game': '{:+.2f}',
                    'Putting': '{:+.2f}'
                }
            )

            st.dataframe(
                styled_df,
                hide_index=True,
                use_container_width=True,
                height=600
            )

            # Download
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="Download Full Data (CSV)",
                data=csv,
                file_name=f"strokes_gained_{primary_stat_col}_{primary_window}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error in SG tab: {str(e)}")
        import traceback
        st.code(traceback.format_exc())