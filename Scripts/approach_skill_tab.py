"""
Approach Skill Tab - Test Version
Full breakdown of DataGolf approach-skill API data:
  - Fairway / Rough toggle
  - L12 / L24 / YTD period toggle
  - Big bucket % numbers from Approach_Buckets.xlsx
  - Per-bucket leaderboard with heatmap
  - Top 5 player cards
  - Field scatter
  - Player deep dive
  - Weakness finder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List


# ── Constants ─────────────────────────────────────────────────────────────────

PERIODS = ["l12", "l24", "ytd"]
PERIOD_LABELS = {"l12": "L12 Months", "l24": "L24 Months", "ytd": "Year to Date"}

FW_BUCKETS = [
    ("50–100",  "50_100_fw"),
    ("100–150", "100_150_fw"),
    ("150–200", "150_200_fw"),
    ("200+",    "over_200_fw"),
]

RGH_BUCKETS = [
    ("Under 150", "under_150_rgh"),
    ("Over 150",  "over_150_rgh"),
]

METRICS = [
    ("sg_per_shot",          "SG / Shot"),
    ("gir_rate",             "GIR %"),
    ("proximity_per_shot",   "Proximity"),
    ("good_shot_rate",       "Good Shot %"),
    ("poor_shot_avoid_rate", "Poor Avoid %"),
    ("shot_count",           "Shots"),
]

# Lower raw value = better (gradient needs negating)
LOWER_IS_BETTER = {"proximity_per_shot"}

BUCKET_PALETTE = ["#4C78A8", "#72B7B2", "#54A24B", "#EECA3B", "#F58518", "#E45756"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_approach_skill(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce")
    return df


def get_col(bucket_prefix: str, metric_suffix: str) -> str:
    return f"{bucket_prefix}_{metric_suffix}"


def fmt_rate(val):
    if pd.isna(val):
        return "—"
    try:
        return f"{float(val) * 100:.1f}%"
    except Exception:
        return "—"


def fmt_sg(val):
    if pd.isna(val):
        return "—"
    try:
        return f"{float(val):+.3f}"
    except Exception:
        return "—"


def fmt_prox(val):
    if pd.isna(val):
        return "—"
    try:
        return f"{float(val):.1f} ft"
    except Exception:
        return "—"


def fmt_count(val):
    if pd.isna(val):
        return "—"
    try:
        return f"{int(val):,}"
    except Exception:
        return "—"


METRIC_FORMATTERS = {
    "sg_per_shot":          fmt_sg,
    "gir_rate":             fmt_rate,
    "proximity_per_shot":   fmt_prox,
    "good_shot_rate":       fmt_rate,
    "poor_shot_avoid_rate": fmt_rate,
    "shot_count":           fmt_count,
}


def is_low_data(df: pd.DataFrame, bucket_prefix: str) -> pd.Series:
    col = get_col(bucket_prefix, "low_data_indicator")
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool)
    return pd.Series(False, index=df.index)


def shorten_name(name: str) -> str:
    parts = str(name).split(",")
    if len(parts) == 2:
        return f"{parts[1].strip()[0]}. {parts[0].strip()}"
    return name


def apply_gradient(sty, disp: pd.DataFrame, metric_suffix: str, metric_label: str):
    """Apply RdYlGn gradient to a column; negate gmap for lower-is-better metrics."""
    raw_col = f"_raw_{metric_suffix}"
    if raw_col not in disp.columns:
        return sty
    gmap = pd.to_numeric(disp[raw_col], errors="coerce")
    if gmap.notna().sum() < 2:
        return sty
    if metric_suffix in LOWER_IS_BETTER:
        # Negate so lower values map to green end of RdYlGn
        sty = sty.background_gradient(
            subset=[metric_label],
            cmap="RdYlGn",
            vmin=-gmap.quantile(0.9),
            vmax=-gmap.quantile(0.1),
            gmap=-gmap,
        )
    else:
        sty = sty.background_gradient(
            subset=[metric_label],
            cmap="RdYlGn",
            vmin=gmap.quantile(0.1),
            vmax=gmap.quantile(0.9),
            gmap=gmap,
        )
    return sty


# ── Main render function ───────────────────────────────────────────────────────

def render_approach_skill_tab(
    approach_skill_path: Path,
    field_ids: Optional[List[int]] = None,
    all_players: Optional[pd.DataFrame] = None,
    approach_buckets_path: Optional[Path] = None,
    event_id: Optional[int] = None,
):
    st.title("Approach Skill")

    # ── Load & filter data ────────────────────────────────────────────────────
    raw = load_approach_skill(approach_skill_path)
    if raw.empty:
        st.error(f"No approach skill data found at {approach_skill_path}")
        st.caption("Run fetch_approach_skill.py to generate this file.")
        return

    if field_ids and len(field_ids) > 0:
        field_set = {int(x) for x in field_ids if pd.notna(x)}
        df_all = raw[raw["dg_id"].isin(field_set)].copy()
    else:
        df_all = raw.copy()

    if df_all.empty:
        st.warning("No players matched the current field.")
        return

    # ── Big bucket % numbers from Approach_Buckets.xlsx ──────────────────────
    bucket_pcts = {}
    event_name_str = ""
    if (approach_buckets_path is not None
            and Path(approach_buckets_path).exists()
            and event_id is not None):
        try:
            bk = pd.read_excel(approach_buckets_path)
            bk.columns = [str(c).strip().lower() for c in bk.columns]
            bk["event_id"] = pd.to_numeric(bk.get("event_id"), errors="coerce")
            event_row = bk[bk["event_id"] == int(event_id)]
            if not event_row.empty:
                r = event_row.iloc[0]
                event_name_str = str(r.get("event_name", "")).strip()
                for label, col in [("50–100",  "50_100"),
                                    ("100–150", "100_150"),
                                    ("150–200", "150_200"),
                                    ("Over 200", "over_200")]:
                    bucket_pcts[label] = pd.to_numeric(r.get(col, np.nan), errors="coerce")
        except Exception:
            pass

    if event_name_str:
        st.markdown(f"**Approach Buckets in Yards for {event_name_str}**")

    if bucket_pcts:
        pct_colors = ["#EF553B", "#FFA15A", "#00CC96", "#90EE90"]
        pct_cols = st.columns(len(bucket_pcts))
        for ci, (label, val) in enumerate(bucket_pcts.items()):
            with pct_cols[ci]:
                val_str = f"{val * 100:.1f}%" if pd.notna(val) else "—"
                st.markdown(
                    f"<div style='font-size:13px; color:#aaa; margin-bottom:4px;'>{label}</div>"
                    f"<div style='font-size:42px; font-weight:900; color:{pct_colors[ci]};'>{val_str}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Selectors — single row ────────────────────────────────────────────────
    ctrl1, ctrl2, _ = st.columns([5, 5, 5])

    with ctrl1:
        period = st.segmented_control(
            "Time Period",
            options=PERIODS,
            format_func=lambda p: PERIOD_LABELS[p],
            default="l24",
            key="approach_period",
        )

    with ctrl2:
        lie = st.segmented_control(
            "Lie",
            options=["Fairway", "Rough"],
            default="Fairway",
            key="approach_lie",
        )

    period = period or "l24"
    lie = lie or "Fairway"

    df = df_all[df_all["time_period"] == period].copy()
    if df.empty:
        st.warning(f"No data for period '{period}'.")
        return

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

    buckets = FW_BUCKETS if lie == "Fairway" else RGH_BUCKETS

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION D — KPI CARDS + MINI HISTOGRAMS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Field Averages by Bucket")
    st.caption("Field median SG/Shot per bucket with distribution of the full field below each card.")

    kpi_cols = st.columns(len(buckets), gap="large")

    for col_idx, (blabel, bprefix) in enumerate(buckets):
        with kpi_cols[col_idx]:
            sg_vals   = pd.to_numeric(df.get(get_col(bprefix, "sg_per_shot"),   pd.Series(dtype=float)), errors="coerce").dropna()
            gir_vals  = pd.to_numeric(df.get(get_col(bprefix, "gir_rate"),      pd.Series(dtype=float)), errors="coerce").dropna()
            prox_vals = pd.to_numeric(df.get(get_col(bprefix, "proximity_per_shot"), pd.Series(dtype=float)), errors="coerce").dropna()

            med_sg   = sg_vals.median()   if len(sg_vals)   > 0 else None
            med_gir  = gir_vals.median()  if len(gir_vals)  > 0 else None
            med_prox = prox_vals.median() if len(prox_vals) > 0 else None

            sg_color = "#00CC96" if (med_sg is not None and med_sg > 0) else "#EF553B"
            sg_txt   = f"{med_sg:+.3f}"       if med_sg   is not None else "—"
            gir_txt  = f"{med_gir * 100:.1f}%" if med_gir  is not None else "—"
            prox_txt = f"{med_prox:.1f} ft"    if med_prox is not None else "—"

            st.markdown(
                f"""
                <div style='border:1px solid rgba(255,255,255,0.1); border-radius:14px;
                            padding:14px 16px; background:rgba(255,255,255,0.03); margin-bottom:8px;'>
                    <div style='font-size:13px; color:#aaa; margin-bottom:6px;'>{blabel}</div>
                    <div style='font-size:28px; font-weight:900; color:{sg_color}; line-height:1;'>{sg_txt}</div>
                    <div style='font-size:11px; color:#888; margin-top:4px;'>SG / Shot (median)</div>
                    <div style='margin-top:10px; display:flex; gap:12px;'>
                        <div>
                            <div style='font-size:15px; font-weight:700; color:#636EFA;'>{gir_txt}</div>
                            <div style='font-size:10px; color:#666;'>GIR</div>
                        </div>
                        <div>
                            <div style='font-size:15px; font-weight:700; color:#FFA15A;'>{prox_txt}</div>
                            <div style='font-size:10px; color:#666;'>Proximity</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if len(sg_vals) > 2:
                fig_hist = go.Figure(go.Histogram(
                    x=sg_vals, nbinsx=20,
                    marker_color=BUCKET_PALETTE[col_idx % len(BUCKET_PALETTE)],
                    opacity=0.85,
                ))
                if med_sg is not None:
                    fig_hist.add_vline(x=med_sg, line_dash="dash",
                                       line_color="white", line_width=1.5)
                fig_hist.update_layout(
                    height=100, margin=dict(t=4, b=4, l=4, r=4),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=True, tickfont=dict(size=8),
                               gridcolor="rgba(128,128,128,0.1)"),
                    yaxis=dict(visible=False), showlegend=False, bargap=0.05,
                )
                st.plotly_chart(fig_hist, use_container_width=True,
                                key=f"hist_{bprefix}_{period}")

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION C — TOP 5 PLAYER CARDS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Top 5 by Bucket")
    st.caption("Best players per distance bucket, sorted by SG/Shot.")

    top5_bucket_labels = [label for label, _ in buckets]
    top5_selected = st.segmented_control(
        "Bucket", options=top5_bucket_labels,
        default=top5_bucket_labels[0], key="top5_bucket_select",
    ) or top5_bucket_labels[0]

    top5_prefix = next(p for l, p in buckets if l == top5_selected)

    top5_sg_col   = get_col(top5_prefix, "sg_per_shot")
    top5_gir_col  = get_col(top5_prefix, "gir_rate")
    top5_prox_col = get_col(top5_prefix, "proximity_per_shot")
    top5_good_col = get_col(top5_prefix, "good_shot_rate")
    top5_poor_col = get_col(top5_prefix, "poor_shot_avoid_rate")
    top5_cnt_col  = get_col(top5_prefix, "shot_count")

    if top5_sg_col in df.columns:
        top5_df = df.copy()
        top5_df[top5_sg_col] = pd.to_numeric(top5_df[top5_sg_col], errors="coerce")
        top5_df = top5_df.dropna(subset=[top5_sg_col]).nlargest(5, top5_sg_col)

        card_cols = st.columns(5, gap="large")
        for ci, (_, prow) in enumerate(top5_df.iterrows()):
            with card_cols[ci]:
                name   = str(prow.get("player_name", ""))
                sg_v   = pd.to_numeric(prow.get(top5_sg_col,   np.nan), errors="coerce")
                gir_v  = pd.to_numeric(prow.get(top5_gir_col,  np.nan), errors="coerce")
                prox_v = pd.to_numeric(prow.get(top5_prox_col, np.nan), errors="coerce")
                good_v = pd.to_numeric(prow.get(top5_good_col, np.nan), errors="coerce")
                poor_v = pd.to_numeric(prow.get(top5_poor_col, np.nan), errors="coerce")
                cnt_v  = pd.to_numeric(prow.get(top5_cnt_col,  np.nan), errors="coerce")

                sg_txt   = f"{sg_v:+.3f}"         if pd.notna(sg_v)   else "—"
                gir_txt  = f"{gir_v * 100:.1f}%"  if pd.notna(gir_v)  else "—"
                prox_txt = f"{prox_v:.1f} ft"      if pd.notna(prox_v) else "—"
                good_txt = f"{good_v * 100:.1f}%"  if pd.notna(good_v) else "—"
                poor_txt = f"{poor_v * 100:.1f}%"  if pd.notna(poor_v) else "—"
                cnt_txt  = f"{int(cnt_v):,} shots" if pd.notna(cnt_v)  else ""
                sg_color = "#00CC96" if (pd.notna(sg_v) and sg_v > 0) else "#EF553B"
                short    = shorten_name(name)

                st.markdown(
                    f"""
                    <div style='border:1px solid rgba(255,255,255,0.12); border-radius:14px;
                                padding:14px 12px; background:rgba(255,255,255,0.03); text-align:center;'>
                        <div style='font-size:11px; color:#888; margin-bottom:2px;'>#{ci + 1}</div>
                        <div style='font-size:13px; font-weight:700; margin-bottom:8px;
                                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{short}</div>
                        <div style='font-size:26px; font-weight:900; color:{sg_color};
                                    line-height:1; margin-bottom:4px;'>{sg_txt}</div>
                        <div style='font-size:10px; color:#666; margin-bottom:10px;'>SG / Shot</div>
                        <div style='display:flex; justify-content:space-around; margin-bottom:6px;'>
                            <div>
                                <div style='font-size:13px; font-weight:700; color:#636EFA;'>{gir_txt}</div>
                                <div style='font-size:9px; color:#666;'>GIR</div>
                            </div>
                            <div>
                                <div style='font-size:13px; font-weight:700; color:#FFA15A;'>{prox_txt}</div>
                                <div style='font-size:9px; color:#666;'>Prox</div>
                            </div>
                        </div>
                        <div style='display:flex; justify-content:space-around;'>
                            <div>
                                <div style='font-size:12px; font-weight:600; color:#00CC96;'>{good_txt}</div>
                                <div style='font-size:9px; color:#666;'>Good</div>
                            </div>
                            <div>
                                <div style='font-size:12px; font-weight:600; color:#72B7B2;'>{poor_txt}</div>
                                <div style='font-size:9px; color:#666;'>Poor Avoid</div>
                            </div>
                        </div>
                        <div style='font-size:9px; color:#555; margin-top:8px;'>{cnt_txt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("SG/Shot data not available for this bucket.")

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION B — FIELD SCATTER
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Field Scatter")
    st.caption("Each dot = a player. X = SG/Shot, Y = GIR%, color = Good Shot Rate. Top 5 labelled.")

    scatter_bucket_label = st.selectbox(
        "Bucket", options=[label for label, _ in buckets], key="scatter_bucket_select",
    )
    scatter_prefix = next(p for l, p in buckets if l == scatter_bucket_label)

    sc_sg_col   = get_col(scatter_prefix, "sg_per_shot")
    sc_gir_col  = get_col(scatter_prefix, "gir_rate")
    sc_good_col = get_col(scatter_prefix, "good_shot_rate")
    sc_poor_col = get_col(scatter_prefix, "poor_shot_avoid_rate")
    sc_prox_col = get_col(scatter_prefix, "proximity_per_shot")
    sc_cnt_col  = get_col(scatter_prefix, "shot_count")
    sc_low_col  = get_col(scatter_prefix, "low_data_indicator")

    if all(c in df.columns for c in [sc_sg_col, sc_gir_col]):
        sc_df = df.copy()
        for c in [sc_sg_col, sc_gir_col, sc_good_col, sc_poor_col, sc_prox_col, sc_cnt_col]:
            if c in sc_df.columns:
                sc_df[c] = pd.to_numeric(sc_df[c], errors="coerce")

        sc_df = sc_df.dropna(subset=[sc_sg_col, sc_gir_col]).copy()
        sc_df["gir_pct"] = sc_df[sc_gir_col] * 100
        sc_df["marker_size"] = pd.to_numeric(
            sc_df.get(sc_low_col, 0), errors="coerce"
        ).fillna(0).astype(bool).map({True: 7, False: 11})
        sc_df["short_name"] = sc_df["player_name"].apply(shorten_name)

        color_vals = (
            sc_df[sc_good_col] * 100
            if sc_good_col in sc_df.columns
            else pd.Series(np.zeros(len(sc_df)), index=sc_df.index)
        )
        cmin = color_vals.quantile(0.05)
        cmax = color_vals.quantile(0.95)

        top5_sc  = sc_df.nlargest(5, sc_sg_col)
        rest_sc  = sc_df[~sc_df.index.isin(top5_sc.index)]
        cd_cols  = [sc_sg_col, "gir_pct", sc_good_col, sc_poor_col, sc_prox_col, sc_cnt_col]
        hover_t  = (
            "<b>%{text}</b><br>"
            "SG/Shot: %{customdata[0]:+.3f}<br>"
            "GIR: %{customdata[1]:.1f}%<br>"
            "Good Shot: %{customdata[2]:.1%}<br>"
            "Poor Avoid: %{customdata[3]:.1%}<br>"
            "Proximity: %{customdata[4]:.1f} ft<br>"
            "Shots: %{customdata[5]:.0f}<extra></extra>"
        )

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=rest_sc[sc_sg_col], y=rest_sc["gir_pct"], mode="markers",
            marker=dict(size=rest_sc["marker_size"],
                        color=color_vals.loc[rest_sc.index],
                        colorscale="RdYlGn", cmin=cmin, cmax=cmax,
                        opacity=0.75,
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                        showscale=False),
            text=rest_sc["short_name"],
            customdata=rest_sc[cd_cols].values,
            hovertemplate=hover_t, showlegend=False,
        ))
        fig_sc.add_trace(go.Scatter(
            x=top5_sc[sc_sg_col], y=top5_sc["gir_pct"], mode="markers+text",
            marker=dict(size=14,
                        color=color_vals.loc[top5_sc.index],
                        colorscale="RdYlGn", cmin=cmin, cmax=cmax,
                        line=dict(width=2, color="white"), showscale=True,
                        colorbar=dict(title="Good Shot %", thickness=12, len=0.6)),
            text=top5_sc["short_name"], textposition="top center",
            textfont=dict(size=11, color="white"),
            customdata=top5_sc[cd_cols].values,
            hovertemplate=hover_t, showlegend=False,
        ))

        med_sg_sc  = sc_df[sc_sg_col].median()
        med_gir_sc = sc_df["gir_pct"].median()
        fig_sc.add_hline(y=med_gir_sc, line_dash="dash",
                         line_color="rgba(255,255,255,0.25)", line_width=1)
        fig_sc.add_vline(x=med_sg_sc, line_dash="dash",
                         line_color="rgba(255,255,255,0.25)", line_width=1)

        x_max = sc_df[sc_sg_col].max()
        x_min = sc_df[sc_sg_col].min()
        y_max = sc_df["gir_pct"].max()
        y_min = sc_df["gir_pct"].min()
        for txt, ax, ay in [
            ("Elite",        x_max * 0.9, y_max * 0.97),
            ("GIR Machines", x_min * 0.9, y_max * 0.97),
            ("SG Grinders",  x_max * 0.9, y_min * 0.97),
        ]:
            fig_sc.add_annotation(x=ax, y=ay, text=f"<b>{txt}</b>", showarrow=False,
                                   font=dict(size=10, color="rgba(255,255,255,0.35)"))

        fig_sc.update_layout(
            height=560, xaxis_title="SG / Shot", yaxis_title="GIR %",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False),
            hovermode="closest", margin=dict(t=30, b=50, l=50, r=60),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Scatter data not available for this bucket.")

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 — BUCKET LEADERBOARDS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Bucket Leaderboards")
    st.caption("Heatmap across all approach metrics. Dimmed rows have low sample size.")

    GRADIENT_METRICS = {"SG / Shot", "GIR %", "Good Shot %", "Poor Avoid %", "Proximity", "Shots"}

    bucket_tabs = st.tabs([label for label, _ in buckets])

    for tab_obj, (bucket_label, bucket_prefix) in zip(bucket_tabs, buckets):
        with tab_obj:
            rows = []
            for _, row in df.iterrows():
                entry = {"Player": row.get("player_name", "Unknown")}
                for metric_suffix, metric_label in METRICS:
                    val = row.get(get_col(bucket_prefix, metric_suffix), np.nan)
                    entry[metric_label] = METRIC_FORMATTERS[metric_suffix](val)
                    entry[f"_raw_{metric_suffix}"] = pd.to_numeric(val, errors="coerce")
                entry["_low_data"] = bool(
                    is_low_data(df.loc[[row.name]], bucket_prefix).iloc[0]
                )
                rows.append(entry)

            if not rows:
                st.info(f"No data for {bucket_label}.")
                continue

            disp = pd.DataFrame(rows)
            if "_raw_sg_per_shot" in disp.columns:
                disp = disp.sort_values(
                    "_raw_sg_per_shot", ascending=False, na_position="last"
                ).reset_index(drop=True)

            low_data_mask = disp["_low_data"].reset_index(drop=True)
            show_cols = ["Player"] + [lbl for _, lbl in METRICS]
            show_cols = [c for c in show_cols if c in disp.columns]
            disp_show = disp[show_cols].copy()

            sty = disp_show.style

            for metric_suffix, metric_label in METRICS:
                if metric_label in GRADIENT_METRICS:
                    sty = apply_gradient(sty, disp, metric_suffix, metric_label)

            def dim_low_data(row, mask=low_data_mask):
                if mask.iloc[row.name]:
                    return ["color: #555; font-style: italic;"] * len(row)
                return [""] * len(row)

            sty = sty.apply(dim_low_data, axis=1)

            st.dataframe(sty, use_container_width=True, hide_index=True, height=600)

            n_low = low_data_mask.sum()
            if n_low > 0:
                st.caption(
                    f"{n_low} players have low data confidence for this bucket (shown dimmed)"
                )

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PLAYER DEEP DIVE
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Player Deep Dive")
    st.caption("Select a player to see their full breakdown across all buckets and all time periods.")

    player_names = sorted(df["player_name"].dropna().unique().tolist())

    if not player_names:
        st.info("No players available.")
    else:
        selected_player = st.selectbox(
            "Select Player", options=player_names, key="approach_player_select",
        )

        player_all_periods = df_all[df_all["player_name"] == selected_player].copy()

        if player_all_periods.empty:
            st.warning(f"No data for {selected_player}.")
        else:
            all_buckets   = FW_BUCKETS + RGH_BUCKETS
            period_colors = {"l12": "#636EFA", "l24": "#00CC96", "ytd": "#FFA15A"}

            st.markdown("#### SG per Shot — All Buckets")
            fig_sg = go.Figure()
            for p in PERIODS:
                p_data = player_all_periods[player_all_periods["time_period"] == p]
                if p_data.empty:
                    continue
                p_row = p_data.iloc[0]
                y_vals   = [pd.to_numeric(p_row.get(get_col(bp, "sg_per_shot"), np.nan), errors="coerce")
                            for _, bp in all_buckets]
                x_labels = [bl for bl, _ in all_buckets]
                fig_sg.add_trace(go.Bar(
                    name=PERIOD_LABELS[p], x=x_labels, y=y_vals,
                    marker_color=period_colors[p],
                    text=[f"{v:+.3f}" if pd.notna(v) else "N/A" for v in y_vals],
                    textposition="outside", textfont=dict(size=10),
                ))

            fig_sg.add_hline(y=0, line_dash="dash",
                             line_color="rgba(255,255,255,0.3)", line_width=1)
            fig_sg.update_layout(
                barmode="group", height=380, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False, title="SG per Shot"),
                margin=dict(t=60, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_sg, use_container_width=True)

            st.markdown(f"#### Full Metric Breakdown — {PERIOD_LABELS[period]}")
            st.caption("Each panel = one metric. Bars = buckets (fairway left, rough right).")

            p_data = player_all_periods[player_all_periods["time_period"] == period]
            if p_data.empty:
                st.info(f"No data for {selected_player} in {PERIOD_LABELS[period]}.")
            else:
                p_row = p_data.iloc[0]
                metric_display = [
                    ("sg_per_shot",          "SG / Shot"),
                    ("gir_rate",             "GIR Rate"),
                    ("proximity_per_shot",   "Proximity (ft)"),
                    ("good_shot_rate",       "Good Shot Rate"),
                    ("poor_shot_avoid_rate", "Poor Avoid Rate"),
                ]
                all_colors = ["#4C78A8", "#72B7B2", "#54A24B", "#EECA3B", "#F58518", "#E45756"]

                fig_detail = make_subplots(
                    rows=1, cols=len(metric_display),
                    subplot_titles=[lbl for _, lbl in metric_display],
                    horizontal_spacing=0.06,
                )
                for col_idx, (ms, ml) in enumerate(metric_display, 1):
                    x_labels = [bl for bl, _ in all_buckets]
                    y_vals   = [pd.to_numeric(p_row.get(get_col(bp, ms), np.nan), errors="coerce")
                                for _, bp in all_buckets]
                    colors   = [all_colors[i % len(all_colors)] for i in range(len(all_buckets))]
                    if ms in ("gir_rate", "good_shot_rate", "poor_shot_avoid_rate"):
                        y_vals = [v * 100 if pd.notna(v) else v for v in y_vals]
                    fig_detail.add_trace(
                        go.Bar(x=x_labels, y=y_vals, marker_color=colors,
                               text=[f"{v:.1f}" if pd.notna(v) else "N/A" for v in y_vals],
                               textposition="outside", textfont=dict(size=9), showlegend=False),
                        row=1, col=col_idx,
                    )
                    fig_detail.update_xaxes(tickangle=-35, row=1, col=col_idx, tickfont=dict(size=9))
                    fig_detail.update_yaxes(gridcolor="rgba(128,128,128,0.15)", row=1, col=col_idx)

                fig_detail.update_layout(
                    height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=60, b=40, l=30, r=30),
                )
                st.plotly_chart(fig_detail, use_container_width=True)

            with st.expander("Shot counts & data confidence", expanded=False):
                count_rows = []
                for blabel, bprefix in all_buckets:
                    row_data = {"Bucket": blabel, "Lie": "Fairway" if "fw" in bprefix else "Rough"}
                    for p in PERIODS:
                        p_data = player_all_periods[player_all_periods["time_period"] == p]
                        if not p_data.empty:
                            cnt = pd.to_numeric(
                                p_data.iloc[0].get(get_col(bprefix, "shot_count"), np.nan), errors="coerce"
                            )
                            low = pd.to_numeric(
                                p_data.iloc[0].get(get_col(bprefix, "low_data_indicator"), 0), errors="coerce"
                            )
                            row_data[f"{PERIOD_LABELS[p]} Shots"]    = fmt_count(cnt)
                            row_data[f"{PERIOD_LABELS[p]} Low Data"] = "Yes" if low else "No"
                    count_rows.append(row_data)
                st.dataframe(pd.DataFrame(count_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — WEAKNESS FINDER
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Weakness Finder")
    st.caption("Ranks players by their weakest bucket relative to the field.")

    wf_col1, wf_col2 = st.columns([1, 3])

    with wf_col1:
        wf_metric_label = st.selectbox(
            "Metric",
            options=[lbl for _, lbl in METRICS if lbl != "Shots"],
            index=0, key="wf_metric",
        )
        wf_metric_suffix = next(s for s, l in METRICS if l == wf_metric_label)

        wf_bucket_label = st.selectbox(
            "Bucket",
            options=[lbl for lbl, _ in buckets],
            key="wf_bucket",
        )
        wf_bucket_prefix = next(p for l, p in buckets if l == wf_bucket_label)

        wf_sort = st.radio(
            "Show", options=["Weakest first", "Strongest first"], key="wf_sort",
        )

    with wf_col2:
        wf_data_col = get_col(wf_bucket_prefix, wf_metric_suffix)

        if wf_data_col not in df.columns:
            st.info(f"Column {wf_data_col} not found in data.")
        else:
            wf_df = df[["player_name", wf_data_col]].copy()
            wf_df[wf_data_col] = pd.to_numeric(wf_df[wf_data_col], errors="coerce")
            wf_df = wf_df.dropna(subset=[wf_data_col])
            wf_df = wf_df.sort_values(
                wf_data_col,
                ascending=(wf_sort == "Weakest first"),
                na_position="last",
            ).reset_index(drop=True)

            if wf_metric_suffix in ("gir_rate", "good_shot_rate", "poor_shot_avoid_rate"):
                wf_df[wf_data_col] = wf_df[wf_data_col] * 100

            n_bars     = min(20, len(wf_df))
            bar_color  = "#EF553B" if wf_sort == "Weakest first" else "#00CC96"
            bar_colors = [bar_color] * min(15, n_bars) + ["#888"] * max(0, n_bars - 15)

            fig_wf = go.Figure(go.Bar(
                x=wf_df["player_name"].head(n_bars),
                y=wf_df[wf_data_col].head(n_bars),
                marker_color=bar_colors,
                text=wf_df[wf_data_col].head(n_bars).apply(
                    lambda v: f"{v:+.3f}" if wf_metric_suffix == "sg_per_shot"
                    else f"{v:.1f}%" if wf_metric_suffix in (
                        "gir_rate", "good_shot_rate", "poor_shot_avoid_rate")
                    else f"{v:.1f}"
                ),
                textposition="outside", textfont=dict(size=9),
            ))
            fig_wf.add_hline(y=0, line_dash="dash",
                             line_color="rgba(255,255,255,0.3)", line_width=1)
            fig_wf.update_layout(
                height=380, xaxis_tickangle=-40,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False,
                           title=wf_metric_label),
                margin=dict(t=20, b=80, l=40, r=20), showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)
