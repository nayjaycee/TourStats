"""
Strokes Gained Tab — Production Version
"""
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from grass_putting_visuals import render_bermuda_putting_visuals

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in [THIS_FILE.parent, *THIS_FILE.parents] if (p / "Data").exists()),
    THIS_FILE.parent,
)
INUSE_DIR = REPO_ROOT / "Data" / "in Use"

# ── Constants ─────────────────────────────────────────────────────────────────
STAT_COLS = ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt"]
WINDOWS   = (12, 24, 36, 60)

STAT_LABELS = {
    "Total SG":             "sg_total",
    "Elite Finish (L36)":   "elite_finish",
    "Driving (OTT)":        "sg_ott",
    "Approach (APP)":       "sg_app",
    "Short Game (ARG)":     "sg_arg",
    "Putting":              "sg_putt",
}
METRICS_LABELS = {k: v for k, v in STAT_LABELS.items() if k != "Elite Finish (L36)"}

CAT_LABELS = ["OTT", "APP", "ARG", "PUTT"]
CAT_STATS  = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


# ── Live data loader ──────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def load_live_stats() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch live tournament stats directly from DataGolf API (event averages)."""
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from src.config import get_secret
        api_key = get_secret("DATAGOLF_API_KEY", required=False)
        if not api_key:
            return None, None

        ALL_STATS = (
            "sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_total,"
            "driving_dist,driving_acc,gir,scrambling,prox_fw,prox_rgh"
        )
        resp = requests.get(
            "https://feeds.datagolf.com/preds/live-tournament-stats",
            params={"stats": ALL_STATS, "round": "event_avg", "display": "value",
                    "file_format": "csv", "key": api_key},
            timeout=20,
        )
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.lower().strip() for c in df.columns]

        for col in ["event_name", "last_updated", "stat_display", "course"]:
            if col in df.columns:
                df[col] = df[col].replace("", np.nan).ffill()

        num_cols = ["dg_id", "total", "round", "thru",
                    "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total",
                    "driving_dist", "driving_acc", "gir", "scrambling", "prox_fw", "prox_rgh"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "dg_id" in df.columns:
            df["dg_id"] = df["dg_id"].astype("Int64")

        event_name = df["event_name"].dropna().iloc[0] if "event_name" in df.columns and not df["event_name"].dropna().empty else "Live Event"
        last_updated = df["last_updated"].dropna().iloc[0] if "last_updated" in df.columns and not df["last_updated"].dropna().empty else ""
        label = f"{event_name}"
        if last_updated:
            try:
                ts = pd.to_datetime(last_updated, utc=True)
                label += f" · {ts.strftime('%-m/%-d %-I:%M%p').lower()} UTC"
            except Exception:
                label += f" · {last_updated}"
        return df, label
    except Exception:
        return None, None


# ── Vectorized stats builder (cached) ────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_stats_df(
    rounds_df: pd.DataFrame,
    player_ids: tuple,
    windows: tuple = WINDOWS,
) -> pd.DataFrame:
    """
    Vectorized rolling SG stats for all players in one pass.
    ~10-20x faster than per-player Python loops for 150-player fields.
    """
    date_col = "round_date" if "round_date" in rounds_df.columns else "event_completed"

    df = rounds_df[rounds_df["dg_id"].isin(player_ids)].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["dg_id", date_col], na_position="first")

    # Coerce all stat columns to numeric once
    for stat in STAT_COLS:
        if stat in df.columns:
            df[stat] = pd.to_numeric(df[stat], errors="coerce")

    # Rolling averages — all windows, all stats, in one vectorized pass
    for stat in STAT_COLS:
        if stat not in df.columns:
            continue
        grp = df.groupby("dg_id")[stat]
        for w in windows:
            df[f"{stat}_L{w}"] = grp.transform(lambda x, _w=w: x.rolling(_w, min_periods=3).mean())

    # Std for Elite Finish Score and Consistent/Volatile
    if "sg_total" in df.columns:
        df["_sg_total_std_L36"] = df.groupby("dg_id")["sg_total"].transform(lambda x: x.rolling(36, min_periods=5).std())
        df["_sg_total_std_L24"] = df.groupby("dg_id")["sg_total"].transform(lambda x: x.rolling(24, min_periods=5).std())

    # Take the most-recent row per player (= rolling value at latest round)
    keep = (
        ["dg_id", "player_name"]
        + [f"{s}_L{w}" for s in STAT_COLS for w in windows if s in df.columns]
        + [c for c in ["_sg_total_std_L36", "_sg_total_std_L24"] if c in df.columns]
    )
    keep = [c for c in keep if c in df.columns]
    latest = df.groupby("dg_id")[keep].last().reset_index(drop=True)

    # Derived: ball striking composite
    for w in windows:
        ott, app = f"sg_ott_L{w}", f"sg_app_L{w}"
        if ott in latest.columns and app in latest.columns:
            latest[f"ball_striking_L{w}"] = latest[ott].fillna(0) + latest[app].fillna(0)

    # Elite Finish Score = mean(L36) − 0.3 × std(L36)
    if "sg_total_L36" in latest.columns and "_sg_total_std_L36" in latest.columns:
        latest["elite_finish_L36"] = (
            latest["sg_total_L36"] - 0.3 * latest["_sg_total_std_L36"].fillna(0)
        )

    return latest


@st.cache_data(show_spinner=False)
def build_ef_history(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run EF formula retrospectively on every 2025/2026 PGA tournament.
    Returns per-tournament results: top-5 picks, hit counts, finishes.
    Cached — only runs once per session.
    """
    date_col = "round_date" if "round_date" in rounds_df.columns else "event_completed"
    test_df  = rounds_df[rounds_df["year"].isin([2025, 2026])].copy()
    test_df["event_completed"] = pd.to_datetime(test_df["event_completed"], errors="coerce")

    tournaments = (
        test_df.groupby(["event_id", "year"], as_index=False)
        .agg(event_name=("event_name", "first"),
             event_end=("event_completed", "max"))
        .sort_values("event_end")
    )

    rows = []
    for _, tourn in tournaments.iterrows():
        event_end = pd.to_datetime(tourn["event_end"], errors="coerce")
        if pd.isna(event_end):
            continue
        cutoff = event_end - pd.Timedelta(days=7)

        ev = test_df[
            (test_df["event_id"] == tourn["event_id"]) &
            (test_df["year"] == tourn["year"])
        ]
        field = (
            ev.groupby("dg_id", as_index=False)
            .agg(player_name=("player_name", "first"), finish_num=("finish_num", "first"))
            .dropna(subset=["finish_num"])
        )
        if len(field) < 20:
            continue

        scores = []
        for _, p in field.iterrows():
            pr = rounds_df[rounds_df["dg_id"] == p["dg_id"]].copy()
            rd = pd.to_datetime(pr[date_col], errors="coerce")
            ec = pd.to_datetime(pr["event_completed"], errors="coerce")
            mask = (rd.notna() & (rd < cutoff)) | (rd.isna() & (ec < cutoff))
            pr = pr[mask].sort_values(date_col, ascending=False, na_position="last").head(36)
            v  = pd.to_numeric(pr["sg_total"], errors="coerce").dropna().values
            if len(v) < 10:
                continue
            scores.append({
                "player_name": p["player_name"],
                "ef_score":    float(np.mean(v) - 0.3 * np.std(v)),
                "finish":      int(p["finish_num"]),
            })

        if len(scores) < 10:
            continue

        pred = pd.DataFrame(scores).sort_values("ef_score", ascending=False)
        top5 = pred.head(5)
        rows.append({
            "event_name":  tourn["event_name"],
            "year":        int(tourn["year"]) if pd.notna(tourn["year"]) else 0,
            "event_end":   event_end,
            "top25_hits":  int((top5["finish"] <= 25).sum()),
            "top10_hits":  int((top5["finish"] <= 10).sum()),
            "top5_hits":   int((top5["finish"] <= 5).sum()),
            "picks":       ", ".join(top5["player_name"].tolist()),
            "finishes":    ", ".join(top5["finish"].astype(str).tolist()),
        })

    return pd.DataFrame(rows)


def render_elite_finish_analysis(
    rounds_df,
    field_ids=None,
    all_players=None,
    cutoff_dt=None,
    summary_top=None,
):
    """Standalone Elite Finish Analysis — current field + odds context + history."""

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("Contender Model")
    col_hdr, col_badge = st.columns([5, 1])
    with col_hdr:
        st.caption(
            "Elite Finish Score = mean(L36) − 0.3 × σ(L36)  ·  "
            "Rewards sustained SG output, penalises boom-or-bust variance"
        )
    with col_badge:
        st.markdown(
            "<div style='background:rgba(0,204,150,0.15);border:1px solid rgba(0,204,150,0.4);"
            "border-radius:8px;padding:6px 10px;text-align:center;margin-top:2px'>"
            "<div style='font-size:18px;font-weight:800;color:#00CC96'>66.6%</div>"
            "<div style='font-size:9px;color:rgba(200,200,200,0.6)'>Top-25 hit rate<br>2025/26 · top-5 picks</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Build current-field stats ────────────────────────────────────────────
    if field_ids and len(field_ids) > 0:
        players_list = field_ids
    elif all_players is not None and "dg_id" in all_players.columns:
        players_list = pd.to_numeric(all_players["dg_id"], errors="coerce").dropna().astype(int).tolist()
    else:
        players_list = rounds_df["dg_id"].unique().tolist()

    rounds_for_stats = rounds_df
    if cutoff_dt is not None:
        _dc    = "round_date" if "round_date" in rounds_df.columns else "event_completed"
        _dates = pd.to_datetime(rounds_df[_dc], errors="coerce")
        rounds_for_stats = rounds_df[_dates.isna() | (_dates < pd.to_datetime(cutoff_dt))].copy()

    with st.spinner("Building stats…"):
        stats_df = build_stats_df(rounds_for_stats, tuple(players_list), windows=WINDOWS)

    if stats_df.empty:
        st.warning("Insufficient data.")
        return

    if all_players is not None and "player_name" not in stats_df.columns:
        name_map = dict(zip(
            pd.to_numeric(all_players["dg_id"], errors="coerce").astype("Int64"),
            all_players["player_name"],
        ))
        stats_df["player_name"] = stats_df["dg_id"].map(name_map).fillna(
            stats_df["dg_id"].apply(lambda x: f"Player {x}")
        )

    if not all(c in stats_df.columns for c in ["elite_finish_L36", "_sg_total_std_L36", "sg_total_L36"]):
        st.warning("Elite Finish columns not available.")
        return

    ef_df = stats_df[["dg_id", "player_name", "sg_total_L36", "_sg_total_std_L36", "elite_finish_L36"]].dropna().copy()
    ef_df = ef_df.rename(columns={
        "sg_total_L36":      "mean_sg",
        "_sg_total_std_L36": "std_sg",
        "elite_finish_L36":  "ef_score",
    })
    ef_df = ef_df.sort_values("ef_score", ascending=False).reset_index(drop=True)
    ef_df["ef_rank"] = range(1, len(ef_df) + 1)

    def _ef_tier(score):
        if score >= 1.0:  return ("Elite",     "#00CC96")
        if score >= 0.5:  return ("Contender", "#66D9A6")
        if score >= 0.0:  return ("Fringe",    "#FFA07A")
        return                   ("Below",     "#EF553B")

    tiers = ef_df["ef_score"].apply(_ef_tier)
    ef_df["tier"]       = [t[0] for t in tiers]
    ef_df["tier_color"] = [t[1] for t in tiers]

    # Merge odds from summary_top if available
    odds_col = None
    if summary_top is not None:
        for c in ["close_odds", "decimal_odds", "odds", "win_prob_est"]:
            if c in summary_top.columns and summary_top[c].notna().any():
                odds_col = c
                break
    if odds_col:
        odds_map = summary_top.set_index("dg_id")[odds_col].to_dict()
        ef_df["odds"] = ef_df["dg_id"].map(odds_map)
        ef_df["implied_prob"] = 100 / ef_df["odds"].where(ef_df["odds"] > 0)
        ef_df["odds_rank"] = ef_df["odds"].rank(method="min", na_option="bottom").astype("Int64")

    # ── Model explanation ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Why this model?", expanded=False):
        st.markdown(
            """
**Formula:** `EF Score = mean(sg_total, L36) − 0.3 × σ(sg_total, L36)`

The Elite Finish score answers a simple question: *which players produce strong strokes-gained numbers reliably, not just occasionally?*
The mean captures sustained output over the last 36 rounds (~9–12 months of PGA starts).
The volatility penalty (0.3 × std dev) discounts boom-or-bust players - a player averaging +1.5 with wild swings
is less predictable than one averaging +1.2 consistently.

**Why L36 and not a longer window?**
We tested five formula variants against every 2025/2026 PGA Tour tournament:

| Formula | Top-25 % | Top-10 % |
|---|---|---|
| **A: mean(L36) - 0.3σ L36** | **66.6%** | **37.2%** |
| E: 0.6×mean(L60) + 0.4×expDecay(L12) - 0.3σ L36 | 63.6% | 34.6% |
| B: mean(L60) - 0.3σ L60 | 63.2% | 34.6% |
| D: exp_decay(L36) - 0.3σ L36 | 63.1% | 33.8% |
| C: mean(L60) - 0.3σ L36 | 62.9% | 34.6% |

The current formula beat every alternative including recency-weighted and long-window variants.
Adding more history (L60) or hot-streak blending (exp decay L12) both hurt performance.
L36 appears to be the right balance: long enough to filter noise, short enough to stay current.

**What the score means:**
- **≥ 1.0 (Elite):** Consistently among the best in the field. High floor, top-25 very likely.
- **0.5 to 1.0 (Contender):** Strong player, competitive most weeks.
- **0.0 to 0.5 (Fringe):** Capable but inconsistent; depends on a peak week.
- **< 0.0 (Below):** Negative expected value relative to field average.

**Scatter plot:** Each dot is a player. Further right = higher mean SG (better output).
Lower = lower std dev (more consistent). The dotted lines are iso-score contours.
Players above a line have a lower EF score for their mean than players below it.
Elite picks live in the bottom-right: high mean, low variance.
            """
        )

    # ── Section 1: Scatter + leaderboard ────────────────────────────────────
    st.markdown("### Current Field")
    col_scatter, col_gap, col_board = st.columns([3, 0.15, 1.85])

    with col_scatter:
        mean_range = np.linspace(ef_df["mean_sg"].min() - 0.3, ef_df["mean_sg"].max() + 0.3, 200)
        fig_ef = go.Figure()

        for ef_val, lc, lname in [
            (1.0, "rgba(0,204,150,0.55)",   "EF = 1.0"),
            (0.5, "rgba(0,204,150,0.30)",   "EF = 0.5"),
            (0.0, "rgba(255,255,255,0.20)", "EF = 0"),
        ]:
            std_iso = (mean_range - ef_val) / 0.3
            mask = std_iso > 0
            fig_ef.add_trace(go.Scatter(
                x=mean_range[mask], y=std_iso[mask],
                mode="lines",
                line=dict(color=lc, dash="dot", width=1.2),
                name=lname, hoverinfo="skip",
            ))

        top15_ids = set(ef_df.head(15)["dg_id"].tolist())
        others_ef = ef_df[~ef_df["dg_id"].isin(top15_ids)]
        top_ef    = ef_df[ef_df["dg_id"].isin(top15_ids)]
        ef_cmin   = ef_df["ef_score"].quantile(0.10)
        ef_cmax   = ef_df["ef_score"].quantile(0.90)

        fig_ef.add_trace(go.Scatter(
            x=others_ef["mean_sg"], y=others_ef["std_sg"],
            mode="markers",
            marker=dict(size=7, opacity=0.5, color=others_ef["ef_score"],
                        colorscale="RdYlGn", cmin=ef_cmin, cmax=ef_cmax),
            text=others_ef["player_name"], customdata=others_ef["ef_score"],
            hovertemplate="<b>%{text}</b><br>Mean: %{x:.2f}  Std: %{y:.2f}  EF: %{customdata:.2f}<extra></extra>",
            showlegend=False,
        ))
        fig_ef.add_trace(go.Scatter(
            x=top_ef["mean_sg"], y=top_ef["std_sg"],
            mode="markers+text",
            marker=dict(
                size=10, color=top_ef["ef_score"],
                colorscale="RdYlGn", cmin=ef_cmin, cmax=ef_cmax,
                line=dict(width=1.5, color="white"),
                showscale=True,
                colorbar=dict(title="EF", thickness=12, len=0.7),
            ),
            text=top_ef["player_name"], textposition="top center",
            textfont=dict(size=8, color="white"),
            customdata=top_ef["ef_score"],
            hovertemplate="<b>%{text}</b><br>Mean: %{x:.2f}  Std: %{y:.2f}  EF: %{customdata:.2f}<extra></extra>",
            showlegend=False,
        ))

        fig_ef.update_layout(
            xaxis_title="Mean SG Total (L36)  →  higher is better",
            yaxis_title="Std Dev (L36)  →  lower is more consistent",
            height=950,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, b=40, l=50, r=20),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    with col_board:
        has_odds = odds_col and "odds" in ef_df.columns and ef_df["odds"].notna().any()

        # Header row
        hdr_odds = "<div style='width:52px;text-align:right;font-size:10px;color:#555'>Odds</div>" if has_odds else ""
        st.markdown(
            f"<div style='display:flex;font-size:10px;color:#555;padding:4px 0;"
            f"border-bottom:1px solid rgba(128,128,128,0.2)'>"
            f"<div style='width:20px'></div><div style='flex:1'>Player</div>"
            f"<div style='width:36px;text-align:right'>EF</div>"
            f"{hdr_odds}"
            f"<div style='width:70px;text-align:right'>Tier</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for r, (_, row) in enumerate(ef_df.head(20).iterrows(), 1):
            color   = row["tier_color"]
            odds_html = ""
            if has_odds and pd.notna(row.get("odds")):
                odds_html = f"<div style='width:52px;text-align:right;font-size:11px;color:#888'>{row['odds']:.0f}</div>"
            elif has_odds:
                odds_html = "<div style='width:52px;text-align:right;font-size:11px;color:#444'>—</div>"
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:4px 0;"
                f"border-bottom:1px solid rgba(128,128,128,0.08)'>"
                f"<div style='width:20px;font-size:10px;color:#555;flex-shrink:0'>{r}</div>"
                f"<div style='flex:1;font-size:12px;font-weight:600;white-space:nowrap;"
                f"overflow:hidden;text-overflow:ellipsis'>{row['player_name']}</div>"
                f"<div style='width:36px;text-align:right;font-size:12px;font-weight:700;"
                f"color:{color}'>{row['ef_score']:+.2f}</div>"
                f"{odds_html}"
                f"<div style='width:70px;text-align:right;margin-left:4px'>"
                f"<span style='font-size:9px;padding:2px 6px;border-radius:8px;"
                f"background:{color}25;color:{color}'>{row['tier']}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # Field tier distribution
        total_ef = len(ef_df)
        for lbl, mask, color in [
            ("Elite (≥ 1.0)",     ef_df["ef_score"] >= 1.0,                                "#00CC96"),
            ("Contender (≥ 0.5)", (ef_df["ef_score"] >= 0.5) & (ef_df["ef_score"] < 1.0), "#66D9A6"),
            ("Fringe (≥ 0.0)",    (ef_df["ef_score"] >= 0.0) & (ef_df["ef_score"] < 0.5), "#FFA07A"),
            ("Below (< 0.0)",     ef_df["ef_score"] < 0.0,                                 "#EF553B"),
        ]:
            n   = mask.sum()
            pct = n / total_ef * 100 if total_ef else 0
            st.markdown(
                f"<div style='margin:4px 0'>"
                f"<div style='display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px'>"
                f"<span style='color:{color}'>{lbl}</span>"
                f"<span style='color:#888'>{n} ({pct:.0f}%)</span></div>"
                f"<div style='background:rgba(128,128,128,0.15);height:5px;border-radius:3px'>"
                f"<div style='background:{color};height:100%;width:{pct:.1f}%;border-radius:3px'>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

    # ── Section 2: Value picks (EF rank vs odds rank) ────────────────────────
    if has_odds and "odds_rank" in ef_df.columns:
        st.divider()
        st.markdown("### Odds Context")
        st.caption("Where the model and the market disagree — EF rank vs odds rank")

        ef_df["rank_delta"] = ef_df["odds_rank"].astype(float) - ef_df["ef_rank"]
        value_df = ef_df[ef_df["odds"].notna()].copy()

        col_val, col_fade = st.columns(2)

        with col_val:
            st.markdown("**Model favours (value plays)**")
            st.caption("EF ranks them much higher than the market")
            top_value = value_df.nlargest(8, "rank_delta")
            for _, row in top_value.iterrows():
                delta = int(row["rank_delta"])
                color = row["tier_color"]
                ip    = row.get("implied_prob")
                ip_str = f"{ip:.1f}%" if pd.notna(ip) else "—"
                st.markdown(
                    f"<div style='display:flex;align-items:center;padding:5px 0;"
                    f"border-bottom:1px solid rgba(128,128,128,0.08)'>"
                    f"<div style='flex:1;font-size:12px;font-weight:600'>{row['player_name']}</div>"
                    f"<div style='font-size:11px;color:#aaa;margin-right:8px'>#{int(row['ef_rank'])} EF</div>"
                    f"<div style='font-size:11px;color:{color};font-weight:700'>{row['ef_score']:+.2f}</div>"
                    f"<div style='width:52px;text-align:right;font-size:11px;color:#888'>{row['odds']:.0f}</div>"
                    f"<div style='width:44px;text-align:right;font-size:10px;color:#00CC96'>+{delta}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with col_fade:
            st.markdown("**Model fades (market overrates)**")
            st.caption("EF ranks them much lower than the market")
            top_fade = value_df.nsmallest(8, "rank_delta")
            for _, row in top_fade.iterrows():
                delta = int(row["rank_delta"])
                color = row["tier_color"]
                st.markdown(
                    f"<div style='display:flex;align-items:center;padding:5px 0;"
                    f"border-bottom:1px solid rgba(128,128,128,0.08)'>"
                    f"<div style='flex:1;font-size:12px;font-weight:600'>{row['player_name']}</div>"
                    f"<div style='font-size:11px;color:#aaa;margin-right:8px'>#{int(row['ef_rank'])} EF</div>"
                    f"<div style='font-size:11px;color:{color};font-weight:700'>{row['ef_score']:+.2f}</div>"
                    f"<div style='width:52px;text-align:right;font-size:11px;color:#888'>{row['odds']:.0f}</div>"
                    f"<div style='width:44px;text-align:right;font-size:10px;color:#EF553B'>{delta}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Section 3: Model history ─────────────────────────────────────────────
    st.divider()
    st.markdown("### Model Performance History")
    st.caption("Retrospective results: top-5 EF picks per tournament vs actual finishes · 2025/2026 PGA Tour")

    with st.spinner("Running historical analysis… (cached after first load)"):
        hist_df = build_ef_history(rounds_df)

    if hist_df.empty:
        st.warning("No historical data available.")
    else:
        n_tourn  = len(hist_df)
        n_picks  = n_tourn * 5
        top25_pct = hist_df["top25_hits"].sum() / n_picks * 100
        top10_pct = hist_df["top10_hits"].sum() / n_picks * 100
        top5_pct  = hist_df["top5_hits"].sum()  / n_picks * 100

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl, sub in [
            (m1, f"{top25_pct:.1f}%", "Top-25 Hit Rate", f"{hist_df['top25_hits'].sum()} / {n_picks} picks"),
            (m2, f"{top10_pct:.1f}%", "Top-10 Hit Rate", f"{hist_df['top10_hits'].sum()} / {n_picks} picks"),
            (m3, f"{top5_pct:.1f}%",  "Top-5 Hit Rate",  f"{hist_df['top5_hits'].sum()} / {n_picks} picks"),
            (m4, str(n_tourn),        "Tournaments",      "2025 + 2026 to date"),
        ]:
            with col:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.04);border-radius:8px;"
                    f"padding:14px;text-align:center'>"
                    f"<div style='font-size:26px;font-weight:800;color:#00CC96'>{val}</div>"
                    f"<div style='font-size:11px;font-weight:600;margin-top:4px'>{lbl}</div>"
                    f"<div style='font-size:10px;color:#666;margin-top:2px'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Rolling trend chart
        hist_df["top25_rate"] = hist_df["top25_hits"] / 5 * 100
        hist_df["rolling10"]  = hist_df["top25_rate"].rolling(10, min_periods=3).mean()
        hist_df["event_label"] = hist_df["event_name"].str[:22] + " '" + hist_df["year"].astype(str).str[-2:]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=hist_df["event_label"], y=hist_df["top25_rate"],
            name="Top-25 %",
            marker_color=[
                "#00CC96" if v >= 60 else "#66D9A6" if v >= 40 else "#FFA07A" if v >= 20 else "#EF553B"
                for v in hist_df["top25_rate"]
            ],
            opacity=0.6,
            hovertemplate="<b>%{x}</b><br>Top-25: %{y:.0f}%<extra></extra>",
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_df["event_label"], y=hist_df["rolling10"],
            name="Rolling 10-tournament avg",
            mode="lines",
            line=dict(color="#00CC96", width=2.5),
            hovertemplate="Rolling avg: %{y:.1f}%<extra></extra>",
        ))
        fig_hist.add_hline(
            y=top25_pct, line_dash="dot", line_color="rgba(255,255,255,0.25)",
            annotation_text=f"Overall {top25_pct:.1f}%",
            annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(200,200,200,0.5)"),
        )
        fig_hist.update_layout(
            yaxis_title="% of top-5 picks finishing top-25",
            yaxis=dict(range=[0, 105], gridcolor="rgba(128,128,128,0.2)"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="rgba(128,128,128,0.1)"),
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=80, l=40, r=20),
            barmode="overlay",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Per-tournament detail table
        with st.expander("Per-tournament breakdown", expanded=False):
            detail = hist_df[["event_end", "event_name", "year", "top25_hits", "top10_hits", "top5_hits", "picks", "finishes"]].copy()
            detail = detail.sort_values("event_end", ascending=False)
            detail["event_end"] = detail["event_end"].dt.strftime("%b %d, %Y")
            detail.columns = ["Date", "Event", "Year", "Top-25", "Top-10", "Top-5", "EF Top-5 Picks", "Finishes"]
            detail.insert(0, "#", range(1, len(detail) + 1))
            st.dataframe(detail, hide_index=True, use_container_width=True, height=500)

    # ── Full ranked table ────────────────────────────────────────────────────
    st.divider()
    with st.expander("Full Field Rankings", expanded=False):
        cols_show = ["player_name", "mean_sg", "std_sg", "ef_score", "tier"]
        if has_odds and "odds" in ef_df.columns:
            cols_show = ["player_name", "mean_sg", "std_sg", "ef_score", "tier", "odds", "implied_prob"]
        disp = ef_df[cols_show].copy()
        disp.insert(0, "Rank", range(1, len(disp) + 1))
        col_rename = {
            "player_name": "Player", "mean_sg": "Mean L36", "std_sg": "Std L36",
            "ef_score": "EF Score", "tier": "Tier", "odds": "Win Odds", "implied_prob": "Implied %",
        }
        disp = disp.rename(columns=col_rename)

        def _cc(val):
            try:
                v = float(val)
                if v > 0.5:  return "background-color:rgba(0,204,150,0.3)"
                if v > 0:    return "background-color:rgba(0,204,150,0.15)"
                if v > -0.5: return "background-color:rgba(239,85,59,0.15)"
                return               "background-color:rgba(239,85,59,0.3)"
            except Exception:
                return ""

        fmt = {"Mean L36": "{:+.3f}", "Std L36": "{:.3f}", "EF Score": "{:+.3f}"}
        if "Implied %" in disp.columns:
            fmt["Implied %"] = "{:.1f}%"
        styled = disp.style.applymap(_cc, subset=["Mean L36", "EF Score"]).format(fmt)
        st.dataframe(styled, hide_index=True, use_container_width=True, height=600)


def _merge_live(stats_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """Merge live event_avg SG columns into stats_df as *_Live columns."""
    live_cols = {s: f"{s}_Live" for s in STAT_COLS if s in live_df.columns}
    if not live_cols:
        return stats_df
    sub = live_df[["dg_id"] + list(live_cols.keys())].copy()
    sub = sub.rename(columns=live_cols)
    for col in live_cols.values():
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return stats_df.merge(sub, on="dg_id", how="left")


# ── Colour helpers ────────────────────────────────────────────────────────────

def _sg_color(val: float) -> str:
    if val > 0.5:   return "#00CC96"
    if val > 0:     return "#66D9A6"
    if val > -0.5:  return "#FFA07A"
    return "#EF553B"


# ── Main render ───────────────────────────────────────────────────────────────

def render_production_sg_tab(
    rounds_df,
    field_ids: Optional[List[int]] = None,
    all_players=None,
    id_to_img: Optional[dict] = None,
    name_to_img: Optional[dict] = None,
    schedule_df=None,
    field_df=None,
    event_id=None,
    cutoff_dt=None,
):
    try:
        st.title("Strokes Gained Analysis")

        # ── Load live data ─────────────────────────────────────────────────
        live_df, live_label = load_live_stats()
        has_live = live_df is not None and not live_df.empty

        # Determine if live data is for the current field event or a prior one
        live_is_current = False
        live_event_name = ""
        field_event_name = ""
        if has_live and "event_name" in live_df.columns:
            live_event_name = str(live_df["event_name"].dropna().iloc[0]).strip() if not live_df["event_name"].dropna().empty else ""
        if field_df is not None and "event_name" in field_df.columns:
            field_event_name = str(field_df["event_name"].dropna().iloc[0]).strip() if not field_df["event_name"].dropna().empty else ""
        if live_event_name and field_event_name:
            live_is_current = live_event_name.lower() == field_event_name.lower()

        if has_live:
            if live_is_current:
                st.info(f"Live data — {live_label}")
            else:
                st.info(
                    f"Showing most recent event data ({live_event_name}). "
                    f"Not yet updated for {field_event_name or 'this week'}."
                )

        # ── Player list ────────────────────────────────────────────────────
        if field_ids and len(field_ids) > 0:
            players_list = field_ids
        elif all_players is not None and "dg_id" in all_players.columns:
            players_list = pd.to_numeric(all_players["dg_id"], errors="coerce").dropna().astype(int).tolist()
        else:
            players_list = rounds_df["dg_id"].unique().tolist()

        # ── Build stats (vectorized + cached) ──────────────────────────────
        # Apply cutoff_dt so current-tournament rounds are excluded from rolling stats
        # (mirrors the same filter used in the overview tab's _render_field_snapshot)
        rounds_for_stats = rounds_df
        if cutoff_dt is not None:
            _date_col = "round_date" if "round_date" in rounds_df.columns else "event_completed"
            _dates = pd.to_datetime(rounds_df[_date_col], errors="coerce")
            rounds_for_stats = rounds_df[_dates.isna() | (_dates < pd.to_datetime(cutoff_dt))].copy()

        with st.spinner("Loading stats…"):
            stats_df = build_stats_df(rounds_for_stats, tuple(players_list), windows=WINDOWS)
            if stats_df.empty:
                st.warning("Insufficient data to calculate stats.")
                return

            # Enrich with player names from all_players if missing
            if all_players is not None and "player_name" not in stats_df.columns:
                name_map = dict(zip(
                    pd.to_numeric(all_players["dg_id"], errors="coerce").astype("Int64"),
                    all_players["player_name"],
                ))
                stats_df["player_name"] = stats_df["dg_id"].map(name_map).fillna(
                    stats_df["dg_id"].apply(lambda x: f"Player {x}")
                )

            # Merge live SG if available
            if has_live:
                stats_df = _merge_live(stats_df, live_df)

        # ── Controls ───────────────────────────────────────────────────────
        col_stat, col_window, col_spacer = st.columns([2, 1, 2])

        with col_stat:
            primary_stat = st.selectbox("Primary Metric", list(STAT_LABELS.keys()), index=0)

        with col_window:
            if primary_stat == "Elite Finish (L36)":
                primary_window = "L36"
                st.selectbox("Window", ["L36"], disabled=True)
            else:
                primary_window = st.selectbox("Window", ["L12", "L24", "L36", "L60"], index=0)

        primary_stat_col = STAT_LABELS[primary_stat]

        # Resolve base sort column (always historical)
        if primary_stat == "Elite Finish (L36)":
            sort_col = "elite_finish_L36"
        else:
            sort_col = f"{primary_stat_col}_{primary_window}"

        if sort_col not in stats_df.columns:
            st.warning(f"No data available for {primary_stat} ({primary_window}).")
            return

        # ── Live blend slider — only during an active/current tournament ──
        live_col = f"{primary_stat_col}_Live"
        live_blend = 0
        if has_live and live_is_current and live_col in stats_df.columns and primary_stat != "Elite Finish (L36)":
            live_blend = st.slider(
                f"Blend live {live_event_name} SG into ranking",
                min_value=0, max_value=60, value=25, step=5,
                format="%d%%",
                help="Blends in-progress round data into the sort. Only available during an active event.",
            )

        # Filter to players with data for the base historical column
        valid = stats_df[stats_df[sort_col].notna()].copy()
        if valid.empty:
            st.warning("No players with sufficient data for the selected metric.")
            return

        # Apply blend: sort_key = (1 - w) * historical + w * live (for players with live data)
        if live_blend > 0 and live_col in valid.columns:
            w = live_blend / 100
            valid["_sort_key"] = (
                (1 - w) * valid[sort_col]
                + w * valid[live_col].fillna(valid[sort_col])  # fall back to historical if no live
            )
        else:
            valid["_sort_key"] = valid[sort_col]

        valid = valid.sort_values("_sort_key", ascending=False).reset_index(drop=True)

        # Write top players to session state for cross-tab defaults
        if len(valid) >= 1:
            st.session_state["weekly_top1_dg_id"] = int(valid.iloc[0]["dg_id"])
        if len(valid) >= 2:
            st.session_state["weekly_top2_dg_id"] = int(valid.iloc[1]["dg_id"])

        # ── Top 10 performer cards ─────────────────────────────────────────
        blend_note = f" · {live_blend}% live blend" if live_blend > 0 else ""
        st.caption(
            f"Ranked by {primary_stat} ({primary_window}){blend_note} · {len(valid)} players · "
            "Component bars use only rounds with full SG data."
        )
        st.markdown("### Top Performers")

        top10 = valid.head(10)

        for row_idx in range(2):
            cols = st.columns(5, gap="medium")
            for col_idx, (_, player) in enumerate(top10.iloc[row_idx * 5 : row_idx * 5 + 5].iterrows()):
                with cols[col_idx]:
                    rank    = row_idx * 5 + col_idx + 1
                    dg_id   = int(player["dg_id"])
                    pname   = player.get("player_name", f"Player {dg_id}")
                    img_url = (id_to_img or {}).get(dg_id) or (name_to_img or {}).get(pname)

                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        initials = "".join(p[0].upper() for p in pname.split(", ")[::-1] if p)[:2]
                        st.markdown(
                            f"<div style='width:100%;aspect-ratio:200/220;background:rgba(80,80,80,0.25);"
                            f"border-radius:6px;display:flex;align-items:center;justify-content:center;"
                            f"font-size:28px;font-weight:700;color:rgba(255,255,255,0.25)'>{initials}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"<div style='text-align:center;font-size:11px;color:#888;margin-top:8px'>#{rank}</div>"
                        f"<div style='text-align:center;font-weight:700;font-size:13px;margin-bottom:6px'>{pname}</div>",
                        unsafe_allow_html=True,
                    )

                    primary_val = player[sort_col]
                    color = _sg_color(primary_val)
                    st.markdown(
                        f"<div style='text-align:center;font-size:22px;font-weight:800;color:{color};margin-bottom:8px'>"
                        f"{primary_val:+.2f}</div>",
                        unsafe_allow_html=True,
                    )

                    # Component breakdown bars
                    # Scale bars against the primary_val total so visual proportions match.
                    # Raw component values are shown as labels; bars are sized relative to total.
                    bar_window = "L12" if primary_window in ("This Wk", "L12") else primary_window
                    raw_vals = {
                        stat: float(player.get(f"{stat}_{bar_window}") or 0)
                        for stat in CAT_STATS
                    }
                    comp_sum = sum(raw_vals.values())
                    # If components sum within 10% of the displayed total, use raw values.
                    # Otherwise scale them so bars reflect share of total (component data
                    # may cover fewer rounds than the total when some tours omit SG breakdown).
                    if abs(primary_val) > 0.01 and abs(comp_sum) > 0.01 and abs(comp_sum - primary_val) / max(abs(primary_val), 0.01) > 0.10:
                        scale_factor = primary_val / comp_sum
                        display_vals = {s: v * scale_factor for s, v in raw_vals.items()}
                    else:
                        display_vals = raw_vals

                    scale = 25
                    for cat, stat in zip(CAT_LABELS, CAT_STATS):
                        val = display_vals[stat]
                        raw = raw_vals[stat]
                        bar_color = _sg_color(raw)
                        if val >= 0:
                            bw = min(val * scale, 50)
                            bp = "margin-left:50%;"
                        else:
                            bw = min(abs(val) * scale, 50)
                            bp = f"margin-left:{50 - bw:.1f}%;"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin:2px 0;font-size:10px'>"
                            f"<div style='width:30px;text-align:right;margin-right:4px;opacity:0.7'>{cat}</div>"
                            f"<div style='flex:1;background:rgba(128,128,128,0.2);height:8px;border-radius:4px;position:relative'>"
                            f"<div style='position:absolute;left:50%;width:1px;height:100%;background:rgba(255,255,255,0.3)'></div>"
                            f"<div style='background:{bar_color};height:100%;width:{bw:.1f}%;{bp}'></div>"
                            f"</div>"
                            f"<div style='width:35px;text-align:left;margin-left:4px;font-weight:600;color:{bar_color}'>{raw:+.1f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Show live stat alongside historical if available
                    if has_live:
                        live_val = player.get(f"{primary_stat_col}_Live")
                        if pd.notna(live_val):
                            lc = _sg_color(live_val)
                            badge_label = "This Wk" if live_is_current else "Last Wk"
                            st.markdown(
                                f"<div style='text-align:center;font-size:10px;margin-top:6px;"
                                f"color:rgba(200,200,200,0.5)'>{badge_label} "
                                f"<span style='color:{lc};font-weight:700'>{live_val:+.2f}</span></div>",
                                unsafe_allow_html=True,
                            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ── Key Metrics ────────────────────────────────────────────────────
        st.markdown("### Key Metrics")

        col_ms, _ = st.columns([2, 3])
        with col_ms:
            metrics_stat = st.selectbox("Analyze", list(METRICS_LABELS.keys()), index=0, key="metrics_stat")

        metrics_stat_col = METRICS_LABELS[metrics_stat]
        metrics_sort_col = f"{metrics_stat_col}_{primary_window}"
        if metrics_sort_col not in valid.columns:
            metrics_sort_col = f"{metrics_stat_col}_L12"

        st.caption(f"Rankings based on {metrics_stat} ({primary_window})")

        col1, col2, col3, col4 = st.columns(4)

        def _mini_list(df_col, players, label, val_fmt, color_fn):
            for rank, (_, p) in enumerate(players.iterrows(), 1):
                val = p[df_col]
                color = color_fn(val)
                st.markdown(
                    f"<div style='padding:8px 0;border-bottom:1px solid rgba(128,128,128,0.12)'>"
                    f"<div style='font-weight:700;font-size:13px'>{rank}. {p['player_name']}</div>"
                    f"<div style='font-size:12px;color:{color};margin-top:3px;font-weight:600'>{val_fmt(val)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with col1:
            st.markdown("#### Top")
            st.caption("Highest averages")
            top5 = valid.nlargest(5, metrics_sort_col) if metrics_sort_col in valid.columns else valid.head(5)
            _mini_list(metrics_sort_col, top5, "top", lambda v: f"{v:+.2f}", _sg_color)

        with col2:
            st.markdown("#### Bottom")
            st.caption("Needs work")
            bot5 = valid.nsmallest(5, metrics_sort_col) if metrics_sort_col in valid.columns else valid.tail(5)
            _mini_list(metrics_sort_col, bot5, "bot", lambda v: f"{v:+.2f}", _sg_color)

        with col3:
            st.markdown("#### Consistent")
            st.caption("Low volatility")
            std_col = "_sg_total_std_L24" if metrics_stat_col == "sg_total" else None
            if std_col and std_col in valid.columns:
                c_players = valid[valid[std_col].notna()].nsmallest(5, std_col)
                for rank, (_, p) in enumerate(c_players.iterrows(), 1):
                    sv = p[std_col]
                    st.markdown(
                        f"<div style='padding:8px 0;border-bottom:1px solid rgba(128,128,128,0.12)'>"
                        f"<div style='font-weight:700;font-size:13px'>{rank}. {p['player_name']}</div>"
                        f"<div style='font-size:12px;color:#636EFA;margin-top:3px;font-weight:600'>σ = {sv:.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("—")

        with col4:
            st.markdown("#### Volatile")
            st.caption("Boom or bust")
            if std_col and std_col in valid.columns:
                v_players = valid[valid[std_col].notna()].nlargest(5, std_col)
                for rank, (_, p) in enumerate(v_players.iterrows(), 1):
                    sv = p[std_col]
                    st.markdown(
                        f"<div style='padding:8px 0;border-bottom:1px solid rgba(128,128,128,0.12)'>"
                        f"<div style='font-weight:700;font-size:13px'>{rank}. {p['player_name']}</div>"
                        f"<div style='font-size:12px;color:#FFA500;margin-top:3px;font-weight:600'>σ = {sv:.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("—")

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ── Elite Finish Analysis ──────────────────────────────────────────
        if (
            "elite_finish_L36" in valid.columns
            and "_sg_total_std_L36" in valid.columns
            and "sg_total_L36" in valid.columns
        ):
            st.markdown("### Elite Finish Analysis")
            st.caption(
                "Elite Finish Score = mean(L36) − 0.3 × σ(L36)  ·  "
                "Rewards sustained SG output and penalises boom-or-bust variance  ·  "
                "66.6% of top-5 picks finish top-25 on 2025/26 data"
            )

            ef_df = valid[
                ["dg_id", "player_name", "sg_total_L36", "_sg_total_std_L36", "elite_finish_L36"]
            ].dropna().copy()
            ef_df = ef_df.rename(columns={
                "sg_total_L36":       "mean_sg",
                "_sg_total_std_L36":  "std_sg",
                "elite_finish_L36":   "ef_score",
            })
            ef_df = ef_df.sort_values("ef_score", ascending=False).reset_index(drop=True)

            def _ef_tier(score):
                if score >= 1.0:  return ("Elite",      "#00CC96")
                if score >= 0.5:  return ("Contender",  "#66D9A6")
                if score >= 0.0:  return ("Fringe",     "#FFA07A")
                return                   ("Below",      "#EF553B")

            tiers = ef_df["ef_score"].apply(_ef_tier)
            ef_df["tier"]       = [t[0] for t in tiers]
            ef_df["tier_color"] = [t[1] for t in tiers]

            col_ef_scatter, col_ef_board = st.columns([3, 2])

            with col_ef_scatter:
                mean_range = np.linspace(ef_df["mean_sg"].min() - 0.3, ef_df["mean_sg"].max() + 0.3, 200)

                fig_ef = go.Figure()

                # Iso-score contour lines
                for ef_val, lc, lname in [
                    (1.0, "rgba(0,204,150,0.55)",   "EF = 1.0"),
                    (0.5, "rgba(0,204,150,0.30)",   "EF = 0.5"),
                    (0.0, "rgba(255,255,255,0.20)", "EF = 0"),
                ]:
                    std_iso = (mean_range - ef_val) / 0.3
                    mask = std_iso > 0
                    fig_ef.add_trace(go.Scatter(
                        x=mean_range[mask], y=std_iso[mask],
                        mode="lines",
                        line=dict(color=lc, dash="dot", width=1.2),
                        name=lname,
                        hoverinfo="skip",
                    ))

                top15_ids  = set(ef_df.head(15)["dg_id"].tolist())
                others_ef  = ef_df[~ef_df["dg_id"].isin(top15_ids)]
                top_ef     = ef_df[ef_df["dg_id"].isin(top15_ids)]
                ef_cmin    = ef_df["ef_score"].quantile(0.10)
                ef_cmax    = ef_df["ef_score"].quantile(0.90)

                fig_ef.add_trace(go.Scatter(
                    x=others_ef["mean_sg"],
                    y=others_ef["std_sg"],
                    mode="markers",
                    marker=dict(
                        size=7, opacity=0.55,
                        color=others_ef["ef_score"],
                        colorscale="RdYlGn",
                        cmin=ef_cmin, cmax=ef_cmax,
                    ),
                    text=others_ef["player_name"],
                    customdata=others_ef["ef_score"],
                    hovertemplate="<b>%{text}</b><br>Mean L36: %{x:.2f}<br>Std: %{y:.2f}<br>EF: %{customdata:.2f}<extra></extra>",
                    showlegend=False,
                ))

                fig_ef.add_trace(go.Scatter(
                    x=top_ef["mean_sg"],
                    y=top_ef["std_sg"],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=top_ef["ef_score"],
                        colorscale="RdYlGn",
                        cmin=ef_cmin, cmax=ef_cmax,
                        line=dict(width=1.5, color="white"),
                        showscale=True,
                        colorbar=dict(title="EF", thickness=12, len=0.7),
                    ),
                    text=top_ef["player_name"],
                    textposition="top center",
                    textfont=dict(size=8, color="white"),
                    customdata=top_ef["ef_score"],
                    hovertemplate="<b>%{text}</b><br>Mean L36: %{x:.2f}<br>Std: %{y:.2f}<br>EF: %{customdata:.2f}<extra></extra>",
                    showlegend=False,
                ))

                fig_ef.update_layout(
                    xaxis_title="Mean SG Total (L36)  →  higher is better",
                    yaxis_title="Std Dev (L36)  →  lower is more consistent",
                    height=480,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                    yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                    ),
                    margin=dict(t=30, b=40, l=50, r=20),
                )
                st.plotly_chart(fig_ef, use_container_width=True)

            with col_ef_board:
                st.markdown("**Top 15 by Elite Finish Score**")

                for rank, (_, row) in enumerate(ef_df.head(15).iterrows(), 1):
                    color = row["tier_color"]
                    st.markdown(
                        f"<div style='display:flex;align-items:center;padding:5px 0;"
                        f"border-bottom:1px solid rgba(128,128,128,0.10)'>"
                        f"<div style='width:20px;font-size:10px;color:#666;flex-shrink:0'>{rank}</div>"
                        f"<div style='flex:1;font-size:12px;font-weight:600;white-space:nowrap;"
                        f"overflow:hidden;text-overflow:ellipsis'>{row['player_name']}</div>"
                        f"<div style='width:36px;text-align:right;font-size:12px;font-weight:700;"
                        f"color:{color}'>{row['ef_score']:+.2f}</div>"
                        f"<div style='width:70px;text-align:right;margin-left:6px'>"
                        f"<span style='font-size:9px;padding:2px 6px;border-radius:8px;"
                        f"background:{color}25;color:{color}'>{row['tier']}</span>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Mean / std detail for top 5
                st.markdown("**Top 5 breakdown**")
                st.markdown(
                    "<div style='display:flex;font-size:10px;color:#666;padding:4px 0;"
                    "border-bottom:1px solid rgba(128,128,128,0.2)'>"
                    "<div style='flex:1'>Player</div>"
                    "<div style='width:48px;text-align:right'>Mean</div>"
                    "<div style='width:40px;text-align:right'>Std</div>"
                    "<div style='width:40px;text-align:right'>EF</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                for _, row in ef_df.head(5).iterrows():
                    color = row["tier_color"]
                    st.markdown(
                        f"<div style='display:flex;align-items:center;font-size:11px;"
                        f"padding:4px 0;border-bottom:1px solid rgba(128,128,128,0.08)'>"
                        f"<div style='flex:1;white-space:nowrap;overflow:hidden;"
                        f"text-overflow:ellipsis'>{row['player_name']}</div>"
                        f"<div style='width:48px;text-align:right;color:#aaa'>{row['mean_sg']:+.2f}</div>"
                        f"<div style='width:40px;text-align:right;color:#aaa'>{row['std_sg']:.2f}</div>"
                        f"<div style='width:40px;text-align:right;font-weight:700;color:{color}'>"
                        f"{row['ef_score']:+.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Field tier distribution
                st.markdown("**Field distribution**")
                tiers_order = [
                    ("Elite (≥ 1.0)",     ef_df["ef_score"] >= 1.0,                                           "#00CC96"),
                    ("Contender (≥ 0.5)", (ef_df["ef_score"] >= 0.5) & (ef_df["ef_score"] < 1.0),            "#66D9A6"),
                    ("Fringe (≥ 0.0)",    (ef_df["ef_score"] >= 0.0) & (ef_df["ef_score"] < 0.5),            "#FFA07A"),
                    ("Below (< 0.0)",     ef_df["ef_score"] < 0.0,                                            "#EF553B"),
                ]
                total_ef = len(ef_df)
                for label, mask, color in tiers_order:
                    n = mask.sum()
                    pct = n / total_ef * 100 if total_ef else 0
                    bar_w = pct
                    st.markdown(
                        f"<div style='margin:4px 0'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:11px;margin-bottom:2px'>"
                        f"<span style='color:{color}'>{label}</span>"
                        f"<span style='color:#888'>{n} ({pct:.0f}%)</span></div>"
                        f"<div style='background:rgba(128,128,128,0.15);height:5px;border-radius:3px'>"
                        f"<div style='background:{color};height:100%;width:{bar_w:.1f}%;"
                        f"border-radius:3px'></div></div></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ── Recent Form Trends ─────────────────────────────────────────────
        st.markdown("### Recent Form Trends")

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            form_stat = st.selectbox("Stat to Compare", list(METRICS_LABELS.keys()), index=0, key="form_stat")

        with col_f2:
            form_w1 = st.selectbox("Recent Window", ["L12", "L24", "L36", "L60"], index=0, key="form_w1")

        with col_f3:
            compare_defaults = [w for w in ["L24", "L36", "L60", "L12"] if w != form_w1]
            form_w2 = st.selectbox("Compare to", compare_defaults, index=0, key="form_w2")

        form_stat_col = METRICS_LABELS[form_stat]
        col1_name = f"{form_stat_col}_Live" if form_w1 == "This Wk" else f"{form_stat_col}_{form_w1}"
        col2_name = f"{form_stat_col}_{form_w2}"

        if col1_name in valid.columns and col2_name in valid.columns:
            trend_df = valid[[col1_name, col2_name, "player_name"]].dropna().copy()
            trend_df["form_trend"] = trend_df[col1_name] - trend_df[col2_name]
            movers = pd.concat([
                trend_df.nlargest(10, "form_trend"),
                trend_df.nsmallest(10, "form_trend"),
            ]).sort_values("form_trend")

            fig = go.Figure(go.Bar(
                y=movers["player_name"],
                x=movers["form_trend"],
                orientation="h",
                marker_color=["#00CC96" if x > 0 else "#EF553B" for x in movers["form_trend"]],
                text=movers["form_trend"].apply(lambda x: f"{x:+.2f}"),
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
            fig.update_layout(
                xaxis_title=f"{form_stat} Change ({form_w1} vs {form_w2})",
                yaxis_title="",
                height=600,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selected comparison windows not available.")

        st.divider()

        # ── Player Strengths Map ───────────────────────────────────────────
        st.markdown("### Player Strengths Map")
        st.caption(f"Tee-to-Green (OTT + APP) vs Short Game (ARG + PUTT) · {primary_window}")

        t2g_col  = f"sg_ott_{primary_window}"
        app_col  = f"sg_app_{primary_window}"
        arg_col  = f"sg_arg_{primary_window}"
        putt_col = f"sg_putt_{primary_window}"

        if all(c in valid.columns for c in [t2g_col, app_col, arg_col, putt_col]):
            scatter_df = valid.copy()
            scatter_df["tee_to_green"] = scatter_df[t2g_col].fillna(0) + scatter_df[app_col].fillna(0)
            scatter_df["short_game"]   = scatter_df[arg_col].fillna(0) + scatter_df[putt_col].fillna(0)

            top10_ids  = set(top10["dg_id"].tolist())
            others     = scatter_df[~scatter_df["dg_id"].isin(top10_ids)]
            top10_data = scatter_df[scatter_df["dg_id"].isin(top10_ids)]

            fig2 = go.Figure()
            for df_part, show_text in [(others, False), (top10_data, True)]:
                fig2.add_trace(go.Scatter(
                    x=df_part["tee_to_green"],
                    y=df_part["short_game"],
                    mode="markers+text" if show_text else "markers",
                    marker=dict(
                        size=10 if show_text else 8,
                        color=df_part[sort_col] if sort_col in df_part.columns else None,
                        colorscale="RdYlGn",
                        showscale=show_text is False,
                        colorbar=dict(title=primary_stat) if not show_text else None,
                        line=dict(width=1 if show_text else 0.5, color="white"),
                    ),
                    text=df_part["player_name"],
                    textposition="top center",
                    textfont=dict(size=8, color="white"),
                    hovertemplate="<b>%{text}</b><br>T2G: %{x:.2f}<br>Short: %{y:.2f}<extra></extra>",
                    showlegend=False,
                    cliponaxis=False,
                ))

            fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
            fig2.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

            mx, my = scatter_df["tee_to_green"].max(), scatter_df["short_game"].max()
            mn_x, mn_y = scatter_df["tee_to_green"].min(), scatter_df["short_game"].min()
            for text, ax, ay, align in [
                ("Elite All-Around", mx * 0.85,  my * 0.9,  "right"),
                ("Short Game God",   mn_x * 0.85, my * 0.9,  "left"),
                ("Ball Strikers",    mx * 0.85,  mn_y * 0.9, "right"),
                ("Needs Work",       mn_x * 0.85, mn_y * 0.9, "left"),
            ]:
                fig2.add_annotation(x=ax, y=ay, text=f"<b>{text}</b>", showarrow=False,
                                    font=dict(size=10, color="rgba(255,255,255,0.4)"), align=align)

            fig2.update_layout(
                xaxis_title="Tee-to-Green (OTT + APP)",
                yaxis_title="Short Game (ARG + PUTT)",
                height=700,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False),
                hovermode="closest",
                margin=dict(t=40, b=40, l=40, r=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            q_elite  = scatter_df[(scatter_df["tee_to_green"] > 0) & (scatter_df["short_game"] > 0)]
            q_sg     = scatter_df[(scatter_df["tee_to_green"] < 0) & (scatter_df["short_game"] > 0)]
            q_bs     = scatter_df[(scatter_df["tee_to_green"] > 0) & (scatter_df["short_game"] < 0)]
            q_nw     = scatter_df[(scatter_df["tee_to_green"] < 0) & (scatter_df["short_game"] < 0)]
            with col_q1: st.markdown("**Elite All-Around**");  st.caption(f"{len(q_elite)} players")
            with col_q2: st.markdown("**Short Game Artists**"); st.caption(f"{len(q_sg)} players")
            with col_q3: st.markdown("**Ball Strikers**");      st.caption(f"{len(q_bs)} players")
            with col_q4: st.markdown("**Needs Work**");         st.caption(f"{len(q_nw)} players")
        else:
            st.warning("Insufficient data for player map.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ── Putting by Grass Type ──────────────────────────────────────────
        st.markdown("### Putting by Grass Type")
        _field_df = field_df
        if _field_df is None and all_players is not None and field_ids is not None:
            _field_df = all_players[all_players["dg_id"].isin(field_ids)][["dg_id", "player_name"]].copy()

        render_bermuda_putting_visuals(
            rounds_df=rounds_df,
            schedule_df=schedule_df,
            field_df=_field_df,
            event_id=event_id,
            cutoff_dt=cutoff_dt,
        )

        st.divider()

        # ── Full Field Table ───────────────────────────────────────────────
        with st.expander("View Complete Field Statistics", expanded=False):
            st.markdown("#### Full Field Breakdown")

            # Always show the selected window + live side by side if available
            disp_window = primary_window if primary_window != "This Wk" else "L12"
            base_cols = {
                "Player":      "player_name",
                f"Total ({disp_window})":  f"sg_total_L{disp_window[1:]}",
                f"Drive ({disp_window})":  f"sg_ott_L{disp_window[1:]}",
                f"App ({disp_window})":    f"sg_app_L{disp_window[1:]}",
                f"ARG ({disp_window})":    f"sg_arg_L{disp_window[1:]}",
                f"Putt ({disp_window})":   f"sg_putt_L{disp_window[1:]}",
            }
            live_extra_cols = {}
            if has_live:
                live_extra_cols = {
                    "Total (Live)": "sg_total_Live",
                    "Drive (Live)": "sg_ott_Live",
                    "App (Live)":   "sg_app_Live",
                    "ARG (Live)":   "sg_arg_Live",
                    "Putt (Live)":  "sg_putt_Live",
                }

            all_col_map = {**base_cols, **live_extra_cols}
            available   = {k: v for k, v in all_col_map.items() if v in valid.columns}

            disp = valid[[v for v in available.values()]].copy()
            disp.columns = list(available.keys())
            disp.insert(0, "Rank", range(1, len(disp) + 1))

            num_cols = [c for c in disp.columns if c not in ("Rank", "Player")]

            def _cell_color(val):
                try:
                    v = float(val)
                    if v > 0.5:   return "background-color:rgba(0,204,150,0.3)"
                    if v > 0:     return "background-color:rgba(0,204,150,0.15)"
                    if v > -0.5:  return "background-color:rgba(239,85,59,0.15)"
                    return "background-color:rgba(239,85,59,0.3)"
                except Exception:
                    return ""

            styled = (
                disp.style
                .applymap(_cell_color, subset=num_cols)
                .format({c: "{:+.2f}" for c in num_cols})
            )

            st.dataframe(styled, hide_index=True, use_container_width=True, height=600)

            st.download_button(
                "Download Full Data (CSV)",
                valid.to_csv(index=False),
                file_name=f"sg_field_{primary_window}.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error in SG tab: {e}")
        import traceback
        st.code(traceback.format_exc())
