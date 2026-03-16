from __future__ import annotations

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from io import StringIO
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# Constants — match overview_tab / weather_tab style exactly
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://feeds.datagolf.com/preds/live-tournament-stats"

_ALL_STATS = (
    "sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_total,"
    "driving_dist,driving_acc,gir,scrambling,prox_fw,prox_rgh"
)

_SG_COLS   = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"]
_BALL_COLS = ["driving_dist", "driving_acc", "gir", "scrambling", "prox_fw", "prox_rgh"]

_SG_LABELS = {
    "sg_putt": "Putt", "sg_arg": "ARG", "sg_app": "APP",
    "sg_ott": "OTT",   "sg_t2g": "T2G", "sg_total": "Total",
}
_SG_COLORS = {
    "sg_putt":  "rgba(255,190,100,0.85)",
    "sg_arg":   "rgba(190,150,255,0.85)",
    "sg_app":   "rgba(255,130,160,0.85)",
    "sg_ott":   "rgba(120,180,255,0.85)",
    "sg_t2g":   "rgba(52,211,153,0.85)",
    "sg_total": "#fbbf24",
}

_SEC = (
    "font-size:11px;font-weight:700;letter-spacing:0.08em;"
    "color:rgba(130,130,130,0.7);text-transform:uppercase;margin-bottom:8px"
)
_SUB = "font-size:12px;color:rgba(140,140,140,0.5);margin-bottom:10px"

_PAR_COLOR = {
    "under": "#ef4444",
    "over":  "#60a5fa",
    "even":  "#22c55e",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _par_fmt(val) -> str:
    try:
        v = int(round(float(val)))
        if v == 0:  return "E"
        return f"+{v}" if v > 0 else str(v)
    except Exception:
        return "—"


def _par_color(val) -> str:
    try:
        v = float(val)
        if v < 0:  return _PAR_COLOR["under"]
        if v > 0:  return _PAR_COLOR["over"]
        return _PAR_COLOR["even"]
    except Exception:
        return "rgba(200,200,200,0.7)"


def _thru_fmt(val) -> str:
    try:
        v = int(float(val))
        return "F" if v == 18 else str(v)
    except Exception:
        return "—"


def _pct_fmt(val) -> str:
    try:
        v = float(val)
        # DataGolf returns ratios (0–1) for gir/scrambling/driving_acc
        if v <= 1.0:
            return f"{v * 100:.1f}%"
        return f"{v:.1f}%"
    except Exception:
        return "—"


def _yds_fmt(val) -> str:
    try:
        return f"{float(val):.1f} yds"
    except Exception:
        return "—"


def _ft_fmt(val) -> str:
    try:
        return f"{float(val):.1f} ft"
    except Exception:
        return "—"


def _short(name: str) -> str:
    parts = str(name).split(",")
    return parts[0].strip() if len(parts) == 2 else name.split()[-1]


def _divider() -> None:
    st.markdown(
        "<div style='height:1px;background:rgba(255,255,255,0.06);margin:24px 0'></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tournament-active guard  (called from Stats.py to decide tab visibility)
# ─────────────────────────────────────────────────────────────────────────────

def is_tournament_live(field_df: pd.DataFrame) -> bool:
    """
    Returns True when:
      - today falls within [date_start, date_end] for the current week's event, AND
      - the earliest R1 tee time has passed (i.e. play has begun).
    Requires this_week_field.csv to be loaded as field_df.
    """
    if field_df is None or field_df.empty:
        return False

    try:
        fdf = field_df.copy()
        fdf.columns = [c.lower().strip() for c in fdf.columns]

        now = pd.Timestamp.now()

        # Date window
        start_col = next((c for c in ["date_start", "start_date"] if c in fdf.columns), None)
        end_col   = next((c for c in ["date_end",   "end_date"]   if c in fdf.columns), None)
        if not start_col or not end_col:
            return False

        date_start = pd.to_datetime(fdf[start_col].iloc[0], errors="coerce")
        date_end   = pd.to_datetime(fdf[end_col].iloc[0],   errors="coerce")
        if pd.isna(date_start) or pd.isna(date_end):
            return False
        if not (date_start.date() <= now.date() <= date_end.date()):
            return False

        # First tee time has passed
        tee_col = next((c for c in ["r1_teetime"] if c in fdf.columns), None)
        if tee_col:
            first_tee = pd.to_datetime(fdf[tee_col].dropna(), errors="coerce").min()
            if pd.notna(first_tee) and now < first_tee:
                return False

        return True

    except Exception:
        return False


def get_current_round(field_df: pd.DataFrame) -> int:
    """Read current_round from the field file; fall back to 1."""
    try:
        fdf = field_df.copy()
        fdf.columns = [c.lower().strip() for c in fdf.columns]
        if "current_round" in fdf.columns:
            val = pd.to_numeric(fdf["current_round"].iloc[0], errors="coerce")
            if pd.notna(val):
                return int(val)
    except Exception:
        pass
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch  — cached 60 s, manual override via session state
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_live(api_key: str, round_param: str) -> pd.DataFrame:
    params = {
        "stats":       _ALL_STATS,
        "round":       round_param,
        "display":     "value",
        "file_format": "csv",
        "key":         api_key,
    }
    resp = requests.get(_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.lower().strip() for c in df.columns]

    # Forward-fill event_name / last_updated — only populated on first row
    for col in ["event_name", "last_updated", "stat_display", "course"]:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan).ffill()

    # Numeric coercion
    num_cols = [
        "dg_id", "total", "round", "thru",
        *_SG_COLS, *_BALL_COLS,
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _load_live(api_key: str, round_param: str) -> pd.DataFrame:
    """Wrapper that respects a manual-refresh flag in session state."""
    if st.session_state.get("_live_force_refresh"):
        _fetch_live.clear()
        st.session_state["_live_force_refresh"] = False
    return _fetch_live(api_key, round_param)


@st.cache_data(ttl=1800, show_spinner=False)  # 30 min — odds move slowly
def _fetch_dk_odds(api_key: str) -> dict:
    """Fetch DraftKings win odds from DataGolf. Returns dg_id -> decimal odds dict."""
    try:
        resp = requests.get(
            "https://feeds.datagolf.com/betting-tools/outrights",
            params={
                "tour":        "pga",
                "market":      "win",
                "odds_format": "decimal",
                "file_format": "json",
                "key":         api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        result  = {}
        for row in payload.get("odds", []):
            dg_id = row.get("dg_id")
            odds  = row.get("draftkings")
            if dg_id is not None and odds is not None:
                try:
                    result[int(dg_id)] = float(odds)
                except (ValueError, TypeError):
                    pass
        return result
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# A — Header bar  (event name, last updated, refresh button)
# ─────────────────────────────────────────────────────────────────────────────

def _render_header(df: pd.DataFrame) -> None:
    """Renders the top control bar — event name, freshness indicator, refresh button."""
    event_name   = df["event_name"].iloc[0]  if "event_name"   in df.columns else "Live Tournament"
    last_updated = df["last_updated"].iloc[0] if "last_updated" in df.columns else ""

    mins = 999
    lu_str = str(last_updated)
    age = ""
    try:
        lu = pd.to_datetime(last_updated, utc=True)
        lu_str = lu.strftime("%-I:%M %p UTC")
        delta = pd.Timestamp.now(tz="UTC") - lu
        mins  = int(delta.total_seconds() // 60)
        age   = f"{mins}m ago" if mins > 0 else "just now"
    except Exception:
        pass

    age_color = "#22c55e" if mins <= 2 else "#fbbf24" if mins <= 10 else "#ef4444"

    hdr_left, hdr_right = st.columns([4, 1], gap="medium")

    with hdr_left:
        st.markdown(
            f"<div style='margin-bottom:4px'>"
            f"<div style='font-size:24px;font-weight:900;color:rgba(255,255,255,0.95)'>"
            f"{event_name}</div>"
            f"<div style='font-size:12px;color:rgba(140,140,140,0.55);margin-top:3px;"
            f"display:flex;align-items:center;gap:6px'>"
            f"<span style='width:7px;height:7px;border-radius:50%;background:{age_color};"
            f"display:inline-block'></span>"
            f"DataGolf updated {lu_str} · {age}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with hdr_right:
        if st.button("Refresh", key="live_refresh", use_container_width=True):
            st.session_state["_live_force_refresh"] = True
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# B — Summary stat chips (leader, low round SG, field avg SG total)
# ─────────────────────────────────────────────────────────────────────────────

def _render_hero_row(df: pd.DataFrame) -> None:
    """
    Two tight bands above the leaderboard:
      1. Score leader + SG category leaders (5 chips, full width)
      2. What's working: top-10 avg SG per category with field avg context (pure HTML, no chart)
    """
    sg_cats_all = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

    clean = df[~df["position"].astype(str).str.upper().isin(["WD","CUT","DQ"])].copy()
    if clean.empty:
        return

    # ── Band 1: leader chips ─────────────────────────────────────────────────
    leader       = clean.iloc[0]
    leader_name  = str(leader.get("player_name", ""))
    leader_score = _par_fmt(leader.get("total"))
    leader_thru  = _thru_fmt(leader.get("thru"))
    leader_rnd   = _par_fmt(leader.get("round"))

    chip_defs = [("LEADER", leader_name, leader_score,
                  _par_color(leader.get("total")), f"Thru {leader_thru} · Rnd {leader_rnd}")]
    for col_name, lbl, color in [
        ("sg_ott",  "OTT LEADER",  _SG_COLORS["sg_ott"]),
        ("sg_app",  "APP LEADER",  _SG_COLORS["sg_app"]),
        ("sg_arg",  "ARG LEADER",  _SG_COLORS["sg_arg"]),
        ("sg_putt", "PUTT LEADER", _SG_COLORS["sg_putt"]),
    ]:
        if col_name in clean.columns and clean[col_name].notna().any():
            br = clean.loc[clean[col_name].idxmax()]
            chip_defs.append((lbl, _short(str(br.get("player_name",""))),
                              f"{br[col_name]:+.2f}", color, ""))

    # DK odds chip for the leader (if odds data present)
    if "dk_odds" in clean.columns and clean["dk_odds"].notna().any():
        leader_odds = clean.iloc[0].get("dk_odds", np.nan)
        if pd.notna(leader_odds):
            # Convert decimal to American
            if leader_odds >= 2.0:
                amer = f"+{int(round((leader_odds - 1) * 100))}"
            else:
                amer = f"{int(round(-100 / (leader_odds - 1)))}"
            chip_defs.append(("DK ODDS", _short(leader_name), amer, "rgba(0,163,255,0.9)", "to win"))

    chip_cols = st.columns(len(chip_defs), gap="small")
    for i, (col, (label, sub, val, color, caption)) in enumerate(zip(chip_cols, chip_defs)):
        is_leader = (i == 0)
        border = "1px solid rgba(255,255,255,0.45)" if is_leader else "1px solid rgba(255,255,255,0.07)"
        with col:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border:{border};"
                f"border-radius:10px;padding:10px 14px;height:80px;box-sizing:border-box'>"
                f"<div style='font-size:9px;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.08em;color:rgba(120,120,120,0.5);margin-bottom:3px'>{label}</div>"
                f"<div style='font-size:11px;color:rgba(170,170,170,0.7);margin-bottom:4px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
                f"{sub}"
                f"{'<span style=\"font-size:9px;color:rgba(100,100,100,0.45);margin-left:6px\">' + caption + '</span>' if caption else ''}"
                f"</div>"
                f"<div style='font-size:20px;font-weight:900;color:{color};line-height:1.1'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Band 2: what's working (SG top-10 vs field, pure HTML) ──────────────
    sg_available = [c for c in sg_cats_all if c in clean.columns and clean[c].notna().any()]
    if len(sg_available) < 2 or len(clean) < 10:
        return

    def _pos_num(p):
        try: return int(str(p).replace("T",""))
        except: return 999

    clean["_pos_num"] = clean["position"].apply(_pos_num)
    top10  = clean.nsmallest(10, "_pos_num")
    field_avg = {c: clean[c].mean() for c in sg_available}
    top_avg   = {c: top10[c].mean() for c in sg_available}
    deltas    = {c: top_avg[c] - field_avg[c] for c in sg_available}

    # Scale: biggest delta = full bar width
    max_delta = max(abs(v) for v in deltas.values()) or 1.0

    # Sort by delta descending so biggest separator is first
    sorted_sg = sorted(sg_available, key=lambda c: deltas[c], reverse=True)

    # Build one HTML block — label | bar (proportional) | top-10 avg | vs field delta
    items_html = ""
    for c in sorted_sg:
        label   = _SG_LABELS.get(c, c)
        color   = _SG_COLORS.get(c, "#aaa")
        t_avg   = top_avg[c]
        f_avg   = field_avg[c]
        delta   = deltas[c]
        bar_pct = max(4, int(abs(delta) / max_delta * 100))
        f_sign  = f"{f_avg:+.2f}"
        d_sign  = f"{delta:+.2f}"
        items_html += (
            f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;"
            f"border-bottom:1px solid rgba(255,255,255,0.04)'>"
            # label
            f"<div style='width:38px;font-size:10px;font-weight:700;color:rgba(160,160,160,0.6);"
            f"text-transform:uppercase;flex-shrink:0'>{label}</div>"
            # bar track
            f"<div style='flex:1;background:rgba(255,255,255,0.05);border-radius:3px;height:6px'>"
            f"<div style='width:{bar_pct}%;height:6px;background:{color};"
            f"border-radius:3px;opacity:0.9'></div></div>"
            # top-10 value
            f"<div style='width:42px;text-align:right;font-size:13px;font-weight:900;"
            f"color:{color};flex-shrink:0'>{t_avg:+.2f}</div>"
            # field avg
            f"<div style='width:70px;font-size:10px;color:rgba(110,110,110,0.55);"
            f"flex-shrink:0'>field {f_sign}</div>"
            # delta badge
            f"<div style='width:44px;text-align:right;font-size:10px;font-weight:700;"
            f"color:rgba(200,200,200,0.5);flex-shrink:0'>+{delta:.2f} edge</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='margin-top:10px;padding:10px 14px;"
        f"background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);"
        f"border-radius:10px'>"
        f"<div style='font-size:9px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:rgba(100,100,100,0.5);margin-bottom:6px'>"
        f"What's winning — Top 10 SG vs field avg</div>"
        f"{items_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# C — Live leaderboard table
# ─────────────────────────────────────────────────────────────────────────────

def _render_leaderboard(df: pd.DataFrame, id_to_img: dict) -> None:
    _divider()
    st.markdown(f"<div style='{_SEC}'>Leaderboard</div>", unsafe_allow_html=True)

    main_df  = df[~df["position"].astype(str).str.upper().isin(["WD", "CUT", "DQ"])].copy()
    other_df = df[df["position"].astype(str).str.upper().isin(["WD", "CUT", "DQ"])].copy()

    th = (
        "font-size:10px;font-weight:700;letter-spacing:0.07em;"
        "color:rgba(130,130,130,0.65);text-transform:uppercase;"
        "padding:7px 8px;border-bottom:1px solid rgba(255,255,255,0.08);"
        "position:sticky;top:0;background:#0e1117;z-index:1;"
    )

    has_odds = "dk_odds" in main_df.columns and main_df["dk_odds"].notna().any()

    table = (
        f"<table style='width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed'>"
        f"<thead><tr>"
        f"<th style='{th};text-align:left;width:46px'>POS</th>"
        f"<th style='{th};text-align:left;'>PLAYER</th>"
        f"<th style='{th};text-align:center;width:55px'>TOTAL</th>"
        f"<th style='{th};text-align:center;width:55px'>RND</th>"
        f"<th style='{th};text-align:center;width:55px'>THRU</th>"
        f"<th style='{th};text-align:center;width:55px'>SG TOT</th>"
        + (f"<th style='{th};text-align:center;width:55px'>ODDS</th>" if has_odds else "")
        + f"<th style='{th};text-align:left;width:400px'>SG BREAKDOWN</th>"
        f"</tr></thead><tbody>"
    )

    # Compute field-wide scale for bar widths
    sg_scale_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    _field_max = 1.0
    for _c in sg_scale_cols:
        if _c in main_df.columns:
            _m = main_df[_c].abs().quantile(0.95)
            if pd.notna(_m) and _m > _field_max:
                _field_max = float(_m)

    def _sg_bar_html(row) -> str:
        components = [
            ("sg_ott",  "OTT",  _SG_COLORS["sg_ott"]),
            ("sg_app",  "APP",  _SG_COLORS["sg_app"]),
            ("sg_arg",  "ARG",  _SG_COLORS["sg_arg"]),
            ("sg_putt", "PUTT", _SG_COLORS["sg_putt"]),
        ]
        vals = [(lbl, color, pd.to_numeric(row.get(col, np.nan), errors="coerce"))
                for col, lbl, color in components]
        vals = [(lbl, color, v) for lbl, color, v in vals if pd.notna(v)]
        if not vals:
            return "<span style='color:rgba(100,100,100,0.4);font-size:11px'>—</span>"

        # Center-baseline layout: label | [neg bar←][→pos bar] | value
        # Half-width = 50% of available space, bar fills proportionally from center
        bars = ""
        neg_col = "rgba(239,68,68,0.80)"
        for lbl, bar_color, v in vals:
            pct    = min(abs(v) / _field_max, 1.0)
            w_pct  = max(1.0, pct * 100)  # percent of half-container, max ~48%
            use_col = bar_color if v >= 0 else neg_col
            val_col = "rgba(200,200,200,0.9)" if v >= 0 else "#ef4444"

            if v >= 0:
                neg_bar = "<div style='flex:1'></div>"
                pos_bar = f"<div style='width:{w_pct}%;height:10px;background:{use_col};border-radius:0 3px 3px 0;flex-shrink:0'></div>"
            else:
                neg_bar = f"<div style='width:{w_pct}%;height:10px;background:{use_col};border-radius:3px 0 0 3px;margin-left:auto;flex-shrink:0'></div>"
                pos_bar = "<div style='flex:1'></div>"

            bars += (
                f"<div style='display:grid;grid-template-columns:32px 1fr 1fr 44px;"
                f"align-items:center;gap:0;margin-bottom:5px'>"
                f"<span style='font-size:10px;color:rgba(130,130,130,0.6);text-align:right;padding-right:6px'>{lbl}</span>"
                # left half (negatives live here)
                f"<div style='display:flex;align-items:center;justify-content:flex-end;"
                f"background:rgba(255,255,255,0.03);border-right:1px solid rgba(255,255,255,0.08);height:10px'>"
                f"{neg_bar}</div>"
                # right half (positives live here)
                f"<div style='display:flex;align-items:center;"
                f"background:rgba(255,255,255,0.03);height:10px'>"
                f"{pos_bar}</div>"
                f"<span style='font-size:10px;font-weight:700;color:{val_col};"
                f"text-align:right;padding-left:6px'>{v:+.2f}</span>"
                f"</div>"
            )
        return f"<div style='padding:3px 0;width:100%'>{bars}</div>"

    def _rows_html(slice_df, muted=False) -> str:
        html = ""
        for idx, (_, row) in enumerate(slice_df.iterrows()):
            pos      = str(row.get("position", "—"))
            name     = str(row.get("player_name", "—"))
            total    = row.get("total", np.nan)
            rnd      = row.get("round", np.nan)
            thru     = row.get("thru", np.nan)
            sg_total = row.get("sg_total", np.nan)

            total_s = _par_fmt(total)
            total_c = _par_color(total)
            rnd_s   = _par_fmt(rnd)
            rnd_c   = _par_color(rnd)
            thru_s  = _thru_fmt(thru)
            sg_s    = f"{sg_total:+.2f}" if pd.notna(sg_total) else "—"
            sg_c    = "#34d399" if pd.notna(sg_total) and sg_total > 0 else "#ef4444" if pd.notna(sg_total) and sg_total < 0 else "rgba(180,180,180,0.6)"
            sg_bar  = _sg_bar_html(row)
            stripe  = "rgba(255,255,255,0.015)" if idx % 2 == 0 else "transparent"
            opacity = "opacity:0.45;" if muted else ""
            top_border = "border-top:2px solid #fbbf24;" if pos == "1" else ""

            # DK odds — convert decimal to American
            dk_raw = row.get("dk_odds", np.nan)
            if pd.notna(dk_raw) and not muted:
                dk_dec = float(dk_raw)
                if dk_dec >= 2.0:
                    dk_s = f"+{int(round((dk_dec - 1) * 100))}"
                else:
                    dk_s = f"{int(round(-100 / (dk_dec - 1)))}"
                dk_cell = (f"<td style='padding:6px 4px;text-align:center;font-size:12px;"
                           f"font-weight:700;color:rgba(0,163,255,0.85)'>{dk_s}</td>")
            else:
                dk_cell = "<td style='padding:6px 4px;text-align:center;font-size:11px;color:rgba(80,80,80,0.5)'>—</td>"

            html += (
                f"<tr style='border-bottom:1px solid rgba(255,255,255,0.04);"
                f"background:{stripe};{opacity}{top_border}'>"
                f"<td style='padding:6px 8px;font-weight:700;color:rgba(255,255,255,0.75);font-size:13px'>{pos}</td>"
                f"<td style='padding:6px 8px;font-weight:600;color:rgba(255,255,255,0.9);font-size:12px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:130px'>{name}</td>"
                f"<td style='padding:6px 4px;text-align:center;font-size:15px;font-weight:800;color:{total_c}'>{total_s}</td>"
                f"<td style='padding:6px 4px;text-align:center;font-size:13px;font-weight:600;color:{rnd_c}'>{rnd_s}</td>"
                f"<td style='padding:6px 4px;text-align:center;font-size:12px;color:rgba(170,170,170,0.6)'>{thru_s}</td>"
                f"<td style='padding:6px 4px;text-align:center;font-size:13px;font-weight:700;color:{sg_c}'>{sg_s}</td>"
                + (dk_cell if has_odds else "")
                + f"<td style='padding:3px 8px'>{sg_bar}</td>"
                f"</tr>"
            )
        return html

    table += _rows_html(main_df)

    if not other_df.empty:
        table += (
            f"<tr><td colspan='{8 if has_odds else 7}' style='padding:6px 8px;font-size:10px;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.08em;color:rgba(100,100,100,0.4);"
            f"border-top:1px solid rgba(255,255,255,0.06)'>Withdrawn / Cut</td></tr>"
        )
        table += _rows_html(other_df, muted=True)

    table += "</tbody></table>"

    # Each row is ~88px tall (4 bars + padding). 25 rows ≈ 2200px + header
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.02);border-radius:10px;"
        f"border:1px solid rgba(255,255,255,0.07);overflow-x:auto;"
        f"overflow-y:auto;max-height:2300px'>{table}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# D — SG snapshot  (horizontal bar, top 15 by SG total)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sg_snapshot(df: pd.DataFrame) -> None:
    _divider()
    st.markdown(f"<div style='{_SEC}'>Strokes Gained Snapshot</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Top 15 by SG Total — component breakdown</div>",
        unsafe_allow_html=True,
    )

    sg_df = df.dropna(subset=["sg_total"]).copy()
    sg_df = sg_df[~sg_df["position"].astype(str).str.upper().isin(["WD","CUT","DQ"])]
    sg_df = sg_df.nlargest(15, "sg_total")

    components = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    fig = go.Figure()

    for comp in components:
        if comp not in sg_df.columns:
            continue
        fig.add_trace(go.Bar(
            y=[_short(n) for n in sg_df["player_name"]],
            x=sg_df[comp],
            name=_SG_LABELS[comp],
            orientation="h",
            marker=dict(color=_SG_COLORS[comp], opacity=0.85),
            hovertemplate=f"<b>%{{y}}</b> — {_SG_LABELS[comp]}: <b>%{{x:+.2f}}</b><extra></extra>",
        ))

    # SG total diamond markers
    fig.add_trace(go.Scatter(
        y=[_short(n) for n in sg_df["player_name"]],
        x=sg_df["sg_total"],
        mode="markers",
        name="Total",
        marker=dict(
            size=10, symbol="diamond",
            color="rgba(255,255,255,0.9)",
            line=dict(color="rgba(0,0,0,0.4)", width=1),
        ),
        hovertemplate="<b>%{y}</b> — SG Total: <b>%{x:+.2f}</b><extra></extra>",
    ))

    fig.update_layout(
        barmode="relative",
        height=max(320, len(sg_df) * 26 + 60),
        template="plotly_dark",
        margin=dict(l=10, r=20, t=10, b=40),
        xaxis=dict(
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
            gridcolor="rgba(255,255,255,0.05)", title="Strokes Gained",
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11),
                   gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="y unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# E — Movers strip  (requires previous snapshot stored in session state)
# ─────────────────────────────────────────────────────────────────────────────

def _render_movers(df: pd.DataFrame) -> None:
    """
    Compares current positions against the previous pull stored in session state.
    On first load, just stores; on subsequent loads, shows movers.
    """
    prev_key = "live_prev_positions"
    curr_positions = (
        df[["dg_id", "position", "player_name"]]
        .dropna(subset=["dg_id"])
        .copy()
    )
    curr_positions["dg_id"] = curr_positions["dg_id"].astype(int)

    prev = st.session_state.get(prev_key)

    if prev is not None and not prev.empty:
        merged = curr_positions.merge(
            prev[["dg_id", "position"]].rename(columns={"position": "prev_pos"}),
            on="dg_id", how="inner",
        )

        def _pos_num(p):
            try:
                return int(str(p).replace("T", "").replace("WD", "999").replace("CUT", "999").replace("DQ", "999"))
            except Exception:
                return 999

        merged["curr_num"] = merged["position"].apply(_pos_num)
        merged["prev_num"] = merged["prev_pos"].apply(_pos_num)
        merged["move"]     = merged["prev_num"] - merged["curr_num"]
        merged             = merged[merged["move"] != 0].copy()

        risers  = merged[merged["move"] > 0].nlargest(4, "move")
        fallers = merged[merged["move"] < 0].nsmallest(4, "move")

        if not risers.empty or not fallers.empty:
            _divider()
            st.markdown(f"<div style='{_SEC}'>Movers Since Last Refresh</div>", unsafe_allow_html=True)

            col_r, col_f = st.columns(2, gap="large")

            def _mover_html(rows, up: bool) -> str:
                html = ""
                for _, row in rows.iterrows():
                    move  = abs(int(row["move"]))
                    color = "#22c55e" if up else "#ef4444"
                    arrow = "▲" if up else "▼"
                    pos   = str(row["position"])
                    name  = str(row["player_name"])
                    html += (
                        f"<div style='display:flex;align-items:center;gap:10px;"
                        f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>"
                        f"<span style='font-size:18px;font-weight:900;color:{color}'>{arrow}{move}</span>"
                        f"<div>"
                        f"<div style='font-size:13px;font-weight:600;color:rgba(215,215,215,0.9)'>{name}</div>"
                        f"<div style='font-size:10px;color:rgba(120,120,120,0.5)'>Now {pos}</div>"
                        f"</div></div>"
                    )
                return html

            with col_r:
                st.markdown(
                    f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.08em;color:#22c55e;margin-bottom:8px'>Rising</div>"
                    + _mover_html(risers, up=True),
                    unsafe_allow_html=True,
                )
            with col_f:
                st.markdown(
                    f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.08em;color:#ef4444;margin-bottom:8px'>Falling</div>"
                    + _mover_html(fallers, up=False),
                    unsafe_allow_html=True,
                )

    # Store current for next comparison
    st.session_state[prev_key] = curr_positions


# ─────────────────────────────────────────────────────────────────────────────
# F — Player strengths map  (T2G vs Short Game, colored by cut status)
# ─────────────────────────────────────────────────────────────────────────────

def _render_strengths_map(df: pd.DataFrame, cut_line=None) -> None:
    needed = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    if not all(c in df.columns for c in needed):
        return

    _divider()
    st.markdown(f"<div style='{_SEC}'>Player Strengths Map</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Tee-to-Green (OTT + APP) vs Short Game (ARG + PUTT) · "
        f"color = leaderboard position</div>",
        unsafe_allow_html=True,
    )

    plot_df = df[~df["position"].astype(str).str.upper().isin(["WD","DQ"])].copy()
    plot_df["sg_t2g_live"]  = pd.to_numeric(plot_df["sg_ott"], errors="coerce") + \
                               pd.to_numeric(plot_df["sg_app"], errors="coerce")
    plot_df["sg_short_live"] = pd.to_numeric(plot_df["sg_arg"], errors="coerce") + \
                                pd.to_numeric(plot_df["sg_putt"], errors="coerce")
    plot_df = plot_df.dropna(subset=["sg_t2g_live", "sg_short_live"]).copy()

    def _pos_num(p):
        try:
            return int(str(p).replace("T",""))
        except Exception:
            return 999

    plot_df["_pos_num"] = plot_df["position"].apply(_pos_num)

    # Projected cut position
    scores = plot_df["total"].dropna().sort_values()
    _cut_n = int(cut_line) if cut_line else 65
    cut_pos = min(_cut_n, len(scores) - 1)
    cut_score = float(scores.iloc[cut_pos - 1]) if len(scores) >= cut_pos else 999

    # Label players: making cut vs projected miss vs top 10
    def _cut_status(row):
        total = row.get("total", np.nan)
        pos   = row["_pos_num"]
        if pos <= 10:
            return "top10"
        if pd.notna(total) and float(total) <= cut_score:
            return "making"
        return "missing"

    plot_df["_status"] = plot_df.apply(_cut_status, axis=1)

    color_map  = {"top10": "#fbbf24", "making": "#34d399", "missing": "#ef4444"}
    size_map   = {"top10": 14,         "making": 9,         "missing": 9}
    opacity_map = {"top10": 1.0,       "making": 0.75,      "missing": 0.55}

    fig = go.Figure()

    # Quadrant reference lines at 0
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.08)", dash="dash", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.08)", dash="dash", width=1))

    # Quadrant labels
    pad = 0.15
    x_max = max(abs(plot_df["sg_t2g_live"].max()), abs(plot_df["sg_t2g_live"].min())) + 0.5
    y_max = max(abs(plot_df["sg_short_live"].max()), abs(plot_df["sg_short_live"].min())) + 0.3
    quad_labels = [
        ( x_max - pad, -y_max + pad, "Ball Strikers",    "right"),
        (-x_max + pad, -y_max + pad, "Needs Work",       "left"),
        (-x_max + pad,  y_max - pad, "Short Game Gods",  "left"),
        ( x_max - pad,  y_max - pad, "Elite All-Around", "right"),
    ]
    for qx, qy, qtxt, qanchor in quad_labels:
        fig.add_annotation(
            x=qx, y=qy, text=qtxt, showarrow=False,
            font=dict(size=10, color="rgba(120,120,120,0.35)"),
            xanchor=qanchor,
        )

    for status in ["missing", "making", "top10"]:
        sub = plot_df[plot_df["_status"] == status]
        if sub.empty:
            continue
        show_text = status == "top10"
        label_map = {"top10": "Top 10", "making": "Making cut", "missing": "Projected miss"}
        fig.add_trace(go.Scatter(
            x=sub["sg_t2g_live"],
            y=sub["sg_short_live"],
            mode="markers+text" if show_text else "markers",
            name=label_map[status],
            text=sub["player_name"].apply(_short) if show_text else None,
            textposition="top center",
            textfont=dict(size=10, color="rgba(255,220,100,0.95)"),
            marker=dict(
                size=size_map[status],
                color=color_map[status],
                opacity=opacity_map[status],
                line=dict(
                    width=2 if show_text else 0.5,
                    color="rgba(0,0,0,0.4)" if show_text else "rgba(255,255,255,0.1)",
                ),
            ),
            customdata=np.stack([
                sub["player_name"],
                sub["sg_t2g_live"].round(2),
                sub["sg_short_live"].round(2),
                sub["position"].astype(str),
                sub["total"].apply(_par_fmt),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pos: %{customdata[3]} (%{customdata[4]})<br>"
                "T2G: %{customdata[1]:+}<br>"
                "Short: %{customdata[2]:+}<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=700,
        template="plotly_dark",
        margin=dict(l=50, r=30, t=20, b=60),
        xaxis=dict(
            title="Tee-to-Green (OTT + APP)",
            zeroline=False,
            gridcolor="rgba(255,255,255,0.04)",
            range=[-x_max, x_max],
        ),
        yaxis=dict(
            title="Short Game (ARG + PUTT)",
            zeroline=False,
            gridcolor="rgba(255,255,255,0.04)",
            range=[-y_max, y_max],
        ),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# G — Proximity table  (prox_fw and prox_rgh, top 15)
# ─────────────────────────────────────────────────────────────────────────────

def _render_proximity(df: pd.DataFrame) -> None:
    has_prox = "prox_fw" in df.columns or "prox_rgh" in df.columns
    if not has_prox:
        return

    _divider()
    st.markdown(f"<div style='{_SEC}'>Proximity to Hole</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Average proximity from fairway and rough · feet</div>",
        unsafe_allow_html=True,
    )

    prox_df = df[~df["position"].astype(str).str.upper().isin(["WD","CUT","DQ"])].copy()
    cols_available = [c for c in ["prox_fw", "prox_rgh"] if c in prox_df.columns]
    prox_df = prox_df.dropna(subset=cols_available, how="all").copy()

    if prox_df.empty:
        st.caption("No proximity data available.")
        return

    # Sort by prox_fw ascending (closer = better) if available, else prox_rgh
    sort_col = "prox_fw" if "prox_fw" in prox_df.columns else "prox_rgh"
    prox_df  = prox_df.dropna(subset=[sort_col]).nsmallest(15, sort_col)

    fig = go.Figure()

    if "prox_fw" in prox_df.columns:
        fig.add_trace(go.Bar(
            y=[_short(n) for n in prox_df["player_name"]],
            x=prox_df["prox_fw"],
            name="Fairway",
            orientation="h",
            marker=dict(color="rgba(56,189,248,0.75)"),
            hovertemplate="<b>%{y}</b> — Fairway: <b>%{x:.1f} ft</b><extra></extra>",
        ))

    if "prox_rgh" in prox_df.columns:
        fig.add_trace(go.Bar(
            y=[_short(n) for n in prox_df["player_name"]],
            x=prox_df["prox_rgh"],
            name="Rough",
            orientation="h",
            marker=dict(color="rgba(249,115,22,0.65)"),
            hovertemplate="<b>%{y}</b> — Rough: <b>%{x:.1f} ft</b><extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        height=max(300, len(prox_df) * 26 + 60),
        template="plotly_dark",
        margin=dict(l=10, r=20, t=10, b=40),
        xaxis=dict(
            title="Feet from hole (lower = better)",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11),
                   gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# H — SG quad panel  (top 10 per category: OTT, APP, ARG, PUTT)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sg_quad(df: pd.DataFrame) -> None:
    cats = [
        ("sg_ott",  "Off the Tee",   _SG_COLORS["sg_ott"]),
        ("sg_app",  "Approach",      _SG_COLORS["sg_app"]),
        ("sg_arg",  "Around Green",  _SG_COLORS["sg_arg"]),
        ("sg_putt", "Putting",       _SG_COLORS["sg_putt"]),
    ]
    available = [(c, lbl, col) for c, lbl, col in cats if c in df.columns]
    if not available:
        return

    _divider()
    st.markdown(f"<div style='{_SEC}'>SG Category Leaders</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Top 10 in each discipline this round</div>",
        unsafe_allow_html=True,
    )

    clean = df[~df["position"].astype(str).str.upper().isin(["WD","CUT","DQ"])].copy()

    cols = st.columns(len(available), gap="medium")
    for col_widget, (stat, label, color) in zip(cols, available):
        sub = clean.dropna(subset=[stat]).nlargest(10, stat)
        names = [_short(n) for n in sub["player_name"]]
        vals  = sub[stat].tolist()

        fig = go.Figure(go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(
                color=vals,
                colorscale=[[0, "rgba(239,68,68,0.6)"], [0.5, "rgba(100,100,100,0.4)"], [1, color]],
                cmin=min(vals) if vals else 0,
                cmax=max(vals) if vals else 1,
            ),
            hovertemplate="<b>%{y}</b>: %{x:+.2f}<extra></extra>",
            text=[f"{v:+.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=9, color="rgba(200,200,200,0.7)"),
        ))
        fig.update_layout(
            title=dict(text=label, font=dict(size=12, color="rgba(200,200,200,0.8)"), x=0.5),
            height=320,
            template="plotly_dark",
            margin=dict(l=0, r=40, t=36, b=20),
            xaxis=dict(
                zeroline=True, zerolinecolor="rgba(255,255,255,0.12)",
                gridcolor="rgba(255,255,255,0.04)", showticklabels=False,
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10),
                       gridcolor="rgba(255,255,255,0.04)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        col_widget.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# I — "Winning with what"  (stacked 100% bar for top 10 by total score)
# ─────────────────────────────────────────────────────────────────────────────

def _render_winning_with_what(df: pd.DataFrame) -> None:
    components = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    available  = [c for c in components if c in df.columns]
    if not available:
        return

    _divider()
    st.markdown(f"<div style='{_SEC}'>Winning With What</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>SG composition for top 10 on the leaderboard — where strokes are coming from</div>",
        unsafe_allow_html=True,
    )

    def _pos_num(p):
        try:
            return int(str(p).replace("T", ""))
        except Exception:
            return 999

    clean = df[~df["position"].astype(str).str.upper().isin(["WD","CUT","DQ"])].copy()
    clean["_pos_num"] = clean["position"].apply(_pos_num)
    top10 = clean.nsmallest(10, "_pos_num").copy()

    if top10.empty:
        return

    # Separate positive and negative contributions per player
    fig = go.Figure()

    for comp in available:
        pos_vals = top10[comp].clip(lower=0)
        neg_vals = top10[comp].clip(upper=0)
        label    = _SG_LABELS.get(comp, comp)
        color    = _SG_COLORS.get(comp, "#aaa")

        fig.add_trace(go.Bar(
            name=label,
            x=[_short(n) for n in top10["player_name"]],
            y=pos_vals,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b> — {label}: <b>%{{y:+.2f}}</b><extra></extra>",
            legendgroup=comp,
        ))
        # Negative portion (same color, slightly more transparent)
        if neg_vals.abs().sum() > 0:
            fig.add_trace(go.Bar(
                name=f"{label} (–)",
                x=[_short(n) for n in top10["player_name"]],
                y=neg_vals,
                marker_color=color,
                opacity=0.45,
                hovertemplate=f"<b>%{{x}}</b> — {label}: <b>%{{y:+.2f}}</b><extra></extra>",
                legendgroup=comp,
                showlegend=False,
            ))

    # SG total line
    if "sg_total" in top10.columns:
        fig.add_trace(go.Scatter(
            x=[_short(n) for n in top10["player_name"]],
            y=top10["sg_total"],
            mode="markers",
            name="SG Total",
            marker=dict(
                size=10, symbol="diamond",
                color="rgba(255,255,255,0.9)",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
            ),
            hovertemplate="<b>%{x}</b> — SG Total: <b>%{y:+.2f}</b><extra></extra>",
        ))

    fig.update_layout(
        barmode="relative",
        height=380,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=10, b=60),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11),
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
            gridcolor="rgba(255,255,255,0.05)", title="SG per Round",
        ),
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# J — Cut line strip  (projected cut on the leaderboard)
# ─────────────────────────────────────────────────────────────────────────────

def _render_cut_strip(df, cut_line=None):
    if cut_line is None:
        return

    clean = df[~df["position"].astype(str).str.upper().isin(["WD","DQ"])].copy()

    total_col = "total"
    if total_col not in clean.columns or clean[total_col].dropna().empty:
        return

    scores = clean[total_col].dropna().sort_values().reset_index(drop=True)
    n = len(scores)
    if n < 10:
        return

    # Infer cut position — standard 65+ties, use 65 as hard cutoff
    _cut_n = int(cut_line) if cut_line else 65
    cut_pos = min(_cut_n, n - 1)
    cut_score = float(scores.iloc[cut_pos - 1])

    _divider()
    st.markdown(f"<div style='{_SEC}'>Score Distribution & Projected Cut</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Projected cut line at {_par_fmt(cut_score)} "
        f"<span style='color:rgba(120,120,120,0.5)'>(top 65 + ties)</span></div>",
        unsafe_allow_html=True,
    )

    # Color each bar by above/below cut
    score_range = list(range(int(scores.min()), int(scores.max()) + 1))
    counts      = scores.value_counts().sort_index()
    bar_colors  = ["#34d399" if s <= cut_score else "rgba(100,100,100,0.4)"
                   for s in score_range]

    fig = go.Figure(go.Bar(
        x=[_par_fmt(s) for s in score_range],
        y=[int(counts.get(s, 0)) for s in score_range],
        marker_color=bar_colors,
        hovertemplate="Score %{x}: <b>%{y} players</b><extra></extra>",
    ))

    # Cut line annotation — use shape+annotation (add_vline doesn't work on categorical axes)
    cut_x_label = _par_fmt(cut_score)
    cut_x_idx   = score_range.index(int(cut_score)) if int(cut_score) in score_range else None
    if cut_x_idx is not None:
        fig.add_shape(
            type="line",
            x0=cut_x_idx - 0.5, x1=cut_x_idx - 0.5,
            y0=0, y1=1, yref="paper",
            line=dict(color="#fbbf24", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=cut_x_idx - 0.5, y=1, yref="paper",
            text=f"Cut {cut_x_label}",
            showarrow=False, xanchor="left",
            font=dict(size=11, color="#fbbf24"),
            bgcolor="rgba(0,0,0,0.4)",
        )

    fig.update_layout(
        height=260,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(title="Score to Par", tickfont=dict(size=11),
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Players", gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.15,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Bubble strip: only top OWGR players on the cut line
    # field_df is passed in via module-level; use session state to pass owgr map
    owgr_map = st.session_state.get("_live_owgr_map", {})  # dg_id -> owgr_rank

    bubble_df = clean[
        (clean[total_col] >= cut_score - 1) &
        (clean[total_col] <= cut_score + 1)
    ].copy().sort_values(total_col)

    if not bubble_df.empty:
        # Attach owgr_rank and filter to top-ranked players only (owgr <= 75, or all if no data)
        if owgr_map:
            bubble_df["_owgr"] = bubble_df["dg_id"].map(owgr_map)
            bubble_df = bubble_df[bubble_df["_owgr"].notna() & (bubble_df["_owgr"] <= 50)].copy()
            bubble_df = bubble_df.sort_values(["total", "_owgr"])

        if not bubble_df.empty:
            st.markdown(
                f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.08em;color:rgba(120,120,120,0.5);margin:12px 0 6px'>"
                f"Notable players on the bubble"
                f"<span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:10px;"
                f"color:rgba(100,100,100,0.4);margin-left:8px'>top 50 OWGR within 1 shot of cut</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            chips = ""
            for _, row in bubble_df.iterrows():
                score    = row.get(total_col, np.nan)
                name     = str(row.get("player_name", ""))
                thru     = _thru_fmt(row.get("thru", np.nan))
                s_fmt    = _par_fmt(score)
                owgr_val = int(row["_owgr"]) if "_owgr" in row.index and pd.notna(row["_owgr"]) else None
                safe     = "in" if pd.notna(score) and float(score) <= cut_score else "out"
                color    = "#34d399" if safe == "in" else "#ef4444"
                owgr_str = f"<span style='opacity:0.4;font-size:9px'> #{owgr_val}</span>" if owgr_val else ""
                chips += (
                    f"<span style='display:inline-block;margin:3px 4px;padding:5px 12px;"
                    f"border-radius:20px;border:1px solid {color};"
                    f"font-size:11px;color:{color};white-space:nowrap'>"
                    f"{name}{owgr_str} <b>{s_fmt}</b>"
                    f"<span style='opacity:0.5;font-size:10px'> thru {thru}</span>"
                    f"</span>"
                )
            st.markdown(
                f"<div style='display:flex;flex-wrap:wrap'>{chips}</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# K — Round delta scatter  (R1 vs R2 SG total)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_round(api_key: str, round_num: int) -> pd.DataFrame:
    params = {
        "stats":       "sg_total,sg_putt,sg_app,sg_ott,sg_arg",
        "round":       str(round_num),
        "display":     "value",
        "file_format": "csv",
        "key":         api_key,
    }
    resp = requests.get(_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ["event_name", "last_updated", "stat_display", "course"]:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan).ffill()
    num_cols = ["dg_id", "total", "round", "thru", "sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _render_round_delta(api_key: str, current_round: int) -> None:
    """
    Fetches R1 and R2 separately and plots SG total R1 vs R2.
    Only shown when current_round >= 2.
    """
    if current_round < 2:
        return

    _divider()
    st.markdown(f"<div style='{_SEC}'>Round-over-Round SG Delta</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>R1 vs R2 SG Total — who improved, who regressed</div>",
        unsafe_allow_html=True,
    )

    try:
        r1 = _fetch_round(api_key, 1)
        r2 = _fetch_round(api_key, 2)
    except Exception as e:
        st.caption(f"Could not load round data: {e}")
        return

    r1 = r1[["dg_id", "player_name", "sg_total"]].dropna(subset=["sg_total"]).copy()
    r2 = r2[["dg_id", "sg_total"]].dropna(subset=["sg_total"]).copy()
    r1 = r1.rename(columns={"sg_total": "sg_r1"})
    r2 = r2.rename(columns={"sg_total": "sg_r2"})

    merged = r1.merge(r2, on="dg_id", how="inner")
    if merged.empty:
        st.caption("R2 data not yet available.")
        return

    merged["delta"]   = merged["sg_r2"] - merged["sg_r1"]
    merged["_color"]  = merged["delta"].apply(
        lambda d: "#34d399" if d > 0.5 else "#ef4444" if d < -0.5 else "rgba(160,160,160,0.6)"
    )
    merged["_size"]   = merged["delta"].abs().clip(0.1, 5) * 4 + 6
    merged["_label"]  = merged["player_name"].apply(_short)

    # Quadrant labels
    x_mid = merged["sg_r1"].median()
    y_mid = merged["sg_r2"].median()

    fig = go.Figure()

    # Diagonal reference (same both rounds)
    lim = max(abs(merged["sg_r1"].max()), abs(merged["sg_r2"].max()), 3) + 0.5
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.08)", dash="dot", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=merged["sg_r1"],
        y=merged["sg_r2"],
        mode="markers+text",
        text=merged.apply(
            lambda r: _short(r["player_name"]) if abs(r["delta"]) > 1.5 else "",
            axis=1,
        ),
        textposition="top center",
        textfont=dict(size=9, color="rgba(220,220,220,0.75)"),
        marker=dict(
            size=merged["_size"],
            color=merged["delta"],
            colorscale=[
                [0,   "#ef4444"],
                [0.5, "rgba(130,130,130,0.5)"],
                [1,   "#34d399"],
            ],
            cmin=-4, cmax=4,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        ),
        customdata=np.stack([
            merged["_label"],
            merged["sg_r1"],
            merged["sg_r2"],
            merged["delta"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "R1 SG: %{customdata[1]:+.2f}<br>"
            "R2 SG: %{customdata[2]:+.2f}<br>"
            "Delta: %{customdata[3]:+.2f}<extra></extra>"
        ),
        showlegend=False,
    ))

    # Quadrant shading
    fig.add_hrect(y0=y_mid, y1=lim,  fillcolor="rgba(52,211,153,0.03)", line_width=0)
    fig.add_hrect(y0=-lim,  y1=y_mid, fillcolor="rgba(239,68,68,0.03)",  line_width=0)

    # Quadrant annotations
    fig.add_annotation(x=-lim+0.2, y=lim-0.3,  text="Improving",   showarrow=False,
                       font=dict(size=9, color="rgba(52,211,153,0.35)"),  xanchor="left")
    fig.add_annotation(x=lim-0.2,  y=-lim+0.3, text="Fading",      showarrow=False,
                       font=dict(size=9, color="rgba(239,68,68,0.35)"),   xanchor="right")
    fig.add_annotation(x=-lim+0.2, y=-lim+0.3, text="Struggling",  showarrow=False,
                       font=dict(size=9, color="rgba(160,160,160,0.3)"),  xanchor="left")
    fig.add_annotation(x=lim-0.2,  y=lim-0.3,  text="Consistent",  showarrow=False,
                       font=dict(size=9, color="rgba(160,160,160,0.3)"),  xanchor="right")

    fig.update_layout(
        height=520,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(
            title="R1 SG Total",
            range=[-lim, lim],
            zeroline=True, zerolinecolor="rgba(255,255,255,0.1)",
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            title="R2 SG Total",
            range=[-lim, lim],
            zeroline=True, zerolinecolor="rgba(255,255,255,0.1)",
            gridcolor="rgba(255,255,255,0.04)",
        ),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Top movers table below scatter
    merged_sorted = merged.sort_values("delta", ascending=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    top_up   = merged_sorted.head(5)
    top_down = merged_sorted.tail(5).iloc[::-1]

    col_u, col_d = st.columns(2, gap="large")

    def _delta_rows(rows, up: bool) -> str:
        html = ""
        for _, row in rows.iterrows():
            d     = row["delta"]
            color = "#34d399" if up else "#ef4444"
            arrow = "▲" if up else "▼"
            html += (
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>"
                f"<span style='font-size:13px;font-weight:600;color:rgba(215,215,215,0.9)'>"
                f"{_short(row['player_name'])}</span>"
                f"<span style='font-size:13px;font-weight:800;color:{color}'>"
                f"{arrow} {abs(d):.2f}</span>"
                f"</div>"
            )
        return html

    with col_u:
        st.markdown(
            f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:#34d399;margin-bottom:8px'>Biggest improvers</div>"
            + _delta_rows(top_up, up=True),
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:#ef4444;margin-bottom:8px'>Biggest fallers</div>"
            + _delta_rows(top_down, up=False),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# L — SG success analysis  (what's driving leaderboard performance)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sg_success_analysis(df: pd.DataFrame) -> None:
    sg_components = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    sg_available  = [c for c in sg_components if c in df.columns]
    if len(sg_available) < 2:
        return

    clean = df[~df["position"].astype(str).str.upper().isin(["WD","CUT","DQ"])].copy()
    if len(clean) < 15:
        return

    def _pos_num(p):
        try:
            return int(str(p).replace("T",""))
        except Exception:
            return 999

    clean["_pos_num"] = clean["position"].apply(_pos_num)
    n      = len(clean)
    top10  = clean.nsmallest(10, "_pos_num")
    bottom = clean.nlargest(max(10, n // 3), "_pos_num")

    # SG axes
    sg_cats   = [c for c in sg_available]
    sg_labels = [_SG_LABELS.get(c, c) for c in sg_cats]
    sg_top    = [top10[c].mean()  for c in sg_cats]
    sg_bot    = [bottom[c].mean() for c in sg_cats]
    sg_all    = [clean[c].mean()  for c in sg_cats]
    deltas    = [t - b for t, b in zip(sg_top, sg_bot)]

    # Extra axes: GIR and scrambling — convert to pct if 0-1
    extra_axes = []
    for col, lbl in [("gir", "GIR"), ("scrambling", "Scramble")]:
        if col in clean.columns:
            scale = 100.0 if clean[col].dropna().max() <= 1.0 else 1.0
            t_val = top10[col].mean()  * scale
            b_val = bottom[col].mean() * scale
            a_val = clean[col].mean()  * scale
            extra_axes.append((lbl, t_val, b_val, a_val))

    all_labels = sg_labels + [e[0] for e in extra_axes]
    top_raw    = sg_top    + [e[1] for e in extra_axes]
    bot_raw    = sg_bot    + [e[2] for e in extra_axes]
    all_raw    = sg_all    + [e[3] for e in extra_axes]

    # Normalize each axis independently: divide by field std so all axes ≈ same range
    # Use the spread across [bot, field, top] per axis as the normalizing factor
    norm_top = []
    norm_bot = []
    norm_all = []

    # Compute SG spread to use as the common reference scale
    sg_spread = max(
        max(abs(sg_top[i] - sg_all[i]) for i in range(len(sg_cats))),
        max(abs(sg_bot[i] - sg_all[i]) for i in range(len(sg_cats))),
        0.01
    )

    for i in range(len(all_labels)):
        is_extra = i >= len(sg_cats)
        t_delta = top_raw[i] - all_raw[i]
        b_delta = bot_raw[i] - all_raw[i]
        if is_extra:
            # Cap extra axes at 50% of SG scale so they don't dominate visually
            axis_spread = max(abs(t_delta), abs(b_delta), 0.01)
            scale_factor = min(sg_spread * 0.5 / axis_spread, 1.0)
            norm_top.append(t_delta * scale_factor / sg_spread)
            norm_bot.append(b_delta * scale_factor / sg_spread)
        else:
            norm_top.append(t_delta / sg_spread)
            norm_bot.append(b_delta / sg_spread)
        norm_all.append(0.0)

    # Offset so all values > 0 (radar collapses on negatives)
    offset = abs(min(norm_bot)) + 0.4
    top_r  = [v + offset for v in norm_top] + [norm_top[0] + offset]
    bot_r  = [v + offset for v in norm_bot] + [norm_bot[0] + offset]
    all_r  = [offset] * (len(all_labels) + 1)  # field avg = flat ring
    theta  = all_labels + [all_labels[0]]

    # Hover shows real values
    top_hover = top_raw + [top_raw[0]]
    bot_hover = bot_raw + [bot_raw[0]]
    all_hover = all_raw + [all_raw[0]]

    _divider()
    st.markdown(f"<div style='{_SEC}'>What's Driving Success This Week</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>SG profile + GIR & scrambling — Top 10 vs Field vs Bottom third</div>",
        unsafe_allow_html=True,
    )

    # Full width spider
    fig = go.Figure()

        # Bottom third — faint outline only
    fig.add_trace(go.Scatterpolar(
            r=bot_r, theta=theta,
            fill="toself",
            fillcolor="rgba(100,100,100,0.07)",
            line=dict(color="rgba(140,140,140,0.25)", width=1, dash="dot"),
            name="Bottom third",
            customdata=bot_hover,
            hovertemplate="<b>%{theta}</b>: %{customdata:+.2f}<extra>Bottom third</extra>",
        ))
        # Field avg
    fig.add_trace(go.Scatterpolar(
            r=all_r, theta=theta,
            fill="toself",
            fillcolor="rgba(180,180,180,0.08)",
            line=dict(color="rgba(180,180,180,0.4)", width=1.5, dash="dash"),
            name="Field avg",
            customdata=all_hover,
            hovertemplate="<b>%{theta}</b>: %{customdata:+.2f}<extra>Field avg</extra>",
        ))
        # Top 10 — bold filled
    fig.add_trace(go.Scatterpolar(
            r=top_r, theta=theta,
            fill="toself",
            fillcolor="rgba(251,191,36,0.25)",
            line=dict(color="rgba(251,191,36,1.0)", width=3),
            name="Top 10",
            customdata=top_hover,
            hovertemplate="<b>%{theta}</b>: %{customdata:+.2f}<extra>Top 10</extra>",
            marker=dict(size=8, color="rgba(251,191,36,1.0)"),
        ))

    r_max = max(top_r) * 1.18
    fig.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.02)",
                radialaxis=dict(visible=False, range=[0, r_max]),
                angularaxis=dict(
                    tickfont=dict(size=14, color="rgba(220,220,220,0.85)"),
                    linecolor="rgba(255,255,255,0.06)",
                    gridcolor="rgba(255,255,255,0.07)",
                ),
            ),
            height=500,
            template="plotly_dark",
            margin=dict(l=60, r=60, t=50, b=60),
            legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center",
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(
                bgcolor="rgba(20,20,20,0.92)",
                font=dict(color="rgba(220,220,220,0.9)", size=12),
                bordercolor="rgba(255,255,255,0.1)",
            ),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Delta bars as a horizontal row below the spider
    sorted_cats = sorted(zip(deltas, sg_cats, sg_labels), reverse=True)
    max_delta   = max(abs(d) for d in deltas) or 1
    delta_cols  = st.columns(len(sorted_cats), gap="medium")
    for col_w, (delta, cat, label) in zip(delta_cols, sorted_cats):
        color = _SG_COLORS.get(cat, "#aaa")
        pct   = abs(delta) / max_delta
        w_pct = max(8, int(pct * 100))
        with col_w:
            st.markdown(
                f"<div style='text-align:center;padding:8px 4px'>"
                f"<div style='font-size:11px;font-weight:700;color:rgba(180,180,180,0.7);"
                f"margin-bottom:6px'>{label}</div>"
                f"<div style='font-size:20px;font-weight:900;color:{color};"
                f"margin-bottom:8px'>{delta:+.2f}</div>"
                f"<div style='background:rgba(255,255,255,0.06);border-radius:4px;height:6px;width:100%'>"
                f"<div style='width:{w_pct}%;height:6px;background:{color};"
                f"border-radius:4px;margin:0 auto;opacity:0.85'></div>"
                f"</div>"
                f"<div style='font-size:9px;color:rgba(100,100,100,0.5);margin-top:4px'>"
                f"top 10 edge</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Key separator callout
    top_cat_delta, top_cat, top_label = sorted_cats[0]
    _tc = _SG_COLORS.get(top_cat, "#aaa")
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);"
        f"border-radius:10px;padding:10px 16px;margin-top:8px;display:inline-block'>"
        f"<span style='font-size:10px;color:rgba(120,120,120,0.5)'>Key separator: </span>"
        f"<span style='font-size:13px;font-weight:700;color:{_tc};margin:0 8px'>{top_label}</span>"
        f"<span style='font-size:11px;color:rgba(160,160,160,0.6)'>leaders gaining "
        f"<b style='color:rgba(220,220,220,0.8)'>{top_cat_delta:+.2f}</b> more than bottom third</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_live_tab(*, field_df, id_to_img, cut_line=None):
    """
    Main entry point — called from Stats.py.

    Parameters
    ----------
    field_df : this_week_field.csv loaded as a DataFrame (used for OWGR)
    id_to_img : dg_id → headshot URL dict
    cut_line  : int or None — cut line from schedule
    """
    api_key = st.secrets.get("DATAGOLF_API_KEY", "")
    if not api_key:
        st.error("DATAGOLF_API_KEY not configured.")
        return

    # Build OWGR map from field file
    if field_df is not None and not field_df.empty:
        fdf = field_df.copy()
        fdf.columns = [c.lower().strip() for c in fdf.columns]
        if "dg_id" in fdf.columns and "owgr_rank" in fdf.columns:
            fdf["dg_id"]     = pd.to_numeric(fdf["dg_id"],     errors="coerce")
            fdf["owgr_rank"] = pd.to_numeric(fdf["owgr_rank"], errors="coerce")
            st.session_state["_live_owgr_map"] = (
                fdf.dropna(subset=["dg_id","owgr_rank"])
                   .set_index("dg_id")["owgr_rank"]
                   .astype(int)
                   .to_dict()
            )

    # Fetch live stats and DK odds in parallel (both cached)
    with st.spinner("Fetching live data…"):
        try:
            df = _load_live(api_key, "event_avg")
        except Exception as e:
            st.error(f"Could not fetch live data: {e}")
            return

    if df.empty:
        st.info("No live data returned — the tournament may not have started yet.")
        return

    # Merge DK odds — fetched from API, cached 30 min in memory
    dk_map = _fetch_dk_odds(api_key)
    df["dg_id"]   = pd.to_numeric(df["dg_id"], errors="coerce")
    df["dk_odds"] = df["dg_id"].map(lambda x: dk_map.get(int(x)) if pd.notna(x) else np.nan)

    _render_header(df)

    # Sections
    _render_hero_row(df)
    _render_movers(df)
    _render_leaderboard(df, id_to_img or {})
    _render_cut_strip(df)
    _render_winning_with_what(df)
    _render_sg_quad(df)
    _render_strengths_map(df, cut_line=cut_line)
