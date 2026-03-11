from __future__ import annotations

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import timedelta
from urllib.parse import urljoin

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL  = "http://api.weatherapi.com/v1/"
_C_WIND    = "#4C78A8"   # blue  — wind
_C_GUST    = "#F58518"   # orange — gust
_C_RAIN    = "#54A24B"   # green — chance of rain
_C_PRECIP  = "#72B7B2"   # teal  — precip
_C_TEMP    = "#E45756"   # red   — temperature
_C_AM      = "#38bdf8"   # early wave marker
_C_PM      = "#f97316"   # late wave marker

_WIND_THRESHOLDS = [
    (25, "#ef4444", "Extreme — likely scoring carnage"),
    (18, "#f97316", "Very windy — major scoring factor"),
    (12, "#fbbf24", "Moderate — will matter on exposed holes"),
    (0,  "#22c55e", "Benign — minimal wind impact"),
]

_SEC = ("font-size:11px;font-weight:700;letter-spacing:0.08em;"
        "color:rgba(130,130,130,0.7);text-transform:uppercase;margin-bottom:8px")
_SUB = "font-size:12px;color:rgba(140,140,140,0.5);margin-bottom:10px"


# ─────────────────────────────────────────────────────────────────────────────
# WeatherAPI helpers
# ─────────────────────────────────────────────────────────────────────────────

class WeatherAPIError(RuntimeError):
    pass


def _api_get(endpoint: str, params: dict, timeout: int = 20) -> dict:
    url = urljoin(_BASE_URL, endpoint.lstrip("/"))
    r = requests.get(url, params=params, timeout=timeout)
    try:
        data = r.json()
    except Exception as e:
        raise WeatherAPIError(f"Non-JSON response ({r.status_code}): {r.text[:200]}") from e
    if r.status_code != 200:
        msg = data.get("error", {}).get("message") or str(data)
        raise WeatherAPIError(f"WeatherAPI error ({r.status_code}): {msg}")
    return data


@st.cache_data(ttl=1800, show_spinner=False)   # cache 30 min
def _fetch_forecast(api_key: str, q: str, days: int) -> dict:
    return _api_get(
        "forecast.json",
        {"key": api_key, "q": q, "days": days, "aqi": "no", "alerts": "no"},
    )


def _forecast_to_hourly(data: dict) -> pd.DataFrame:
    loc  = data["location"]
    rows = []
    for fd in data["forecast"]["forecastday"]:
        for h in fd["hour"]:
            rows.append({
                "dt_local":       h.get("time"),
                "date":           fd.get("date"),
                "temp_f":         h.get("temp_f"),
                "feelslike_f":    h.get("feelslike_f"),
                "wind_mph":       h.get("wind_mph"),
                "wind_dir":       h.get("wind_dir"),
                "gust_mph":       h.get("gust_mph"),
                "chance_of_rain": h.get("chance_of_rain"),
                "precip_in":      h.get("precip_in"),
                "cloud":          h.get("cloud"),
                "condition":      (h.get("condition") or {}).get("text", ""),
                "location_name":  loc.get("name", ""),
                "tz_id":          loc.get("tz_id", ""),
            })
    df = pd.DataFrame(rows)
    df["dt_local"] = pd.to_datetime(df["dt_local"], errors="coerce")
    for c in ["temp_f","feelslike_f","wind_mph","gust_mph","chance_of_rain","precip_in","cloud"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("dt_local").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tee time helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tee_times(path: str) -> pd.DataFrame:
    """
    Load this_week_field.csv.
    Expected columns: player_name, dg_id, r1_teetime, r1_wave, r2_teetime, r2_wave,
                      r3_teetime (optional), r4_teetime (optional)
    """
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        for rnd in [1, 2, 3, 4]:
            col = f"r{rnd}_teetime"
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _get_wave_times(tee_df: pd.DataFrame, round_num: int) -> dict[str, list]:
    """
    Returns {"early": [datetime, ...], "late": [datetime, ...]}
    for the two tee time waves of a given round.
    Uses r{n}_wave to split AM/PM. Falls back to splitting on median time.
    """
    col_time = f"r{round_num}_teetime"
    col_wave = f"r{round_num}_wave"

    if tee_df.empty or col_time not in tee_df.columns:
        return {"early": [], "late": []}

    df = tee_df[[col_time] + ([col_wave] if col_wave in tee_df.columns else [])].dropna(subset=[col_time])
    if df.empty:
        return {"early": [], "late": []}

    if col_wave in df.columns:
        wave_vals = df[col_wave].astype(str).str.lower().str.strip()
        early_mask = wave_vals.isin(["1", "am", "early", "morning", "a"])
        late_mask  = ~early_mask
    else:
        # fallback: split on median tee time
        med = df[col_time].median()
        early_mask = df[col_time] <= med
        late_mask  = ~early_mask

    # Deduplicate to unique tee times (one dot per distinct time slot)
    early = sorted(df.loc[early_mask, col_time].dropna().unique().tolist())
    late  = sorted(df.loc[late_mask,  col_time].dropna().unique().tolist())
    return {"early": early, "late": late}


# ─────────────────────────────────────────────────────────────────────────────
# Wind condition badge
# ─────────────────────────────────────────────────────────────────────────────

def _get_top_groups(
    tee_df: pd.DataFrame,
    round_num: int,
    n_groups: int = 8,
    featured_dg_ids: list[int] | None = None,
) -> list[dict]:
    """
    Returns groups containing featured players (from favourites + hottest panels).
    If featured_dg_ids is empty/None, falls back to top n_groups by best OWGR.
    Groups are deduplicated — if two featured players share a tee time, the group
    appears only once. Within each group all players are shown.
    Each entry: {"teetime": pd.Timestamp, "players": [str, ...], "best_owgr": int, "start_hole": str}
    """
    col_time = f"r{round_num}_teetime"
    col_hole = f"r{round_num}_start_hole"

    if tee_df.empty or col_time not in tee_df.columns:
        return []

    needed = [col_time, "player_name"]
    if "dg_id" in tee_df.columns:
        needed.append("dg_id")
    if col_hole in tee_df.columns:
        needed.append(col_hole)
    if "owgr_rank" in tee_df.columns:
        needed.append("owgr_rank")

    df = tee_df[[c for c in needed if c in tee_df.columns]].dropna(subset=[col_time, "player_name"]).copy()
    if df.empty:
        return []

    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    df = df.dropna(subset=[col_time])

    if "owgr_rank" in df.columns:
        df["owgr_rank"] = pd.to_numeric(df["owgr_rank"], errors="coerce")
    else:
        df["owgr_rank"] = 9999

    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")

    group_keys = [col_time] + ([col_hole] if col_hole in df.columns else [])
    all_groups = []
    for key_vals, grp in df.groupby(group_keys):
        teetime    = grp[col_time].iloc[0]
        players    = grp["player_name"].dropna().tolist()
        dg_ids     = set(grp["dg_id"].dropna().astype(int).tolist()) if "dg_id" in grp.columns else set()
        best_owgr  = int(grp["owgr_rank"].dropna().min()) if grp["owgr_rank"].notna().any() else 9999
        start_hole = str(int(grp[col_hole].iloc[0])) if col_hole in grp.columns and pd.notna(grp[col_hole].iloc[0]) else "—"
        all_groups.append({
            "teetime": teetime, "players": players, "dg_ids": dg_ids,
            "best_owgr": best_owgr, "start_hole": start_hole,
        })

    # Filter to groups containing a featured player; fallback to OWGR-ranked if none
    if featured_dg_ids:
        featured_set = set(featured_dg_ids)
        filtered = [g for g in all_groups if g["dg_ids"] & featured_set]
        if filtered:
            filtered.sort(key=lambda g: g["teetime"])
            return filtered  # show all matched groups (already deduped by group key)

    # Fallback: top n_groups by OWGR
    all_groups.sort(key=lambda g: g["best_owgr"])
    top = all_groups[:n_groups]
    top.sort(key=lambda g: g["teetime"])
    return top


def _wind_badge(peak_wind: float) -> str:
    for threshold, color, label in _WIND_THRESHOLDS:
        if peak_wind >= threshold:
            return (
                f"<div style='display:inline-block;background:{color}22;border:1px solid {color};"
                f"border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700;color:{color}'>"
                f"Peak {peak_wind:.0f} mph — {label}</div>"
            )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Daily summary cards
# ─────────────────────────────────────────────────────────────────────────────

def _render_daily_cards(hourly_df: pd.DataFrame, round_dates: list[pd.Timestamp]):
    """Four compact summary cards, one per round day."""
    cols = st.columns(4, gap="small")
    for ci, rdate in enumerate(round_dates):
        day = hourly_df[hourly_df["dt_local"].dt.date == rdate.date()].copy()
        with cols[ci]:
            if day.empty:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);"
                    f"border-radius:10px;padding:12px;text-align:center'>"
                    f"<div style='font-size:11px;font-weight:700;color:rgba(130,130,130,0.6)'>R{ci+1}</div>"
                    f"<div style='font-size:12px;color:rgba(100,100,100,0.5);margin-top:6px'>No forecast</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                continue

            peak_wind  = float(day["wind_mph"].max())
            peak_gust  = float(day["gust_mph"].max())
            max_temp   = float(day["temp_f"].max())
            min_temp   = float(day["temp_f"].min())
            max_rain   = float(day["chance_of_rain"].max())
            cond       = day.iloc[len(day)//2]["condition"]  # midday condition

            # wind color
            wc = next(c for t, c, _ in _WIND_THRESHOLDS if peak_wind >= t)

            rain_s = f"{max_rain:.0f}%" if max_rain > 5 else "—"
            rain_c = "#60a5fa" if max_rain > 40 else "#f97316" if max_rain > 15 else "rgba(140,140,140,0.5)"

            st.markdown(
                f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);"
                f"border-radius:10px;padding:12px 10px;text-align:center'>"
                f"<div style='font-size:11px;font-weight:700;color:rgba(130,130,130,0.6);text-transform:uppercase'>R{ci+1} · {rdate.strftime('%a %b %d')}</div>"
                f"<div style='font-size:22px;font-weight:800;color:{wc};margin:6px 0 2px'>{peak_wind:.0f}</div>"
                f"<div style='font-size:10px;color:rgba(120,120,120,0.55)'>peak mph · gusts {peak_gust:.0f}</div>"
                f"<div style='margin-top:8px;font-size:12px;color:rgba(200,200,200,0.7)'>{max_temp:.0f}° / {min_temp:.0f}°F</div>"
                f"<div style='font-size:12px;font-weight:700;color:{rain_c};margin-top:2px'>Rain {rain_s}</div>"
                f"<div style='font-size:10px;color:rgba(110,110,110,0.55);margin-top:4px'>{cond}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main chart — single wind panel, rain strip, group timeline
# ─────────────────────────────────────────────────────────────────────────────

def _render_weather_chart(
    day_df: pd.DataFrame,
    round_num: int,
    round_date: pd.Timestamp,
    wave_times: dict,
    top_groups: list | None = None,
):
    if day_df.empty:
        st.info(f"No forecast data available for R{round_num} ({round_date.strftime('%b %d')}).")
        return

    x      = day_df["dt_local"]
    wind   = day_df["wind_mph"]
    gust   = day_df["gust_mph"]
    rain   = day_df["chance_of_rain"].clip(0, 100)
    precip = day_df["precip_in"].clip(lower=0)
    temp   = day_df["temp_f"]

    peak_wind = float(wind.max())
    wind_color, _, _ = next((c, s, d) for t, c, s, d in [
        (25, "#ef4444", "Extreme", "Extreme — likely scoring carnage"),
        (18, "#f97316", "Very windy", "Very windy — major scoring factor"),
        (12, "#fbbf24", "Moderate", "Moderate — will matter on exposed holes"),
        (0,  "#22c55e", "Benign", "Benign — minimal wind impact"),
    ] if peak_wind >= t)

    # ── Single wind panel ─────────────────────────────────────────────────────
    fig = go.Figure()

    # Wave background tints — AM steel blue, PM warm amber, clearly distinct
    for wave_key, fill_color in [
        ("early", "rgba(56,189,248,0.10)"),   # steel blue
        ("late",  "rgba(251,191,36,0.08)"),    # warm amber
    ]:
        times = wave_times.get(wave_key, [])
        if not times:
            continue
        t_start = pd.Timestamp(min(times))
        t_end   = pd.Timestamp(max(times))
        fig.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor=fill_color, line_width=0,
        )

    # Gust fill area
    fig.add_trace(go.Scatter(
        x=pd.concat([x, x.iloc[::-1]]),
        y=pd.concat([gust, wind.iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(245,133,24,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Gust line — thin, muted, same color family as wind
    fig.add_trace(go.Scatter(
        x=x, y=gust,
        mode="lines",
        line=dict(color="rgba(76,120,168,0.35)", width=1, dash="dot"),
        name="Gust",
        hovertemplate="Gust: <b>%{y:.0f} mph</b><extra></extra>",
    ))

    # Wind line — always blue, fill tinted by severity
    fig.add_trace(go.Scatter(
        x=x, y=wind,
        mode="lines",
        line=dict(color=_C_WIND, width=3),
        name="Wind",
        hovertemplate="%{x|%H:%M} — <b>%{y:.0f} mph</b><extra></extra>",
        fill="tozeroy",
        fillcolor=f"rgba({int(wind_color[1:3],16)},{int(wind_color[3:5],16)},{int(wind_color[5:7],16)},0.08)",
    ))

    # Rain chance — scaled to primary y-axis so 100% = top of chart
    y_max = max(float(gust.max()) * 1.15, 5)
    rain_scaled = rain / 100.0 * y_max
    fig.add_trace(go.Scatter(
        x=x, y=rain_scaled,
        mode="lines",
        line=dict(color="rgba(84,162,75,0.0)"),
        fill="tozeroy",
        fillcolor="rgba(84,162,75,0.15)",
        name="Rain chance",
        customdata=rain,
        hovertemplate="Rain: <b>%{customdata:.0f}%</b><extra></extra>",
    ))

    # Top 3 peak rain labels — above the fill at the peak x position
    import numpy as np
    rain_vals   = rain.values
    scaled_vals = rain_scaled.values
    x_vals      = x.values
    peaks = []
    for i in range(1, len(rain_vals) - 1):
        v = float(rain_vals[i])
        if v >= 20 and v >= float(rain_vals[i - 1]) and v >= float(rain_vals[i + 1]):
            peaks.append((v, i))
    peaks.sort(reverse=True)
    labeled = []
    for v, i in peaks:
        too_close = any(
            abs(int((x_vals[i] - px) / np.timedelta64(1, 'm'))) < 120
            for px in labeled
        )
        if not too_close and len(labeled) < 3:
            fig.add_annotation(
                x=pd.Timestamp(x_vals[i]).isoformat(), y=float(scaled_vals[i]),
                xref="x", yref="y",
                text=f"{v:.0f}%",
                showarrow=False,
                font=dict(size=10, color="rgba(84,162,75,1.0)", family="monospace"),
                yanchor="bottom", xanchor="center",
                yshift=6,
            )
            labeled.append(x_vals[i])

    # Temp line — right axis, very subtle
    fig.add_trace(go.Scatter(
        x=x, y=temp,
        mode="lines",
        line=dict(color="rgba(228,87,86,0.35)", width=1, dash="dot"),
        name="Temp °F",
        yaxis="y3",
        hovertemplate="Temp: <b>%{y:.0f}°F</b><extra></extra>",
    ))

    # x clip
    x_min = round_date.replace(hour=6,  minute=0)
    x_max = round_date.replace(hour=19, minute=0)

    fig.update_layout(
        height=300,
        template="plotly_dark",
        margin=dict(l=40, r=100, t=20, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        showlegend=False,
        yaxis=dict(
            range=[0, y_max],
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=10),
            title=dict(text="mph", font=dict(size=9, color="rgba(150,150,150,0.5)")),
        ),

        yaxis3=dict(
            overlaying="y", side="right",
            showgrid=False,
            tickfont=dict(size=9, color="rgba(228,87,86,0.4)"),
            title=dict(text="°F", font=dict(size=9, color="rgba(228,87,86,0.4)")),
            range=[float(temp.min()) - 15, float(temp.max()) + 15],
        ),
        xaxis=dict(
            range=[x_min, x_max],
            tickformat="%-I%p",
            gridcolor="rgba(255,255,255,0.03)",
            tickfont=dict(size=10),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Wind badge — inline, compact
    badge = _wind_badge(peak_wind)
    if badge:
        st.markdown(badge, unsafe_allow_html=True)

    # ── Chart legend ──────────────────────────────────────────────────────────
    def _swatch(color: str, shape: str = "line") -> str:
        if shape == "line":
            return f"<span style='display:inline-block;width:22px;height:2px;background:{color};vertical-align:middle;margin-right:5px;border-radius:1px'></span>"
        if shape == "dot-line":
            return (
                f"<span style='display:inline-block;width:22px;height:2px;"
                f"background:repeating-linear-gradient(90deg,{color} 0,{color} 4px,transparent 4px,transparent 7px);"
                f"vertical-align:middle;margin-right:5px'></span>"
            )
        if shape == "fill":
            return f"<span style='display:inline-block;width:22px;height:10px;background:{color};vertical-align:middle;margin-right:5px;border-radius:2px'></span>"

    fill_hex  = wind_color
    fill_rgba = f"rgba({int(fill_hex[1:3],16)},{int(fill_hex[3:5],16)},{int(fill_hex[5:7],16)},0.25)"

    items = [
        (_swatch("#4C78A8", "line"),     "Wind speed"),
        (_swatch("rgba(76,120,168,0.4)", "dot-line"), "Gust"),
        (_swatch(fill_rgba, "fill"),     "Wind fill — color = severity"),
        (_swatch("rgba(228,87,86,0.4)", "dot-line"), "Temp °F"),
        (_swatch("rgba(84,162,75,0.5)", "fill"),     "Rain chance"),
        (_swatch("rgba(56,189,248,0.18)", "fill"),   "AM wave window"),
        (_swatch("rgba(251,191,36,0.18)", "fill"),   "PM wave window"),
    ]

    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:14px 20px;margin-top:10px;margin-bottom:4px'>"
    for swatch, label in items:
        legend_html += (
            f"<span style='display:flex;align-items:center;font-size:10px;"
            f"color:rgba(130,130,130,0.55)'>{swatch}{label}</span>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


def _render_group_timeline(top_groups: list, round_date: pd.Timestamp) -> None:
    """
    Timeline strip: numbered dots linked to rows below.
    AM groups = steel blue, PM groups = amber.
    """
    if not top_groups:
        return

    day_start  = round_date.replace(hour=6,  minute=0)
    day_end    = round_date.replace(hour=19, minute=0)
    total_mins = (day_end - day_start).seconds / 60

    def pct(t: pd.Timestamp) -> float:
        mins = max(0, min((t - day_start).seconds / 60, total_mins))
        return mins / total_mins * 100

    def wave_color(t: pd.Timestamp) -> str:
        return "#38bdf8" if t.hour < 12 else "#fbbf24"

    st.markdown(
        "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.08em;color:rgba(100,100,100,0.5);margin:20px 0 14px'>Featured groups</div>",
        unsafe_allow_html=True,
    )

    # Timeline bar with numbered dots
    bar_html = (
        "<div style='position:relative;height:2px;background:rgba(255,255,255,0.07);"
        "margin:8px 0 36px;border-radius:1px'>"
    )
    for gi, grp in enumerate(top_groups):
        p   = pct(grp["teetime"])
        col = wave_color(grp["teetime"])
        num = gi + 1
        bar_html += (
            f"<div style='position:absolute;left:{p:.1f}%;top:-10px;transform:translateX(-50%);"
            f"width:20px;height:20px;border-radius:50%;background:{col};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:10px;font-weight:700;color:rgba(0,0,0,0.75)'>{num}</div>"
        )
    bar_html += "</div>"
    st.markdown(bar_html, unsafe_allow_html=True)

    # Group rows with matching number badge
    rows_html = ""
    for gi, grp in enumerate(top_groups):
        col      = wave_color(grp["teetime"])
        num      = gi + 1
        tee_str  = grp["teetime"].strftime("%-I:%M %p")
        hole_str = f"· H{grp['start_hole']}" if grp.get("start_hole", "—") != "—" else ""
        names    = "  ·  ".join(grp["players"])
        badge    = (
            f"<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"width:18px;height:18px;border-radius:50%;background:{col};"
            f"font-size:9px;font-weight:700;color:rgba(0,0,0,0.75);flex-shrink:0'>{num}</span>"
        )
        rows_html += (
            f"<div style='display:flex;align-items:center;gap:10px;padding:5px 0;"
            f"border-bottom:1px solid rgba(255,255,255,0.04)'>"
            f"{badge}"
            f"<span style='font-size:11px;color:rgba(150,150,150,0.55);white-space:nowrap;min-width:80px'>{tee_str}{hole_str}</span>"
            f"<span style='font-size:12px;color:rgba(215,215,215,0.9);font-weight:500'>{names}</span>"
            f"</div>"
        )

    st.markdown(f"<div style='margin-bottom:12px'>{rows_html}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _render_group_table(top_groups: list) -> None:
    """Render a clean legend table of featured groups below the chart."""
    if not top_groups:
        return

    _CIRCLE = ["①","②","③","④","⑤","⑥","⑦","⑧"]

    rows_html = ""
    for gi, grp in enumerate(top_groups):
        symbol     = _CIRCLE[gi] if gi < len(_CIRCLE) else str(gi + 1)
        tee_str    = grp["teetime"].strftime("%-I:%M %p")
        hole_str   = f"Hole {grp['start_hole']}" if grp.get("start_hole", "—") != "—" else ""
        meta       = f"{tee_str}  {hole_str}".strip()
        names      = "  ·  ".join(grp["players"])
        owgr_str   = f"OWGR #{grp['best_owgr']}" if grp['best_owgr'] < 9999 else ""
        rows_html += (
            f"<tr>"
            f"<td style='padding:5px 10px 5px 4px;font-size:14px;color:rgba(200,200,200,0.5);white-space:nowrap'>{symbol}</td>"
            f"<td style='padding:5px 10px;font-size:11px;color:rgba(160,160,160,0.55);white-space:nowrap'>{meta}</td>"
            f"<td style='padding:5px 10px;font-size:12px;color:rgba(220,220,220,0.85);font-weight:600'>{names}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<div style='margin-top:4px'>"
        f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;"
        f"color:rgba(120,120,120,0.5);margin-bottom:6px'>Featured groups</div>"
        f"<table style='border-collapse:collapse;width:100%'>{rows_html}</table>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_weather_tab(
    *,
    api_key: str,
    schedule_df: pd.DataFrame,
    event_id: int | float,
    tee_times_path: str | None = None,
    featured_dg_ids: list[int] | None = None,
):
    """
    Render the weather tab for the current event.

    Parameters
    ----------
    api_key         : WeatherAPI key
    schedule_df     : schedule DataFrame OR this_week_field.csv DataFrame.
                      Recognised date columns : start_date, date_start, event_date, tournament_date.
                      Recognised location cols: weather_q, course_name, event_name.
    event_id        : the current event's event_id
    tee_times_path  : path to this_week_field.csv (optional; waves shown if provided)
    featured_dg_ids : dg_ids from outright favourites + hottest players panels.
                      When provided, only groups containing a featured player are shown;
                      groups are deduplicated so shared players appear only once.
    """

    # ── Lookup event row ──────────────────────────────────────────────────────
    sdf = schedule_df.copy()
    sdf["event_id"] = pd.to_numeric(sdf["event_id"], errors="coerce")
    eid = float(event_id) if event_id is not None else None

    row = sdf[sdf["event_id"] == eid]
    if row.empty:
        row = sdf[sdf["event_id"] == int(eid)]
    if row.empty:
        st.warning(f"Event {event_id} not found in schedule.")
        return
    row = row.iloc[0]

    event_name  = str(row.get("event_name", "")).strip()
    course_name = str(row.get("course_name", "")).strip()

    # ── Resolve start_date from multiple possible column names ─────────────────
    start_date = pd.NaT
    for _col in ["start_date", "date_start", "event_date", "tournament_date", "date"]:
        if _col in row.index:
            start_date = pd.to_datetime(row[_col], errors="coerce")
            if not pd.isna(start_date):
                break

    # ── Resolve weather query string ───────────────────────────────────────────
    weather_q = ""
    for _col in ["weather_q", "weather_location", "location"]:
        if _col in row.index:
            _val = str(row[_col]).strip()
            if _val.lower() not in {"nan", "none", "null", "", "<unset>"}:
                weather_q = _val
                break

    # Fall back to course_name — WeatherAPI handles venue names well
    if not weather_q and course_name and course_name.lower() not in {"nan", "none", "null"}:
        weather_q = course_name

    if not weather_q or pd.isna(start_date):
        missing = []
        if not weather_q:
            missing.append("a location/course name")
        if pd.isna(start_date):
            missing.append("a start date (tried: start_date, date_start, event_date)")
        st.warning(
            f"Weather unavailable — could not find {chr(39).join(missing)} "
            f"for event_id={event_id}. "
            f"Schedule columns present: {list(sdf.columns)}"
        )
        return

    # R1–R4 dates
    round_dates = [start_date + timedelta(days=i) for i in range(4)]

    # ── Fetch forecast ────────────────────────────────────────────────────────
    today        = pd.Timestamp.now().normalize()
    r1_days_away = (round_dates[0] - today).days
    r4_days_away = (round_dates[-1] - today).days

    # Event already finished
    if r4_days_away < -1:
        st.info("This event has already concluded — forecast not available.")
        return

    # Event is beyond the 14-day forecast window
    if r1_days_away > 13:
        days_until_window = r1_days_away - 13
        st.markdown(
            "<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);"
            "border-radius:10px;padding:20px;text-align:center;color:rgba(180,180,180,0.7)'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:6px'>Forecast not yet available</div>"
            f"<div style='font-size:13px'>The {event_name} starts in {r1_days_away} days. "
            f"Hourly forecasts will appear in approximately {days_until_window} days.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # WeatherAPI day 1 = today, +2 to avoid off-by-one at boundary, cap at 14
    days_ahead = max(1, min(r4_days_away + 2, 14))

    with st.spinner("Fetching forecast..."):
        try:
            raw      = _fetch_forecast(api_key, weather_q, days_ahead)
            hourly   = _forecast_to_hourly(raw)
        except WeatherAPIError as e:
            st.error(f"Weather API error: {e}")
            return

    # ── Load tee times ────────────────────────────────────────────────────────
    tee_df = _load_tee_times(tee_times_path) if tee_times_path else pd.DataFrame()

    # ── Header ────────────────────────────────────────────────────────────────
    loc_name = hourly["location_name"].iloc[0] if not hourly.empty else weather_q
    st.markdown(
        f"<div style='margin-bottom:12px'>"
        f"<span style='font-size:20px;font-weight:800;color:rgba(255,255,255,0.92)'>{event_name}</span>"
        f"<span style='font-size:12px;color:rgba(160,160,160,0.45);margin-left:10px'>"
        f"{course_name} · {loc_name}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Daily summary cards ───────────────────────────────────────────────────
    st.markdown(f"<div style='{_SEC}'>4-Day Forecast</div>", unsafe_allow_html=True)
    _render_daily_cards(hourly, round_dates)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    # ── Round selector tabs ───────────────────────────────────────────────────
    st.markdown(f"<div style='{_SEC}'>Hourly Breakdown</div>", unsafe_allow_html=True)

    # Default to the next upcoming round
    today_date = pd.Timestamp.now().date()
    default_rnd = 0
    for i, rd in enumerate(round_dates):
        if rd.date() >= today_date:
            default_rnd = i
            break

    tab_labels = [f"R{i+1} · {rd.strftime('%a %b %d')}" for i, rd in enumerate(round_dates)]
    tabs = st.tabs(tab_labels)

    for ti, (tab, rd) in enumerate(zip(tabs, round_dates)):
        with tab:
            day_df     = hourly[hourly["dt_local"].dt.date == rd.date()].copy()
            wave_times = _get_wave_times(tee_df, ti + 1)

            # Wave summary line if tee times available
            if wave_times["early"] or wave_times["late"]:
                parts = []
                if wave_times["early"]:
                    t = pd.Timestamp(min(wave_times["early"]))
                    parts.append(
                        f"<span style='color:{_C_AM};font-weight:700'>Early wave</span>"
                        f"<span style='color:rgba(160,160,160,0.6)'> from {t.strftime('%I:%M %p')}</span>"
                    )
                if wave_times["late"]:
                    t = pd.Timestamp(min(wave_times["late"]))
                    parts.append(
                        f"<span style='color:{_C_PM};font-weight:700'>Late wave</span>"
                        f"<span style='color:rgba(160,160,160,0.6)'> from {t.strftime('%I:%M %p')}</span>"
                    )
                st.markdown(
                    f"<div style='font-size:12px;margin-bottom:8px'>"
                    f"Tee times: {' · '.join(parts)}</div>",
                    unsafe_allow_html=True,
                )

            top_groups = _get_top_groups(tee_df, ti + 1, n_groups=8, featured_dg_ids=featured_dg_ids or []) if not tee_df.empty else []
            _render_weather_chart(day_df, ti + 1, rd, wave_times, top_groups)
            _render_group_timeline(top_groups, rd)

    # ── Forecast note ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:10px;color:rgba(100,100,100,0.45);margin-top:8px'>"
        f"Forecast via WeatherAPI · refreshed every 30 min · "
        f"Shaded bands show tee time wave windows · 6am–7pm local time shown</div>",
        unsafe_allow_html=True,
    )
