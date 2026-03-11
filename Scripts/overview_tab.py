from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Cancelled / suspended event annotations
# Add any year → note pairs here to show a footnote in Past Winners
# ─────────────────────────────────────────────────────────────────────────────

# Cancelled / suspended event annotations
# Key: (event_id, year) → note string
# Only events where a round was actually started before cancellation get an annotation.
# Events simply not held (e.g. most 2020 events) produce no data rows and need no annotation.
# ─────────────────────────────────────────────────────────────────────────────

_CANCELLED_YEARS: dict[tuple[int, int], str] = {
    (11, 2020): "Cancelled after R1 — COVID-19 (Hideki Matsuyama led)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(x) -> str:
    if x is None: return "—"
    s = str(x).strip()
    return "—" if s.lower() in {"nan","none","null","","<unset>"} else s

def _money(x) -> str:
    v = pd.to_numeric(x, errors="coerce")
    return f"${v:,.0f}" if pd.notna(v) else "—"

def _odds_to_american(d) -> str:
    try:
        d = float(d)
        if d <= 1.0: return "—"
        return f"+{int(round((d-1)*100))}" if d >= 2.0 else f"{int(round(-100/(d-1)))}"
    except: return "—"

def _odds_to_implied(d) -> str:
    try: return f"{1/float(d)*100:.1f}%"
    except: return "—"

def _label(text: str) -> str:
    return (
        f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:rgba(120,120,120,0.55);margin-bottom:10px'>{text}</div>"
    )

def _divider() -> None:
    st.markdown(
        "<div style='height:1px;background:rgba(255,255,255,0.06);margin:24px 0'></div>",
        unsafe_allow_html=True,
    )

def _headshot(name: str, img_url, size: int = 32) -> str:
    if img_url and str(img_url).lower() not in {"nan","none","null",""}:
        return (
            f"<img src='{img_url}' style='width:{size}px;height:{size}px;border-radius:50%;"
            f"object-fit:cover;flex-shrink:0'>"
        )
    initials = "".join(w[0] for w in name.split() if w)[:2].upper()
    return (
        f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
        f"background:rgba(255,255,255,0.06);display:flex;align-items:center;"
        f"justify-content:center;font-size:{max(9,size//3)}px;font-weight:700;"
        f"color:rgba(180,180,180,0.7);flex-shrink:0'>{initials}</div>"
    )

def _owgr_badge(owgr) -> str:
    """Coloured rank pill — gold top10, silver top50, muted otherwise."""
    if owgr is None or (isinstance(owgr, float) and np.isnan(owgr)):
        return "<span style='font-size:10px;color:rgba(100,100,100,0.45)'>OWGR —</span>"
    n = int(owgr)
    if n <= 10:
        color, bg = "#fbbf24", "rgba(251,191,36,0.12)"
    elif n <= 50:
        color, bg = "rgba(200,200,200,0.8)", "rgba(255,255,255,0.07)"
    else:
        color, bg = "rgba(130,130,130,0.6)", "rgba(255,255,255,0.03)"
    return (
        f"<span style='font-size:9px;font-weight:700;color:{color};"
        f"background:{bg};border-radius:4px;padding:1px 5px'>#{n}</span>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A — Hero header
# ─────────────────────────────────────────────────────────────────────────────

def _render_hero(selected_row: pd.Series) -> None:
    event_name  = _clean(selected_row.get("event_name"))
    course_name = _clean(selected_row.get("course_name"))
    location    = _clean(selected_row.get("location"))
    purse       = _money(selected_row.get("purse"))
    winner_shr  = _money(selected_row.get("winner_share"))
    champ       = _clean(selected_row.get("defending_champ"))
    event_type  = _clean(selected_row.get("event_type"))
    img_url = str(selected_row.get("banner_image","") or selected_row.get("image","") or "").strip()
    if img_url.lower() in {"nan","none","null","","<unset>"}: img_url = None

    try:
        start_dt = pd.to_datetime(selected_row.get("start_date") or selected_row.get("event_date"))
        date_str = f"{start_dt.strftime('%b %d')} – {(start_dt + pd.Timedelta(days=3)).strftime('%b %d, %Y')}"
    except: date_str = "—"

    if img_url:
        st.markdown(
            f"<div style='border-radius:14px;overflow:hidden;margin-bottom:18px;height:500px;position:relative'>"
            f"<img src='{img_url}' style='width:100%;height:100%;object-fit:cover;object-position:center center;display:block'>"
            f"<div style='position:absolute;inset:0;background:linear-gradient(to bottom,rgba(0,0,0,0) 40%,rgba(10,10,10,0.8))'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    badge = ""
    if event_type not in {"—","REGULAR"}:
        badge = (
            f"<span style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;"
            f"background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);"
            f"border-radius:4px;padding:2px 7px;vertical-align:middle;margin-left:10px'>{event_type}</span>"
        )
    loc_part = f" · {location}" if location != "—" else ""
    st.markdown(
        f"<div style='margin-bottom:6px'>"
        f"<div style='font-size:28px;font-weight:900;line-height:1.1'>{event_name}{badge}</div>"
        f"<div style='font-size:13px;color:rgba(150,150,150,0.6);margin-top:5px'>"
        f"{course_name}{loc_part} · {date_str}</div></div>",
        unsafe_allow_html=True,
    )

    oad_purse    = _money(selected_row.get("oad_purse"))
    oad_winner   = _money(selected_row.get("oad_winner_share"))

    # Show OAD-adjusted values if they differ from actual purse
    try:
        _p_raw  = float(selected_row.get("purse") or 0)
        _oad_p  = float(selected_row.get("oad_purse") or 0)
        oad_differs = abs(_p_raw - _oad_p) > 1
    except: oad_differs = False

    pills = [("Purse", purse), ("Winner's Share", winner_shr), ("Defending Champ", champ)]
    if oad_differs:
        pills += [("OAD Purse", oad_purse), ("OAD Winner's Share", oad_winner)]

    pills_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 4px;align-items:center'>"
    for lbl, val in pills:
        is_oad = lbl.startswith("OAD")
        bg     = "rgba(139,92,246,0.10)" if is_oad else "rgba(255,255,255,0.04)"
        border = "rgba(139,92,246,0.3)"  if is_oad else "rgba(255,255,255,0.07)"
        lbl_color = "rgba(139,92,246,0.7)" if is_oad else "rgba(120,120,120,0.55)"
        pills_html += (
            f"<div style='background:{bg};border:1px solid {border};"
            f"border-radius:20px;padding:4px 13px;white-space:nowrap'>"
            f"<span style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;"
            f"color:{lbl_color};margin-right:5px'>{lbl}</span>"
            f"<span style='font-size:13px;font-weight:700;color:rgba(225,225,225,0.9)'>{val}</span>"
            f"</div>"
        )
    st.markdown(pills_html + "</div>", unsafe_allow_html=True)

    # ── Course stats strip ────────────────────────────────────────────────────
    def _fmt_yardage(v):
        try: return f"{int(float(v)):,} yds"
        except: return None

    def _fmt_rating(v):
        try: return f"{float(v):.1f}"
        except: return None

    def _fmt_slope(v):
        try: return str(int(float(v)))
        except: return None

    def _fmt_par(v):
        try: return f"Par {int(float(v))}"
        except: return None

    course_stat_items = []

    par_val = selected_row.get("par")
    par_fmt = _fmt_par(par_val)
    if par_fmt:
        course_stat_items.append(("", par_fmt, "#60a5fa"))          # blue pill — most prominent

    yardage_val = selected_row.get("yardage")
    yardage_fmt = _fmt_yardage(yardage_val)
    if yardage_fmt:
        course_stat_items.append(("", yardage_fmt, None))

    rating_val = selected_row.get("course_rating")
    slope_val  = selected_row.get("slope")
    rating_fmt = _fmt_rating(rating_val)
    slope_fmt  = _fmt_slope(slope_val)
    if rating_fmt and slope_fmt:
        course_stat_items.append(("Rating / Slope", f"{rating_fmt} / {slope_fmt}", None))
    elif rating_fmt:
        course_stat_items.append(("Rating", rating_fmt, None))

    greens_val = _clean(selected_row.get("greens_type"))
    if greens_val and greens_val != "—":
        course_stat_items.append(("Greens", greens_val, None))

    fairways_val = _clean(selected_row.get("fairways_type"))
    if fairways_val and fairways_val != "—":
        course_stat_items.append(("Fairways", fairways_val, None))

    rough_val = _clean(selected_row.get("rough_type"))
    if rough_val and rough_val != "—":
        course_stat_items.append(("Rough", rough_val, None))

    record_val  = _clean(selected_row.get("course_record"))
    record_holder = _clean(selected_row.get("course_record_holder"))
    if record_val and record_val != "—":
        record_display = record_val
        if record_holder and record_holder != "—":
            record_display = f"{record_val} ({record_holder})"
        course_stat_items.append(("Course Record", record_display, None))

    if course_stat_items:
        strip_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 18px;align-items:center'>"
        for lbl, val, accent in course_stat_items:
            if accent:
                # Prominent coloured pill (par)
                strip_html += (
                    f"<div style='background:rgba(96,165,250,0.12);border:1px solid rgba(96,165,250,0.3);"
                    f"border-radius:20px;padding:4px 13px;font-size:13px;font-weight:800;"
                    f"color:{accent};white-space:nowrap'>{val}</div>"
                )
            elif lbl:
                # Labelled pill
                strip_html += (
                    f"<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);"
                    f"border-radius:20px;padding:4px 13px;white-space:nowrap'>"
                    f"<span style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;"
                    f"color:rgba(120,120,120,0.55);margin-right:5px'>{lbl}</span>"
                    f"<span style='font-size:13px;font-weight:600;color:rgba(210,210,210,0.85)'>{val}</span>"
                    f"</div>"
                )
            else:
                # Plain pill (yardage etc)
                strip_html += (
                    f"<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);"
                    f"border-radius:20px;padding:4px 13px;font-size:13px;font-weight:600;"
                    f"color:rgba(210,210,210,0.85);white-space:nowrap'>{val}</div>"
                )
        strip_html += "</div>"
        st.markdown(strip_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# B — Course DNA  (redesigned)
# ─────────────────────────────────────────────────────────────────────────────

_IMP_COLS = {
    "imp_dist": ("Distance",     "#f97316", ""),
    "imp_acc":  ("Accuracy",     "#38bdf8", ""),
    "imp_app":  ("Approach",     "#34d399", ""),
    "imp_arg":  ("Around Green", "#a78bfa", ""),
    "imp_putt": ("Putting",      "#fbbf24", ""),
}

def _render_course_dna(course_fit_df, course_num) -> None:
    _divider()
    st.markdown(_label("Course DNA — What It Takes to Win"), unsafe_allow_html=True)

    if course_fit_df is None or course_num is None:
        st.caption("Course fit data not available."); return

    cf = course_fit_df.copy()
    cf.columns = [c.lower().strip() for c in cf.columns]
    cn_col = next((c for c in ["course_num","course","course_id"] if c in cf.columns), None)
    if cn_col is None: st.caption("course_num column not found."); return

    cf[cn_col] = pd.to_numeric(cf[cn_col], errors="coerce")
    row = cf.loc[cf[cn_col] == int(course_num)]
    if row.empty: st.caption(f"No course fit data for course_num={course_num}."); return
    row = row.iloc[0]

    avail = {k: v for k, v in _IMP_COLS.items() if k in row.index}
    if not avail: st.caption("No importance columns found."); return

    weights = {k: float(pd.to_numeric(row[k], errors="coerce") or 0) for k in avail}
    total   = sum(weights.values()) or 1
    sorted_keys = sorted(weights, key=lambda k: weights[k], reverse=True)

    primary = _clean(row.get("primary_skill",""))
    primary_key = f"imp_{primary.lower()}" if primary != "—" else sorted_keys[0]
    primary_label = avail.get(primary_key, avail[sorted_keys[0]])[0]
    primary_color = avail.get(primary_key, avail[sorted_keys[0]])[1]

    pred_pct = float(pd.to_numeric(row.get("predictability_pct", 0), errors="coerce") or 0)
    if pred_pct >= 0.80:
        pred_label = "High"
        pred_color = "#34d399"
        pred_blurb = "Elite ball-strikers have a reliable edge — skill beats luck here."
    elif pred_pct >= 0.50:
        pred_label = "Medium"
        pred_color = "#fbbf24"
        pred_blurb = "Skill matters, but variance plays a meaningful role."
    else:
        pred_label = "Low"
        pred_color = "#f87171"
        pred_blurb = "Course history and hot form may matter more than rankings."

    # ── Left: horizontal bar tiles ────────────────────────────────────────────
    col_bars, col_meta = st.columns([3, 2], gap="large")

    with col_bars:
        bars_html = "<div style='display:flex;flex-direction:column;gap:6px;padding-top:4px'>"
        max_val = max(weights[k] for k in sorted_keys) * 100
        for k in sorted_keys:
            label, color, icon = avail[k]
            pct   = weights[k] * 100
            width = max(4, pct / max(max_val, 1) * 100)
            is_primary = (k == primary_key)
            border = f"border:1px solid {color}44;" if is_primary else "border:1px solid rgba(255,255,255,0.04);"
            glow   = f"box-shadow:0 0 12px {color}22;" if is_primary else ""
            bars_html += (
                f"<div style='background:rgba(255,255,255,0.03);{border}{glow}"
                f"border-radius:8px;padding:9px 12px;display:flex;align-items:center;gap:10px'>"
                # icon + label
                f"<span style='font-size:14px;flex-shrink:0'>{icon}</span>"
                f"<div style='flex:1;min-width:0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>"
                f"<span style='font-size:12px;font-weight:{'800' if is_primary else '600'};"
                f"color:{'rgba(255,255,255,0.95)' if is_primary else 'rgba(200,200,200,0.7)'}'>{label}"
                + (f" <span style='font-size:9px;font-weight:700;color:{color};"
                   f"background:{color}22;border-radius:3px;padding:1px 5px;margin-left:4px'>PRIMARY</span>"
                   if is_primary else "") +
                f"</span>"
                f"<span style='font-size:13px;font-weight:800;color:{color}'>{pct:.1f}%</span>"
                f"</div>"
                # bar track
                f"<div style='height:5px;background:rgba(255,255,255,0.05);border-radius:3px'>"
                f"<div style='height:5px;width:{width:.0f}%;border-radius:3px;"
                f"background:linear-gradient(90deg,{color}cc,{color}55)'></div>"
                f"</div>"
                f"</div></div>"
            )
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)

    # ── Right: primary skill card + predictability ─────────────────────────────
    with col_meta:
        # Donut-style ring using an inline SVG — shows skill split visually
        # Build SVG arc segments
        cx, cy, r_outer, r_inner = 80, 80, 68, 44
        import math
        def arc_path(start_deg, end_deg, ro, ri, cx, cy):
            def pt(deg, radius):
                rad = math.radians(deg - 90)
                return cx + radius * math.cos(rad), cy + radius * math.sin(rad)
            x1o,y1o = pt(start_deg, ro); x2o,y2o = pt(end_deg, ro)
            x1i,y1i = pt(start_deg, ri); x2i,y2i = pt(end_deg, ri)
            large = 1 if (end_deg - start_deg) > 180 else 0
            return (f"M {x1o:.1f} {y1o:.1f} A {ro} {ro} 0 {large} 1 {x2o:.1f} {y2o:.1f} "
                    f"L {x2i:.1f} {y2i:.1f} A {ri} {ri} 0 {large} 0 {x1i:.1f} {y1i:.1f} Z")

        gap_deg = 3
        current = 0
        segments = ""
        for k in sorted_keys:
            label, color, icon = avail[k]
            pct = weights[k] / total
            sweep = pct * 360 - gap_deg
            if sweep > 0:
                path = arc_path(current, current + sweep, r_outer, r_inner, cx, cy)
                opacity = "1" if k == primary_key else "0.35"
                segments += f"<path d='{path}' fill='{color}' opacity='{opacity}'/>"
            current += pct * 360

        donut_svg = (
            f"<svg width='160' height='160' viewBox='0 0 160 160'>"
            f"{segments}"
            f"<text x='80' y='74' text-anchor='middle' font-size='11' font-weight='700' "
            f"fill='rgba(150,150,150,0.6)' font-family='sans-serif'>PRIMARY</text>"
            f"<text x='80' y='92' text-anchor='middle' font-size='14' font-weight='900' "
            f"fill='{primary_color}' font-family='sans-serif'>{primary_label}</text>"
            f"</svg>"
        )

        st.markdown(
            f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);"
            f"border-radius:12px;padding:16px;display:flex;flex-direction:column;align-items:center'>"
            f"<div style='display:flex;justify-content:center'>{donut_svg}</div>"
            # Predictability row
            f"<div style='width:100%;margin-top:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
            f"<span style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;"
            f"color:rgba(120,120,120,0.5)'>Predictability</span>"
            f"<span style='font-size:11px;font-weight:800;color:{pred_color}'>{pred_label}</span>"
            f"</div>"
            f"<div style='height:5px;background:rgba(255,255,255,0.06);border-radius:3px;margin-bottom:8px'>"
            f"<div style='height:5px;width:{pred_pct*100:.0f}%;border-radius:3px;"
            f"background:linear-gradient(90deg,{pred_color},{pred_color}88)'></div></div>"
            f"<div style='font-size:11px;color:rgba(140,140,140,0.6);line-height:1.5;text-align:center'>"
            f"{pred_blurb}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# C — Odds + Form  (with OWGR badges)
# ─────────────────────────────────────────────────────────────────────────────

def _render_field_snapshot(field_ev, rounds_df, cutoff_dt, id_to_img, field_df=None) -> list:
    """Returns list of featured dg_ids (favourites + hottest) for weather tee-time filtering."""
    _divider()
    if field_ev.empty:
        st.caption("Field data not yet available for this event.")
        return []

    fev = field_ev.copy()
    for c in ["dg_id","owgr_rank"]:
        if c in fev.columns: fev[c] = pd.to_numeric(fev[c], errors="coerce")
    odds_col = next((c for c in ["close_odds","decimal_odds","odds","win_prob_est"] if c in fev.columns), None)
    fev["_odds"] = pd.to_numeric(fev[odds_col], errors="coerce") if odds_col else np.nan
    fev = fev.dropna(subset=["dg_id"]).copy()
    fev["dg_id"] = fev["dg_id"].astype(int)

    # Merge owgr_rank from field_df (this_week_field.csv) if not already present or all null
    if field_df is not None and not field_df.empty and "dg_id" in field_df.columns and "owgr_rank" in field_df.columns:
        fdf = field_df[["dg_id","owgr_rank"]].copy()
        fdf["dg_id"] = pd.to_numeric(fdf["dg_id"], errors="coerce").dropna().astype(int)
        fdf["owgr_rank"] = pd.to_numeric(fdf["owgr_rank"], errors="coerce")
        fdf = fdf.drop_duplicates("dg_id")
        if fev["owgr_rank"].isna().all() if "owgr_rank" in fev.columns else True:
            fev = fev.drop(columns=["owgr_rank"], errors="ignore").merge(fdf, on="dg_id", how="left")
        else:
            # fill nulls only
            fev = fev.merge(fdf.rename(columns={"owgr_rank":"_owgr_fill"}), on="dg_id", how="left")
            fev["owgr_rank"] = fev["owgr_rank"].fillna(fev["_owgr_fill"])
            fev = fev.drop(columns=["_owgr_fill"])

    col_odds, col_form = st.columns(2, gap="large")

    featured_ids: list[int] = []  # dg_ids from both panels for weather tee-time filter

    with col_odds:
        st.markdown(_label("Outright Favourites"), unsafe_allow_html=True)
        has_odds = fev["_odds"].notna().any() and (fev["_odds"] > 1).any()
        if has_odds:
            for rank_pos, (_, p) in enumerate(fev[fev["_odds"] > 1].nsmallest(10, "_odds").iterrows(), 1):
                name     = _clean(p.get("player_name"))
                american = _odds_to_american(p["_odds"])
                implied  = _odds_to_implied(p["_odds"])
                owgr     = p.get("owgr_rank")
                hs       = _headshot(name, id_to_img.get(int(p["dg_id"])))
                owgr_html = _owgr_badge(owgr)
                featured_ids.append(int(p["dg_id"]))
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;"
                    f"height:52px;overflow:hidden;border-bottom:1px solid rgba(255,255,255,0.04)'>"
                    f"<span style='font-size:10px;color:rgba(100,100,100,0.4);min-width:14px;text-align:right'>"
                    f"{rank_pos}</span>"
                    f"{hs}<div style='flex:1;min-width:0;overflow:hidden'>"
                    f"<div style='font-size:13px;font-weight:600;color:rgba(215,215,215,0.9);"
                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{name}</div>"
                    f"<div style='margin-top:2px'>{owgr_html}</div>"
                    f"</div><div style='text-align:right;flex-shrink:0'>"
                    f"<div style='font-size:15px;font-weight:800;color:#fbbf24'>{american}</div>"
                    f"<div style='font-size:10px;color:rgba(110,110,110,0.5)'>{implied}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Odds not yet available for this event.")

    with col_form:
        st.markdown(_label("Hottest Players — Last 12 Rounds"), unsafe_allow_html=True)
        field_ids = fev["dg_id"].tolist()
        r = rounds_df.copy()
        for c in ["dg_id","sg_total"]:
            if c in r.columns: r[c] = pd.to_numeric(r[c], errors="coerce")
        date_col = "round_date" if "round_date" in r.columns else "event_completed"
        r[date_col] = pd.to_datetime(r[date_col], errors="coerce")
        cutoff_ts = pd.to_datetime(cutoff_dt) if cutoff_dt is not None else pd.Timestamp.now()
        r = r[(r["dg_id"].isin(field_ids)) & (r[date_col] < cutoff_ts)].dropna(subset=["sg_total"]).copy()
        r = r.sort_values(["dg_id", date_col], ascending=[True,False])

        form = (
            r.groupby("dg_id", group_keys=False)
             .apply(lambda g: g.head(12))
             .reset_index(drop=True)
             .groupby("dg_id")["sg_total"].mean()
             .reset_index().rename(columns={"sg_total":"sg_l12"})
        )
        form["dg_id"] = form["dg_id"].astype(int)
        _meta = [c for c in ["dg_id","player_name","owgr_rank"] if c in fev.columns]
        form = (
            form.merge(fev[_meta].drop_duplicates("dg_id"), on="dg_id", how="left")
                .dropna(subset=["sg_l12"]).nlargest(10,"sg_l12")
        )
        if "owgr_rank" not in form.columns:
            form["owgr_rank"] = np.nan
        if form.empty:
            st.caption("Not enough round data.")
        else:
            sg_max = float(form["sg_l12"].max())
            for rank_pos, (_, p) in enumerate(form.iterrows(), 1):
                name   = _clean(p.get("player_name"))
                sg_val = float(p["sg_l12"])
                bar_w  = max(4, sg_val/sg_max*100) if sg_max > 0 else 4
                owgr   = p.get("owgr_rank")
                hs     = _headshot(name, id_to_img.get(int(p["dg_id"])))
                owgr_html = _owgr_badge(owgr)
                featured_ids.append(int(p["dg_id"]))
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;"
                    f"height:52px;overflow:hidden;border-bottom:1px solid rgba(255,255,255,0.04)'>"
                    f"<span style='font-size:10px;color:rgba(100,100,100,0.4);min-width:14px;text-align:right'>"
                    f"{rank_pos}</span>"
                    f"{hs}<div style='flex:1;min-width:0;overflow:hidden'>"
                    f"<div style='font-size:13px;font-weight:600;color:rgba(215,215,215,0.9);"
                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{name}</div>"
                    f"<div style='display:flex;align-items:center;gap:6px;margin-top:2px'>"
                    f"{owgr_html}"
                    f"<div style='flex:1;height:3px;background:rgba(255,255,255,0.05);border-radius:2px'>"
                    f"<div style='height:3px;width:{bar_w:.0f}%;background:#34d399;border-radius:2px'></div></div>"
                    f"</div></div>"
                    f"<div style='text-align:right;flex-shrink:0'>"
                    f"<div style='font-size:15px;font-weight:800;color:#34d399'>+{sg_val:.2f}</div>"
                    f"<div style='font-size:10px;color:rgba(110,110,110,0.4)'>SG/Rnd</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    return list(dict.fromkeys(featured_ids))  # deduplicated, order preserved


# ─────────────────────────────────────────────────────────────────────────────
# D — Event history chart + past winners  (with cancelled year support)
# ─────────────────────────────────────────────────────────────────────────────

def _render_event_history(rounds_df, event_id, course_num) -> None:
    _divider()
    st.markdown(_label("Event History"), unsafe_allow_html=True)
    if event_id is None: st.caption("No event_id — history unavailable."); return

    # Build cancelled years relevant to THIS event only
    eid_int = int(event_id)
    cancelled_this_event = {yr: note for (eid, yr), note in _CANCELLED_YEARS.items() if eid == eid_int}

    r = rounds_df.copy()
    for c in ["event_id","year","finish_num","round_num","course_num","sg_total","round_score","cum_to_par","to_par"]:
        if c in r.columns: r[c] = pd.to_numeric(r[c], errors="coerce")
    if "round_date" in r.columns: r["round_date"] = pd.to_datetime(r["round_date"], errors="coerce")
    r = r[r["event_id"] == int(event_id)].copy()
    if r.empty: st.caption("No historical data found for this event."); return

    score_col = "cum_to_par" if "cum_to_par" in r.columns else ("to_par" if "to_par" in r.columns else None)
    if score_col:
        win_scores = (
            r[r["finish_num"] == 1].dropna(subset=["year"])
            .groupby("year")[score_col].min().reset_index()
            .rename(columns={score_col:"winning_score"})
        )
        win_scores["year"] = win_scores["year"].astype(int)
    else:
        win_scores = pd.DataFrame()

    diff_years: set = set()
    if course_num is not None and "course_num" in r.columns:
        cby = r.groupby("year")["course_num"].first().reset_index()
        cby["year"] = cby["year"].astype(int)
        diff_years = set(cby.loc[cby["course_num"] != int(course_num), "year"].tolist())

    avg_score = (
        r.dropna(subset=["year","round_score"])
        .groupby(["year","round_num"])["round_score"].mean().reset_index()
        .groupby("year")["round_score"].mean().reset_index()
        .rename(columns={"round_score":"avg_score"})
    )
    avg_score["year"] = avg_score["year"].astype(int)

    if win_scores.empty and avg_score.empty:
        st.caption("Insufficient data."); return

    all_years = sorted(set(
        (win_scores["year"].tolist() if not win_scores.empty else []) +
        (avg_score["year"].tolist() if not avg_score.empty else [])
    ))

    fig = go.Figure()
    if not avg_score.empty:
        fig.add_trace(go.Bar(
            x=avg_score["year"], y=avg_score["avg_score"],
            marker=dict(color="rgba(148,163,184,0.10)", line=dict(width=0)),
            hovertemplate="<b>%{x}</b> avg round: %{y:.1f}<extra></extra>",
            yaxis="y2", showlegend=False,
        ))
    if not win_scores.empty:
        fig.add_trace(go.Scatter(
            x=win_scores["year"], y=win_scores["winning_score"],
            mode="lines", line=dict(color="rgba(251,191,36,0.25)", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        for _, rw in win_scores.iterrows():
            yr = int(rw["year"]); sc = rw["winning_score"]; diff = yr in diff_years
            fig.add_trace(go.Scatter(
                x=[yr], y=[sc], mode="markers",
                marker=dict(
                    size=11, symbol="diamond" if diff else "circle",
                    color="rgba(148,163,184,0.55)" if diff else "#fbbf24",
                    line=dict(color="rgba(0,0,0,0.3)", width=1),
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{yr}</b>: {sc:+.0f}" +
                    (" ◆ different course" if diff else "") +
                    "<extra></extra>"
                ),
            ))
            fig.add_annotation(
                x=yr, y=sc, yshift=14,
                text=f"{sc:+.0f}" + (" ◆" if diff else ""),
                showarrow=False,
                font=dict(size=9, color="rgba(148,163,184,0.55)" if diff else "rgba(251,191,36,0.85)"),
            )

    # Mark cancelled years on the chart with a red dashed line
    for cancel_yr in cancelled_this_event:
        if cancel_yr in all_years:
            fig.add_vline(
                x=cancel_yr,
                line=dict(color="rgba(239,68,68,0.3)", width=1, dash="dot"),
                annotation_text="✕",
                annotation_font=dict(color="rgba(239,68,68,0.6)", size=10),
                annotation_position="top",
            )

    fig.update_layout(
        height=260, template="plotly_dark",
        margin=dict(l=10, r=20, t=16, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=False,
        xaxis=dict(tickmode="array", tickvals=all_years, tickfont=dict(size=10),
                   gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(
            title=dict(text="Winning score (to par)", font=dict(size=9, color="rgba(150,150,150,0.4)")),
            tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.04)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False,
            tickfont=dict(size=9, color="rgba(148,163,184,0.3)"),
            title=dict(text="Avg round", font=dict(size=8, color="rgba(148,163,184,0.25)")),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if diff_years:
        st.markdown(
            f"<div style='font-size:11px;color:rgba(148,163,184,0.45);margin-top:-8px;margin-bottom:8px'>"
            f"◆ {', '.join(str(y) for y in sorted(diff_years))} — played at a different course. "
            f"Scores may not be directly comparable.</div>",
            unsafe_allow_html=True,
        )

    # Past winners table
    _divider()
    st.markdown(_label("Past Winners"), unsafe_allow_html=True)
    winners = (
        r[r["finish_num"] == 1].dropna(subset=["year"])
        .drop_duplicates(subset=["year"]).sort_values("year", ascending=False)
    )

    # Build a set of all years to display — include cancelled years even if no winner row
    displayed_years = set(winners["year"].astype(int).tolist()) | set(cancelled_this_event.keys())
    all_winner_years = sorted(displayed_years, reverse=True)

    if "player_name" not in winners.columns and not cancelled_this_event:
        st.caption("No winner data found."); return

    name_by_year = {}
    if "player_name" in winners.columns:
        for _, pw in winners.iterrows():
            name_by_year[int(pw["year"])] = _clean(pw.get("player_name"))

    rows_html = ""
    footnotes_used = []

    for yr in all_winner_years:
        is_cancelled = yr in cancelled_this_event
        diff = yr in diff_years
        ws_row = win_scores[win_scores["year"] == yr] if not win_scores.empty else pd.DataFrame()

        if is_cancelled:
            note_text = cancelled_this_event[yr]
            footnotes_used.append((yr, note_text))
            rows_html += (
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);opacity:0.55'>"
                f"<div style='display:flex;align-items:baseline;gap:12px'>"
                f"<span style='font-size:11px;color:rgba(110,110,110,0.5);min-width:36px'>{yr}</span>"
                f"<span style='font-size:13px;font-style:italic;color:rgba(180,180,180,0.5)'>No champion</span>"
                f"<span style='font-size:10px;color:rgba(239,68,68,0.6);background:rgba(239,68,68,0.08);"
                f"border-radius:3px;padding:1px 5px;margin-left:4px'>✕ CANCELLED *</span>"
                f"</div><span style='font-size:12px;color:rgba(100,100,100,0.4)'>—</span></div>"
            )
        else:
            name = name_by_year.get(yr, "—")
            score_str = f"{int(ws_row['winning_score'].iloc[0]):+d}" if not ws_row.empty else "—"
            course_note = (
                "<span style='font-size:10px;color:rgba(148,163,184,0.4);margin-left:8px'>◆ diff. course</span>"
                if diff else ""
            )
            rows_html += (
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>"
                f"<div style='display:flex;align-items:baseline;gap:12px'>"
                f"<span style='font-size:11px;color:rgba(110,110,110,0.5);min-width:36px'>{yr}</span>"
                f"<span style='font-size:13px;font-weight:600;color:rgba(210,210,210,0.9)'>{name}</span>"
                f"{course_note}</div>"
                f"<span style='font-size:13px;font-weight:700;color:rgba(251,191,36,0.8)'>{score_str}</span>"
                f"</div>"
            )

    st.markdown(f"<div>{rows_html}</div>", unsafe_allow_html=True)

    # Footnotes for cancelled years
    if footnotes_used:
        fn_html = "<div style='margin-top:10px'>"
        for yr, note in footnotes_used:
            fn_html += (
                f"<div style='font-size:10px;color:rgba(239,68,68,0.5);margin-bottom:3px'>"
                f"* {yr}: {note}</div>"
            )
        fn_html += "</div>"
        st.markdown(fn_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# E — Weather (delegates to weather_tab)
# ─────────────────────────────────────────────────────────────────────────────

def _render_weather_section(api_key, schedule_df, event_id, tee_times_path, featured_dg_ids=None) -> None:
    _divider()
    st.markdown(_label("Weather Forecast"), unsafe_allow_html=True)
    try:
        from weather_tab import render_weather_tab

        sdf = schedule_df.copy() if schedule_df is not None else pd.DataFrame()
        date_cols = {"start_date", "date_start", "event_date", "tournament_date"}
        has_date = any(c in sdf.columns for c in date_cols)
        if not has_date and tee_times_path:
            try:
                sdf = pd.read_csv(tee_times_path)
                sdf.columns = [c.lower().strip() for c in sdf.columns]
            except Exception:
                pass

        sdf["event_id"] = pd.to_numeric(sdf["event_id"], errors="coerce")
        eid = float(event_id) if event_id is not None else None

        # Check whether the tee times file is for this event; if not, suppress
        # tee-time visuals and show a note instead of wrong groups.
        tee_times_match = False
        if tee_times_path:
            try:
                tt = pd.read_csv(tee_times_path, nrows=5)
                tt.columns = [c.lower().strip() for c in tt.columns]
                if "event_id" in tt.columns:
                    tt_eid = pd.to_numeric(tt["event_id"], errors="coerce").dropna()
                    if not tt_eid.empty and float(tt_eid.iloc[0]) == eid:
                        tee_times_match = True
            except Exception:
                pass

        if tee_times_path and not tee_times_match:
            st.markdown(
                "<div style='font-size:11px;color:rgba(148,163,184,0.45);margin-bottom:8px'>"
                "Tee times not available for this event — field data is for a different week. "
                "Wave windows and featured groups will not be shown.</div>",
                unsafe_allow_html=True,
            )

        render_weather_tab(
            api_key=api_key,
            schedule_df=sdf,
            event_id=eid,
            tee_times_path=tee_times_path if tee_times_match else None,
            featured_dg_ids=featured_dg_ids if tee_times_match else [],
        )
    except Exception as e:
        st.caption(f"Weather unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_overview_tab(
    *,
    selected_row: pd.Series,
    rounds_df: pd.DataFrame,
    field_ev: pd.DataFrame,
    event_id,
    course_num,
    cutoff_dt,
    course_fit_df=None,
    id_to_img=None,
    weather_api_key=None,
    schedule_df=None,
    tee_times_path=None,
    field_df=None,
) -> None:
    # Auto-load field_df from tee_times_path if not provided
    if field_df is None and tee_times_path:
        try:
            field_df = pd.read_csv(tee_times_path)
            field_df.columns = [c.lower().strip() for c in field_df.columns]
        except Exception:
            field_df = None
    _render_hero(selected_row)
    _render_course_dna(course_fit_df, course_num)
    featured_ids = _render_field_snapshot(field_ev, rounds_df, cutoff_dt, id_to_img or {}, field_df=field_df)
    _render_event_history(rounds_df, event_id, course_num)
    if weather_api_key and event_id is not None:
        _render_weather_section(weather_api_key, schedule_df, event_id, tee_times_path, featured_dg_ids=featured_ids)
