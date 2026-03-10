from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants / helpers
# ─────────────────────────────────────────────────────────────────────────────

_TOUR_LABELS = {
    "pga": "PGA Tour", "euro": "DP World Tour",
    "kft": "Korn Ferry Tour", "alt": "Other",
}
_EXCLUDE_EVENTS = {"pga q-school","korn ferry q-school","korn ferry","q-school"}

def _tour_display(t): return _TOUR_LABELS.get(str(t).lower().strip(), str(t).upper())

def _par_fmt(val):
    try:
        v = int(round(float(val)))
        if v == 0: return "E"
        return f"+{v}" if v > 0 else str(v)
    except Exception: return "–"

def _par_color(val):
    try:
        v = float(val)
        if v < 0: return "#ef4444"
        if v > 0: return "#60a5fa"
        return "#22c55e"
    except Exception: return "rgba(200,200,200,0.7)"

def _score_color(score_val, course_par_val):
    try:
        diff = int(float(score_val)) - int(float(course_par_val))
        if diff < 0:  return "#ef4444"
        if diff == 0: return "#22c55e"
        return "#60a5fa"
    except Exception: return "rgba(200,200,200,0.7)"

def _score_fmt(val):
    try: return str(int(float(val)))
    except Exception: return ""

def _move_html(cur, prev):
    try:
        d = int(float(prev)) - int(float(cur))
        if d > 0: return f"<span style='color:#22c55e;font-size:10px'>▲{d}</span>"
        if d < 0: return f"<span style='color:#ef4444;font-size:10px'>▼{abs(d)}</span>"
        return "<span style='color:rgba(180,180,180,0.4);font-size:10px'>–</span>"
    except Exception: return ""

def _hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def _short(name):
    parts = str(name).split(",")
    if len(parts) == 2: return parts[0].strip()
    return name.split()[-1] if name else name

def _md(text):
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

def _get_hs(dg_id, player_name, id_to_img, name_to_img):
    img = None
    if id_to_img and dg_id:
        try: img = id_to_img.get(int(dg_id))
        except Exception: pass
    if img is None and name_to_img and player_name:
        img = name_to_img.get(str(player_name))
    return img

def _sort_by_finish(df):
    """
    Sort leaderboard rows by round_position first, then break ties with finish_num.
    Both columns sorted ascending (lower = better).
    Falls back gracefully if finish_num is missing.
    """
    sort_cols = ["round_position"]
    if "finish_num" in df.columns:
        sort_cols.append("finish_num")
    return df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

COLORS = [
    "#f97316","#38bdf8","#a78bfa","#34d399","#fb7185",
    "#fbbf24","#60a5fa","#e879f9","#4ade80","#f472b6",
]
SG_COLORS = {
    "sg_ott": "rgba(120,180,255,0.85)",
    "sg_app": "rgba(255,130,160,0.85)",
    "sg_arg": "rgba(190,150,255,0.85)",
    "sg_putt":"rgba(255,190,100,0.85)",
}
SG_LABELS = {"sg_ott":"OTT","sg_app":"APP","sg_arg":"ARG","sg_putt":"PUTT"}

_SEC = ("font-size:11px;font-weight:700;letter-spacing:0.08em;"
        "color:rgba(130,130,130,0.7);text-transform:uppercase;margin-bottom:8px")
_SUB = "font-size:12px;color:rgba(140,140,140,0.5);margin-bottom:10px"
_DIV = "<div style='margin-bottom:6px'></div>"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cum_score(ev, player_name, through_round):
    pev = ev[ev["player_name"] == player_name].copy()
    pev["round_num"] = pd.to_numeric(pev["round_num"], errors="coerce")
    pev = pev.dropna(subset=["round_num"]).sort_values("round_num")
    row = pev[pev["round_num"] == through_round]
    if row.empty: return np.nan
    ctp = pd.to_numeric(row.iloc[0].get("cum_to_par", np.nan), errors="coerce")
    if pd.notna(ctp): return float(ctp)
    subset = pev[pev["round_num"] <= through_round]
    vals = pd.to_numeric(subset["to_par"], errors="coerce").dropna()
    return float(vals.sum()) if len(vals) > 0 else np.nan

def _build_narrative(ev, sel_rnd, rounds_avail, rnd_sel):
    lines = []
    rdata = ev[ev["round_num"] == sel_rnd].copy()
    for c in ["round_score","to_par","round_position","birdies","bogies","doubles_or_worse","course_par"]:
        if c in rdata.columns:
            rdata[c] = pd.to_numeric(rdata[c], errors="coerce")
    is_final = (rnd_sel == "Final" or sel_rnd == max(rounds_avail))

    # Use finish_num to break ties when identifying leader/winner
    rdata_sorted = rdata.dropna(subset=["round_position"]).copy()
    if "finish_num" in rdata_sorted.columns:
        rdata_sorted = rdata_sorted.sort_values(["round_position","finish_num"], ascending=[True,True])
    else:
        rdata_sorted = rdata_sorted.sort_values("round_position")

    if not rdata_sorted.empty:
        lname = rdata_sorted.iloc[0].get("player_name","")
        ltp   = rdata_sorted.iloc[0].get("cum_to_par", rdata_sorted.iloc[0].get("to_par", np.nan))
        ltp_s = _par_fmt(ltp)
        lines.append(f"**{lname}** {'wins' if is_final else f'leads after R{sel_rnd}'} at **{ltp_s}**.")

    if "round_score" in rdata.columns and "course_par" in rdata.columns:
        avg = rdata["round_score"].mean(); par = rdata["course_par"].median()
        if pd.notna(avg) and pd.notna(par):
            diff = avg - par
            if diff < -0.5:
                lines.append(f"R{sel_rnd} played easy — field averaged **{diff:+.1f}** to par ({avg:.1f}).")
            elif diff > 0.5:
                lines.append(f"R{sel_rnd} was tough — field averaged **{diff:+.1f}** to par ({avg:.1f}).")
            else:
                lines.append(f"R{sel_rnd} played close to par — field average **{avg:.1f}**.")

    if "birdies" in rdata.columns and "bogies" in rdata.columns:
        ab = rdata["birdies"].mean(); abg = rdata["bogies"].mean()
        if pd.notna(ab) and pd.notna(abg):
            lines.append(f"Players averaged **{ab:.1f} birdies** and **{abg:.1f} bogeys**.")

    if sel_rnd > min(rounds_avail):
        prv = max(r for r in rounds_avail if r < sel_rnd)
        prev_pos = ev[ev["round_num"]==prv][["dg_id","player_name","round_position"]].copy()
        curr_pos = rdata[["dg_id","player_name","round_position"]].copy()
        merged = curr_pos.merge(
            prev_pos[["dg_id","round_position"]].rename(columns={"round_position":"prev"}),
            on="dg_id", how="inner")
        merged["move"] = merged["prev"] - merged["round_position"]
        if not merged.empty:
            best = merged.loc[merged["move"].idxmax()]
            worst = merged.loc[merged["move"].idxmin()]
            if pd.notna(best["move"]) and best["move"] > 3:
                lines.append(f"Biggest mover: **{_short(best['player_name'])}** climbed **{int(best['move'])} spots** into T{int(best['round_position'])}.")
            if pd.notna(worst["move"]) and worst["move"] < -4:
                lines.append(f"Biggest faller: **{_short(worst['player_name'])}** tumbled **{abs(int(worst['move']))} spots**.")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_event_browser_tab(*, rounds_df: pd.DataFrame, ID_TO_IMG: dict = None, NAME_TO_IMG: dict = None):

    df = rounds_df.copy()
    num_cols = ["dg_id","event_id","year","round_num","round_position",
                "round_score","finish_num","to_par","cum_to_par","course_par",
                "birdies","bogies","doubles_or_worse","eagles_or_better","pars",
                "gir","scrambling","driving_dist","driving_acc","prox_fw","prox_rgh",
                "great_shots","poor_shots","sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    name_col = "event_name" if "event_name" in df.columns else None
    if not name_col: st.info("No event_name column."); return

    if "tour" not in df.columns: df["tour"] = "pga"
    df["tour"] = df["tour"].astype(str).str.lower().str.strip()

    # ── Selectors ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2,1,3], gap="medium")
    tours = [t for t in ["pga","euro","kft"] if t in df["tour"].unique()] + \
            [t for t in df["tour"].unique() if t not in ["pga","euro","kft"]]
    with c1: sel_tour = st.selectbox("Tour", tours, format_func=_tour_display, key="eb_tour")
    tdf = df[df["tour"]==sel_tour].copy()
    years = sorted(tdf["year"].dropna().astype(int).unique(), reverse=True)
    with c2: sel_year = st.selectbox("Year", years, key="eb_year")
    ydf = tdf[tdf["year"]==sel_year].copy()

    emeta = (
        ydf.dropna(subset=["event_id",name_col])
        .groupby(["event_id",name_col], as_index=False)
        .agg(latest=("round_date","max"))
        .sort_values("latest", ascending=False)
    )
    emeta = emeta[~emeta[name_col].str.lower().str.contains("q-school|korn ferry q", na=False)]

    eid_opts  = emeta["event_id"].astype(int).tolist()
    eid_names = dict(zip(emeta["event_id"].astype(int), emeta[name_col]))
    with c3: sel_eid = st.selectbox("Event", eid_opts,
                                     format_func=lambda x: eid_names.get(x,str(x)), key="eb_event")

    ev = ydf[ydf["event_id"]==sel_eid].copy()
    etitle = eid_names.get(sel_eid, f"Event {sel_eid}")
    if ev.empty: st.info("No data for this event."); return

    rounds_avail   = sorted(ev["round_num"].dropna().astype(int).unique())
    n_players      = ev["dg_id"].nunique()
    latest_date    = ev["round_date"].max()
    date_str       = latest_date.strftime("%b %d, %Y") if pd.notna(latest_date) else ""
    course_name    = (ev["course_name"].dropna().iloc[0]
                      if "course_name" in ev.columns and not ev["course_name"].dropna().empty else "")
    sg_avail       = [c for c in ["sg_ott","sg_app","sg_arg","sg_putt"] if c in ev.columns]
    has_to_par     = "to_par"     in ev.columns
    has_cum_to_par = "cum_to_par" in ev.columns
    has_course_par = "course_par" in ev.columns

    # ── Round selector ─────────────────────────────────────────────────────────
    rnd_opts = ["Final"] + [f"R{r}" for r in rounds_avail]
    rnd_sel  = st.radio("Round", rnd_opts, horizontal=True, key="eb_rnd")
    sel_rnd  = max(rounds_avail) if rnd_sel=="Final" else int(rnd_sel[1:])
    is_final = (rnd_sel == "Final")
    rnds_so_far = [r for r in rounds_avail if r <= sel_rnd]

    ev_rnd   = ev[ev["round_num"]==sel_rnd].copy()
    final_rd = ev[ev["round_num"]==max(rounds_avail)].copy()
    final_rd["round_position"] = pd.to_numeric(final_rd["round_position"], errors="coerce")
    if "finish_num" in final_rd.columns:
        final_rd["finish_num"] = pd.to_numeric(final_rd["finish_num"], errors="coerce")

    # ── Event header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin:4px 0 10px'>"
        f"<span style='font-size:24px;font-weight:800;color:rgba(255,255,255,0.95)'>{etitle}</span>"
        f"<span style='font-size:13px;color:rgba(180,180,180,0.45);margin-left:12px'>"
        f"{course_name}{' · ' if course_name else ''}{date_str} · {n_players} players</span></div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    narrative = _build_narrative(ev, sel_rnd, rounds_avail, rnd_sel)

    # FIX 1: Break ties with finish_num when identifying the winner/leader
    winner_row = _sort_by_finish(final_rd.dropna(subset=["round_position"])).head(1)

    col_narr, col_leader = st.columns([1, 1], gap="large")

    # ── Narrative ─────────────────────────────────────────────────────────────
    with col_narr:
        if narrative:
            label = "Tournament Summary" if is_final else f"R{sel_rnd} Story"
            bullet_html = "".join(
                f"<div style='font-size:13px;color:rgba(220,220,220,0.85);margin-bottom:5px'>· {_md(line)}</div>"
                for line in narrative
            )
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border-left:3px solid #38bdf8;"
                f"border-radius:0 8px 8px 0;padding:14px 16px;height:100%'>"
                f"<div style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
                f"color:#38bdf8;text-transform:uppercase;margin-bottom:10px'>{label}</div>"
                f"{bullet_html}</div>",
                unsafe_allow_html=True,
            )

    # ── Leader / Winner card ──────────────────────────────────────────────────
    with col_leader:
        if not winner_row.empty:
            winner_id   = int(winner_row.iloc[0]["dg_id"])
            winner_name = winner_row.iloc[0].get("player_name","")
            winner_data = ev[ev["dg_id"]==winner_id].sort_values("round_num")
            card_label  = "Winner" if is_final else f"Leader after R{sel_rnd}"

            w_round_cells = ""
            for rnum in rounds_avail:
                wrow   = winner_data[winner_data["round_num"]==rnum]
                is_cur = (rnum == sel_rnd)
                bg_c   = "background:rgba(56,189,248,0.12);border-radius:6px;padding:4px 10px;" if is_cur else "padding:4px 10px;"
                if wrow.empty:
                    w_round_cells += (
                        f"<div style='text-align:center;{bg_c}'>"
                        f"<div style='font-size:10px;color:rgba(100,100,100,0.5)'>R{rnum}</div>"
                        f"<div style='font-size:20px;color:rgba(100,100,100,0.35)'>–</div>"
                        f"<div style='font-size:10px;color:rgba(100,100,100,0.35)'>–</div></div>"
                    )
                    continue
                wr    = wrow.iloc[0]
                scr   = pd.to_numeric(wr.get("round_score", np.nan), errors="coerce")
                cp    = pd.to_numeric(wr.get("course_par",  np.nan), errors="coerce")
                tp    = pd.to_numeric(wr.get("to_par",      np.nan), errors="coerce")
                pos   = pd.to_numeric(wr.get("round_position", np.nan), errors="coerce")
                scr_c = _score_color(scr, cp) if pd.notna(scr) and pd.notna(cp) else "rgba(200,200,200,0.7)"
                w_round_cells += (
                    f"<div style='text-align:center;{bg_c}'>"
                    f"<div style='font-size:10px;color:rgba(140,140,140,0.55)'>R{rnum}</div>"
                    f"<div style='font-size:22px;font-weight:800;color:{scr_c}'>{_score_fmt(scr) or '–'}</div>"
                    f"<div style='font-size:12px;font-weight:700;color:{_par_color(tp)}'>{_par_fmt(tp)}</div>"
                    f"<div style='font-size:9px;color:rgba(120,120,120,0.5)'>pos {int(pos) if pd.notna(pos) else '–'}</div></div>"
                )

            sg_pills = ""
            for sg_col, sg_lbl in [("sg_ott","OTT"),("sg_app","APP"),("sg_arg","ARG"),("sg_putt","PUTT")]:
                if sg_col in winner_data.columns:
                    val = pd.to_numeric(winner_data[sg_col], errors="coerce").mean()
                    if pd.notna(val):
                        col = "#34d399" if val > 0.3 else ("#ef4444" if val < -0.1 else "rgba(180,180,180,0.7)")
                        sg_pills += (
                            f"<div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:4px 10px;text-align:center'>"
                            f"<div style='font-size:12px;font-weight:700;color:{col}'>{val:+.2f}</div>"
                            f"<div style='font-size:9px;color:rgba(120,120,120,0.55)'>{sg_lbl}</div></div>"
                        )

            fin_row = winner_data[winner_data["round_num"]==max(rounds_avail)]
            fin_ctp = fin_row.iloc[0].get("cum_to_par", np.nan) if not fin_row.empty else np.nan
            if pd.isna(fin_ctp) and "to_par" in winner_data.columns:
                fin_ctp = pd.to_numeric(winner_data["to_par"], errors="coerce").sum()
            total_strokes = sum(
                int(float(winner_data[winner_data["round_num"]==r].iloc[0]["round_score"]))
                for r in rounds_avail
                if not winner_data[winner_data["round_num"]==r].empty
                and pd.notna(winner_data[winner_data["round_num"]==r].iloc[0].get("round_score"))
            )

            hs_url = _get_hs(winner_id, winner_name, ID_TO_IMG, NAME_TO_IMG)
            hs_html = (
                f"<img src='{hs_url}' style='width:52px;height:52px;border-radius:50%;"
                f"border:2px solid rgba(255,255,255,0.15);object-fit:cover;flex-shrink:0' "
                f"onerror=\"this.style.display='none'\">"
            ) if hs_url else ""

            sg_row = f"<div style='display:flex;gap:6px;flex-wrap:wrap;margin-top:10px'>{sg_pills}</div>" if sg_pills else ""
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.09);"
                f"border-radius:12px;padding:14px 16px'>"
                f"<div style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:rgba(130,130,130,0.65);"
                f"text-transform:uppercase;margin-bottom:10px'>{card_label}</div>"
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:12px'>"
                f"{hs_html}"
                f"<div>"
                f"<div style='font-size:18px;font-weight:800;color:rgba(255,255,255,0.95)'>{winner_name}</div>"
                f"<div style='font-size:12px;color:rgba(140,140,140,0.55);margin-top:2px'>"
                f"{total_strokes} · "
                f"<span style='color:{_par_color(fin_ctp)};font-weight:700'>{_par_fmt(fin_ctp)}</span> total</div>"
                f"</div></div>"
                f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{w_round_cells}</div>"
                f"{sg_row}</div>",
                unsafe_allow_html=True,
            )

    # ── Superlatives row ──────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    _render_superlatives(ev, ev_rnd, sel_rnd, rounds_avail, is_final, ID_TO_IMG, NAME_TO_IMG)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — SCORING CONDITIONS + LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(f"<div style='{_SEC}'>Scoring Conditions by Round</div>", unsafe_allow_html=True)

    cond_cols = st.columns(len(rounds_avail), gap="small")
    for ci, rnum in enumerate(rounds_avail):
        rslice    = ev[ev["round_num"]==rnum]
        is_sel    = (rnum == sel_rnd)
        avg_score = rslice["round_score"].mean()  if "round_score" in rslice.columns else np.nan
        avg_par   = rslice["course_par"].median() if "course_par"  in rslice.columns else np.nan
        avg_birdie= rslice["birdies"].mean()      if "birdies"     in rslice.columns else np.nan
        avg_bogey = rslice["bogies"].mean()       if "bogies"      in rslice.columns else np.nan
        avg_dbl   = rslice["doubles_or_worse"].mean() if "doubles_or_worse" in rslice.columns else np.nan
        diff      = (avg_score - avg_par) if pd.notna(avg_score) and pd.notna(avg_par) else np.nan
        border    = "border:1px solid #38bdf8;" if is_sel else "border:1px solid rgba(255,255,255,0.07);"
        bg        = "background:rgba(56,189,248,0.06);" if is_sel else "background:rgba(255,255,255,0.02);"
        lbl_col   = "#38bdf8" if is_sel else "rgba(140,140,140,0.6)"
        diff_s   = _par_fmt(diff) if pd.notna(diff) else "–"
        diff_col = _par_color(diff) if pd.notna(diff) else "rgba(200,200,200,0.5)"
        score_s  = f"{avg_score:.1f}" if pd.notna(avg_score) else "–"
        birdie_s = f"{avg_birdie:.1f}" if pd.notna(avg_birdie) else "–"
        bogey_s  = f"{avg_bogey:.1f}" if pd.notna(avg_bogey) else "–"
        dbl_s    = f"{avg_dbl:.1f}" if pd.notna(avg_dbl) else "–"
        with cond_cols[ci]:
            st.markdown(
                f"<div style='{border}{bg}border-radius:10px;padding:12px 10px;text-align:center'>"
                f"<div style='font-size:11px;font-weight:700;color:{lbl_col};text-transform:uppercase;margin-bottom:6px'>R{rnum}</div>"
                f"<div style='font-size:22px;font-weight:800;color:{diff_col};line-height:1'>{diff_s}</div>"
                f"<div style='font-size:10px;color:rgba(140,140,140,0.55);margin-top:2px'>avg {score_s}</div>"
                f"<div style='display:flex;justify-content:space-around;margin-top:10px'>"
                f"<div><div style='font-size:13px;font-weight:700;color:#ef4444'>{birdie_s}</div>"
                f"<div style='font-size:9px;color:rgba(120,120,120,0.55)'>birdies</div></div>"
                f"<div><div style='font-size:13px;font-weight:700;color:#60a5fa'>{bogey_s}</div>"
                f"<div style='font-size:9px;color:rgba(120,120,120,0.55)'>bogeys</div></div>"
                f"<div><div style='font-size:13px;font-weight:700;color:rgba(180,180,180,0.55)'>{dbl_s}</div>"
                f"<div style='font-size:9px;color:rgba(120,120,120,0.55)'>doubles</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Leaderboard ────────────────────────────────────────────────────────────
    st.markdown(f"<div style='margin-top:16px;{_SEC}'>Leaderboard</div>", unsafe_allow_html=True)
    _render_leaderboard(ev, ev_rnd, sel_rnd, rounds_avail, is_final, final_rd,
                        has_to_par, has_cum_to_par, has_course_par)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C — SCORE TRACKER
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    _render_score_tracker(ev, sel_rnd, rounds_avail, is_final, final_rd)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION D — TOP 5 BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    if sg_avail:
        st.divider()
        _render_top5_breakdown(ev, ev_rnd, sel_rnd, rounds_avail, is_final, final_rd, sg_avail)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E — SHOT QUALITY HEATMAP
    # ══════════════════════════════════════════════════════════════════════════
    if all(c in ev.columns for c in ["birdies","bogies","doubles_or_worse"]):
        st.divider()
        _render_heatmap(ev, sel_rnd, rounds_avail, final_rd)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION F — STROKES GAINED
    # ══════════════════════════════════════════════════════════════════════════
    if sg_avail:
        st.divider()
        _render_sg_section(ev, ev_rnd, sel_rnd, rounds_avail, is_final, final_rd, sg_avail)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION G — BALL STRIKING
    # ══════════════════════════════════════════════════════════════════════════
    has_driving  = all(c in ev.columns for c in ["driving_dist","driving_acc"])
    has_approach = all(c in ev.columns for c in ["gir","scrambling"])
    if has_driving or has_approach:
        st.divider()
        _render_ball_striking(ev, ev_rnd, sel_rnd, is_final, final_rd, has_driving, has_approach)


# ─────────────────────────────────────────────────────────────────────────────
# Superlatives
# ─────────────────────────────────────────────────────────────────────────────

def _render_superlatives(ev, ev_rnd, sel_rnd, rounds_avail, is_final, id_to_img=None, name_to_img=None):
    cards = []

    if "round_score" in ev_rnd.columns:
        ev_rnd_clean = ev_rnd.dropna(subset=["round_score"]).copy()
        if not ev_rnd_clean.empty:
            best = ev_rnd_clean.nsmallest(1, "round_score").iloc[0]
            scr  = int(float(best["round_score"]))
            cp   = pd.to_numeric(best.get("course_par", np.nan), errors="coerce")
            tp   = scr - int(float(cp)) if pd.notna(cp) else np.nan
            tp_s = _par_fmt(tp)
            tp_c = _par_color(tp)
            dg   = int(best["dg_id"]) if pd.notna(best.get("dg_id")) else None
            hs   = _get_hs(dg, best["player_name"], id_to_img, name_to_img) if dg else None
            hs_h = (f"<img src='{hs}' style='width:40px;height:40px;border-radius:50%;"
                    f"border:2px solid rgba(255,255,255,0.12);object-fit:cover;margin-right:10px;flex-shrink:0'"
                    f" onerror=\"this.style.display='none'\">") if hs else ""
            cards.append((
                "Low Round",
                hs_h,
                _short(best["player_name"]),
                f"<span style='font-size:20px;font-weight:800;color:{tp_c}'>{tp_s}</span>"
                f"<span style='font-size:13px;color:rgba(160,160,160,0.6);margin-left:6px'>({scr})</span>",
                "#fbbf24",
            ))

    if sel_rnd > min(rounds_avail):
        prv = max(r for r in rounds_avail if r < sel_rnd)
        prev_pos = ev[ev["round_num"]==prv][["dg_id","player_name","round_position"]].copy()
        curr_pos = ev_rnd[["dg_id","player_name","round_position"]].copy()
        for df_ in [prev_pos, curr_pos]:
            df_["round_position"] = pd.to_numeric(df_["round_position"], errors="coerce")
        merged = curr_pos.merge(
            prev_pos[["dg_id","round_position"]].rename(columns={"round_position":"prev"}),
            on="dg_id", how="inner")
        merged["move"] = merged["prev"] - merged["round_position"]
        merged = merged.dropna(subset=["move"])
        if not merged.empty:
            best = merged.loc[merged["move"].idxmax()]
            if best["move"] > 0:
                dg  = int(best["dg_id"]) if pd.notna(best.get("dg_id")) else None
                hs  = _get_hs(dg, best["player_name"], id_to_img, name_to_img) if dg else None
                hs_h = (
                    f"<img src='{hs}' style='width:40px;height:40px;border-radius:50%;"
                    f"border:2px solid rgba(255,255,255,0.12);object-fit:cover;margin-right:10px;flex-shrink:0'"
                    f" onerror=\"this.style.display='none'\">"
                ) if hs else ""
                cards.append((
                    "Biggest Mover",
                    hs_h,
                    _short(best["player_name"]),
                    f"<span style='font-size:20px;font-weight:800;color:#22c55e'>▲{int(best['move'])}</span>"
                    f"<span style='font-size:13px;color:rgba(160,160,160,0.6);margin-left:6px'>spots to T{int(best['round_position'])}</span>",
                    "#22c55e",
                ))

    if not cards:
        return

    sup_cols = st.columns(len(cards), gap="medium")
    for ci, (title, hs_html, name, stat_html, accent) in enumerate(cards):
        with sup_cols[ci]:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);"
                f"border-top:2px solid {accent};border-radius:0 0 10px 10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;letter-spacing:0.08em;"
                f"color:rgba(130,130,130,0.65);text-transform:uppercase;margin-bottom:8px'>{title}</div>"
                f"<div style='display:flex;align-items:center'>"
                f"{hs_html}"
                f"<div>"
                f"<div style='font-size:14px;font-weight:700;color:rgba(255,255,255,0.9);margin-bottom:3px'>{name}</div>"
                f"<div>{stat_html}</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def _render_leaderboard(ev, ev_rnd, sel_rnd, rounds_avail, is_final, final_rd,
                         has_to_par, has_cum_to_par, has_course_par):

    lb = ev[ev["round_num"]==sel_rnd].copy()
    lb["round_position"] = pd.to_numeric(lb["round_position"], errors="coerce")

    # Always map finish_num and fin_text from final_rd so they are available on all round slices
    if "finish_num" in final_rd.columns:
        fn_map  = final_rd.set_index("dg_id")["finish_num"].to_dict()
        lb["finish_num"] = pd.to_numeric(lb["dg_id"].map(fn_map), errors="coerce")
    elif "finish_num" in lb.columns:
        lb["finish_num"] = pd.to_numeric(lb["finish_num"], errors="coerce")

    if "fin_text" in final_rd.columns:
        ft_map = final_rd.set_index("dg_id")["fin_text"].to_dict()
        lb["_fin_text"] = lb["dg_id"].map(ft_map)

    if is_final:
        # Final view: sort by finish_num (the definitive result)
        lb = lb.dropna(subset=["finish_num"]).sort_values("finish_num", ascending=True).reset_index(drop=True)
    else:
        # In-progress: sort by round_position then break ties with finish_num
        lb = _sort_by_finish(lb.dropna(subset=["round_position"]))

    if sel_rnd > min(rounds_avail):
        prv = max(r for r in rounds_avail if r < sel_rnd)
        pp  = ev[ev["round_num"]==prv][["dg_id","round_position"]].rename(columns={"round_position":"prev_pos"})
        lb  = lb.merge(pp, on="dg_id", how="left")
    else:
        lb["prev_pos"] = np.nan

    _drop = [c for c in ["round_score","to_par","cum_to_par","course_par"] if c in lb.columns]
    lb = lb.drop(columns=_drop)

    for rnum in rounds_avail:
        cols = ["dg_id","round_score"]
        if has_to_par:     cols.append("to_par")
        if has_cum_to_par: cols.append("cum_to_par")
        if has_course_par: cols.append("course_par")
        rslice = ev[ev["round_num"]==rnum][[c for c in cols if c in ev.columns]].copy()
        for c in rslice.columns[1:]: rslice[c] = pd.to_numeric(rslice[c], errors="coerce")
        rslice = rslice.rename(columns={
            "round_score":f"round_score_r{rnum}", "to_par":f"_tp{rnum}",
            "cum_to_par":f"_ctp{rnum}", "course_par":f"_cp{rnum}",
        })
        lb = lb.merge(rslice, on="dg_id", how="left")

    PAGE_SIZE = 25
    n_pages   = max(1, int(np.ceil(len(lb) / PAGE_SIZE)))
    if "eb_page" not in st.session_state: st.session_state["eb_page"] = 0
    if st.session_state.get("_eb_rnd_prev") != sel_rnd:
        st.session_state["eb_page"] = 0
        st.session_state["_eb_rnd_prev"] = sel_rnd
    page_idx = min(st.session_state["eb_page"], n_pages - 1)

    def _page_slice(df, page):
        start = page * PAGE_SIZE
        end   = min(start + PAGE_SIZE, len(df))
        if end < len(df):
            bp = df.iloc[end-1]["round_position"]
            while end < len(df) and df.iloc[end]["round_position"] == bp: end += 1
        return df.iloc[start:end]

    lb_page = _page_slice(lb, page_idx)

    if n_pages > 1:
        _pc1, _pc2, _pc3 = st.columns([1,1,20])
        with _pc1:
            if st.button("‹", key="eb_prev", disabled=(page_idx==0)):
                st.session_state["eb_page"] = page_idx - 1; st.rerun()
        with _pc2:
            if st.button("›", key="eb_next", disabled=(page_idx==n_pages-1)):
                st.session_state["eb_page"] = page_idx + 1; st.rerun()

    th_s = ("font-size:10px;font-weight:700;letter-spacing:0.07em;color:rgba(130,130,130,0.65);"
            "text-transform:uppercase;padding:7px 6px;border-bottom:1px solid rgba(255,255,255,0.08)")
    all_rounds = [1,2,3,4]
    rnd_hdr = ""
    for rnum in all_rounds:
        is_cur = (rnum == sel_rnd); has_d = rnum in rounds_avail
        bg  = "background:rgba(255,255,255,0.07);border-radius:3px;" if is_cur else ""
        col = ("color:rgba(255,255,255,0.9)" if is_cur else
               "color:rgba(150,150,150,0.6)" if has_d else "color:rgba(75,75,75,0.6)")
        rnd_hdr += f"<th style='{th_s};{bg}{col};text-align:center;width:52px'>R{rnum}</th>"

    table = (
        f"<table style='width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed'>"
        f"<thead><tr>"
        f"<th style='{th_s};text-align:left;width:48px'>POS</th>"
        f"<th style='{th_s};width:32px'></th>"
        f"<th style='{th_s};text-align:left'>PLAYER</th>"
        f"<th style='{th_s};text-align:center;width:58px'>TOTAL</th>"
        f"<th style='{th_s};text-align:center;width:58px'>ROUND</th>"
        f"{rnd_hdr}"
        f"<th style='{th_s};text-align:center;width:64px'>STROKES</th>"
        f"</tr></thead><tbody>"
    )

    for idx, (_, row) in enumerate(lb_page.iterrows()):
        if is_final:
            fin_num = pd.to_numeric(row.get("finish_num", np.nan), errors="coerce")
            if pd.notna(fin_num) and int(fin_num) == 1:
                pos_txt = "🏆"
            else:
                pos_txt = str(row.get("_fin_text", "")).strip()
                if not pos_txt or pos_txt == "nan":
                    pos_txt = str(int(fin_num)) if pd.notna(fin_num) else "—"
        else:
            pos_txt = str(row.get("round_position_text","")).strip()
            if not pos_txt or pos_txt == "nan":
                rp = row.get("round_position")
                pos_txt = str(int(rp)) if pd.notna(rp) else "—"

        pname     = str(row.get("player_name","")).strip()
        move_html = _move_html(row.get("round_position"), row.get("prev_pos"))
        total_par_val = row.get(f"_ctp{sel_rnd}", row.get(f"_tp{sel_rnd}", np.nan))
        rnd_par_val   = row.get(f"_tp{sel_rnd}", np.nan)
        total_strokes, total_valid = 0, False
        for rnum in rounds_avail:
            if rnum <= sel_rnd:
                v = row.get(f"round_score_r{rnum}", np.nan)
                if pd.notna(v): total_strokes += int(float(v)); total_valid = True
        rnd_cells = ""
        for rnum in all_rounds:
            is_cur  = (rnum == sel_rnd); has_d = rnum in rounds_avail
            bg_cell = "background:rgba(255,255,255,0.05);" if is_cur else ""
            if has_d and rnum <= sel_rnd:
                sv = row.get(f"round_score_r{rnum}", np.nan)
                cv = row.get(f"_cp{rnum}", np.nan)
                st_ = _score_fmt(sv) or "–"
                sc_ = (_score_color(sv, cv) if pd.notna(sv) and pd.notna(cv) else "rgba(200,200,200,0.7)")
            else:
                st_ = "–"; sc_ = "rgba(75,75,75,0.5)"
            rnd_cells += (f"<td style='padding:7px 4px;text-align:center;{bg_cell}"
                          f"color:{sc_};font-size:12px;font-weight:600'>{st_}</td>")
        stripe = "rgba(255,255,255,0.015)" if idx % 2 == 0 else "transparent"
        table += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.04);background:{stripe}'>"
            f"<td style='padding:7px 6px;font-weight:700;color:rgba(255,255,255,0.85);font-size:13px'>{pos_txt}</td>"
            f"<td style='padding:7px 4px;text-align:center'>{move_html}</td>"
            f"<td style='padding:7px 8px;font-weight:600;color:rgba(255,255,255,0.9);font-size:13px'>{pname}</td>"
            f"<td style='padding:7px 6px;text-align:center;font-size:14px;font-weight:800;color:{_par_color(total_par_val)}'>{_par_fmt(total_par_val)}</td>"
            f"<td style='padding:7px 6px;text-align:center;font-size:13px;font-weight:600;color:{_par_color(rnd_par_val)}'>{_par_fmt(rnd_par_val) if pd.notna(rnd_par_val) else '–'}</td>"
            f"{rnd_cells}"
            f"<td style='padding:7px 6px;text-align:center;font-size:12px;color:rgba(170,170,170,0.6)'>{str(total_strokes) if total_valid else '–'}</td>"
            f"</tr>"
        )
    table += "</tbody></table>"
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.02);border-radius:10px;"
        f"border:1px solid rgba(255,255,255,0.07);overflow-x:auto'>{table}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Score Tracker
# ─────────────────────────────────────────────────────────────────────────────

def _render_score_tracker(ev, sel_rnd, rounds_avail, is_final, final_rd):
    st.markdown(f"<div style='{_SEC}'>Score Tracker</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Cumulative score vs par · lower = better · "
        f"field median shown as grey band</div>",
        unsafe_allow_html=True,
    )

    all_names = sorted(ev["player_name"].dropna().unique())
    # Use tie-broken sort for default 5
    default_5 = _sort_by_finish(final_rd.dropna(subset=["round_position"])).head(5)["player_name"].dropna().tolist()
    highlight = st.multiselect("Highlight players", options=all_names, default=default_5,
                               key="eb_hl", placeholder="Pick players to track")
    players_to_draw = highlight if highlight else default_5

    rnds_show = [r for r in rounds_avail if r <= sel_rnd]

    field_p25, field_p75, field_med = {}, {}, {}
    for rnum in rnds_show:
        rslice = ev[ev["round_num"]==rnum]
        vals = pd.to_numeric(rslice.get("cum_to_par", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(vals) >= 4:
            field_p25[rnum] = float(vals.quantile(0.25))
            field_p75[rnum] = float(vals.quantile(0.75))
            field_med[rnum] = float(vals.median())

    fig = go.Figure()

    if len(field_p25) == len(rnds_show) and len(rnds_show) > 1:
        x_b  = rnds_show + list(reversed(rnds_show))
        y_b  = [field_p75[r] for r in rnds_show] + [field_p25[r] for r in reversed(rnds_show)]
        fig.add_trace(go.Scatter(x=x_b, y=y_b, fill="toself",
            fillcolor="rgba(150,150,150,0.07)", line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rnds_show, y=[field_med[r] for r in rnds_show],
            mode="lines", line=dict(color="rgba(140,140,140,0.3)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip"))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dash"))

    all_y = []
    for ci, pname in enumerate(players_to_draw):
        color  = COLORS[ci % len(COLORS)]
        fill_c = _hex_to_rgba(color, 0.1)
        x_pts, y_pts = [], []
        for rnum in rnds_show:
            val = _cum_score(ev, pname, rnum)
            x_pts.append(rnum); y_pts.append(val)
        while y_pts and pd.isna(y_pts[-1]):
            x_pts.pop(); y_pts.pop()
        if not x_pts: continue
        all_y.extend([v for v in y_pts if pd.notna(v)])
        last_x, last_y = x_pts[-1], y_pts[-1]
        label_s = _par_fmt(last_y) if pd.notna(last_y) else ""

        fig.add_trace(go.Scatter(x=x_pts, y=y_pts, fill="tozeroy", fillcolor=fill_c,
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))

        hover_texts = []
        for rnum, val in zip(x_pts, y_pts):
            rnd_row = ev[(ev["player_name"]==pname) & (ev["round_num"]==rnum)]
            rtp = pd.to_numeric(rnd_row.iloc[0].get("to_par", np.nan), errors="coerce") if not rnd_row.empty else np.nan
            pos = rnd_row.iloc[0].get("round_position", np.nan) if not rnd_row.empty else np.nan
            hover_texts.append(
                f"<b>{_short(pname)}</b><br>R{rnum}: {_par_fmt(rtp)} (round)<br>"
                f"Total: <b>{_par_fmt(val)}</b><br>Position: {int(pos) if pd.notna(pos) else '–'}")

        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts, mode="lines+markers",
            name=f"{_short(pname)} ({label_s})",
            line=dict(color=color, width=3),
            marker=dict(size=[11 if r==sel_rnd else 7 for r in x_pts], color=color,
                        line=dict(color="white", width=[2 if r==sel_rnd else 1 for r in x_pts])),
            text=hover_texts, hovertemplate="%{text}<extra></extra>",
        ))
        if pd.notna(last_y):
            fig.add_annotation(
                x=last_x+0.08, y=last_y,
                text=f"<b>{_short(pname)}</b> <span style='color:rgba(180,180,180,0.7);font-size:10px'>{label_s}</span>",
                showarrow=False, xanchor="left", font=dict(size=11, color=color),
            )

    y_min = (min(all_y)-2) if all_y else -20
    y_max = (max(all_y)+2) if all_y else 5
    y_min = min(y_min, -1); y_max = max(y_max, 1)
    tick_step = 2 if (y_max - y_min) < 20 else 5
    tick_vals = list(range(int(y_min)-1, int(y_max)+2, tick_step))

    fig.update_layout(
        height=440, template="plotly_dark",
        margin=dict(l=50, r=150, t=20, b=50),
        xaxis=dict(tickmode="array", tickvals=rnds_show,
                   ticktext=[f"R{r}" for r in rnds_show],
                   range=[min(rnds_show)-0.3, max(rnds_show)+0.7] if rnds_show else [0,1],
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=13)),
        yaxis=dict(title="Score vs Par", range=[y_max, y_min],
                   tickmode="array", tickvals=tick_vals,
                   ticktext=[_par_fmt(v) for v in tick_vals],
                   gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(size=11)),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shot quality heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _render_heatmap(ev, sel_rnd, rounds_avail, final_rd):
    st.markdown(f"<div style='{_SEC}'>Shot Quality Heatmap</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Top 20 finishers · birdies − bogeys − 2×doubles · "
        f"<span style='color:#34d399'>green = birdie-heavy</span> · "
        f"<span style='color:#60a5fa'>blue = bogey-heavy</span> · selected round highlighted</div>",
        unsafe_allow_html=True,
    )
    # Use tie-broken sort for top 20
    top20 = _sort_by_finish(final_rd.dropna(subset=["round_position"])).head(20)["player_name"].dropna().tolist()
    if not top20: return
    hmap_z, hmap_text = [], []
    for pname in top20:
        pev = ev[ev["player_name"]==pname]
        row_vals = []
        for rnum in rounds_avail:
            pr = pev[pev["round_num"]==rnum]
            if pr.empty: row_vals.append(np.nan)
            else:
                b  = float(pd.to_numeric(pr.iloc[0].get("birdies",0), errors="coerce") or 0)
                bg = float(pd.to_numeric(pr.iloc[0].get("bogies",0),  errors="coerce") or 0)
                d  = float(pd.to_numeric(pr.iloc[0].get("doubles_or_worse",0), errors="coerce") or 0)
                row_vals.append(b - bg - 2*d)
        hmap_z.append(row_vals)
        hmap_text.append([_par_fmt(v) if pd.notna(v) else "" for v in row_vals])
    all_vals = [v for row in hmap_z for v in row if pd.notna(v)]
    zmax     = max(abs(min(all_vals)), abs(max(all_vals)), 1) if all_vals else 5
    fig = go.Figure(go.Heatmap(
        z=hmap_z, x=[f"R{r}" for r in rounds_avail], y=[_short(n) for n in top20],
        text=hmap_text, texttemplate="%{text}",
        textfont=dict(size=11, color="rgba(255,255,255,0.85)"),
        colorscale=[[0.0,"rgba(96,165,250,0.9)"],[0.5,"rgba(25,25,35,0.9)"],[1.0,"rgba(52,211,153,0.9)"]],
        zmid=0, zmin=-zmax, zmax=zmax, showscale=False, xgap=3, ygap=2,
        hovertemplate="<b>%{y}</b> · %{x}<br>Score: <b>%{z:+.0f}</b><extra></extra>",
    ))
    sel_col_idx = rounds_avail.index(sel_rnd) if sel_rnd in rounds_avail else None
    if sel_col_idx is not None:
        fig.add_shape(type="rect", x0=sel_col_idx-0.5, x1=sel_col_idx+0.5,
                      y0=-0.5, y1=len(top20)-0.5,
                      line=dict(color="#38bdf8", width=2), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(
        height=max(300, len(top20)*26+60), template="plotly_dark",
        margin=dict(l=10,r=10,t=10,b=30),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SG Section
# ─────────────────────────────────────────────────────────────────────────────

def _render_sg_section(ev, ev_rnd, sel_rnd, rounds_avail, is_final, final_rd, sg_avail):
    src_label = "Full event averages" if is_final else f"R{sel_rnd} only"
    st.markdown(f"<div style='{_SEC}'>Strokes Gained Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='{_SUB}'>{src_label}</div>", unsafe_allow_html=True)

    sg_src = ev.copy() if is_final else ev_rnd.copy()
    for c in sg_avail + (["sg_total"] if "sg_total" in ev.columns else []):
        sg_src[c] = pd.to_numeric(sg_src[c], errors="coerce")
    sg_player_avg = (
        sg_src.groupby("player_name")[sg_avail + (["sg_total"] if "sg_total" in sg_src.columns else [])]
        .mean().reset_index()
    )

    st.markdown(f"<div style='{_SUB}'>Top & bottom 5 per category</div>", unsafe_allow_html=True)
    cat_cols = st.columns(len(sg_avail), gap="medium")
    for ci, c in enumerate(sg_avail):
        with cat_cols[ci]:
            label = SG_LABELS[c]; color = list(SG_COLORS.values())[ci]
            top5 = sg_player_avg.nlargest(5,c)[["player_name",c]].dropna()
            bot5 = sg_player_avg.nsmallest(5,c)[["player_name",c]].dropna()
            combined = pd.concat([top5,bot5]).drop_duplicates("player_name").sort_values(c,ascending=False)
            st.markdown(
                f"<div style='text-align:center;font-size:12px;font-weight:700;color:{color};"
                f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>SG: {label}</div>",
                unsafe_allow_html=True)
            fig_tb = go.Figure()
            fig_tb.add_trace(go.Bar(
                y=[_short(n) for n in combined["player_name"]], x=combined[c], orientation="h",
                marker=dict(color=[color if v>=0 else "#ef4444" for v in combined[c]]),
                hovertemplate="%{y}: <b>%{x:+.2f}</b><extra></extra>",
            ))
            fig_tb.update_layout(height=260, template="plotly_dark",
                margin=dict(l=5,r=10,t=5,b=20),
                xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
                           gridcolor="rgba(255,255,255,0.05)", title="SG"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_tb, use_container_width=True)

    all_names = sorted(ev["player_name"].dropna().unique())
    default_5 = _sort_by_finish(final_rd.dropna(subset=["round_position"])).head(5)["player_name"].dropna().tolist()
    highlight = st.session_state.get("eb_hl", default_5)
    players_to_draw = [p for p in highlight if p in all_names] or default_5

    if players_to_draw:
        st.markdown(
            f"<div style='{_SUB}'>SG breakdown for highlighted players · "
            f"{'event avg' if is_final else f'R{sel_rnd}'}</div>",
            unsafe_allow_html=True,
        )
        hl_cols = st.columns(min(len(players_to_draw), 5), gap="medium")
        for ci, pname in enumerate(players_to_draw[:5]):
            with hl_cols[ci]:
                color = COLORS[ci % len(COLORS)]
                st.markdown(
                    f"<div style='text-align:center;font-size:12px;font-weight:700;"
                    f"color:{color};margin-bottom:4px'>{_short(pname)}</div>",
                    unsafe_allow_html=True)
                pev = ev[ev["player_name"]==pname].copy()
                if not is_final:
                    pev = pev[pev["round_num"]==sel_rnd].copy()
                for c in sg_avail + (["sg_total"] if "sg_total" in ev.columns else []):
                    pev[c] = pd.to_numeric(pev[c], errors="coerce")
                pev = pev.sort_values("round_num")
                x_r = pev["round_num"].astype(int).tolist()
                if not x_r: continue

                fig_hl = go.Figure()
                for c in sg_avail:
                    fig_hl.add_trace(go.Bar(
                        x=x_r, y=pev[c], name=SG_LABELS[c],
                        marker=dict(color=SG_COLORS[c]),
                        hovertemplate=f"{SG_LABELS[c]}: <b>%{{y:.2f}}</b><extra></extra>"))
                if "sg_total" in pev.columns:
                    fig_hl.add_trace(go.Scatter(
                        x=x_r, y=pev["sg_total"], mode="markers",
                        name="Total", marker=dict(size=10, color="rgba(100,255,180,0.9)",
                                                   symbol="diamond", line=dict(color="white",width=1.5)),
                        hovertemplate="SG Total: <b>%{y:.2f}</b><extra></extra>"))
                fig_hl.update_layout(
                    barmode="relative", height=220, template="plotly_dark",
                    margin=dict(l=5,r=5,t=5,b=30),
                    xaxis=dict(tickmode="array", tickvals=x_r,
                               ticktext=[f"R{r}" for r in x_r],
                               gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.2)",
                               gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center",
                                font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                    showlegend=(ci==0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hl, use_container_width=True)

    has_t2g_cols = all(c in ev.columns for c in ["sg_ott","sg_app","sg_arg","sg_putt"])
    if has_t2g_cols:
        st.markdown(
            f"<div style='{_SUB}'>Tee-to-Green (OTT+APP) vs Short Game (ARG+PUTT) · "
            f"{'event avg' if is_final else f'R{sel_rnd}'} · "
            f"darker green = better finish</div>",
            unsafe_allow_html=True,
        )
        sdf = sg_src.copy()
        sdf["t2g"]   = pd.to_numeric(sdf.get("sg_ott",0), errors="coerce") + \
                       pd.to_numeric(sdf.get("sg_app",0), errors="coerce")
        sdf["short"] = pd.to_numeric(sdf.get("sg_arg",0), errors="coerce") + \
                       pd.to_numeric(sdf.get("sg_putt",0), errors="coerce")
        sdf = sdf.dropna(subset=["t2g","short"]).merge(
            final_rd[["dg_id","round_position"]].rename(columns={"round_position":"fin_pos"}),
            on="dg_id", how="left")
        sdf["fin_pos"] = pd.to_numeric(sdf["fin_pos"], errors="coerce")

        rnd_pos = ev_rnd[["dg_id","round_position"]].copy()
        rnd_pos["round_position"] = pd.to_numeric(rnd_pos["round_position"], errors="coerce")
        top10_rnd_ids = set(rnd_pos.nsmallest(10,"round_position")["dg_id"].tolist()) if not is_final else set()

        fig_t2g = go.Figure()
        for tx, ty, txt in [
            (sdf["t2g"].min()*0.85, sdf["short"].max()*0.85, "Short Game Gods"),
            (sdf["t2g"].max()*0.85, sdf["short"].max()*0.85, "Elite All-Around"),
            (sdf["t2g"].max()*0.85, sdf["short"].min()*0.85, "Ball Strikers"),
            (sdf["t2g"].min()*0.85, sdf["short"].min()*0.85, "Struggling"),
        ]:
            fig_t2g.add_annotation(x=tx, y=ty, text=txt, showarrow=False,
                font=dict(size=9, color="rgba(120,120,120,0.4)"), xanchor="center")

        fig_t2g.add_trace(go.Scatter(
            x=sdf["t2g"], y=sdf["short"],
            mode="markers",
            marker=dict(
                size=[13 if row["dg_id"] in top10_rnd_ids else 8 for _, row in sdf.iterrows()],
                color=-sdf["fin_pos"].fillna(sdf["fin_pos"].max()),
                colorscale="RdYlGn", showscale=True,
                colorbar=dict(title="Finish", tickfont=dict(size=9), thickness=12),
                opacity=0.8,
                line=dict(color=[("white" if row["dg_id"] in top10_rnd_ids else "rgba(255,255,255,0.15)")
                                  for _, row in sdf.iterrows()],
                          width=[2 if row["dg_id"] in top10_rnd_ids else 0.5 for _, row in sdf.iterrows()]),
            ),
            text=sdf["player_name"].apply(_short),
            hovertemplate="<b>%{text}</b><br>T2G: %{x:+.2f}<br>Short: %{y:+.2f}<extra></extra>",
            showlegend=False,
        ))

        top10_sdf = sdf[sdf["dg_id"].isin(top10_rnd_ids)]
        if not top10_sdf.empty:
            rnd_pos_dict = dict(zip(rnd_pos["dg_id"], rnd_pos["round_position"]))
            for _, row in top10_sdf.iterrows():
                rpos = rnd_pos_dict.get(row["dg_id"])
                lbl  = f"{int(rpos)} · {_short(row['player_name'])}" if pd.notna(rpos) else _short(row["player_name"])
                fig_t2g.add_annotation(
                    x=row["t2g"], y=row["short"]+0.04,
                    text=f"<b>{lbl}</b>", showarrow=False, xanchor="center",
                    font=dict(size=9, color="rgba(255,255,255,0.75)"),
                )

        fig_t2g.add_hline(y=0, line=dict(color="rgba(255,255,255,0.1)", dash="dot"))
        fig_t2g.add_vline(x=0, line=dict(color="rgba(255,255,255,0.1)", dash="dot"))
        fig_t2g.update_layout(
            height=460, template="plotly_dark",
            margin=dict(l=60,r=60,t=30,b=60),
            xaxis=dict(title="Tee-to-Green (OTT + APP)", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            yaxis=dict(title="Short Game (ARG + PUTT)", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_t2g, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ball Striking
# ─────────────────────────────────────────────────────────────────────────────

def _render_ball_striking(ev, ev_rnd, sel_rnd, is_final, final_rd, has_driving, has_approach):
    src_label = "Event averages" if is_final else f"R{sel_rnd}"
    st.markdown(f"<div style='{_SEC}'>Ball Striking</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>{src_label} · top round performers labelled · "
        f"green = better finish</div>",
        unsafe_allow_html=True,
    )

    if is_final:
        agg_cols = [c for c in ["driving_dist","driving_acc","gir","scrambling"] if c in ev.columns]
        scatter_src = ev.groupby(["dg_id","player_name"], as_index=False).agg({c:"mean" for c in agg_cols})
    else:
        scatter_src = ev_rnd.copy()

    scatter_src = scatter_src.merge(
        final_rd[["dg_id","round_position"]].rename(columns={"round_position":"fin_pos"}),
        on="dg_id", how="left")
    scatter_src["fin_pos"] = pd.to_numeric(scatter_src["fin_pos"], errors="coerce")

    rnd_pos = ev_rnd[["dg_id","round_position"]].copy()
    rnd_pos["round_position"] = pd.to_numeric(rnd_pos["round_position"], errors="coerce")
    top5_rnd_ids = set(rnd_pos.nsmallest(5,"round_position")["dg_id"].tolist())

    def _make_scatter(sdf, x_col, y_col, x_title, y_title, chart_title, x_pct=False, y_pct=False):
        sdf = sdf.dropna(subset=[x_col, y_col, "fin_pos"]).copy()
        sdf["player_name"] = sdf["player_name"].astype(str)
        xm = 100 if x_pct and sdf[x_col].max() <= 1 else 1
        ym = 100 if y_pct and sdf[y_col].max() <= 1 else 1
        top_ids = sdf[sdf["dg_id"].isin(top5_rnd_ids)].index
        rest_ids = sdf.index.difference(top_ids)

        fig = go.Figure()
        for idx_set, sz, show_text in [(rest_ids,8,False),(top_ids,13,True)]:
            sub = sdf.loc[sdf.index.isin(idx_set)]
            fig.add_trace(go.Scatter(
                x=sub[x_col]*xm, y=sub[y_col]*ym,
                mode="markers+text" if show_text else "markers",
                marker=dict(
                    size=sz,
                    color=-sub["fin_pos"].fillna(999),
                    colorscale="RdYlGn", showscale=False,
                    opacity=0.8 if not show_text else 1.0,
                    line=dict(width=2 if show_text else 0.5,
                              color="white" if show_text else "rgba(255,255,255,0.15)"),
                ),
                text=sub["player_name"].apply(_short) if show_text else None,
                textposition="top center",
                textfont=dict(size=10, color="rgba(255,255,255,0.85)"),
                hovertemplate=f"<b>%{{text}}</b><br>{x_title}: %{{x:.1f}}<br>{y_title}: %{{y:.1f}}<extra></extra>",
                showlegend=False,
            ))
        mx = (sdf[x_col]*xm).median(); my = (sdf[y_col]*ym).median()
        fig.add_hline(y=my, line=dict(color="rgba(255,255,255,0.08)", dash="dot"))
        fig.add_vline(x=mx, line=dict(color="rgba(255,255,255,0.08)", dash="dot"))
        fig.update_layout(
            height=380, template="plotly_dark",
            title=dict(text=chart_title, font=dict(size=13,color="rgba(200,200,200,0.8)"), x=0),
            margin=dict(l=20,r=20,t=40,b=50),
            xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.05)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    n_cols   = int(has_driving) + int(has_approach)
    scat_cols = st.columns(n_cols, gap="large")
    col_ptr  = 0

    if has_driving:
        with scat_cols[col_ptr]:
            st.plotly_chart(_make_scatter(
                scatter_src, "driving_dist","driving_acc",
                "Driving Distance (yds)","Driving Accuracy %",
                "Driving: Distance vs Accuracy", y_pct=True),
                use_container_width=True)
        col_ptr += 1

    if has_approach:
        with scat_cols[col_ptr]:
            st.plotly_chart(_make_scatter(
                scatter_src, "gir","scrambling",
                "GIR %","Scrambling %",
                "GIR % vs Scrambling %", x_pct=True, y_pct=True),
                use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Top 5 Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _render_top5_breakdown(ev, ev_rnd, sel_rnd, rounds_avail, is_final, final_rd, sg_avail):
    src_label = "Full event averages" if is_final else f"R{sel_rnd} only"

    st.markdown(f"<div style='{_SEC}'>Top 5 Breakdown — What Separated Them</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>Top 5 finishers vs field average · {src_label}</div>",
        unsafe_allow_html=True,
    )

    final_rd_clean = final_rd.dropna(subset=["round_position"]).copy()

    # FIX 3: Break ties with finish_num when identifying the top 5
    top5_rows = _sort_by_finish(final_rd_clean).head(5)

    if top5_rows.empty:
        st.info("Not enough data to build Top 5 breakdown.")
        return
    top5_names = top5_rows["player_name"].dropna().tolist()
    top5_ids   = top5_rows["dg_id"].dropna().tolist()

    sg_src = ev.copy() if is_final else ev_rnd.copy()
    score_src = ev.copy() if is_final else ev_rnd.copy()
    for c in sg_avail + ["sg_total","birdies","bogies","doubles_or_worse","round_score","to_par"]:
        if c in sg_src.columns:
            sg_src[c] = pd.to_numeric(sg_src[c], errors="coerce")
        if c in score_src.columns:
            score_src[c] = pd.to_numeric(score_src[c], errors="coerce")

    agg_cols = [c for c in sg_avail + ["sg_total","birdies","bogies","doubles_or_worse","to_par"]
                if c in sg_src.columns]
    player_avg = sg_src.groupby("player_name")[agg_cols].mean().reset_index()
    field_avg  = sg_src[agg_cols].mean()

    # ── Section 1: Top 5 identity cards ───────────────────────────────────────
    card_cols = st.columns(5, gap="small")
    for ci, pname in enumerate(top5_names[:5]):
        prow = top5_rows[top5_rows["player_name"] == pname]
        pos  = int(prow.iloc[0]["round_position"]) if not prow.empty else ci + 1

        p_rounds = ev[ev["player_name"] == pname].sort_values("round_num")
        round_cells = ""
        for rnum in rounds_avail:
            rrow = p_rounds[p_rounds["round_num"] == rnum]
            if rrow.empty:
                round_cells += f"<div style='text-align:center;padding:0 4px'><div style='font-size:9px;color:rgba(100,100,100,0.5)'>R{rnum}</div><div style='font-size:13px;color:rgba(100,100,100,0.35)'>–</div></div>"
                continue
            tp  = pd.to_numeric(rrow.iloc[0].get("to_par", np.nan), errors="coerce")
            scr = pd.to_numeric(rrow.iloc[0].get("round_score", np.nan), errors="coerce")
            cp  = pd.to_numeric(rrow.iloc[0].get("course_par", np.nan), errors="coerce")
            is_sel_rnd = (rnum == sel_rnd)
            bg = "background:rgba(56,189,248,0.15);border-radius:4px;" if is_sel_rnd else ""
            scr_c = _score_color(scr, cp) if pd.notna(scr) and pd.notna(cp) else "rgba(200,200,200,0.7)"
            round_cells += (
                f"<div style='text-align:center;padding:0 4px;{bg}'>"
                f"<div style='font-size:9px;color:rgba(120,120,120,0.55)'>R{rnum}</div>"
                f"<div style='font-size:14px;font-weight:700;color:{scr_c}'>{_par_fmt(tp)}</div>"
                f"</div>"
            )

        pavg = player_avg[player_avg["player_name"] == pname]
        best_sg_lbl = ""
        if not pavg.empty:
            sg_vals = {c: float(pavg.iloc[0][c]) for c in sg_avail if c in pavg.columns and pd.notna(pavg.iloc[0][c])}
            if sg_vals:
                best_sg_col = max(sg_vals, key=sg_vals.get)
                best_sg_val = sg_vals[best_sg_col]
                field_val   = float(field_avg.get(best_sg_col, 0))
                edge        = best_sg_val - field_val
                best_sg_lbl = (
                    f"<div style='font-size:10px;color:rgba(150,150,150,0.6);margin-top:6px'>"
                    f"Edge: <span style='color:#34d399;font-weight:700'>SG {SG_LABELS.get(best_sg_col,best_sg_col)} "
                    f"{edge:+.2f}</span> vs field</div>"
                )

        color = COLORS[ci % len(COLORS)]
        with card_cols[ci]:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);"
                f"border-top:3px solid {color};border-radius:8px;padding:12px 10px;text-align:center'>"
                f"<div style='font-size:11px;color:rgba(140,140,140,0.5);font-weight:600'>T{pos}</div>"
                f"<div style='font-size:14px;font-weight:800;color:rgba(240,240,240,0.9);margin:3px 0 8px'>"
                f"{_short(pname)}</div>"
                f"<div style='display:flex;justify-content:center;gap:2px;margin-bottom:6px'>{round_cells}</div>"
                f"{best_sg_lbl}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    # ── Section 2: SG — Top 5 vs Field (grouped bar) ──────────────────────────
    st.markdown(
        f"<div style='{_SUB}'>Strokes Gained — Top 5 vs Field Average · {src_label}</div>",
        unsafe_allow_html=True,
    )

    sg_fig = go.Figure()
    sg_cat_labels = [SG_LABELS.get(c, c) for c in sg_avail]

    field_vals_sg = [float(field_avg.get(c, 0)) for c in sg_avail]
    sg_fig.add_trace(go.Bar(
        name="Field Avg",
        x=sg_cat_labels,
        y=field_vals_sg,
        marker=dict(color="rgba(120,120,120,0.35)", line=dict(color="rgba(255,255,255,0.1)", width=1)),
        hovertemplate="Field Avg · %{x}: <b>%{y:+.2f}</b><extra></extra>",
    ))

    for ci, pname in enumerate(top5_names[:5]):
        pavg = player_avg[player_avg["player_name"] == pname]
        if pavg.empty:
            continue
        vals = [float(pavg.iloc[0][c]) if c in pavg.columns and pd.notna(pavg.iloc[0][c]) else 0
                for c in sg_avail]
        sg_fig.add_trace(go.Bar(
            name=_short(pname),
            x=sg_cat_labels,
            y=vals,
            marker=dict(color=COLORS[ci % len(COLORS)], opacity=0.85),
            hovertemplate=f"<b>{_short(pname)}</b> · %{{x}}: <b>%{{y:+.2f}}</b><extra></extra>",
        ))

    sg_fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    sg_fig.update_layout(
        barmode="group",
        height=340,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(tickfont=dict(size=12), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="SG", zeroline=False, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(sg_fig, use_container_width=True)

    # ── Section 3: Scoring stats table ────────────────────────────────────────
    stat_cols_avail = [c for c in ["birdies","bogies","doubles_or_worse","to_par"] if c in player_avg.columns]
    if stat_cols_avail:
        st.markdown(
            f"<div style='{_SUB}'>Scoring Stats — Top 5 vs Field Average · {src_label}</div>",
            unsafe_allow_html=True,
        )

        col_labels = {
            "birdies": "Birdies", "bogies": "Bogeys",
            "doubles_or_worse": "Doubles+", "to_par": "To Par",
        }
        col_colors = {
            "birdies": "#34d399", "bogies": "#ef4444",
            "doubles_or_worse": "#f97316", "to_par": "#38bdf8",
        }

        rows_html = ""
        field_row_cells = f"<td style='padding:7px 10px;font-size:11px;color:rgba(160,160,160,0.6);font-style:italic'>Field Avg</td>"
        for c in stat_cols_avail:
            fv = float(field_avg.get(c, 0))
            fmt = _par_fmt(fv) if c == "to_par" else f"{fv:.2f}"
            field_row_cells += f"<td style='padding:7px 10px;text-align:center;font-size:12px;color:rgba(160,160,160,0.55)'>{fmt}</td>"
        rows_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>{field_row_cells}</tr>"

        for ci, pname in enumerate(top5_names[:5]):
            pavg = player_avg[player_avg["player_name"] == pname]
            if pavg.empty:
                continue
            color = COLORS[ci % len(COLORS)]
            prow_cells = f"<td style='padding:7px 10px;font-size:12px;font-weight:700;color:{color}'>{_short(pname)}</td>"
            for c in stat_cols_avail:
                pv   = float(pavg.iloc[0][c]) if c in pavg.columns and pd.notna(pavg.iloc[0][c]) else 0.0
                fv   = float(field_avg.get(c, 0))
                better = (pv > fv) if c == "birdies" else (pv < fv)
                val_color = col_colors[c] if better else "rgba(220,220,220,0.75)"
                fmt = _par_fmt(pv) if c == "to_par" else f"{pv:.2f}"
                prow_cells += (
                    f"<td style='padding:7px 10px;text-align:center;font-size:13px;"
                    f"font-weight:700;color:{val_color}'>{fmt}</td>"
                )
            rows_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.04)'>{prow_cells}</tr>"

        header_cells = "<th style='padding:7px 10px;text-align:left;font-size:10px;color:rgba(120,120,120,0.6);font-weight:600;text-transform:uppercase;letter-spacing:0.07em'>Player</th>"
        for c in stat_cols_avail:
            lbl = col_labels.get(c, c)
            clr = col_colors.get(c, "rgba(180,180,180,0.5)")
            header_cells += (
                f"<th style='padding:7px 10px;text-align:center;font-size:10px;"
                f"color:{clr};font-weight:700;text-transform:uppercase;letter-spacing:0.07em'>{lbl}</th>"
            )

        table_html = (
            f"<div style='overflow-x:auto'>"
            f"<table style='width:100%;border-collapse:collapse;background:rgba(255,255,255,0.02);"
            f"border-radius:8px;overflow:hidden'>"
            f"<thead><tr style='border-bottom:1px solid rgba(255,255,255,0.08)'>{header_cells}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    # ── Section 4: How they won it ─────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='{_SUB}'>How They Won It — biggest SG edge vs field · {src_label}</div>",
        unsafe_allow_html=True,
    )

    all_edges_all_players = {}
    for pname in top5_names[:5]:
        pavg = player_avg[player_avg["player_name"] == pname]
        if pavg.empty:
            continue
        for c in sg_avail:
            if c not in pavg.columns:
                continue
            pv = float(pavg.iloc[0][c]) if pd.notna(pavg.iloc[0][c]) else 0.0
            fv = float(field_avg.get(c, 0))
            all_edges_all_players.setdefault(pname, {})[c] = pv - fv
    shared_max_abs = max(
        (abs(v) for edges in all_edges_all_players.values() for v in edges.values()),
        default=1,
    ) or 1

    edge_cols = st.columns(min(len(top5_names), 5), gap="medium")
    for ci, pname in enumerate(top5_names[:5]):
        pavg = player_avg[player_avg["player_name"] == pname]
        if pavg.empty:
            continue
        color = COLORS[ci % len(COLORS)]
        sg_edges = all_edges_all_players.get(pname, {})

        if not sg_edges:
            continue

        sorted_edges = sorted(sg_edges.items(), key=lambda x: x[1], reverse=True)
        bars_html = ""
        for cat, edge in sorted_edges:
            bar_w  = max(4, int(abs(edge) / shared_max_abs * 100))
            bar_c  = "#34d399" if edge >= 0 else "#ef4444"
            lbl    = SG_LABELS.get(cat, cat)
            bars_html += (
                f"<div style='display:flex;align-items:center;margin-bottom:5px;gap:6px'>"
                f"<div style='font-size:10px;color:rgba(160,160,160,0.6);width:34px;text-align:right;flex-shrink:0'>{lbl}</div>"
                f"<div style='flex:1;height:14px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden'>"
                f"<div style='width:{bar_w}%;height:100%;background:{bar_c};border-radius:3px'></div></div>"
                f"<div style='font-size:11px;font-weight:700;color:{bar_c};width:40px;flex-shrink:0'>{edge:+.2f}</div>"
                f"</div>"
            )

        with edge_cols[ci]:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);"
                f"border-top:3px solid {color};border-radius:8px;padding:12px'>"
                f"<div style='font-size:12px;font-weight:700;color:{color};margin-bottom:10px'>{_short(pname)}</div>"
                f"{bars_html}"
                f"</div>",
                unsafe_allow_html=True,
            )
