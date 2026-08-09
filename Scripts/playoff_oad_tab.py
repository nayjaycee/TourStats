"""
playoff_oad_tab.py
==================
Pick-5 OAD playoff league: FedEx St. Jude, BMW Championship, Tour Championship.

Rules:
  - Pick 5 golfers per event; each golfer can only be used once across all 3 events.
  - Points = actual earnings. Tour Championship = 0.5x multiplier.
  - Highest cumulative points wins.

Field logic mirrors oad_game_theory_tab.py:
  - Live field from this_week_field.csv / next_week_field.csv if event_id matches
  - Past filed event → use players from that event's CSV
  - Future event → synthetic pool based on event_type
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT   = Path(__file__).parent.parent
INUSE  = ROOT / "Data" / "in Use"
OAD_DIR = ROOT / "Data" / "OAD"

THIS_WEEK_PATH = INUSE / "this_week_field.csv"
NEXT_WEEK_PATH = INUSE / "next_week_field.csv"
ODDS_PATH      = INUSE / "this_week_odds.csv"
ROUNDS_PATH    = INUSE / "combined_roundlevel_2024_present.csv"
ALLROUNDS_PATH = ROOT / "Data" / "Clean" / "Combined" / "combined_rounds_all_2017_2026.csv"
LINEUPS_PATH   = OAD_DIR / "pick5_lineups.json"

try:
    from season_config import cfg as _scfg
    SCHEDULE_PATH = _scfg.SCHED_PATH
except ImportError:
    SCHEDULE_PATH = INUSE / "OAD_2026_Schedule.xlsx"

PLAYOFF_EVENT_IDS = {27, 28, 60}
TOUR_CHAMP_ID     = 60
TOUR_CHAMP_MULT   = 0.5
PICKS_PER_ENTRY   = 5

_MAJOR_EVENT_IDS = {14, 26, 33, 100}
_SIG_EVENT_IDS   = {5, 7, 9, 11, 12, 23, 34, 480, 541}
_ASIAN_CUTOFF    = 10000


def _to_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_pick5_events(events_dir: str) -> pd.DataFrame:
    d = ROOT / events_dir
    if not d.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(d.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            df["dg_id"]     = pd.to_numeric(df.get("dg_id", pd.Series(dtype=float)), errors="coerce")
            df["_event_id"] = pd.to_numeric(df["event_id"], errors="coerce") if "event_id" in df.columns else np.nan
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["Username"]  = combined["Username"].astype(str).str.strip().str.strip('"')
    combined["_username"] = combined["Username"].str.lower().str.strip()
    combined["_earn_raw"] = (
        combined["Earnings"].astype(str)
        .str.replace(r'[\$,\"]', "", regex=True).str.strip()
        .replace({"--": "0", "": "0"})
        .pipe(pd.to_numeric, errors="coerce").fillna(0)
    )
    combined["_mult"]         = np.where(combined["_event_id"] == TOUR_CHAMP_ID, TOUR_CHAMP_MULT, 1.0)
    combined["_earnings_num"] = combined["_earn_raw"] * combined["_mult"]
    return combined


@st.cache_data(ttl=300)
def _load_field_odds():
    this_week = pd.read_csv(THIS_WEEK_PATH) if THIS_WEEK_PATH.exists() else pd.DataFrame()
    next_week = pd.read_csv(NEXT_WEEK_PATH) if NEXT_WEEK_PATH.exists() else pd.DataFrame()
    odds      = pd.read_csv(ODDS_PATH)      if ODDS_PATH.exists()      else pd.DataFrame()
    return this_week, next_week, odds


@st.cache_data(ttl=3600)
def _get_regular_player_pool(min_events: int = 10) -> pd.DataFrame:
    if not ROUNDS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ROUNDS_PATH, usecols=["dg_id", "player_name", "tour", "event_id", "round_date"], low_memory=False)
    df["event_id"]   = pd.to_numeric(df["event_id"], errors="coerce")
    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    df = df[~((df["tour"].str.upper() == "PGA") & (df["event_id"] >= _ASIAN_CUTOFF))]
    cutoff_365 = pd.Timestamp.now() - pd.Timedelta(days=365)
    cutoff_120 = pd.Timestamp.now() - pd.Timedelta(days=120)
    pga        = df[(df["tour"].str.lower() == "pga") & (df["round_date"] >= cutoff_365)]
    recent_ids = set(pga[pga["round_date"] >= cutoff_120]["dg_id"].unique())
    ev_counts  = pga.groupby("dg_id")["event_id"].nunique()
    qualified  = set(ev_counts[ev_counts >= min_events].index) & recent_ids
    pool = pga[pga["dg_id"].isin(qualified)].drop_duplicates("dg_id")
    return pool[["dg_id", "player_name"]].reset_index(drop=True)


@st.cache_data(ttl=3600)
def _get_major_player_pool() -> pd.DataFrame:
    if not ROUNDS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ROUNDS_PATH, usecols=["dg_id", "player_name", "tour", "event_id", "round_date"], low_memory=False)
    df["event_id"]   = pd.to_numeric(df["event_id"], errors="coerce")
    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    df = df[~((df["tour"].str.upper() == "PGA") & (df["event_id"] >= _ASIAN_CUTOFF))]
    cutoff_365 = pd.Timestamp.now() - pd.Timedelta(days=365)
    cutoff_120 = pd.Timestamp.now() - pd.Timedelta(days=120)
    pga        = df[(df["tour"].str.lower() == "pga") & (df["round_date"] >= cutoff_365)]
    recent_ids = set(pga[pga["round_date"] >= cutoff_120]["dg_id"].unique())
    pool = pga[pga["event_id"].isin(_MAJOR_EVENT_IDS) & pga["dg_id"].isin(recent_ids)].drop_duplicates("dg_id")
    return pool[["dg_id", "player_name"]].reset_index(drop=True)


@st.cache_data(ttl=3600)
def _load_all_rounds_slim() -> pd.DataFrame:
    """Load full round history keeping only columns needed for course history."""
    path = ALLROUNDS_PATH if ALLROUNDS_PATH.exists() else ROUNDS_PATH
    if not path.exists():
        return pd.DataFrame()
    want = ["dg_id", "event_id", "sg_total", "finish_num", "round_date"]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in want if c in available]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["dg_id"]    = pd.to_numeric(df["dg_id"],    errors="coerce")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    if "sg_total"    in df.columns: df["sg_total"]    = pd.to_numeric(df["sg_total"],    errors="coerce")
    if "finish_num"  in df.columns: df["finish_num"]  = pd.to_numeric(df["finish_num"],  errors="coerce")
    if "round_date"  in df.columns: df["round_date"]  = pd.to_datetime(df["round_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def _compute_course_history(field_ids: tuple[int, ...], event_id: int, cutoff_str: str) -> pd.DataFrame:
    """Per-player avg SG total, avg finish, and number of starts at this event (historical)."""
    rdf = _load_all_rounds_slim()
    if rdf.empty or not event_id:
        return pd.DataFrame()
    cutoff = pd.Timestamp(cutoff_str)
    ev = rdf[
        (rdf["event_id"] == event_id) &
        rdf["dg_id"].isin(field_ids)
    ].copy()
    if "round_date" in ev.columns:
        ev = ev[ev["round_date"] < cutoff]
    if ev.empty:
        return pd.DataFrame()

    ev["_year"] = ev["round_date"].dt.year if "round_date" in ev.columns else 0

    starts_s = (
        ev.drop_duplicates(["dg_id", "_year"])
        .groupby("dg_id")["_year"].count()
        .rename("hist_starts")
    )
    avg_sg_s = ev.groupby("dg_id")["sg_total"].mean().rename("hist_sg_avg") if "sg_total" in ev.columns else pd.Series(dtype=float, name="hist_sg_avg")

    if "finish_num" in ev.columns:
        avg_fin_s = (
            ev.dropna(subset=["finish_num"])
            .drop_duplicates(["dg_id", "_year"])
            .groupby("dg_id")["finish_num"].mean()
            .rename("hist_avg_finish")
        )
    else:
        avg_fin_s = pd.Series(dtype=float, name="hist_avg_finish")

    return starts_s.to_frame().join(avg_sg_s, how="left").join(avg_fin_s, how="left").reset_index()


# ---------------------------------------------------------------------------
# Saved lineups helpers
# ---------------------------------------------------------------------------

def _load_saved_lineups() -> list[dict]:
    if not LINEUPS_PATH.exists():
        return []
    try:
        return json.loads(LINEUPS_PATH.read_text()).get("lineups", [])
    except Exception:
        return []


def _save_lineup(name: str, picks_by_eid: dict[int, set[int]], name_map: dict[int, str]) -> None:
    lineups = _load_saved_lineups()
    entry = {
        "name":     name.strip(),
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "picks":    {
            str(eid): [{"dg_id": did, "name": name_map.get(did, str(did))} for did in sorted(dids)]
            for eid, dids in picks_by_eid.items()
        },
    }
    # Replace if name already exists, otherwise append
    replaced = False
    for i, lu in enumerate(lineups):
        if lu.get("name", "").lower() == entry["name"].lower():
            lineups[i] = entry
            replaced = True
            break
    if not replaced:
        lineups.append(entry)
    LINEUPS_PATH.write_text(json.dumps({"lineups": lineups}, indent=2))


def _delete_lineup(name: str) -> None:
    lineups = [lu for lu in _load_saved_lineups() if lu.get("name", "") != name]
    LINEUPS_PATH.write_text(json.dumps({"lineups": lineups}, indent=2))


def _best_decimal_odds(row: pd.Series) -> float:
    books = ("draftkings", "fanduel", "betmgm", "caesars", "pinnacle", "bet365")
    vals  = [row.get(b, np.nan) for b in books]
    vals  = [v for v in vals if pd.notna(v) and v > 1]
    return float(min(vals)) if vals else np.nan


def _dec_to_pct(dec: float) -> float:
    return round(1 / dec * 100, 1) if pd.notna(dec) and dec > 1 else np.nan


def _model_tier(pct: float) -> str:
    if pct >= 10: return "Elite"
    if pct >= 5:  return "Contender"
    if pct >= 2:  return "Fringe"
    return              "Long Shot"


def _tier_style(val):
    colors = {"Elite": "#00CC96", "Contender": "#66D9A6", "Fringe": "#FFA07A", "Long Shot": "#EF553B"}
    c = colors.get(val, "")
    return f"color:{c};font-weight:600;" if c else ""


def _style_avail(val):
    if val is True:  return "background-color:#1a3a1a;color:#4ade80;"
    if val is False: return "background-color:#3a1a1a;color:#f87171;"
    return ""


def _last_first(nm: str) -> str:
    pts = str(nm).split(",")
    return f"{pts[0].strip()}, {pts[1].strip()[0]}." if len(pts) == 2 else nm


# ---------------------------------------------------------------------------
# Leaderboard (in-memory, sum of 5 picks with multipliers)
# ---------------------------------------------------------------------------

def _build_leaderboard(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["PLACE", "PLAYER", "_username", "EARNINGS"])
    lb = (
        events_df.groupby("_username")["_earnings_num"]
        .sum().sort_values(ascending=False).reset_index()
        .rename(columns={"_earnings_num": "EARNINGS"})
    )
    lb["EARNINGS"] = lb["EARNINGS"].astype(int)
    name_map  = events_df.drop_duplicates("_username").set_index("_username")["Username"].to_dict()
    lb["PLAYER"] = lb["_username"].map(name_map)
    lb["PLACE"]  = lb["EARNINGS"].rank(method="min", ascending=False).astype(int)
    return lb.sort_values("PLACE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Season charts
# ---------------------------------------------------------------------------

def _render_standings_chart(events_df: pd.DataFrame, sched: pd.DataFrame,
                             lb: pd.DataFrame, our_username: str) -> None:
    today = pd.Timestamp.now().normalize()
    eid_info: dict[int, tuple] = {}
    for _, row in sched.iterrows():
        eid = pd.to_numeric(row.get("event_id"), errors="coerce")
        if pd.notna(eid) and pd.notna(row.get("start_date")):
            eid_info[int(eid)] = (row["start_date"], str(row["event_name"]))

    if events_df.empty or "_event_id" not in events_df.columns:
        st.info("No completed playoff events yet.")
        return

    edf = events_df.dropna(subset=["_event_id"]).copy()
    edf["_event_id"] = edf["_event_id"].astype(int)
    filed_eids: list[int] = sorted(
        {eid for eid in edf["_event_id"].unique() if eid in eid_info and eid_info[eid][0] <= today},
        key=lambda e: eid_info[e][0],
    )
    if not filed_eids:
        st.info("No completed playoff events yet.")
        return

    # SUM 5 picks per entry per event (pick 5 format)
    picks = edf.groupby(["_username", "_event_id"])["_earnings_num"].sum().reset_index()
    pivot = (
        picks.pivot(index="_username", columns="_event_id", values="_earnings_num")
        .reindex(columns=filed_eids, fill_value=0).fillna(0)
    )
    ranks = pivot.cumsum(axis=1).rank(method="min", ascending=False, axis=0).astype(int)

    last_eid      = filed_eids[-1]
    current_ranks = ranks[last_eid].sort_values()
    top3_users    = list(current_ranks.index[:3])
    highlight     = list(dict.fromkeys(([our_username] if our_username else []) + top3_users))

    lb_rank: dict[str, int] = {
        str(row.get("_username", "")).lower(): int(row.get("PLACE", 999))
        for _, row in lb.iterrows()
    } if not lb.empty else {}

    disp_names: dict[str, str] = {}
    for u in highlight:
        rows = edf[edf["_username"] == u]["Username"]
        disp_names[u] = str(rows.iloc[0]) if not rows.empty else u

    short_labels = [
        (eid_info[e][1][:20] + "…" if len(eid_info[e][1]) > 20 else eid_info[e][1])
        for e in filed_eids
    ]

    fig = go.Figure()
    line_colors = ["#c084fc", "#818cf8", "#94a3b8"]
    for user in highlight:
        y_vals   = [int(ranks.loc[user, e]) if user in ranks.index and e in ranks.columns else None for e in filed_eids]
        rank_now = lb_rank.get(user, int(current_ranks.get(user, len(pivot))))
        if user == our_username:
            color, width, label = "#facc15", 3, f"Us (#{rank_now})"
        else:
            idx   = top3_users.index(user)
            color = line_colors[idx]
            width = 2
            label = f"#{rank_now}: {disp_names[user]}"
        fig.add_trace(go.Scatter(
            x=short_labels, y=y_vals, mode="lines+markers", name=label,
            line=dict(color=color, width=width), marker=dict(size=5), connectgaps=True,
        ))

    fig.add_hline(y=5, line_dash="dash", line_color="rgba(74,222,128,0.45)", line_width=1.5,
                  annotation_text=" Top 5", annotation_position="right",
                  annotation_font_color="rgba(74,222,128,0.8)", annotation_font_size=11)
    fig.update_layout(
        yaxis=dict(autorange="reversed", title="Rank", gridcolor="rgba(128,128,128,0.2)"),
        xaxis=dict(tickangle=-30, gridcolor="rgba(128,128,128,0.1)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(t=10, b=60, l=50, r=110),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="p5_rank_chart")


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render(cfg, sched: pd.DataFrame, selected_row=None) -> None:
    from oad_league_engine import compute_ef_scores, compute_field_stats, compute_course_fit_scores

    events_dir = cfg.events_dir or "Data/OAD/OAD_playoff_events"
    events_df  = _load_pick5_events(events_dir)
    lb         = _build_leaderboard(events_df)

    our_username = cfg.our_username.lower().strip() if cfg.our_username and cfg.our_username.lower() not in ("tbd", "") else None
    our_display  = cfg.our_display if cfg.our_display and cfg.our_display.lower() != "tbd" else "Us"
    n_league     = cfg.league_size
    cutline      = cfg.cutline

    if selected_row is None:
        st.info("No event selected.")
        return

    sel_name     = str(selected_row.get("event_name", ""))
    sel_event_id = int(selected_row["event_id"]) if pd.notna(selected_row.get("event_id")) else None
    _sd          = selected_row.get("start_date")
    sel_start    = pd.Timestamp(_sd) if pd.notna(_sd) else pd.NaT
    today        = pd.Timestamp.now().normalize()
    course_num   = int(selected_row["course_num"]) if pd.notna(selected_row.get("course_num")) else None
    cutoff_str   = (sel_start if pd.notna(sel_start) else today).isoformat()

    # oad_winner_share already stores the effective points value for this contest
    # (Tour Champ = $5M because 0.5x is baked into the schedule value)
    winner_share = int(selected_row["oad_winner_share"]) if pd.notna(selected_row.get("oad_winner_share")) else 0
    mult         = TOUR_CHAMP_MULT if sel_event_id == TOUR_CHAMP_ID else 1.0

    # Warn if non-playoff event selected
    if sel_event_id not in PLAYOFF_EVENT_IDS:
        st.warning(f"**{sel_name}** is not a Pick 5 playoff event. Navigate to St. Jude, BMW, or Tour Championship.")

    filed_event_ids: set[int] = set()
    if not events_df.empty and "_event_id" in events_df.columns:
        filed_event_ids = set(events_df["_event_id"].dropna().astype(int)) & PLAYOFF_EVENT_IDS

    # Standings chart
    st.markdown("### Season Standings")
    _render_standings_chart(events_df, sched, lb, our_username or "")
    st.divider()

    # ── Our position ─────────────────────────────────────────────────────────
    our_row      = lb[lb["_username"] == our_username] if our_username and not lb.empty else pd.DataFrame()
    our_place    = int(our_row["PLACE"].iloc[0])    if not our_row.empty else None
    our_earnings = int(our_row["EARNINGS"].iloc[0]) if not our_row.empty else 0

    teams_ahead     = lb[lb["PLACE"] < our_place].copy() if our_place else lb[lb["PLACE"] <= cutline].copy()
    ahead_usernames = set(teams_ahead["_username"].dropna())
    n_ahead         = len(ahead_usernames)

    cutline_row  = lb[lb["PLACE"] == cutline]
    cutline_earn = int(cutline_row["EARNINGS"].iloc[0]) if not cutline_row.empty else None
    gap_to_cut   = (cutline_earn - our_earnings) if cutline_earn is not None else None

    team_rank: dict[str, int] = {
        row["_username"]: int(row["PLACE"])
        for _, row in teams_ahead.iterrows() if pd.notna(row.get("_username"))
    }

    chasers          = lb[lb["EARNINGS"] < our_earnings].copy() if our_place else lb.iloc[0:0].copy()
    n_chasers        = len(chasers)
    chaser_usernames = set(chasers["_username"].dropna()) if "_username" in chasers.columns else set()

    # ── team_used: picks from playoff events BEFORE selected event ───────────
    eid_to_date: dict[int, pd.Timestamp] = {}
    for _, row in sched.iterrows():
        eid = pd.to_numeric(row.get("event_id"), errors="coerce")
        if pd.notna(eid) and pd.notna(row.get("start_date")):
            eid_to_date[int(eid)] = row["start_date"]

    cutoff_date     = sel_start if pd.notna(sel_start) else today
    prior_event_ids = {
        eid for eid in filed_event_ids
        if eid_to_date.get(eid, pd.Timestamp.max) < cutoff_date
    }
    prior_events = events_df[events_df["_event_id"].isin(prior_event_ids)] if not events_df.empty else pd.DataFrame()

    team_used: dict[str, set[int]] = {}
    if not prior_events.empty:
        for uname, grp in prior_events.groupby("_username"):
            team_used[str(uname)] = set(grp["dg_id"].dropna().astype(int).tolist())

    # This event's filed picks (5 per entry)
    this_week_events = events_df[events_df["_event_id"] == sel_event_id] if (sel_event_id and not events_df.empty) else pd.DataFrame()
    this_week_ownership: dict[int, int] = {}
    this_week_picks: dict[str, set[int]] = {}   # username → set of 5 dg_ids
    if not this_week_events.empty:
        for _, row in this_week_events.iterrows():
            did   = row.get("dg_id")
            uname = row.get("_username", "")
            if pd.notna(did) and uname:
                did_int = int(did)
                this_week_ownership[did_int] = this_week_ownership.get(did_int, 0) + 1
                this_week_picks.setdefault(str(uname), set()).add(did_int)
    have_this_week = bool(this_week_picks)

    # ── Field (same 4-step logic as 3.5k) ──────────────────────────────────
    this_week_df, next_week_df, odds_df = _load_field_odds()
    field_ids: set[int]          = set()
    field_name_map: dict[int, str] = {}

    def _field_event_id(df):
        return int(df["event_id"].iloc[0]) if not df.empty and "event_id" in df.columns and pd.notna(df["event_id"].iloc[0]) else None

    matched_field = pd.DataFrame()
    matched_from_this_week = False
    if sel_event_id is not None:
        if _field_event_id(this_week_df) == sel_event_id:
            matched_field = this_week_df
            matched_from_this_week = True
        elif _field_event_id(next_week_df) == sel_event_id:
            matched_field = next_week_df

    if not matched_field.empty:
        field_ids      = set(matched_field["dg_id"].dropna().astype(int).tolist())
        field_name_map = {int(r["dg_id"]): str(r["player_name"]) for _, r in matched_field.iterrows()}
        st.info(f"Using live field ({len(field_ids)} players).")
    elif not this_week_events.empty:
        valid          = this_week_events.dropna(subset=["dg_id"])
        field_ids      = set(valid["dg_id"].astype(int).tolist())
        field_name_map = {int(r["dg_id"]): str(r.get("Player", r["dg_id"])) for _, r in valid.iterrows()}
        st.info(f"Using filed event picks as field ({len(field_ids)} players).")
    else:
        sel_event_type = str(selected_row.get("event_type", "PLAYOFFS")).upper()
        if "MAJOR" in sel_event_type:
            pool      = _get_major_player_pool()
            pool_desc = "MAJOR field"
        else:
            pool      = _get_regular_player_pool(min_events=10)
            pool_desc = "REGULAR field (10+ PGA Tour events, last 365 days)"
        for _, r in pool.iterrows():
            did = int(r["dg_id"])
            field_ids.add(did)
            field_name_map[did] = str(r["player_name"])
        st.info(f"Synthetic {pool_desc}: {len(field_ids)} players.")

    # ── Odds ────────────────────────────────────────────────────────────────
    win_pct_map: dict[int, float]   = {}
    dg_model_map: dict[int, float]  = {}
    dg_hist_map: dict[int, float]   = {}
    best_odds_map: dict[int, float] = {}

    odds_available = matched_from_this_week and not odds_df.empty
    if odds_available:
        win_odds = odds_df[odds_df["market"] == "win"] if "market" in odds_df.columns else odds_df
        for _, row in win_odds.iterrows():
            did  = int(row["dg_id"])
            best = _best_decimal_odds(row)
            best_odds_map[did] = best
            win_pct_map[did]   = _dec_to_pct(best)
            if pd.notna(row.get("datagolf_baseline")):
                dg_model_map[did] = round(_dec_to_pct(float(row["datagolf_baseline"])), 1)
            if pd.notna(row.get("datagolf_base_history_fit")):
                dg_hist_map[did]  = round(_dec_to_pct(float(row["datagolf_base_history_fit"])), 1)

    # ── EF scores ───────────────────────────────────────────────────────────
    ef_map: dict[int, dict] = {}
    if field_ids:
        field_tuple = tuple(sorted(field_ids))
        ef_df = compute_ef_scores(field_tuple)
        if ef_df is not None and not ef_df.empty:
            for _, r in ef_df.iterrows():
                ef_map[int(r["dg_id"])] = {
                    "ef_score":  r["ef_score"],
                    "composite": r["composite"],
                    "ef_rank":   r["ef_rank"],
                }

    # ── Course fit ──────────────────────────────────────────────────────────
    cf_map: dict[int, float] = {}
    if course_num is not None and field_ids:
        cf_df = compute_course_fit_scores(tuple(sorted(field_ids)), course_num, cutoff_str)
        if not cf_df.empty:
            cf_map = dict(zip(cf_df["dg_id"].astype(int), cf_df["course_fit"]))

    # ── Course history ───────────────────────────────────────────────────────
    hist_map: dict[int, dict] = {}
    if sel_event_id is not None and field_ids:
        hist_df = _compute_course_history(tuple(sorted(field_ids)), sel_event_id, cutoff_str)
        if not hist_df.empty:
            for _, _hr in hist_df.iterrows():
                _did_h = int(_hr["dg_id"])
                hist_map[_did_h] = {
                    "hist_sg":      _hr.get("hist_sg_avg", np.nan),
                    "hist_finish":  _hr.get("hist_avg_finish", np.nan),
                    "hist_starts":  int(_hr.get("hist_starts", 0)),
                }

    # ── Our used players ────────────────────────────────────────────────────
    our_used: set[int] = team_used.get(our_username, set()) if our_username else set()
    best_pos           = (our_place - n_ahead) if our_place and n_ahead >= 0 else None

    # ── Metrics header ───────────────────────────────────────────────────────
    st.subheader(f"Pick 5 — {sel_name}")
    if mult != 1.0:
        st.caption(f"Points = earnings × {mult:.1f} for this event")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Our Rank",        f"#{our_place}" if our_place else "-")
    c2.metric("Our Points",      f"${our_earnings:,.0f}")
    c3.metric(f"Gap to #{cutline}",
              f"${gap_to_cut:,.0f}" if gap_to_cut is not None else "-",
              delta="In range" if gap_to_cut is not None and winner_share > gap_to_cut else "Out of range",
              delta_color="normal" if gap_to_cut is not None and winner_share > gap_to_cut else "inverse")
    c4.metric("Winner Share",   f"${winner_share:,.0f}" if winner_share else "-")
    c5.metric("Players Used",   f"{len(our_used)} / {(len(filed_event_ids) - (1 if sel_event_id in filed_event_ids else 0)) * PICKS_PER_ENTRY}")

    if n_chasers > 0:
        st.caption(f"**{n_chasers} teams** are behind us.")

    st.divider()

    # ── Our season summary ───────────────────────────────────────────────────
    with st.expander(f"Our Season — {our_display}", expanded=False):
        if our_username and not events_df.empty:
            our_df = events_df[events_df["_username"] == our_username].copy()
            all_our_eids = prior_event_ids | ({sel_event_id} if sel_event_id and sel_event_id in filed_event_ids else set())
            our_df = our_df[our_df["_event_id"].isin(all_our_eids)]
            if not our_df.empty:
                our_df["_sort_date"] = our_df["_event_id"].map(eid_to_date)
                our_df = our_df.sort_values("_sort_date")
                display = our_df[["Tournament", "Player", "Position"]].copy() if "Tournament" in our_df.columns else our_df[["_event_id", "Player"]].copy()
                display["Earnings"] = our_df["_earn_raw"].apply(lambda x: f"${int(x):,}" if x else "$0")
                display["Points"]   = our_df["_earnings_num"].apply(lambda x: f"${int(x):,}" if x else "$0")
                total_pts = int(our_df["_earnings_num"].sum())
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.caption(f"Total points: **${total_pts:,.0f}**")
            else:
                st.info("No pick history yet.")
        else:
            st.info("Set `our_username` in oad_leagues.json (pick5 league) to track your picks.")

    st.divider()

    # ── Build game theory table ──────────────────────────────────────────────
    if not field_ids:
        st.warning("No field data available.")
        return

    rows = []
    for did in sorted(field_ids):
        name     = field_name_map.get(did, str(did))
        dg_hist  = dg_hist_map.get(did, np.nan)
        dg_model = dg_model_map.get(did, np.nan)
        win_pct  = win_pct_map.get(did, np.nan)
        best_dec = best_odds_map.get(did, np.nan)

        model_score = dg_hist if pd.notna(dg_hist) else (dg_model if pd.notna(dg_model) else np.nan)
        tier        = _model_tier(model_score) if pd.notna(model_score) else ""

        # League-wide burned (prior events)
        league_burned = [u for u, ids in team_used.items() if did in ids]
        n_used        = len(league_burned)
        league_pct    = round(n_used / n_league * 100) if n_league > 0 else 0

        burned_ahead   = [u for u in league_burned if u in ahead_usernames]
        burned_chasers = [u for u in league_burned if u in chaser_usernames]
        n_ahead_used   = len(burned_ahead)
        n_behind_used  = len(burned_chasers)
        burned_ahead_pct = round(n_ahead_used / n_ahead   * 100) if n_ahead   > 0 else 0
        chaser_pct       = round(n_behind_used / n_chasers * 100) if n_chasers > 0 else 0

        ranks    = [team_rank[u] for u in burned_ahead if u in team_rank]
        avg_rank = round(float(np.mean(ranks)), 1) if ranks else np.nan

        # This week ownership (count of entries picking this player as one of 5)
        own_week = round(this_week_ownership.get(did, 0) / n_league * 100) if n_league > 0 else 0

        if have_this_week:
            picking_ahead    = [u for u, picks in this_week_picks.items() if u in ahead_usernames and did in picks]
            picking_behind   = [u for u, picks in this_week_picks.items() if u in chaser_usernames and did in picks]
            n_used_wk        = this_week_ownership.get(did, 0)
            n_ahead_used_wk  = len(picking_ahead)
            n_behind_used_wk = len(picking_behind)
            ahead_used_wk_pct = round(n_ahead_used_wk / n_ahead   * 100) if n_ahead   > 0 else 0
        else:
            n_used_wk = n_ahead_used_wk = n_behind_used_wk = 0
            ahead_used_wk_pct = np.nan

        we_can_use = did not in our_used
        we_used    = did in our_used
        ef_info    = ef_map.get(did, {})

        _hi = hist_map.get(did, {})
        rows.append({
            "dg_id":              did,
            "Player":             name,
            "Tier":               tier,
            "DG w/ Fit":          dg_hist,
            "DG Model %":         dg_model,
            "Win % (odds)":       win_pct,
            "Best Odds":          f"{best_dec:.1f}" if pd.notna(best_dec) else "-",
            "Course Fit":         cf_map.get(did, np.nan),
            "Hist SG":            _hi.get("hist_sg",     np.nan),
            "Hist Finish":        _hi.get("hist_finish", np.nan),
            "Hist Starts":        _hi.get("hist_starts", np.nan),
            "Used":               n_used,
            "Ahead Used":         n_ahead_used,
            "Behind Used":        n_behind_used,
            "% Used League":      league_pct,
            "% Ahead Used":       burned_ahead_pct,
            "% Behind Used":      chaser_pct,
            "Used (wk)":          n_used_wk         if have_this_week else np.nan,
            "Ahead Used (wk)":    n_ahead_used_wk   if have_this_week else np.nan,
            "% Ahead Used (wk)":  ahead_used_wk_pct if have_this_week else 0,
            "Avg Rank (users)":   avg_rank,
            "EF Score":           ef_info.get("ef_score",  np.nan),
            "EF Composite":       ef_info.get("composite", np.nan),
            "EF Rank":            ef_info.get("ef_rank",   np.nan),
            "We Can Use":         we_can_use,
            "We Used":            we_used,
        })

    gt_df = pd.DataFrame(rows)
    _has_ef = bool(ef_map)

    # Composite score
    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

    _signals, _weights = [], []
    if odds_available and gt_df["DG w/ Fit"].notna().any():
        _signals.append(_norm(gt_df["DG w/ Fit"].fillna(0)));     _weights.append(0.40)
    if odds_available and gt_df["Win % (odds)"].notna().any():
        _signals.append(_norm(gt_df["Win % (odds)"].fillna(0)));  _weights.append(0.25)
    if _has_ef and gt_df["EF Composite"].notna().any():
        _signals.append(_norm(gt_df["EF Composite"].fillna(0)));  _weights.append(0.35)

    if _signals:
        total_w = sum(_weights)
        _blend  = sum(s * (w / total_w) for s, w in zip(_signals, _weights))
        gt_df["Score"] = _blend * (1 + gt_df["% Ahead Used"] / 100)
    elif odds_available:
        gt_df["Score"] = gt_df["DG w/ Fit"] * (1 + gt_df["% Ahead Used"] / 100)
    else:
        gt_df["Score"] = gt_df["% Ahead Used"].fillna(0)

    gt_df = gt_df.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True)

    # ── Differentiation scatter ──────────────────────────────────────────────
    st.markdown(f"### Differentiation — {sel_name}")
    y_col   = "DG w/ Fit" if odds_available else ("EF Composite" if _has_ef else "% Used League")
    y_title = "DG w/ Fit Win %  →  higher is better" if odds_available else ("EF Composite" if _has_ef else "% Used League")
    st.caption(
        "X = % of teams ahead that have used this player (prior events).  "
        f"Y = {y_title}.  Top-left = high win chance + low prior usage among teams ahead."
    )

    chart_df = gt_df.dropna(subset=[y_col]).copy()
    chart_df["_status"] = chart_df.apply(
        lambda r: "We Used" if r["We Used"] else "Available" if r["We Can Use"] else "Already Used",
        axis=1,
    )
    color_map = {"Available": "#4ade80", "We Used": "#facc15", "Already Used": "#f87171"}
    top_idx   = set(chart_df.nlargest(min(8, len(chart_df)), y_col).index)

    fig = go.Figure()
    for status, color in color_map.items():
        sub = chart_df[chart_df["_status"] == status]
        if sub.empty:
            continue
        is_top = sub.index.isin(top_idx)
        for df_part, show_text in [(sub[~is_top], False), (sub[is_top], True)]:
            if df_part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=df_part["% Ahead Used"], y=df_part[y_col],
                mode="markers+text" if show_text else "markers",
                name=status if not show_text else None,
                marker=dict(size=10 if show_text else 7, color=color,
                            opacity=0.95 if show_text else 0.5,
                            line=dict(color="white" if show_text else "rgba(255,255,255,0.1)",
                                      width=1.5 if show_text else 0.5)),
                text=df_part["Player"].apply(_last_first) if show_text else None,
                textposition="top center", textfont=dict(size=8, color="rgba(255,255,255,0.85)"),
                customdata=df_part[["Player"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>" + y_title + ": %{y:.1f}<br>% Ahead Used: %{x:.0f}%<extra></extra>",
                showlegend=not show_text,
            ))

    if not chart_df.empty:
        fig.add_vline(x=float(chart_df["% Ahead Used"].median()), line_dash="dot", line_color="rgba(128,128,128,0.3)", line_width=1)
        fig.add_hline(y=float(chart_df[y_col].median()),           line_dash="dot", line_color="rgba(128,128,128,0.3)", line_width=1)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=500,
        xaxis=dict(title="% of Teams Ahead That Have Used This Player (Prior Events)", gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=40, l=50, r=20),
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True, key="p5_scatter")
    st.divider()

    # ── Field table ───────────────────────────────────────────────────────────
    st.markdown("### Field Table")
    fc1, fc2, fc3 = st.columns(3)
    show_avail = fc1.checkbox("Available to us only", value=False, key="p5_gt_avail")
    min_burned = fc2.slider("Min % burned (ahead)", 0, 100, 0, 5, key="p5_gt_burn")
    min_model  = fc3.slider("Min DG w/ Fit %", 0.0, 25.0, 0.0, 0.5, key="p5_gt_model", disabled=not odds_available)

    disp = gt_df.copy()
    if show_avail:
        disp = disp[disp["We Can Use"]]
    if min_burned > 0:
        disp = disp[disp["% Ahead Used"] >= min_burned]
    if min_model > 0 and odds_available:
        disp = disp[disp["DG w/ Fit"].notna() & (disp["DG w/ Fit"] >= min_model)]

    _ef_cols   = ["EF Score", "EF Composite", "EF Rank"] if _has_ef else []
    _has_cf    = bool(cf_map)
    _has_hist  = bool(hist_map)
    _cf_cols   = ["Course Fit"] if _has_cf else []
    _hist_cols = ["Hist SG", "Hist Finish", "Hist Starts"] if _has_hist else []
    _base_cols = (
        ["Score", "Player", "Tier", "DG w/ Fit", "Win % (odds)"]
        + _ef_cols + _cf_cols + _hist_cols
        + ["Used", "Ahead Used", "Behind Used"]
    )
    _wk_cols   = ["Used (wk)", "Ahead Used (wk)"] if have_this_week else []
    _meta_cols = ["We Can Use", "We Used"]
    show_cols  = _base_cols + _wk_cols + _meta_cols
    if not odds_available:
        show_cols = [c for c in show_cols if c not in ("DG w/ Fit", "Win % (odds)")]
    if n_chasers == 0:
        show_cols = [c for c in show_cols if "Behind" not in c]

    def _fmt_count_pct(total):
        def _fn(v):
            if pd.isna(v): return "-"
            pct = round(int(v) / total * 100) if total > 0 else 0
            return f"{int(v)} ({pct}%)"
        return _fn

    fmt = {
        "DG w/ Fit": "{:.1f}%", "Win % (odds)": "{:.1f}%", "Score": "{:.3f}",
        "EF Score": "{:.2f}", "EF Composite": "{:.2f}", "EF Rank": "{:.0f}",
        "Course Fit": "{:.2f}", "Hist SG": "{:+.2f}", "Hist Finish": "{:.1f}", "Hist Starts": "{:.0f}",
    }
    _fmt_all = {k: v for k, v in fmt.items() if k in disp.columns}
    _fmt_all["Used"]            = _fmt_count_pct(n_league)
    _fmt_all["Ahead Used"]      = _fmt_count_pct(n_ahead)
    _fmt_all["Behind Used"]     = _fmt_count_pct(n_chasers)
    _fmt_all["Used (wk)"]       = _fmt_count_pct(n_league)
    _fmt_all["Ahead Used (wk)"] = _fmt_count_pct(n_ahead)

    styled = (
        disp[[c for c in show_cols if c in disp.columns]].style
        .applymap(_style_avail, subset=["We Can Use", "We Used"])
        .applymap(_tier_style, subset=["Tier"])
        .format(_fmt_all, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)
    st.divider()

    # ── Recommendation (available to us) ─────────────────────────────────────
    st.markdown("### Recommendation — Available to Us")
    st.caption(f"Pick {PICKS_PER_ENTRY} players. Score blends model quality × prior-event usage among teams ahead.")

    avail_rec = gt_df[gt_df["We Can Use"] & gt_df["Score"].notna()].sort_values("Score", ascending=False).head(20)
    if not avail_rec.empty:
        rec_cols = _base_cols + _wk_cols
        if not odds_available:
            rec_cols = [c for c in rec_cols if c not in ("DG w/ Fit", "Win % (odds)")]
        styled_rec = (
            avail_rec[[c for c in rec_cols if c in avail_rec.columns]].style
            .applymap(_tier_style, subset=["Tier"])
            .format({k: v for k, v in _fmt_all.items() if k in avail_rec.columns}, na_rep="-")
        )
        st.dataframe(styled_rec, use_container_width=True, hide_index=True)
    else:
        st.info("No available players.")

    st.divider()

    # ── Lineup Builder ───────────────────────────────────────────────────────
    with st.expander("Lineup Builder", expanded=True):
        st.caption(
            "Plan your 5 picks per event across all 3 playoff events. "
            "Players already used in prior events are excluded. "
            "Each player can only appear in one event."
        )

        # Build option list sorted by Score (available to us only)
        _lb_rows = gt_df[gt_df["We Can Use"]].sort_values("Score", ascending=False, na_position="last")
        _lb_label_to_did: dict[str, int] = {}
        _lb_did_to_label: dict[int, str] = {}
        for _, _r in _lb_rows.iterrows():
            _did  = int(_r["dg_id"])
            _nm   = _r["Player"]
            _ef   = _r.get("EF Score",  np.nan)
            _win  = _r.get("DG w/ Fit", _r.get("Win % (odds)", np.nan))
            _cf   = _r.get("Course Fit", np.nan)
            _hs   = _r.get("Hist SG",    np.nan)
            _parts: list[str] = []
            if pd.notna(_ef):  _parts.append(f"EF:{_ef:.2f}")
            if pd.notna(_win): _parts.append(f"W:{_win:.1f}%")
            if pd.notna(_cf):  _parts.append(f"CF:{_cf:.2f}")
            if pd.notna(_hs):  _parts.append(f"HS:{_hs:+.2f}")
            _lbl = f"{_nm} ({', '.join(_parts)})" if _parts else _nm
            _lb_label_to_did[_lbl] = _did
            _lb_did_to_label[_did] = _lbl
        _lb_opts = list(_lb_label_to_did.keys())

        _ev_names = {27: "FedEx St. Jude", 28: "BMW Championship", 60: "Tour Championship"}
        _ev_mults = {27: 1.0, 28: 1.0, 60: 0.5}
        _ev_keys  = {27: "p5_plan_27",    28: "p5_plan_28",       60: "p5_plan_60"}
        _ev_order = [27, 28, 60]

        _cols = st.columns(3)
        _picks_by_eid: dict[int, list[str]] = {}
        for _col, _eid in zip(_cols, _ev_order):
            with _col:
                st.markdown(f"**{_ev_names[_eid]}**")
                _picks_by_eid[_eid] = st.multiselect(
                    f"picks_{_eid}",
                    options=_lb_opts,
                    max_selections=PICKS_PER_ENTRY,
                    key=_ev_keys[_eid],
                    label_visibility="collapsed",
                )

        # Resolve labels → dg_ids
        _dids_by_eid: dict[int, set[int]] = {
            _eid: {_lb_label_to_did[l] for l in _picks if l in _lb_label_to_did}
            for _eid, _picks in _picks_by_eid.items()
        }
        _all_selected: set[int] = set().union(*_dids_by_eid.values()) if _dids_by_eid else set()

        # Detect cross-event conflicts
        _conflicts: set[int] = set()
        for _i, _eid_a in enumerate(_ev_order):
            for _eid_b in _ev_order[_i + 1:]:
                _conflicts |= _dids_by_eid[_eid_a] & _dids_by_eid[_eid_b]
        if _conflicts:
            _cnames = [
                gt_df[gt_df["dg_id"] == d]["Player"].iloc[0]
                for d in _conflicts
                if not gt_df[gt_df["dg_id"] == d].empty
            ]
            st.error(f"Same player in multiple events: {', '.join(_cnames)}. Each player can only appear once.")

        # ── Save / Load lineups ──────────────────────────────────────────────
        st.divider()
        _saved = _load_saved_lineups()
        _sc1, _sc2 = st.columns([3, 1])
        with _sc1:
            _lu_name_input = st.text_input(
                "Lineup name", placeholder="e.g. Plan A", key="p5_lu_name_input", label_visibility="collapsed"
            )
        with _sc2:
            _do_save = st.button("Save lineup", key="p5_lu_save", use_container_width=True)

        if _do_save:
            if not _lu_name_input.strip():
                st.warning("Enter a name before saving.")
            elif not _all_selected:
                st.warning("No players selected yet.")
            else:
                _name_lookup = {int(_r["dg_id"]): _r["Player"] for _, _r in gt_df.iterrows()}
                _save_lineup(_lu_name_input.strip(), _dids_by_eid, _name_lookup)
                st.success(f"Saved **{_lu_name_input.strip()}**.")
                _saved = _load_saved_lineups()  # refresh

        if _saved:
            _lu_names = [lu["name"] for lu in _saved]
            _lc1, _lc2, _lc3 = st.columns([3, 1, 1])
            with _lc1:
                _sel_lu = st.selectbox(
                    "Saved lineups", _lu_names, key="p5_lu_sel", label_visibility="collapsed"
                )
            with _lc2:
                _do_load = st.button("Load", key="p5_lu_load", use_container_width=True)
            with _lc3:
                _do_delete = st.button("Delete", key="p5_lu_delete", use_container_width=True)

            if _do_delete and _sel_lu:
                _delete_lineup(_sel_lu)
                st.success(f"Deleted **{_sel_lu}**.")
                st.rerun()

            if _do_load and _sel_lu:
                _lu_data = next((lu for lu in _saved if lu["name"] == _sel_lu), None)
                if _lu_data:
                    for _k_eid, _k_key in _ev_keys.items():
                        _saved_picks = _lu_data["picks"].get(str(_k_eid), [])
                        _restored = []
                        for _sp in _saved_picks:
                            _sp_did = int(_sp["dg_id"])
                            _sp_lbl = _lb_did_to_label.get(_sp_did)
                            if _sp_lbl and _sp_lbl in _lb_opts:
                                _restored.append(_sp_lbl)
                            # else: player no longer in available pool, skip
                        st.session_state[_k_key] = _restored
                    st.success(f"Loaded **{_sel_lu}**.")
                    st.rerun()

            # Show saved lineup preview on hover via caption
            _lu_preview = next((lu for lu in _saved if lu["name"] == _sel_lu), None)
            if _lu_preview:
                _preview_parts = []
                for _peid, _pname in _ev_names.items():
                    _ppicks = [p["name"] for p in _lu_preview["picks"].get(str(_peid), [])]
                    if _ppicks:
                        _preview_parts.append(f"**{_pname}:** {', '.join(_ppicks)}")
                if _preview_parts:
                    st.caption("  |  ".join(_preview_parts) + f"  _(saved {_lu_preview.get('saved_at', '')[:10]})_")

        st.divider()

        # Plan summary table
        _summary_rows = []
        for _eid in _ev_order:
            for _lbl in _picks_by_eid[_eid]:
                _did2 = _lb_label_to_did.get(_lbl)
                if _did2 is None:
                    continue
                _r2 = gt_df[gt_df["dg_id"] == _did2]
                if _r2.empty:
                    continue
                _r2   = _r2.iloc[0]
                _ef2  = _r2.get("EF Score",  np.nan)
                _win2 = _r2.get("DG w/ Fit", _r2.get("Win % (odds)", np.nan))
                _cf2  = _r2.get("Course Fit", np.nan)
                _hs2  = _r2.get("Hist SG",    np.nan)
                _hf2  = _r2.get("Hist Finish", np.nan)
                _summary_rows.append({
                    "Event":        _ev_names[_eid],
                    "Player":       _r2["Player"],
                    "Score":        round(float(_r2["Score"]), 3) if pd.notna(_r2.get("Score")) else None,
                    "EF Score":     round(float(_ef2), 2)         if pd.notna(_ef2)              else None,
                    "Win % (DG)":   round(float(_win2), 1)        if pd.notna(_win2)             else None,
                    "Course Fit":   round(float(_cf2), 2)         if pd.notna(_cf2)              else None,
                    "Hist SG":      round(float(_hs2), 2)         if pd.notna(_hs2)              else None,
                    "Hist Finish":  round(float(_hf2), 1)         if pd.notna(_hf2)              else None,
                    "Pts Mult":     f"{_ev_mults[_eid]:.1f}x",
                    "Conflict":     "YES" if _did2 in _conflicts else "",
                })

        if _summary_rows:
            st.markdown("#### Plan Summary")
            _sum_df = pd.DataFrame(_summary_rows)
            if _sum_df["Conflict"].eq("").all():
                _sum_df = _sum_df.drop(columns=["Conflict"])
            _sum_fmt = {c: v for c, v in {
                "Score": "{:.3f}", "EF Score": "{:.2f}", "Win % (DG)": "{:.1f}%",
                "Course Fit": "{:.2f}", "Hist SG": "{:+.2f}", "Hist Finish": "{:.1f}",
            }.items() if c in _sum_df.columns}
            st.dataframe(
                _sum_df.style.format(_sum_fmt, na_rep="-"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Select players above to build your plan.")

        # Still available (not selected, not used)
        _still_avail = gt_df[
            gt_df["We Can Use"] & ~gt_df["dg_id"].isin(_all_selected)
        ].sort_values("Score", ascending=False, na_position="last")
        _sa_cols = [
            c for c in [
                "Player", "Score", "Tier", "EF Score", "EF Composite",
                "DG w/ Fit", "Win % (odds)", "Course Fit", "Hist SG", "Hist Finish", "Hist Starts",
            ]
            if c in _still_avail.columns
        ]
        with st.expander(f"Still available ({len(_still_avail)} players not yet planned)", expanded=False):
            _sa_fmt = {c: v for c, v in {
                "Score": "{:.3f}", "EF Score": "{:.2f}", "EF Composite": "{:.2f}",
                "DG w/ Fit": "{:.1f}%", "Win % (odds)": "{:.1f}%",
                "Course Fit": "{:.2f}", "Hist SG": "{:+.2f}", "Hist Finish": "{:.1f}", "Hist Starts": "{:.0f}",
            }.items() if c in _sa_cols}
            _sa_styler = _still_avail[_sa_cols].head(60).style.format(_sa_fmt, na_rep="-")
            if "Tier" in _sa_cols:
                _sa_styler = _sa_styler.applymap(_tier_style, subset=["Tier"])
            st.dataframe(_sa_styler, use_container_width=True, hide_index=True)

    st.divider()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    with st.expander("Optimizer", expanded=False):
        st.caption(
            "Multi-stage ILP optimizer. Simulates all 3 playoff events path-dependently "
            "(FSJ → BMW50 → TC30) and finds the highest-EV 15-player allocation."
        )
        _opt_used_dg = our_used  # dg_ids already spent in prior contest events
        _oc1, _oc2, _oc3 = st.columns(3)
        _opt_n_sims = _oc1.select_slider("Sims", [1000, 5000, 10000, 25000], value=10000, key="p5_opt_sims")
        _opt_seed   = _oc2.number_input("Seed", value=42, step=1, key="p5_opt_seed")
        _run_opt    = _oc3.button("Run Optimizer", key="p5_opt_run", use_container_width=True, type="primary")

        if _run_opt or "p5_opt_result" in st.session_state:
            if _run_opt:
                try:
                    from pick5_optimizer import (
                        load_and_match as _opt_load,
                        build_skill_ratings as _opt_skill,
                        run_simulation as _opt_sim,
                        run_optimizer as _opt_run,
                        shadow_prices as _opt_shadow,
                        STANDINGS_PATH as _OPT_SCHED_PATH,
                    )
                    with st.spinner(f"Running {_opt_n_sims:,} simulations…"):
                        _opt_st   = _opt_load()
                        _opt_sk   = _opt_skill(_opt_st["dg_id"].dropna().astype(int).tolist(),
                                               cutoff=pd.Timestamp(cutoff_str))
                        _opt_sr   = _opt_sim(_opt_st, _opt_sk, n_sims=int(_opt_n_sims), seed=int(_opt_seed))
                        _opt_res  = _opt_run(_opt_st, _opt_sk, sim_result=_opt_sr,
                                             our_used_dg_ids=_opt_used_dg)
                        _opt_shad = _opt_shadow(_opt_st, _opt_sk, sim_result=_opt_sr,
                                                our_used_dg_ids=_opt_used_dg)
                    st.session_state["p5_opt_result"]  = _opt_res
                    st.session_state["p5_opt_shadow"]  = _opt_shad
                    st.session_state["p5_opt_n_sims"]  = _opt_n_sims
                except Exception as _opt_err:
                    st.error(f"Optimizer error: {_opt_err}")

            _opt_res  = st.session_state.get("p5_opt_result")
            _opt_shad = st.session_state.get("p5_opt_shadow")
            _opt_nsim = st.session_state.get("p5_opt_n_sims", _opt_n_sims)

            if _opt_res is not None:
                # ── Point-estimate plan ──────────────────────────────────────
                st.markdown("#### Optimal Plan (point-estimate mode)")
                _pv1, _pv2, _pv3, _pv4 = st.columns(4)
                _pv1.metric("Expected Total", f"${_opt_res.mean_plan_pts:,.0f}")
                _pv2.metric("Median",         f"${_opt_res.median_plan_pts:,.0f}")
                _pv3.metric("P10",            f"${_opt_res.p10_plan_pts:,.0f}")
                _pv4.metric("P90",            f"${_opt_res.p90_plan_pts:,.0f}")

                if not _opt_res.plan_df.empty:
                    _plan_fmt = {"ev_pts": "{:,.0f}"}
                    st.dataframe(_opt_res.plan_df[["event_name", "player", "fec_rank", "ev_pts"]]
                                 .style.format(_plan_fmt, na_rep="-"),
                                 use_container_width=True, hide_index=True)

                # ── EV distribution chart ────────────────────────────────────
                import plotly.graph_objects as _go
                _pts = _opt_res.plan_total_pts
                _fig_dist = _go.Figure()
                _fig_dist.add_trace(_go.Histogram(
                    x=_pts, nbinsx=80, name="Sims",
                    marker_color="rgba(99,102,241,0.7)",
                ))
                _fig_dist.add_vline(x=float(_pts.mean()), line_dash="dash", line_color="#facc15",
                                    annotation_text=f" Mean ${_pts.mean():,.0f}", annotation_font_color="#facc15")
                _fig_dist.update_layout(
                    xaxis_title="Total Contest Points ($)", yaxis_title="Sims",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(t=10, b=40, l=50, r=10),
                )
                st.plotly_chart(_fig_dist, use_container_width=True, key="p5_opt_dist")

                # ── Frequency table ──────────────────────────────────────────
                st.markdown("#### Selection Frequency Across Sims")
                st.caption("How often each player appears in the optimal FSJ/BMW/TC allocation across all simulation paths.")
                _freq_show = _opt_res.freq_df[
                    (_opt_res.freq_df["freq_fsj_pct"] > 0) |
                    (_opt_res.freq_df["freq_bmw_pct"] > 0) |
                    (_opt_res.freq_df["freq_tc_pct"]  > 0)
                ].sort_values("freq_tc_pct", ascending=False).head(30)
                if not _freq_show.empty:
                    _freq_fmt = {c: "{:.1f}%" for c in ["freq_fsj_pct", "freq_bmw_pct", "freq_tc_pct"]}
                    _freq_fmt.update({"ev_27": "${:,.0f}", "ev_28": "${:,.0f}", "ev_60": "${:,.0f}"})
                    st.dataframe(
                        _freq_show[["player", "fec_rank", "freq_fsj_pct", "freq_bmw_pct", "freq_tc_pct",
                                    "ev_27", "ev_28", "ev_60"]]
                        .rename(columns={"freq_fsj_pct": "FSJ%", "freq_bmw_pct": "BMW%", "freq_tc_pct": "TC%",
                                         "ev_27": "EV FSJ", "ev_28": "EV BMW", "ev_60": "EV TC"})
                        .style.format(_freq_fmt, na_rep="-"),
                        use_container_width=True, hide_index=True,
                    )

                # ── Reserve list ─────────────────────────────────────────────
                st.markdown("#### Reserve List — P(qualifies for BMW / TC)")
                _res_fmt = {
                    "p_bmw": "{:.1f}%", "p_tc": "{:.1f}%",
                    "ev_27": "${:,.0f}", "ev_28": "${:,.0f}", "ev_60": "${:,.0f}", "ev_total": "${:,.0f}",
                }
                st.dataframe(
                    _opt_res.reserve_df.head(40)
                    [["player", "fec_rank", "p_bmw", "p_tc", "ev_27", "ev_28", "ev_60", "ev_total"]]
                    .rename(columns={"p_bmw": "P(BMW)", "p_tc": "P(TC)",
                                     "ev_27": "EV FSJ", "ev_28": "EV BMW",
                                     "ev_60": "EV TC",  "ev_total": "EV Total"})
                    .style.format(_res_fmt, na_rep="-"),
                    use_container_width=True, hide_index=True,
                )

                # ── Shadow prices ────────────────────────────────────────────
                if _opt_shad is not None and not _opt_shad.empty:
                    st.markdown("#### Shadow Prices — Value of Burning at FSJ vs Saving")
                    st.caption("Positive = burn at FSJ; Negative = hold for BMW/TC.")
                    _sp_fmt = {"ev_fsj": "${:,.0f}", "ev_bmw": "${:,.0f}", "ev_tc": "${:,.0f}",
                               "p_bmw_pct": "{:.1f}%", "p_tc_pct": "{:.1f}%",
                               "opp_cost": "${:,.0f}", "shadow_price": "${:,.0f}"}
                    _shad_show = _opt_shad[
                        (_opt_shad["ev_fsj"] > 0) | (_opt_shad["shadow_price"].notna())
                    ].head(40)
                    st.dataframe(
                        _shad_show[["player", "fec_rank", "ev_fsj", "ev_bmw", "ev_tc",
                                    "p_bmw_pct", "p_tc_pct", "opp_cost", "shadow_price"]]
                        .rename(columns={"p_bmw_pct": "P(BMW)", "p_tc_pct": "P(TC)",
                                         "opp_cost": "Opp Cost", "shadow_price": "Shadow $",
                                         "ev_fsj": "EV FSJ", "ev_bmw": "EV BMW", "ev_tc": "EV TC"})
                        .style.format(_sp_fmt, na_rep="-"),
                        use_container_width=True, hide_index=True,
                    )

    st.divider()

    # ── Teams ahead — best available ──────────────────────────────────────────
    st.markdown("### Teams Ahead — Best Available")
    _display_key = "DG w/ Fit" if odds_available else "Score"
    st.caption(f"Top 5 players each team ahead can still use (sorted by Score).")

    player_info = gt_df.set_index("dg_id")[[c for c in ["Player", "Score", _display_key, "Tier"] if c in gt_df.columns]].to_dict("index")

    team_rows = []
    for _, team_row in teams_ahead.sort_values("PLACE").iterrows():
        uname    = team_row["_username"]
        used_ids = team_used.get(uname, set())
        avail_ids = sorted(
            [d for d in field_ids if d not in used_ids],
            key=lambda d: player_info.get(d, {}).get("Score") or -999,
            reverse=True,
        )[:5]
        picks = []
        for did in avail_ids:
            info = player_info.get(did, {})
            last = info.get("Player", str(did)).split(",")[0].strip()
            val  = info.get(_display_key, np.nan)
            picks.append(f"{last} ({val:.1f}{'%' if 'Fit' in _display_key else ''})" if pd.notna(val) else last)

        # This week's picks for this team (up to 5)
        picking_now = ""
        if have_this_week:
            tw = this_week_picks.get(uname, set())
            names = [field_name_map.get(d, str(d)).split(",")[0].strip() for d in tw if d in field_ids]
            picking_now = ", ".join(sorted(names)) if names else "-"

        row_data = {
            "Place":   int(team_row["PLACE"]),
            "Team":    team_row["PLAYER"],
            "Points":  f"${int(team_row['EARNINGS']):,.0f}",
        }
        if have_this_week:
            row_data["Picking Now (5)"] = picking_now
        for i in range(5):
            row_data[f"Avail {i+1}"] = picks[i] if i < len(picks) else "-"
        team_rows.append(row_data)

    if team_rows:
        st.dataframe(pd.DataFrame(team_rows), use_container_width=True, hide_index=True,
                     height=min(50 + len(team_rows) * 35, 600))

    # ── Team inspector ────────────────────────────────────────────────────────
    with st.expander("Inspect a competitor's pick history", expanded=False):
        if teams_ahead.empty:
            st.info("No teams ahead.")
        else:
            sel_team  = st.selectbox("Team", options=teams_ahead["PLAYER"].tolist(), key="p5_team_sel")
            match_row = lb[lb["PLAYER"] == sel_team]
            sel_uname = match_row["_username"].iloc[0] if not match_row.empty else None
            if sel_uname and not events_df.empty:
                team_hist = events_df[events_df["_username"] == sel_uname].copy()
                team_hist = team_hist[team_hist["_event_id"].isin(prior_event_ids)]
                team_hist["In Field"] = team_hist["dg_id"].apply(
                    lambda d: "YES" if pd.notna(d) and int(d) in field_ids else ""
                )
                team_hist["Earnings"] = team_hist["_earn_raw"].apply(lambda x: f"${int(x):,}" if x else "$0")
                team_hist["Points"]   = team_hist["_earnings_num"].apply(lambda x: f"${int(x):,}" if x else "$0")
                if "Tournament" in team_hist.columns:
                    st.dataframe(team_hist[["Tournament", "Player", "Position", "In Field", "Earnings", "Points"]],
                                 use_container_width=True, hide_index=True)

                used_ids       = team_used.get(sel_uname, set())
                avail_for_team = field_ids - used_ids
                col1, col2 = st.columns(2)
                col1.metric("Players burned (prior events)", len(used_ids))
                col2.metric("Still available this event",   len(avail_for_team))

                if have_this_week:
                    tw = this_week_picks.get(sel_uname, set())
                    if tw:
                        names = [field_name_map.get(d, str(d)) for d in tw]
                        st.info(f"Picking this week: **{', '.join(names)}**")

    st.divider()

    # ── Field stats ───────────────────────────────────────────────────────────
    st.markdown("### Field Stats — Available Players")
    _cutoff_fs = sel_start if pd.notna(sel_start) else today
    fstats = compute_field_stats(tuple(sorted(field_ids)), _cutoff_fs.isoformat())

    if fstats.empty:
        st.warning("Insufficient round data.")
        return

    avail_map_fs = gt_df.set_index("dg_id")[["We Can Use", "We Used"]].to_dict("index")
    fstats["Available"] = fstats["dg_id"].map(lambda d: avail_map_fs.get(d, {}).get("We Can Use", True))
    fstats["We Used"]   = fstats["dg_id"].map(lambda d: avail_map_fs.get(d, {}).get("We Used", False))

    sc1, sc2 = st.columns([1, 3])
    win_sel   = sc1.segmented_control("Window", ["L12", "L24", "L36"], default="L12", key="p5_stats_win")
    avail_only = sc2.checkbox("Available to us only", value=True, key="p5_stats_avail")
    if not win_sel:
        win_sel = "L12"
    _w = win_sel[1:]

    disp_fs = fstats.copy()
    if avail_only:
        disp_fs = disp_fs[disp_fs["Available"]]

    col_map = {
        "Player":                  "player_name",
        f"SG Total ({win_sel})":   f"sg_total_L{_w}",
        f"OTT ({win_sel})":        f"sg_ott_L{_w}",
        f"APP ({win_sel})":        f"sg_app_L{_w}",
        f"ARG ({win_sel})":        f"sg_arg_L{_w}",
        f"Putt ({win_sel})":       f"sg_putt_L{_w}",
        f"T2G ({win_sel})":        f"sg_t2g_L{_w}",
        "Contender":               "contender_L36",
        "Momentum":                "momentum",
    }
    for ow in ("L12", "L24", "L36"):
        if ow != win_sel:
            col_map[f"SG Total ({ow})"] = f"sg_total_{ow}"

    available_cols = {k: v for k, v in col_map.items() if v in disp_fs.columns}
    show = disp_fs[[v for v in available_cols.values()]].copy()
    show.columns = list(available_cols.keys())
    sort_key = f"SG Total ({win_sel})"
    if sort_key in show.columns:
        show = show.sort_values(sort_key, ascending=False).reset_index(drop=True)
    show.insert(0, "Rank", range(1, len(show) + 1))
    show["Available"] = disp_fs["Available"].values[:len(show)]
    show["We Used"]   = disp_fs["We Used"].values[:len(show)]

    sg_cols = [c for c in show.columns if c not in ("Rank", "Player", "Available", "We Used", "Contender", "Momentum")]
    fmt_map  = {c: "{:+.2f}" for c in sg_cols}

    def _sg_color(val):
        try:
            v = float(val)
            if v > 0.5:  return "background-color:rgba(0,204,150,0.3)"
            if v > 0:    return "background-color:rgba(0,204,150,0.15)"
            if v > -0.5: return "background-color:rgba(239,85,59,0.15)"
            return "background-color:rgba(239,85,59,0.3)"
        except Exception:
            return ""

    styler = show.style.applymap(_sg_color, subset=[c for c in sg_cols if c in show.columns])
    styler = styler.applymap(_style_avail, subset=[c for c in ["Available", "We Used"] if c in show.columns])
    styler = styler.format(fmt_map, na_rep="-")
    st.caption(f"{len(show)} players")
    st.dataframe(styler, use_container_width=True, hide_index=True, height=700)
