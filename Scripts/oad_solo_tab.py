"""
oad_solo_tab.py - Solo OAD League (1k)
=======================================
Solo One-and-Done league. Data lives in a single flat CSV updated weekly.

CSV columns: username, entryId, eventName, eventDate, selection, winnings
Our username: AKsREVENGE (case-insensitive match)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -- Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
INUSE          = ROOT / "Data" / "in Use"
OAD_DIR        = ROOT / "Data" / "OAD"

SOLO_PATH      = OAD_DIR / "solo_OaD.csv"
THIS_WEEK_PATH = INUSE / "this_week_field.csv"
ODDS_PATH      = INUSE / "this_week_odds.csv"
SCHEDULE_PATH  = INUSE / "OAD_2026_Schedule.xlsx"
ROUNDS_PATH    = INUSE / "combined_roundlevel_2024_present.csv"
ALL_PLAYERS_PATH = INUSE / "All_players.xlsx"

OUR_USERNAME = "aksrevenge"   # lowercase match key
OUR_DISPLAY  = "AKsREVENGE"  # display label


def _to_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


# -- Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_solo() -> pd.DataFrame:
    """Load the flat solo CSV and normalise columns for downstream use."""
    if not SOLO_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SOLO_PATH)
    df["_username"]  = df["username"].astype(str).str.strip().str.lower()
    df["_slug"]      = df["eventName"].apply(_to_slug)
    df["_winnings"]  = pd.to_numeric(df["winnings"], errors="coerce").fillna(0)
    df["eventDate"]  = pd.to_datetime(df["eventDate"], errors="coerce")
    return df


@st.cache_data(ttl=300)
def _load_schedule() -> pd.DataFrame:
    if not SCHEDULE_PATH.exists():
        return pd.DataFrame()
    s = pd.read_excel(SCHEDULE_PATH)
    s["start_date"] = pd.to_datetime(s["start_date"], errors="coerce")
    s["_slug"] = s["event_name"].apply(_to_slug)
    return s


@st.cache_data(ttl=300)
def _load_field_odds():
    field = pd.read_csv(THIS_WEEK_PATH) if THIS_WEEK_PATH.exists() else pd.DataFrame()
    odds  = pd.read_csv(ODDS_PATH)      if ODDS_PATH.exists()      else pd.DataFrame()
    return field, odds


_NAME_ALIASES: dict[str, str] = {
    # solo file name (lower) → rounds canonical name (lower)
    "nicolas echavarria": "echavarria, nico",
    "john keefer":        "keefer, johnny",
    "hao-tong li":        "li, haotong",
}


@st.cache_data(ttl=3600)
def _build_name_to_dgid() -> dict[str, int]:
    """Build player name (lower) → dg_id supporting both 'Last, First' and 'First Last' formats."""
    lookup: dict[str, int] = {}

    def _add(name: str, did: int) -> None:
        key = str(name).strip().lower()
        lookup[key] = did
        # If "Last, First" format, also register as "First Last"
        if "," in key:
            parts = key.split(",", 1)
            flipped = f"{parts[1].strip()} {parts[0].strip()}"
            lookup[flipped] = did

    if ROUNDS_PATH.exists():
        df = pd.read_csv(ROUNDS_PATH, usecols=["dg_id", "player_name"], low_memory=False)
        for _, r in df.drop_duplicates("dg_id").iterrows():
            if pd.notna(r["player_name"]):
                _add(str(r["player_name"]), int(r["dg_id"]))
    if ALL_PLAYERS_PATH.exists():
        ap = pd.read_excel(ALL_PLAYERS_PATH)
        nc = next((c for c in ap.columns if "name" in c.lower()), None)
        ic = next((c for c in ap.columns if "dg_id" in c.lower()), None)
        if nc and ic:
            for _, r in ap.dropna(subset=[nc, ic]).iterrows():
                _add(str(r[nc]), int(r[ic]))
    return lookup


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
    return "Long Shot"


def _tier_style(val):
    colors = {"Elite": "#00CC96", "Contender": "#66D9A6", "Fringe": "#FFA07A", "Long Shot": "#EF553B"}
    c = colors.get(val, "")
    return f"color:{c};font-weight:600;" if c else ""


def _style_avail(val):
    if val is True:  return "background-color:#1a3a1a;color:#4ade80;"
    if val is False: return "background-color:#3a1a1a;color:#f87171;"
    return ""


# ─────────────────────────────────────────────────────────────────────────────

def render_oad_solo_tab() -> None:
    df        = _load_solo()
    sched     = _load_schedule()
    field_df, odds_df = _load_field_odds()

    if df.empty:
        st.warning("solo_OaD.csv not found.")
        return

    name_to_dgid = _build_name_to_dgid()
    dgid_to_name = {v: k for k, v in name_to_dgid.items()}

    # -- Map every selection to a dg_id ────────────────────────────────────────
    def _resolve(name: str) -> int | None:
        key = str(name).strip().lower()
        key = _NAME_ALIASES.get(key, key)
        return name_to_dgid.get(key)

    df["dg_id"] = df["selection"].apply(_resolve)

    # -- Leaderboard ───────────────────────────────────────────────────────────
    lb = (
        df.groupby("username")["_winnings"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"username": "PLAYER", "_winnings": "EARNINGS"})
    )
    lb["EARNINGS"]  = lb["EARNINGS"].astype(int)
    lb["_username"] = lb["PLAYER"].str.strip().str.lower()
    lb["PLACE"]     = lb["EARNINGS"].rank(method="min", ascending=False).astype(int)

    our_row      = lb[lb["_username"] == OUR_USERNAME]
    our_place    = int(our_row["PLACE"].iloc[0])    if not our_row.empty else None
    our_earnings = int(our_row["EARNINGS"].iloc[0]) if not our_row.empty else 0

    # Teams strictly ahead
    teams_ahead      = lb[lb["PLACE"] < our_place].copy() if our_place else lb[lb["PLACE"] <= 40].copy()
    ahead_usernames  = set(teams_ahead["_username"].dropna())
    n_ahead          = len(ahead_usernames)

    target_rank   = (our_place - 1) if our_place else 40
    target_row    = lb[lb["PLACE"] == target_rank]
    target_earn   = int(target_row["EARNINGS"].iloc[0]) if not target_row.empty else None
    gap_to_next   = (target_earn - our_earnings) if target_earn is not None else None

    rank7_row       = lb[lb["PLACE"] == 7]
    rank7_earnings  = int(rank7_row["EARNINGS"].iloc[0]) if not rank7_row.empty else None
    gap_to_40       = (rank7_earnings - our_earnings) if rank7_earnings is not None else None

    team_rank: dict[str, int] = {
        row["_username"]: int(row["PLACE"])
        for _, row in teams_ahead.iterrows() if pd.notna(row.get("_username"))
    }

    all_usernames = set(df["_username"].dropna().unique())
    n_league      = len(all_usernames)

    # -- Event selector ─────────────────────────────────────────────────────────
    current_event_name = field_df["event_name"].iloc[0] if not field_df.empty else ""
    current_slug       = _to_slug(current_event_name)
    today              = pd.Timestamp.now().normalize()
    filed_slugs: set[str] = set(df["_slug"].unique())

    event_options: list[dict] = []
    if not sched.empty:
        for _, row in sched.sort_values("start_date").iterrows():
            slug       = row["_slug"]
            name       = row["event_name"]
            start      = row["start_date"]
            etype      = str(row.get("event_type", "REGULAR")).upper()
            winner_sh  = int(row["oad_winner_share"]) if pd.notna(row.get("oad_winner_share")) else 0
            is_filed   = slug in filed_slugs
            is_current = slug == current_slug
            is_future  = pd.notna(start) and start > today
            is_major   = etype == "MAJOR"

            if is_filed or (is_future and pd.notna(start)):
                label = name
                if is_current:
                    label = f"{name} (current)"
                elif is_future:
                    label = f"{name} - {start.strftime('%b %-d')} {'[MAJOR]' if is_major else ''}"
                event_options.append({
                    "label": label.strip(), "name": name, "slug": slug,
                    "start_date": start, "is_current": is_current,
                    "is_future": is_future, "is_filed": is_filed,
                    "is_major": is_major, "winner_share": winner_sh,
                    "event_type": etype,
                })

    default_idx = next((i for i, e in enumerate(event_options) if e["is_current"]), 0)

    st.subheader("OAD Solo (1k)")
    st.caption(OUR_DISPLAY)

    sel_label = st.selectbox(
        "Event", options=[e["label"] for e in event_options],
        index=default_idx, key="solo_event_select",
    )
    sel_event = next((e for e in event_options if e["label"] == sel_label),
                     event_options[default_idx] if event_options else None)
    if sel_event is None:
        st.info("No events found.")
        return

    sel_slug       = sel_event["slug"]
    sel_name       = sel_event["name"]
    sel_is_major   = sel_event["is_major"]
    sel_event_type = sel_event["event_type"]
    sel_start      = sel_event["start_date"]
    winner_share   = sel_event["winner_share"]

    st.divider()

    # -- Cutoff: team_used = only events before selected event ─────────────────
    slug_to_date: dict[str, pd.Timestamp] = {}
    if not sched.empty:
        for _, row in sched.iterrows():
            if pd.notna(row["start_date"]):
                slug_to_date[row["_slug"]] = row["start_date"]

    cutoff_date = sel_start if pd.notna(sel_start) else today
    prior_slugs = {
        s for s in filed_slugs
        if s in slug_to_date and slug_to_date[s] < cutoff_date
    }
    prior_events = df[df["_slug"].isin(prior_slugs)]

    # team_used[username] = set of dg_ids burned in prior events
    team_used: dict[str, set[int]] = {}
    for uname, grp in prior_events.groupby("_username"):
        team_used[uname] = set(grp["dg_id"].dropna().astype(int).tolist())

    # This week's picks (from filed event data)
    this_week_events = df[df["_slug"] == sel_slug]
    this_week_ownership: dict[int, int] = {}
    this_week_picks: dict[str, int] = {}
    if not this_week_events.empty:
        for _, row in this_week_events.iterrows():
            did = row.get("dg_id")
            if pd.notna(did):
                did_int = int(did)
                this_week_ownership[did_int] = this_week_ownership.get(did_int, 0) + 1
                uname = row.get("_username", "")
                if uname:
                    this_week_picks[uname] = did_int

    # Also pull from this_week_field.csv if current event
    if sel_event["is_current"] and not field_df.empty and "username" in field_df.columns:
        for _, r in field_df.dropna(subset=["dg_id", "username"]).iterrows():
            uname = str(r["username"]).strip().lower()
            if uname not in this_week_picks:
                this_week_picks[uname] = int(r["dg_id"])

    have_this_week = bool(this_week_picks)

    # -- Field ─────────────────────────────────────────────────────────────────
    _MAJOR_IDS = {14, 26, 33, 100}
    _SIG_IDS   = {5, 7, 9, 11, 12, 23, 34, 480, 541}

    field_ids: set[int]        = set()
    field_name_map: dict[int, str] = {}

    if sel_event["is_current"] and not field_df.empty:
        field_ids      = set(field_df["dg_id"].dropna().astype(int).tolist())
        field_name_map = {int(r["dg_id"]): str(r["player_name"]) for _, r in field_df.iterrows()}
        st.info(f"Using live field ({len(field_ids)} players).")
    else:
        try:
            _rdf = pd.read_csv(
                ROUNDS_PATH,
                usecols=["dg_id", "player_name", "tour", "event_id", "round_date"],
                low_memory=False,
            )
            _rdf["round_date"] = pd.to_datetime(_rdf["round_date"], errors="coerce")
            _cutoff_365 = pd.Timestamp.now() - pd.Timedelta(days=365)
            _cutoff_120 = pd.Timestamp.now() - pd.Timedelta(days=120)
            _pga = _rdf[(_rdf["tour"].str.lower() == "pga") & (_rdf["round_date"] >= _cutoff_365)]
            _recent_ids = set(_pga[_pga["round_date"] >= _cutoff_120]["dg_id"].unique())

            if sel_is_major:
                _pool = _pga[_pga["event_id"].isin(_MAJOR_IDS) & _pga["dg_id"].isin(_recent_ids)].drop_duplicates("dg_id")
                _desc = f"MAJOR field: {len(_pool)} players"
            elif sel_event_type == "SIGNATURE":
                _sig_c = _pga[_pga["event_id"].isin(_SIG_IDS)].groupby("dg_id")["event_id"].nunique()
                _qual  = set(_sig_c[_sig_c >= 2].index) & _recent_ids
                _pool  = _pga[_pga["dg_id"].isin(_qual)].drop_duplicates("dg_id")
                _desc  = f"SIGNATURE field: {len(_pool)} players"
            else:
                _ec   = _pga.groupby("dg_id")["event_id"].nunique()
                _qual = set(_ec[_ec >= 15].index) & _recent_ids
                _pool = _pga[_pga["dg_id"].isin(_qual)].drop_duplicates("dg_id")
                _desc = f"REGULAR field: {len(_pool)} players"

            for _, r in _pool.iterrows():
                did = int(r["dg_id"])
                field_ids.add(did)
                field_name_map[did] = str(r["player_name"])
            st.info(f"Synthetic {_desc}.")
        except Exception as _e:
            st.sidebar.warning(f"Synthetic field error: {_e}")

    # -- Odds ──────────────────────────────────────────────────────────────────
    win_pct_map: dict[int, float]   = {}
    dg_model_map: dict[int, float]  = {}
    dg_hist_map: dict[int, float]   = {}
    best_odds_map: dict[int, float] = {}

    odds_available = sel_event["is_current"] and not odds_df.empty
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

    # -- Our used players ──────────────────────────────────────────────────────
    our_used: set[int] = team_used.get(OUR_USERNAME, set())

    # -- Jumpable / chasers ────────────────────────────────────────────────────
    our_new_earnings   = our_earnings + winner_share
    jumpable           = teams_ahead[teams_ahead["EARNINGS"] <= our_new_earnings]
    n_jumpable         = len(jumpable)
    jumpable_usernames = set(jumpable["_username"].dropna())

    if our_place and winner_share > 0:
        chasers = lb[
            (lb["EARNINGS"] < our_earnings) &
            (lb["EARNINGS"] > our_earnings - winner_share)
        ].copy()
    else:
        chasers = pd.DataFrame()
    n_chasers        = len(chasers)
    chaser_usernames = set(chasers["_username"].dropna())

    # -- Dashboard header ──────────────────────────────────────────────────────
    best_pos = (our_place - n_jumpable) if our_place else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Our Rank", f"#{our_place}" if our_place else "-")
    c2.metric("Our Earnings", f"${our_earnings:,.0f}")
    if gap_to_40 is not None:
        label_7 = "In range" if (winner_share > gap_to_40) else "Out of range"
        c3.metric(
            "Gap to #7", f"${gap_to_40:,.0f}",
            delta=label_7,
            delta_color="normal" if (winner_share > gap_to_40) else "inverse",
        )
    else:
        c3.metric("Gap to #7", "-")
    c4.metric("Winner's Share", f"${winner_share:,.0f}" if winner_share else "-")
    c5.metric(
        "Best rank if pick wins", f"#{best_pos}" if best_pos else "-",
        delta=f"+{n_jumpable} spots" if n_jumpable else None,
    )

    if n_chasers > 0:
        st.caption(
            f"**{n_chasers} teams** are within one winner's share of catching us "
            f"(earnings between ${our_earnings - winner_share:,.0f} and ${our_earnings:,.0f})."
        )

    st.divider()

    # -- Our season ────────────────────────────────────────────────────────────
    with st.expander(f"Our Season - {OUR_DISPLAY}", expanded=False):
        all_our_slugs = prior_slugs | ({sel_slug} if sel_slug in filed_slugs else set())
        our_picks_df  = df[(df["_username"] == OUR_USERNAME) & (df["_slug"].isin(all_our_slugs))].copy()

        if not our_picks_df.empty and not sched.empty:
            slug_order = {row["_slug"]: row["start_date"] for _, row in sched.iterrows() if pd.notna(row["start_date"])}
            our_picks_df["_sort_date"] = our_picks_df["_slug"].map(lambda s: slug_order.get(s, pd.NaT))
            our_picks_df = our_picks_df.sort_values("_sort_date")

        if not our_picks_df.empty:
            display = our_picks_df[["eventName", "selection"]].copy()
            display.columns = ["Tournament", "Player"]
            display["Earnings"] = our_picks_df["_winnings"].apply(lambda x: f"${int(x):,}" if x != 0 else "$0")
            total_earned = int(our_picks_df["_winnings"].sum())
            n_events     = len(our_picks_df)
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption(
                f"**{n_events} events**  |  Total pick earnings: **${total_earned:,.0f}**  |  "
                f"Avg per event: **${total_earned // n_events:,.0f}**"
            )
        else:
            st.info("No pick history found.")

    st.divider()

    # -- Build game theory table ───────────────────────────────────────────────
    rows = []
    for did in sorted(field_ids):
        name     = field_name_map.get(did, str(did))
        dg_hist  = dg_hist_map.get(did, np.nan)
        dg_model = dg_model_map.get(did, np.nan)
        win_pct  = win_pct_map.get(did, np.nan)
        best_dec = best_odds_map.get(did, np.nan)

        model_score = dg_hist if pd.notna(dg_hist) else (dg_model if pd.notna(dg_model) else np.nan)
        tier        = _model_tier(model_score) if pd.notna(model_score) else ""

        burned_ahead     = [u for u, ids in team_used.items() if u in ahead_usernames and did in ids]
        burned_ahead_pct = round(len(burned_ahead) / n_ahead * 100) if n_ahead > 0 else 0
        ranks            = [team_rank[u] for u in burned_ahead if u in team_rank]
        avg_rank         = round(float(np.mean(ranks)), 1) if ranks else np.nan

        jump_burned = [u for u in burned_ahead if u in jumpable_usernames]
        jump_pct    = round(len(jump_burned) / n_jumpable * 100) if n_jumpable > 0 else 0

        chaser_burned = [u for u, ids in team_used.items() if u in chaser_usernames and did in ids]
        chaser_pct    = round(len(chaser_burned) / n_chasers * 100) if n_chasers > 0 else 0

        league_burned = [u for u, ids in team_used.items() if did in ids]
        league_pct    = round(len(league_burned) / n_league * 100) if n_league > 0 else 0

        own_week = round(this_week_ownership.get(did, 0) / n_league * 100) if n_league > 0 else 0

        if have_this_week:
            picking_ahead     = [u for u, d in this_week_picks.items() if u in ahead_usernames and d == did]
            picking_ahead_pct = round(len(picking_ahead) / n_ahead * 100) if n_ahead > 0 else 0
        else:
            picking_ahead_pct = np.nan

        if have_this_week and winner_share > 0:
            our_new  = our_earnings + winner_share
            new_rank = 1
            for _, t_row in lb.iterrows():
                if t_row["_username"] == OUR_USERNAME:
                    continue
                t_uname = t_row.get("_username", "")
                t_earn  = int(t_row["EARNINGS"])
                t_new   = t_earn + winner_share if this_week_picks.get(t_uname) == did else t_earn
                if t_new > our_new:
                    new_rank += 1
        else:
            new_rank = np.nan

        we_can_use = did not in our_used
        we_used    = did in our_used

        rows.append({
            "dg_id":               did,
            "Player":              name,
            "Tier":                tier,
            "DG w/ Fit":           dg_hist,
            "DG Model %":          dg_model,
            "Win % (odds)":        win_pct,
            "Best Odds":           f"{best_dec:.1f}" if pd.notna(best_dec) else "-",
            "% Burned (ahead)":    burned_ahead_pct,
            "% Burned (jump)":     jump_pct,
            "% Burned (chasers)":  chaser_pct,
            "% Burned (league)":   league_pct,
            "Picking Now (ahead)": picking_ahead_pct,
            "Own % this week":     own_week,
            "Avg Rank (users)":    avg_rank,
            "New Rank":            new_rank,
            "We Can Use":          we_can_use,
            "We Used":             we_used,
        })

    gt_df    = pd.DataFrame(rows)
    sort_col = "DG w/ Fit" if odds_available else "% Burned (ahead)"
    gt_df    = gt_df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    # -- Differentiation scatter ───────────────────────────────────────────────
    st.markdown(f"### Differentiation - {sel_name}")
    y_col   = "DG w/ Fit" if odds_available else "% Burned (league)"
    y_title = "DG w/ Fit Win %  ->  higher is better" if odds_available else "% Burned (league)  ->  popularity proxy"
    st.caption(
        "X = % of teams ahead that have burned (can't reuse) this player.  "
        f"Y = {'DG model w/ history fit win %' if odds_available else '% burned league-wide'}.  "
        "Top-right = high win chance + fewer competitors ahead."
    )

    chart_df = gt_df.dropna(subset=[y_col]).copy()
    chart_df["_status"] = chart_df.apply(
        lambda r: "We Used"   if r["We Used"]   else
                  "Available" if r["We Can Use"] else "Already Used/Excl.",
        axis=1,
    )
    color_map = {"Available": "#4ade80", "We Used": "#facc15", "Already Used/Excl.": "#f87171"}
    top_idx   = set(chart_df.nlargest(min(8, len(chart_df)), y_col).index)

    def _last_first(nm):
        pts = str(nm).split(",")
        return f"{pts[0].strip()}, {pts[1].strip()[0]}." if len(pts) == 2 else nm

    _hover = (
        "<b>%{customdata[0]}</b><br>"
        + (
            "DG w/ Fit: %{y:.1f}%<br>"
            "DG Model: %{customdata[1]:.1f}%<br>"
            "Win % (odds): %{customdata[2]:.1f}%  -  %{customdata[5]}<br>"
            if odds_available else
            "League burned: %{y:.0f}%<br>"
        )
        + "% Burned (ahead): %{x:.0f}%  -  Avg rank: %{customdata[3]}<br>"
          "% Burned (jump): %{customdata[4]:.0f}%"
          "<extra></extra>"
    )
    cd_cols = ["Player", "DG Model %", "Win % (odds)", "Avg Rank (users)", "% Burned (jump)", "Best Odds"]

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
                x=df_part["% Burned (ahead)"], y=df_part[y_col],
                mode="markers+text" if show_text else "markers",
                name=status if not show_text else None,
                marker=dict(
                    size=10 if show_text else 7, color=color,
                    opacity=0.95 if show_text else 0.5,
                    line=dict(
                        color="white" if show_text else "rgba(255,255,255,0.1)",
                        width=1.5 if show_text else 0.5,
                    ),
                ),
                text=df_part["Player"].apply(_last_first) if show_text else None,
                textposition="top center",
                textfont=dict(size=8, color="rgba(255,255,255,0.85)"),
                customdata=df_part[cd_cols].values,
                hovertemplate=_hover,
                showlegend=not show_text,
            ))

    if not chart_df.empty:
        med_x = float(chart_df["% Burned (ahead)"].median())
        med_y = float(chart_df[y_col].median())
        fig.add_vline(x=med_x, line_dash="dot", line_color="rgba(128,128,128,0.3)", line_width=1)
        fig.add_hline(y=med_y, line_dash="dot", line_color="rgba(128,128,128,0.3)", line_width=1)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        xaxis=dict(title="% of Teams Ahead - Burned (can't reuse)", gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=40, l=50, r=20),
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -- Field table ───────────────────────────────────────────────────────────
    st.markdown("### Field Table")

    fc1, fc2, fc3 = st.columns(3)
    show_avail = fc1.checkbox("Available to us only", value=False, key="solo_gt_avail")
    min_burned = fc2.slider("Min % burned (ahead)", 0, 100, 0, 5, key="solo_gt_min_used")
    min_model  = fc3.slider("Min DG w/ Fit %", 0.0, 25.0, 0.0, 0.5, key="solo_gt_min_model",
                            disabled=not odds_available)

    disp = gt_df.copy()
    if show_avail:
        disp = disp[disp["We Can Use"]]
    if min_burned > 0:
        disp = disp[disp["% Burned (ahead)"] >= min_burned]
    if min_model > 0 and odds_available:
        disp = disp[disp["DG w/ Fit"].notna() & (disp["DG w/ Fit"] >= min_model)]

    show_cols = [
        "Player", "Tier",
        "DG w/ Fit", "DG Model %", "Win % (odds)", "Best Odds",
        "% Burned (ahead)", "% Burned (jump)", "% Burned (chasers)", "% Burned (league)",
        "Picking Now (ahead)", "Own % this week",
        "Avg Rank (users)", "New Rank",
        "We Can Use", "We Used",
    ]
    if not odds_available:
        show_cols = [c for c in show_cols if c not in ("DG w/ Fit", "DG Model %", "Win % (odds)", "Best Odds")]
    if not have_this_week:
        show_cols = [c for c in show_cols if c not in ("Picking Now (ahead)", "New Rank")]
    if n_chasers == 0:
        show_cols = [c for c in show_cols if c != "% Burned (chasers)"]

    fmt = {
        "DG w/ Fit":           "{:.1f}%",
        "DG Model %":          "{:.1f}%",
        "Win % (odds)":        "{:.1f}%",
        "% Burned (ahead)":    "{:.0f}%",
        "% Burned (jump)":     "{:.0f}%",
        "% Burned (chasers)":  "{:.0f}%",
        "% Burned (league)":   "{:.0f}%",
        "Picking Now (ahead)": "{:.0f}%",
        "Own % this week":     "{:.0f}%",
        "Avg Rank (users)":    "{:.1f}",
        "New Rank":            "{:.0f}",
    }
    styled = (
        disp[[c for c in show_cols if c in disp.columns]].style
        .applymap(_style_avail, subset=["We Can Use", "We Used"])
        .applymap(_tier_style, subset=["Tier"])
        .format({k: v for k, v in fmt.items() if k in disp.columns}, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    st.divider()

    # -- Recommendation ────────────────────────────────────────────────────────
    st.markdown("### Recommendation - Available to Us")
    if odds_available:
        st.caption("Score = DG w/ Fit % x (1 + % Burned (ahead) / 100)")
        score_col = "DG w/ Fit"
    else:
        st.caption("Score = % Burned (league) x (1 + % Burned (ahead) / 100)  -  odds not yet available")
        score_col = "% Burned (league)"

    avail = gt_df[gt_df["We Can Use"] & gt_df[score_col].notna()].copy()
    avail["Score"] = avail[score_col] * (1 + avail["% Burned (ahead)"] / 100)
    avail = avail.sort_values("Score", ascending=False).head(15)

    if not avail.empty:
        rec_cols = [
            "Player", "Tier",
            "DG w/ Fit", "DG Model %", "Win % (odds)", "Best Odds",
            "% Burned (ahead)", "% Burned (jump)", "% Burned (chasers)", "% Burned (league)",
            "Picking Now (ahead)", "Own % this week",
            "Avg Rank (users)", "New Rank", "Score",
        ]
        if not odds_available:
            rec_cols = [c for c in rec_cols if c not in ("DG w/ Fit", "DG Model %", "Win % (odds)", "Best Odds")]
        if not have_this_week:
            rec_cols = [c for c in rec_cols if c not in ("Picking Now (ahead)", "New Rank")]
        if n_chasers == 0:
            rec_cols = [c for c in rec_cols if c != "% Burned (chasers)"]

        fmt_rec = {**fmt, "Score": "{:.2f}"}
        styled_rec = (
            avail[[c for c in rec_cols if c in avail.columns]].style
            .applymap(_tier_style, subset=["Tier"])
            .format({k: v for k, v in fmt_rec.items() if k in avail.columns}, na_rep="-")
        )
        st.dataframe(styled_rec, use_container_width=True, hide_index=True)
    else:
        st.info("No available players found.")

    st.divider()

    # -- Teams ahead: best available ───────────────────────────────────────────
    st.markdown("### Teams Ahead - Best Available Picks")
    sort_key = "DG w/ Fit" if odds_available else "% Burned (league)"
    st.caption(f"Top 5 players each team ahead can still use, ranked by {sort_key}.")

    player_info = gt_df.set_index("dg_id")[["Player", sort_key, "Tier"]].to_dict("index")

    team_rows = []
    for _, team_row in teams_ahead.sort_values("PLACE").iterrows():
        uname     = team_row["_username"]
        used_ids  = team_used.get(uname, set())
        avail_ids = sorted(
            [d for d in field_ids if d not in used_ids],
            key=lambda d: player_info.get(d, {}).get(sort_key) or 0,
            reverse=True,
        )[:5]
        picks = []
        for did in avail_ids:
            info = player_info.get(did, {})
            last = info.get("Player", str(did)).split(",")[0].strip()
            val  = info.get(sort_key, np.nan)
            picks.append(f"{last} ({val:.1f}%)" if pd.notna(val) else last)
        this_pick_did  = this_week_picks.get(uname)
        this_pick_name = field_name_map.get(this_pick_did, "").split(",")[0].strip() if this_pick_did else "-"
        team_rows.append({
            "Place":       int(team_row["PLACE"]),
            "Team":        team_row["PLAYER"],
            "Earnings":    f"${int(team_row['EARNINGS']):,.0f}",
            "Picking Now": this_pick_name,
            "Pick 1":      picks[0] if len(picks) > 0 else "-",
            "Pick 2":      picks[1] if len(picks) > 1 else "-",
            "Pick 3":      picks[2] if len(picks) > 2 else "-",
            "Pick 4":      picks[3] if len(picks) > 3 else "-",
            "Pick 5":      picks[4] if len(picks) > 4 else "-",
        })

    if not have_this_week:
        team_rows_df = pd.DataFrame(team_rows).drop(columns=["Picking Now"], errors="ignore")
    else:
        team_rows_df = pd.DataFrame(team_rows)

    st.dataframe(team_rows_df, use_container_width=True, hide_index=True,
                 height=min(50 + len(team_rows) * 35, 600))

    st.divider()

    # -- Team inspector ────────────────────────────────────────────────────────
    with st.expander("Inspect a competitor's pick history", expanded=False):
        sel_team  = st.selectbox("Select team", options=teams_ahead["PLAYER"].tolist(), key="solo_team_select")
        match_row = lb[lb["PLAYER"] == sel_team]
        sel_uname = match_row["_username"].iloc[0] if not match_row.empty else None
        if sel_uname:
            team_hist = df[df["_username"] == sel_uname].copy()
            team_hist = team_hist[team_hist["_slug"].isin(prior_slugs)].sort_values("eventName")
            team_hist["In Field"] = team_hist["dg_id"].apply(
                lambda d: "YES" if pd.notna(d) and int(d) in field_ids else ""
            )
            team_hist["Earnings"] = team_hist["_winnings"].apply(
                lambda x: f"${int(x):,}" if x != 0 else "$0"
            )
            st.dataframe(
                team_hist[["eventName", "selection", "In Field", "Earnings"]].rename(
                    columns={"eventName": "Tournament", "selection": "Player"}
                ),
                use_container_width=True, hide_index=True,
            )
            used_ids_team  = team_used.get(sel_uname, set())
            avail_for_team = field_ids - used_ids_team
            col1, col2 = st.columns(2)
            col1.metric("Players burned (prior events)", len(used_ids_team))
            col2.metric("Still available this event", len(avail_for_team))

    st.divider()

    # -- Full leaderboard ──────────────────────────────────────────────────────
    with st.expander("Full Leaderboard", expanded=True):
        lb_display = lb[["PLACE", "PLAYER", "EARNINGS"]].copy()
        lb_display["EARNINGS"] = lb_display["EARNINGS"].apply(lambda x: f"${x:,}")

        def _highlight_us(row):
            if row["PLAYER"].strip().lower() == OUR_USERNAME:
                return ["background-color:#1a3060"] * len(row)
            return [""] * len(row)

        st.dataframe(
            lb_display.style.apply(_highlight_us, axis=1),
            use_container_width=True,
            hide_index=True,
            height=min(50 + len(lb_display) * 35, 1800),
        )
