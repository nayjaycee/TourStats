"""
refresh_data.py — command-line data refresh script

Usage:
    python Scripts/refresh_data.py --blocks 1,2,3   # Monday full refresh
    python Scripts/refresh_data.py --blocks 1        # Mid-week odds/field only
    python Scripts/refresh_data.py --blocks 4        # Live odds only (during tournament)

Blocks:
    1 = Field + Odds -> Fields.xlsx, this_week_field.csv, All_players.xlsx
    2 = Rounds -> combined_rounds_all_2017_2026.csv, Finishes.csv
    3 = Approach skill -> approach_skill_all_periods.csv
    4 = Live odds + pre-tournament model -> this_week_live_odds.csv

Blocks:
    5 = Live SG stats -> Data/live/live_latest.csv  (run frequently during rounds)

Cron schedule (add via: crontab -e):
    # Monday full refresh at 8am, 12pm, 4pm ET
    0 8,12,16 * * 1 cd /Users/joshmacbook/python_projects/GolfStats && .venv/bin/python Scripts/refresh_data.py --blocks 1,2,3 >> /tmp/golfstats_refresh.log 2>&1

    # Tue/Wed field+odds refresh at 9am and 3pm ET
    0 9,15 * * 2,3 cd /Users/joshmacbook/python_projects/GolfStats && .venv/bin/python Scripts/refresh_data.py --blocks 1 >> /tmp/golfstats_refresh.log 2>&1

    # Thu-Sun: next-week field check twice a day (tracks withdrawals/additions)
    0 9,17 * * 4,5,6,0 cd /Users/joshmacbook/python_projects/GolfStats && .venv/bin/python Scripts/refresh_data.py --blocks 1 >> /tmp/golfstats_refresh.log 2>&1

    # Live odds every 30 min Thu-Sun during tournament hours (7am-8pm ET)
    */30 7-20 * * 4,5,6,0 cd /Users/joshmacbook/python_projects/GolfStats && .venv/bin/python Scripts/refresh_data.py --blocks 4 >> /tmp/golfstats_refresh.log 2>&1

    # Live SG stats every 5 min Thu-Sun during tournament hours (7am-8pm ET)
    */5 7-20 * * 4,5,6,0 cd /Users/joshmacbook/python_projects/GolfStats && .venv/bin/python Scripts/refresh_data.py --blocks 5 >> /tmp/golfstats_live.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_secret

DATA_ROOT   = PROJECT_ROOT / "Data"
RAW_DIR     = DATA_ROOT / "Raw"
CLEAN_DIR   = DATA_ROOT / "Clean" / "Combined"
INUSE_DIR   = DATA_ROOT / "in Use"

YEAR                = 2026
TOUR_KEYS           = ["PGA", "EURO", "LIV"]
EXCLUDED_EVENT_IDS  = {18}

DG_ROUNDS_URL = "https://feeds.datagolf.com/historical-raw-data/rounds"
DG_FIELD_URL  = "https://feeds.datagolf.com/field-updates"
DG_ODDS_URL   = "https://feeds.datagolf.com/betting-tools/outrights"
DG_PREDS_URL  = "https://feeds.datagolf.com/preds/pre-tournament"
DG_APPROACH_URL = "https://feeds.datagolf.com/preds/approach-skill"

DG_API_KEY = get_secret("DATAGOLF_API_KEY")
if not DG_API_KEY:
    raise RuntimeError("Missing DATAGOLF_API_KEY")

TOURS = {
    "PGA":  {"folder": "PGA",  "prefix": "PGA",  "api_tour": "PGA"},
    "EURO": {"folder": "EURO", "prefix": "EURO", "api_tour": "euro"},
    "LIV":  {"folder": "LIV",  "prefix": "liv",  "api_tour": "liv"},
}

TIMEOUT  = 60
SESSION  = requests.Session()
ET       = pytz.timezone("America/New_York")

CLEAN_YEAR_PATH     = CLEAN_DIR / f"combined_rounds_{YEAR}.csv"
INUSE_ALL_PATH      = INUSE_DIR / "combined_rounds_all_2017_2026.csv"
FINISHES_PATH       = INUSE_DIR / "Finishes.csv"
THIS_WEEK_FIELD_CSV  = INUSE_DIR / "this_week_field.csv"
NEXT_WEEK_FIELD_CSV  = INUSE_DIR / "next_week_field.csv"
THIS_WEEK_ODDS_CSV   = INUSE_DIR / "this_week_odds.csv"
LIVE_ODDS_PATH       = INUSE_DIR / "this_week_live_odds.csv"
APPROACH_PATH        = INUSE_DIR / "approach_skill_all_periods.csv"
FIELDS_XLSX          = INUSE_DIR / "Fields.xlsx"
ALL_PLAYERS_XLSX     = INUSE_DIR / "All_players.xlsx"
SCHED_XLSX           = INUSE_DIR / "OAD_2026_Schedule.xlsx"
CHANGELOG_PATH       = INUSE_DIR / "data_changelog.json"
FIELD_STATUS_PATH    = INUSE_DIR / "field_status.json"
LIVE_DIR             = DATA_ROOT / "live"
LIVE_LATEST_PATH     = LIVE_DIR / "live_latest.csv"

# ── Utilities ─────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}", flush=True)

def coerce_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def write_excel(path: Path, df: pd.DataFrame, sheet_name: str = "Sheet1") -> None:
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
        df.to_excel(w, index=False, sheet_name=sheet_name)

def pull_df(url: str, params: Dict[str, Any], *, save_path: Optional[Path] = None) -> pd.DataFrame:
    resp = SESSION.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if save_path:
        df.to_csv(save_path, index=False)
        _log(f"Saved -> {save_path.name} | rows: {len(df):,}")
    return df

def write_changelog(entry: dict) -> None:
    log = json.loads(CHANGELOG_PATH.read_text()) if CHANGELOG_PATH.exists() else {"events": []}
    log["events"] = ([entry] + log["events"])[:20]
    CHANGELOG_PATH.write_text(json.dumps(log, indent=2))

def git_push(paths: List[Path], message: str) -> None:
    """Stage specific files, commit, and push to origin/main.
    Requires git to be configured with push credentials (SSH key or credential helper)."""
    try:
        for p in paths:
            rel = Path(p).relative_to(PROJECT_ROOT) if Path(p).is_absolute() else Path(p)
            subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", str(rel)],
                           check=True, capture_output=True)
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "diff", "--cached", "--quiet"],
            capture_output=True)
        if result.returncode == 0:
            _log("git_push: nothing staged, skipping commit")
            return
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "commit", "-m", message],
            check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "push", "origin", "main"],
            check=True, capture_output=True)
        _log(f"git_push: pushed — {message}")
    except subprocess.CalledProcessError as e:
        _log(f"git_push failed: {e.stderr.decode() if e.stderr else e}")


def _field_diff(old_names: list, new_names: list):
    old_set, new_set = set(old_names), set(new_names)
    return sorted(new_set - old_set), sorted(old_set - new_set)

def _update_field_status(key: str, event_id: int, event_name: str,
                         player_names: list, now_et: pd.Timestamp) -> None:
    fs = json.loads(FIELD_STATUS_PATH.read_text()) if FIELD_STATUS_PATH.exists() else {}
    prev = fs.get(key, {})
    old_names = prev.get("player_names", [])
    added, withdrawn = _field_diff(old_names, player_names)
    ts = now_et.strftime("%Y-%m-%d %H:%M:%S ET")
    if added or withdrawn:
        note = f"+{len(added)} / -{len(withdrawn)}"
    else:
        note = "No changes" if old_names else "Initial pull"
    change = {"timestamp": ts, "added": added, "withdrawn": withdrawn, "note": note}
    prev_changes = prev.get("recent_changes", [])
    fs[key] = {
        "event_id":       event_id,
        "event_name":     event_name,
        "player_count":   len(player_names),
        "player_names":   player_names,
        "last_updated":   ts,
        "recent_changes": ([change] + prev_changes)[:10],
    }
    FIELD_STATUS_PATH.write_text(json.dumps(fs, indent=2))
    _log(f"field_status.json [{key}] updated | +{len(added)} / -{len(withdrawn)}")

def _schedule_start_et(sched_df: pd.DataFrame, event_id: int) -> pd.Timestamp:
    row = sched_df.loc[sched_df["event_id"] == event_id]
    if row.empty:
        raise RuntimeError(f"No schedule row for event_id={event_id}")
    start = pd.to_datetime(row["start_date"].iloc[0], errors="coerce")
    if pd.isna(start):
        raise RuntimeError(f"NaT start_date for event_id={event_id}")
    return pd.Timestamp(start.date()).tz_localize(ET)

NON_NUMERIC_FINISH = {"CUT", "WD", "DQ", "MDF", "NAN"}

def clean_finish(val):
    v = str(val).upper().strip()
    if v in NON_NUMERIC_FINISH:
        return v
    if v.startswith("T") and v[1:].isdigit():
        return int(v[1:])
    if v.isdigit():
        return int(v)
    return v

LIV_EVENT_ID_MAP = {k.strip().lower(): v for k, v in {
    "adelaide": 1012, "andalucia": 1024, "bangkok": 1006, "bedminster": 1003,
    "boston": 1004, "chicago": 1005, "dallas": 1031,
    "dallas (team finalstroke play)": 1026, "dc": 1015, "greenbrier": 1017,
    "hong kong": 1020, "houston": 1022, "indianapolis": 1032, "jeddah": 1007,
    "korea": 1029, "las vegas": 1019, "london": 1001, "mayakoba": 1009,
    "mexico city": 1028, "miami": 1021, "miami (team finalstroke play)": 1008,
    "michigan (team finalstroke play)": 1033, "nashville": 1023, "orlando": 1011,
    "portland": 1002, "promotions event": 1018, "riyadh": 1027, "singapore": 1013,
    "tucson": 1010, "tulsa": 1014, "united kingdom": 1025, "valderrama": 1016,
    "virginia": 1030,
}.items()}

def patch_liv_ids(df: pd.DataFrame) -> pd.DataFrame:
    if not {"tour", "event_name"}.issubset(df.columns):
        return df
    out = df.copy()
    out["_tour"]  = out["tour"].astype(str).str.strip().str.upper()
    out["_ename"] = out["event_name"].astype(str).str.strip().str.lower()
    target = out["_tour"].eq("LIV") & out["_ename"].isin(LIV_EVENT_ID_MAP)
    if target.any():
        new_ids = out.loc[target, "_ename"].map(LIV_EVENT_ID_MAP).astype("Int64")
        for col in ["event_id", "course_num"]:
            if col in out.columns:
                out.loc[target, col] = new_ids
    return out.drop(columns=["_tour", "_ename"], errors="ignore")

def add_round_date(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"event_completed", "round_num", "tour", "event_name"}
    if not needed.issubset(df.columns):
        return df
    out = df.copy()
    out["_ec_dt"] = pd.to_datetime(out["event_completed"], errors="coerce")
    if out["_ec_dt"].isna().all():
        return df
    out["round_num"] = pd.to_numeric(out["round_num"], errors="coerce")
    max_round = out.groupby(["tour", "event_name", "_ec_dt"])["round_num"].transform("max")
    out["round_date"] = (
        out["_ec_dt"] - pd.to_timedelta(max_round - out["round_num"], unit="D")
    ).dt.date.astype(str)
    return out.drop(columns=["_ec_dt"], errors="ignore")

def add_round_positions(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["to_par", "cum_to_par", "round_position", "round_position_text"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.copy()
    df["to_par"] = df["round_score"] - df["course_par"]
    df = df.sort_values(["event_id", "season", "dg_id", "round_num"]).reset_index(drop=True)
    df["cum_to_par"] = df.groupby(["event_id", "season", "dg_id"])["to_par"].cumsum()

    missing_par  = df.groupby(["event_id", "season"])["course_par"].apply(lambda x: x.isna().all())
    missing_keys = missing_par[missing_par].index
    if len(missing_keys) > 0:
        mask = df.set_index(["event_id", "season"]).index.isin(missing_keys)
        df.loc[mask, "cum_to_par"] = (
            df[mask].groupby(["event_id", "season", "dg_id"])["round_score"].cumsum().values
        )

    def rank_group(g):
        valid = g["cum_to_par"].notna()
        pos   = pd.Series(np.nan, index=g.index)
        if valid.any():
            pos[valid] = g.loc[valid, "cum_to_par"].rank(method="min", ascending=True).astype(int)
        return pos

    df["round_position"] = (
        df.groupby(["event_id", "season", "round_num"], group_keys=False)
        .apply(lambda g: rank_group(g), include_groups=False)
    )
    df["round_position"] = pd.array(
        df["round_position"].where(df["round_position"].isna(), df["round_position"].astype("Int64")),
        dtype="Int64",
    )
    tie_counts = df.groupby(["event_id", "season", "round_num", "round_position"])["dg_id"].transform("count")
    df["round_position_text"] = df["round_position"].astype(str)
    df.loc[df["round_position"].isna(), "round_position_text"] = np.nan
    df.loc[tie_counts > 1, "round_position_text"] = "T" + df.loc[tie_counts > 1, "round_position"].astype(str)
    return df

def build_finishes(full_path: Path, out_path: Path, year: int = YEAR) -> None:
    df = pd.read_csv(full_path, low_memory=False)
    df["year"] = pd.to_numeric(df.get("season", df.get("year")), errors="coerce").astype("Int64")
    df = df[df["year"] == year].copy()
    df = coerce_int(df, ["event_id", "dg_id"])
    df["finish_num"] = pd.to_numeric(df.get("finish_num"), errors="coerce")
    sort_cols = [c for c in ["year", "event_id", "dg_id", "round_num", "round_date"] if c in df.columns]
    final = df.sort_values(sort_cols).groupby(["year", "event_id", "dg_id"], as_index=False).tail(1).copy()
    fn = final["finish_num"]
    final["made_cut"] = fn.notna().astype(int)
    final["CUT"]      = (1 - final["made_cut"]).astype(int)
    final["win"]      = ((fn == 1)  & fn.notna()).astype(int)
    final["top_5"]    = ((fn <= 5)  & fn.notna()).astype(int)
    final["top_10"]   = ((fn <= 10) & fn.notna()).astype(int)
    final["top_25"]   = ((fn <= 25) & fn.notna()).astype(int)
    if "finish_text" not in final.columns and "fin_text" in final.columns:
        final["finish_text"] = final["fin_text"]
    cols = ["year", "event_name", "event_id", "event_completed",
            "player_name", "dg_id", "finish_text", "finish_num",
            "win", "top_5", "top_10", "top_25", "made_cut", "CUT"]
    for c in cols:
        if c not in final.columns:
            final[c] = pd.NA
    final[cols].sort_values(["event_id", "finish_num", "player_name"], na_position="last").to_csv(out_path, index=False)
    _log(f"Finishes: {len(final):,} rows -> {out_path.name}")


# ── Blocks ────────────────────────────────────────────────────────────────────

def run_block1() -> None:
    """Field + Odds -> Fields.xlsx, this_week_field.csv, All_players.xlsx"""
    _log("=== BLOCK 1: Field + Odds ===")
    now_et = pd.Timestamp.now(tz=ET)

    # Pull field (don't save yet)
    field_raw = pull_df(DG_FIELD_URL, {"tour": "PGA", "file_format": "csv", "key": DG_API_KEY})
    field_raw = coerce_int(field_raw, ["event_id", "dg_id"])
    field = field_raw[["event_name", "event_id", "dg_id", "player_name", "owgr_rank"]].dropna(
        subset=["event_id", "dg_id", "player_name"]).copy()

    ids = sorted(field["event_id"].dropna().unique().tolist())
    if len(ids) == 0:
        raise RuntimeError("No event_id found in field feed")
    # DataGolf occasionally returns two events (transition period between weeks).
    # Use the one with the latest start_date as the "primary" field event.
    if len(ids) > 1:
        _log(f"Multiple event_ids in field feed: {ids} — selecting latest by start_date")
        sched_tmp = read_excel(SCHED_XLSX)
        sched_tmp["event_id"]   = pd.to_numeric(sched_tmp["event_id"], errors="coerce").astype("Int64")
        sched_tmp["start_date"] = pd.to_datetime(sched_tmp["start_date"], errors="coerce")
        id_to_start = dict(zip(sched_tmp["event_id"].dropna().astype(int), sched_tmp["start_date"]))
        ids_sorted  = sorted(ids, key=lambda x: id_to_start.get(int(x), pd.NaT) or pd.Timestamp.min)
        field_event_id = int(ids_sorted[-1])  # latest start_date = next week
        # Save secondary (older) event as this_week if it's still active
        other_id = int(ids_sorted[0])
        other_raw = field_raw[field_raw["event_id"] == other_id]
        if not other_raw.empty:
            other_raw.to_csv(THIS_WEEK_FIELD_CSV, index=False)
            _log(f"Saved this_week_field.csv (event_id={other_id}) | rows: {len(other_raw):,}")
        # Primary (next week) goes to next_week_field.csv
        field_raw = field_raw[field_raw["event_id"] == field_event_id]
        field     = field[field["event_id"] == field_event_id]
        field_raw.to_csv(NEXT_WEEK_FIELD_CSV, index=False)
        _log(f"Saved next_week_field.csv (event_id={field_event_id}) | rows: {len(field_raw):,}")
    else:
        field_event_id = int(ids[0])
    field_event_name = field[field["event_id"] == field_event_id]["event_name"].iloc[0]
    _log(f"Field: event_id={field_event_id} | {field_event_name} | players={len(field):,}")

    # Load schedule first so we know if the tournament has started before touching odds
    sched = read_excel(SCHED_XLSX)
    sched["event_id"]   = pd.to_numeric(sched["event_id"], errors="coerce").astype("Int64")
    sched["start_date"] = pd.to_datetime(sched["start_date"], errors="coerce")
    field_start = _schedule_start_et(sched, field_event_id)
    field_started = now_et >= field_start
    _log(f"Event start={field_start} | started={field_started}")

    # Pull odds — only save and use when the odds event matches the field event.
    # Before a tournament the endpoint may still return last week's event; once underway
    # it returns live odds for the active event. In both mismatch cases we preserve whatever
    # closing odds were already saved and don't corrupt Fields.xlsx with wrong-event data.
    odds_event_match = False
    try:
        odds_raw = pull_df(DG_ODDS_URL,
                           {"tour": "pga", "market": "win", "odds_format": "decimal",
                            "file_format": "csv", "key": DG_API_KEY})
        odds_event_name = ""
        if "event_name" in odds_raw.columns:
            odds_event_name = str(odds_raw["event_name"].dropna().iloc[0]) if not odds_raw["event_name"].dropna().empty else ""
        odds_event_match = (
            odds_event_name.strip().lower() == field_event_name.strip().lower()
        )
        if odds_event_match and not field_started:
            odds_raw.to_csv(THIS_WEEK_ODDS_CSV, index=False)
            _log(f"Saved -> this_week_odds.csv | rows: {len(odds_raw):,}")
        elif not odds_event_match:
            _log(f"Odds event mismatch: API returned '{odds_event_name}' but field is '{field_event_name}' — skipping odds save")
        else:
            _log("Tournament in progress — odds pulled but this_week_odds.csv NOT overwritten (preserving closing odds)")
    except Exception as e:
        if THIS_WEEK_ODDS_CSV.exists():
            _log(f"Odds pull failed ({e}), using cache")
            odds_raw = pd.read_csv(THIS_WEEK_ODDS_CSV)
            odds_event_match = True  # treat cached file as valid
        else:
            odds_raw = pd.DataFrame()

    odds_raw = coerce_int(odds_raw, ["event_id", "dg_id"])
    # Only build close_odds lookup when odds are for the correct event
    if odds_event_match and not field_started:
        close_odds_by_id = dict(zip(
            odds_raw["dg_id"].dropna().astype(int),
            pd.to_numeric(odds_raw.get("datagolf_base_history_fit"), errors="coerce")
        ))
    else:
        close_odds_by_id = {}  # odds not yet available or wrong event — leave NaN in Fields.xlsx

    # Active tournament window: Thu start through Mon 6am
    ACTIVE_WINDOW = pd.Timedelta(days=4, hours=6)

    # Determine if an *existing* this_week_field.csv belongs to a still-active tournament
    active_tournament_in_window = False
    if THIS_WEEK_FIELD_CSV.exists():
        try:
            ex = pd.read_csv(THIS_WEEK_FIELD_CSV)
            ex_ids = pd.to_numeric(ex.get("event_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique()
            if len(ex_ids) == 1:
                ex_eid = int(ex_ids[0])
                ex_row = sched.loc[sched["event_id"] == ex_eid]
                if not ex_row.empty and ex_eid != field_event_id:
                    ex_start = pd.to_datetime(ex_row["start_date"].iloc[0], errors="coerce")
                    ex_end   = pd.Timestamp(ex_start.date()).tz_localize(ET) + ACTIVE_WINDOW
                    if now_et <= ex_end:
                        active_tournament_in_window = True
                        _log(f"Active tournament (event_id={ex_eid}) still in window until {ex_end} — routing to next_week_field.csv")
        except Exception as e:
            _log(f"Could not check existing field ({e}) — saving as this_week")

    player_names = sorted(field["player_name"].dropna().tolist())

    if active_tournament_in_window:
        # Save as next week's field
        field_raw.to_csv(NEXT_WEEK_FIELD_CSV, index=False)
        _log(f"Saved next_week_field.csv | rows: {len(field_raw):,}")
        _update_field_status("next_week", field_event_id, field_event_name, player_names, now_et)
    else:
        # Save as this week's field
        field_raw.to_csv(THIS_WEEK_FIELD_CSV, index=False)
        _log(f"Saved this_week_field.csv | rows: {len(field_raw):,}")
        _update_field_status("this_week", field_event_id, field_event_name, player_names, now_et)

    # Update Fields.xlsx
    def _ensure_cols(df):
        for c in ["year", "event_name", "event_id", "event_completed",
                  "player_name", "dg_id", "close_odds", "field_pulled_at", "odds_pulled_at"]:
            if c not in df.columns:
                df[c] = pd.NA
        return coerce_int(df, ["year", "event_id", "dg_id"])

    fields = _ensure_cols(read_excel(FIELDS_XLSX))
    mask   = (fields["year"] == YEAR) & (fields["event_id"] == field_event_id)
    existing_odds = dict(zip(fields.loc[mask, "dg_id"].dropna().astype(int), fields.loc[mask, "close_odds"]))

    wk = field[["event_name", "event_id", "dg_id", "player_name"]].copy()
    wk["year"]            = YEAR
    wk["event_completed"] = pd.Timestamp(field_start.date())
    wk["field_pulled_at"] = now_et.strftime("%Y-%m-%d %H:%M:%S %Z")
    if field_started:
        wk["close_odds"]     = wk["dg_id"].astype(int).map(existing_odds)
        wk["odds_pulled_at"] = pd.NA
    else:
        wk["close_odds"]     = wk["dg_id"].astype(int).map(close_odds_by_id)
        wk["odds_pulled_at"] = now_et.strftime("%Y-%m-%d %H:%M:%S %Z")

    wk = wk[["year", "event_name", "event_id", "event_completed",
              "player_name", "dg_id", "close_odds", "field_pulled_at", "odds_pulled_at"]]
    fields_new = pd.concat([fields.loc[~mask], wk], ignore_index=True)
    fields_new["player_name"] = fields_new["player_name"].astype(str)
    fields_new["event_name"]  = fields_new["event_name"].astype(str)
    write_excel(FIELDS_XLSX, fields_new)
    _log(f"Fields.xlsx updated | total={len(fields_new):,}")

    # Update owgr in All_players.xlsx
    ap = read_excel(ALL_PLAYERS_XLSX)
    ap["dg_id"] = pd.to_numeric(ap["dg_id"], errors="coerce").astype("Int64")
    owgr_map = dict(zip(field["dg_id"].dropna().astype(int), field["owgr_rank"]))
    ap["owgr"] = ap.apply(
        lambda r: owgr_map.get(int(r["dg_id"]), r.get("owgr")) if pd.notna(r["dg_id"]) else r.get("owgr"), axis=1
    )
    write_excel(ALL_PLAYERS_XLSX, ap)
    _log(f"All_players.xlsx owgr updated for {len(owgr_map):,} players")

    write_changelog({
        "timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "type": "field_odds_refresh",
        "event_name": field_event_name,
        "details": f"Field ({len(field):,} players) and odds updated for {field_event_name}",
    })

    push_paths = [
        THIS_WEEK_FIELD_CSV, FIELDS_XLSX, ALL_PLAYERS_XLSX, FIELD_STATUS_PATH, CHANGELOG_PATH,
    ]
    if not field_started:
        push_paths.append(THIS_WEEK_ODDS_CSV)
    if active_tournament_in_window and NEXT_WEEK_FIELD_CSV.exists():
        push_paths.append(NEXT_WEEK_FIELD_CSV)
    git_push(
        [p for p in push_paths if Path(p).exists()],
        f"data: field/odds refresh {now_et.strftime('%Y-%m-%d %H:%M ET')}",
    )
    _log("Block 1 complete.")


def run_block2() -> None:
    """Rounds refresh -> combined_rounds_all_2017_2026.csv + Finishes.csv"""
    _log("=== BLOCK 2: Rounds Refresh ===")

    for tour_key in TOUR_KEYS:
        cfg      = TOURS[tour_key]
        out_path = (RAW_DIR / cfg["folder"]).mkdir(parents=True, exist_ok=True) or \
                   RAW_DIR / cfg["folder"] / f"{cfg['prefix']}_rounds_{YEAR}.csv"
        params   = {"tour": cfg["api_tour"], "event_id": "all", "year": YEAR,
                    "file_format": "csv", "key": DG_API_KEY}
        df = pull_df(DG_ROUNDS_URL, params, save_path=RAW_DIR / cfg["folder"] / f"{cfg['prefix']}_rounds_{YEAR}.csv")
        coerce_int(df, ["event_id", "dg_id", "year", "round_num"]).to_csv(
            RAW_DIR / cfg["folder"] / f"{cfg['prefix']}_rounds_{YEAR}.csv", index=False)

    dfs = []
    for tour_key in TOUR_KEYS:
        cfg   = TOURS[tour_key]
        fpath = RAW_DIR / cfg["folder"] / f"{cfg['prefix']}_rounds_{YEAR}.csv"
        df    = pd.read_csv(fpath)
        df["tour"] = tour_key
        df["year"] = YEAR
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = coerce_int(combined, ["event_id", "dg_id", "year", "round_num"])

    if "fin_text" in combined.columns:
        combined["fin_text"]   = combined["fin_text"].astype(str).str.upper().str.strip()
        combined["finish_num"] = combined["fin_text"].apply(clean_finish)

    combined = patch_liv_ids(combined)
    combined = combined[~combined["event_id"].isin(EXCLUDED_EVENT_IDS)].copy()
    combined = add_round_date(combined)

    _log("Computing round positions...")
    combined = add_round_positions(combined)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CLEAN_YEAR_PATH, index=False)
    _log(f"Clean year file: {len(combined):,} rows (with positions)")

    build_finishes(CLEAN_YEAR_PATH, FINISHES_PATH)

    write_changelog({
        "timestamp": pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d %H:%M:%S ET"),
        "type": "rounds_refresh",
        "event_name": "",
        "details": f"Rounds refreshed — {len(combined):,} rows across {combined['event_id'].nunique()} events. Finishes rebuilt.",
    })

    git_push(
        [CLEAN_YEAR_PATH, FINISHES_PATH, CHANGELOG_PATH],
        f"data: rounds refresh {pd.Timestamp.now(tz=ET).strftime('%Y-%m-%d %H:%M ET')}",
    )
    _log("Block 2 complete.")


def run_block3() -> None:
    """Approach skill refresh"""
    _log("=== BLOCK 3: Approach Skill ===")
    periods = ["l12", "l24", "ytd"]
    base_url = "https://feeds.datagolf.com/preds/approach-skill"
    frames = []
    for period in periods:
        try:
            resp    = SESSION.get(base_url, params={"period": period, "file_format": "json", "key": DG_API_KEY}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows    = payload if isinstance(payload, list) else (
                payload.get("players") or payload.get("data") or payload.get("results") or [])
            last_updated = None if isinstance(payload, list) else (
                payload.get("last_updated") or payload.get("updated_at"))
            df = pd.DataFrame(rows)
            df["time_period"]  = period
            df["last_updated"] = last_updated
            df["fetched_at"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            frames.append(df)
            _log(f"Approach {period}: {len(df)} rows")
        except Exception as e:
            _log(f"Approach {period} failed: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        meta = ["time_period", "last_updated", "fetched_at"]
        combined = combined[meta + [c for c in combined.columns if c not in meta]]
        combined.to_csv(APPROACH_PATH, index=False)
        _log(f"approach_skill_all_periods.csv: {len(combined):,} rows")

        write_changelog({
            "timestamp": pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d %H:%M:%S ET"),
            "type": "approach_refresh",
            "event_name": "",
            "details": f"Approach skill refreshed — {len(combined):,} rows (l12/l24/ytd)",
        })
        git_push(
            [APPROACH_PATH, CHANGELOG_PATH],
            f"data: approach skill refresh {pd.Timestamp.now(tz=ET).strftime('%Y-%m-%d %H:%M ET')}",
        )
    _log("Block 3 complete.")


def run_block4() -> None:
    """Live odds + pre-tournament model -> this_week_live_odds.csv"""
    _log("=== BLOCK 4: Live Odds ===")
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        resp = SESSION.get(
            "https://feeds.datagolf.com/betting-tools/outrights",
            params={"tour": "pga", "market": "win", "odds_format": "decimal",
                    "file_format": "json", "key": DG_API_KEY},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        payload      = resp.json()
        odds_rows    = payload.get("odds", [])
        odds_df      = pd.DataFrame(odds_rows)
        odds_updated = payload.get("last_updated", "")
        _log(f"Outrights: {len(odds_df)} players | updated: {odds_updated}")
    except Exception as e:
        if LIVE_ODDS_PATH.exists():
            _log(f"Outrights pull failed ({e}), using cache")
            odds_df      = pd.read_csv(LIVE_ODDS_PATH)
            odds_updated = ""
        else:
            raise

    dg_base_col = next(
        (c for c in odds_df.columns if "datagolf" in c.lower() and "history" in c.lower()),
        next((c for c in odds_df.columns if c.lower().startswith("datagolf")), None)
    )
    book_cols = [c for c in odds_df.columns
                 if c not in ("dg_id", "player_name") and not c.lower().startswith("datagolf")]
    if book_cols:
        nb = odds_df[book_cols].apply(pd.to_numeric, errors="coerce")
        odds_df["best_book_odds"] = nb.max(axis=1)
        odds_df["best_book"]      = nb.idxmax(axis=1).where(nb.notna().any(axis=1))
    else:
        odds_df["best_book_odds"] = pd.NA
        odds_df["best_book"]      = pd.NA

    keep = ["dg_id", "player_name", "best_book_odds", "best_book"]
    if dg_base_col:
        keep.insert(2, dg_base_col)
    odds_clean = odds_df[keep].copy()
    if dg_base_col:
        odds_clean.rename(columns={dg_base_col: "baseline_odds"}, inplace=True)
    else:
        odds_clean["baseline_odds"] = pd.NA
    odds_clean["last_updated_odds"] = odds_updated
    odds_clean["dg_id"] = pd.to_numeric(odds_clean["dg_id"], errors="coerce").astype("Int64")

    try:
        resp = SESSION.get(DG_PREDS_URL,
                           params={"tour": "pga", "file_format": "json", "key": DG_API_KEY},
                           timeout=TIMEOUT)
        resp.raise_for_status()
        payload      = resp.json()
        preds_df     = pd.DataFrame(payload.get("players", []))
        preds_updated = payload.get("last_updated", "")
        _log(f"Pre-tournament: {len(preds_df)} players | updated: {preds_updated}")
    except Exception as e:
        _log(f"Pre-tournament pull failed: {e}")
        preds_df      = pd.DataFrame()
        preds_updated = ""

    if not preds_df.empty:
        preds_df["dg_id"] = pd.to_numeric(preds_df.get("dg_id", pd.NA), errors="coerce").astype("Int64")
        preds_df.rename(columns={"win": "dg_win_prob", "proj_total": "proj_score",
                                  "datagolf_rank": "model_rank"}, inplace=True)
        keep_p = ["dg_id"] + [c for c in ["dg_win_prob", "make_cut", "top_5", "top_10", "proj_score", "model_rank"]
                               if c in preds_df.columns]
        preds_clean = preds_df[keep_p].copy()
        preds_clean["last_updated_preds"] = preds_updated
    else:
        preds_clean = pd.DataFrame(columns=["dg_id", "last_updated_preds"])

    merged = odds_clean.merge(preds_clean, on="dg_id", how="left")
    merged["fetched_at"] = fetched_at
    col_order = ["player_name", "dg_id", "dg_win_prob", "model_rank", "proj_score",
                 "baseline_odds", "best_book_odds", "best_book", "make_cut", "top_5", "top_10",
                 "last_updated_odds", "last_updated_preds", "fetched_at"]
    merged = merged[[c for c in col_order if c in merged.columns]]
    merged.to_csv(LIVE_ODDS_PATH, index=False)
    _log(f"this_week_live_odds.csv: {len(merged):,} rows")

    write_changelog({
        "timestamp": pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d %H:%M:%S ET"),
        "type": "live_odds_refresh",
        "event_name": "",
        "details": f"Live odds refreshed — {len(merged):,} players",
    })
    _log("Block 4 complete.")


def run_block5() -> None:
    """Live SG stats -> Data/live/live_latest.csv (overwritten every run)"""
    _log("=== BLOCK 5: Live SG Stats ===")

    ALL_STATS = (
        "sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_total,"
        "driving_dist,driving_acc,gir,scrambling,prox_fw,prox_rgh"
    )

    resp = SESSION.get(
        "https://feeds.datagolf.com/preds/live-tournament-stats",
        params={"stats": ALL_STATS, "round": "event_avg", "display": "value",
                "file_format": "csv", "key": DG_API_KEY},
        timeout=20,
    )
    resp.raise_for_status()

    from io import StringIO as _StringIO
    import numpy as _np
    df = pd.read_csv(_StringIO(resp.text))
    df.columns = [c.lower().strip() for c in df.columns]

    # Forward-fill sparse columns
    for col in ["event_name", "last_updated", "stat_display", "course"]:
        if col in df.columns:
            df[col] = df[col].replace("", _np.nan).ffill()

    num_cols = ["dg_id", "total", "round", "thru",
                "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total",
                "driving_dist", "driving_acc", "gir", "scrambling", "prox_fw", "prox_rgh"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LIVE_LATEST_PATH, index=False)

    event_name = df["event_name"].dropna().iloc[0] if "event_name" in df.columns and not df["event_name"].dropna().empty else ""
    last_updated = df["last_updated"].dropna().iloc[0] if "last_updated" in df.columns and not df["last_updated"].dropna().empty else ""
    _log(f"live_latest.csv: {len(df):,} players | {event_name} | updated: {last_updated}")
    _log("Block 5 complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GolfStats data refresh")
    parser.add_argument("--blocks", default="1,2,3,4",
                        help="Comma-separated block numbers to run (default: 1,2,3,4)")
    args    = parser.parse_args()
    blocks  = [int(b.strip()) for b in args.blocks.split(",")]

    _log(f"Running blocks: {blocks}")
    if 1 in blocks:
        run_block1()
    if 2 in blocks:
        run_block2()
    if 3 in blocks:
        run_block3()
    if 4 in blocks:
        run_block4()
    if 5 in blocks:
        run_block5()
    _log("All done.")
