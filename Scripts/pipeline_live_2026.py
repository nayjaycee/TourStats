# Scripts/pipeline_live_2026.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "Data" / "Raw"
DATA_CLEAN_COMBINED = PROJECT_ROOT / "Data" / "Clean" / "Combined"
DATA_IN_USE = PROJECT_ROOT / "Data" / "in Use"

DATA_CLEAN_COMBINED.mkdir(parents=True, exist_ok=True)

SEASON = 2026

# --- Inputs ---
FROZEN_ALL_YEARS_PATH = DATA_IN_USE / "combined_rounds_all_2017_2026.csv"  # frozen baseline

# --- Outputs ---
OUT_2026_CLEAN = DATA_CLEAN_COMBINED / f"combined_rounds_{SEASON}.csv"
OUT_ALL_YEARS = DATA_IN_USE / f"combined_rounds_all_2017_{SEASON}.csv"

# --- Run metadata (for guardrails + debugging) ---
RUN_META_PATH = DATA_CLEAN_COMBINED / f"run_meta_rounds_{SEASON}.json"

# DataGolf endpoint
ROUNDS_URL = "https://feeds.datagolf.com/historical-raw-data/rounds"

# Tours to pull
TOURS: Dict[str, Dict[str, str]] = {
    "PGA":  {"folder": "PGA",  "prefix": "PGA"},
    "EURO": {"folder": "EURO", "prefix": "EURO"},
    "LIV":  {"folder": "LIV",  "prefix": "liv"},
}

# Contract columns (what your app expects / what you posted)
CONTRACT_COLS = [
    "tour","year","season","event_completed","event_name","event_id",
    "player_name","dg_id","fin_text","round_num",
    "course_name","course_num","course_par","start_hole","teetime","round_score",
    "sg_putt","sg_arg","sg_app","sg_ott","sg_t2g","sg_total",
    "driving_dist","driving_acc","gir","scrambling","prox_rgh","prox_fw",
    "great_shots","poor_shots",
    "eagles_or_better","birdies","pars","bogies","doubles_or_worse",
    "finish_num","round_date",
]

# Non-numeric finish statuses you already use
NON_NUMERIC_FINISH = {"CUT", "WD", "DQ", "MDF", "NAN"}

# LIV mapping (your map)
LIV_EVENT_ID_MAP_RAW = {
    "adelaide":                         1012,
    "andalucia":                        1024,
    "bangkok":                          1006,
    "bedminster":                       1003,
    "boston":                           1004,
    "chicago":                          1005,
    "dallas":                           1031,
    "dallas (team finalstroke play)":   1026,
    "dc":                               1015,
    "greenbrier":                       1017,
    "hong kong":                        1020,
    "houston":                          1022,
    "indianapolis":                     1032,
    "jeddah":                           1007,
    "korea":                            1029,
    "las vegas":                        1019,
    "london":                           1001,
    "mayakoba":                         1009,
    "mexico city":                      1028,
    "miami":                            1021,
    "miami (team finalstroke play)":    1008,
    "michigan (team finalstroke play)": 1033,
    "nashville":                        1023,
    "orlando":                          1011,
    "portland":                         1002,
    "promotions event":                 1018,
    "riyadh":                           1027,
    "singapore":                        1013,
    "tucson":                           1010,
    "tulsa":                            1014,
    "united kingdom":                   1025,
    "valderrama":                       1016,
    "virginia":                         1030,
}
LIV_EVENT_ID_MAP = {k.strip().lower(): v for k, v in LIV_EVENT_ID_MAP_RAW.items()}


# ============================================================
# UTIL
# ============================================================

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def _read_run_meta() -> dict:
    if RUN_META_PATH.exists():
        try:
            return json.loads(RUN_META_PATH.read_text())
        except Exception:
            return {}
    return {}


def _write_run_meta(meta: dict) -> None:
    RUN_META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))


def clean_finish(val) -> object:
    v = str(val).upper().strip()
    if v in NON_NUMERIC_FINISH:
        return v
    if v.startswith("T") and v[1:].isdigit():
        return int(v[1:])
    if v.isdigit():
        return int(v)
    return v


def _ensure_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CONTRACT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    # keep extra cols if DataGolf adds them? -> NO: we enforce contract for stability
    out = out[CONTRACT_COLS].copy()
    return out


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # minimal cleanup only (no dtype "fixing")
    for c in ["tour", "event_name", "player_name", "course_name"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
            out.loc[out[c].str.lower().isin({"nan", "none", "null"}), c] = ""
    if "tour" in out.columns:
        out["tour"] = out["tour"].astype(str).str.upper().str.strip()
    return out


def _apply_liv_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    tour_clean = out.get("tour", pd.Series(index=out.index, dtype=object)).astype(str).str.upper().str.strip()
    name_clean = out.get("event_name", pd.Series(index=out.index, dtype=object)).astype(str).str.strip().str.lower()

    liv_mask = tour_clean.eq("LIV")
    in_map = name_clean.isin(LIV_EVENT_ID_MAP.keys())
    target = liv_mask & in_map

    if target.any():
        new_ids = name_clean.loc[target].map(LIV_EVENT_ID_MAP).astype("Int64")
        out.loc[target, "event_id"] = new_ids
        out.loc[target, "course_num"] = new_ids

    return out


def _add_round_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    round_date derived from event_completed (end date) and round_num.
    Uses max(round_num) within (tour, event_name, event_completed) groups.
    """
    out = df.copy()
    if out.empty:
        return out

    if "event_completed" not in out.columns or "round_num" not in out.columns:
        return out

    # parse end date
    event_completed_dt = pd.to_datetime(out["event_completed"], errors="coerce")

    # round_num numeric for computing offsets (no global dtype fixes; just this calc)
    rn = pd.to_numeric(out["round_num"], errors="coerce")

    # group by fields that define an event instance
    gcols = []
    for c in ["tour", "event_name"]:
        if c in out.columns:
            gcols.append(c)
    gcols.append("_event_completed_dt")

    tmp = out.copy()
    tmp["_event_completed_dt"] = event_completed_dt
    tmp["_round_num_num"] = rn

    # if missing, can't do anything
    if tmp["_event_completed_dt"].isna().all() or tmp["_round_num_num"].isna().all():
        return out

    max_round = tmp.groupby(gcols)["_round_num_num"].transform("max")
    offset_days = max_round - tmp["_round_num_num"]

    round_date_dt = tmp["_event_completed_dt"] - pd.to_timedelta(offset_days, unit="D")
    out["round_date"] = pd.to_datetime(round_date_dt, errors="coerce").dt.date.astype(str)

    # clean "NaT" strings -> empty
    out.loc[out["round_date"].astype(str).str.lower().isin({"nat", "nan", "none"}), "round_date"] = ""

    return out


def _add_finish_num(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fin_text" not in out.columns:
        out["fin_text"] = pd.NA
    out["fin_text"] = out["fin_text"].astype(str).str.upper().str.strip()
    out["finish_num"] = out["fin_text"].apply(clean_finish)
    return out


# ============================================================
# PULL
# ============================================================

def pull_rounds_csv(
    api_key: str,
    tour: str,
    season: int,
    timeout: int = 90,
) -> pd.DataFrame:
    params = {
        "tour": tour.lower(),
        "event_id": "all",
        "season": int(season),
        "file_format": "csv",
        "key": api_key,
    }

    resp = requests.get(ROUNDS_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    text = resp.text
    df = pd.read_csv(StringIO(text), low_memory=False)
    return df


def raw_out_path(tour_name: str, season: int) -> Path:
    folder = DATA_RAW / TOURS[tour_name]["folder"] / "Rounds"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{TOURS[tour_name]['prefix']}_rounds_{season}.csv"


# ============================================================
# CLEAN
# ============================================================

def clean_rounds_df(df: pd.DataFrame, tour_name: str, season: int) -> pd.DataFrame:
    out = df.copy()

    # Ensure required identifiers exist (but don't hard-cast)
    if "tour" not in out.columns:
        out["tour"] = tour_name
    if "season" not in out.columns:
        out["season"] = int(season)
    if "year" not in out.columns:
        # your combined uses both year + season; for DataGolf rounds, year typically matches season
        out["year"] = int(season)

    out = _normalize_strings(out)

    # deterministic transforms
    out = _add_finish_num(out)

    # LIV patch before round_date so ids are correct (not required for round_date, but consistent)
    out = _apply_liv_mapping(out)

    out = _add_round_date(out)

    # enforce contract (stable for app)
    out = _ensure_contract_columns(out)

    # final tour/year/season stamps (keep consistent)
    out["tour"] = tour_name
    out["season"] = int(season)
    out["year"] = int(season)

    return out


# ============================================================
# GUARDRAILS
# ============================================================

@dataclass(frozen=True)
class Guardrails:
    min_total_rows: int = 50_000               # "something is very wrong" floor
    max_drop_frac_vs_prev: float = 0.30        # abort if rows drop >30% vs previous successful run
    require_each_tour_nonempty: bool = True


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in CONTRACT_COLS if c not in df.columns]
    if missing:
        return False, f"Missing contract columns: {missing}"
    extra = [c for c in df.columns if c not in CONTRACT_COLS]
    if extra:
        return False, f"Unexpected extra columns (schema drift): {extra}"
    return True, "ok"


def should_publish(
    combined_2026: pd.DataFrame,
    per_tour_rows: Dict[str, int],
    guard: Guardrails,
) -> Tuple[bool, str]:
    ok, msg = validate_schema(combined_2026)
    if not ok:
        return False, msg

    total_rows = int(len(combined_2026))
    if total_rows < guard.min_total_rows:
        return False, f"Total rows too small: {total_rows:,} < {guard.min_total_rows:,}"

    if guard.require_each_tour_nonempty:
        empty_tours = [t for t, n in per_tour_rows.items() if n <= 0]
        if empty_tours:
            return False, f"One or more tours returned 0 rows: {empty_tours}"

    # compare to previous successful run
    meta = _read_run_meta()
    prev_total = int(meta.get("last_success_total_rows", 0) or 0)
    if prev_total > 0:
        drop = (prev_total - total_rows) / float(prev_total)
        if drop > guard.max_drop_frac_vs_prev:
            return False, f"Row-count collapse vs prev: prev={prev_total:,}, now={total_rows:,}, drop={drop:.1%}"

    return True, "ok"


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_refresh_rounds_2026(api_key: str, guard: Optional[Guardrails] = None) -> None:
    guard = guard or Guardrails()

    run_started = _now_iso()
    per_tour_rows: Dict[str, int] = {}
    cleaned: Dict[str, pd.DataFrame] = {}

    print(f"[{run_started}] Starting full refresh for season={SEASON}")

    # 1) Pull + save raw + clean
    for tour_name in TOURS.keys():
        print(f"[INFO] Pulling {tour_name} rounds season={SEASON}")
        df_raw = pull_rounds_csv(api_key=api_key, tour=tour_name, season=SEASON)

        # Save raw snapshot
        raw_path = raw_out_path(tour_name, SEASON)
        _atomic_write_csv(df_raw, raw_path)
        print(f"[INFO] Saved raw: {raw_path}  | rows: {len(df_raw):,}")

        df_clean = clean_rounds_df(df_raw, tour_name=tour_name, season=SEASON)
        cleaned[tour_name] = df_clean
        per_tour_rows[tour_name] = int(len(df_clean))
        print(f"[INFO] Cleaned {tour_name}: rows={len(df_clean):,}")

    # 2) Combine
    combined_2026 = pd.concat([cleaned[t] for t in TOURS.keys()], ignore_index=True)
    combined_2026 = combined_2026[CONTRACT_COLS].copy()

    # 3) Guardrails before publishing
    ok, reason = should_publish(combined_2026, per_tour_rows, guard)
    if not ok:
        print(f"[ABORT] Not publishing. Reason: {reason}")
        meta = _read_run_meta()
        meta.update({
            "last_attempt_ts": run_started,
            "last_attempt_status": "aborted",
            "last_attempt_reason": reason,
            "last_attempt_per_tour_rows": per_tour_rows,
            "last_attempt_total_rows": int(len(combined_2026)),
        })
        _write_run_meta(meta)
        return

    # 4) Publish combined_rounds_2026.csv
    _atomic_write_csv(combined_2026, OUT_2026_CLEAN)
    print(f"[OK] Wrote 2026 clean: {OUT_2026_CLEAN} | rows: {len(combined_2026):,}")

    # 5) Build all-years file for app
    if not FROZEN_ALL_YEARS_PATH.exists():
        raise FileNotFoundError(f"Frozen baseline missing: {FROZEN_ALL_YEARS_PATH}")

    frozen = pd.read_csv(FROZEN_ALL_YEARS_PATH, low_memory=False)
    # enforce contract on frozen too (keeps app stable)
    frozen = _ensure_contract_columns(frozen)

    all_years = pd.concat([frozen, combined_2026], ignore_index=True)
    all_years = all_years[CONTRACT_COLS].copy()

    _atomic_write_csv(all_years, OUT_ALL_YEARS)
    print(f"[OK] Wrote all-years: {OUT_ALL_YEARS} | rows: {len(all_years):,}")

    # 6) Success meta
    meta = _read_run_meta()
    meta.update({
        "last_success_ts": run_started,
        "last_success_status": "success",
        "last_success_per_tour_rows": per_tour_rows,
        "last_success_total_rows": int(len(combined_2026)),
        "outputs": {
            "combined_2026": str(OUT_2026_CLEAN),
            "all_years": str(OUT_ALL_YEARS),
        }
    })
    _write_run_meta(meta)
    print(f"[DONE] Success. Meta: {RUN_META_PATH}")


# ============================================================
# CLI ENTRY
# ============================================================

def _get_api_key_from_env() -> Optional[str]:
    # preferred: export DATAGOLF_API_KEY="..."
    return os.getenv("DATAGOLF_API_KEY")


if __name__ == "__main__":
    key = _get_api_key_from_env()
    if not key:
        raise RuntimeError("Missing DATAGOLF_API_KEY env var. Set it, then rerun.")
    run_full_refresh_rounds_2026(api_key=key)
