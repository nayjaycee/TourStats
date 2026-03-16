#!/usr/bin/env python3
"""
refresh_live_odds.py
--------------------
Pulls DraftKings win odds from DataGolf and writes:
    /Users/joshmacbook/python_projects/GolfStats/Data/in Use/this_week_live_odds.csv

Schedule with cron (every 30 min, Thu-Sun during tournament weeks):
    crontab -e
    */30 * * * 4,5,6,0 /Users/joshmacbook/python_projects/GolfStats/Scripts/refresh_live_odds.py >> /tmp/refresh_live_odds.log 2>&1

Or run manually any time:
    python3 refresh_live_odds.py
"""

import ast
import sys
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ── Config ───────────────────────────────────────────────────────────────────
BOOK          = "draftkings"
INUSE_DIR     = Path("/Users/joshmacbook/python_projects/GolfStats/Data/in Use")
OUT_PATH      = INUSE_DIR / "this_week_live_odds.csv"
TIMEOUT       = 30

# API key — reads from environment or falls back to src.config
DG_API_KEY = os.environ.get("DATAGOLF_API_KEY", "")
if not DG_API_KEY:
    try:
        sys.path.insert(0, str(Path("/Users/joshmacbook/python_projects/GolfStats")))
        from src.config import get_secret
        DG_API_KEY = get_secret("DATAGOLF_API_KEY")
    except Exception as e:
        print(f"[ERROR] Could not load API key: {e}")
        sys.exit(1)

# ── Pull ─────────────────────────────────────────────────────────────────────
def run():
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{fetched_at}] Pulling DraftKings odds...")

    try:
        resp = requests.get(
            "https://feeds.datagolf.com/betting-tools/outrights",
            params={
                "tour":        "pga",
                "market":      "win",
                "odds_format": "decimal",
                "file_format": "json",
                "key":         DG_API_KEY,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}")
        sys.exit(1)

    odds_rows    = payload.get("odds", [])
    odds_updated = payload.get("last_updated", "")

    if not odds_rows:
        print("[WARN] No odds rows returned — tournament may not be active.")
        sys.exit(0)

    rows = []
    for r in odds_rows:
        # Flatten baseline_odds dict
        baseline_raw = r.get("baseline_odds", {}) or {}
        if isinstance(baseline_raw, str):
            try:
                baseline_raw = ast.literal_eval(baseline_raw)
            except Exception:
                baseline_raw = {}
        baseline_val = baseline_raw.get("baseline_history_fit")

        rows.append({
            "player_name":        r.get("player_name"),
            "dg_id":              r.get("dg_id"),
            f"{BOOK}_odds":       pd.to_numeric(r.get(BOOK), errors="coerce"),
            "baseline_odds":      pd.to_numeric(baseline_val, errors="coerce") if baseline_val else None,
            "last_updated":       odds_updated,
            "fetched_at":         fetched_at,
        })

    df = pd.DataFrame(rows)
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    INUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    n_with_odds = df[f"{BOOK}_odds"].notna().sum()
    print(f"[OK] {len(df)} players, {n_with_odds} with {BOOK} odds -> {OUT_PATH.name}")
    print(f"     DG last updated: {odds_updated}")

if __name__ == "__main__":
    run()
