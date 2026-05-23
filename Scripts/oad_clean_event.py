#!/usr/bin/env python3
"""
oad_clean_event.py — OAD Event File Cleaner
============================================
Run this whenever you add a new OAD event CSV to Data/OAD/OAD_events/ (3.5k)
or want to append a new event to Data/OAD/solo_OaD.csv (1k).

Usage:
    python Scripts/oad_clean_event.py <path/to/event.csv>

Auto-detects format:
  - 3.5k:  has columns Rank, Username, Player, Earnings (OAD_events/ format)
  - 1k:    has columns username, entryId, selection, winnings (solo_OaD format)

For 3.5k:
  - Maps Player names → dg_id via combined_roundlevel + All_players lookup
  - Saves dg_id column back to the same file
  - Optionally zeros all Earnings (e.g. mid-tournament projections)

For 1k:
  - Appends rows to solo_OaD.csv (deduplicates on entryId)
  - Prints a summary of what was added
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT          = Path(__file__).parent.parent
INUSE         = ROOT / "Data" / "in Use"
OAD_DIR       = ROOT / "Data" / "OAD"
ROUNDS_PATH   = INUSE / "combined_roundlevel_2024_present.csv"
ALL_PLAYERS   = INUSE / "All_players.xlsx"
SOLO_PATH     = OAD_DIR / "solo_OaD.csv"

# ── Name aliases: raw name (lower) → canonical form for lookup ────────────────
NAME_ALIASES: dict[str, str] = {
    "nicolas echavarria": "echavarria, nico",
    "john keefer":        "keefer, johnny",
    "hao-tong li":        "li, haotong",
    "cam davis":          "davis, cam",
    "byeong hun an":      "an, byeong-hun",
    "k.h. lee":           "lee, kyoung-hoon",
}


def _build_lookup() -> dict[str, int]:
    """Build name (lower) → dg_id. Registers both 'Last, First' and 'First Last'."""
    lookup: dict[str, int] = {}

    def _add(name: str, did: int) -> None:
        key = str(name).strip().lower()
        if not key or key in ("nan", "none"):
            return
        lookup[key] = did
        if "," in key:
            parts = key.split(",", 1)
            lookup[f"{parts[1].strip()} {parts[0].strip()}"] = did

    if ROUNDS_PATH.exists():
        df = pd.read_csv(ROUNDS_PATH, usecols=["dg_id", "player_name"], low_memory=False)
        for _, r in df.drop_duplicates("dg_id").iterrows():
            if pd.notna(r["player_name"]):
                _add(str(r["player_name"]), int(r["dg_id"]))

    if ALL_PLAYERS.exists():
        ap = pd.read_excel(ALL_PLAYERS)
        nc = next((c for c in ap.columns if "name" in c.lower()), None)
        ic = next((c for c in ap.columns if "dg_id" in c.lower()), None)
        if nc and ic:
            for _, r in ap.dropna(subset=[nc, ic]).iterrows():
                _add(str(r[nc]), int(r[ic]))

    return lookup


def _resolve_name(name: str, lookup: dict[str, int]) -> int | None:
    key = str(name).strip().lower()
    key = NAME_ALIASES.get(key, key)
    return lookup.get(key)


def _detect_format(df: pd.DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)
    if "entryid" in cols or "selection" in cols:
        return "1k"
    if "player" in cols and "username" in cols:
        return "3k"
    return "unknown"


def clean_3k(path: Path, zero_earnings: bool | None = None) -> None:
    df = pd.read_csv(path)
    print(f"\n3.5k file: {path.name}  ({len(df)} rows)")

    lookup = _build_lookup()
    print(f"  Lookup built: {len(lookup):,} name entries")

    # Map dg_id
    df["dg_id"] = df["Player"].apply(lambda n: _resolve_name(n, lookup) or 0)
    matched     = (df["dg_id"] > 0).sum()
    unmatched   = df[df["dg_id"] == 0]["Player"].dropna().unique().tolist()

    print(f"  Matched: {matched}/{len(df)}  |  Unmatched: {len(unmatched)}")
    if unmatched:
        print("  Unmatched names:")
        for n in unmatched[:20]:
            print(f"    - {n}")

    # Zero earnings prompt
    if zero_earnings is None:
        ans = input("\n  Zero all Earnings? (y/N): ").strip().lower()
        zero_earnings = ans == "y"

    if zero_earnings:
        df["Earnings"] = "$0"
        print("  Earnings zeroed.")

    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def clean_1k(path: Path) -> None:
    new_df = pd.read_csv(path)
    print(f"\n1k file: {path.name}  ({len(new_df)} rows)")

    # Normalise columns
    new_df.columns = [c.strip() for c in new_df.columns]

    if not SOLO_PATH.exists():
        new_df.to_csv(SOLO_PATH, index=False)
        print(f"  Created {SOLO_PATH} with {len(new_df)} rows.")
        return

    existing = pd.read_csv(SOLO_PATH)
    before   = len(existing)

    # Deduplicate on entryId if present
    if "entryId" in new_df.columns and "entryId" in existing.columns:
        existing_ids = set(existing["entryId"].astype(str))
        new_rows     = new_df[~new_df["entryId"].astype(str).isin(existing_ids)]
    else:
        new_rows = new_df

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(SOLO_PATH, index=False)

    added = len(combined) - before
    print(f"  Added {added} new rows to {SOLO_PATH}  (total: {len(combined)})")
    if added == 0:
        print("  (All rows already existed — nothing new appended.)")

    # Summary by event
    if "eventName" in new_rows.columns:
        for ev, grp in new_rows.groupby("eventName"):
            print(f"    {ev}: {len(grp)} picks")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Scripts/oad_clean_event.py <path/to/event.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    df  = pd.read_csv(path)
    fmt = _detect_format(df)

    if fmt == "3k":
        zero = "--zero" in sys.argv
        clean_3k(path, zero_earnings=zero if zero else None)
    elif fmt == "1k":
        clean_1k(path)
    else:
        print(f"Cannot detect format. Columns: {list(df.columns)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
