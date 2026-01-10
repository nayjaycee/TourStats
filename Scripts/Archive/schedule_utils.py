from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work as package AND when run directly
# ------------------------------------------------------------
try:
    # Normal package-style imports
    from .config import ODDS_AND_RESULTS_PATH
    from .odds_results import load_odds_and_results
except ImportError:
    # Fallback when running this file directly
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Scripts.Archive.config import ODDS_AND_RESULTS_PATH
    from Scripts.Archive.odds_results import load_odds_and_results


# ============================================================
# CONFIG / NORMALIZATION
# ============================================================

@dataclass
class TierConfig:
    """
    Mapping from messy event_tier strings to canonical buckets.
    """
    major_labels: tuple = ("major", "majors")
    signature_labels: tuple = (
        "signature",
        "signature event",
        "designated",
        "elevated",
        "invitationals",   # if you’ve labeled any this way
    )
    regular_labels: tuple = (
        "regular",
        "standard",
        "pga",
        "full-field",
    )


def normalize_event_tier(raw: str, cfg: Optional[TierConfig] = None) -> str:
    """
    Map raw event_tier text into one of:
        'major', 'signature', 'regular', 'other'
    """
    if cfg is None:
        cfg = TierConfig()

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "other"

    s = str(raw).strip().lower()

    if any(lbl in s for lbl in cfg.major_labels):
        return "major"
    if any(lbl in s for lbl in cfg.signature_labels):
        return "signature"
    if any(lbl in s for lbl in cfg.regular_labels):
        return "regular"

    # If your sheet uses explicit names like "Major", "Sig", etc.,
    # we can extend this later once we see your actual values.
    return "other"


def _guess_purse_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to guess which column is the event purse.

    We’ll keep this flexible so we don’t keep rewriting if the
    Excel changes. It just gives us *one* numeric purse column.
    """
    candidates = [
        "purse",
        "total_purse",
        "purse_usd",
        "event_purse",
        "winner_share_total_purse",  # if you’ve got something like this
    ]

    cols_norm = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_norm:
            return cols_norm[cand]

    # No obvious purse column
    return None


# ============================================================
# BUILD EVENT SCHEDULE
# ============================================================

def build_event_schedule(
    odds_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a canonical event schedule from Odds_and_Results.xlsx.

    Output columns (where available):
      - season (int)
      - event_id (Int64)
      - event_name (str)
      - event_tier_raw (original text)
      - event_tier_norm in {'major', 'signature', 'regular', 'other'}
      - event_completed (Timestamp, usually the start date / odds date)
      - purse (float) if we can find a column that looks like total purse

    Strategy:
      - load odds/results
      - normalize columns
      - group by (season, event_id, event_name, event_tier) and
        take earliest date + first purse.
    """
    if odds_results is None:
        odds_results = load_odds_and_results(copy=True)

    df = odds_results.copy()

    # Normalize key columns
    df.columns = df.columns.astype(str)
    cols_lower = {c.lower(): c for c in df.columns}

    # ------------------------------------------------------------
    # EVENT DATE: Always use event_completed from Odds_and_Results.xlsx
    # ------------------------------------------------------------
    if "event_completed" not in cols_lower:
        raise ValueError("build_event_schedule: expected an 'event_completed' column in Odds_and_Results.xlsx.")

    date_col = cols_lower["event_completed"]

    # Parse event_completed, even if it's in a weird format
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        infer_datetime_format=True,
        utc=False,
    )

    if df[date_col].isna().any():
        print(
            "Warning: some event_completed values could not be parsed to datetime; "
            "those rows will have NaT event_completed."
        )

    # Season
    if "season" in cols_lower:
        season_col = cols_lower["season"]
    else:
        season_col = "season"
        df[season_col] = df[date_col].dt.year

    # event_id
    event_id_col = None
    for cand in ["event_id", "event_id_fixed"]:
        if cand in cols_lower:
            event_id_col = cols_lower[cand]
            break
    if event_id_col is None:
        raise ValueError("build_event_schedule: expected an 'event_id' or 'event_id_fixed' column in odds/results.")

    df[event_id_col] = pd.to_numeric(df[event_id_col], errors="coerce").astype("Int64")

    # event_name
    event_name_col = None
    for cand in ["event_name", "tournament", "event"]:
        if cand in cols_lower:
            event_name_col = cols_lower[cand]
            break
    if event_name_col is None:
        raise ValueError("build_event_schedule: expected an 'event_name' column in odds/results.")

    # event_tier (raw)
    event_tier_col = None
    for cand in ["event_tier", "tier", "event_type"]:
        if cand in cols_lower:
            event_tier_col = cols_lower[cand]
            break
    if event_tier_col is None:
        # Not fatal — we’ll treat everything as 'other' and can fill later
        df["event_tier_raw"] = np.nan
    else:
        df["event_tier_raw"] = df[event_tier_col]

    # purse
    purse_col = _guess_purse_column(df)
    if purse_col is not None:
        df[purse_col] = pd.to_numeric(df[purse_col], errors="coerce")
    # else: we’ll leave purse as NaN in the schedule

    # Build group key
    grp_cols = [season_col, event_id_col, event_name_col, "event_tier_raw"]

    # Aggregate:
    #  - earliest date we see for the event in that season (odds markets)
    #  - first purse value if present
    agg_dict: dict[str, tuple] = {
        date_col: ("event_completed", "min"),
    }

    if purse_col is not None:
        agg_dict[purse_col] = ("purse", "first")

    grouped = (
        df.groupby(grp_cols)
          .agg(**agg_dict)
          .reset_index()
    )

    # Rename key columns to canonical names
    grouped = grouped.rename(
        columns={
            season_col: "season",
            event_id_col: "event_id",
            event_name_col: "event_name",
        }
    )

    # Normalize tiers
    grouped["event_tier_norm"] = grouped["event_tier_raw"].apply(normalize_event_tier)

    # Ensure types
    grouped["season"] = pd.to_numeric(grouped["season"], errors="coerce").astype("Int64")
    grouped["event_id"] = pd.to_numeric(grouped["event_id"], errors="coerce").astype("Int64")
    grouped["event_completed"] = pd.to_datetime(grouped["event_completed"], errors="coerce")

    # Sort schedule for convenience
    grouped = grouped.sort_values(["season", "event_completed", "event_name"]).reset_index(drop=True)

    return grouped


def load_event_schedule() -> pd.DataFrame:
    """
    Convenience wrapper:
      - calls load_odds_and_results()
      - then build_event_schedule()
    """
    odds = load_odds_and_results(copy=True)
    return build_event_schedule(odds_results=odds)


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Running schedule_utils.py self-test...")

    try:
        sched = load_event_schedule()
        print("Schedule columns:", sched.columns.tolist())
        print("Sample rows:")
        print(
            sched[
                [
                    "season",
                    "event_id",
                    "event_name",
                    "event_tier_raw",
                    "event_tier_norm",
                    "event_completed",
                ] + (["purse"] if "purse" in sched.columns else [])
            ].head(15)
        )

        print("\nSeason counts:")
        print(sched["season"].value_counts().sort_index())

        print("\nTier breakdown:")
        print(sched["event_tier_norm"].value_counts())

    except Exception as e:
        print("Self-test failed:", e)