from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work as package AND when run directly
# ------------------------------------------------------------
try:
    # Normal case: imported inside OAD package
    from .config import ODDS_AND_RESULTS_PATH
except ImportError:
    # Fallback: run as a standalone script
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Scripts.Archive.config import ODDS_AND_RESULTS_PATH


# ============================================================
# LOADING / CLEANING
# ============================================================

def load_odds_and_results(copy: bool = True) -> pd.DataFrame:
    """
    Load the combined odds + results workbook:

        /Data/in Use/Odds_and_Results.xlsx

    This is intentionally light on assumptions about column names.
    It will:
      - lower/strip column names
      - parse date column into datetime
      - ensure dg_id is numeric if present
      - ensure decimal odds (if present) is numeric

    We WILL refine this as we start using more columns.
    """
    df = pd.read_excel(ODDS_AND_RESULTS_PATH)

    # Normalize column names to snake-ish form
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Fix date column if present
    for cand in ["date", "event_date", "round_date"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            # For convenience, expose a unified 'date' column
            if cand != "date":
                df["date"] = df[cand]
            break

    # Ensure season if not present but we have a usable date
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = df["date"].dt.year

    # dg_id normalization
    for cand in ["dg_id", "dgid", "datagolf_id"]:
        if cand in df.columns:
            df["dg_id"] = pd.to_numeric(df[cand], errors="coerce").astype("Int64")
            break

    # Decimal odds normalization – we will expect a column like 'decimal_odds'
    dec_candidates = ["decimal_odds", "dec_odds", "odds_decimal"]
    for cand in dec_candidates:
        if cand in df.columns:
            df["decimal_odds"] = pd.to_numeric(df[cand], errors="coerce")
            break

    return df.copy() if copy else df


# ============================================================
# IMPLIED PROBS / EV
# ============================================================

def add_implied_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add implied probability from decimal odds:

        implied_prob = 1 / decimal_odds

    Requires a 'decimal_odds' column.
    """
    if "decimal_odds" not in df.columns:
        raise ValueError("add_implied_prob: expected a 'decimal_odds' column.")

    out = df.copy()
    dec = pd.to_numeric(out["decimal_odds"], errors="coerce")
    # Avoid division by zero or nonsense
    out["implied_prob"] = np.where(
        (dec > 1.0) & np.isfinite(dec),
        1.0 / dec,
        np.nan,
    )
    return out


def compute_ev_from_purse(df: pd.DataFrame, purse_col: str = "purse") -> pd.DataFrame:
    """
    Compute simple expected value for a win-only market:

        EV = implied_prob * purse

    Assumes:
      - 'implied_prob' is present
      - purse_col exists and is numeric
    """
    if "implied_prob" not in df.columns:
        raise ValueError("compute_ev_from_purse: expected 'implied_prob' column.")
    if purse_col not in df.columns:
        raise ValueError(f"compute_ev_from_purse: expected '{purse_col}' column.")

    out = df.copy()
    purse = pd.to_numeric(out[purse_col], errors="coerce")
    out["ev"] = out["implied_prob"] * purse
    return out


# ============================================================
# LATEST ODDS BY PLAYER / TIER / DATE
# ============================================================

def get_latest_odds_by_player_tier(
    odds: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    event_tier: Optional[str] = None,
    tier_col: str = "event_tier",
    player_col: str = "dg_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    For each player, find their most recent odds at (optionally) a given tier
    BEFORE as_of_date.

    This matches your intended logic for:
      - "most recent odds for that player at the same level of event
         prior to the date"

    Expected columns (we'll clean them upstream as needed):
      - player_col (default 'dg_id')
      - date_col   (default 'date' as datetime)
      - tier_col   (if event_tier is used)
      - 'decimal_odds' (for later implied prob)

    Returns a DataFrame with ONE row per player, containing the last
    odds row before as_of_date.
    """
    if player_col not in odds.columns:
        raise ValueError(f"get_latest_odds_by_player_tier: expected '{player_col}' column in odds.")
    if date_col not in odds.columns:
        raise ValueError(f"get_latest_odds_by_player_tier: expected '{date_col}' column in odds.")

    df = odds.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    as_of_ts = pd.to_datetime(as_of_date)

    # Filter by date
    df = df[df[date_col] < as_of_ts].copy()

    # Optional tier filter
    if event_tier is not None:
        if tier_col not in df.columns:
            raise ValueError(
                f"get_latest_odds_by_player_tier: event_tier='{event_tier}' "
                f"requested but '{tier_col}' column not found."
            )
        tier_norm = str(event_tier).strip().lower()
        df["tier_norm"] = df[tier_col].astype(str).str.strip().str.lower()
        df = df[df["tier_norm"] == tier_norm].copy()

    if df.empty:
        # No odds at all before that date for this tier
        return df.iloc[0:0].copy()

    # Sort by date, keep most recent per player
    df = df.sort_values([player_col, date_col], ascending=[True, False])

    latest = df.groupby(player_col, as_index=False).head(1).reset_index(drop=True)

    # Drop helper column if we created it
    if "tier_norm" in latest.columns:
        latest = latest.drop(columns=["tier_norm"])

    return latest


def get_latest_odds_for_players(
    odds: pd.DataFrame,
    dg_ids: Iterable[int],
    as_of_date: str | pd.Timestamp,
    event_tier: Optional[str] = None,
    tier_col: str = "event_tier",
    player_col: str = "dg_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Convenience wrapper:
      - filter odds to the given dg_ids
      - then call get_latest_odds_by_player_tier

    Returns one row per dg_id present in odds before as_of_date at that tier.
    """
    dg_ids = list(dg_ids)
    df = odds.copy()

    if player_col not in df.columns:
        raise ValueError(f"get_latest_odds_for_players: expected '{player_col}' column in odds.")

    df = df[df[player_col].isin(dg_ids)].copy()
    return get_latest_odds_by_player_tier(
        df,
        as_of_date=as_of_date,
        event_tier=event_tier,
        tier_col=tier_col,
        player_col=player_col,
        date_col=date_col,
    )


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Running odds_results.py self-test...")
    try:
        odds_df = load_odds_and_results(copy=True)
        print("Columns:", odds_df.columns.tolist())

        if "decimal_odds" in odds_df.columns:
            odds_df = add_implied_prob(odds_df)
            print("Sample with implied_prob:")
            print(odds_df[["dg_id", "decimal_odds", "implied_prob"]].head())

        # If we have event_tier and date, try a toy latest-odds call
        if "event_tier" in odds_df.columns and "date" in odds_df.columns and "dg_id" in odds_df.columns:
            sample_date = odds_df["date"].max()
            sample_tier = odds_df["event_tier"].iloc[0]
            latest = get_latest_odds_by_player_tier(
                odds_df,
                as_of_date=sample_date,
                event_tier=sample_tier,
            )
            print(f"Latest odds rows for tier='{sample_tier}' as of {sample_date}:")
            print(latest.head())

    except Exception as e:
        print("Self-test failed:", e)