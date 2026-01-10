from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work as package AND when run directly
# ------------------------------------------------------------
try:
    # Normal case: imported inside OAD package
    from Scripts.data_loading import load_combined_rounds
except ImportError:
    # Fallback: run as a standalone script
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Scripts.Archive.data_loading import load_combined_rounds


# ============================================================
# CONFIG
# ============================================================

@dataclass
class RollingConfig:
    windows: Sequence[int] = (40, 24, 12)  # L40, L24, L12
    # columns we will compute rolling means for
    stat_cols: Sequence[str] = (
        "sg_total",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "driving_dist",
        "driving_acc",
        "round_score",
    )


# ============================================================
# HELPERS
# ============================================================

def _ensure_ts_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single timestamp column 'ts_round' for ordering rounds.

    Priority:
      1) round_date (if present)
      2) event_completed (if present)

    Uses case-insensitive matching for column names.
    """
    out = df.copy()

    # Case-insensitive lookup
    cols_lower = {c.lower(): c for c in out.columns}

    rd_col = cols_lower.get("round_date")
    ec_col = cols_lower.get("event_completed")

    if rd_col is not None:
        out[rd_col] = pd.to_datetime(out[rd_col], errors="coerce")
    if ec_col is not None:
        out[ec_col] = pd.to_datetime(out[ec_col], errors="coerce")

    if rd_col is not None:
        # Start with round_date
        out["ts_round"] = out[rd_col]
        # Fill missing from event_completed if available
        if ec_col is not None:
            mask = out["ts_round"].isna()
            out.loc[mask, "ts_round"] = out.loc[mask, ec_col]
    elif ec_col is not None:
        # No round_date, fall back to event_completed only
        out["ts_round"] = out[ec_col]
    else:
        raise ValueError(
            "stats_engine: neither 'round_date' nor 'event_completed' columns "
            "were found in combined rounds."
        )

    out["ts_round"] = pd.to_datetime(out["ts_round"], errors="coerce")
    out = out[out["ts_round"].notna()].copy()

    return out

def _to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ============================================================
# MAIN ROLLING STATS FUNCTION
# ============================================================

def compute_rolling_stats_for_field(
    combined_rounds: Optional[pd.DataFrame] = None,
    as_of_date: str | pd.Timestamp = None,
    dg_ids: Optional[Iterable[int]] = None,
    cfg: Optional[RollingConfig] = None,
) -> pd.DataFrame:
    """
    Compute rolling stats (L40, L24, L12 by default) for a set of players
    using ALL tours, considering only rounds BEFORE `as_of_date`.

    Inputs:
      - combined_rounds: if None, will load via load_combined_rounds()
      - as_of_date: cutoff timestamp; only rounds with ts_round < as_of_date are used.
        If None, uses the max ts_round in combined_rounds.
      - dg_ids: optional iterable of dg_ids to restrict to (field).
        If None, computes for all players with rounds before as_of_date.
      - cfg: RollingConfig; controls windows and stat_cols.

    Output:
      One row per dg_id with columns:
        - dg_id
        - player_name_latest
        - n_rounds_total (before cutoff)
        - For each window W in cfg.windows:
            l{W}_rounds
            l{W}_{stat} for each stat in cfg.stat_cols
    """
    if cfg is None:
        cfg = RollingConfig()

    # 1) Load and prep base data
    if combined_rounds is None:
        combined_rounds = load_combined_rounds(copy=False)

    df = combined_rounds.copy()

    # Ensure dg_id is numeric
    if "dg_id" not in df.columns:
        raise ValueError("compute_rolling_stats_for_field: 'dg_id' column is missing in combined_rounds.")
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    # Ensure time column
    df = _ensure_ts_round(df)

    # Cutoff date
    if as_of_date is None:
        cutoff = df["ts_round"].max()
    else:
        cutoff = pd.to_datetime(as_of_date)
    df = df[df["ts_round"] < cutoff].copy()

    if df.empty:
        raise ValueError(f"No rounds found before as_of_date={cutoff}.")

    # Restrict to field if dg_ids provided
    if dg_ids is not None:
        dg_ids = [int(x) for x in dg_ids]
        df = df[df["dg_id"].isin(dg_ids)].copy()
        if df.empty:
            raise ValueError("compute_rolling_stats_for_field: no rounds for provided dg_ids before cutoff.")

    # Convert stat columns to numeric
    df = _to_numeric_inplace(df, cfg.stat_cols)

    # Sort by recency within player
    df = df.sort_values(["dg_id", "ts_round"], ascending=[True, False])

    # 2) Group and aggregate
    windows = list(cfg.windows)
    stat_cols = list(cfg.stat_cols)

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        n_total = len(g)

        out: dict[str, object] = {
            "dg_id": g["dg_id"].iloc[0],
            "n_rounds_total": n_total,
        }

        # For each window, compute round counts + means
        for W in windows:
            sub = g.head(W)
            prefix = f"l{W}"

            out[f"{prefix}_rounds"] = len(sub)

            for stat in stat_cols:
                if stat in sub.columns:
                    out[f"{prefix}_{stat}"] = sub[stat].mean(skipna=True)
                else:
                    out[f"{prefix}_{stat}"] = np.nan

        return pd.Series(out)

    res = (
        df.groupby("dg_id", group_keys=False)
          .apply(_agg_player)
          .reset_index(drop=True)
    )

    return res


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Running stats_engine.py self-test...")

    try:
        combined = load_combined_rounds(copy=False)
        combined = _ensure_ts_round(combined)
        cutoff = combined["ts_round"].max()

        # Take a small sample of players for quick test
        sample_ids = (
            combined["dg_id"]
            .dropna()
            .astype(int)
            .unique()[:20]
            .tolist()
        )

        stats_df = compute_rolling_stats_for_field(
            combined_rounds=combined,
            as_of_date=cutoff,
            dg_ids=sample_ids,
        )

        print("Rolling stats sample:")
        print(
            stats_df[
                [
                    "dg_id",
                    "n_rounds_total",
                    "l40_rounds",
                    "l40_sg_total",
                    "l24_sg_total",
                    "l12_sg_total",
                    "l40_driving_dist",
                    "l40_driving_acc",
                ]
            ].head()
        )

    except Exception as e:
        print("Self-test failed:", e)