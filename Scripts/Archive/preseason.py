from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work as package AND when run directly
# ------------------------------------------------------------
try:
    # Normal case: imported inside OAD package
    from .config import (
        EVENT_SKILL_PATH,
        IN_USE_DIR,
    )
    from .data_loading import load_combined_rounds
except ImportError:
    # Fallback: run as a standalone script
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Scripts.Archive.config import (
        EVENT_SKILL_PATH,
        IN_USE_DIR,
    )
    from Scripts.Archive.data_loading import load_combined_rounds


# ============================================================
# CONFIG FOR PRESEASON HISTORY
# ============================================================

@dataclass
class PreseasonConfig:
    lookback_years: int = 2          # use previous N seasons
    min_rounds: int = 100            # historical rounds required
    decay_lambda_years: float = 0.35 # controls how fast older years are downweighted
    field_k: float = 0.25            # weight scaling for avg_skill


# ============================================================
# HELPERS
# ============================================================

def _weighted_mean(v: pd.Series, w: pd.Series) -> float:
    v = pd.to_numeric(v, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = v.notna() & w.notna()
    if not mask.any():
        return np.nan
    v_eff = v[mask]
    w_eff = w[mask]
    s = w_eff.sum()
    if s <= 0 or not np.isfinite(s):
        return np.nan
    return float((v_eff * w_eff).sum() / s)


def _latest_name(hist: pd.DataFrame) -> pd.Series:
    """
    Get the most recent player_name for each dg_id based on year/event_completed/round_num.
    """
    df = hist.copy()
    # Sort chronologically
    df = df.sort_values(
        ["dg_id", "year", "event_completed", "round_num"],
        ascending=[True, True, True, True],
    )
    latest = (
        df.groupby("dg_id")["player_name"]
          .last()
          .reset_index()
          .rename(columns={"player_name": "player_name_latest"})
    )
    return latest


# ============================================================
# MAIN: BUILD PRESEASON HISTORY
# ============================================================

def build_preseason_history(
    combined_rounds: Optional[pd.DataFrame] = None,
    target_season: int = 2024,
    cfg: Optional[PreseasonConfig] = None,
) -> pd.DataFrame:
    """
    Build preseason history for a given target season.

    Logic:
      - Identify all players who have at least one PGA Tour round in target_season.
      - Look back `lookback_years` seasons before target_season (all tours).
      - Join event_skill (avg_skill) by (year, event_id).
      - Compute, per player:
          * hist_rounds                (# of historical rounds)
          * mean_sg_total              (simple mean)
          * mean_sg_total_field        (field-strength weighted)
          * dec_sg_total               (time-decayed mean)
          * dec_sg_total_field         (time-decayed, field-weighted)
          * mean_round_score           (simple mean round_score)
      - Filter to players with at least min_rounds.
      - Attach latest player_name.
      - Sort by dec_sg_total_field descending.

    Returns:
      DataFrame with one row per player for the target season.
    """
    if cfg is None:
        cfg = PreseasonConfig()

    # 1) Load combined rounds
    if combined_rounds is None:
        combined_rounds = load_combined_rounds(copy=False)

    df = combined_rounds.copy()

    # Basic normalization
    df["tour"] = df["tour"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    # 2) Identify players who play PGA in target_season
    mask_target_pga = (df["tour"] == "PGA") & (df["season"] == target_season)
    pga_players = (
        df.loc[mask_target_pga, "dg_id"]
          .dropna()
          .astype("Int64")
          .unique()
          .tolist()
    )

    if not pga_players:
        raise ValueError(f"No PGA players found in combined rounds for season={target_season}.")

    # 3) Historical window: previous N seasons
    min_year = target_season - cfg.lookback_years
    max_year = target_season - 1

    hist = df[
        df["year"].between(min_year, max_year)
        & df["dg_id"].isin(pga_players)
    ].copy()

    if hist.empty:
        raise ValueError(
            f"No historical rounds found for players in season={target_season} "
            f"using lookback_years={cfg.lookback_years}."
        )

    # 4) Load event_skill (avg_skill) and join
    esk = pd.read_excel(EVENT_SKILL_PATH)
    esk = esk.copy()
    esk.columns = (
        esk.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Expect columns: year, event_id, avg_skill (we'll be robust)
    if "year" not in esk.columns:
        raise ValueError("event_skill.xlsx must contain a 'year' column.")
    # Try to locate event_id and avg_skill columns
    event_id_col = None
    for cand in ["event_id", "event_id_fixed"]:
        if cand in esk.columns:
            event_id_col = cand
            break
    if event_id_col is None:
        raise ValueError("event_skill.xlsx must contain 'event_id' or 'event_id_fixed'.")

    avg_skill_col = None
    for cand in ["avg_skill", "field_strength", "avg_skill_round1"]:
        if cand in esk.columns:
            avg_skill_col = cand
            break
    if avg_skill_col is None:
        raise ValueError("event_skill.xlsx must contain an 'avg_skill' (or similar) column.")

    esk["year"] = pd.to_numeric(esk["year"], errors="coerce").astype("Int64")
    esk[event_id_col] = pd.to_numeric(esk[event_id_col], errors="coerce").astype("Int64")
    esk[avg_skill_col] = pd.to_numeric(esk[avg_skill_col], errors="coerce")

    # minimal join frame
    esk_join = esk[["year", event_id_col, avg_skill_col]].rename(
        columns={event_id_col: "event_id", avg_skill_col: "avg_skill"}
    )

    # Join into hist
    hist = hist.merge(
        esk_join,
        on=["year", "event_id"],
        how="left",
    )

    # 5) Build weights
    # Field-strength weight
    hist["field_w"] = 1.0 + cfg.field_k * hist["avg_skill"].fillna(0.0)

    # Time weight in years (relative to max_year in window)
    # e.g. for target=2025, hist years 2023–2024 → more weight on 2024
    prev_max_year = max_year
    age_years = (prev_max_year - hist["year"]).astype(float)
    hist["time_w"] = np.exp(-cfg.decay_lambda_years * age_years)

    # Combined field*time weight
    hist["field_time_w"] = hist["field_w"] * hist["time_w"]

    # 6) Aggregate per player
    # Ensure numeric sg_total and round_score
    hist["sg_total"] = pd.to_numeric(hist["sg_total"], errors="coerce")
    hist["round_score"] = pd.to_numeric(hist["round_score"], errors="coerce")

    def _agg_group(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "hist_rounds": len(g),
                "mean_sg_total": _weighted_mean(g["sg_total"], pd.Series(1.0, index=g.index)),
                "mean_sg_total_field": _weighted_mean(g["sg_total"], g["field_w"]),
                "dec_sg_total": _weighted_mean(g["sg_total"], g["time_w"]),
                "dec_sg_total_field": _weighted_mean(g["sg_total"], g["field_time_w"]),
                "mean_round_score": _weighted_mean(g["round_score"], pd.Series(1.0, index=g.index)),
            }
        )

    agg = (
        hist.groupby("dg_id", as_index=False)
            .apply(_agg_group)
            .reset_index(drop=True)
    )

    # 7) Filter by min_rounds
    agg = agg[agg["hist_rounds"] >= cfg.min_rounds].copy()
    if agg.empty:
        raise ValueError(
            f"No players meet min_rounds={cfg.min_rounds} for target_season={target_season}."
        )

    # 8) Attach latest player_name
    latest = _latest_name(hist)
    out = agg.merge(latest, on="dg_id", how="left")

    # 9) Sort by strongest signal (decayed field-weighted SG)
    out = out.sort_values("dec_sg_total_field", ascending=False).reset_index(drop=True)

    # Add target season for clarity
    out["target_season"] = target_season

    return out


def save_preseason_history(
    preseason_df: pd.DataFrame,
    target_season: int,
    base_dir: Optional[str | "pd._typing.FilePath"] = None,
) -> str:
    """
    Save preseason table to IN_USE_DIR (or a custom dir) and
    return the path as a string.

    Filename pattern:
        preseason_{target_season}.csv
    """
    if base_dir is None:
        out_dir = IN_USE_DIR
    else:
        from pathlib import Path
        out_dir = Path(base_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"preseason_{target_season}.csv"
    out_path = out_dir / fname
    preseason_df.to_csv(out_path, index=False)
    return str(out_path)


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Running preseason.py self-test...")

    try:
        pre_2024 = build_preseason_history(target_season=2024)
        print("Preseason 2024 sample:")
        print(pre_2024.head())
        print(f"Players in preseason 2024 table: {len(pre_2024)}")

        path_2024 = save_preseason_history(pre_2024, target_season=2024)
        print(f"Saved preseason_2024 to: {path_2024}")

    except Exception as e:
        print("Self-test failed:", e)