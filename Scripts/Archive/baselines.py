from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work both as package AND when run directly
# ------------------------------------------------------------
try:
    # Normal case: imported as part of the OAD package
    from .config import (
        HALF_LIVES_ROUNDS,
        MIN_ROUNDS_BASELINE,
        RECENT_WINDOWS,
    )
    from .data_loading import load_combined_rounds
except ImportError:
    # Fallback: running this file directly (not recommended for normal use)
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Scripts.Archive.config import (
        HALF_LIVES_ROUNDS,
        MIN_ROUNDS_BASELINE,
        RECENT_WINDOWS,
    )
    from Scripts.Archive.data_loading import load_combined_rounds

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have a single 'ts_round' column for time ordering:
      - Prefer round_date if present.
      - Fall back to event_completed.
    Rows with no usable date are dropped.
    """
    if "round_date" in df.columns:
        ts = pd.to_datetime(df["round_date"], errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)

    if "event_completed" in df.columns:
        # Use event_completed where round_date is missing
        mask = ts.isna()
        if mask.any():
            ts.loc[mask] = pd.to_datetime(
                df.loc[mask, "event_completed"], errors="coerce"
            )

    ts = pd.to_datetime(ts, errors="coerce")
    df = df.copy()
    df["ts_round"] = ts

    # Drop rows with no valid timestamp
    df = df[df["ts_round"].notna()].copy()
    return df


def _filter_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Keep only rounds strictly before the as_of timestamp.
    """
    return df[df["ts_round"] < as_of].copy()


# ============================================================
# EXP-DECAY BASELINES
# ============================================================

def compute_exp_decay_baselines(
    rounds: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    max_rounds: int = 150,
    min_rounds: int = MIN_ROUNDS_BASELINE,
    half_lives: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """
    Compute exponential-decay baselines for each player (dg_id) as of as_of_date.

    Uses up to `max_rounds` most recent rounds before as_of_date and applies
    half-life weights per stat:

        HALF_LIVES_ROUNDS = {
            "driving_dist": 25,
            "driving_acc": 25,
            "sg_app": 60,
            "sg_arg": 60,
            "sg_putt": 120,
            "sg_total": 80,
        }

    Returns a DataFrame with columns:
        dg_id, player_name, n_rounds,
        skill_dist, skill_acc, skill_app, skill_arg, skill_putt, skill_total
    """
    if half_lives is None:
        half_lives = HALF_LIVES_ROUNDS

    if "dg_id" not in rounds.columns:
        raise ValueError("Expected 'dg_id' in rounds DataFrame.")

    as_of_ts = pd.to_datetime(as_of_date)

    df = _ensure_timestamp(rounds)
    df = _filter_as_of(df, as_of_ts)

    # Only rows with at least sg_total; other stats may be NaN and will be handled via np.nanmean
    # We'll still keep them in – decay weights will act on whatever is present.
    df = df.copy()
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    # Sort by recency and cap to max_rounds per player
    df = df.sort_values(["dg_id", "ts_round"], ascending=[True, False])
    df["round_index"] = df.groupby("dg_id").cumcount()
    df = df[df["round_index"] < max_rounds].copy()

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        out: dict[str, object] = {
            "dg_id": g["dg_id"].iloc[0],
            "player_name": g["player_name"].iloc[0] if "player_name" in g.columns else None,
            "n_rounds": n,
        }

        if n < min_rounds:
            for col in ["skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt", "skill_total"]:
                out[col] = np.nan
            return pd.Series(out)

        # index 0 = most recent
        idx = np.arange(n, dtype=float)

        def _decayed_mean(values: np.ndarray, hl_rounds: int | None) -> float:
            # If no half-life or all NaN -> return NaN
            if hl_rounds is None or np.all(np.isnan(values)):
                return np.nan
            lam = np.log(2.0) / float(hl_rounds)
            w = np.exp(-lam * idx)
            # zero out weights where value is NaN
            mask = ~np.isnan(values)
            if not mask.any():
                return np.nan
            w_eff = w[mask]
            v_eff = values[mask]
            return float(np.sum(w_eff * v_eff) / np.sum(w_eff))

        # Distance / Accuracy
        dist_vals = g["driving_dist"].to_numpy(dtype=float) if "driving_dist" in g.columns else np.full(n, np.nan)
        acc_vals = g["driving_acc"].to_numpy(dtype=float) if "driving_acc" in g.columns else np.full(n, np.nan)

        out["skill_dist"] = _decayed_mean(dist_vals, half_lives.get("driving_dist"))
        out["skill_acc"] = _decayed_mean(acc_vals, half_lives.get("driving_acc"))

        # SG stats
        for stat_key, out_key in [
            ("sg_app", "skill_app"),
            ("sg_arg", "skill_arg"),
            ("sg_putt", "skill_putt"),
            ("sg_total", "skill_total"),
        ]:
            vals = g[stat_key].to_numpy(dtype=float) if stat_key in g.columns else np.full(n, np.nan)
            out[out_key] = _decayed_mean(vals, half_lives.get(stat_key))

        return pd.Series(out)

    res = (
        df.groupby("dg_id", group_keys=False)
          .apply(_agg_player, include_groups=False)
          .reset_index(drop=True)
    )

    # Keep only players with at least one non-null skill
    skill_cols = ["skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt", "skill_total"]
    mask_any = res[skill_cols].notna().any(axis=1)
    res = res[mask_any].reset_index(drop=True)

    return res


def compute_exp_decay_baselines_from_all(
    as_of_date: str | pd.Timestamp,
    max_rounds: int = 150,
    min_rounds: int = MIN_ROUNDS_BASELINE,
) -> pd.DataFrame:
    """
    Convenience wrapper: load combined rounds (2017–2025) and compute
    exp-decay baselines as of as_of_date.
    """
    all_rounds = load_combined_rounds(copy=False)
    return compute_exp_decay_baselines(
        all_rounds,
        as_of_date=as_of_date,
        max_rounds=max_rounds,
        min_rounds=min_rounds,
    )


# ============================================================
# RECENT-FORM WINDOWS (L40 / L24 / L12)
# ============================================================

def compute_recent_form(
    rounds: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    windows: Iterable[int] = RECENT_WINDOWS,
    stats: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Compute recent-form summary for each player as of as_of_date.

    For each dg_id and each window N in `windows`, compute:
        - mean sg_total, sg_app, sg_arg, sg_putt
        - mean driving_dist, driving_acc
        - mean round_score

    All windows are based on the N most recent rounds *before* as_of_date
    for that player (ordered by ts_round).

    Returns a wide DataFrame with columns like:
        dg_id, player_name,
        sg_total_L40, sg_app_L40, ..., driving_dist_L40, ...,
        sg_total_L24, ..., round_score_L12, ...
        plus n_rounds_L40, n_rounds_L24, n_rounds_L12
    """
    if stats is None:
        stats = [
            "sg_total",
            "sg_app",
            "sg_arg",
            "sg_putt",
            "driving_dist",
            "driving_acc",
            "round_score",
        ]

    as_of_ts = pd.to_datetime(as_of_date)
    windows = sorted(set(int(w) for w in windows), reverse=True)  # largest first

    if "dg_id" not in rounds.columns:
        raise ValueError("Expected 'dg_id' in rounds DataFrame.")

    df = _ensure_timestamp(rounds)
    df = _filter_as_of(df, as_of_ts)

    df = df.copy()
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")
    df = df.sort_values(["dg_id", "ts_round"], ascending=[True, False])

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        out: dict[str, object] = {
            "dg_id": g["dg_id"].iloc[0],
            "player_name": g["player_name"].iloc[0] if "player_name" in g.columns else None,
        }

        n_total = len(g)

        for N in windows:
            gN = g.head(N)
            out[f"n_rounds_L{N}"] = len(gN)

            for stat in stats:
                colname = f"{stat}_L{N}"
                if stat not in gN.columns:
                    out[colname] = np.nan
                    continue
                vals = pd.to_numeric(gN[stat], errors="coerce")
                out[colname] = float(vals.mean()) if len(vals) > 0 else np.nan

        return pd.Series(out)

    res = (
        df.groupby("dg_id", group_keys=False)
          .apply(_agg_player, include_groups=False)
          .reset_index(drop=True)
    )

    return res


def compute_recent_form_for_season(
    season: int,
    as_of_date: str | pd.Timestamp,
    windows: Iterable[int] = RECENT_WINDOWS,
    stats: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load combined rounds, subset to a season,
    and compute recent form as of as_of_date.
    """
    all_rounds = load_combined_rounds(copy=False)
    df_season = all_rounds[all_rounds["season"] == season].copy()
    return compute_recent_form(
        df_season,
        as_of_date=as_of_date,
        windows=windows,
        stats=stats,
    )