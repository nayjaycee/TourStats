"""
field_adjusted_sg.py
====================
Iterative field-strength adjustment for strokes gained metrics.

The Problem
-----------
SG is zero-sum within a field.  Gaining +2.0 vs. a weak DP World Tour field
is not the same as gaining +2.0 vs. a loaded PGA Tour Signature event.

The Algorithm
-------------
We bootstrap player skill ratings from the round data itself using
alternating optimization (similar to an iterative least-squares approach):

    1. Initialize: each player's skill = their career mean sg_total
    2. For each event: field_quality = mean skill of all players in that event
    3. For each round:  adj_sg = raw_sg + (field_quality - reference_quality)
       where reference_quality = mean field_quality of PGA Tour events
    4. Re-estimate skill ratings using recency-weighted mean of adj_sg
       (exponential decay: recent rounds count more)
    5. Repeat 2-4 for n_iter iterations until convergence

Component Adjustments
---------------------
Field quality is derived from sg_total. The same scalar is distributed
evenly across the four components (putt, app, arg, ott), so they remain
internally consistent:
    adj_sg_putt = sg_putt + field_adj / 4
    adj_sg_app  = sg_app  + field_adj / 4
    adj_sg_arg  = sg_arg  + field_adj / 4
    adj_sg_ott  = sg_ott  + field_adj / 4
    adj_sg_t2g  = sg_t2g  + field_adj * 3/4
    adj_sg_total = sg_total + field_adj

This ensures adj_putt + adj_app + adj_arg + adj_ott == adj_total (modulo
any pre-existing rounding in the raw data).

Decay Parameter (decay_lambda)
-------------------------------
Controls how fast older rounds lose influence in skill estimation.
Weight for a round that is i positions back from the player's most
recent round: exp(-decay_lambda * i)

    decay_lambda = 0.00  →  all rounds equal weight
    decay_lambda = 0.03  →  half-life ≈ 23 rounds  (default)
    decay_lambda = 0.05  →  half-life ≈ 14 rounds
    decay_lambda = 0.10  →  half-life ≈  7 rounds

Public API
----------
    compute_field_adjusted_sg(rounds_df, n_iter, decay_lambda, min_rounds, reference_tour)
        → rounds_df with added columns:
            field_quality, field_adj,
            adj_sg_total, adj_sg_putt, adj_sg_app, adj_sg_arg, adj_sg_ott, adj_sg_t2g

    get_field_quality_summary(rounds_df_adjusted)
        → per-event field quality table (diagnostic helper)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────

SG_COMPONENTS = ["sg_putt", "sg_app", "sg_arg", "sg_ott"]
SG_TOTAL      = "sg_total"
SG_T2G        = "sg_t2g"

# Component weights that sum adj_components → adj_total
# putt (1/4) + app (1/4) + arg (1/4) + ott (1/4) = total (4/4)
# t2g = app + arg + ott → weight is 3/4
_COMPONENT_WEIGHT: dict[str, float] = {
    "sg_putt": 1 / 4,
    "sg_app":  1 / 4,
    "sg_arg":  1 / 4,
    "sg_ott":  1 / 4,
    "sg_t2g":  3 / 4,
    "sg_total": 1.0,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _weighted_mean(values: np.ndarray, decay_lambda: float) -> float:
    """Recency-weighted mean.  values[0] is most recent, values[-1] is oldest."""
    n = len(values)
    if n == 0:
        return np.nan
    weights = np.exp(-decay_lambda * np.arange(n))
    weights /= weights.sum()
    return float(np.dot(weights, values))


def _estimate_skill_ratings(
    rounds: pd.DataFrame,
    adj_col: str,
    date_col: str,
    decay_lambda: float,
    min_rounds: int,
    fallback_by_tour: dict[str, float],
) -> dict[int, float]:
    """
    For each player, compute recency-weighted mean of adj_col as their skill rating.
    Players with fewer than min_rounds rounds get the fallback value for their tour.
    """
    ratings: dict[int, float] = {}
    for dg_id, grp in rounds.groupby("dg_id", sort=False):
        grp_sorted = grp.sort_values(date_col, ascending=False, na_position="last")
        vals = grp_sorted[adj_col].dropna().values
        if len(vals) >= min_rounds:
            ratings[int(dg_id)] = _weighted_mean(vals, decay_lambda)
        else:
            # fall back to tour-average for this player's primary tour
            primary_tour = grp["tour"].mode().iloc[0] if "tour" in grp.columns else "PGA"
            ratings[int(dg_id)] = fallback_by_tour.get(str(primary_tour).upper(), 0.0)
    return ratings


def _compute_event_field_quality(
    rounds: pd.DataFrame,
    skill_ratings: dict[int, float],
    fallback: float = 0.0,
) -> pd.Series:
    """
    For each (event_id, season) pair return the mean skill rating of the field.
    Returns a Series indexed by the rounds DataFrame's index.
    """
    # Vectorised lookup
    player_skill = rounds["dg_id"].map(skill_ratings).fillna(fallback)
    group_keys = ["event_id", "season"] if "season" in rounds.columns else ["event_id"]
    event_means = player_skill.groupby([rounds[k] for k in group_keys]).transform("mean")
    return event_means


# ── main public function ───────────────────────────────────────────────────────

def compute_field_adjusted_sg(
    rounds_df: pd.DataFrame,
    n_iter: int = 8,
    decay_lambda: float = 0.03,
    min_rounds: int = 15,
    reference_tour: str = "PGA",
) -> pd.DataFrame:
    """
    Compute field-strength-adjusted SG for every round.

    Parameters
    ----------
    rounds_df : pd.DataFrame
        Round-level data.  Must contain: dg_id, event_id, tour, sg_total,
        and at least one of round_date / event_completed for ordering.
    n_iter : int
        Number of alternating-optimization iterations (default 8; converges by ~5).
    decay_lambda : float
        Exponential decay rate per round for recency weighting (default 0.03,
        half-life ≈ 23 rounds).  0.0 = no decay.
    min_rounds : int
        Minimum rounds a player needs for a "trusted" skill estimate.
        Players below this threshold get the tour-average fallback.
    reference_tour : str
        Tour whose average field quality is used as the 0-point (default "PGA").

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
            field_quality, field_adj,
            adj_sg_total, adj_sg_putt, adj_sg_app, adj_sg_arg, adj_sg_ott, adj_sg_t2g
    """
    df = rounds_df.copy()

    # ── date column ──────────────────────────────────────────────────────────
    if "round_date" in df.columns and df["round_date"].notna().any():
        date_col = "round_date"
    elif "event_completed" in df.columns:
        date_col = "event_completed"
    else:
        raise ValueError("rounds_df must have round_date or event_completed.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # ── filter to rounds with valid sg_total ─────────────────────────────────
    # Preserve original index so we can align results back without a merge key collision
    df = df.reset_index(drop=True)
    valid_mask = df[SG_TOTAL].notna()
    valid = df[valid_mask].copy()

    if valid.empty:
        # Return with NaN adj columns if no valid data
        for col in [f"adj_{c}" for c in SG_COMPONENTS + [SG_T2G, SG_TOTAL]]:
            df[col] = np.nan
        df["field_quality"] = np.nan
        df["field_adj"] = np.nan
        return df

    valid["dg_id"] = pd.to_numeric(valid["dg_id"], errors="coerce")
    valid = valid.dropna(subset=["dg_id"])
    valid["dg_id"] = valid["dg_id"].astype(int)

    # ── fallback skill per tour (mean raw sg_total) ───────────────────────────
    if "tour" in valid.columns:
        fallback_by_tour: dict[str, float] = (
            valid.groupby("tour")[SG_TOTAL].mean().to_dict()
        )
        fallback_by_tour = {k.upper(): v for k, v in fallback_by_tour.items()}
    else:
        fallback_by_tour = {"PGA": 0.0}

    global_fallback = float(valid[SG_TOTAL].mean())

    # ── Step 1: initialize skill ratings from career mean sg_total ───────────
    skill_ratings: dict[int, float] = (
        valid.groupby("dg_id")[SG_TOTAL].mean().to_dict()
    )

    # ── working adj column (iteratively refined) ──────────────────────────────
    valid = valid.sort_values(date_col, na_position="last").copy()
    valid["_adj_sg"] = valid[SG_TOTAL].copy()

    for _iter in range(n_iter):
        # Step 2: field quality per event
        valid["_field_quality"] = _compute_event_field_quality(
            valid, skill_ratings, fallback=global_fallback
        )

        # Step 3: reference quality = median of per-event field quality for reference_tour events
        # Using per-event median (not per-round mean) so that sub-tour outlier events
        # tagged as the reference tour (e.g. Chinese Tour tagged as "PGA") don't contaminate
        # the reference anchor.
        ref_mask = valid["tour"].str.upper() == reference_tour.upper() if "tour" in valid.columns else pd.Series(True, index=valid.index)
        ref_rounds = valid.loc[ref_mask]
        _ref_group_keys = ["event_id", "season"] if "season" in ref_rounds.columns else ["event_id"]
        ref_event_means = ref_rounds.groupby(_ref_group_keys)["_field_quality"].mean()
        reference_quality = float(ref_event_means.median())

        # Step 3 continued: adjusted sg
        valid["_field_adj"] = valid["_field_quality"] - reference_quality
        valid["_adj_sg"] = valid[SG_TOTAL] + valid["_field_adj"]

        # Step 4: re-estimate skill ratings
        skill_ratings = _estimate_skill_ratings(
            valid,
            adj_col="_adj_sg",
            date_col=date_col,
            decay_lambda=decay_lambda,
            min_rounds=min_rounds,
            fallback_by_tour=fallback_by_tour,
        )

    # ── Final pass: compute definitive field_quality and field_adj ────────────
    valid["_field_quality"] = _compute_event_field_quality(
        valid, skill_ratings, fallback=global_fallback
    )
    ref_mask = valid["tour"].str.upper() == reference_tour.upper() if "tour" in valid.columns else pd.Series(True, index=valid.index)
    ref_rounds = valid.loc[ref_mask]
    _ref_group_keys = ["event_id", "season"] if "season" in ref_rounds.columns else ["event_id"]
    ref_event_means = ref_rounds.groupby(_ref_group_keys)["_field_quality"].mean()
    reference_quality = float(ref_event_means.median())
    valid["field_quality"] = valid["_field_quality"]
    valid["field_adj"]     = valid["_field_quality"] - reference_quality

    # ── Apply adjustment to all SG components ────────────────────────────────
    for sg_col, weight in _COMPONENT_WEIGHT.items():
        if sg_col in valid.columns:
            valid[f"adj_{sg_col}"] = valid[sg_col] + valid["field_adj"] * weight

    # ── Drop working columns ─────────────────────────────────────────────────
    valid = valid.drop(columns=["_adj_sg", "_field_quality", "_field_adj"], errors="ignore")

    # ── Assign adj columns back via index alignment ───────────────────────────
    adj_cols = ["field_quality", "field_adj"] + [
        f"adj_{c}" for c in SG_COMPONENTS + [SG_T2G, SG_TOTAL]
        if f"adj_{c}" in valid.columns
    ]
    for col in adj_cols:
        df[col] = np.nan
        df.loc[valid.index, col] = valid[col].values

    return df


# ── diagnostic helper ─────────────────────────────────────────────────────────

def get_field_quality_summary(rounds_df_adjusted: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a per-event summary of field quality, sorted strongest → weakest.
    Useful for validating the adjustment.
    """
    if "field_quality" not in rounds_df_adjusted.columns:
        raise ValueError("Run compute_field_adjusted_sg first.")

    group_keys = ["event_id", "event_name", "tour", "season"] \
        if all(c in rounds_df_adjusted.columns for c in ["event_name", "season"]) \
        else ["event_id", "tour"]

    available_keys = [k for k in group_keys if k in rounds_df_adjusted.columns]

    summary = (
        rounds_df_adjusted
        .groupby(available_keys)["field_quality"]
        .mean()
        .reset_index()
        .rename(columns={"field_quality": "avg_field_quality"})
        .sort_values("avg_field_quality", ascending=False)
        .reset_index(drop=True)
    )
    summary["avg_field_quality"] = summary["avg_field_quality"].round(3)
    return summary
