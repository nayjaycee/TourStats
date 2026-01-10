from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Imports that work both as package AND when run directly
# ------------------------------------------------------------
try:
    # Normal case: imported as part of the OAD package
    from .config import (
        HALF_LIVES_ROUNDS,
        COURSE_FIT_SHRINK_K,
        DERIVED_DIR,
    )
    from .data_loading import load_combined_rounds
except ImportError:
    # Fallback: running this file directly (not the main use case, but keep it working)
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Scripts.Archive.config import (
        HALF_LIVES_ROUNDS,
        COURSE_FIT_SHRINK_K,
        DERIVED_DIR,
    )
    from Scripts.Archive.data_loading import load_combined_rounds


# ============================================================
# DATA STRUCTURES
# ============================================================

ATTR_COLS = ["driving_dist", "driving_acc", "sg_app", "sg_arg", "sg_putt"]


@dataclass
class CourseFitConfig:
    max_rounds_per_player: int = 150
    min_rounds_per_player: int = 20
    min_obs_per_course: int = 80  # event-player-round observations per course
    shrink_k: float = COURSE_FIT_SHRINK_K


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
        mask = ts.isna()
        if mask.any():
            ts.loc[mask] = pd.to_datetime(
                df.loc[mask, "event_completed"], errors="coerce"
            )

    ts = pd.to_datetime(ts, errors="coerce")
    df = df.copy()
    df["ts_round"] = ts
    df = df[df["ts_round"].notna()].copy()
    return df


def _build_player_attribute_skills(
    hist: pd.DataFrame,
    cfg: CourseFitConfig,
) -> pd.DataFrame:
    """
    Build long-run attribute skills for each player using exponential decay in rounds.
    Uses HALF_LIVES_ROUNDS for the relevant attributes.
    """
    if "dg_id" not in hist.columns:
        raise ValueError("Expected 'dg_id' in historical rounds for course fit.")

    df = _ensure_timestamp(hist)

    df = df.copy()
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    # Sort by recency within player and cap to max_rounds
    df = df.sort_values(["dg_id", "ts_round"], ascending=[True, False])
    df["round_index"] = df.groupby("dg_id").cumcount()
    df = df[df["round_index"] < cfg.max_rounds_per_player].copy()

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        out: dict[str, object] = {
            "dg_id": g["dg_id"].iloc[0],
            "player_name": g["player_name"].iloc[0] if "player_name" in g.columns else None,
            "n_rounds_attr": n,
        }

        if n < cfg.min_rounds_per_player:
            for col in ["skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt"]:
                out[col] = np.nan
            return pd.Series(out)

        idx = np.arange(n, dtype=float)

        def _decayed_mean(values: np.ndarray, hl_rounds: int | None) -> float:
            if hl_rounds is None or np.all(np.isnan(values)):
                return np.nan
            lam = np.log(2.0) / float(hl_rounds)
            w = np.exp(-lam * idx)
            mask = ~np.isnan(values)
            if not mask.any():
                return np.nan
            w_eff = w[mask]
            v_eff = values[mask]
            return float(np.sum(w_eff * v_eff) / np.sum(w_eff))

        # Distance and accuracy
        dist_vals = g["driving_dist"].to_numpy(dtype=float) if "driving_dist" in g.columns else np.full(n, np.nan)
        acc_vals = g["driving_acc"].to_numpy(dtype=float) if "driving_acc" in g.columns else np.full(n, np.nan)

        out["skill_dist"] = _decayed_mean(dist_vals, HALF_LIVES_ROUNDS.get("driving_dist"))
        out["skill_acc"] = _decayed_mean(acc_vals, HALF_LIVES_ROUNDS.get("driving_acc"))

        # SG splits
        for stat_key, out_key in [
            ("sg_app", "skill_app"),
            ("sg_arg", "skill_arg"),
            ("sg_putt", "skill_putt"),
        ]:
            vals = g[stat_key].to_numpy(dtype=float) if stat_key in g.columns else np.full(n, np.nan)
            out[out_key] = _decayed_mean(vals, HALF_LIVES_ROUNDS.get(stat_key))

        return pd.Series(out)

    res = (
        df.groupby("dg_id", group_keys=False)
          .apply(_agg_player)
          .reset_index(drop=True)
    )

    # Keep only players with at least one valid skill
    skill_cols = ["skill_dist", "skill_acc", "skill_app", "skill_putt"]
    mask_any = res[skill_cols].notna().any(axis=1)
    res = res[mask_any].reset_index(drop=True)

    return res


def _standardize_attributes(skills: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-scores for each attribute to the skills table.
    """
    df = skills.copy()
    for col, zcol in [
        ("skill_dist", "skill_dist_z"),
        ("skill_acc", "skill_acc_z"),
        ("skill_app", "skill_app_z"),
        ("skill_arg", "skill_arg_z"),
        ("skill_putt", "skill_putt_z"),
    ]:
        mu = df[col].mean(skipna=True)
        sd = df[col].std(ddof=0, skipna=True)
        if sd == 0 or np.isnan(sd):
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - mu) / sd
    return df


# ============================================================
# MAIN: BUILD COURSE-FIT IMPORTANCE
# ============================================================

def build_course_fit_5attr(
    combined_rounds: Optional[pd.DataFrame] = None,
    target_season: int = 2025,
    config: Optional[CourseFitConfig] = None,
) -> pd.DataFrame:
    """
    Build course-fit profiles for all PGA Tour courses used in `target_season`,
    using only rounds with season < target_season.

    We:
      1. Identify all courses (course_num, course_name) that host PGA events
         in target_season.
      2. Build long-run attribute skills for each player from pre-target_season
         rounds (all tours), with exp-decay weighting by HALF_LIVES_ROUNDS.
      3. For each course, take all pre-target_season PGA rounds at that course,
         join to player attribute z-scores, and regress sg_total on the
         5 z-attributes.
      4. Shrink course-specific coefficients toward 0 using:
            coef_shrunk = coef_raw * (n_obs / (n_obs + shrink_k))
      5. Convert |coef_shrunk| to importance weights that sum to 1.

    Returns a DataFrame with columns:
        course_num, course_name, profile_season,
        n_obs, n_players,
        coef_dist, coef_acc, coef_app, coef_arg, coef_putt,
        imp_dist, imp_acc, imp_app, imp_arg, imp_putt
    """
    if config is None:
        config = CourseFitConfig()

    if combined_rounds is None:
        combined_rounds = load_combined_rounds(copy=False)

    df = combined_rounds.copy()

    # Basic normalization
    df["tour"] = df["tour"].astype(str).str.upper()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")
    df["course_num"] = pd.to_numeric(df["course_num"], errors="coerce").astype("Int64")

    # Identify PGA courses for the target season
    mask_target = (df["tour"] == "PGA") & (df["season"] == target_season)
    courses_target = (
        df.loc[mask_target, ["course_num", "course_name"]]
        .dropna(subset=["course_num"])
        .drop_duplicates()
    )

    if courses_target.empty:
        raise ValueError(f"No PGA rounds found for season={target_season} in combined rounds.")

    # Historical rounds (pre-target season)
    hist = df[df["season"] < target_season].copy()
    if hist.empty:
        raise ValueError(f"No historical rounds with season < {target_season}.")

    # Build attribute skills for all players (all tours) pre-target-season
    skills = _build_player_attribute_skills(hist, cfg=config)
    if skills.empty:
        raise ValueError("No player attribute skills could be computed (check MIN_ROUNDS / data).")

    skills_z = _standardize_attributes(skills)

    # Restrict historical rounds to PGA only for course fit estimation
    hist_pga = hist[hist["tour"] == "PGA"].copy()
    hist_pga = _ensure_timestamp(hist_pga)

    # Merge course + player attributes (z-scores)
    merged = hist_pga.merge(
        skills_z[
            [
                "dg_id",
                "skill_dist_z",
                "skill_acc_z",
                "skill_app_z",
                "skill_arg_z",
                "skill_putt_z",
            ]
        ],
        on="dg_id",
        how="inner",
    )

    # Only rows where sg_total is present
    merged["sg_total"] = pd.to_numeric(merged["sg_total"], errors="coerce")
    merged = merged[merged["sg_total"].notna()].copy()

    # We'll estimate per-course importance
    records: list[dict[str, object]] = []

    for _, crow in courses_target.iterrows():
        cnum = int(crow["course_num"])
        cname = crow["course_name"]

        df_c = merged[merged["course_num"] == cnum].copy()
        if len(df_c) < config.min_obs_per_course:
            # Not enough data to say anything meaningful
            continue

        # Prepare X, y
        X_cols = ["skill_dist_z", "skill_acc_z", "skill_app_z", "skill_arg_z", "skill_putt_z"]
        X = df_c[X_cols].to_numpy(dtype=float)
        y = df_c["sg_total"].to_numpy(dtype=float)

        # If all-zero variance or something pathological, skip
        if np.all(np.isnan(X)) or np.all(np.isnan(y)):
            continue

        # Replace any remaining NaNs with 0 in X (no signal) and mask them via shrinkage
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Simple linear regression via normal equations to avoid sklearn dependency
        # (X'X + λI)^(-1) X'y; here λ is tiny ridge to avoid singularity
        n_obs = len(df_c)
        XTX = X.T @ X
        XTy = X.T @ y
        ridge = 1e-6 * np.eye(XTX.shape[0])
        try:
            coefs_raw = np.linalg.solve(XTX + ridge, XTy)
        except np.linalg.LinAlgError:
            # Singular – skip this course
            continue

        # Shrink toward 0 based on sample size
        shrink_factor = n_obs / (n_obs + config.shrink_k)
        coefs_shrunk = coefs_raw * shrink_factor

        abs_shrunk = np.abs(coefs_shrunk)
        s = abs_shrunk.sum()
        if s == 0 or np.isnan(s):
            # No signal – treat as flat, importance is NaN
            imp = [np.nan] * 5
        else:
            imp = (abs_shrunk / s).tolist()

        rec: dict[str, object] = {
            "course_num": cnum,
            "course_name": cname,
            "profile_season": target_season,
            "n_obs": n_obs,
            "n_players": df_c["dg_id"].nunique(),
        }

        # Raw coefficients (shrunk)
        rec["coef_dist"] = float(coefs_shrunk[0])
        rec["coef_acc"] = float(coefs_shrunk[1])
        rec["coef_app"] = float(coefs_shrunk[2])
        rec["coef_arg"] = float(coefs_shrunk[3])
        rec["coef_putt"] = float(coefs_shrunk[4])

        # Importance weights
        rec["imp_dist"] = float(imp[0])
        rec["imp_acc"] = float(imp[1])
        rec["imp_app"] = float(imp[2])
        rec["imp_arg"] = float(imp[3])
        rec["imp_putt"] = float(imp[4])

        records.append(rec)

    course_fit = pd.DataFrame(records)
    if course_fit.empty:
        raise ValueError(
            f"No course-fit profiles produced for season={target_season}. "
            f"Check min_obs_per_course={config.min_obs_per_course} and data coverage."
        )

    return course_fit.sort_values("course_name").reset_index(drop=True)


def save_course_fit_5attr(
    course_fit_df: pd.DataFrame,
    target_season: int,
    base_dir: Optional[str | "pd._typing.FilePath"] = None,
) -> str:
    """
    Save the course-fit table to DERIVED_DIR (or a custom dir) and
    return the path as a string.

    Filename pattern:
        course_fit_5attr_pre{target_season}.csv
    """
    if base_dir is None:
        out_dir = DERIVED_DIR
    else:
        out_dir = (base_dir if isinstance(base_dir, pd.Path) else pd.Path(base_dir))  # type: ignore

    out_dir = DERIVED_DIR if base_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"course_fit_5attr_pre{target_season}.csv"
    out_path = out_dir / fname
    course_fit_df.to_csv(out_path, index=False)
    return str(out_path)


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Running course_fit.py self-test...")
    try:
        cf_2025 = build_course_fit_5attr(target_season=2025)
        print(cf_2025.head())
        print(f"Built course-fit profiles for {len(cf_2025)} courses (pre-2025).")
    except Exception as e:
        print(f"Self-test failed: {e}")