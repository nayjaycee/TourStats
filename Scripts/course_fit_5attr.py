# course_fit_5attr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from Scripts.config import COURSE_FIT_TEMPLATE, PLAYER_SKILL_TEMPLATE
from Scripts.data_io import load_rounds


@dataclass
class CourseFitConfig:
    max_rounds: int = 150
    min_rounds: int = 20
    min_obs_per_course: int = 80
    shrink_k: float = 400.0


def _compute_player_skills(
    hist: pd.DataFrame,
    max_rounds: int,
    min_rounds: int,
) -> pd.DataFrame:
    """
    Internal helper: compute 5-attribute exp-decay skills per player.
    """
    half_lives = {
        "driving_dist": 25,
        "driving_acc": 25,
        "sg_app": 60,
        "sg_arg": 60,
        "sg_putt": 120,
    }

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        if "dg_id" in g.columns:
            dg_id_val = g["dg_id"].iloc[0]
        else:
            dg_id_val = g.name

        g = g.copy().sort_values("event_completed", ascending=False)
        g = g.head(max_rounds)

        n = len(g)
        out = {
            "dg_id": dg_id_val,
            "player_name": g["player_name"].iloc[0],
            "n_rounds": n,
        }

        if n < min_rounds:
            for col in [
                "skill_dist",
                "skill_acc",
                "skill_app",
                "skill_arg",
                "skill_putt",
            ]:
                out[col] = np.nan
            return pd.Series(out)

        idx = np.arange(n)

        for stat, hl in half_lives.items():
            lam = np.log(2) / hl
            w = np.exp(-lam * idx)
            w = w / w.sum()

            vals = g[stat].to_numpy()
            skill = float(np.sum(w * vals))
            key = {
                "driving_dist": "skill_dist",
                "driving_acc": "skill_acc",
                "sg_app": "skill_app",
                "sg_arg": "skill_arg",
                "sg_putt": "skill_putt",
            }[stat]
            out[key] = skill

        return pd.Series(out)

    skills = (
        hist.groupby("dg_id", group_keys=False)
        .apply(_agg_player, include_groups=False)
        .reset_index(drop=True)
    )

    skill_cols = ["skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt"]
    skills = skills.dropna(subset=skill_cols)

    return skills


def _scale_0_1_like_players(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max scaling (same as player skill scaling). Forces 0 and 1 to exist per col."""
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        lo = float(np.nanmin(x.to_numpy()))
        hi = float(np.nanmax(x.to_numpy()))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[c] = (x - lo) / (hi - lo)
        else:
            out[c] = np.nan
        out[c] = out[c].clip(0.0, 1.0)
    return out


def _scale_0_1_percentile(df: pd.DataFrame, cols: list[str], lo_q: float = 0.05, hi_q: float = 0.95) -> pd.DataFrame:
    """
    DG-ish display scaling: winsorize then min-max, which *usually* avoids hard 0/1
    unless values hit the winsor caps.
    """
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        lo = float(x.quantile(lo_q))
        hi = float(x.quantile(hi_q))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            xw = x.clip(lo, hi)
            out[c] = (xw - lo) / (hi - lo)
        else:
            out[c] = np.nan
        out[c] = out[c].clip(0.0, 1.0)
    return out


def build_course_fit_5attr(
    combined: pd.DataFrame,
    target_season: int,
    cfg: CourseFitConfig = CourseFitConfig(),
    target_courses: Optional[pd.DataFrame] = None,
    course_scale: str = "player_minmax",  # "player_minmax" or "percentile"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes:
      1) course_profiles: DG-style 5-attribute course importances for courses
      2) player_skills: exp-decay skills per player
    """

    df = combined.copy()

    df["tour"] = df["tour"].astype(str).str.upper()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")
    df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")

    for col in ["sg_app", "sg_arg", "sg_putt", "sg_total", "driving_dist", "driving_acc"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in combined DataFrame.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Historical PGA rounds before target_season
    hist = df[(df["tour"] == "PGA") & (df["season"] < target_season)].copy()
    if hist.empty:
        raise ValueError(f"No historical PGA rounds with season < {target_season}.")

    attr_cols = [
        "driving_dist", "driving_acc", "sg_app", "sg_arg", "sg_putt", "sg_total",
        "dg_id", "event_completed", "event_id", "round_num", "event_name", "course_num", "course_name", "player_name"
    ]
    missing_hist = [c for c in attr_cols if c not in hist.columns]
    if missing_hist:
        raise ValueError(f"Missing required columns in hist: {missing_hist}")

    hist = hist.dropna(subset=["driving_dist","driving_acc","sg_app","sg_arg","sg_putt","sg_total","dg_id","event_completed","event_id","round_num","course_num"])
    hist["course_num"] = pd.to_numeric(hist["course_num"], errors="coerce")
    hist = hist.dropna(subset=["course_num"]).copy()
    hist["course_num"] = hist["course_num"].astype(int)

    hist = hist.sort_values(["dg_id", "event_completed", "event_id", "round_num"], ascending=[True, True, True, True]).reset_index(drop=True)

    # 1) player skills
    skills = _compute_player_skills(hist, cfg.max_rounds, cfg.min_rounds)
    skill_cols = ["skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt"]
    if skills.empty:
        raise ValueError("No players with valid 5-attribute skills for course fit.")

    # 2) event-level performance merged with skills
    event_perf = (
        hist.groupby(["event_id", "event_name", "course_num", "course_name", "dg_id"], as_index=False)
            .agg(event_sg_total=("sg_total", "sum"))
    )

    event_perf = event_perf.merge(skills[["dg_id"] + skill_cols], on="dg_id", how="inner")
    if event_perf.empty:
        raise ValueError("No event-player rows after merging skills; check data coverage.")

    event_perf["event_sg_total_centered"] = (
        event_perf["event_sg_total"]
        - event_perf.groupby(["event_id", "course_num"])["event_sg_total"].transform("mean")
    )

    # 3) global z-scores for skills
    z_map = {"skill_dist": "z_dist", "skill_acc": "z_acc", "skill_app": "z_app", "skill_arg": "z_arg", "skill_putt": "z_putt"}
    for col, zname in z_map.items():
        mu = event_perf[col].mean()
        sd = event_perf[col].std(ddof=0)
        event_perf[zname] = 0.0 if (sd == 0 or np.isnan(sd)) else (event_perf[col] - mu) / sd

    # 4) target courses
    if target_courses is not None:
        courses_target = target_courses.copy()
        if "course_num" not in courses_target.columns:
            raise ValueError("target_courses must include a 'course_num' column.")

        if "course_name" not in courses_target.columns:
            name_map = (
                df[(df["tour"] == "PGA") & (df["season"] < target_season)]
                .dropna(subset=["course_num", "course_name"])
                .groupby("course_num")["course_name"]
                .agg(lambda x: x.value_counts().index[0])
                .rename("course_name")
                .reset_index()
            )
            courses_target = courses_target.merge(name_map, on="course_num", how="left")
        else:
            courses_target["course_name"] = courses_target["course_name"].astype(str)

        courses_target["course_num"] = pd.to_numeric(courses_target["course_num"], errors="coerce")
        courses_target = (
            courses_target.dropna(subset=["course_num"])
            .drop_duplicates(subset=["course_num"])
            .sort_values("course_num")
            .reset_index(drop=True)
        )
        if courses_target.empty:
            raise ValueError("target_courses resolved to empty after cleaning.")
    else:
        courses_target = (
            df[(df["tour"] == "PGA") & (df["season"] == target_season)]
            .loc[:, ["course_num", "course_name"]]
            .dropna(subset=["course_num"])
            .drop_duplicates()
            .sort_values("course_num")
            .reset_index(drop=True)
        )
        if courses_target.empty:
            raise ValueError(f"No PGA courses found for season {target_season}.")

    # 5) course regressions -> pow_*
    profiles = []
    for _, row in courses_target.iterrows():
        cnum = int(pd.to_numeric(row["course_num"], errors="coerce"))
        cname = str(row.get("course_name", ""))

        df_c = event_perf[event_perf["course_num"] == cnum].copy()
        if len(df_c) < cfg.min_obs_per_course:
            continue

        df_c = df_c.dropna(subset=["event_sg_total_centered","z_dist","z_acc","z_app","z_arg","z_putt"])
        if len(df_c) < cfg.min_obs_per_course:
            continue

        X = df_c[["z_dist","z_acc","z_app","z_arg","z_putt"]].to_numpy()
        y = df_c["event_sg_total_centered"].to_numpy()

        model = LinearRegression()
        model.fit(X, y)

        r2 = float(model.score(X, y)) if len(y) > 3 else 0.0
        r2 = float(np.clip(r2, 0.0, 1.0))
        predictability = float(np.sqrt(r2))

        abs_b = np.abs(model.coef_)
        pow_dist, pow_acc, pow_app, pow_arg, pow_putt = abs_b.tolist()

        # downweight noisy courses + shrinkage
        pow_dist *= predictability
        pow_acc  *= predictability
        pow_app  *= predictability
        pow_arg  *= predictability
        pow_putt *= predictability

        n_obs = int(len(df_c))
        shrink = float(n_obs / (n_obs + cfg.shrink_k))

        pow_dist *= shrink
        pow_acc  *= shrink
        pow_app  *= shrink
        pow_arg  *= shrink
        pow_putt *= shrink

        profiles.append({
            "course_num": cnum,
            "course_name": cname,
            "pow_dist": pow_dist,
            "pow_acc":  pow_acc,
            "pow_app":  pow_app,
            "pow_arg":  pow_arg,
            "pow_putt": pow_putt,
            "r2": r2,
            "n_obs": n_obs,
            "n_events": int(df_c["event_id"].nunique()),
            "n_players": int(df_c["dg_id"].nunique()),
            "target_season": int(target_season),
        })

    course_profiles = pd.DataFrame(profiles).sort_values("course_num").reset_index(drop=True)
    if course_profiles.empty:
        raise ValueError("No course profiles were created (min_obs_per_course too high or missing coverage).")

    # 6) scale pow_* -> imp_* (choose scaling)
    pow_cols = ["pow_dist","pow_acc","pow_app","pow_arg","pow_putt"]

    if course_scale == "player_minmax":
        course_profiles = _scale_0_1_like_players(course_profiles, pow_cols)
    elif course_scale == "percentile":
        course_profiles = _scale_0_1_percentile(course_profiles, pow_cols, lo_q=0.05, hi_q=0.95)
    else:
        raise ValueError("course_scale must be 'player_minmax' or 'percentile'")

    course_profiles = course_profiles.rename(columns={
        "pow_dist": "imp_dist",
        "pow_acc":  "imp_acc",
        "pow_app":  "imp_app",
        "pow_arg":  "imp_arg",
        "pow_putt": "imp_putt",
    })

    return course_profiles, skills

def build_and_save_course_fit_and_skills_with_targets(
    target_season: int,
    target_courses: pd.DataFrame,
    cfg: CourseFitConfig = CourseFitConfig(),
) -> None:
    """
    Convenience wrapper for preseason / OAD use:
      - loads rounds
      - builds course profiles ONLY for target_courses
      - builds player skills from historical PGA rounds
      - saves both to disk
    """
    combined = load_rounds()

    course_profiles, skills = build_course_fit_5attr(
        combined=combined,
        target_season=target_season,
        cfg=cfg,
        target_courses=target_courses,
    )

    course_path = COURSE_FIT_TEMPLATE.with_name(
        str(COURSE_FIT_TEMPLATE.name).format(season=target_season)
    )
    course_path.parent.mkdir(parents=True, exist_ok=True)
    course_profiles.to_csv(course_path, index=False)

    skills_path = PLAYER_SKILL_TEMPLATE.with_name(
        str(PLAYER_SKILL_TEMPLATE.name).format(season=target_season)
    )
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    skills.to_csv(skills_path, index=False)

    print(f"Requested courses: {len(target_courses)}")
    print(f"Profiles created:  {len(course_profiles)}")
    print(f"Saved course profiles to: {course_path}")
    print(f"Saved player skills to:   {skills_path}")
