# course_fit_5attr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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


def build_course_fit_5attr(
    combined: pd.DataFrame,
    target_season: int,
    cfg: CourseFitConfig = CourseFitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function: computes:

    1) course_profiles: DG-style 5-attribute course importances for courses
       used in `target_season`.
    2) player_skills: per-player 5-attribute exp-decayed skills based on
       historical PGA rounds prior to target_season.

    Returns
    -------
    (course_profiles, player_skills)
    """

    df = combined.copy()

    df["tour"] = df["tour"].astype(str).str.upper()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")
    df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")

    for col in [
        "sg_app",
        "sg_arg",
        "sg_putt",
        "sg_total",
        "driving_dist",
        "driving_acc",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Required column '{col}' not found in combined DataFrame.")

    # Historical PGA rounds before target_season
    hist = df[(df["tour"] == "PGA") & (df["season"] < target_season)].copy()
    if hist.empty:
        raise ValueError(f"No historical PGA rounds with season < {target_season}.")

    attr_cols = [
        "driving_dist",
        "driving_acc",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "sg_total",
        "dg_id",
        "event_completed",
    ]
    hist = hist.dropna(subset=attr_cols)

    hist = (
        hist.sort_values(
            ["dg_id", "event_completed", "event_id", "round_num"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )

    # 1) player skills
    skills = _compute_player_skills(hist, cfg.max_rounds, cfg.min_rounds)
    skill_cols = ["skill_dist", "skill_acc", "skill_app", "skill_arg", "skill_putt"]
    if skills.empty:
        raise ValueError("No players with valid 5-attribute skills for course fit.")

    # 2) event-level performance merged with skills
    event_perf = (
        hist.groupby(
            ["event_id", "event_name", "course_num", "course_name", "dg_id"],
            as_index=False,
        ).agg(event_sg_total=("sg_total", "sum"))
    )

    event_perf = event_perf.merge(
        skills[["dg_id"] + skill_cols], on="dg_id", how="inner"
    )
    if event_perf.empty:
        raise ValueError("No event-player rows after merging skills; check data coverage.")

    event_perf["event_sg_total_centered"] = (
        event_perf["event_sg_total"]
        - event_perf.groupby(["event_id", "course_num"])["event_sg_total"].transform(
            "mean"
        )
    )

    # 3) global z-scores for skills
    z_map = {
        "skill_dist": "z_dist",
        "skill_acc": "z_acc",
        "skill_app": "z_app",
        "skill_arg": "z_arg",
        "skill_putt": "z_putt",
    }
    for col, zname in z_map.items():
        mu = event_perf[col].mean()
        sd = event_perf[col].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            event_perf[zname] = 0.0
        else:
            event_perf[zname] = (event_perf[col] - mu) / sd

    # 4) target-season PGA courses
    courses_target = (
        df[(df["tour"] == "PGA") & (df["season"] == target_season)]
        .loc[:, ["course_num", "course_name"]]
        .dropna(subset=["course_num"])
        .drop_duplicates()
        .sort_values("course_num")
    )
    if courses_target.empty:
        raise ValueError(f"No PGA courses found for season {target_season}.")

    profiles = []
    for _, row in courses_target.iterrows():
        cnum = row["course_num"]
        cname = row["course_name"]

        df_c = event_perf[event_perf["course_num"] == cnum].copy()
        if len(df_c) < cfg.min_obs_per_course:
            continue

        df_c = df_c.dropna(
            subset=[
                "event_sg_total_centered",
                "z_dist",
                "z_acc",
                "z_app",
                "z_arg",
                "z_putt",
            ]
        )
        if len(df_c) < cfg.min_obs_per_course:
            continue

        X = df_c[["z_dist", "z_acc", "z_app", "z_arg", "z_putt"]].values
        y = df_c["event_sg_total_centered"].values

        model = LinearRegression()
        model.fit(X, y)
        coefs = model.coef_

        abs_coefs = np.abs(coefs)
        s = abs_coefs.sum()
        if s == 0:
            imp_dist = imp_acc = imp_app = imp_arg = imp_putt = np.nan
        else:
            imp_vals = abs_coefs / s
            imp_dist, imp_acc, imp_app, imp_arg, imp_putt = imp_vals.tolist()

        n_obs = len(df_c)
        shrink = n_obs / (n_obs + cfg.shrink_k)
        imp_dist *= shrink
        imp_acc *= shrink
        imp_app *= shrink
        imp_arg *= shrink
        imp_putt *= shrink

        profiles.append(
            {
                "course_num": cnum,
                "course_name": cname,
                "imp_dist": imp_dist,
                "imp_acc": imp_acc,
                "imp_app": imp_app,
                "imp_arg": imp_arg,
                "imp_putt": imp_putt,
                "n_obs": n_obs,
                "n_events": df_c["event_id"].nunique(),
                "n_players": df_c["dg_id"].nunique(),
                "target_season": target_season,
            }
        )

    course_profiles = pd.DataFrame(profiles).sort_values("course_num").reset_index(
        drop=True
    )
    return course_profiles, skills


def build_and_save_course_fit_and_skills(target_season: int) -> None:
    """
    Convenience wrapper:
      - loads rounds
      - builds course profiles & player skills
      - saves them to disk under the standard filenames.
    """
    combined = load_rounds()
    course_profiles, skills = build_course_fit_5attr(combined, target_season)

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

    print(f"Saved course profiles to: {course_path}")
    print(f"Saved player skills to:   {skills_path}")