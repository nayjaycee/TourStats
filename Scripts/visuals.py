from __future__ import annotations

from typing import Iterable, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Scripts.data_io import load_rounds


# -----------------------------
# Helper: subset weekly tables
# -----------------------------

def subset_weekly_for_players(
    weekly: Dict[str, pd.DataFrame],
    dg_ids: Iterable[int],
) -> Dict[str, pd.DataFrame]:
    """
    Given the full `weekly` dict from build_weekly_view and a list of dg_ids,
    return slices of the key tables for just those players.

    Returns a dict with:
      - performance
      - event_history
      - ytd
      - summary

    (Also keeps the old keys table_performance / table_event_history / table_ytd
     for backwards compatibility.)
    """
    ids = [int(x) for x in dg_ids]

    mapping = {
        "performance": "table_performance",
        "event_history": "table_event_history",
        "ytd": "table_ytd",
        "summary": "summary",
    }

    out: Dict[str, pd.DataFrame] = {}

    for short_key, weekly_key in mapping.items():
        df = weekly.get(weekly_key)
        if df is None or isinstance(df, pd.DataFrame) and df.empty or (
            isinstance(df, pd.DataFrame) and "dg_id" not in df.columns and short_key != "summary"
        ):
            out[short_key] = pd.DataFrame()
            # also keep the original key empty for safety
            out[weekly_key] = pd.DataFrame()
            continue

        if short_key == "summary":
            sub = df.copy()
        else:
            sub = df[df["dg_id"].isin(ids)].copy()

        sub = sub.reset_index(drop=True)
        out[short_key] = sub
        out[weekly_key] = sub  # backwards-compatible

    return out


# -----------------------------
# Visual 1: recent form (SG windows)
# -----------------------------

def plot_recent_form_sg(
    perf_slice: pd.DataFrame,
    windows: List[int] = (40, 24, 12),
    stat: str = "sg_total",
):
    """
    For each player in perf_slice, plot their SG stat (e.g. sg_total)
    over the specified windows (e.g. L40 / L24 / L12).

    x-axis: window (e.g. 40, 24, 12)
    y-axis: SG value
    One line per player.
    """
    if perf_slice.empty:
        print("No data in performance slice.")
        return

    cols = [f"{stat}_L{w}" for w in windows]
    missing = [c for c in cols if c not in perf_slice.columns]
    if missing:
        print(f"Missing columns for stat '{stat}': {missing}")
        return

    fig, ax = plt.subplots()
    x = np.arange(len(windows))

    for _, row in perf_slice.iterrows():
        y = [row.get(f"{stat}_L{w}", np.nan) for w in windows]
        label = f"{row.get('player_name', 'dg_' + str(int(row['dg_id'])))} ({int(row['dg_id'])})"
        ax.plot(x, y, marker="o", label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{w}" for w in windows])
    ax.set_xlabel("Window (rounds)")
    ax.set_ylabel(f"{stat} (strokes per round)")
    ax.set_title(f"Recent {stat} by window")
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Visual 2: SG profile (ball-striking vs putting)
# -----------------------------

def plot_sg_component_profile(
    perf_slice: pd.DataFrame,
    window: int = 24,
):
    """
    For each player, show a bar profile of SG components at a given window:

        sg_ott_L{window}, sg_app_L{window}, sg_arg_L{window}, sg_putt_L{window}

    Bars grouped by player.
    """
    if perf_slice.empty:
        print("No data in performance slice.")
        return

    comps = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    cols = [f"{c}_L{window}" for c in comps]
    missing = [c for c in cols if c not in perf_slice.columns]
    if missing:
        print(f"Missing columns for SG components at L{window}: {missing}")
        return

    n_players = len(perf_slice)
    x = np.arange(len(comps))
    width = 0.8 / max(n_players, 1)

    fig, ax = plt.subplots()

    for i, (_, row) in enumerate(perf_slice.iterrows()):
        values = [row.get(f"{c}_L{window}", np.nan) for c in comps]
        offset = (i - (n_players - 1) / 2) * width
        xpos = x + offset
        label = f"{row.get('player_name', 'dg_' + str(int(row['dg_id'])))} ({int(row['dg_id'])})"
        ax.bar(xpos, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(["OTT", "APP", "ARG", "PUTT"])
    ax.set_xlabel(f"Components (L{window})")
    ax.set_ylabel("Strokes gained per round")
    ax.set_title(f"SG component profile (L{window})")
    ax.axhline(0.0, linewidth=1)
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Visual 3: YTD profile
# -----------------------------

def plot_ytd_profile(
    ytd_slice: pd.DataFrame,
):
    """
    For each player, show a YTD profile:

      - ytd_starts
      - ytd_made_cut_pct
      - ytd_top25
      - ytd_top5
      - ytd_wins

    Bars grouped by metric, with one bar per player per metric.
    """
    if ytd_slice.empty:
        print("No data in YTD slice.")
        return

    metrics = ["ytd_starts", "ytd_made_cut_pct", "ytd_top25", "ytd_top5", "ytd_wins"]
    metrics = [m for m in metrics if m in ytd_slice.columns]
    if not metrics:
        print("No YTD metrics available in slice.")
        return

    n_players = len(ytd_slice)
    x = np.arange(len(metrics))
    width = 0.8 / max(n_players, 1)

    fig, ax = plt.subplots()

    for i, (_, row) in enumerate(ytd_slice.iterrows()):
        values = [row.get(m, np.nan) for m in metrics]
        offset = (i - (n_players - 1) / 2) * width
        xpos = x + offset
        label = f"{row.get('player_name', 'dg_' + str(int(row['dg_id'])))} ({int(row['dg_id'])})"
        ax.bar(xpos, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(["starts", "MC%", "T25", "T5", "wins"][: len(metrics)])
    ax.set_xlabel("YTD metrics")
    ax.set_ylabel("Value (counts or proportion)")
    ax.set_title("YTD profile")
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Visual 4: EV current vs EV future max (+ optional course fit)
# -----------------------------

def plot_ev_current_vs_future_max(
    perf_slice: pd.DataFrame,
    show_course_fit: bool = True,
) -> pd.DataFrame:
    """
    For each player in perf_slice, show:

      - ev_current
      - ev_future_max   (best single future event)
      - (optional) course_fit_score on secondary axis

    No ev_future_total here on purpose.

    Returns the small comparison table used to build the chart.
    """
    if perf_slice.empty:
        print("No data in performance slice.")
        return perf_slice

    needed = ["dg_id", "ev_current", "ev_future_max"]
    missing = [c for c in needed if c not in perf_slice.columns]
    if missing:
        print("Missing EV columns:", missing)
        return perf_slice

    df = perf_slice.copy()
    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

    # make sure numeric
    for col in ["ev_current", "ev_future_max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # compute pct if not there
    if "ev_current_vs_future_max_pct" not in df.columns:
        denom = df["ev_future_max"].replace(0, np.nan)
        df["ev_current_vs_future_max_pct"] = df["ev_current"] / denom

    # sort by ev_current descending by default
    df = df.sort_values("ev_current", ascending=False)

    players = df["dg_id"].astype(int).tolist()
    x = np.arange(len(players))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.bar(x - width / 2, df["ev_current"], width=width, label="EV current")
    ax1.bar(x + width / 2, df["ev_future_max"], width=width, label="EV future max")
    ax1.set_ylabel("EV (winner-share units)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in players])
    ax1.set_title("EV current vs EV future max")

    # annotate burned % on top of current bar
    for i, (_, row) in enumerate(df.iterrows()):
        pct = row["ev_current_vs_future_max_pct"]
        if pd.notna(pct) and np.isfinite(pct):
            ax1.text(
                i - width / 2,
                row["ev_current"] * 1.02,
                f"{pct * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # optional course fit on secondary axis
    if show_course_fit and "course_fit_score" in df.columns:
        cfit = pd.to_numeric(df["course_fit_score"], errors="coerce")
        if cfit.notna().any():
            ax2 = ax1.twinx()
            ax2.plot(x, cfit, marker="o", linestyle="--", label="course_fit_score")
            ax2.set_ylabel("Course fit (z-score)")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        else:
            ax1.legend(loc="upper left")
    else:
        ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    cols_out = [
        "dg_id",
        "decimal_odds",
        "ev_current",
        "ev_future_max",
        "ev_current_vs_future_max_pct",
        "course_fit_score",
    ]
    cols_out = [c for c in cols_out if c in df.columns]
    return df[cols_out]


# -----------------------------
# Visual 5: Course vs player radar using 5-attr skills
# -----------------------------

def _build_course_vector_5attr(course_row: pd.Series) -> Dict[str, float]:
    """
    Take a course_fit row with imp_dist/imp_acc/imp_app/imp_arg/imp_putt
    and return a normalized 5-attr vector on 0–1 scale.
    """
    raw = {
        "Dist": float(course_row.get("imp_dist", 0.0) or 0.0),
        "Acc":  float(course_row.get("imp_acc",  0.0) or 0.0),
        "App":  float(course_row.get("imp_app",  0.0) or 0.0),
        "Arg":  float(course_row.get("imp_arg",  0.0) or 0.0),
        "Putt": float(course_row.get("imp_putt", 0.0) or 0.0),
    }
    vals = np.array(list(raw.values()), dtype=float)
    if np.allclose(vals, 0.0):
        vals = np.ones_like(vals)
    vals = vals / vals.max()
    return dict(zip(raw.keys(), vals))


def _build_player_vectors_5attr(
    player_skills_df: pd.DataFrame,
    dg_ids: Iterable[int],
) -> Dict[int, Dict[str, float]]:
    """
    From player_skill_5attr table, build per-dg_id normalized 5-attr skills:

      skill_dist, skill_acc, skill_app, skill_arg, skill_putt

    Each attribute is min–max scaled across the *entire* skills table so
    the 0–1 scale is meaningful across players.
    """
    skills = player_skills_df.copy()
    skills["dg_id"] = pd.to_numeric(skills["dg_id"], errors="coerce").astype("Int64")

    attr_map = {
        "Dist": "skill_dist",
        "Acc":  "skill_acc",
        "App":  "skill_app",
        "Arg":  "skill_arg",
        "Putt": "skill_putt",
    }

    # global min/max per attribute
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for label, col in attr_map.items():
        if col not in skills.columns:
            raise KeyError(f"Column '{col}' missing from player_skills_df.")
        x = pd.to_numeric(skills[col], errors="coerce")
        mins[label] = float(x.min())
        maxs[label] = float(x.max())

    profiles: Dict[int, Dict[str, float]] = {}
    for dg in [int(x) for x in dg_ids]:
        row = skills[skills["dg_id"] == dg]
        if row.empty:
            continue
        row = row.iloc[0]
        prof = {}
        for label, col in attr_map.items():
            val = float(row[col])
            lo, hi = mins[label], maxs[label]
            if hi > lo:
                scaled = (val - lo) / (hi - lo)
            else:
                scaled = 0.5
            prof[label] = scaled
        profiles[dg] = prof

    return profiles

def plot_course_vs_players_radar_from_skills(
    weekly: dict,
    course_fit_df: pd.DataFrame,
    player_skills_df: pd.DataFrame,
    dg_ids,
    label_players_with_names: bool = True,
) -> None:
    """
    Radar chart comparing course profile vs player 5-attr skill profiles.

    Key differences from the previous version:
      - Course profile uses the *global* scale of imp_* across all courses,
        NOT re-scaled so that this course's max = 1.
      - Player profiles are min–max scaled using the full player_skills_df
        (season-wide), NOT re-scaled per-event.

    Axes (in DataGolf-like order): DIST, ACC, APP, ARG, PUTT
    Radial axis: 0–1 (same scale for course + players across all events).
    """
    dg_ids = [int(x) for x in dg_ids]

    # --- event / course context ---
    sched_row = weekly["schedule_row"].iloc[0]
    course_num = int(sched_row["course_num"])
    course_name = str(sched_row.get("course_name", f"course {course_num}"))
    event_name = str(sched_row.get("event_name", ""))

    # --- course profile (global scale) ---
    cf = course_fit_df.copy()
    cf["course_num"] = pd.to_numeric(cf["course_num"], errors="coerce").astype("Int64")

    row = cf[cf["course_num"] == course_num]
    if row.empty:
        print(f"No course_fit row found for course_num={course_num}")
        return
    row = row.iloc[0]

    imp_cols = ["imp_dist", "imp_acc", "imp_app", "imp_arg", "imp_putt"]
    for c in imp_cols:
        if c not in cf.columns:
            raise KeyError(f"Column '{c}' missing from course_fit_df.")

    # Global max over all courses / attributes so 1.0 is fixed tour-wide
    global_max_imp = float(cf[imp_cols].to_numpy().max())
    if not np.isfinite(global_max_imp) or global_max_imp <= 0:
        global_max_imp = 1.0  # defensive fallback

    course_vals_raw = np.array(
        [
            float(row.get("imp_dist", 0.0) or 0.0),
            float(row.get("imp_acc", 0.0) or 0.0),
            float(row.get("imp_app", 0.0) or 0.0),
            float(row.get("imp_arg", 0.0) or 0.0),
            float(row.get("imp_putt", 0.0) or 0.0),
        ],
        dtype=float,
    )
    # Put on 0–1 scale using GLOBAL max, not per-course max
    course_vals = np.clip(course_vals_raw / global_max_imp, 0.0, 1.0)

    # --- player 5-attr profiles (global scale from player_skills_df) ---
    skills = player_skills_df.copy()
    skills["dg_id"] = pd.to_numeric(skills["dg_id"], errors="coerce").astype("Int64")

    attr_map = {
        "DIST": "skill_dist",
        "ACC": "skill_acc",
        "APP": "skill_app",
        "ARG": "skill_arg",
        "PUTT": "skill_putt",
    }

    for col in attr_map.values():
        if col not in skills.columns:
            raise KeyError(f"Column '{col}' missing from player_skills_df.")

    mins = {}
    maxs = {}
    for label, col in attr_map.items():
        x = pd.to_numeric(skills[col], errors="coerce")
        mins[label] = float(x.min())
        maxs[label] = float(x.max())

    player_profiles: dict[int, dict[str, float]] = {}
    for dg in dg_ids:
        r = skills[skills["dg_id"] == dg]
        if r.empty:
            continue
        r = r.iloc[0]

        prof = {}
        for label, col in attr_map.items():
            val = float(r.get(col, np.nan))
            lo, hi = mins[label], maxs[label]
            if not np.isfinite(val) or hi <= lo:
                prof[label] = np.nan
            else:
                prof[label] = (val - lo) / (hi - lo)  # global 0–1 scale

        player_profiles[dg] = prof

    if not player_profiles:
        print("No matching dg_ids found in player_skills_df.")
        return

    # --- build radar plot ---
    axes_labels = ["DIST", "ACC", "APP", "ARG", "PUTT"]
    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # course polygon
    course_vals_closed = np.concatenate([course_vals, course_vals[:1]])
    ax.plot(angles, course_vals_closed, linestyle="--", linewidth=2, label="Course")
    ax.fill(angles, course_vals_closed, alpha=0.15)

    # players
    field_df = weekly.get("field")
    for dg, prof in player_profiles.items():
        vals = [prof[label] for label in axes_labels]
        vals = np.array(vals, dtype=float)
        if np.isnan(vals).all():
            continue
        vals_closed = np.concatenate([vals, vals[:1]])

        # try to use player_name if available
        label = f"dg_id {dg}"
        if label_players_with_names and isinstance(field_df, pd.DataFrame):
            if "dg_id" in field_df.columns and "player_name" in field_df.columns:
                r = field_df[field_df["dg_id"] == dg]
                if not r.empty:
                    label = f"{r.iloc[0]['player_name']} ({dg})"

        ax.plot(angles, vals_closed, linewidth=2, label=label)
        ax.fill(angles, vals_closed, alpha=0.10)

    # cosmetics
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])

    title_parts = [course_name]
    if event_name:
        title_parts.append(f"({event_name})")
    ax.set_title(" / ".join(title_parts), pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    plt.show()

def print_event_history_for_player(
    weekly: dict,
    dg_id: int,
    min_year: int = 2017,
) -> None:
    """
    Print historical event results (pre-current-season) for a given dg_id.

    Pulls from combined rounds (load_rounds) and:
      - filters to this event_id
      - filters to years < season_year
      - aggregates per-year finish + SG
    """

    sched_row = weekly["schedule_row"].iloc[0]
    event_id = int(sched_row["event_id"])
    season_year = int(sched_row["year"])

    rounds = load_rounds().copy()

    # enforce types
    for col in ["year", "event_id", "dg_id", "finish_num"]:
        if col in rounds.columns:
            rounds[col] = pd.to_numeric(rounds[col], errors="coerce")

    rounds["dg_id"] = rounds["dg_id"].astype("Int64")

    mask = (
        (rounds["event_id"] == event_id)
        & (rounds["dg_id"] == int(dg_id))
        & (rounds["year"] >= min_year)
        & (rounds["year"] < season_year)   # exclude the season I'm running
    )
    hist = rounds[mask].copy()

    if hist.empty:
        print(
            f"No historical rounds for dg_id {dg_id} at event_id {event_id} "
            f"before {season_year} (years >= {min_year})."
        )
        return

    hist["sg_total"] = pd.to_numeric(hist["sg_total"], errors="coerce")

    per_year = (
        hist.groupby(["year"], as_index=False)
        .agg(
            sg_event=("sg_total", "sum"),
            finish_num=("finish_num", "min"),
            fin_text=("fin_text", "first"),
            event_name=("event_name", "first"),
        )
        .sort_values("year")
    )

    name = (
        per_year["event_name"].dropna().iloc[-1]
        if "event_name" in per_year.columns and not per_year["event_name"].isna().all()
        else f"event_id {event_id}"
    )

    print(
        f"Event history for dg_id {dg_id} at {name} "
        f"(event_id {event_id}), years {min_year}–{season_year-1}"
    )
    display(per_year[["year", "finish_num", "fin_text", "sg_event"]])

    n_starts = len(per_year)
    best_finish = per_year["finish_num"].min()
    top10 = (per_year["finish_num"] <= 10).sum()
    top25 = (per_year["finish_num"] <= 25).sum()
    wins = (per_year["finish_num"] == 1).sum()

    print("\nSummary:")
    print(f"Starts: {n_starts}")
    print(f"Best finish: {best_finish}")
    print(f"Top-10s: {top10}")
    print(f"Top-25s: {top25}")
    print(f"Wins: {wins}")
