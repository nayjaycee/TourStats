# Scripts/notebook_support.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Set, Dict, Any, Optional, List

import numpy as np
import pandas as pd


# -----------------------------
# Display + warnings
# -----------------------------
def configure_notebook_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.display.float_format = "{:,.2f}".format


# -----------------------------
# Paths / config
# -----------------------------
@dataclass(frozen=True)
class OADPaths:
    project_root: Path
    picks_log_path: Path
    picks_path: Path
    course_imp_path: Path


def build_default_paths(project_root: Path, season_year: int) -> OADPaths:
    project_root = Path(project_root)

    picks_log_path = project_root / "Data" / "in Use" / f"OAD_{season_year}_manual_picks.csv"
    picks_path = project_root / "Data" / "in Use" / f"OAD_{season_year}_manual_picks.csv"
    course_imp_path = project_root / "Data" / "in Use" / f"course_fit_{season_year}_dg_style_5attr.csv"

    return OADPaths(
        project_root=project_root,
        picks_log_path=picks_log_path,
        picks_path=picks_path,
        course_imp_path=course_imp_path,
    )


def load_course_importance(course_imp_path: Path) -> pd.DataFrame:
    return pd.read_csv(course_imp_path)


# -----------------------------
# Used-player filtering
# -----------------------------
def get_used_dg_ids_from_picks(picks_path: Path) -> Set[int]:
    if not Path(picks_path).exists():
        return set()

    df = pd.read_csv(picks_path)
    if "dg_id" not in df.columns:
        return set()

    used = (
        pd.to_numeric(df["dg_id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    return set(used)


def apply_used_filter_to_weekly(weekly: Dict[str, Any], picks_path: Path) -> Tuple[Dict[str, Any], Set[int]]:
    used = get_used_dg_ids_from_picks(picks_path)
    weekly["used_dg_ids"] = used

    if not used:
        return weekly, used

    table_keys = [
        "table_performance",
        "table_performance_top50",
        "table_event_history",
        "table_event_history_top50",
        "table_ytd",
        "table_ytd_top50",
        "pattern_candidates",
    ]

    for key in table_keys:
        df = weekly.get(key)
        if isinstance(df, pd.DataFrame) and "dg_id" in df.columns:
            # be defensive: Int64 can have <NA>
            dg = pd.to_numeric(df["dg_id"], errors="coerce")
            mask = ~dg.fillna(-1).astype(int).isin(used)
            weekly[key] = df.loc[mask].copy()

    return weekly, used


# -----------------------------
# Condensed table + top-table filters
# -----------------------------
def build_condensed_table(weekly: Dict[str, Any]) -> pd.DataFrame:
    if "summary" not in weekly or not isinstance(weekly["summary"], pd.DataFrame):
        raise KeyError("weekly['summary'] missing or not a DataFrame")

    summary = weekly["summary"].copy()

    if "dg_id" in summary.columns:
        summary["dg_id"] = pd.to_numeric(summary["dg_id"], errors="coerce").astype("Int64")

    used_ids = weekly.get("used_dg_ids", set())
    if used_ids:
        used_ids = {int(x) for x in used_ids}
        dg = pd.to_numeric(summary["dg_id"], errors="coerce").fillna(-1).astype(int)
        summary = summary.loc[~dg.isin(used_ids)].copy()

    cols = [
        "dg_id",
        "is_shortlist",
        "decision_context",
        "course_type",
        "decision_score",
        "pct_ytd_avg_sg_total",
        "pct_ytd_made_cut_pct",
        "pct_event_hist_sg",
        "pct_course_hist_sg",
        "final_rank_score",
        "oad_score",
        "pct_sg_total_L12",
        "pct_ev_current_adj",
        "pct_oad_score",
        "decimal_odds",
        "ev_current_adj",
        "ev_future_max",
        "ev_current_to_future_max_ratio",
        "sg_total_L40",
        "sg_total_L24",
        "sg_total_L12",
        "starts_event",
        "made_cut_pct_event",
        "top25_event",
        "top10_event",
        "top5_event",
        "wins_event",
        "prev_finish_num_event",
        "ytd_starts",
        "ytd_made_cut_pct",
        "ytd_top25",
        "ytd_top10",
        "ytd_top5",
        "ytd_wins",
        "ytd_avg_sg_total",
    ]
    cols = [c for c in cols if c in summary.columns]
    out = summary[cols].copy()

    # robust sort: prefer oad_score, then ev_current_adj, then ev_current
    for candidate in ["decision_score", "final_rank_score", "oad_score", "ev_current_adj", "ev_current"]:
        if candidate in out.columns:
            out = out.sort_values(candidate, ascending=False)
            break

    return out.reset_index(drop=True)


def filter_top_tables_by_ids(weekly: Dict[str, Any], dg_ids: Iterable[int]) -> Dict[str, Any]:
    ids = {int(x) for x in dg_ids}
    out = weekly.copy()

    for key in ["table_performance_top50", "table_event_history_top50", "table_ytd_top50"]:
        df = out.get(key)
        if isinstance(df, pd.DataFrame) and "dg_id" in df.columns:
            dg = pd.to_numeric(df["dg_id"], errors="coerce").fillna(-1).astype(int)
            out[key] = df.loc[dg.isin(ids)].copy()

    return out


# -----------------------------
# Picks log utilities
# -----------------------------
PICKS_LOG_COLUMNS = [
    "year",
    "event_id",
    "event_name",
    "dg_id",
    "player_name",
    "decimal_odds",
    "ev_current",
    "ev_future_total",
    "ev_current_pct_of_future",
    "finish_num",
    "finish_text",
    "winnings",
]


def load_picks_log(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=PICKS_LOG_COLUMNS)


def log_pick(
    season_year: int,
    event_id: int,
    dg_id: int,
    weekly_raw: Dict[str, Any],     # must be unfiltered weekly
    odds_df: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    path = Path(path)
    log_df = load_picks_log(path)

    if "schedule_row" not in weekly_raw or not isinstance(weekly_raw["schedule_row"], pd.DataFrame):
        raise KeyError("weekly_raw['schedule_row'] missing or not a DataFrame")

    event_name = weekly_raw["schedule_row"].iloc[0]["event_name"]

    perf = weekly_raw.get("table_performance")
    if not isinstance(perf, pd.DataFrame) or "dg_id" not in perf.columns:
        raise KeyError("weekly_raw['table_performance'] missing or malformed")

    row = perf.loc[perf["dg_id"] == dg_id]
    if row.empty:
        raise ValueError(f"dg_id {dg_id} not found in table_performance.")
    row = row.iloc[0]

    # player name (if available)
    player_name = None
    field_df = weekly_raw.get("field")
    if isinstance(field_df, pd.DataFrame) and {"dg_id", "player_name"}.issubset(field_df.columns):
        r = field_df.loc[field_df["dg_id"] == dg_id]
        if not r.empty:
            player_name = r.iloc[0]["player_name"]

    # actual result from odds/results
    mask = (
        (odds_df["year"] == season_year) &
        (odds_df["event_id"] == event_id) &
        (odds_df["dg_id"] == dg_id)
    )
    res = odds_df.loc[mask]
    if res.empty:
        finish_num = np.nan
        finish_text = None
        winnings = 0.0
    else:
        r0 = res.iloc[0]
        finish_num = r0.get("finish_num", np.nan)
        finish_text = r0.get("finish_text", None)
        winnings = r0.get("Winnings", 0.0)

    new_row = {
        "year": season_year,
        "event_id": event_id,
        "event_name": event_name,
        "dg_id": dg_id,
        "player_name": player_name,
        "decimal_odds": row.get("decimal_odds", np.nan),
        "ev_current": row.get("ev_current", np.nan),
        "ev_future_total": row.get("ev_future_total", np.nan),
        "ev_current_pct_of_future": row.get("ev_current_pct_of_future", np.nan),
        "finish_num": finish_num,
        "finish_text": finish_text,
        "winnings": winnings,
    }

    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    log_df.to_csv(path, index=False)
    return log_df


def inspect_picks_log(path: Path) -> pd.DataFrame:
    df = load_picks_log(path)
    if df.empty:
        print("Picks log is empty.")
        return df

    df = df.copy()
    df["cum_winnings"] = df["winnings"].cumsum()

    print("Manual picks log:")
    display(df)

    print("\nCumulative winnings:")
    display(df[["event_name", "player_name", "winnings", "cum_winnings"]])

    return df


def preview_pick_result(picks_log: pd.DataFrame) -> None:
    if picks_log.empty:
        print("No picks logged yet.")
        return

    df = picks_log.copy()
    df["cum_winnings"] = df["winnings"].cumsum()

    row = df.iloc[-1]
    player = row.get("player_name", "")
    finish = row.get("finish_num", np.nan)
    winnings = row.get("winnings", 0.0)
    cum = row.get("cum_winnings", winnings)

    print("Latest Pick Summary:")
    print("---------------------")
    print(f"Player:        {player} (dg_id {int(row['dg_id'])})")
    print(f"Finish:        {finish}")
    print(f"Winnings:      ${winnings:,.0f}")
    print(f"Cumulative:    ${cum:,.0f}")


# -----------------------------
# Maintenance: add names to picks
# -----------------------------
def add_player_names_to_picks_file(
    picks_path: Path,
    rounds_df: pd.DataFrame,
    overwrite: bool = True,
) -> pd.DataFrame:
    picks_path = Path(picks_path)
    picks = pd.read_csv(picks_path)

    if "dg_id" not in picks.columns:
        raise ValueError("picks file missing dg_id column")

    name_map = (
        rounds_df[["dg_id", "player_name"]]
        .dropna(subset=["dg_id", "player_name"])
        .drop_duplicates()
        .copy()
    )

    name_map["dg_id"] = pd.to_numeric(name_map["dg_id"], errors="coerce").astype("Int64")
    picks["dg_id"] = pd.to_numeric(picks["dg_id"], errors="coerce").astype("Int64")

    out = picks.merge(name_map, on="dg_id", how="left")

    if overwrite:
        out.to_csv(picks_path, index=False)

    return out

from typing import Optional, Iterable
import pandas as pd

def print_preseason_reports(
    season: int,
    longlist: pd.DataFrame,
    baseline_df: pd.DataFrame,
    big_hist_df: pd.DataFrame,
    render_preseason_player_report,
    dg_ids: Optional[Iterable[int]] = None,
    max_n: Optional[int] = None,
) -> None:
    """
    Prints the preseason report for a list of dg_ids.
    - If dg_ids is None: uses longlist['dg_id']
    - max_n optionally limits number printed
    """
    if dg_ids is None:
        dg_ids = longlist["dg_id"].dropna().astype(int).tolist()
    else:
        dg_ids = [int(x) for x in dg_ids]

    if max_n is not None:
        dg_ids = dg_ids[: int(max_n)]

    for dg in dg_ids:
        print("=" * 80)
        print(
            render_preseason_player_report(
                season=season,
                dg_id=dg,
                longlist=longlist,
                baseline_df=baseline_df,
                big_hist_df=big_hist_df,
            )
        )
        print("\n")