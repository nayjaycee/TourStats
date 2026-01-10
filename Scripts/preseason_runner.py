# Scripts/preseason_runner.py (recommended) or append to Scripts/preseason.py

from __future__ import annotations
from typing import Iterable, Optional
import pandas as pd

from Scripts.preseason import (
    build_preseason_package,
    compute_preseason_baseline_windows_for_longlist,
    render_preseason_player_report,
)
from Scripts.data_io import load_rounds
from Scripts.schedule import build_season_schedule


def run_preseason_reports(
    season_year: int,
    top_n: int = 30,
    windows: tuple[int, ...] = (40, 24, 12),
    dg_ids: Optional[Iterable[int]] = None,
    display_longlist: bool = True,
) -> dict:
    """
    Reproduces your exact working notebook preseason flow:
      1) build_preseason_package
      2) compute_preseason_baseline_windows_for_longlist
      3) print render_preseason_player_report for each dg_id

    Returns dict with longlist, big_hist_df, baseline_df, rounds_df, sched
    so the notebook can keep using them.
    """

    # 1) package
    pre_pkg = build_preseason_package(season_year, top_n=top_n)
    longlist = pre_pkg["longlist"]
    big_hist_df = pre_pkg["major_sig_history"]

    if display_longlist:
        print("Preseason long list:")
        display(longlist)

    # 2) baselines as of first OAD event
    rounds_df = load_rounds()
    sched = build_season_schedule(season_year)
    baseline_df = compute_preseason_baseline_windows_for_longlist(
        season_year,
        longlist,
        windows=windows,
    )

    # choose who to print
    if dg_ids is None:
        dg_list = longlist["dg_id"].dropna().astype(int).tolist()
    else:
        dg_list = [int(x) for x in dg_ids]

    # 3) reports
    for dg in dg_list:
        print("=" * 80)
        print(
            render_preseason_player_report(
                season=season_year,
                dg_id=dg,
                longlist=longlist,
                baseline_df=baseline_df,
                big_hist_df=big_hist_df,
            )
        )
        print("\n")

    return {
        "longlist": longlist,
        "big_hist_df": big_hist_df,
        "baseline_df": baseline_df,
        "rounds_df": rounds_df,
        "sched": sched,
    }
