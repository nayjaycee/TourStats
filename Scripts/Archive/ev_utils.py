from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from Scripts.Archive.odds_results import load_odds_and_results
from Scripts.Archive.oad_schedule import load_oad_for_season

# NEW imports for weekly EV engine
try:
    from .stats_engine import compute_rolling_stats_for_field
    from .odds_results import add_implied_prob, compute_ev_from_purse
except ImportError:
    from Scripts.Archive.stats_engine import compute_rolling_stats_for_field
    from Scripts.Archive.odds_results import add_implied_prob, compute_ev_from_purse


# ============================================================
# ODDS NORMALIZATION
# ============================================================

def _get_decimal_odds_column(df: pd.DataFrame) -> str:
    """
    Return the odds column name.

    We *require* 'close_odds'. No guessing, no heuristics.
    """
    if "close_odds" not in df.columns:
        raise ValueError("Expected column 'close_odds' in odds/results data.")
    return "close_odds"


def add_decimal_odds_and_implied_prob(
    odds_df: pd.DataFrame,
    decimal_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Ensure that odds_df has:
      - a numeric decimal odds column: decimal_odds_std
      - an implied probability column: implied_prob

    We treat 'close_odds' as the decimal odds column unless you override it.
    """
    df = odds_df.copy()

    # Decide which column to use for decimal odds
    if decimal_col is None:
        decimal_col = _get_decimal_odds_column(df)

    # Normalize to numeric
    df[decimal_col] = pd.to_numeric(df[decimal_col], errors="coerce")

    # Standardized name
    df["decimal_odds_std"] = df[decimal_col]

    # Basic sanity: require > 1 (decimal odds of 1.0 or less are nonsense)
    mask_valid = df["decimal_odds_std"] > 1.0
    if not mask_valid.any():
        raise ValueError(
            f"Decimal odds in column '{decimal_col}' look invalid (<= 1.0 or NaN). "
            f"Check Odds_and_Results.xlsx."
        )

    # Implied probability (no overround adjustment)
    df["implied_prob"] = 1.0 / df["decimal_odds_std"]

    return df, decimal_col


# ============================================================
# EV CALCULATION
# ============================================================

def add_ev_for_events(
    season: int,
    odds_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For a single season (2024 or 2025):

      1. Filter Odds_and_Results to that year.
      2. Normalize odds -> decimal_odds_std + implied_prob.
      3. Load the OAD calendar for that season (OAD_2024.xlsx or OAD_2025.xlsx).
      4. Merge on (year, event_id).
      5. Compute EV per player-event = implied_prob * purse.

    Assumptions:
      - Odds_and_Results has columns:
          year, event_id, player_name, dg_id, close_odds, ...
      - OAD file has columns:
          start_date, event_id_fixed, purse, ...

    Output: DataFrame with all odds columns plus:
      - decimal_odds_std
      - implied_prob
      - purse (from OAD)
      - ev_raw  (implied_prob * purse)
    """
    if season not in (2024, 2025):
        raise ValueError("add_ev_for_events: season must be 2024 or 2025.")

    # 1) Load odds + filter to season
    if odds_df is None:
        odds_df = load_odds_and_results(copy=True)

    df = odds_df.copy()
    if "year" not in df.columns:
        raise ValueError("Odds_and_Results must contain a 'year' column.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"] == season].copy()

    if df.empty:
        raise ValueError(f"No odds rows found for season {season} in Odds_and_Results.")

    # Ensure event_id is numeric
    if "event_id" not in df.columns:
        raise ValueError("Odds_and_Results must contain an 'event_id' column.")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")

    # 2) Normalize odds
    df_norm, _ = add_decimal_odds_and_implied_prob(df)

    # 3) Load OAD calendar for this season (from Excel)
    oad = load_oad_for_season(season)

    # 4) Primary merge: (year, event_id)
    merged = df_norm.merge(
        oad[["year", "event_id", "event_name", "purse"]],
        on=["year", "event_id"],
        how="left",
        suffixes=("", "_oad"),
    )

    # 4b) Fallback patch: if purse is NaN, try matching by (year, event_name)
    # This is mainly a diagnostic to reveal ID mismatches.
    missing_mask = merged["purse"].isna()
    if missing_mask.any():
        # Build a simple (year, event_name_norm) merge on the OAD side
        oad_fallback = oad.copy()
        oad_fallback["event_name_norm"] = (
            oad_fallback["event_name"].astype(str).str.strip().str.lower()
        )

        merged["event_name_norm"] = (
            merged["event_name"].astype(str).str.strip().str.lower()
        )

        fb = merged[missing_mask].merge(
            oad_fallback[["year", "event_name_norm", "purse"]],
            on=["year", "event_name_norm"],
            how="left",
            suffixes=("", "_fb"),
        )

        # Where fallback purse is available, use it
        patched = merged.copy()
        patched.loc[missing_mask, "purse_fb"] = fb["purse_fb"].values

        use_fb = patched["purse"].isna() & patched["purse_fb"].notna()
        if use_fb.any():
            patched.loc[use_fb, "purse"] = patched.loc[use_fb, "purse_fb"]

        # Optional: print which events still have missing purse
        still_missing = patched["purse"].isna()
        if still_missing.any():
            debug_events = (
                patched.loc[still_missing, ["year", "event_id", "event_name"]]
                .drop_duplicates()
                .sort_values(["year", "event_id"])
            )
            print("EV WARNING: These events still have no purse after fallback merge:")
            print(debug_events.to_string(index=False))

        merged = patched.drop(columns=["event_name_norm", "purse_fb"], errors="ignore")

    # 5) Compute EV
    if "purse" not in merged.columns:
        merged["purse"] = np.nan

    merged["ev_raw"] = merged["implied_prob"] * merged["purse"]

    return merged


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Running ev_utils.py self-test...")

    try:
        for yr in (2024, 2025):
            print(f"\n=== Season {yr} ===")
            ev_df = add_ev_for_events(season=yr)

            print("Columns in EV dataframe:")
            print(ev_df.columns.tolist())

            cols_to_show = [
                c
                for c in [
                    "year",
                    "event_id",
                    "event_name",
                    "decimal_odds_std",
                    "implied_prob",
                    "purse",
                    "ev_raw",
                ]
                if c in ev_df.columns
            ]

            print("\nSample EV rows:")
            print(ev_df[cols_to_show].head(20))

            print("\nEV summary (non-null):")
            print(ev_df["ev_raw"].dropna().describe())

    except Exception as e:
        print("Self-test failed:", e)


        # ============================================================
        # WEEKLY EV ENGINE
        # ============================================================

        def _summarize_event_history(
                master: pd.DataFrame,
                event_id: int,
        ) -> pd.DataFrame:
            """
            Per-player history at THIS event across all years.

            Uses existing flag columns if present:
              win, top_5, top_10, top_25, made_cut
            and only falls back to finish_position logic if needed.
            """
            df = master.copy()

            if "event_id" not in df.columns:
                raise ValueError("_summarize_event_history: master must contain 'event_id'.")

            df = df[df["event_id"] == int(event_id)].copy()
            if df.empty:
                return df.iloc[0:0].copy()

            # Normalize ids / basics
            for col in ["dg_id", "season", "year"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            # Standardize column names we care about
            # (if you ran load_odds_and_results these are already lowercased)
            col_map = {c.lower(): c for c in df.columns}

            def get(col_lower: str) -> Optional[str]:
                return col_map.get(col_lower)

            # Main numeric fields
            fin_col = get("finish_num") or get("finish_position")
            if fin_col:
                df[fin_col] = pd.to_numeric(df[fin_col], errors="coerce")
            sg_col = get("sg_total")
            if sg_col:
                df[sg_col] = pd.to_numeric(df[sg_col], errors="coerce")
            score_col = get("round_score")
            if score_col:
                df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

            # Use existing flags if present
            def coerce_flag(col_lower: str) -> Optional[str]:
                c = get(col_lower)
                if c is None:
                    return None
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                return c

            win_flag_col = coerce_flag("win")
            top5_flag_col = coerce_flag("top_5")
            top10_flag_col = coerce_flag("top_10")
            top25_flag_col = coerce_flag("top_25")
            made_flag_col = coerce_flag("made_cut")

            # If we don’t have flags, derive from finish number
            if fin_col and made_flag_col is None:
                df["made_cut_flag"] = df[fin_col].notna().astype(int)
                made_flag_col = "made_cut_flag"

            if fin_col and top25_flag_col is None:
                df["top_25_flag"] = df[fin_col].between(1, 25, inclusive="both").astype(int)
                top25_flag_col = "top_25_flag"

            if fin_col and top10_flag_col is None:
                df["top_10_flag"] = df[fin_col].between(1, 10, inclusive="both").astype(int)
                top10_flag_col = "top_10_flag"

            if fin_col and top5_flag_col is None:
                df["top_5_flag"] = df[fin_col].between(1, 5, inclusive="both").astype(int)
                top5_flag_col = "top_5_flag"

            if fin_col and win_flag_col is None:
                df["win_flag"] = (df[fin_col] == 1).astype(int)
                win_flag_col = "win_flag"

            # Sort chronologically for "previous" result
            date_col = get("event_completed") or get("event_date")
            sort_cols = ["dg_id"]
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                sort_cols.append(date_col)
            elif get("season"):
                sort_cols.append(get("season"))
            df = df.sort_values(sort_cols, ascending=True)

            def _agg(g: pd.DataFrame) -> pd.Series:
                starts = len(g)

                def s(col: Optional[str]) -> int:
                    return int(g[col].sum()) if col and col in g.columns else 0

                made = s(made_flag_col)
                out = {
                    "dg_id": g["dg_id"].iloc[0],
                    "event_starts": starts,
                    "event_made_cuts": made,
                    "event_cut_pct": made / starts if starts > 0 else np.nan,
                    "event_top25": s(top25_flag_col),
                    "event_top10": s(top10_flag_col),
                    "event_top5": s(top5_flag_col),
                    "event_wins": s(win_flag_col),
                    "event_avg_round_score": g[score_col].mean(skipna=True) if score_col else np.nan,
                    "event_avg_sg_total": g[sg_col].mean(skipna=True) if sg_col else np.nan,
                }

                last = g.iloc[-1]
                out["event_prev_finish_position"] = (
                    last[fin_col] if fin_col and fin_col in last.index else np.nan
                )
                out["event_prev_round_score"] = (
                    last[score_col] if score_col and score_col in last.index else np.nan
                )

                return pd.Series(out)

            hist = (
                df.groupby("dg_id", group_keys=False)
                .apply(_agg)
                .reset_index(drop=True)
            )
            return hist


        def _summarize_ytd_history(
                master: pd.DataFrame,
                season: int,
                as_of_date: pd.Timestamp,
        ) -> pd.DataFrame:
            """
            Per-player YTD form in a given season up to as_of_date.

            Uses existing flags (win, top_5, top_10, top_25, made_cut) if present.
            """
            df = master.copy()

            # Season filter
            col_map = {c.lower(): c for c in df.columns}
            season_col = col_map.get("season") or col_map.get("year")
            if season_col is None:
                raise ValueError("_summarize_ytd_history: master must have 'season' or 'year'.")

            df[season_col] = pd.to_numeric(df[season_col], errors="coerce").astype("Int64")
            df = df[df[season_col] == int(season)]

            # Date cutoff
            date_col = col_map.get("event_completed") or col_map.get("event_date")
            if date_col is None:
                raise ValueError("_summarize_ytd_history: master must have event_completed or event_date.")
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[df[date_col] < as_of_date].copy()

            if df.empty:
                return df.iloc[0:0].copy()

            df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")

            fin_col = col_map.get("finish_num") or col_map.get("finish_position")
            if fin_col:
                df[fin_col] = pd.to_numeric(df[fin_col], errors="coerce")
            sg_col = col_map.get("sg_total")
            if sg_col:
                df[sg_col] = pd.to_numeric(df[sg_col], errors="coerce")
            score_col = col_map.get("round_score")
            if score_col:
                df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

            def coerce_flag(col_lower: str) -> Optional[str]:
                c = col_map.get(col_lower)
                if c is None:
                    return None
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                return c

            win_flag_col = coerce_flag("win")
            top5_flag_col = coerce_flag("top_5")
            top10_flag_col = coerce_flag("top_10")
            top25_flag_col = coerce_flag("top_25")
            made_flag_col = coerce_flag("made_cut")

            # Fallback from finish numbers if needed
            if fin_col and made_flag_col is None:
                df["made_cut_flag"] = df[fin_col].notna().astype(int)
                made_flag_col = "made_cut_flag"
            if fin_col and top25_flag_col is None:
                df["top_25_flag"] = df[fin_col].between(1, 25, inclusive="both").astype(int)
                top25_flag_col = "top_25_flag"
            if fin_col and top10_flag_col is None:
                df["top_10_flag"] = df[fin_col].between(1, 10, inclusive="both").astype(int)
                top10_flag_col = "top_10_flag"
            if fin_col and top5_flag_col is None:
                df["top_5_flag"] = df[fin_col].between(1, 5, inclusive="both").astype(int)
                top5_flag_col = "top_5_flag"
            if fin_col and win_flag_col is None:
                df["win_flag"] = (df[fin_col] == 1).astype(int)
                win_flag_col = "win_flag"

            def _agg(g: pd.DataFrame) -> pd.Series:
                starts = len(g)

                def s(col: Optional[str]) -> int:
                    return int(g[col].sum()) if col and col in g.columns else 0

                made = s(made_flag_col)

                return pd.Series(
                    {
                        "dg_id": g["dg_id"].iloc[0],
                        "ytd_starts": starts,
                        "ytd_made_cuts": made,
                        "ytd_cut_pct": made / starts if starts > 0 else np.nan,
                        "ytd_top25": s(top25_flag_col),
                        "ytd_top10": s(top10_flag_col),
                        "ytd_top5": s(top5_flag_col),
                        "ytd_wins": s(win_flag_col),
                        "ytd_avg_round_score": g[score_col].mean(skipna=True) if score_col else np.nan,
                        "ytd_avg_sg_total": g[sg_col].mean(skipna=True) if sg_col else np.nan,
                    }
                )

            ytd = (
                df.groupby("dg_id", group_keys=False)
                .apply(_agg)
                .reset_index(drop=True)
            )
            return ytd

        def _compute_course_fit_score(
                week_df: pd.DataFrame,
                course_fit: Optional[pd.DataFrame],
                course_num: Optional[int],
        ) -> pd.DataFrame:
            """
            Compute a simple course-fit score per player by:
              - taking L40 stats (dist, acc, sg_app, sg_arg, sg_putt)
              - z-scoring across the week field
              - multiplying by course importance weights (imp_*)

            Requires week_df to already contain:
              l40_driving_dist, l40_driving_acc,
              l40_sg_app, l40_sg_arg, l40_sg_putt.

            If course_fit or course_num is missing, just returns week_df with
            course_fit_score = NaN.
            """
            df = week_df.copy()
            df["course_fit_score"] = np.nan

            if course_fit is None or course_num is None:
                return df

            cf_row = course_fit[course_fit["course_num"] == int(course_num)]
            if cf_row.empty:
                return df

            cf_row = cf_row.iloc[0]
            weights = {
                "dist": cf_row.get("imp_dist", np.nan),
                "acc": cf_row.get("imp_acc", np.nan),
                "app": cf_row.get("imp_app", np.nan),
                "arg": cf_row.get("imp_arg", np.nan),
                "putt": cf_row.get("imp_putt", np.nan),
            }

            # If all NaN, nothing to do
            if all(pd.isna(v) for v in weights.values()):
                return df

            # Z-score the L40 stats within the field
            stat_map = {
                "dist": "l40_driving_dist",
                "acc": "l40_driving_acc",
                "app": "l40_sg_app",
                "arg": "l40_sg_arg",
                "putt": "l40_sg_putt",
            }

            for key, col in stat_map.items():
                if col not in df.columns:
                    df[col] = np.nan
                vals = pd.to_numeric(df[col], errors="coerce")
                mu = vals.mean(skipna=True)
                sd = vals.std(ddof=0, skipna=True)
                if sd == 0 or np.isnan(sd):
                    df[f"{col}_z"] = 0.0
                else:
                    df[f"{col}_z"] = (vals - mu) / sd

            score = 0.0
            for key, w in weights.items():
                if pd.isna(w):
                    continue
                colz = f"{stat_map[key]}_z"
                score += w * df[colz]

            df["course_fit_score"] = score

            # Drop intermediate z-columns
            drop_cols = [c for c in df.columns if c.endswith("_z")]
            df = df.drop(columns=drop_cols, errors="ignore")

            return df


        def build_week_ev_table(
                env: Dict[str, pd.DataFrame],
                odds: pd.DataFrame,
                master: pd.DataFrame,
                week_index: int,
                shortlist_ids: Optional[Sequence[int]] = None,
                exclude_used: bool = False,
                used_ids: Optional[Sequence[int]] = None,
                top_k: int = 30,
                include_future_ev: bool = True,  # placeholder; currently not used
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """
            Build the detailed EV table for a single week/event.

            env must contain at least:
              - 'sched': schedule with columns
                  ['season', 'week_index', 'event_id', 'event_name',
                   'event_date' or 'start_date', 'tour', 'purse', 'course_num']
              - optionally 'course_fit': course-fit importance table.

            odds is expected to be *win* market odds with at least:
              - 'season' or 'year'
              - 'event_id'
              - 'dg_id'
              - 'decimal_odds' (or a column we can treat as such)
              - 'event_tier' (optional; used later for future EV logic)

            master is the season master results with:
              - 'season' or 'year'
              - 'event_id', 'dg_id'
              - finish_position, sg_total, round_score, etc. where available.

            Returns:
              week_full : detailed DataFrame (one row per player in field)
              week_view : slimmed-down view for decision-making
            """
            sched = env.get("sched")
            if sched is None:
                raise ValueError("build_week_ev_table: env must contain a 'sched' DataFrame.")

            course_fit = env.get("course_fit")

            # ---- identify week/event row ----
            wk_rows = sched[sched["week_index"] == week_index]
            if wk_rows.empty:
                raise ValueError(f"build_week_ev_table: no schedule rows for week_index={week_index}.")

            if len(wk_rows) > 1:
                # For now, just take the first; if you have multiple tours per week
                # we can extend this to handle multiple events simultaneously.
                wk = wk_rows.iloc[0]
            else:
                wk = wk_rows.iloc[0]

            season = int(wk.get("season") or wk.get("year"))
            event_id = int(wk["event_id"])
            event_name = str(wk["event_name"])
            tour = str(wk.get("tour", ""))
            course_num = int(wk["course_num"]) if "course_num" in wk.index and not pd.isna(wk["course_num"]) else None

            # event date / as-of date
            as_of = None
            for cand in ("event_date", "start_date", "event_completed"):
                if cand in wk.index:
                    as_of = pd.to_datetime(wk[cand])
                    break
            if as_of is None or pd.isna(as_of):
                raise ValueError("build_week_ev_table: schedule must have event_date/start_date/event_completed.")

            # ---- odds for THIS event ----
            df_odds = odds.copy()

            # normalize season/year column
            season_col = "year" if "year" in df_odds.columns else ("season" if "season" in df_odds.columns else None)
            if season_col is None:
                raise ValueError("build_week_ev_table: odds must contain 'year' or 'season'.")

            df_odds[season_col] = pd.to_numeric(df_odds[season_col], errors="coerce").astype("Int64")

            if season_col is None:
                raise ValueError("build_week_ev_table: odds must contain 'season' or 'year'.")

            df_odds["event_id"] = pd.to_numeric(df_odds["event_id"], errors="coerce").astype("Int64")
            df_odds["dg_id"] = pd.to_numeric(df_odds["dg_id"], errors="coerce").astype("Int64")

            df_odds_evt = df_odds[(df_odds[season_col] == season) & (df_odds["event_id"] == event_id)].copy()
            if df_odds_evt.empty:
                raise ValueError(
                    f"build_week_ev_table: no odds rows found for season={season}, event_id={event_id}."
                )

            # Ensure decimal odds + implied_prob + EV
            if "decimal_odds" not in df_odds_evt.columns:
                # Re-use close_odds if present or raise; this is stricter than your
                # big combined file but that's intentional for OAD
                if "close_odds" in df_odds_evt.columns:
                    df_odds_evt["decimal_odds"] = pd.to_numeric(
                        df_odds_evt["close_odds"], errors="coerce"
                    )
                else:
                    raise ValueError(
                        "build_week_ev_table: expected 'decimal_odds' or 'close_odds' in odds for event."
                    )

            df_odds_evt = add_implied_prob(df_odds_evt)

            # attach purse from schedule if not present
            if "purse" not in df_odds_evt.columns or df_odds_evt["purse"].isna().all():
                df_odds_evt["purse"] = float(wk["purse"])

            df_odds_evt = compute_ev_from_purse(df_odds_evt, purse_col="purse")
            df_odds_evt = df_odds_evt.rename(columns={"ev": "ev_current"})

            # Restrict to shortlist if provided
            if shortlist_ids:
                shortlist_ids = [int(x) for x in shortlist_ids]
                df_odds_evt = df_odds_evt[df_odds_evt["dg_id"].isin(shortlist_ids)]

            # Exclude already-used players if requested
            if exclude_used and used_ids:
                used_ids = [int(x) for x in used_ids]
                df_odds_evt = df_odds_evt[~df_odds_evt["dg_id"].isin(used_ids)]

            if df_odds_evt.empty:
                raise ValueError(
                    f"build_week_ev_table: no candidates left after shortlist/used filters "
                    f"for week_index={week_index}."
                )

            # ---- rolling stats (L40 / L24 / L12) ----
            field_ids = (
                df_odds_evt["dg_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            rolling = compute_rolling_stats_for_field(
                combined_rounds=None,  # will load combined internally
                as_of_date=as_of,
                dg_ids=field_ids,
            )

            # ---- event-specific history ----
            event_hist = _summarize_event_history(master=master, event_id=event_id)

            # ---- YTD history ----
            ytd_hist = _summarize_ytd_history(master=master, season=season, as_of_date=as_of)

            # ---- base merge: odds + rolling + histories ----
            base_cols = [
                "dg_id",
                "decimal_odds",
                "implied_prob",
                "purse",
                "ev_current",
            ]

            week = df_odds_evt.merge(
                rolling,
                on="dg_id",
                how="left",
                suffixes=("", "_roll"),
            )

            week = week.merge(event_hist, on="dg_id", how="left")
            week = week.merge(ytd_hist, on="dg_id", how="left")

            # ---- course-fit score ----
            week = _compute_course_fit_score(
                week_df=week,
                course_fit=course_fit,
                course_num=course_num,
            )

            # ---- future EV placeholder ----
            # TODO: implement your full future-EV logic based on:
            #   - event_tier, player category (top / mid / LIV)
            #   - participation heuristics (prior majors/signature starts)
            #   - latest odds by tier from odds_results.get_latest_odds_by_player_tier
            week["ev_future_total"] = np.nan

            # For now, define ev_total = ev_current (no future component yet)
            week["ev_total"] = week["ev_current"]

            # Metadata
            week["year"] = season
            week["week_index"] = week_index
            week["event_id"] = event_id
            week["event_name"] = event_name
            week["tour"] = tour
            week["start_date"] = as_of

            # ---- ordering / trimming ----
            week_full = week.copy()

            # A leaner view for decision-making
            view_cols = []
            for col in [
                "year",
                "week_index",
                "event_id",
                "event_name",
                "tour",
                "start_date",
                "dg_id",
                "decimal_odds",
                "implied_prob",
                "purse",
                "ev_current",
                "ev_future_total",
                "ev_total",
                "course_fit_score",
                "l40_rounds",
                "l40_sg_total",
                "l24_sg_total",
                "l12_sg_total",
                "l40_driving_dist",
                "l40_driving_acc",
                "event_starts",
                "event_made_cuts",
                "event_cut_pct",
                "event_top25",
                "event_top10",
                "event_top5",
                "event_wins",
                "event_prev_finish_position",
                "ytd_starts",
                "ytd_made_cuts",
                "ytd_cut_pct",
                "ytd_top25",
                "ytd_top10",
                "ytd_top5",
                "ytd_wins",
                "ytd_avg_sg_total",
            ]:
                if col in week_full.columns:
                    view_cols.append(col)

            week_view = (
                week_full[view_cols]
                .sort_values("ev_total", ascending=False)
                .reset_index(drop=True)
            )

            # Limit to top_k in the view if requested (but keep full detail in week_full)
            if top_k is not None and top_k > 0 and len(week_view) > top_k:
                week_view = week_view.head(top_k).copy()

            return week_full, week_view