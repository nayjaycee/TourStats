# Scripts/league.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeagueCutoff:
    current_event_id: int
    current_event_order: int
    through_event_order: int  # current - 1


def build_event_order_map(schedule_df: pd.DataFrame) -> Dict[int, int]:
    """
    Build event_id -> event_order.

    Preference:
      1) event_order (authoritative if present)
      2) week_num (legacy alt ordering)
      3) start_date (fallback)
    """
    if "event_id" not in schedule_df.columns:
        raise ValueError("schedule_df must contain 'event_id'")

    s = schedule_df.copy()
    s["event_id"] = pd.to_numeric(s["event_id"], errors="coerce")
    s = s.dropna(subset=["event_id"]).copy()
    s["event_id"] = s["event_id"].astype(int)

    # 1) preferred: event_order
    if "event_order" in s.columns:
        s["event_order"] = pd.to_numeric(s["event_order"], errors="coerce")
        s = s.dropna(subset=["event_order"]).copy()
        s["event_order"] = s["event_order"].astype(int)

        s = (
            s.sort_values(["event_order", "event_id"])
             .drop_duplicates(subset=["event_id"], keep="first")
             .reset_index(drop=True)
        )
        return dict(zip(s["event_id"], s["event_order"]))

    # 2) next: week_num
    if "week_num" in s.columns:
        s["week_num"] = pd.to_numeric(s["week_num"], errors="coerce")
        s = s.dropna(subset=["week_num"]).copy()
        s["week_num"] = s["week_num"].astype(int)

        s = (
            s.sort_values(["week_num", "event_id"])
             .drop_duplicates(subset=["event_id"], keep="first")
             .reset_index(drop=True)
        )
        return dict(zip(s["event_id"], s["week_num"]))

    # 3) fallback: start_date
    if "start_date" not in s.columns:
        raise ValueError("schedule_df must contain 'event_order', 'week_num', or 'start_date'")

    s["start_date"] = pd.to_datetime(s["start_date"], errors="coerce")
    s = s.dropna(subset=["start_date"]).copy()

    s = (
        s.sort_values("start_date")
         .drop_duplicates(subset=["event_id"], keep="first")
         .reset_index(drop=True)
    )
    s["event_order"] = np.arange(1, len(s) + 1, dtype=int)
    return dict(zip(s["event_id"], s["event_order"]))



def compute_cutoff(order_map: Dict[int, int], current_event_id: int) -> Optional[LeagueCutoff]:
    cur_order = order_map.get(int(current_event_id))
    if cur_order is None:
        return None
    through = max(int(cur_order) - 1, 0)
    return LeagueCutoff(
        current_event_id=int(current_event_id),
        current_event_order=int(cur_order),
        through_event_order=int(through),
    )


def _detect_raw_is_cumulative(per_entry: pd.DataFrame) -> bool:
    """
    Detection at ENTRY grain:
    If most entries have non-decreasing raw_winnings over >=3 events,
    treat it as already-cumulative.

    NOTE: This expects per_entry to have exactly one value per (entry_id, event_order)
          representing "raw_winnings" for that event_order.
    """
    if per_entry["event_order"].nunique() < 3:
        return False

    tmp = per_entry.sort_values(["entry_id", "event_order"]).copy()

    def pct_nondecreasing(x: pd.Series) -> float:
        v = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy()
        if len(v) < 3:
            return 0.0
        return float(np.mean(np.diff(v) >= -1e-9))

    scores = (
        tmp.groupby("entry_id")["raw_winnings"]
           .apply(pct_nondecreasing)
           .reset_index(name="pct")
    )
    return float(scores["pct"].median()) >= 0.90


def build_league_standings_through_prior(
    league_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    current_event_id: int,
    *,
    username_you: str = "You",
    top_n: int = 7,
    raw_is_cumulative: Optional[bool] = None,  # None = auto-detect
) -> Tuple[pd.DataFrame, Optional[LeagueCutoff], Optional[bool]]:
    """
    ENTRY-based standings through the PRIOR event.

    Returns:
      out (long df): entry_id, username, label, event_order, raw_winnings (delta), cum_winnings
      cutoff
      raw_is_cumulative_used (bool or None if cutoff None)
    """
    need = {"league_id", "entry_id", "username", "event_id", "raw_winnings"}
    missing = need - set(league_df.columns)
    if missing:
        raise ValueError(f"league_df missing required columns: {sorted(list(missing))}")

    order_map = build_event_order_map(schedule_df)
    cutoff = compute_cutoff(order_map, current_event_id=current_event_id)
    if cutoff is None:
        return pd.DataFrame(), None, None
    if cutoff.through_event_order <= 0:
        return pd.DataFrame(), cutoff, None

    df = league_df.copy()
    df["entry_id"] = df["entry_id"].astype(str)
    df["username"] = df["username"].astype(str)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["raw_winnings"] = pd.to_numeric(df["raw_winnings"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["event_id"]).copy()
    df["event_id"] = df["event_id"].astype(int)

    # map to event_order
    df["event_order"] = df["event_id"].map(order_map)
    df = df.dropna(subset=["event_order"]).copy()
    df["event_order"] = df["event_order"].astype(int)

    # through prior event only
    df = df[df["event_order"] <= int(cutoff.through_event_order)].copy()

    # ---- dedupe safely: compute BOTH max and sum, decide later ----
    # If raw is cumulative: duplicates are usually repeats -> max is correct.
    # If raw is per-event: duplicates could be split rows -> sum is safer.
    agg = (
        df.groupby(["entry_id", "username", "event_id", "event_order"], as_index=False)
          .agg(
              raw_max=("raw_winnings", "max"),
              raw_sum=("raw_winnings", "sum"),
          )
    )

    # decide if raw is already cumulative (entry-level) using raw_max (safer signal)
    if raw_is_cumulative is None:
        raw_is_cumulative_used = _detect_raw_is_cumulative(
            agg.rename(columns={"raw_max": "raw_winnings"})[["entry_id", "event_order", "raw_winnings"]]
        )
    else:
        raw_is_cumulative_used = bool(raw_is_cumulative)

    # choose which deduped raw to use
    agg["raw_winnings"] = agg["raw_max"] if raw_is_cumulative_used else agg["raw_sum"]
    df = agg[["entry_id", "username", "event_id", "event_order", "raw_winnings"]].copy()

    # build full grid so lines don't break
    max_x = int(cutoff.through_event_order)
    all_orders = pd.DataFrame({"event_order": np.arange(1, max_x + 1, dtype=int)})

    entries = df[["entry_id", "username"]].drop_duplicates().copy()
    grid = entries.assign(_k=1).merge(all_orders.assign(_k=1), on="_k").drop(columns=["_k"])

    out = grid.merge(df[["entry_id", "event_order", "raw_winnings"]], on=["entry_id", "event_order"], how="left")
    out["raw_winnings"] = out["raw_winnings"].fillna(np.nan)  # keep NaN until we interpret
    out = out.sort_values(["entry_id", "event_order"]).reset_index(drop=True)

    if raw_is_cumulative_used:
        # raw_winnings is season-to-date: forward fill -> cum, then diff -> per-event delta
        out["cum_winnings"] = (
            pd.to_numeric(out["raw_winnings"], errors="coerce")
              .groupby(out["entry_id"])
              .ffill()
              .fillna(0.0)
        )
        out["raw_winnings"] = out.groupby("entry_id")["cum_winnings"].diff().fillna(out["cum_winnings"])
    else:
        # raw_winnings is per-event: fill missing with 0 then cumsum
        out["raw_winnings"] = pd.to_numeric(out["raw_winnings"], errors="coerce").fillna(0.0)
        out["cum_winnings"] = out.groupby("entry_id")["raw_winnings"].cumsum()

    # build a readable label: username if unique, else username + short entry_id
    name_counts = entries.groupby("username")["entry_id"].nunique()
    multi = set(name_counts[name_counts > 1].index.tolist())

    def _label(u: str, eid: str) -> str:
        return f"{u} ({eid[-4:]})" if u in multi else u

    out["label"] = [_label(u, eid) for u, eid in zip(out["username"], out["entry_id"])]

    # rank top N entries at cutoff week; always keep all of "You" username’s entries
    cut_week = int(cutoff.through_event_order)
    at_cut = out[out["event_order"] == cut_week][["entry_id", "username", "label", "cum_winnings"]].copy()
    at_cut = at_cut.sort_values("cum_winnings", ascending=False)

    keep_entries = set(at_cut.head(int(top_n))["entry_id"].tolist())
    keep_entries |= set(at_cut.loc[at_cut["username"] == str(username_you), "entry_id"].tolist())

    out = out[out["entry_id"].isin(keep_entries)].copy()
    out = out.sort_values(["cum_winnings", "label", "event_order"], ascending=[False, True, True])

    return out, cutoff, raw_is_cumulative_used