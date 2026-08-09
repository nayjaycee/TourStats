"""
pick5_optimizer.py
==================
Multi-stage assignment optimizer for the FedEx Playoffs "5 & Done" contest.

Rules:
  - Pick exactly 5 golfers per event (15 total across 3 events).
  - Each golfer used at most once across all 3 events.
  - Points = earnings × event multiplier (TC is 0.5x).
  - No 36-hole cuts — every player in field earns money. Only a WD scores zero.

Events:
  FedEx St. Jude   (event_id=27) — Top 70 FedEx Cup  — $20M — 1.0x
  BMW Championship (event_id=28) — Top 50 after FSJ   — $20M — 1.0x
  TOUR Championship (event_id=60) — Top 30 after BMW  — $40M — 0.5x

Priorities (per brief):
  1. Name matcher — exact, not fuzzy. Fails loud on unresolved names.
  2. Payout tables — verified anchors only; rest interpolated and flagged.
  3. Simulation — path-dependent sequence (FSJ → BMW → TC), N≥10,000 sims.
  4. Optimizer — ILP (PuLP) for exact solve; greedy for frequency mode.
  5. What-if harness — lock/exclude, shadow prices, EV gap.

TODO: Rename metric1/metric2/metric3 in standings CSV once user confirms meanings.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pulp

ROOT  = Path(__file__).parent.parent
INUSE = ROOT / "Data" / "in Use"
OAD   = ROOT / "Data" / "OAD"

STANDINGS_PATH    = OAD   / "fedexcup_standings.csv"
ALL_PLAYERS_PATH  = INUSE / "All_players.xlsx"
ROUNDS_PATH       = INUSE / "combined_roundlevel_2024_present.csv"

# ── Event config ───────────────────────────────────────────────────────────────
EVENT_IDS    = [27, 28, 60]
EVENT_NAMES  = {27: "FedEx St. Jude", 28: "BMW Championship", 60: "TOUR Championship"}
EVENT_PURSE  = {27: 20_000_000, 28: 20_000_000, 60: 40_000_000}
EVENT_MULT   = {27: 1.0, 28: 1.0, 60: 0.5}
EVENT_FIELD  = {27: 70, 28: 50, 60: 30}   # max field size per event
EVENT_IDX    = {27: 0, 28: 1, 60: 2}       # internal array index

# ── TODO: rename once user confirms ────────────────────────────────────────────
# metric1 range: -0.43 to 2.38 (looks like SG total / round)
# metric2 range:  8.08 to 90.1 (unknown — could be win% or points-per-event)
# metric3 range: -0.21 to 2.83 (looks like another SG composite)
METRIC_COLS = {
    "metric1": "metric1",   # TODO: rename to real column meaning
    "metric2": "metric2",   # TODO: rename to real column meaning
    "metric3": "metric3",   # TODO: rename to real column meaning
}

# ── Course skill weights per event (brief: FSJ approach-heavy; BMW ball-striking) ──
COURSE_WEIGHTS = {
    27: {"sg_total": 0.35, "sg_app": 0.35, "sg_t2g": 0.20, "sg_ott": 0.10},  # TPC Southwind
    28: {"sg_total": 0.50, "sg_t2g": 0.35, "sg_app": 0.15},                   # Bellerive (no recent data)
    60: {"sg_total": 0.70, "sg_app": 0.20, "sg_t2g": 0.10},                   # East Lake
}

# ── Simulation constants ───────────────────────────────────────────────────────
ROUND_SIGMA      = 2.8    # typical round-to-round SG std dev for PGA Tour player
N_ROUNDS_PER_EV  = 4
DEFAULT_N_SIMS   = 10_000

# ── Manual dg_id overrides for names that don't resolve automatically ──────────
_FORCED_DG_IDS: dict[str, int] = {
    # standings name → dg_id
    "Johnny KEEFER":       32457,   # rounds data has "Keefer, John"
}

# Alias map: standings name → canonical "Last, First" spelling to force-match
_ALIAS_MAP: dict[str, str] = {
    # Add one-off fixes here as needed
    # "First LAST": "Last, First",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Name matching
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_accents(s: str) -> str:
    """Remove diacritical marks: Åberg→Aberg, Højgaard→Hojgaard."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _norm_name(s: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    s = _strip_accents(str(s))
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _standings_to_key(raw: str) -> str:
    """
    Convert standings format 'First LAST' (or 'First Middle LAST') to lookup key.
    Uses rfind(' ') so multi-word first names (Si Woo, Min Woo) are handled correctly.
    Key format: 'last first' (normalized).
    """
    raw = raw.strip()
    idx = raw.rfind(" ")
    if idx == -1:
        return _norm_name(raw)
    last  = raw[idx + 1:]
    first = raw[:idx]
    return f"{_norm_name(last)} {_norm_name(first)}"


def _players_to_key(player_name: str) -> str:
    """Convert 'Last, First' (our system) to 'last first' lookup key."""
    parts = player_name.split(",", 1)
    if len(parts) == 2:
        return f"{_norm_name(parts[0])} {_norm_name(parts[1])}"
    return _norm_name(player_name)


def build_name_lookup(
    all_players_path: Path = ALL_PLAYERS_PATH,
    rounds_path: Path = ROUNDS_PATH,
) -> dict[str, tuple[int, str]]:
    """
    Build key → (dg_id, canonical_name) lookup from All_players.xlsx,
    augmented with any names found in the rounds data but missing from All_players.
    """
    lookup: dict[str, tuple[int, str]] = {}

    # Primary: All_players.xlsx
    if all_players_path.exists():
        ap = pd.read_excel(all_players_path)
        for _, r in ap.iterrows():
            key = _players_to_key(str(r["player_name"]))
            lookup[key] = (int(r["dg_id"]), str(r["player_name"]))

    # Augment: rounds data (for players not in All_players)
    if rounds_path.exists():
        rdf = pd.read_csv(rounds_path, usecols=["dg_id", "player_name"], low_memory=False)
        rdf = rdf.dropna(subset=["dg_id", "player_name"]).drop_duplicates("dg_id")
        for _, r in rdf.iterrows():
            key = _players_to_key(str(r["player_name"]))
            if key not in lookup:
                lookup[key] = (int(r["dg_id"]), str(r["player_name"]))

    return lookup


def load_standings(
    standings_path: Path = STANDINGS_PATH,
    lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load fedexcup_standings.csv and resolve each player to a dg_id.

    Raises ValueError listing all unresolved names so nothing is silently dropped.
    The known collisions (Fitzpatrick ×2, Kim ×3) are handled by full-name matching.
    """
    if lookup is None:
        lookup = build_name_lookup()

    df = pd.read_csv(standings_path)
    df.columns = [c.strip() for c in df.columns]

    dg_ids   = []
    can_names = []
    unresolved = []

    for _, row in df.iterrows():
        raw = str(row["player"])

        # 1. Check forced overrides first
        if raw in _FORCED_DG_IDS:
            dg_ids.append(_FORCED_DG_IDS[raw])
            can_names.append(raw)
            continue

        # 2. Check alias map
        alias_name = _ALIAS_MAP.get(raw)
        if alias_name:
            key = _players_to_key(alias_name)
        else:
            key = _standings_to_key(raw)

        if key in lookup:
            did, cname = lookup[key]
            dg_ids.append(did)
            can_names.append(cname)
        else:
            unresolved.append(raw)
            dg_ids.append(np.nan)
            can_names.append(raw)

    if unresolved:
        raise ValueError(
            f"Unresolved standings names — add to _FORCED_DG_IDS or _ALIAS_MAP:\n"
            + "\n".join(f"  {n!r}  key={_standings_to_key(n)!r}" for n in unresolved)
        )

    df["dg_id"]  = dg_ids
    df["can_name"] = can_names
    df["dg_id"]  = pd.to_numeric(df["dg_id"], errors="coerce").astype("Int64")
    df["fec_rank"]   = df["rank"].astype(int)
    df["fec_points"] = pd.to_numeric(df["points"], errors="coerce")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Payout tables
# ═══════════════════════════════════════════════════════════════════════════════

PAYOUT_PATH = OAD / "playoff_payout_scales.csv"


def _load_payout_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load exact payout values from playoff_payout_scales.csv.
    Returns (FSJ_PAYOUT, BMW_PAYOUT, TC_PAYOUT) as 1-indexed arrays
    (index 0 = sentinel 0; index r = payout for finish rank r).
    """
    df = pd.read_csv(PAYOUT_PATH)
    df = df.rename(columns=str.strip)

    # FSJ: positions 1-70
    fsj_rows = df[df["position"] <= 70].sort_values("position")
    fsj_arr  = np.concatenate([[0.0], fsj_rows["fsj_usd"].values.astype(float)])

    # BMW: positions 1-50 (same dollar values as FSJ)
    bmw_rows = df[df["position"] <= 50].sort_values("position")
    bmw_arr  = np.concatenate([[0.0], bmw_rows["bmw_usd"].values.astype(float)])

    # TC: positions 1-30 (full $40M; 0.5x applied via EVENT_MULT at sim time)
    tc_rows = df[df["position"] <= 30].sort_values("position")
    tc_arr  = np.concatenate([[0.0], tc_rows["tc_usd"].values.astype(float)])

    return fsj_arr, bmw_arr, tc_arr


FSJ_PAYOUT, BMW_PAYOUT, TC_PAYOUT = _load_payout_tables()


def _build_payout_table(n_players: int, purse: int, anchors: dict[int, int]) -> np.ndarray:
    """Log-linear interpolation between anchor points (used for FCP points only)."""
    positions  = np.arange(1, n_players + 1, dtype=float)
    anchor_pos = sorted(anchors.keys())
    anchor_val = [float(anchors[p]) for p in anchor_pos]
    log_anchor = np.log(np.maximum(anchor_val, 1))
    log_vals   = np.interp(positions, anchor_pos, log_anchor)
    raw        = np.exp(log_vals)
    for pos, val in anchors.items():
        raw[pos - 1] = val
    return np.concatenate([[0.0], raw])


# Approximate FedEx Cup playoff points per finish (used to determine BMW/TC fields).
# Source: historical estimates — treat as approximate.
_FSJ_FCP_ANCHORS = {1: 2500, 2: 1500, 5: 700, 10: 450, 20: 280, 35: 165, 50: 100, 70: 35}
_BMW_FCP_ANCHORS = {1: 2500, 2: 1500, 5: 700, 10: 450, 20: 280, 35: 165, 50: 100}

FSJ_FCP_PTS = _build_payout_table(70, 0, _FSJ_FCP_ANCHORS)
BMW_FCP_PTS = _build_payout_table(50, 0, _BMW_FCP_ANCHORS)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Skill ratings
# ═══════════════════════════════════════════════════════════════════════════════

def build_skill_ratings(
    dg_ids: list[int],
    rounds_path: Path = ROUNDS_PATH,
    cutoff: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute per-event composite SG rating for each player from rolling stats.
    Returns DataFrame with columns: dg_id, mu_27, mu_28, mu_60, sigma.

    mu_XX = expected SG per round at event XX, weighted by course characteristics.
    Falls back to median of field for players with no data.
    """
    if not rounds_path.exists():
        return pd.DataFrame({"dg_id": dg_ids,
                             "mu_27": 0.0, "mu_28": 0.0, "mu_60": 0.0, "sigma": ROUND_SIGMA})

    df = pd.read_csv(rounds_path, low_memory=False)
    df["dg_id"]      = pd.to_numeric(df["dg_id"],      errors="coerce")
    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    for c in ["sg_total", "sg_app", "sg_ott", "sg_t2g", "sg_arg", "sg_putt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if cutoff is not None:
        df = df[df["round_date"] < cutoff]

    df = df[df["dg_id"].isin(dg_ids)]

    # Use last 12 rounds (L12) for each player
    cutoff_12 = pd.Timestamp.now() - pd.Timedelta(days=365)
    recent = df[df["round_date"] >= cutoff_12]

    rows = []
    for did in dg_ids:
        pr = recent[recent["dg_id"] == did]
        if len(pr) < 4:
            pr = df[df["dg_id"] == did].sort_values("round_date").tail(24)

        def wmean(cols: dict[str, float]) -> float:
            total, wsum = 0.0, 0.0
            for col, wt in cols.items():
                if col in pr.columns:
                    vals = pr[col].dropna()
                    if len(vals) > 0:
                        total += vals.mean() * wt
                        wsum  += wt
            return total / wsum if wsum > 0 else 0.0

        mu = {eid: wmean(COURSE_WEIGHTS[eid]) for eid in EVENT_IDS}

        # Per-player sigma from round-to-round variance in sg_total
        if "sg_total" in pr.columns and len(pr["sg_total"].dropna()) >= 4:
            sigma = float(pr["sg_total"].dropna().std())
            sigma = max(min(sigma, 4.0), 1.5)   # clamp to reasonable range
        else:
            sigma = ROUND_SIGMA

        rows.append({"dg_id": did, "mu_27": mu[27], "mu_28": mu[28], "mu_60": mu[60], "sigma": sigma})

    skill = pd.DataFrame(rows)

    # Fill missing players with field median
    for col in ["mu_27", "mu_28", "mu_60"]:
        med = skill[col].median()
        skill[col] = skill[col].fillna(med)
    skill["sigma"] = skill["sigma"].fillna(ROUND_SIGMA)

    return skill


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Simulation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    """Output of run_simulation()."""
    n_sims:       int
    player_ids:   list[int]               # dg_id for each index position
    # Shape: (n_sims, n_players, 3) — points per player per event per sim
    points:       np.ndarray
    # Shape: (n_sims, n_players, 3) — bool, True if player was in this event's field in this sim
    eligible:     np.ndarray
    # Shape: (n_sims, n_players, 3) — finish rank (1-based, 0 = not in field)
    finish:       np.ndarray
    # Event order matches EVENT_IDS = [27, 28, 60]
    fec_points_after: np.ndarray          # (n_sims, n_players, 3) running FCP total after each event


def run_simulation(
    standings: pd.DataFrame,
    skill:     pd.DataFrame,
    n_sims:    int = DEFAULT_N_SIMS,
    seed:      int = 42,
) -> SimResult:
    """
    Simulate the full 3-event playoff sequence path-dependently.

    standings: output of load_standings() — must have dg_id, fec_points, bucket columns.
    skill:     output of build_skill_ratings() — must have dg_id, mu_27, mu_28, mu_60, sigma.

    Only FSJ-eligible players (bucket in TC30/BMW50/FSJ70) are simulated.
    """
    rng = np.random.default_rng(seed)

    # Align on player order using FSJ field (all non-OUT entries)
    fsj_pool = standings[standings["bucket"].isin(["TC30", "BMW50", "FSJ70"])].copy()
    fsj_pool = fsj_pool.merge(skill, on="dg_id", how="left")
    fsj_pool["mu_27"]   = fsj_pool["mu_27"].fillna(0.0)
    fsj_pool["mu_28"]   = fsj_pool["mu_28"].fillna(0.0)
    fsj_pool["mu_60"]   = fsj_pool["mu_60"].fillna(0.0)
    fsj_pool["sigma"]   = fsj_pool["sigma"].fillna(ROUND_SIGMA)
    fsj_pool["fec_pts"] = fsj_pool["fec_points"].fillna(0.0)
    fsj_pool = fsj_pool.reset_index(drop=True)

    n_players = len(fsj_pool)
    player_ids = fsj_pool["dg_id"].tolist()

    mu27   = fsj_pool["mu_27"].values
    mu28   = fsj_pool["mu_28"].values
    mu60   = fsj_pool["mu_60"].values
    sigmas = fsj_pool["sigma"].values
    init_fcp = fsj_pool["fec_pts"].values

    # Generate all raw scores upfront: (n_sims, n_players, n_rounds)
    def _gen_scores(mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
        # Shape: (n_players, n_sims, n_rounds)
        raw = rng.normal(
            mu[:, None, None],
            sig[:, None, None],
            (n_players, n_sims, N_ROUNDS_PER_EV),
        )
        return raw.sum(axis=2).T  # → (n_sims, n_players)

    scores_fsj = _gen_scores(mu27, sigmas)
    scores_bmw = _gen_scores(mu28, sigmas)
    scores_tc  = _gen_scores(mu60, sigmas)

    # Output arrays
    points   = np.zeros((n_sims, n_players, 3), dtype=np.float64)
    eligible = np.zeros((n_sims, n_players, 3), dtype=bool)
    finish   = np.zeros((n_sims, n_players, 3), dtype=np.int32)
    fcp_snap = np.zeros((n_sims, n_players, 3), dtype=np.float64)

    for s in range(n_sims):
        # ── FSJ: all n_players eligible ────────────────────────────────────
        sc = scores_fsj[s]
        order = np.argsort(-sc)
        fsj_rank = np.empty(n_players, dtype=int)
        fsj_rank[order] = np.arange(1, n_players + 1)

        fsj_earn = FSJ_PAYOUT[fsj_rank]                    # $
        fsj_pts  = fsj_earn * EVENT_MULT[27]               # contest points
        fsj_fcp  = FSJ_FCP_PTS[fsj_rank]                   # approximate FCP addition

        points[s, :, 0]   = fsj_pts
        eligible[s, :, 0] = True
        finish[s, :, 0]   = fsj_rank

        fcp_after_fsj = init_fcp + fsj_fcp
        fcp_snap[s, :, 0] = fcp_after_fsj

        # ── BMW: top 50 by FCP after FSJ ───────────────────────────────────
        bmw_field_idx = np.argsort(-fcp_after_fsj)[:EVENT_FIELD[28]]
        bmw_elig = np.zeros(n_players, dtype=bool)
        bmw_elig[bmw_field_idx] = True

        masked = np.where(bmw_elig, scores_bmw[s], -np.inf)
        order  = np.argsort(-masked)
        bmw_rank = np.zeros(n_players, dtype=int)
        bmw_rank[order[:EVENT_FIELD[28]]] = np.arange(1, EVENT_FIELD[28] + 1)

        bmw_earn = np.where(bmw_elig, BMW_PAYOUT[bmw_rank], 0.0)
        bmw_pts  = bmw_earn * EVENT_MULT[28]
        bmw_fcp  = np.where(bmw_elig, BMW_FCP_PTS[np.maximum(bmw_rank, 1)], 0.0)

        points[s, :, 1]   = bmw_pts
        eligible[s, :, 1] = bmw_elig
        finish[s, :, 1]   = bmw_rank

        fcp_after_bmw = fcp_after_fsj + bmw_fcp
        fcp_snap[s, :, 1] = fcp_after_bmw

        # ── TC: top 30 by FCP after BMW ────────────────────────────────────
        tc_field_idx = np.argsort(-fcp_after_bmw)[:EVENT_FIELD[60]]
        tc_elig = np.zeros(n_players, dtype=bool)
        tc_elig[tc_field_idx] = True

        masked  = np.where(tc_elig, scores_tc[s], -np.inf)
        order   = np.argsort(-masked)
        tc_rank = np.zeros(n_players, dtype=int)
        tc_rank[order[:EVENT_FIELD[60]]] = np.arange(1, EVENT_FIELD[60] + 1)

        tc_earn = np.where(tc_elig, TC_PAYOUT[np.maximum(tc_rank, 1)], 0.0) * EVENT_MULT[60]
        tc_pts  = tc_earn

        points[s, :, 2]   = tc_pts
        eligible[s, :, 2] = tc_elig
        finish[s, :, 2]   = tc_rank
        fcp_snap[s, :, 2] = fcp_after_bmw   # TC doesn't change field further

    return SimResult(
        n_sims=n_sims, player_ids=player_ids,
        points=points, eligible=eligible, finish=finish, fec_points_after=fcp_snap,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizerResult:
    """Output of run_optimizer()."""
    # (a) Frequency mode: selection frequency for current week's event
    freq_df: pd.DataFrame                   # columns: dg_id, player, freq_pct, mean_pts_if_picked

    # (b) Point-estimate mode: single solve on mean EVs
    plan_df: pd.DataFrame                   # columns: dg_id, player, event_id, event_name, ev_pts

    # (c) Reserve list: P(qualifies for BMW/TC)
    reserve_df: pd.DataFrame                # columns: dg_id, player, p_bmw, p_tc, ev_bmw, ev_tc

    # (d) Total contest point distribution for the PE plan
    plan_total_pts: np.ndarray              # shape (n_sims,)

    mean_plan_pts:   float
    median_plan_pts: float
    p10_plan_pts:    float
    p90_plan_pts:    float


def _solve_ilp(
    ev_matrix:    np.ndarray,   # (n_players, 3) — expected points per player per event
    elig_matrix:  np.ndarray,   # (n_players, 3) bool — whether player can play each event
    n_picks:      int = 5,
    locked:       Optional[dict[int, int]] = None,   # player_idx → event_idx (0/1/2)
    excluded:     Optional[set[int]] = None,          # player_idx set (banned)
    our_used:     Optional[set[int]] = None,          # player_idx of already-used players
    msg:          bool = False,
) -> Optional[dict[tuple[int,int], int]]:
    """
    Solve the ILP: maximize total EV subject to 5 per event + once per player.
    Returns {(player_idx, event_idx): 1} for selected slots, or None if infeasible.
    """
    n_players = ev_matrix.shape[0]
    excluded  = excluded  or set()
    our_used  = our_used  or set()
    locked    = locked    or {}

    prob = pulp.LpProblem("pick5", pulp.LpMaximize)

    # Decision variables: x[p, e] ∈ {0, 1}
    x = {
        (p, e): pulp.LpVariable(f"x_{p}_{e}", cat="Binary")
        for p in range(n_players)
        for e in range(3)
        if elig_matrix[p, e] and p not in excluded and p not in our_used
    }

    # Objective
    prob += pulp.lpSum(ev_matrix[p, e] * x[p, e] for (p, e) in x)

    # Exactly 5 per event
    for e in range(3):
        prob += pulp.lpSum(x.get((p, e), 0) for p in range(n_players)) == n_picks

    # Each player used at most once
    for p in range(n_players):
        prob += pulp.lpSum(x.get((p, e), 0) for e in range(3)) <= 1

    # Locked constraints
    for p_idx, e_idx in locked.items():
        if (p_idx, e_idx) in x:
            prob += x[p_idx, e_idx] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=msg))

    if prob.status != 1:
        return None

    return {(p, e): int(round(v.value())) for (p, e), v in x.items() if v.value() and v.value() > 0.5}


def run_optimizer(
    standings:   pd.DataFrame,
    skill:       pd.DataFrame,
    sim_result:  Optional[SimResult] = None,
    n_sims:      int = DEFAULT_N_SIMS,
    our_used_dg_ids: Optional[set[int]] = None,
    locked_dg:   Optional[dict[int, int]] = None,   # dg_id → event_id
    excluded_dg: Optional[set[int]] = None,
    seed:        int = 42,
) -> OptimizerResult:
    """
    Run the full optimizer. If sim_result is already computed, pass it in to skip re-simulation.

    locked_dg:   force player (dg_id) into a specific event_id
    excluded_dg: ban player (dg_id) from all events
    our_used_dg_ids: players already spent in prior events of the contest
    """
    if sim_result is None:
        sim_result = run_simulation(standings, skill, n_sims=n_sims, seed=seed)

    sr      = sim_result
    n_p     = len(sr.player_ids)
    pid_arr = np.array(sr.player_ids)

    our_used_dg_ids = our_used_dg_ids or set()
    excluded_dg     = excluded_dg     or set()
    locked_dg       = locked_dg       or {}

    # Convert dg_ids to array indices
    pid_to_idx = {pid: i for i, pid in enumerate(sr.player_ids)}
    our_used_idx = {pid_to_idx[d] for d in our_used_dg_ids if d in pid_to_idx}
    excluded_idx = {pid_to_idx[d] for d in excluded_dg      if d in pid_to_idx}
    locked_idx   = {pid_to_idx[d]: EVENT_IDX[e] for d, e in locked_dg.items() if d in pid_to_idx}

    # ── (a) Reserve list ─────────────────────────────────────────────────────
    p_bmw = sr.eligible[:, :, 1].mean(axis=0)   # (n_players,) P(in BMW field)
    p_tc  = sr.eligible[:, :, 2].mean(axis=0)   # (n_players,) P(in TC field)
    ev_27 = sr.points[:, :, 0].mean(axis=0)
    ev_28 = sr.points[:, :, 1].mean(axis=0)
    ev_60 = sr.points[:, :, 2].mean(axis=0)

    dg_name_map = dict(zip(standings["dg_id"], standings["player"]))

    reserve_df = pd.DataFrame({
        "dg_id":  pid_arr,
        "player": [dg_name_map.get(d, str(d)) for d in pid_arr],
        "fec_rank": [
            standings.loc[standings["dg_id"] == d, "fec_rank"].iloc[0]
            if d in standings["dg_id"].values else np.nan
            for d in pid_arr
        ],
        "p_bmw":  (p_bmw * 100).round(1),
        "p_tc":   (p_tc  * 100).round(1),
        "ev_27":  ev_27.round(0),
        "ev_28":  ev_28.round(0),
        "ev_60":  ev_60.round(0),
        "ev_total": (ev_27 + ev_28 + ev_60).round(0),
    }).sort_values("fec_rank").reset_index(drop=True)

    # ── (b) Frequency mode (greedy per sim) ──────────────────────────────────
    # For each sim, greedily assign best available players to each event
    # in reverse order (TC first, most scarce), then BMW, then FSJ.
    freq_counts = np.zeros((n_p,), dtype=int)  # how often chosen for FSJ specifically
    all_freq    = np.zeros((n_p, 3), dtype=int)

    for s in range(sr.n_sims):
        elig_s = sr.eligible[s]    # (n_p, 3)
        pts_s  = sr.points[s]      # (n_p, 3)
        used   = set(our_used_idx) | set(excluded_idx)
        assign = {}

        for e_idx, eid in [(2, 60), (1, 28), (0, 27)]:   # TC first
            avail = [p for p in range(n_p) if elig_s[p, e_idx] and p not in used]
            avail.sort(key=lambda p: pts_s[p, e_idx], reverse=True)
            for p in avail[:5]:
                assign[p] = e_idx
                used.add(p)

        for p, e in assign.items():
            all_freq[p, e] += 1

    freq_df = pd.DataFrame({
        "dg_id":          pid_arr,
        "player":         [dg_name_map.get(d, str(d)) for d in pid_arr],
        "fec_rank":       [
            standings.loc[standings["dg_id"] == d, "fec_rank"].iloc[0]
            if d in standings["dg_id"].values else np.nan
            for d in pid_arr
        ],
        "freq_fsj_pct":   (all_freq[:, 0] / sr.n_sims * 100).round(1),
        "freq_bmw_pct":   (all_freq[:, 1] / sr.n_sims * 100).round(1),
        "freq_tc_pct":    (all_freq[:, 2] / sr.n_sims * 100).round(1),
        "ev_27":          ev_27.round(0),
        "ev_28":          ev_28.round(0),
        "ev_60":          ev_60.round(0),
    })
    freq_df = freq_df[freq_df["fec_rank"].notna()].sort_values("fec_rank").reset_index(drop=True)

    # ── (c) Point-estimate mode (single ILP on mean EVs) ────────────────────
    ev_matrix   = np.stack([ev_27, ev_28, ev_60], axis=1)   # (n_p, 3)
    elig_mean   = sr.eligible.mean(axis=0) > 0.05            # eligible in >5% of sims
    solution    = _solve_ilp(
        ev_matrix, elig_mean, n_picks=5,
        locked=locked_idx, excluded=excluded_idx, our_used=our_used_idx,
    )

    plan_rows = []
    if solution:
        for (p_idx, e_idx), _ in solution.items():
            eid = EVENT_IDS[e_idx]
            plan_rows.append({
                "dg_id":      int(pid_arr[p_idx]),
                "player":     dg_name_map.get(int(pid_arr[p_idx]), str(pid_arr[p_idx])),
                "fec_rank":   reserve_df.loc[reserve_df["dg_id"] == int(pid_arr[p_idx]), "fec_rank"].squeeze()
                              if int(pid_arr[p_idx]) in reserve_df["dg_id"].values else np.nan,
                "event_id":   eid,
                "event_name": EVENT_NAMES[eid],
                "ev_pts":     round(ev_matrix[p_idx, e_idx], 0),
            })
    plan_df = pd.DataFrame(plan_rows).sort_values(["event_id", "ev_pts"], ascending=[True, False]).reset_index(drop=True)

    # ── (d) Total points distribution for the PE plan ────────────────────────
    if solution and sr.n_sims > 0:
        plan_pts_sims = np.zeros(sr.n_sims)
        for (p_idx, e_idx) in solution:
            plan_pts_sims += sr.points[:, p_idx, e_idx]
    else:
        plan_pts_sims = np.zeros(sr.n_sims)

    return OptimizerResult(
        freq_df=freq_df,
        plan_df=plan_df,
        reserve_df=reserve_df,
        plan_total_pts=plan_pts_sims,
        mean_plan_pts=float(plan_pts_sims.mean()),
        median_plan_pts=float(np.median(plan_pts_sims)),
        p10_plan_pts=float(np.percentile(plan_pts_sims, 10)),
        p90_plan_pts=float(np.percentile(plan_pts_sims, 90)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. What-if harness
# ═══════════════════════════════════════════════════════════════════════════════

def score_lineup(
    lineup:     dict[int, int],   # dg_id → event_id
    sim_result: SimResult,
) -> dict:
    """
    Score a user-specified 15-player lineup (5 per event) against all sims.
    Returns EV, distribution stats, and validity checks.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(sim_result.player_ids)}
    pts_per_sim = np.zeros(sim_result.n_sims)
    issues = []

    used_dg: set[int] = set()
    ev_counts: dict[int, int] = {}

    for dg_id, event_id in lineup.items():
        if event_id not in EVENT_IDX:
            issues.append(f"dg_id={dg_id}: unknown event_id {event_id}")
            continue
        e_idx = EVENT_IDX[event_id]

        if dg_id not in pid_to_idx:
            issues.append(f"dg_id={dg_id}: not in simulation player pool")
            continue
        if dg_id in used_dg:
            issues.append(f"dg_id={dg_id}: used in multiple events (O&D violation)")
            continue

        p_idx = pid_to_idx[dg_id]
        used_dg.add(dg_id)
        ev_counts[event_id] = ev_counts.get(event_id, 0) + 1
        pts_per_sim += sim_result.points[:, p_idx, e_idx]

    for eid in EVENT_IDS:
        n = ev_counts.get(eid, 0)
        if n != 5:
            issues.append(f"{EVENT_NAMES[eid]}: {n} picks (need exactly 5)")

    return {
        "ev":     float(pts_per_sim.mean()),
        "median": float(np.median(pts_per_sim)),
        "p10":    float(np.percentile(pts_per_sim, 10)),
        "p90":    float(np.percentile(pts_per_sim, 90)),
        "pts_distribution": pts_per_sim,
        "issues": issues,
    }


def shadow_prices(
    standings:     pd.DataFrame,
    skill:         pd.DataFrame,
    sim_result:    Optional[SimResult] = None,
    our_used_dg_ids: Optional[set[int]] = None,
    n_sims:        int = DEFAULT_N_SIMS,
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Compute shadow price for each FSJ-eligible player:
      shadow_price = EV(use now at FSJ) - opportunity_cost_of_using_now
      opportunity_cost = P(qualifies_for_later) * EV(best available at later event if saved)

    Higher shadow price → more valuable to use at FSJ now.
    Negative shadow price → better to save for BMW or TC.
    """
    if sim_result is None:
        sim_result = run_simulation(standings, skill, n_sims=n_sims, seed=seed)

    sr          = sim_result
    pid_arr     = np.array(sr.player_ids)
    pid_to_idx  = {int(p): i for i, p in enumerate(pid_arr)}
    our_used    = our_used_dg_ids or set()
    our_used_idx = {pid_to_idx[d] for d in our_used if d in pid_to_idx}

    ev_27 = sr.points[:, :, 0].mean(axis=0)
    ev_28 = sr.points[:, :, 1].mean(axis=0)
    ev_60 = sr.points[:, :, 2].mean(axis=0)
    p_bmw = sr.eligible[:, :, 1].mean(axis=0)
    p_tc  = sr.eligible[:, :, 2].mean(axis=0)

    # Opportunity cost = value lost by NOT using player at BMW/TC.
    # Marginal player at each event = the 6th-best alternative (we pick 5, so they'd
    # replace the weakest of our 5 picks, not the outright best).
    n_picks = 5

    def _marginal_loss(ev_all: np.ndarray, idx: int) -> float:
        """How much we lose at this event by removing player idx from the top-5 pool."""
        ev_others = np.delete(ev_all, idx)
        ev_sorted = np.sort(ev_others)[::-1]  # descending
        threshold = ev_sorted[n_picks - 1] if len(ev_sorted) >= n_picks else 0.0
        # Player i is in the top 5 only if their EV beats the threshold
        if ev_all[idx] >= threshold:
            marginal_6th = ev_sorted[n_picks] if len(ev_sorted) > n_picks else 0.0
            return float(ev_all[idx] - marginal_6th)
        return 0.0

    opp_cost_bmw = np.array([_marginal_loss(ev_28, i) for i in range(len(pid_arr))])
    opp_cost_tc  = np.array([_marginal_loss(ev_60, i) for i in range(len(pid_arr))])
    opp_cost     = p_bmw * opp_cost_bmw + p_tc * opp_cost_tc
    shadow   = ev_27 - opp_cost

    dg_name_map = dict(zip(standings["dg_id"], standings["player"]))
    fec_rank_map = dict(zip(standings["dg_id"], standings["fec_rank"]))

    df = pd.DataFrame({
        "dg_id":        pid_arr,
        "player":       [dg_name_map.get(d, str(d)) for d in pid_arr],
        "fec_rank":     [fec_rank_map.get(d, np.nan) for d in pid_arr],
        "ev_fsj":       ev_27.round(0),
        "ev_bmw":       ev_28.round(0),
        "ev_tc":        ev_60.round(0),
        "p_bmw_pct":    (p_bmw * 100).round(1),
        "p_tc_pct":     (p_tc  * 100).round(1),
        "opp_cost":     opp_cost.round(0),
        "shadow_price": shadow.round(0),
    })
    df = df[~df["dg_id"].isin(our_used)]
    return df.sort_values("shadow_price", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_match(
    standings_path: Path = STANDINGS_PATH,
    all_players_path: Path = ALL_PLAYERS_PATH,
    rounds_path: Path = ROUNDS_PATH,
) -> pd.DataFrame:
    """Load and dg_id-match standings in one call."""
    lookup = build_name_lookup(all_players_path, rounds_path)
    return load_standings(standings_path, lookup)


def get_bucket_field(standings: pd.DataFrame, event_id: int) -> pd.DataFrame:
    """Return projected field for a given event based on current bucket assignments."""
    bucket_map = {27: ["TC30", "BMW50", "FSJ70"], 28: ["TC30", "BMW50"], 60: ["TC30"]}
    buckets = bucket_map.get(event_id, [])
    return standings[standings["bucket"].isin(buckets)].copy()


if __name__ == "__main__":
    print("Loading standings…")
    st = load_and_match()
    print(f"  {len(st)} players matched, {st['dg_id'].isna().sum()} unresolved")
    print("\nBuilding skill ratings…")
    dg_ids = st["dg_id"].dropna().astype(int).tolist()
    sk = build_skill_ratings(dg_ids)
    print(f"  {len(sk)} players with skill data")
    print("\nRunning simulation (10,000 sims)…")
    sr = run_simulation(st, sk, n_sims=10_000)
    print(f"  Done. points.shape = {sr.points.shape}")
    print("\nRunning optimizer…")
    result = run_optimizer(st, sk, sim_result=sr)
    print("\n── Point-estimate plan ──")
    print(result.plan_df.to_string(index=False))
    print(f"\nExpected total points: ${result.mean_plan_pts:,.0f}")
    print(f"  P10/Median/P90: ${result.p10_plan_pts:,.0f} / ${result.median_plan_pts:,.0f} / ${result.p90_plan_pts:,.0f}")
    print("\n── Reserve list (top 20 by FEC rank) ──")
    print(result.reserve_df.head(20).to_string(index=False))
