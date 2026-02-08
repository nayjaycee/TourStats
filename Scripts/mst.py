from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

SEASON_YEAR = 2026

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Player Stats", layout="wide")
st.title("Player Stats")


# ============================================================
# Paths (AUTO-DETECT between your local OAD project and public repo)
# ============================================================
THIS_FILE = Path(__file__).resolve()

CANDIDATE_DATA_DIRS = [
    # Local OAD project layout
    THIS_FILE.parents[1] / "Data" / "MST",      # .../OAD/Scripts/mst.py -> .../OAD/Data/MST
    # Public repo layout (if mst.py is at repo root)
    THIS_FILE.parent / "data" / "mst",          # .../oad-public/mst.py -> .../oad-public/data/mst
    THIS_FILE.parent / "data" / "MST",
]

def _pick_data_dir(cands: list[Path]) -> Path:
    for d in cands:
        if d.exists():
            return d
    # fall back to the first candidate; we’ll error with exact missing file paths below
    return cands[0]

DATA_DIR = _pick_data_dir(CANDIDATE_DATA_DIRS)

# Required MST Excel inputs
SCHED_PATH  = DATA_DIR / "OAD_2026_Schedule.xlsx"
SKILL_PATH  = DATA_DIR / "app_skill.xlsx"
FIELDS_PATH = DATA_DIR / "Fields.xlsx"

# Bucket filename varies in your worlds; try both
BUCKET_PATH_A = DATA_DIR / "Approach_Buckets.xlsx"
BUCKET_PATH_B = DATA_DIR / "Approach Buckets.xlsx"
BUCKET_PATH = BUCKET_PATH_A if BUCKET_PATH_A.exists() else BUCKET_PATH_B

# Round-level combined CSV: local OAD project path
ROUNDS_CANDIDATES = [
    THIS_FILE.parents[1] / "Data" / "in Use" / "combined_rounds_all_2017_2026.csv",  # .../OAD/Data/in Use/...
    THIS_FILE.parent / "data" / "combined_rounds_all_2017_2026.csv",                 # if you later copy into public repo
]
ROUNDS_PATH = next((p for p in ROUNDS_CANDIDATES if p.exists()), ROUNDS_CANDIDATES[0])

# Hard fail early with a helpful message (no guessing)
missing = [p for p in [SCHED_PATH, SKILL_PATH, FIELDS_PATH, BUCKET_PATH, ROUNDS_PATH] if not p.exists()]
if missing:
    st.error("Missing required file(s):")
    for p in missing:
        st.code(str(p))
    st.stop()


# ============================================================
# Loaders (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_schedule() -> pd.DataFrame:
    df = pd.read_excel(SCHED_PATH)
    # common normalizations
    for c in ["event_id", "course_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["start_date", "event_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "event_name" in df.columns:
        df["event_name"] = df["event_name"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_fields() -> pd.DataFrame:
    df = pd.read_excel(FIELDS_PATH)

    st.sidebar.write("Fields loaded rows:", len(df))
    st.sidebar.write("Fields year min/max:",
                     pd.to_numeric(df["year"], errors="coerce").min(),
                     pd.to_numeric(df["year"], errors="coerce").max())
    st.sidebar.write("Fields unique years (sample):",
                     sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique().tolist())[:20])

    # normalize types
    for c in ["year", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

    # normalize binary outcome flags
    flag_cols = ["Win", "Top_5", "Top_10", "Top_25", "Made_Cut", "CUT"]
    for c in flag_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_player_skill() -> pd.DataFrame:
    df = pd.read_excel(SKILL_PATH)
    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_approach_buckets() -> pd.DataFrame:
    df = pd.read_excel(BUCKET_PATH)
    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    if "event_name" in df.columns:
        df["event_name"] = df["event_name"].astype(str)
    for c in ["start_date", "event_date", "event_completed"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_rounds_minimal() -> pd.DataFrame:
    """
    Load only the columns we need for rolling stats.
    This keeps memory and compute sane.
    """
    usecols = [
        "tour", "dg_id", "player_name", "event_id", "year",
        "round_date", "event_completed",
        "round_score",
        "sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg", "sg_t2g",
    ]
    # read_csv will ignore missing usecols if you pass them? No — it errors.
    # So we do a two-pass: read header, intersect columns.
    with open(ROUNDS_PATH, "r") as f:
        header = f.readline().strip().split(",")
    cols = [c for c in usecols if c in header]

    df = pd.read_csv(ROUNDS_PATH, usecols=cols)

    if "tour" in df.columns:
        df["tour"] = df["tour"].astype(str).str.lower().str.strip()

    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")

    # prefer round_date; fallback to event_completed
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")

    for c in ["round_score", "sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg", "sg_t2g"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


# ============================================================
# Rolling stats (L40/L24/L12) from round-level data
# ============================================================
def _player_rolling_for_windows(g: pd.DataFrame, windows: Sequence[int]) -> pd.Series:
    out: Dict[str, float] = {}
    stats = [
        "sg_total",
        "sg_putt",
        "sg_app",
        "sg_ott",
        "sg_arg",
        "sg_t2g",
        "round_score",
    ]
    for w in windows:
        sub = g.head(w)
        for stat in stats:
            col = f"{stat}_L{w}"
            out[col] = float(sub[stat].mean()) if stat in sub.columns else np.nan
    return pd.Series(out)

@st.cache_data(show_spinner=False)
def compute_rolling_stats(
    rounds_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    dg_ids: Iterable[int],
    windows: Sequence[int] = (40, 24, 12),
) -> pd.DataFrame:
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = rounds_df.copy()

    # PGA only (per your requirements)
    if "tour" in df.columns:
        df = df[df["tour"] == "pga"].copy()

    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df[df["dg_id"].isin([int(x) for x in dg_ids])].copy()

    # pick date column
    if "round_date" in df.columns and df["round_date"].notna().any():
        date_col = "round_date"
    elif "event_completed" in df.columns:
        date_col = "event_completed"
    else:
        raise ValueError("Rounds data must include round_date or event_completed.")

    df = df[df[date_col] < ts].copy()
    if df.empty:
        return pd.DataFrame({"dg_id": list(dg_ids)})

    # sort most recent first per player
    df = df.sort_values(["dg_id", date_col], ascending=[True, False])

    rolled = (
        df.groupby("dg_id", group_keys=False)
          .apply(_player_rolling_for_windows, windows=windows)
          .reset_index()
    )
    return rolled


# ============================================================
# YTD outcomes from Fields.xlsx
# ============================================================
def compute_ytd_from_fields(fields: pd.DataFrame, year: int) -> pd.DataFrame:
    df = fields.copy()
    df = df[pd.to_numeric(df.get("year"), errors="coerce") == int(year)].copy()
    if df.empty:
        return pd.DataFrame(columns=["dg_id", "player_name", "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts", "ytd_made_cut_pct"])

    # enforce one row per (dg_id,event_id)
    df = df.dropna(subset=["dg_id", "event_id"]).copy()
    df = df.drop_duplicates(subset=["dg_id", "event_id"], keep="last")

    out = (
        df.groupby(["dg_id", "player_name"], as_index=False)
          .agg(
              ytd_starts=("event_id", "count"),
              ytd_wins=("Win", "sum") if "Win" in df.columns else ("event_id", "size"),
              ytd_top10=("Top_10", "sum") if "Top_10" in df.columns else ("event_id", "size"),
              ytd_top25=("Top_25", "sum") if "Top_25" in df.columns else ("event_id", "size"),
              ytd_made_cuts=("Made_Cut", "sum") if "Made_Cut" in df.columns else ("event_id", "size"),
          )
    )
    out["ytd_made_cut_pct"] = np.where(out["ytd_starts"] > 0, out["ytd_made_cuts"] / out["ytd_starts"], np.nan)
    return out


# ============================================================
# Approach fit (player bucket skill × tournament bucket mix)
# ============================================================
def compute_approach_fit(
    dg_ids: List[int],
    skill_df: pd.DataFrame,
    buckets_df: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    base = pd.DataFrame({"dg_id": dg_ids}).drop_duplicates()
    base["dg_id"] = pd.to_numeric(base["dg_id"], errors="coerce")

    b = buckets_df[pd.to_numeric(buckets_df.get("event_id"), errors="coerce") == int(event_id)].copy()
    if b.empty:
        base["approach_fit_score"] = np.nan
        return base

    row = b.iloc[0]

    # tournament mix columns (your file)
    tour_bucket_cols = ["50_100", "100_150", "150_200", "over_200"]
    weights = {c: float(pd.to_numeric(row.get(c), errors="coerce") or 0.0) for c in tour_bucket_cols}
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}

    # player columns (your skill file)
    player_value_cols = {
        "50_100": "50_100_fw_value",
        "100_150": "100_150_fw_value",
        "150_200": "150_200_fw_value",
        "over_200": "over_200_fw_value",
    }

    s_df = skill_df.copy()
    s_df["dg_id"] = pd.to_numeric(s_df.get("dg_id"), errors="coerce")
    s_df = s_df[s_df["dg_id"].isin([int(x) for x in dg_ids])].copy()

    if s_df.empty:
        base["approach_fit_score"] = np.nan
        return base

    score = np.zeros(len(s_df), dtype=float)
    for bucket, w in weights.items():
        col = player_value_cols.get(bucket)
        if col and col in s_df.columns:
            vals = pd.to_numeric(s_df[col], errors="coerce").fillna(0.0).to_numpy()
            score += w * vals

    out = s_df[["dg_id"]].copy()
    out["approach_fit_score"] = score

    # Optional sample size proxy
    count_cols = ["50_100_fw_shot_count", "100_150_fw_shot_count", "150_200_fw_shot_count", "over_200_fw_shot_count"]
    present = [c for c in count_cols if c in s_df.columns]
    if present:
        tmp = s_df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["approach_samples"] = tmp.sum(axis=1).astype(int)

    return base.merge(out, on="dg_id", how="left")


# ============================================================
# Sidebar controls (event selector, field toggle)
# ============================================================
schedule = load_schedule()
fields = load_fields()
skills = load_player_skill()
buckets = load_approach_buckets()
rounds = load_rounds_minimal()
field_ids: list[int] = []

import os

st.sidebar.write("FIELDS_PATH:", str(FIELDS_PATH))
st.sidebar.write("FIELDS exists?:", os.path.exists(FIELDS_PATH))
st.sidebar.write("FIELDS cwd:", os.getcwd())



with st.sidebar:
    st.header("Controls")

    # --- schedule ordering ---
    sched = schedule.copy()

    date_col = "start_date" if "start_date" in sched.columns else ("event_date" if "event_date" in sched.columns else None)
    if date_col:
        sched[date_col] = pd.to_datetime(sched[date_col], errors="coerce")
        sched = sched.sort_values(date_col, na_position="last")
    else:
        sched = sched.reset_index(drop=True)

    # stable row selection (works even if event_id is missing)
    sched = sched.reset_index(drop=True)
    sched["__row_id"] = sched.index.astype(int)

    # event dropdown shows ONLY event_name
    sched["__event_label"] = sched.get("event_name", "").astype(str)
    sched.loc[sched["__event_label"].str.strip().isin(["", "nan", "None"]), "__event_label"] = "Unknown event"

    labels = sched["__event_label"].tolist()

    selected_idx = st.selectbox(
        "Event",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        index=0,
    )

    selected_row = sched.iloc[int(selected_idx)]

    # event_id can be NA; that's OK
    event_id_val = pd.to_numeric(selected_row.get("event_id"), errors="coerce")
    event_id = int(event_id_val) if pd.notna(event_id_val) else None

    st.write("Selected schedule row raw event_id:", selected_row.get("event_id"))
    st.write("Parsed event_id used for lookup:", event_id)

    # default True; we'll disable later if the selected event has no field rows
    if "only_in_field" not in st.session_state:
        st.session_state.only_in_field = True

    only_in_field = st.toggle(
        "Only players in this week's field",
        key="only_in_field"
    )

# Pull schedule row for this event_id (first match)
# Selected schedule row (by index), independent of event_id
sched_row = selected_row

# Cutoff date for rolling stats: day before start_date/event_date if available, else "today"
if "start_date" in sched_row.index and pd.notna(sched_row.get("start_date")):
    cutoff = pd.to_datetime(sched_row["start_date"]) - pd.Timedelta(days=1)
elif "event_date" in sched_row.index and pd.notna(sched_row.get("event_date")):
    cutoff = pd.to_datetime(sched_row["event_date"]) - pd.Timedelta(days=1)
else:
    cutoff = pd.Timestamp.today()

if event_id is None:
    st.info("This schedule row has no event_id yet, so field/odds/YTD-by-event can’t be shown for it.")
elif not field_ids:
    st.warning("No field uploaded yet for this event (Fields.xlsx has 0 rows for this event_id).")

# ============================================================
# Build universe and field (2026 only)
# ============================================================

field_ev = pd.DataFrame()
field_ids: list[int] = []

if event_id is not None:
    f = fields.copy()
    f["event_id"] = pd.to_numeric(f["event_id"], errors="coerce")
    f["year"] = pd.to_numeric(f["year"], errors="coerce")
    f["dg_id"] = pd.to_numeric(f["dg_id"], errors="coerce")

    # OFFICIAL FIELD = 2026 ONLY
    field_ev = f[
        (f["event_id"] == int(event_id)) &
        (f["year"] == SEASON_YEAR)
    ].dropna(subset=["dg_id"])

    field_ev = field_ev.drop_duplicates(subset=["dg_id"], keep="last")
    field_ids = field_ev["dg_id"].astype(int).tolist()

# Auto-disable toggle if field not uploaded yet
if only_in_field and not field_ids:
    st.warning(
        "No 2026 field uploaded yet for this event. Showing scouting universe instead."
    )
    only_in_field = False

# ------------------------------------------------------------
# Universe definition:
#  - all players in 2026 (any appearance)
#  - plus players in 2025 with >= 4 starts
# ------------------------------------------------------------
f_univ = fields.copy()

# normalize keys once
for c in ["year", "event_id", "dg_id"]:
    if c in f_univ.columns:
        f_univ[c] = pd.to_numeric(f_univ[c], errors="coerce")

if "player_name" in f_univ.columns:
    f_univ["player_name"] = f_univ["player_name"].astype(str)

# keep only rows with ids
f_univ = f_univ.dropna(subset=["dg_id", "year"]).copy()
f_univ["dg_id"] = f_univ["dg_id"].astype(int)
f_univ["year"] = f_univ["year"].astype(int)

# starts = unique events per player in 2025
starts_2025 = (
    f_univ[f_univ["year"] == 2025]
    .dropna(subset=["event_id"])
    .drop_duplicates(subset=["dg_id", "event_id"])
    .groupby("dg_id")["event_id"]
    .size()
    .rename("starts_2025")
    .reset_index()
)

ids_2025_4plus = set(starts_2025.loc[starts_2025["starts_2025"] >= 4, "dg_id"].astype(int).tolist())
ids_2026_all   = set(f_univ.loc[f_univ["year"] == 2026, "dg_id"].astype(int).unique().tolist())

universe_ids = sorted(ids_2026_all | ids_2025_4plus)

# pick a canonical name per dg_id (prefer 2026 name, fallback to latest)
name_pool = f_univ[f_univ["dg_id"].isin(universe_ids)].copy()
name_pool["is_2026"] = (name_pool["year"] == 2026).astype(int)
name_pool = name_pool.sort_values(["dg_id", "is_2026", "year"], ascending=[True, False, False])

universe = name_pool.drop_duplicates(subset=["dg_id"], keep="first")[["dg_id", "player_name"]].copy()

if only_in_field:
    base_ids = field_ids
else:
    base_ids = universe["dg_id"].astype(int).unique().tolist()

base = universe[universe["dg_id"].isin(base_ids)].copy()


# ============================================================
# Compute: rolling stats + YTD + approach fit + odds (if present)
# ============================================================
with st.spinner("Computing rolling stats (L40/L24/L12) from round-level data..."):
    rolling = compute_rolling_stats(rounds_df=rounds, as_of_date=cutoff, dg_ids=base_ids, windows=(40, 24, 12))

ytd = compute_ytd_from_fields(fields, SEASON_YEAR)
fit = compute_approach_fit(base_ids, skills, buckets, event_id) if event_id is not None else pd.DataFrame({"dg_id": base_ids, "approach_fit_score": np.nan})
fit = compute_approach_fit(base_ids, skills, buckets, event_id)

# Merge
out = base.merge(rolling, on="dg_id", how="left").merge(ytd, on=["dg_id", "player_name"], how="left").merge(fit, on="dg_id", how="left")
out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
out = out.dropna(subset=["dg_id"]).copy()
out["dg_id"] = out["dg_id"].astype(int)
out = out.drop_duplicates(subset=["dg_id"], keep="first")

# Odds from Fields if present
odds_candidates = ["close_odds", "decimal_odds", "odds", "win_prob_est"]
odds_col = next((c for c in odds_candidates if c in field_ev.columns), None)
if odds_col:
    tmp = field_ev[["dg_id", odds_col]].copy()
    tmp["dg_id"] = pd.to_numeric(tmp["dg_id"], errors="coerce")
    out = out.merge(tmp, on="dg_id", how="left")

# Clean numeric display
for c in ["ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts"]:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
if "ytd_made_cut_pct" in out.columns:
    out["ytd_made_cut_pct"] = pd.to_numeric(out["ytd_made_cut_pct"], errors="coerce").fillna(0.0)

# Round rolling columns to readable precision
for c in out.columns:
    if any(c.startswith(x) for x in ["sg_", "round_score"]):
        out[c] = pd.to_numeric(out[c], errors="coerce")
# SG: 3 decimals; score: 2
for c in [col for col in out.columns if col.startswith("sg_")]:
    out[c] = out[c].round(3)
for c in [col for col in out.columns if col.startswith("round_score_")]:
    out[c] = out[c].round(2)
if "approach_fit_score" in out.columns:
    out["approach_fit_score"] = pd.to_numeric(out["approach_fit_score"], errors="coerce").round(3)
if "ytd_made_cut_pct" in out.columns:
    out["ytd_made_cut_pct"] = out["ytd_made_cut_pct"].round(3)


# ============================================================
# UI layout
# ============================================================
tab1, tab2, tab3 = st.tabs(["Field Table", "Tournament Buckets", "Debug"])

with tab1:
    st.subheader("Players (rolling SG L40/L24/L12 + YTD + approach fit)")

    sort_choices = []
    for c in ["sg_total_L12", "sg_total_L24", "sg_total_L40", "approach_fit_score", "ytd_wins", "ytd_top10"]:
        if c in out.columns:
            sort_choices.append(c)
    sort_by = st.selectbox("Sort by", sort_choices, index=0 if sort_choices else None)

    out_show = out.copy()
    if sort_by:
        out_show = out_show.sort_values(sort_by, ascending=False)

    # Column set: prioritize total/approach/putting
    display_cols = [
        "player_name",
        odds_col if odds_col else None,
        "sg_total_L12", "sg_total_L24", "sg_total_L40",
        "sg_app_L12", "sg_app_L24", "sg_app_L40",
        "sg_putt_L12", "sg_putt_L24", "sg_putt_L40",
        "sg_ott_L12", "sg_ott_L24", "sg_ott_L40",
        "sg_arg_L12", "sg_arg_L24", "sg_arg_L40",
        "sg_t2g_L12", "sg_t2g_L24", "sg_t2g_L40",
        "round_score_L12", "round_score_L24", "round_score_L40",
        "approach_fit_score", "approach_samples",
        "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cut_pct",
    ]
    display_cols = [c for c in display_cols if c and c in out_show.columns]

    st.dataframe(out_show[display_cols], use_container_width=True, height=650)

with tab2:
    st.subheader("Tournament approach bucket mix")

    b = buckets[pd.to_numeric(buckets.get("event_id"), errors="coerce") == event_id].copy()
    if b.empty:
        st.info("No approach bucket distribution found for this event_id in Approach Buckets.")
    else:
        row = b.iloc[0]
        mix_cols = [c for c in ["50_100", "100_150", "150_200", "over_200"] if c in b.columns]
        mix = {c: float(pd.to_numeric(row.get(c), errors="coerce") or 0.0) for c in mix_cols}
        total = sum(mix.values())
        if total > 0:
            mix = {k: v / total for k, v in mix.items()}
        st.write(mix)

with tab3:
    st.subheader("Debug (paths + row counts)")
    st.write("DATA_DIR:", str(DATA_DIR))
    st.write("ROUNDS_PATH:", str(ROUNDS_PATH))
    st.write("Schedule rows:", len(schedule))
    st.write("Fields rows:", len(fields))
    st.write("Skills rows:", len(skills))
    st.write("Buckets rows:", len(buckets))
    st.write("Rounds rows (minimal cols loaded):", len(rounds))
    st.write("Selected event_id:", event_id)
    st.write("Field size:", len(field_ids))
