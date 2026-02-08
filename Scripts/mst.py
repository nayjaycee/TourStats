import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="OAD Field Dashboard", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
SCHED_PATH  = DATA_DIR / "OAD_2026_Schedule.xlsx"
SKILL_PATH  = DATA_DIR / "app_skill.xlsx"
BUCKET_PATH = DATA_DIR / "Approach_Buckets.xlsx"
FIELDS_PATH = DATA_DIR / "Fields.xlsx"

# -------------------------
# Loaders (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def load_schedule() -> pd.DataFrame:
    df = pd.read_excel(SCHED_PATH)
    # normalize
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    return df

@st.cache_data(show_spinner=False)
def load_fields() -> pd.DataFrame:
    df = pd.read_excel(FIELDS_PATH)
    # normalize
    for c in ["event_id", "dg_id", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
    # normalize boolean flags (1/0)
    flag_cols = ["Win", "Top_5", "Top_10", "Top_25", "Made_Cut", "CUT"]
    for c in flag_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # unify player_name col
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)
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
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "event_name" in df.columns:
        df["event_name"] = df["event_name"].astype(str)
    return df

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_ytd(df_fields: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    YTD outcomes per player from Fields.xlsx flags.
    Assumes one row per (player,event) for that year (or close to it).
    """
    df = df_fields.copy()
    df = df[df["year"] == year].copy()

    if df.empty:
        return pd.DataFrame(columns=["dg_id", "player_name", "starts", "wins", "top10", "top25", "made_cuts", "made_cut_pct"])

    # Deduplicate to one row per player-event if needed
    key_cols = [c for c in ["year", "event_id", "dg_id"] if c in df.columns]
    if len(key_cols) >= 2:
        df = df.dropna(subset=["dg_id", "event_id"]).drop_duplicates(subset=key_cols, keep="last")

    out = (
        df.groupby(["dg_id", "player_name"], as_index=False)
          .agg(
              starts=("event_id", "count") if "event_id" in df.columns else ("dg_id", "count"),
              wins=("Win", "sum") if "Win" in df.columns else ("dg_id", "sum"),
              top5=("Top_5", "sum") if "Top_5" in df.columns else ("dg_id", "sum"),
              top10=("Top_10", "sum") if "Top_10" in df.columns else ("dg_id", "sum"),
              top25=("Top_25", "sum") if "Top_25" in df.columns else ("dg_id", "sum"),
              made_cuts=("Made_Cut", "sum") if "Made_Cut" in df.columns else ("dg_id", "sum"),
          )
    )
    out["made_cut_pct"] = np.where(out["starts"] > 0, out["made_cuts"] / out["starts"], np.nan)
    return out

def compute_approach_fit(
    field_dg_ids: list[int],
    skill_df: pd.DataFrame,
    buckets_df: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    """
    Dot-product: tournament bucket mix × player bucket values.
    Requires bucket % columns in buckets_df and corresponding *_fw_value columns in skill_df.
    """
    b = buckets_df[buckets_df["event_id"] == event_id].copy()
    if b.empty:
        return pd.DataFrame({"dg_id": field_dg_ids})

    b = b.iloc[0]

    # Expected tournament bucket columns (from your screenshot)
    tour_bucket_cols = ["50_100", "100_150", "150_200", "over_200"]
    tour_weights = {}
    for c in tour_bucket_cols:
        if c in b.index:
            tour_weights[c] = float(pd.to_numeric(b[c], errors="coerce") or 0.0)
        else:
            tour_weights[c] = 0.0

    # Normalize weights to sum to 1 if they look like percentages
    s = sum(tour_weights.values())
    if s > 0:
        tour_weights = {k: v / s for k, v in tour_weights.items()}

    # Map tournament buckets to player skill columns
    # (Based on your app_skill screenshot: 50_100_fw_value, 100_150_fw_value, 150_200_fw_value, over_200_fw_value)
    player_value_cols = {
        "50_100": "50_100_fw_value",
        "100_150": "100_150_fw_value",
        "150_200": "150_200_fw_value",
        "over_200": "over_200_fw_value",
    }

    s_df = skill_df.copy()
    s_df = s_df[pd.to_numeric(s_df["dg_id"], errors="coerce").isin(field_dg_ids)].copy()

    # Compute score
    score = np.zeros(len(s_df), dtype=float)
    for bucket, w in tour_weights.items():
        col = player_value_cols.get(bucket)
        if col and col in s_df.columns:
            vals = pd.to_numeric(s_df[col], errors="coerce").fillna(0.0).to_numpy()
            score += w * vals

    out = s_df[["dg_id", "player_name"]].copy() if "player_name" in s_df.columns else s_df[["dg_id"]].copy()
    out["approach_fit_score"] = score

    # Include counts (optional, can help interpret reliability)
    # Example: sum of bucket shot counts as a proxy for sample size
    count_cols = ["50_100_fw_shot_count", "100_150_fw_shot_count", "150_200_fw_shot_count", "over_200_fw_shot_count"]
    present_counts = [c for c in count_cols if c in s_df.columns]
    if present_counts:
        cnt = pd.to_numeric(s_df[present_counts], errors="coerce").fillna(0).sum(axis=1)
        out["approach_samples"] = cnt.astype(int)

    return out

# -------------------------
# UI
# -------------------------
st.title("OAD Public Field Dashboard")

schedule = load_schedule()
fields = load_fields()
skills = load_player_skill()
buckets = load_approach_buckets()

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    # Event selector from schedule
    sched = schedule.copy()
    if "start_date" in sched.columns:
        sched = sched.sort_values("start_date")
    else:
        sched = sched.sort_values("event_id")

    # Build label
    def _label(r):
        eid = int(r["event_id"]) if pd.notna(r.get("event_id")) else -1
        nm = str(r.get("event_name", "")).strip()
        dt = r.get("start_date")
        if pd.notna(dt):
            return f"{eid:02d} — {nm} ({pd.to_datetime(dt).date()})"
        return f"{eid:02d} — {nm}"

    sched["__label"] = sched.apply(_label, axis=1)
    labels = sched["__label"].tolist()
    label_to_eid = dict(zip(labels, sched["event_id"].astype(int)))

    selected_label = st.selectbox("Event", labels, index=0)
    event_id = int(label_to_eid[selected_label])

    # Determine season year for YTD (use schedule start_date year if available; else user pick)
    default_year = 2026
    if "start_date" in sched.columns:
        yr_guess = pd.to_datetime(sched.loc[sched["event_id"] == event_id, "start_date"].iloc[0], errors="coerce")
        if pd.notna(yr_guess):
            default_year = int(yr_guess.year)

    ytd_year = st.selectbox("YTD Year (for flags)", sorted(fields["year"].dropna().astype(int).unique().tolist()), index=None)
    if ytd_year is None:
        ytd_year = default_year

    only_in_field = st.toggle("Only players in selected event field", value=True)

# Field for selected event (from Fields.xlsx)
field_ev = fields[fields["event_id"] == event_id].copy() if "event_id" in fields.columns else fields.head(0).copy()
field_ids = sorted(field_ev["dg_id"].dropna().astype(int).unique().tolist()) if "dg_id" in field_ev.columns else []

# Universe table (from skills list or from fields)
universe = None
if "dg_id" in skills.columns:
    universe = skills[["dg_id", "player_name"]].drop_duplicates().copy()
else:
    universe = fields[["dg_id", "player_name"]].drop_duplicates().copy()

if only_in_field:
    universe = universe[universe["dg_id"].astype(int).isin(field_ids)].copy()

# Compute YTD outcomes
ytd = compute_ytd(fields, int(ytd_year))

# Compute approach fit
fit = compute_approach_fit(field_ids if only_in_field else universe["dg_id"].astype(int).tolist(), skills, buckets, event_id)

# Merge into one view
out = universe.merge(ytd, on=["dg_id", "player_name"], how="left")
out = out.merge(fit.drop(columns=["player_name"], errors="ignore"), on="dg_id", how="left")

# Clean display
for c in ["starts", "wins", "top5", "top10", "top25", "made_cuts"]:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
if "made_cut_pct" in out.columns:
    out["made_cut_pct"] = pd.to_numeric(out["made_cut_pct"], errors="coerce").fillna(0.0)

# Optional: include odds if present
odds_col = pick_first_existing(field_ev, ["close_odds", "odds", "decimal_odds"])
if odds_col and not field_ev.empty:
    odds_tmp = field_ev[["dg_id", odds_col]].copy()
    odds_tmp["dg_id"] = pd.to_numeric(odds_tmp["dg_id"], errors="coerce")
    out = out.merge(odds_tmp, on="dg_id", how="left")

# Display
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Players")
    sort_col = "approach_fit_score" if "approach_fit_score" in out.columns else ("wins" if "wins" in out.columns else "starts")
    out_show = out.sort_values(sort_col, ascending=False).reset_index(drop=True)

    show_cols = ["player_name", "approach_fit_score", "approach_samples", "starts", "wins", "top10", "top25", "made_cut_pct"]
    if odds_col:
        show_cols.insert(2, odds_col)
    show_cols = [c for c in show_cols if c in out_show.columns]

    st.dataframe(out_show[show_cols], use_container_width=True, height=650)

with c2:
    st.subheader("Tournament approach mix")
    b = buckets[buckets["event_id"] == event_id].copy()
    if b.empty:
        st.info("No approach bucket distribution found for this event_id in Approach_Buckets.xlsx.")
    else:
        row = b.iloc[0]
        mix_cols = [c for c in ["50_100", "100_150", "150_200", "over_200"] if c in b.columns]
        mix = {c: float(pd.to_numeric(row[c], errors="coerce") or 0.0) for c in mix_cols}
        total = sum(mix.values())
        if total > 0:
            mix = {k: v / total for k, v in mix.items()}
        st.write(mix)

st.caption("MVP: event selector + field toggle + YTD outcomes + approach-fit score. Rolling L40/L24/L12 will be added via precomputed snapshots or a round-level source.")
