from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Paths (repo-relative; works locally + Streamlit Cloud)
# ============================================================
THIS_FILE = Path(__file__).resolve()

def _pick_repo_root(start: Path) -> Path:
    # find ALL parents that contain a Data/ folder
    candidates = [p for p in [start.parent] + list(start.parents) if (p / "Data").exists()]
    if not candidates:
        raise RuntimeError("Could not find repo root containing Data/")
    # pick the HIGHEST one (closest to filesystem root), not the first
    return candidates[-1]

REPO_ROOT = _pick_repo_root(THIS_FILE)
DATA_ROOT = REPO_ROOT / "Data"
MST_DIR   = DATA_ROOT / "MST"
INUSE_DIR = DATA_ROOT / "in Use"

# ---- Inputs ----
ALL_PLAYERS_PATH = MST_DIR / "All_players.xlsx"
SCHED_PATH       = MST_DIR / "OAD_2026_Schedule.xlsx"
SKILL_PATH       = MST_DIR / "app_skill.xlsx"
FIELDS_PATH      = MST_DIR / "Fields.xlsx"

BUCKET_PATH_A = MST_DIR / "Approach_Buckets.xlsx"
BUCKET_PATH_B = MST_DIR / "Approach Buckets.xlsx"
BUCKET_PATH   = BUCKET_PATH_A if BUCKET_PATH_A.exists() else BUCKET_PATH_B

MST_EVENTLEVEL_2017_2023 = MST_DIR / "combined_eventlevel_pga_2017_2023_mean.csv"

ROUNDLEVEL_2024P_CANDIDATES = [
    MST_DIR / "combined_roundlevel_2024_present.csv",
    MST_DIR / "combined_roundlevel_2024_present_mean.csv",
    MST_DIR / "combined_roundlevel_2024_present_mean_only.csv",
    MST_DIR / "combined_roundlevel_2024_present_pga.csv",
]
MST_ROUNDLEVEL_2024_PRESENT = next((p for p in ROUNDLEVEL_2024P_CANDIDATES if p.exists()), None)

ROUNDS_PATH = INUSE_DIR / "combined_rounds_all_2017_2026.csv"

# ---- Hard fail early ----
required = [
    ALL_PLAYERS_PATH, SCHED_PATH, SKILL_PATH, FIELDS_PATH, BUCKET_PATH,
    MST_EVENTLEVEL_2017_2023,
    ROUNDS_PATH,
]
missing = [p for p in required if not p.exists()]

if MST_ROUNDLEVEL_2024_PRESENT is None:
    missing.append(MST_DIR / "combined_roundlevel_2024_present*.csv (not found)")
elif not MST_ROUNDLEVEL_2024_PRESENT.exists():
    missing.append(MST_ROUNDLEVEL_2024_PRESENT)

if missing:
    st.error("Missing required file(s):")
    for p in missing:
        st.code(str(p))
    st.stop()

# ============================================================
# LOADERS (MST split files)
# ============================================================
@st.cache_data(show_spinner=False)
def load_mst_roundlevel_2024_present() -> pd.DataFrame:
    if not MST_ROUNDLEVEL_2024_PRESENT.exists():
        raise FileNotFoundError(f"Missing: {MST_ROUNDLEVEL_2024_PRESENT}")
    df = pd.read_csv(MST_ROUNDLEVEL_2024_PRESENT, low_memory=False)

    # normalize types
    for c in ["dg_id", "event_id", "year", "round_num", "course_num", "finish_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # dates
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_mst_eventlevel_2017_2023() -> pd.DataFrame:
    if not MST_EVENTLEVEL_2017_2023.exists():
        raise FileNotFoundError(f"Missing: {MST_EVENTLEVEL_2017_2023}")
    df = pd.read_csv(MST_EVENTLEVEL_2017_2023, low_memory=False)

    # normalize types
    for c in ["dg_id", "event_id", "year", "course_num", "finish_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # event/tournament date columns (best-effort)
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


SEASON_YEAR = 2026

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="MST OaD", layout="wide")
st.title("One and Done")

st.markdown(
    """
    <style>
      /* Remove extra top padding in the main app area */
      .block-container { padding-top: 3.0rem; }

      /* Slightly tighten header area too (Streamlit versions vary) */
      header[data-testid="stHeader"] { height: 0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Loaders (cached)
# ============================================================
def get_pre_event_cutoff_date(sched: pd.DataFrame, event_id: int) -> Optional[pd.Timestamp]:
    """Returns (event_date/start_date - 1 day)."""
    if sched is None or sched.empty:
        return None

    s = sched.copy()
    s["event_id"] = pd.to_numeric(s.get("event_id"), errors="coerce")
    row = s.loc[s["event_id"] == int(event_id)].head(1)
    if row.empty:
        return None

    for c in ["event_date", "start_date"]:
        if c in row.columns:
            dt = pd.to_datetime(row.iloc[0][c], errors="coerce")
            if pd.notna(dt):
                return dt - pd.Timedelta(days=1)

    # fallback: any date-like column
    date_cols = [c for c in row.columns if "date" in c.lower()]
    if date_cols:
        dt = pd.to_datetime(row.iloc[0][date_cols[0]], errors="coerce")
        if pd.notna(dt):
            return dt - pd.Timedelta(days=1)

    return None


def _event_end_table_roundlevel(rounds: pd.DataFrame) -> pd.DataFrame:
    """year,event_id,event_end from round_date max (field-wide)."""
    if rounds is None or rounds.empty:
        return pd.DataFrame(columns=["year", "event_id", "event_end"])

    df = rounds.copy()
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    if "round_date" in df.columns and df["round_date"].notna().any():
        df = df.dropna(subset=["year", "event_id", "round_date"]).copy()
        ends = (
            df.groupby(["year", "event_id"], as_index=False)["round_date"]
              .max()
              .rename(columns={"round_date": "event_end"})
        )
    else:
        ends = df[["year", "event_id"]].drop_duplicates().copy()
        ends["event_end"] = pd.to_datetime(ends["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")

    ends["year"] = pd.to_numeric(ends["year"], errors="coerce").astype("Int64")
    ends["event_id"] = pd.to_numeric(ends["event_id"], errors="coerce").astype("Int64")
    ends = ends.dropna(subset=["year", "event_id"]).copy()
    ends["year"] = ends["year"].astype(int)
    ends["event_id"] = ends["event_id"].astype(int)
    ends["event_end"] = pd.to_datetime(ends["event_end"], errors="coerce")
    return ends


def _event_end_table_eventlevel(ev: pd.DataFrame) -> pd.DataFrame:
    """year,event_id,event_end from best available event date column."""
    if ev is None or ev.empty:
        return pd.DataFrame(columns=["year", "event_id", "event_end"])

    df = ev.copy()
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")

    # Pick a likely date column
    candidates = ["event_end", "end_date", "tournament_end", "event_date", "start_date", "date"]
    dcol = next((c for c in candidates if c in df.columns), None)
    if dcol is None:
        # any date-like column
        date_cols = [c for c in df.columns if "date" in c.lower()]
        dcol = date_cols[0] if date_cols else None

    if dcol is not None:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        ends = (
            df.dropna(subset=["year", "event_id"])
              .groupby(["year", "event_id"], as_index=False)[dcol]
              .max()
              .rename(columns={dcol: "event_end"})
        )
    else:
        ends = df[["year", "event_id"]].drop_duplicates().copy()
        ends["event_end"] = pd.to_datetime(ends["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")

    ends["year"] = pd.to_numeric(ends["year"], errors="coerce").astype("Int64")
    ends["event_id"] = pd.to_numeric(ends["event_id"], errors="coerce").astype("Int64")
    ends = ends.dropna(subset=["year", "event_id"]).copy()
    ends["year"] = ends["year"].astype(int)
    ends["event_id"] = ends["event_id"].astype(int)
    ends["event_end"] = pd.to_datetime(ends["event_end"], errors="coerce")
    return ends


def build_last_n_events_table_mst(
    rounds_2024p: pd.DataFrame,
    ev_2017_2023: pd.DataFrame,
    dg_id: int,
    n: int = 25,
    date_max: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Returns last N TOURNAMENTS pre-event for a player:
      - 2024+ from roundlevel aggregated to event
      - 2017–2023 from eventlevel directly (if columns exist)
    """
    parts = []

    # ---- 2024+ from round-level ----
    r = rounds_2024p.copy()
    if not r.empty and "dg_id" in r.columns:
        r["dg_id"] = pd.to_numeric(r["dg_id"], errors="coerce")
        r = r.loc[r["dg_id"] == int(dg_id)].copy()
        if not r.empty:
            ends_r = _event_end_table_roundlevel(rounds_2024p)
            r["year"] = pd.to_numeric(r.get("year"), errors="coerce")
            r["event_id"] = pd.to_numeric(r.get("event_id"), errors="coerce")

            fin_col = "fin_text" if "fin_text" in r.columns else ("finish_text" if "finish_text" in r.columns else None)
            if fin_col is None:
                r["fin_text"] = ""
                fin_col = "fin_text"

            def _first_non_null_str(s: pd.Series) -> str:
                s2 = s.dropna().astype(str)
                return s2.iloc[0] if len(s2) else ""

            t = (
                r.groupby(["year", "event_id", "event_name"], as_index=False)
                .agg(
                    Finish=(fin_col, _first_non_null_str),
                    SG_Total=("sg_total", "mean") if "sg_total" in r.columns else ("event_id", "size"),
                )
            )

            t = t.merge(ends_r, on=["year", "event_id"], how="left")
            t["event_end"] = pd.to_datetime(t["event_end"], errors="coerce")
            t["source"] = "2024+ roundlevel"
            parts.append(t)

    # ---- 2017-2023 from event-level ----
    e = ev_2017_2023.copy()
    if not e.empty and "dg_id" in e.columns:
        e["dg_id"] = pd.to_numeric(e["dg_id"], errors="coerce")
        e = e.loc[e["dg_id"] == int(dg_id)].copy()
        if not e.empty:
            ends_e = _event_end_table_eventlevel(ev_2017_2023)

            # best-guess columns for display
            name_col = "event_name" if "event_name" in e.columns else ("tournament" if "tournament" in e.columns else None)
            if name_col is None:
                e["event_name"] = e.get("event_id", "").astype(str)
                name_col = "event_name"

            fin_col = "fin_text" if "fin_text" in e.columns else ("finish_text" if "finish_text" in e.columns else None)
            if fin_col is None:
                e["fin_text"] = ""
                fin_col = "fin_text"

            # sg_total might be mean-per-round or mean-per-event; we just show it as-is
            sg_col = "sg_total" if "sg_total" in e.columns else None

            t = e[["year", "event_id", name_col, fin_col] + ([sg_col] if sg_col else [])].copy()
            t = t.rename(columns={name_col: "event_name", fin_col: "Finish"})
            if sg_col:
                t = t.rename(columns={sg_col: "SG_Total"})
            else:
                t["SG_Total"] = np.nan

            t = t.merge(ends_e, on=["year", "event_id"], how="left")
            t["event_end"] = pd.to_datetime(t["event_end"], errors="coerce")
            t["source"] = "2017-2023 eventlevel"
            parts.append(t)

    if not parts:
        return pd.DataFrame(columns=["Event", "Finish", "SG Total", "Year", "event_id"])

    out = pd.concat(parts, ignore_index=True)
    if date_max is not None and pd.notna(date_max):
        out = out.loc[out["event_end"].notna() & (out["event_end"] <= pd.to_datetime(date_max))].copy()

    out = out.sort_values(["event_end", "year", "event_id"], ascending=[False, False, False]).head(int(n)).copy()

    out["SG Total"] = pd.to_numeric(out.get("SG_Total"), errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    out = out.rename(columns={"event_name": "Event", "year": "Year"})
    return out[["Event", "Finish", "SG Total", "Year", "event_id"]].reset_index(drop=True)


def _last_n_rounds_pre_event(rounds_2024p: pd.DataFrame, dg_id: int, cutoff_dt: Optional[pd.Timestamp], n: int = 40) -> pd.DataFrame:
    """Uses ONLY 2024-present roundlevel."""
    df = rounds_2024p.copy()
    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce")
    df = df.loc[df["dg_id"] == int(dg_id)].copy()
    if df.empty:
        return df

    ends = _event_end_table_roundlevel(rounds_2024p)
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")
    df = df.dropna(subset=["year", "event_id"]).copy()
    df["year"] = df["year"].astype(int)
    df["event_id"] = df["event_id"].astype(int)

    df = df.merge(ends, on=["year", "event_id"], how="left")
    df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")

    if cutoff_dt is not None and pd.notna(cutoff_dt):
        df = df.loc[df["event_end"].notna() & (df["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

    if "round_date" in df.columns and df["round_date"].notna().any():
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
        df = df.sort_values(["round_date", "event_id", "round_num"], ascending=True)
    else:
        if "round_num" in df.columns:
            df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce")
        df = df.sort_values(["year", "event_id", "round_num"], ascending=True)

    return df.tail(int(n)).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_all_players() -> pd.DataFrame:
    df = pd.read_excel(ALL_PLAYERS_PATH)

    # Normalize column names for safety
    df.columns = [c.strip().lower() for c in df.columns]

    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

    # allow image to be missing, but create it
    if "image" not in df.columns:
        df["image"] = None
    else:
        df["image"] = df["image"].astype(str).str.strip()
        df.loc[df["image"].isin(["", "nan", "None"]), "image"] = None

    df = df.dropna(subset=["dg_id"]).drop_duplicates(subset=["dg_id"]).copy()
    df["dg_id"] = df["dg_id"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def build_headshot_maps(all_players: pd.DataFrame) -> tuple[dict[int, str], dict[str, str]]:
    """
    Returns:
      id_to_img: dg_id(int) -> image_url(str)
      name_to_img: player_name(str) -> image_url(str)
    """
    df = all_players.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "dg_id" not in df.columns or "player_name" not in df.columns:
        raise ValueError("All_players.xlsx must have columns: dg_id, player_name (and optionally image)")

    if "image" not in df.columns:
        df["image"] = None

    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df.dropna(subset=["dg_id"]).copy()
    df["dg_id"] = df["dg_id"].astype(int)

    df["player_name"] = df["player_name"].astype(str)

    # normalize image values
    df["image"] = df["image"].astype(str).str.strip()
    df.loc[df["image"].isin(["", "nan", "None"]), "image"] = None

    id_to_img = {}
    name_to_img = {}

    for _, row in df.iterrows():
        dg_id = int(row["dg_id"])
        name = str(row["player_name"])
        img = row["image"]
        if isinstance(img, str) and img.strip():
            id_to_img[dg_id] = img
            name_to_img[name] = img

    return id_to_img, name_to_img


def get_headshot_url(dg_id: int | None, player_name: str | None,
                     id_to_img: dict[int, str], name_to_img: dict[str, str]) -> str | None:
    if dg_id is not None and dg_id in id_to_img:
        return id_to_img[dg_id]
    if player_name is not None and player_name in name_to_img:
        return name_to_img[player_name]
    return None

def show_headshot(img_value: str | None, width: int = 90) -> None:
    """
    img_value can be:
      - a public URL (https://...)
      - a repo-relative/absolute file path that exists in the deployed repo
    """
    if not img_value:
        return

    s = str(img_value).strip()
    if not s or s.lower() in {"none", "nan"}:
        return

    p = Path(s)
    if p.exists():
        st.image(str(p), width=width)
        return

    # also try relative to REPO_ROOT (handles "Data/MST/headshots/x.png")
    p2 = REPO_ROOT / s
    if p2.exists():
        st.image(str(p2), width=width)
        return

    # otherwise treat as a URL
    st.image(s, width=width)


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

    # normalize types
    for c in ["year", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "event_completed" in df.columns:
        # Excel serial dates often look like 46058; convert those correctly
        ec = df["event_completed"]
        if pd.api.types.is_numeric_dtype(ec):
            df["event_completed"] = pd.to_datetime(ec, unit="D", origin="1899-12-30", errors="coerce")
        else:
            df["event_completed"] = pd.to_datetime(ec, errors="coerce")

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
        "birdies", "eagles_or_better",
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

    for c in ["round_score", "sg_total", "sg_putt", "sg_app", "sg_ott", "sg_arg", "sg_t2g", "birdies",
              "eagles_or_better"]:
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
        "birdies",
        "eagles_or_better",
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

def compute_ytd_from_fields_today(fields: pd.DataFrame, year: int) -> pd.DataFrame:
    df = fields.copy()

    # --- normalize key columns ---
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce")
    df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce")

    # basic filter
    df = df[df["year"] == int(year)].copy()
    df = df.dropna(subset=["dg_id", "event_id"]).copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "dg_id", "player_name",
            "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts", "ytd_made_cut_pct"
        ])

    df["dg_id"] = df["dg_id"].astype(int)
    df["event_id"] = df["event_id"].astype(int)
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

    # --- TODAY cutoff (not event-based) ---
    today = pd.Timestamp.today().normalize()

    # --- results gate: row counts only if results exist ---
    # If your Fields file has finish_num / finish_text / winnings, use them as the "results exist" signal.
    has_result = pd.Series(False, index=df.index)

    if "finish_num" in df.columns:
        has_result |= pd.to_numeric(df["finish_num"], errors="coerce").notna()

    if "finish_text" in df.columns:
        has_result |= df["finish_text"].notna()

    if "winnings" in df.columns:
        has_result |= pd.to_numeric(df["winnings"], errors="coerce").notna()

    # If you also have event_completed, require it AND require it to be < today.
    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
        has_result &= df["event_completed"].notna() & (df["event_completed"] < today)

    # FINAL FILTER: only completed-with-results rows count for YTD
    df = df[has_result].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "dg_id", "player_name",
            "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts", "ytd_made_cut_pct"
        ])

    # --- normalize outcome flags (only AFTER we've filtered to real results) ---
    for c in ["Win", "Top_10", "Top_25", "Made_Cut", "CUT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

    # one row per (player,event)
    df = df.drop_duplicates(subset=["dg_id", "event_id"], keep="last").copy()

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

    out["ytd_made_cut_pct"] = np.where(
        out["ytd_starts"] > 0,
        out["ytd_made_cuts"] / out["ytd_starts"],
        np.nan
    )
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
all_players_df = load_all_players()
ID_TO_IMG, NAME_TO_IMG = build_headshot_maps(all_players_df)
skills = load_player_skill()
buckets = load_approach_buckets()
rounds = load_rounds_minimal()

with st.sidebar:

    # --- schedule ordering ---
    sched = schedule.copy()

    date_col = "start_date" if "start_date" in sched.columns else ("event_date" if "event_date" in sched.columns else None)
    if date_col:
        sched[date_col] = pd.to_datetime(sched[date_col], errors="coerce")
        sched = sched.sort_values(date_col, na_position="last")
    else:
        sched = sched.reset_index(drop=True)

    # stable row selection
    sched = sched.reset_index(drop=True)
    sched["__row_id"] = sched.index.astype(int)

    # label shown in dropdown
    sched["__event_label"] = sched.get("event_name", "").astype(str)
    sched.loc[sched["__event_label"].str.strip().isin(["", "nan", "None"]), "__event_label"] = "Unknown event"

    # pick default = next upcoming (same logic you already have)
    today = pd.Timestamp.today().normalize()

    date_col2 = None
    if "start_date" in sched.columns and sched["start_date"].notna().any():
        date_col2 = "start_date"
    elif "event_date" in sched.columns and sched["event_date"].notna().any():
        date_col2 = "event_date"

    if date_col2:
        sched[date_col2] = pd.to_datetime(sched[date_col2], errors="coerce")
        is_upcoming = sched[date_col2] >= today
        sched = sched.assign(__is_upcoming=is_upcoming)
        sched = sched.sort_values(["__is_upcoming", date_col2], ascending=[False, True]).drop(columns="__is_upcoming")

        upcoming = sched.loc[sched[date_col2] >= today, "__row_id"].tolist()
        default_row_id = int(upcoming[0]) if len(upcoming) else int(sched["__row_id"].iloc[0])
    else:
        default_row_id = int(sched["__row_id"].iloc[0])

    # build a stable mapping for labels
    row_ids = [int(x) for x in sched["__row_id"].tolist()]
    label_by_id = dict(zip(row_ids, sched["__event_label"].tolist()))

    selected_row_id = st.selectbox(
        "Event",
        options=row_ids,
        format_func=lambda rid: label_by_id.get(int(rid), "Unknown event"),
        index=row_ids.index(int(default_row_id)),
        key="event_select_row_id",
    )

    selected_row = sched.loc[sched["__row_id"].astype(int) == int(selected_row_id)].iloc[0]

    course_num_val = pd.to_numeric(selected_row.get("course_num"), errors="coerce")
    course_num = int(course_num_val) if pd.notna(course_num_val) else None
    st.session_state["selected_course_num"] = course_num

    # image (optional) from schedule column "image"
    img_url = selected_row.get("image", None)
    img_url = None if img_url is None else str(img_url).strip()

    if img_url and img_url.lower() not in {"nan", "none", "null", "<unset>", ""}:
        st.image(img_url, use_container_width=True)


    def _clean_text(x) -> str:
        if x is None:
            return "—"
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null", "<unset>"}:
            return "—"
        return s


    def _as_money(x) -> str:
        v = pd.to_numeric(x, errors="coerce")
        return f"${v:,.0f}" if pd.notna(v) else "—"


    course = _clean_text(selected_row.get("course_name"))
    champ = _clean_text(selected_row.get("defending_champ"))
    purse = _as_money(selected_row.get("purse"))
    winshr = _as_money(selected_row.get("winner_share"))

    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 14px 14px;
            margin-top: 12px;
        ">
          <div style="font-size: 22px; font-weight: 800; margin-bottom: 10px;">Event Details</div>

          <div style="display:flex; justify-content:space-between; gap:12px; margin:6px 0;">
            <div style="opacity:0.72; font-size:13px; white-space:nowrap;">Course</div>
            <div style="font-weight:800; font-size:14px; text-align:right;">{course}</div>
          </div>

          <div style="display:flex; justify-content:space-between; gap:12px; margin:6px 0;">
            <div style="opacity:0.72; font-size:13px; white-space:nowrap;">Defending Champion</div>
            <div style="font-weight:800; font-size:14px; text-align:right;">{champ}</div>
          </div>

          <div style="display:flex; justify-content:space-between; gap:12px; margin:6px 0;">
            <div style="opacity:0.72; font-size:13px; white-space:nowrap;">Purse</div>
            <div style="font-weight:900; font-size:15px; text-align:right;">{purse}</div>
          </div>

          <div style="display:flex; justify-content:space-between; gap:12px; margin:6px 0;">
            <div style="opacity:0.72; font-size:13px; white-space:nowrap;">Winner&apos;s Share</div>
            <div style="font-weight:900; font-size:15px; text-align:right;">{winshr}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # event_id can be NA; that's OK
    event_id_val = pd.to_numeric(selected_row.get("event_id"), errors="coerce")
    event_id = int(event_id_val) if pd.notna(event_id_val) else None

    field_ids = []
    field_ev = pd.DataFrame()

    if event_id is not None and "event_id" in fields.columns:
        tmp = fields.copy()
        tmp["event_id"] = pd.to_numeric(tmp["event_id"], errors="coerce")
        tmp["dg_id"] = pd.to_numeric(tmp["dg_id"], errors="coerce")
        field_ev = tmp[(tmp["event_id"] == int(event_id))].dropna(subset=["dg_id"]).drop_duplicates(subset=["dg_id"])
        field_ids = field_ev["dg_id"].astype(int).tolist()

    # default True; we'll disable later if the selected event has no field rows
    if "only_in_field" not in st.session_state:
        st.session_state.only_in_field = True

    st.markdown("---")
    st.caption("Filters")
    only_in_field = st.toggle(
        "Only players in this week's field",
        value=st.session_state.only_in_field,
        key="only_in_field_toggle_sidebar",  # prevents duplicate-id issues if it appears elsewhere later
    )
    st.session_state.only_in_field = only_in_field

    if only_in_field and not field_ids:
        st.warning("Field Not Yet Released - Showing All Players Instead.")
        only_in_field = False
# -------------------------
# Player exclusion controls
# -------------------------
    if "excluded_players" not in st.session_state:
        st.session_state["excluded_players"] = []

    player_universe = st.session_state.get("player_universe", [])

    st.markdown("---")
    if not player_universe:
        st.caption("Player list will appear once Tab 1 loads for the selected event.")

    excluded = st.multiselect(
        "Filter OUT players",
        options=player_universe,
        key="excluded_players",  # widget owns the value
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
for w in (12, 24, 40):
    b = f"birdies_L{w}"
    e = f"eagles_or_better_L{w}"
    if b in rolling.columns and e in rolling.columns:
        rolling[f"birdies_or_better_L{w}"] = rolling[b].fillna(0) + rolling[e].fillna(0)

ytd = compute_ytd_from_fields_today(fields, year=2026)
if event_id is not None:
    fit = compute_approach_fit(base_ids, skills, buckets, event_id)
else:
    fit = pd.DataFrame({"dg_id": base_ids, "approach_fit_score": np.nan})


# Merge
out = base.merge(rolling, on="dg_id", how="left").merge(ytd, on=["dg_id", "player_name"], how="left").merge(fit, on="dg_id", how="left")
out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
out = out.dropna(subset=["dg_id"]).copy()
out["dg_id"] = out["dg_id"].astype(int)
out = out.drop_duplicates(subset=["dg_id"], keep="first")
# Round ALL numeric columns to 1 decimal for display consistency
num_cols = out.select_dtypes(include=[np.number]).columns
out[num_cols] = out[num_cols].round(1)

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

# ============================================================
# DISPLAY: unique + readable column names (Styler-safe)
# ============================================================
WINDOW_LABELS = {
    "L12": "Last 12 Rounds",
    "L24": "Last 24 Rounds",
    "L40": "Last 40 Rounds",
}

STAT_LABELS = {
    "sg_total": "SG Total",
    "sg_app": "SG Approach",
    "sg_putt": "SG Putting",
    "sg_ott": "SG Off the Tee",
    "sg_arg": "SG Around the Green",
    "sg_t2g": "SG Tee to Green",
    "round_score": "Round Score",
    "birdies_or_better": "Birdies or Better",
    "birdies": "Birdies",
    "eagles_or_better": "Eagles or Better",
}

def pretty_col(c: str, odds_col: str | None = None) -> str:
    if c == "player_name":
        return "Player"
    if odds_col and c == odds_col:
        return "Odds"

    # match patterns like "sg_total_L12"
    if "_L" in c:
        base, w = c.rsplit("_L", 1)
        w = f"L{w}"
        base_label = STAT_LABELS.get(base, base.replace("_", " ").title())
        w_label = WINDOW_LABELS.get(w, w)
        return f"{base_label} — {w_label}"

    # non-window cols
    if c.startswith("ytd_"):
        # example: ytd_top10 -> YTD Top10
        return c.replace("_", " ").upper()

    return c.replace("_", " ").title()

def render_table(
    title: str,
    df: pd.DataFrame,
    cols: list[str],
    gradient_cols: list[str],
    odds_col: str | None = None,
    height: int = 400,
):
    st.subheader(title)

    show = df[cols].copy()
    show = show.reset_index(drop=True)  # unique index for Styler

    # Build a UNIQUE rename map for just the columns in this table
    rename_map = {c: pretty_col(c, odds_col=odds_col) for c in show.columns}
    show = show.rename(columns=rename_map)

    # gradient subset must use renamed column names
    grad_cols = [rename_map.get(c, c) for c in gradient_cols if c in rename_map]

    # identify "lower is better" columns (Round Score only)
    low_is_good = [c for c in grad_cols if c.startswith("Round Score")]
    high_is_good = [c for c in grad_cols if c not in low_is_good]

    sty = show.style
    if high_is_good:
        sty = sty.background_gradient(subset=high_is_good, cmap="RdYlGn")
    if low_is_good:
        sty = sty.background_gradient(subset=low_is_good, cmap="RdYlGn_r")

    # 1 decimal formatting for numeric columns (excluding Player)
    fmt = {c: "{:.1f}" for c in show.columns if c != "Player" and pd.api.types.is_numeric_dtype(show[c])}
    sty = sty.format(fmt)

    st.dataframe(sty, use_container_width=True, height=height)

def heat_table(df: pd.DataFrame, sg_cols, precision=2):
    show = df.copy()

    # convert formatted "+1.23" strings back to numbers for coloring
    for c in sg_cols:
        if c in show.columns:
            show[c] = (
                show[c].astype(str)
                      .str.replace("+", "", regex=False)
                      .replace({"": None, "nan": None, "None": None})
            )
            show[c] = pd.to_numeric(show[c], errors="coerce")

    # build styler
    sty = show.style

    # center 0 so negatives/positives diverge (red↔green look)
    # pick a built-in cmap; "RdYlGn" is classic
    sty = sty.background_gradient(subset=[c for c in sg_cols if c in show.columns], cmap="RdYlGn", axis=None)

    # format back to +/- strings for display
    fmt = {c: (lambda x: "" if pd.isna(x) else f"{x:+.{precision}f}") for c in sg_cols if c in show.columns}
    sty = sty.format(fmt)

    return sty

# ============================================================
# summary_top (used by H2H + Deep Dive player pools)
# ============================================================
# Use the already-built merged table `out` (which already respects only_in_field via base_ids/base)
summary_top = out.copy()

# Give it a stable "top-to-bottom" sort for defaults in Tab4
_default_sort = "sg_total_L12" if "sg_total_L12" in summary_top.columns else None
if _default_sort:
    summary_top = summary_top.sort_values(_default_sort, ascending=False)

summary_top = summary_top.reset_index(drop=True)

tab1, tab2, tab3, tab4 = st.tabs(["Weekly", "Approach Buckets", "H2H", "Deep Dive"])

with tab1:
    st.subheader("Players (rolling L12/L24/L40)")

    # ----- sort selector (pretty labels, real columns) -----
    SORT_LABELS = {
        "sg_total_L12": "SG Total — Last 12 Rounds",
        "sg_total_L24": "SG Total — Last 24 Rounds",
        "sg_total_L40": "SG Total — Last 40 Rounds",
        "sg_app_L12": "SG Approach — Last 12 Rounds",
        "sg_putt_L12": "SG Putting — Last 12 Rounds",
        "ytd_wins": "YTD Wins",
        "ytd_top10": "YTD Top 10s",
    }

    sort_choices = [c for c in [
        "sg_total_L12", "sg_total_L24", "sg_total_L40",
        "sg_app_L12", "sg_putt_L12",
        "ytd_wins", "ytd_top10"
    ] if c in out.columns]

    sort_by = st.selectbox(
        "Sort by",
        options=sort_choices,
        index=0 if sort_choices else 0,
        format_func=lambda c: SORT_LABELS.get(c, c),
    )

    # use sort_by as before (still the real column name)
    out_show = out.copy()
    if sort_by and sort_by in out_show.columns:
        out_show = out_show.sort_values(sort_by, ascending=False)

    name_col = "player_name"
    odds = odds_col if odds_col else None

    # Apply "Only in field" from sidebar (uses dg_id)
    if st.session_state.get("only_in_field", False) and field_ids and "dg_id" in out_show.columns:
        out_show["dg_id"] = pd.to_numeric(out_show["dg_id"], errors="coerce")
        out_show = out_show[out_show["dg_id"].isin([int(x) for x in field_ids])]

    # --- Sidebar-driven filters ---
    player_q = st.session_state.get("player_q", "").strip()
    top_n = int(st.session_state.get("player_top_n", 0) or 0)
    min_sg_total_l24 = float(st.session_state.get("min_sg_total_l24", 0.0))
    min_sg_app_l24 = float(st.session_state.get("min_sg_app_l24", -99.0))
    use_odds_range = bool(st.session_state.get("use_odds_range", False))

    player_name_options = out_show[name_col].dropna().sort_values().unique().tolist()
    st.session_state["player_universe"] = player_name_options

    # Apply exclusions (filter OUT)
    excluded_players = st.session_state.get("excluded_players", [])
    if excluded_players:
        out_show = out_show[~out_show[name_col].isin(excluded_players)]


    if player_q:
        out_show = out_show[out_show[name_col].str.contains(player_q, case=False, na=False)]

    # Odds range (compute bounds from current df)
    if use_odds_range and odds and odds in out_show.columns:
        odds_series = pd.to_numeric(out_show[odds], errors="coerce").dropna()
        if not odds_series.empty:
            lo, hi = float(odds_series.min()), float(odds_series.max())
            odds_min, odds_max = st.slider(
                "Odds range",
                min_value=lo,
                max_value=hi,
                value=(lo, hi),
                key="odds_range_slider",
            )
            out_show[odds] = pd.to_numeric(out_show[odds], errors="coerce")
            out_show = out_show[out_show[odds].between(odds_min, odds_max)]

    # Thresholds
    if "sg_total_L24" in out_show.columns:
        out_show["sg_total_L24"] = pd.to_numeric(out_show["sg_total_L24"], errors="coerce")
        out_show = out_show[out_show["sg_total_L24"] >= min_sg_total_l24]

    if "sg_app_L24" in out_show.columns and min_sg_app_l24 > -50:  # treat -99 as "ignore"
        out_show["sg_app_L24"] = pd.to_numeric(out_show["sg_app_L24"], errors="coerce")
        out_show = out_show[out_show["sg_app_L24"] >= min_sg_app_l24]

    # Top N
    if top_n > 0:
        out_show = out_show.head(top_n)

    name_col = "player_name"
    odds = odds_col if odds_col else None

    # 1) Odds + SG Total
    total_cols = [name_col] + ([odds] if odds else []) + ["sg_total_L12", "sg_total_L24", "sg_total_L40"]
    total_grad = ["sg_total_L12", "sg_total_L24", "sg_total_L40"]
    render_table("Odds + SG Total", out_show, total_cols, total_grad, odds_col=odds_col, height=420)

    # 2) Approach
    app_cols = [name_col, "sg_app_L12", "sg_app_L24", "sg_app_L40"]
    app_grad = ["sg_app_L12", "sg_app_L24", "sg_app_L40"]
    render_table("Approach", out_show, app_cols, app_grad, odds_col=None, height=420)

    # 3) Putting
    putt_cols = [name_col, "sg_putt_L12", "sg_putt_L24", "sg_putt_L40"]
    putt_grad = ["sg_putt_L12", "sg_putt_L24", "sg_putt_L40"]
    render_table("Putting", out_show, putt_cols, putt_grad, odds_col=None, height=420)

    # 4) Off the tee
    ott_cols = [name_col, "sg_ott_L12", "sg_ott_L24", "sg_ott_L40"]
    ott_grad = ["sg_ott_L12", "sg_ott_L24", "sg_ott_L40"]
    render_table("Off the Tee", out_show, ott_cols, ott_grad, odds_col=None, height=420)

    # 5) Around the green
    arg_cols = [name_col, "sg_arg_L12", "sg_arg_L24", "sg_arg_L40"]
    arg_grad = ["sg_arg_L12", "sg_arg_L24", "sg_arg_L40"]
    render_table("Around the Green", out_show, arg_cols, arg_grad, odds_col=None, height=420)

    # 6) Tee to green
    t2g_cols = [name_col, "sg_t2g_L12", "sg_t2g_L24", "sg_t2g_L40"]
    t2g_grad = ["sg_t2g_L12", "sg_t2g_L24", "sg_t2g_L40"]
    render_table("Tee to Green", out_show, t2g_cols, t2g_grad, odds_col=None, height=420)

    # 7) Avg score + birdies/better
    score_cols = [
        name_col,
        "round_score_L12", "round_score_L24", "round_score_L40",
        "birdies_or_better_L12", "birdies_or_better_L24", "birdies_or_better_L40",
    ]
    score_grad = [
        "round_score_L12", "round_score_L24", "round_score_L40",
        "birdies_or_better_L12", "birdies_or_better_L24", "birdies_or_better_L40",
    ]
    render_table("Scoring (Avg Score + Birdies-or-Better)", out_show, score_cols, score_grad, odds_col=None, height=420)

    # 8) YTD (your upstream YTD calc must already respect cutoff)
    ytd_cols = [name_col, "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cut_pct"]
    ytd_grad = ["ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cut_pct"]
    render_table("Year to Date (completed events only; before cutoff)", out_show, ytd_cols, ytd_grad, odds_col=None, height=420)



with tab2:
    # Event name for the selected event_id (fallback to schedule label)
    event_name = None
    if event_id is not None:
        bname = buckets.loc[pd.to_numeric(buckets.get("event_id"), errors="coerce") == int(event_id), "event_name"]
        if "event_name" in buckets.columns and not bname.empty:
            event_name = str(bname.iloc[0])

    if not event_name:
        event_name = str(selected_row.get("__event_label", "Selected Event"))

    st.subheader(f"Approach Buckets in Yards for {event_name}")

    if event_id is None:
        st.info("No event_id for this schedule row yet.")
        st.stop()

    # ----------------------------
    # Pull the course bucket mix for this event
    # ----------------------------
    b = buckets[pd.to_numeric(buckets.get("event_id"), errors="coerce") == int(event_id)].copy()

    mix_cols = [c for c in ["50_100", "100_150", "150_200", "over_200"] if c in buckets.columns]
    if b.empty or not mix_cols:
        st.info("No approach bucket distribution found for this event_id in Approach Buckets.")
        mix = {c: np.nan for c in ["50–100", "100–150", "150–200", "Over 200"]}
    else:
        row = b.iloc[0]
        raw = {c: float(pd.to_numeric(row.get(c), errors="coerce") or 0.0) for c in mix_cols}
        total = sum(raw.values())
        if total > 0:
            raw = {k: v / total for k, v in raw.items()}

        mix = {
            "50–100": raw.get("50_100", np.nan),
            "100–150": raw.get("100_150", np.nan),
            "150–200": raw.get("150_200", np.nan),
            "Over 200": raw.get("over_200", np.nan),
        }
    # ----------------------------
    # Header row: course mix across top (colored high -> low)
    # ----------------------------
    mix_items = [
        ("50–100", mix.get("50–100", np.nan)),
        ("100–150", mix.get("100–150", np.nan)),
        ("150–200", mix.get("150–200", np.nan)),
        ("Over 200", mix.get("Over 200", np.nan)),
    ]

    # rank only non-nan values (high -> low)
    vals = [(lab, v) for lab, v in mix_items if pd.notna(v)]
    vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)

    # color ramp: highest -> lowest
    rank_colors = ["#2ecc71", "#a3e635", "#facc15", "#fb7185"]
    color_map = {lab: rank_colors[min(i, len(rank_colors) - 1)] for i, (lab, _) in enumerate(vals_sorted)}


    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%" if pd.notna(v) else "—"


    def _mix_card(label: str, value: float):
        col = color_map.get(label, "#e5e7eb")  # fallback
        st.markdown(
            f"""
            <div style="padding:6px 2px;">
              <div style="font-size:20px; color:#e5e7eb; margin-bottom:4px;">{label}</div>
              <div style="font-size:64px; font-weight:800; line-height:1; color:{col};">{_pct(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _mix_card("50–100", mix.get("50–100", np.nan))
    with c2:
        _mix_card("100–150", mix.get("100–150", np.nan))
    with c3:
        _mix_card("150–200", mix.get("150–200", np.nan))
    with c4:
        _mix_card("Over 200", mix.get("Over 200", np.nan))

    st.markdown("---")

    # ----------------------------
    # Player table (this week's field if toggle on; otherwise universe)
    # Uses the same base_ids you already computed above.
    # ----------------------------

    # Expect these columns in app_skill.xlsx
    # (Shots + Value per bucket)
    col_map = {
        "50 – 100 SG": "50_100_fw_value",
        "100 – 150 SG": "100_150_fw_value",
        "150 – 200 SG": "150_200_fw_value",
        "Over 200 SG": "over_200_fw_value",
    }

    s = skills.copy()
    s["dg_id"] = pd.to_numeric(s.get("dg_id"), errors="coerce")
    s = s[s["dg_id"].isin([int(x) for x in base_ids])].copy()

    # Keep player name column consistent with your app
    # (your skills file uses player_name)
    if "player_name" not in s.columns and "Player" in s.columns:
        s["player_name"] = s["Player"]

    # Build display frame
    keep_cols = ["player_name"] + [v for v in col_map.values() if v in s.columns]
    s = s[keep_cols].copy()

    # Rename to friendly headers
    rename_map = {"player_name": "Player"}
    rename_map.update({v: k for k, v in col_map.items() if v in s.columns})
    s = s.rename(columns=rename_map)

    # Coerce types (values only)
    val_cols = [c for c in s.columns if c.endswith("SG")]
    for c in val_cols:
        s[c] = pd.to_numeric(s[c], errors="coerce")

    # Scale SG to more readable units
    SG_SCALE = 100.0
    if val_cols:
        s[val_cols] = s[val_cols] * SG_SCALE

    # Note above the selector
    st.caption(f"Note: SG columns are scaled to SG per 100 shots (original values ×{int(SG_SCALE)}).")
    st.caption(
        "None indicates shots recorded in this distance bucket (often DP World Tour players or rookies).")

    # Sort by a chosen bucket value
    default_sort = "100 – 150 SG" if "100 – 150 SG" in val_cols else (val_cols[0] if val_cols else None)

    sort_choice = st.selectbox(
        "Sort by",
        options=val_cols,
        index=(val_cols.index(default_sort) if default_sort else 0),
        key="tab2_sort_by",
    )

    if sort_choice and sort_choice in s.columns:
        s = s.sort_values(sort_choice, ascending=False)

    # Style + format (now scaled)
    sty = s.style
    if val_cols:
        sty = sty.background_gradient(subset=val_cols, cmap="RdYlGn")

    # If you're using "per 100 shots", 1 decimal usually reads better.
    sty = sty.format({c: "{:.1f}" for c in val_cols})

    st.dataframe(sty, use_container_width=True, height=800)


with tab3:
    st.header("Head-to-Head Comparison")

    if event_id is None:
        st.info("No event_id for this schedule row yet.")
        st.stop()

    rounds_2024p = load_mst_roundlevel_2024_present()
    ev_2017_2023 = load_mst_eventlevel_2017_2023()

    cutoff_dt = get_pre_event_cutoff_date(sched, int(event_id))

    # pool from summary_top (already reflects your field filtering in mst.py)
    pool = summary_top[["dg_id", "player_name"]].dropna().drop_duplicates().copy()
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    # --- build name<->id maps FROM POOL (authoritative for this event/field) ---
    name_to_id = dict(zip(pool["player_name"], pool["dg_id"]))
    id_to_name = dict(zip(pool["dg_id"], pool["player_name"]))

    player_options = sorted(pool["player_name"].unique().tolist())

    # defaults (first two players)
    default_a = 0
    default_b = 1 if len(player_options) > 1 else 0

    if len(pool) < 2:
        st.info("Need at least two players available to compare.")
        st.stop()

    pool["label"] = pool["player_name"]
    id_to_label = dict(zip(pool["dg_id"], pool["label"]))
    id_to_name = dict(zip(pool["dg_id"], pool["player_name"]))

    opts = pool["dg_id"].tolist()

    selA, selB = st.columns(2, gap="large")

    with selA:
        st.markdown("### Player A")
        a1, a2 = st.columns([6, 2], vertical_alignment="center")

        with a1:
            name_a = st.selectbox(" ", player_options, index=default_a, key="h2h_a", label_visibility="collapsed")

        with a2:
            dg_a = int(name_to_id[name_a])
            url_a = get_headshot_url(dg_a, name_a, ID_TO_IMG, NAME_TO_IMG)
            show_headshot(url_a, width=90)

    with selB:
        st.markdown("### Player B")
        b1, b2 = st.columns([6, 2], vertical_alignment="center")

        with b1:
            name_b = st.selectbox(" ", player_options, index=default_b, key="h2h_b", label_visibility="collapsed")

        with b2:
            dg_b = int(name_to_id[name_b])
            url_b = get_headshot_url(dg_b, name_b, ID_TO_IMG, NAME_TO_IMG)
            show_headshot(url_b, width=90)

    dg_a = int(name_to_id[name_a])
    dg_b = int(name_to_id[name_b])

    if dg_a == dg_b:
        st.warning("Pick two different players.")
        st.stop()

    st.subheader("Recent tournaments")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"#### {name_a}")
        tA = build_last_n_events_table_mst(rounds_2024p, ev_2017_2023, dg_a, n=25, date_max=cutoff_dt)
        st.dataframe(
            tA[["Event", "Finish", "SG Total", "Year"]] if not tA.empty else tA,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown(f"#### {name_b}")
        tB = build_last_n_events_table_mst(rounds_2024p, ev_2017_2023, dg_b, n=25, date_max=cutoff_dt)
        st.dataframe(
            tB[["Event", "Finish", "SG Total", "Year"]] if not tB.empty else tB,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ---- SG per round line chart (ONLY 2024+ because that’s all you have at round-level)
    st.subheader("Last 40 Strokes Gained (Smoothed)")

    if "sg_total" not in rounds_2024p.columns:
        st.warning("roundlevel_2024_present is missing sg_total; cannot plot.")
    else:
        ra = _last_n_rounds_pre_event(rounds_2024p, dg_a, cutoff_dt, n=40)
        rb = _last_n_rounds_pre_event(rounds_2024p, dg_b, cutoff_dt, n=40)

        if ra.empty or rb.empty:
            st.info("Not enough 2024+ round data to plot both players.")
        else:
            ra["sg"] = pd.to_numeric(ra["sg_total"], errors="coerce")
            rb["sg"] = pd.to_numeric(rb["sg_total"], errors="coerce")
            ra = ra.dropna(subset=["sg"]).copy()
            rb = rb.dropna(subset=["sg"]).copy()

            ra["round_index"] = range(1, len(ra) + 1)
            rb["round_index"] = range(1, len(rb) + 1)
            ra["player"] = name_a
            rb["player"] = name_b

            plot_df = pd.concat([ra[["round_index", "sg", "player"]],
                                 rb[["round_index", "sg", "player"]]], ignore_index=True)

            # ---- smoothing controls ----
            smooth_window = st.slider("Smoothing (moving average window)", 1, 15, 5, key="mst_smooth_w")

            plot_df = plot_df.sort_values(["player", "round_index"]).copy()
            plot_df["sg_smooth"] = (
                plot_df.groupby("player")["sg"]
                .transform(lambda s: s.rolling(window=smooth_window, min_periods=1).mean())
            )

            fig = px.line(
                plot_df,
                x="round_index",
                y="sg_smooth",
                color="player",
                markers=False,
            )

            # Force colors (one non-blue)
            # NOTE: trace order follows legend order; first trace gets first color, etc.
            fig.update_traces(line=dict(width=3))
            fig.update_traces(selector=dict(name=name_a), line=dict(color="orange"))
            fig.update_traces(selector=dict(name=name_b), line=dict(color="deepskyblue"))

            # Wider feel + legend under chart
            fig.update_layout(
                height=600,  # bigger chart area
                margin=dict(l=20, r=20, t=30, b=80),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    title_text=""
                ),
            )
            fig.update_yaxes(zeroline=True, title="SG (smoothed)")
            fig.update_xaxes(title="round_index")

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Performance by season: 2017-2023 eventlevel + 2024+ aggregated from roundlevel
    st.subheader("Performance by season")

    def _perf_by_season_mst(dg_id: int) -> pd.DataFrame:
        rows = []

        # 2017-2023 from eventlevel
        e = ev_2017_2023.copy()
        if not e.empty and "dg_id" in e.columns:
            e["dg_id"] = pd.to_numeric(e["dg_id"], errors="coerce")
            e = e.loc[e["dg_id"] == int(dg_id)].copy()

            if not e.empty:
                ends_e = _event_end_table_eventlevel(ev_2017_2023)
                e["year"] = pd.to_numeric(e.get("year"), errors="coerce")
                e["event_id"] = pd.to_numeric(e.get("event_id"), errors="coerce")
                e = e.dropna(subset=["year", "event_id"]).copy()
                e["year"] = e["year"].astype(int)
                e["event_id"] = e["event_id"].astype(int)
                e = e.merge(ends_e, on=["year", "event_id"], how="left")
                e["event_end"] = pd.to_datetime(e["event_end"], errors="coerce")

                if cutoff_dt is not None and pd.notna(cutoff_dt):
                    e = e.loc[e["event_end"].notna() & (e["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

                # starts
                starts = e.groupby("year")["event_id"].nunique().rename("starts")

                # wins if finish_num exists
                wins = pd.Series(0, index=starts.index)
                if "finish_num" in e.columns:
                    fn = pd.to_numeric(e["finish_num"], errors="coerce")
                    w = e.loc[fn == 1].groupby("year")["event_id"].nunique()
                    wins.loc[w.index] = w

                # average SG (best effort)
                sg_col = "sg_total" if "sg_total" in e.columns else None
                avg_sg = e.groupby("year")[sg_col].mean() if sg_col else pd.Series(np.nan, index=starts.index)

                tmp = pd.DataFrame({
                    "year": starts.index.astype(int),
                    "starts": starts.values,
                    "wins": wins.values,
                    "avg_sg_per_event": avg_sg.reindex(starts.index).values,
                })
                rows.append(tmp)

        # 2024+ from roundlevel aggregated to event then to year
        r = rounds_2024p.copy()
        if not r.empty:
            r["dg_id"] = pd.to_numeric(r.get("dg_id"), errors="coerce")
            r = r.loc[r["dg_id"] == int(dg_id)].copy()
            if not r.empty:
                ends_r = _event_end_table_roundlevel(rounds_2024p)
                r["year"] = pd.to_numeric(r.get("year"), errors="coerce")
                r["event_id"] = pd.to_numeric(r.get("event_id"), errors="coerce")
                r = r.dropna(subset=["year", "event_id"]).copy()
                r["year"] = r["year"].astype(int)
                r["event_id"] = r["event_id"].astype(int)
                r = r.merge(ends_r, on=["year", "event_id"], how="left")
                r["event_end"] = pd.to_datetime(r["event_end"], errors="coerce")

                if cutoff_dt is not None and pd.notna(cutoff_dt):
                    r = r.loc[r["event_end"].notna() & (r["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

                # event-level totals
                # event-level per-round averages (NOT totals)
                if "sg_total" in r.columns:
                    ev_pr = (
                        r.groupby(["year", "event_id"], as_index=False)["sg_total"]
                        .mean()
                        .rename(columns={"sg_total": "sg_event_pr"})
                    )
                else:
                    ev_pr = r.groupby(["year", "event_id"], as_index=False).size().rename(
                        columns={"size": "sg_event_pr"})

                starts = ev_pr.groupby("year")["event_id"].nunique().rename("starts")
                avg_sg = ev_pr.groupby("year")["sg_event_pr"].mean().rename("avg_sg_per_event")

                wins = pd.Series(0, index=starts.index)
                if "finish_num" in r.columns:
                    fn = pd.to_numeric(r["finish_num"], errors="coerce")
                    # best finish per event from round rows
                    best = (
                        r.assign(finish_num_n=fn)
                         .groupby(["year", "event_id"], as_index=False)["finish_num_n"]
                         .min()
                    )
                    w = best.loc[best["finish_num_n"] == 1].groupby("year")["event_id"].nunique()
                    wins.loc[w.index] = w

                tmp = pd.DataFrame({
                    "year": starts.index.astype(int),
                    "starts": starts.values,
                    "wins": wins.values,
                    "avg_sg_per_event": avg_sg.reindex(starts.index).values,
                })
                rows.append(tmp)

        if not rows:
            return pd.DataFrame(columns=["year", "starts", "wins", "avg_sg_per_event"])

        out = pd.concat(rows, ignore_index=True)
        out = out.groupby("year", as_index=False).agg(
            starts=("starts", "sum"),
            wins=("wins", "sum"),
            avg_sg_per_event=("avg_sg_per_event", "mean"),
        )
        out = out.sort_values("year", ascending=False).reset_index(drop=True)
        return out


    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"#### {name_a}")
        st.dataframe(_perf_by_season_mst(dg_a), use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"#### {name_b}")
        st.dataframe(_perf_by_season_mst(dg_b), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Stat comparison table (last 40 rounds pre-event; 2024+ only)
    st.subheader("Stat comparison (last 40 rounds)")

    stats = [
        "sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt",
        "driving_dist", "driving_acc",
        "gir", "scrambling", "prox_rgh", "prox_fw",
        "great_shots", "poor_shots",
        "birdies", "bogies", "doubles_or_worse",
        "round_score",
    ]
    stats = [c for c in stats if c in rounds_2024p.columns]
    lower_better = {"round_score", "bogies", "doubles_or_worse", "poor_shots", "prox_rgh", "prox_fw"}

    ra40 = _last_n_rounds_pre_event(rounds_2024p, dg_a, cutoff_dt, n=40)
    rb40 = _last_n_rounds_pre_event(rounds_2024p, dg_b, cutoff_dt, n=40)

    def _means(df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in stats:
            out[c] = float(pd.to_numeric(df.get(c), errors="coerce").mean()) if df is not None and not df.empty else np.nan
        return out

    ma = _means(ra40)
    mb = _means(rb40)

    rows = []
    for c in stats:
        va, vb = ma.get(c, np.nan), mb.get(c, np.nan)
        win = ""
        if np.isfinite(va) and np.isfinite(vb) and va != vb:
            if c in lower_better:
                win = "◀" if va < vb else "▶"
            else:
                win = "◀" if va > vb else "▶"
        rows.append([c, va, win, vb])

    comp = pd.DataFrame(rows, columns=["stat", name_a, "winner", name_b])

    # round displayed numbers to 1 decimal (keep winner arrows as-is)
    for col in [name_a, name_b]:
        comp[col] = pd.to_numeric(comp[col], errors="coerce").round(1)


    def _highlight_better_value_cell(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # If winner == "◀", Player A (name_a) is better; if "▶", Player B (name_b) is better
        mask_a = df["winner"] == "◀"
        mask_b = df["winner"] == "▶"

        styles.loc[mask_a, name_a] = "background-color: rgba(255,255,255,0.18); font-weight: 700;"
        styles.loc[mask_b, name_b] = "background-color: rgba(255,255,255,0.18); font-weight: 700;"

        # optional: center the arrow column
        styles.loc[:, "winner"] = "text-align: center;"
        return styles


    sty = (
        comp.style
        .apply(_highlight_better_value_cell, axis=None)
        .format({name_a: "{:.1f}", name_b: "{:.1f}"})
    )

    # --- format to 1 decimal + highlight the better value cell (not the winner arrow col) ---
    lower_better = {"round_score", "bogies", "doubles_or_worse", "poor_shots", "prox_rgh", "prox_fw"}

    a_col = name_a
    b_col = name_b


    def _highlight_winner_row(row):
        stat = row["stat"]
        va = row[a_col]
        vb = row[b_col]

        styles = {c: "" for c in row.index}

        # only highlight if both numbers exist
        if pd.notna(va) and pd.notna(vb):
            if stat in lower_better:
                better_a = va < vb
                better_b = vb < va
            else:
                better_a = va > vb
                better_b = vb > va

            if better_a:
                styles[a_col] = "background-color: rgba(0, 200, 0, 0.25); font-weight: 700;"
            elif better_b:
                styles[b_col] = "background-color: rgba(0, 200, 0, 0.25); font-weight: 700;"

        return [styles[c] for c in row.index]


    sty = (
        comp.style
        .apply(_highlight_winner_row, axis=1)
        .format({a_col: "{:.1f}", b_col: "{:.1f}"})
        .set_properties(subset=[a_col, "winner", b_col], **{"text-align": "center"})
    )

    st.dataframe(sty, use_container_width=True, hide_index=True, height=670)

with tab4:
    st.header("Player Deep Dive")

    rounds_2024p = load_mst_roundlevel_2024_present()
    ev_2017_2023 = load_mst_eventlevel_2017_2023()
    cutoff_dt = get_pre_event_cutoff_date(sched, int(event_id))

    # -------------------------
    # Player selector (use current pool)
    # -------------------------
    pool = summary_top[["dg_id", "player_name"]].dropna().drop_duplicates().copy()
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    pool["label"] = pool["player_name"] + " — " + pool["dg_id"].astype(str)
    labels = pool["label"].tolist()
    label_to_id = dict(zip(labels, pool["dg_id"]))
    label_to_name = dict(zip(pool["dg_id"], pool["player_name"]))

    default_idx = 0
    if isinstance(summary_top, pd.DataFrame) and not summary_top.empty and "dg_id" in summary_top.columns:
        top_dg = int(summary_top.iloc[0]["dg_id"])
        for i, lab in enumerate(labels):
            if int(label_to_id[lab]) == top_dg:
                default_idx = i
                break

    sel_label = st.selectbox("Player", labels, index=default_idx, key="mst_dd_player")
    dg_id_sel = int(label_to_id[sel_label])
    player_name_sel = str(label_to_name.get(dg_id_sel, f"dg_{dg_id_sel}"))

    # --- Title row: name + headshot (mobile-friendly)
    title_col, img_col = st.columns([10, 2], vertical_alignment="center")

    with title_col:
        st.header(player_name_sel)

    with img_col:
        url = get_headshot_url(dg_id_sel, player_name_sel, ID_TO_IMG, NAME_TO_IMG)
        show_headshot(url, width=180)

    st.subheader("Course History")

    course_num = st.session_state.get("selected_course_num", None)

    if course_num is None:
        st.info("Course history unavailable (no course_num for the selected event).")
    else:
        out_parts = []

        # -------------------------
        # A) 2024+ from roundlevel
        # -------------------------
        r = rounds_2024p.copy()
        r["dg_id"] = pd.to_numeric(r.get("dg_id"), errors="coerce")
        r = r.loc[r["dg_id"] == int(dg_id_sel)].copy()

        if "course_num" not in r.columns:
            st.info("Course history unavailable for 2024+ (roundlevel has no course_num column).")
        else:
            r["course_num"] = pd.to_numeric(r["course_num"], errors="coerce")
            r = r.loc[r["course_num"] == int(course_num)].copy()

            if not r.empty:
                # add event_end so we can apply cutoff + sort cleanly
                ends_r = _event_end_table_roundlevel(rounds_2024p)
                r["year"] = pd.to_numeric(r.get("year"), errors="coerce")
                r["event_id"] = pd.to_numeric(r.get("event_id"), errors="coerce")
                r = r.dropna(subset=["year", "event_id"]).copy()
                r["year"] = r["year"].astype(int)
                r["event_id"] = r["event_id"].astype(int)

                r = r.merge(ends_r, on=["year", "event_id"], how="left")
                r["event_end"] = pd.to_datetime(r["event_end"], errors="coerce")

                if cutoff_dt is not None and pd.notna(cutoff_dt):
                    r = r.loc[r["event_end"].notna() & (r["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

                fin_col = "fin_text" if "fin_text" in r.columns else (
                    "finish_text" if "finish_text" in r.columns else None)
                if fin_col is None:
                    r["fin_text"] = ""
                    fin_col = "fin_text"


                def _first_non_null_str(s: pd.Series) -> str:
                    s2 = s.dropna().astype(str)
                    return s2.iloc[0] if len(s2) else ""


                agg_kwargs = {
                    "Finish": (fin_col, _first_non_null_str),
                }
                if "sg_total" in r.columns: agg_kwargs["SG Total"] = ("sg_total", "mean")
                if "sg_app" in r.columns:   agg_kwargs["SG APP"] = ("sg_app", "mean")
                if "sg_ott" in r.columns:   agg_kwargs["SG OTT"] = ("sg_ott", "mean")
                if "sg_arg" in r.columns:   agg_kwargs["SG ARG"] = ("sg_arg", "mean")
                if "sg_putt" in r.columns:  agg_kwargs["SG PUTT"] = ("sg_putt", "mean")

                t24 = (
                    r.groupby(["year", "event_id", "event_name"], as_index=False)
                    .agg(**agg_kwargs)
                    .rename(columns={"year": "Year", "event_name": "Event"})
                )
                t24["event_end"] = r.groupby(["year", "event_id"])["event_end"].max().values
                t24["source"] = "2024+ roundlevel"
                out_parts.append(t24)

        # -------------------------
        # B) 2017-2023 from eventlevel means file
        # -------------------------
        e = ev_2017_2023.copy()
        e["dg_id"] = pd.to_numeric(e.get("dg_id"), errors="coerce")
        e = e.loc[e["dg_id"] == int(dg_id_sel)].copy()

        if e.empty:
            pass
        elif "course_num" not in e.columns:
            st.info("Course history unavailable for 2017–2023 (eventlevel has no course_num column).")
        else:
            e["course_num"] = pd.to_numeric(e["course_num"], errors="coerce")
            e = e.loc[e["course_num"] == int(course_num)].copy()

            if not e.empty:
                # apply cutoff + attach event_end for sorting
                ends_e = _event_end_table_eventlevel(ev_2017_2023)
                e["year"] = pd.to_numeric(e.get("year"), errors="coerce")
                e["event_id"] = pd.to_numeric(e.get("event_id"), errors="coerce")
                e = e.dropna(subset=["year", "event_id"]).copy()
                e["year"] = e["year"].astype(int)
                e["event_id"] = e["event_id"].astype(int)

                e = e.merge(ends_e, on=["year", "event_id"], how="left")
                e["event_end"] = pd.to_datetime(e["event_end"], errors="coerce")

                if cutoff_dt is not None and pd.notna(cutoff_dt):
                    e = e.loc[e["event_end"].notna() & (e["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

                name_col = "event_name" if "event_name" in e.columns else (
                    "tournament" if "tournament" in e.columns else None)
                if name_col is None:
                    e["event_name"] = e.get("event_id", "").astype(str)
                    name_col = "event_name"

                fin_col = "fin_text" if "fin_text" in e.columns else (
                    "finish_text" if "finish_text" in e.columns else None)
                if fin_col is None:
                    e["fin_text"] = ""
                    fin_col = "fin_text"

                cols = ["year", "event_id", name_col, fin_col, "event_end"]
                for c in ["sg_total", "sg_app", "sg_ott", "sg_arg", "sg_putt"]:
                    if c in e.columns:
                        cols.append(c)

                t17 = e[cols].copy().rename(columns={
                    "year": "Year",
                    name_col: "Event",
                    fin_col: "Finish",
                    "sg_total": "SG Total",
                    "sg_app": "SG APP",
                    "sg_ott": "SG OTT",
                    "sg_arg": "SG ARG",
                    "sg_putt": "SG PUTT",
                })
                t17["source"] = "2017–2023 eventlevel"
                out_parts.append(t17)

        # -------------------------
        # Combine + display
        # -------------------------
        if not out_parts:
            st.info("No course history found for this player at this course.")
        else:
            course_hist = pd.concat(out_parts, ignore_index=True)
            course_hist["event_end"] = pd.to_datetime(course_hist.get("event_end"), errors="coerce")
            course_hist = course_hist.sort_values(["event_end", "Year", "event_id"],
                                                  ascending=[False, False, False]).copy()

            # format SG columns
            for c in ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]:
                if c in course_hist.columns:
                    course_hist[c] = pd.to_numeric(course_hist[c], errors="coerce").map(
                        lambda x: f"{x:+.2f}" if pd.notna(x) else ""
                    )

            show_cols = [c for c in
                         ["Year", "Event", "Finish", "SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]
                         if c in course_hist.columns]
            sg_cols = ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]
            sty = heat_table(course_hist[show_cols], sg_cols=sg_cols, precision=2)
            st.dataframe(sty, use_container_width=True, hide_index=True)

    st.subheader("Last 60 Rounds")
    st.caption("Rounds without bars indicate non-PGA Tour events (SG breakdown not available).")

    # choose what to plot
    metric_mode = st.selectbox(
        "Chart",
        ["Stacked SG components + SG Total line", "Single metric line (legacy)"],
        index=0,
        key="mst_dd_chart_mode",
    )

    df = _last_n_rounds_pre_event(rounds_2024p, dg_id_sel, cutoff_dt, n=60)
    if df.empty:
        st.info("Not enough rounds pre-event for this player.")
    else:
        df = df.copy()
        df["round_index"] = range(1, len(df) + 1)

        # Coerce required cols
        for c in ["sg_total", "sg_app", "sg_ott", "sg_arg", "sg_putt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if metric_mode == "Stacked SG components + SG Total line":
            # require at least sg_total for the line
            if "sg_total" not in df.columns:
                st.info("sg_total not available to chart.")
            else:
                base = df.copy()
                base["round_index"] = range(1, len(base) + 1)

                # -------------------------
                # 1) LINE: show SG Total wherever it's available
                # -------------------------
                line_df = base.copy()
                line_df["sg_total"] = pd.to_numeric(line_df["sg_total"], errors="coerce")
                line_df = line_df.dropna(subset=["sg_total"]).copy()

                # -------------------------
                # 2) BARS: only where all 4 components exist (no fake zeros)
                # -------------------------
                comps = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
                have_all_comps = all(c in base.columns for c in comps)

                long = None
                bar_df = pd.DataFrame()

                if have_all_comps:
                    bar_df = base.copy()
                    for c in comps:
                        bar_df[c] = pd.to_numeric(bar_df[c], errors="coerce")

                    # Only keep rounds where ALL components are present
                    bar_df = bar_df.dropna(subset=comps).copy()

                    if not bar_df.empty:
                        # Total from components (for bar height)
                        bar_df["sg_total_calc"] = bar_df[comps].sum(axis=1)

                        # If everything is literally zero, skip bars
                        abs_sum = (
                                bar_df["sg_ott"].abs()
                                + bar_df["sg_app"].abs()
                                + bar_df["sg_arg"].abs()
                                + bar_df["sg_putt"].abs()
                        )
                        bar_df = bar_df.loc[abs_sum > 0].copy()

                        label_map = {
                            "seg_ott": "SG OTT",
                            "seg_app": "SG APP",
                            "seg_arg": "SG ARG",
                            "seg_putt": "SG PUTT",
                        }
                        long["component"] = long["component"].map(label_map)

                # -------------------------
                # 3) PLOT: TRUE stacked bars (components) + SG total line
                # -------------------------
                fig = go.Figure()

                # x as categorical so bars stack at each index
                line_x = line_df["round_index"].astype(int).astype(str)

                # Bars: only where we have all components (bar_df already filtered)
                if have_all_comps and (bar_df is not None) and (not bar_df.empty):
                    bar_x = bar_df["round_index"].astype(int).astype(str)

                    # IMPORTANT: stack the REAL components (not seg_*), so it’s a real SG breakdown
                    BAR_COLS = {
                        "SG OTT": "sg_ott",
                        "SG APP": "sg_app",
                        "SG ARG": "sg_arg",
                        "SG PUTT": "sg_putt",
                    }

                    COLOR_MAP = {
                        "SG OTT": "rgba(120, 180, 255, 0.55)",
                        "SG APP": "rgba(255, 170, 190, 0.55)",
                        "SG ARG": "rgba(190, 150, 255, 0.55)",
                        "SG PUTT": "rgba(255, 190, 120, 0.55)",
                    }

                    for label, col in BAR_COLS.items():
                        if col in bar_df.columns:
                            fig.add_trace(
                                go.Bar(
                                    x=bar_x,
                                    y=pd.to_numeric(bar_df[col], errors="coerce"),
                                    name=label,
                                    marker=dict(color=COLOR_MAP.get(label)),
                                )
                            )

                # SG Total line (always)
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_df["sg_total"],
                        mode="lines+markers",
                        name="SG Total",
                    )
                )

                fig.update_layout(
                    barmode="relative",  # TRUE stacking
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend_title_text="",
                )

                fig.update_yaxes(zeroline=True)
                fig.update_xaxes(type="category")  # force categorical axis

                st.plotly_chart(fig, use_container_width=True)

                if (not have_all_comps) or (bar_df is None) or bar_df.empty:
                    st.caption("Bars hidden on rounds where SG component breakdown (OTT/APP/ARG/PUTT) is unavailable.")



        else:
            metric_options = [c for c in ["sg_total", "sg_t2g", "sg_app", "sg_ott", "sg_arg", "sg_putt", "round_score"]
                              if c in df.columns]
            if not metric_options:
                st.info("No metrics found in roundlevel_2024_present to chart.")
            else:
                metric = st.selectbox("Metric", metric_options, index=0, key="mst_dd_metric")
                df["metric"] = pd.to_numeric(df.get(metric), errors="coerce")
                df2 = df.dropna(subset=["metric"]).copy()

                if df2.empty:
                    st.info("No data for this metric.")
                else:
                    fig = px.line(df2, x="round_index", y="metric", markers=True)
                    fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
                    fig.update_yaxes(zeroline=True)
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------
    # Tournament breakdown: 2024+ (event totals) + 2017-2023 (eventlevel rows)
    # -------------------------
    st.subheader("Tournament breakdown")

    # 2024+ aggregate to event totals
    r = rounds_2024p.copy()
    r["dg_id"] = pd.to_numeric(r.get("dg_id"), errors="coerce")
    r = r.loc[r["dg_id"] == int(dg_id_sel)].copy()

    out_parts = []

    if not r.empty:
        ends_r = _event_end_table_roundlevel(rounds_2024p)
        r["year"] = pd.to_numeric(r.get("year"), errors="coerce")
        r["event_id"] = pd.to_numeric(r.get("event_id"), errors="coerce")
        r = r.dropna(subset=["year", "event_id"]).copy()
        r["year"] = r["year"].astype(int)
        r["event_id"] = r["event_id"].astype(int)
        r = r.merge(ends_r, on=["year", "event_id"], how="left")
        r["event_end"] = pd.to_datetime(r["event_end"], errors="coerce")

        if cutoff_dt is not None and pd.notna(cutoff_dt):
            r = r.loc[r["event_end"].notna() & (r["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

        fin_col = "fin_text" if "fin_text" in r.columns else ("finish_text" if "finish_text" in r.columns else None)
        if fin_col is None:
            r["fin_text"] = ""
            fin_col = "fin_text"

        def _first_non_null_str(s: pd.Series) -> str:
            s2 = s.dropna().astype(str)
            return s2.iloc[0] if len(s2) else ""


        agg_cols = {
            "Finish": (fin_col, _first_non_null_str),
            "SG Total": ("sg_total", "mean") if "sg_total" in r.columns else ("event_id", "size"),
            "SG APP": ("sg_app", "mean") if "sg_app" in r.columns else None,
            "SG OTT": ("sg_ott", "mean") if "sg_ott" in r.columns else None,
            "SG ARG": ("sg_arg", "mean") if "sg_arg" in r.columns else None,
            "SG PUTT": ("sg_putt", "mean") if "sg_putt" in r.columns else None,
        }

        agg_cols = {k: v for k, v in agg_cols.items() if v is not None}

        t24 = (
            r.groupby(["year", "event_id", "event_name"], as_index=False)
             .agg(**agg_cols)
             .rename(columns={"year": "Year", "event_name": "Event"})
        )
        t24["event_end"] = r.groupby(["year", "event_id"])["event_end"].max().values
        t24["source"] = "2024+ roundlevel (event totals)"
        out_parts.append(t24)

    # 2017-2023 eventlevel
    e = ev_2017_2023.copy()
    e["dg_id"] = pd.to_numeric(e.get("dg_id"), errors="coerce")
    e = e.loc[e["dg_id"] == int(dg_id_sel)].copy()

    if not e.empty:
        ends_e = _event_end_table_eventlevel(ev_2017_2023)
        e["year"] = pd.to_numeric(e.get("year"), errors="coerce")
        e["event_id"] = pd.to_numeric(e.get("event_id"), errors="coerce")
        e = e.dropna(subset=["year", "event_id"]).copy()
        e["year"] = e["year"].astype(int)
        e["event_id"] = e["event_id"].astype(int)
        e = e.merge(ends_e, on=["year", "event_id"], how="left")
        e["event_end"] = pd.to_datetime(e["event_end"], errors="coerce")

        if cutoff_dt is not None and pd.notna(cutoff_dt):
            e = e.loc[e["event_end"].notna() & (e["event_end"] <= pd.to_datetime(cutoff_dt))].copy()

        name_col = "event_name" if "event_name" in e.columns else ("tournament" if "tournament" in e.columns else None)
        if name_col is None:
            e["event_name"] = e.get("event_id", "").astype(str)
            name_col = "event_name"

        fin_col = "fin_text" if "fin_text" in e.columns else ("finish_text" if "finish_text" in e.columns else None)
        if fin_col is None:
            e["fin_text"] = ""
            fin_col = "fin_text"

        cols = ["year", "event_id", name_col, fin_col, "event_end"]
        if "sg_total" in e.columns:
            cols.append("sg_total")
        t17 = e[cols].copy()
        t17 = t17.rename(columns={"year": "Year", name_col: "Event", fin_col: "Finish", "sg_total": "SG Total"})
        t17["source"] = "2017-2023 eventlevel"
        out_parts.append(t17)

    if not out_parts:
        st.info("No tournament breakdown available.")
    else:
        out = pd.concat(out_parts, ignore_index=True)
        out["event_end"] = pd.to_datetime(out.get("event_end"), errors="coerce")
        out = out.sort_values(["event_end", "Year", "event_id"], ascending=[False, False, False]).head(30).copy()

        # format SG columns
        for c in ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")

        show_cols = [c for c in ["Year", "Event", "Finish", "SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"] if c in out.columns]
        sg_cols = ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]
        sty = heat_table(out[show_cols], sg_cols=sg_cols, precision=2)
        st.dataframe(sty, use_container_width=True, hide_index=True)

    st.divider()

