from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional, Dict, List
import streamlit.components.v1 as components
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import html

st.set_page_config(
    page_title="Stats",
    layout="wide",
    initial_sidebar_state="expanded",
)

THIS_FILE = Path(__file__).resolve()

def _pick_repo_root(start: Path) -> Path:
    candidates = [p for p in [start.parent] + list(start.parents) if (p / "Data").exists()]
    if not candidates:
        raise RuntimeError("Could not find repo root containing Data/")
    return candidates[-1]

REPO_ROOT = _pick_repo_root(THIS_FILE)
DATA_ROOT = REPO_ROOT / "Data"
INUSE_DIR = DATA_ROOT / "in Use"

ALL_PLAYERS_PATH = INUSE_DIR / "All_players.xlsx"
SCHED_PATH       = INUSE_DIR / "OAD_2026_Schedule.xlsx"
SKILL_PATH       = INUSE_DIR / "app_skill.xlsx"
FIELDS_PATH      = INUSE_DIR / "Fields.xlsx"
FINISHES_PATH = INUSE_DIR / "Finishes.csv"
BUCKET_PATH_A = INUSE_DIR / "Approach_Buckets.xlsx"
BUCKET_PATH_B = INUSE_DIR / "Approach Buckets.xlsx"
BUCKET_PATH   = BUCKET_PATH_A if BUCKET_PATH_A.exists() else BUCKET_PATH_B

EVENTLEVEL_2017_2023 = INUSE_DIR / "combined_eventlevel_pga_2017_2023_mean.csv"

ROUNDLEVEL_2024P_CANDIDATES = [
    INUSE_DIR / "combined_roundlevel_2024_present.csv",
    INUSE_DIR / "combined_roundlevel_2024_present_mean.csv",
    INUSE_DIR / "combined_roundlevel_2024_present_mean_only.csv",
    INUSE_DIR / "combined_roundlevel_2024_present_pga.csv",
]
ROUNDLEVEL_2024_PRESENT = next((p for p in ROUNDLEVEL_2024P_CANDIDATES if p.exists()), None)

ROUNDS_PATH = INUSE_DIR / "combined_rounds_all_2017_2026.csv"

required = [
    ALL_PLAYERS_PATH, SCHED_PATH, SKILL_PATH, FIELDS_PATH, BUCKET_PATH,
    EVENTLEVEL_2017_2023,
    ROUNDS_PATH,
    FINISHES_PATH,
]

missing = [p for p in required if not p.exists()]

if ROUNDLEVEL_2024_PRESENT is None:
    missing.append(INUSE_DIR / "combined_roundlevel_2024_present*.csv (not found)")
elif not ROUNDLEVEL_2024_PRESENT.exists():
    missing.append(ROUNDLEVEL_2024_PRESENT)

if missing:
    st.error("Missing required file(s):")
    for p in missing:
        st.code(str(p))
    st.stop()

@st.cache_data(show_spinner=False)
def load_roundlevel_2024_present() -> pd.DataFrame:
    if not ROUNDLEVEL_2024_PRESENT.exists():
        raise FileNotFoundError(f"Missing: {ROUNDLEVEL_2024_PRESENT}")
    df = pd.read_csv(ROUNDLEVEL_2024_PRESENT, low_memory=False)

    for c in ["dg_id", "event_id", "year", "round_num", "course_num", "finish_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_eventlevel_2017_2023() -> pd.DataFrame:
    if not EVENTLEVEL_2017_2023.exists():
        raise FileNotFoundError(f"Missing: {EVENTLEVEL_2017_2023}")
    df = pd.read_csv(EVENTLEVEL_2017_2023, low_memory=False)

    for c in ["dg_id", "event_id", "year", "course_num", "finish_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


SEASON_YEAR = 2026

st.title("One and Done")

HERO_H = 240  # <-- tune: 230-280

st.markdown("""
<style>
/* Deep Dive layout helpers */
.dd-wrap{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 16px 18px;
}

.dd-title{
  font-size: 28px;
  font-weight: 900;
  margin: 0 0 10px 0;
}
.dd-kpi { flex: 1 1 0; min-width: 0; min-height: 52px; }

.dd-kpi-row{
  display: flex;
  gap: 14px;
  margin-top: 10px;
}

.dd-kpi{
  flex: 1;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  border-radius: 16px;
  padding: 12px 12px;
  min-height: 64px;
}

.dd-kpi-label{
  font-size: 12px;
  opacity: 0.75;
  margin-bottom: 6px;
}

.dd-kpi-val{
  font-size: 24px;
  font-weight: 900;
  line-height: 1;
}

.dd-img-card{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 18px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 330px; /* keeps it stable */
}

.dd-img-card img{
  border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.dd-photo {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.hero-card{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  border-radius: 14px;
  padding: 14px 16px;
}

.hero-title{
  font-size: 26px;
  font-weight: 800;
  line-height: 1.1;
  margin: 0;
}

.hero-meta-row{
  margin-top: 6px;
  font-size: 14px;
  color: rgba(255,255,255,0.75);
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.hero-meta{ white-space: nowrap; }
.hero-meta-sep{ opacity: 0.55; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* center it and remove any "card" vibe */
.dd-float-wrap{
  width: 100%;
  display:flex;
  justify-content:center;
  align-items:flex-start;
  padding-top: 8px;
}

/* floating image box: NO border, NO background, just shadow */
.dd-float-imgbox{
  border-radius: 22px;
  overflow: hidden;
  background: transparent;
  border: none;
  box-shadow:
    0 18px 45px rgba(0,0,0,0.55),
    0 2px 10px rgba(0,0,0,0.35);
}

/* actual image crop */
.dd-float-imgbox img{
  width: 100%;
  height: 100%;
  display:block;
  object-fit: cover;
  object-position: center 20%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Deep Dive hero row */
.dd-hero-left {
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
}

.dd-hero-title {
  font-size: 28px;
  font-weight: 900;
  margin: 0 0 10px 0;
}

.dd-kpi-row {
  display: flex;
  gap: 12px;
  margin-top: 12px;
  flex-wrap: nowrap;
}

.dd-kpi {
  flex: 1 1 0;
  min-width: 0;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.035);
  border-radius: 14px;
  padding: 10px 12px;
}

.dd-kpi-label {
  font-size: 11px;
  opacity: 0.70;
  margin-bottom: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dd-kpi-val {
  font-size: 20px;
  font-weight: 850;
  line-height: 1.0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Right headshot card */
.dd-photo-card {
  width: 100%;
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.02);
  box-shadow:
    0 18px 45px rgba(0,0,0,0.45),
    0 2px 10px rgba(0,0,0,0.25);
}

.dd-photo-card{
  width: 100%;
  height: 360px;                 /* taller */
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;

  display: flex;
  align-items: center;
  justify-content: center;
}

/* Image: bigger, still no-crop */
.dd-photo-card img{
  height: 100%;
  width: auto;                   /* don’t stretch to full width */
  max-width: 100%;               /* don’t overflow */
  object-fit: contain;
  object-position: center;

  border-radius: 22px;           /* rounding on image itself */
  display: block;
}
</style>
""", unsafe_allow_html=True)

def load_finishes_2026() -> pd.DataFrame:
    p = INUSE_DIR / "Finishes.csv"   # this file is 2026-only per your notebook
    df = pd.read_csv(p)

    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df["event_id"] = pd.to_numeric(df.get("event_id"), errors="coerce").astype("Int64")
    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce").astype("Int64")

    for c in ["event_completed", "win", "top_5", "top_10", "top_25", "made_cut", "CUT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df

def show_headshot_cropped_card(img_value: str | None, height_px: int = 250) -> None:
    if not img_value:
        st.empty()
        return

    s = str(img_value).strip().strip('"').strip("'").strip()
    if not s or s.lower() in {"none", "nan"}:
        st.empty()
        return

    if s.startswith("http://") or s.startswith("https://"):
        safe_src = html.escape(s, quote=True)
        st.markdown(
            f"""
            <div class="dd-photo-card" style="height:{height_px}px;">
              <img src="{safe_src}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    p = Path(s)
    p2 = REPO_ROOT / s
    if p.exists():
        st.image(str(p), use_container_width=True)
    elif p2.exists():
        st.image(str(p2), use_container_width=True)
    else:
        st.image(s, use_container_width=True)

def show_headshot_cropped_floating(img_value: str | None, height_px: int = 320, max_width_px: int = 420) -> None:
    if not img_value:
        return

    s = str(img_value).strip().strip('"').strip("'").strip()
    if not s or s.lower() in {"none", "nan"}:
        return

    if s.startswith("http://") or s.startswith("https://"):
        safe_src = html.escape(s, quote=True)
        st.markdown(
            f"""
            <div class="dd-float-wrap">
              <div class="dd-float-imgbox" style="height:{height_px}px; width:min(100%, {max_width_px}px);">
                <img src="{safe_src}" />
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    p = Path(s)
    p2 = REPO_ROOT / s
    if p.exists():
        st.image(str(p), use_container_width=True)
    elif p2.exists():
        st.image(str(p2), use_container_width=True)
    else:
        st.image(s, use_container_width=True)


    p = Path(s)
    p2 = REPO_ROOT / s

    if p.exists():
        st.image(str(p), use_container_width=True)
        return
    if p2.exists():
        st.image(str(p2), use_container_width=True)
        return

    st.image(s, use_container_width=True)


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

    candidates = ["event_end", "end_date", "tournament_end", "event_date", "start_date", "date"]
    dcol = next((c for c in candidates if c in df.columns), None)
    if dcol is None:
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


def build_last_n_events_table(
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

    e = ev_2017_2023.copy()
    if not e.empty and "dg_id" in e.columns:
        e["dg_id"] = pd.to_numeric(e["dg_id"], errors="coerce")
        e = e.loc[e["dg_id"] == int(dg_id)].copy()
        if not e.empty:
            ends_e = _event_end_table_eventlevel(ev_2017_2023)

            name_col = "event_name" if "event_name" in e.columns else ("tournament" if "tournament" in e.columns else None)
            if name_col is None:
                e["event_name"] = e.get("event_id", "").astype(str)
                name_col = "event_name"

            fin_col = "fin_text" if "fin_text" in e.columns else ("finish_text" if "finish_text" in e.columns else None)
            if fin_col is None:
                e["fin_text"] = ""
                fin_col = "fin_text"

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

    df.columns = [c.strip().lower() for c in df.columns]

    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

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

    p2 = REPO_ROOT / s
    if p2.exists():
        st.image(str(p2), width=width)
        return

    st.image(s, width=width)


@st.cache_data(show_spinner=False)
def load_schedule() -> pd.DataFrame:
    df = pd.read_excel(SCHED_PATH)
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

    for c in ["year", "event_id", "dg_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "event_completed" in df.columns:
        ec = df["event_completed"]
        if pd.api.types.is_numeric_dtype(ec):
            df["event_completed"] = pd.to_datetime(ec, unit="D", origin="1899-12-30", errors="coerce")
        else:
            df["event_completed"] = pd.to_datetime(ec, errors="coerce")

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

    flag_cols = ["Win", "Top_5", "Top_10", "Top_25", "Made_Cut", "CUT"]
    for c in flag_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")  # blank -> NaN
            s = s.where(s.isna(), s.clip(0, 1))
            df[c] = s.astype("Int64")  # nullable int keeps <NA>

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
    with open(ROUNDS_PATH, "r") as f:
        header = f.readline().strip().split(",")
    cols = [c for c in usecols if c in header]

    df = pd.read_csv(ROUNDS_PATH, usecols=cols)

    if "tour" in df.columns:
        df["tour"] = df["tour"].astype(str).str.lower().str.strip()

    if "dg_id" in df.columns:
        df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")

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

def _player_meta_from_all_players(all_players: pd.DataFrame, dg_id: int) -> dict:
    """
    Returns a dict of player meta fields from all_players for a given dg_id.
    Safe if columns are missing or player not found.
    """
    meta = {"owgr": None, "country": None}

    if all_players is None or not isinstance(all_players, pd.DataFrame) or all_players.empty:
        return meta

    ap = all_players.copy()
    if "dg_id" not in ap.columns:
        return meta

    ap["dg_id"] = pd.to_numeric(ap["dg_id"], errors="coerce")
    row = ap.loc[ap["dg_id"] == int(dg_id)]
    if row.empty:
        return meta

    r0 = row.iloc[0]

    owgr_col = None
    for c in ["owgr", "OWGR", "owgr_rank", "owgr_current"]:
        if c in ap.columns:
            owgr_col = c
            break
    if owgr_col:
        v = pd.to_numeric(r0.get(owgr_col), errors="coerce")
        meta["owgr"] = int(v) if pd.notna(v) else None

    for c in ["country", "country_code", "ctry", "nation"]:
        if c in ap.columns:
            val = r0.get(c)
            meta["country"] = str(val) if pd.notna(val) else None
            break

    return meta

def render_player_hero(
    *,
    dg_id: int,
    player_name: str,
    all_players: pd.DataFrame,
    ID_TO_IMG: dict,
    NAME_TO_IMG: dict,
    odds: float | None = None,
    headshot_width: int = 120,
    image_only: bool = False,   # NEW
):

    meta = _player_meta_from_all_players(all_players, dg_id)
    owgr = meta.get("owgr", None)

    if odds is not None:
        try:
            odds = float(odds)
            if not np.isfinite(odds):
                odds = None
        except Exception:
            odds = None

    url = get_headshot_url(dg_id, player_name, ID_TO_IMG, NAME_TO_IMG)
    if image_only:
        show_headshot(url, width=headshot_width)
        return


    c_img, c_txt = st.columns([2, 8], vertical_alignment="center")
    with c_img:
        show_headshot(url, width=headshot_width)

    with c_txt:
        parts = []
        if owgr is not None:
            parts.append(f'<span class="hero-meta">OWGR: {owgr}</span>')
        if odds is not None:
            if parts:
                parts.append('<span class="hero-meta-sep">•</span>')
            parts.append(f'<span class="hero-meta">Odds: {odds:.1f}</span>')

        meta_html = f'<div class="hero-meta-row">{" ".join(parts)}</div>' if parts else ""

        st.markdown(
            f"""
            <div class="hero-card">
              <div class="hero-title">{player_name}</div>
              {meta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    if "tour" in df.columns:
        df = df[df["tour"] == "pga"].copy()

    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df[df["dg_id"].isin([int(x) for x in dg_ids])].copy()

    if "round_date" in df.columns and df["round_date"].notna().any():
        date_col = "round_date"
    elif "event_completed" in df.columns:
        date_col = "event_completed"
    else:
        raise ValueError("Rounds data must include round_date or event_completed.")

    df = df[df[date_col] < ts].copy()
    if df.empty:
        return pd.DataFrame({"dg_id": list(dg_ids)})

    df = df.sort_values(["dg_id", date_col], ascending=[True, False])

    rolled = (
        df.groupby("dg_id", group_keys=False)
          .apply(_player_rolling_for_windows, windows=windows)
          .reset_index()
    )
    return rolled

@st.cache_data(show_spinner=False)
def load_finishes() -> pd.DataFrame:
    df = pd.read_csv(FINISHES_PATH)

    for c in ["dg_id", "event_id", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str)

    flag_cols = ["win", "top_5", "top_10", "top_25", "made_cut", "CUT"]
    for c in flag_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

    return df


def compute_ytd_from_finishes(finishes: pd.DataFrame, year: int = 2026) -> pd.DataFrame:
    """
    YTD from Finishes.csv:
      - only rows in `year`
      - only rows with event_completed present (completed events)
      - starts = unique event_id per dg_id
      - wins/top10/top25/made_cut = sums of 0/1 flags (de-duped per event)
    """
    if finishes is None or finishes.empty:
        return pd.DataFrame(columns=[
            "dg_id", "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts", "ytd_made_cut_pct"
        ])

    df = finishes.copy()

    if "year" not in df.columns:
        raise ValueError("Finishes.csv must include a 'year' column.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"] == int(year)].copy()

    if "event_completed" in df.columns:
        df["event_completed"] = pd.to_datetime(df["event_completed"], errors="coerce")
        df = df[df["event_completed"].notna()].copy()

    if "dg_id" not in df.columns:
        raise ValueError("Finishes.csv must include 'dg_id'.")

    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df.dropna(subset=["dg_id"]).copy()
    df["dg_id"] = df["dg_id"].astype(int)

    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
        df = df.dropna(subset=["event_id"]).copy()
        df["event_id"] = df["event_id"].astype(int)
        df = df.drop_duplicates(subset=["dg_id", "event_id"], keep="last")

        starts = df.groupby("dg_id")["event_id"].nunique()
    else:
        starts = df.groupby("dg_id").size()

    def _sum_flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0, index=starts.index)
        s = df.groupby("dg_id")[col].sum()
        return s.reindex(starts.index, fill_value=0)

    wins  = _sum_flag("win")
    top10 = _sum_flag("top_10")
    top25 = _sum_flag("top_25")
    made  = _sum_flag("made_cut")

    out = pd.DataFrame({
        "dg_id": starts.index.astype(int),
        "ytd_starts": starts.values,
        "ytd_wins": wins.values,
        "ytd_top10": top10.values,
        "ytd_top25": top25.values,
        "ytd_made_cuts": made.values,
    })

    out["ytd_made_cut_pct"] = np.where(
        out["ytd_starts"] > 0,
        out["ytd_made_cuts"] / out["ytd_starts"],
        np.nan
    )

    for c in ["ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    return out

def _finish_sort_key(fin: str) -> float:
    """
    Convert finish text to a sortable numeric:
      1, T3, T16 -> 1, 3, 16
      CUT -> 999
      DNP/WD/DQ -> 9999
      blank -> 1e9
    Lower is better.
    """
    if fin is None:
        return 1e9
    s = str(fin).strip().upper()
    if s == "" or s in {"NONE", "NAN"}:
        return 1e9
    if s in {"CUT", "MC"}:
        return 999.0
    if s in {"DNP", "WD", "DQ"}:
        return 9999.0
    s = s.replace("T", "")
    try:
        return float(pd.to_numeric(s, errors="coerce"))
    except Exception:
        return 1e9


def _best_finish_text(series: pd.Series) -> str:
    vals = [str(x) for x in series.dropna().astype(str).tolist() if str(x).strip() != ""]
    if not vals:
        return ""
    vals_sorted = sorted(vals, key=_finish_sort_key)
    return vals_sorted[0]

@st.cache_data(show_spinner=False)
def compute_ytd_sg_total(
    rounds_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    dg_ids: Iterable[int],
    year: int,
) -> pd.DataFrame:
    """
    YTD SG Total = mean sg_total across PGA rounds in `year` with round_date < as_of_date.
    Returns: dg_id, ytd_sg_total
    """
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = rounds_df.copy()

    if "tour" in df.columns:
        df = df[df["tour"] == "pga"].copy()

    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce")
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df["sg_total"] = pd.to_numeric(df.get("sg_total"), errors="coerce")

    if "round_date" not in df.columns:
        return pd.DataFrame({"dg_id": list(dg_ids), "ytd_sg_total": np.nan})

    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

    df = df.dropna(subset=["dg_id", "year", "round_date", "sg_total"]).copy()
    df = df[df["dg_id"].isin([int(x) for x in dg_ids])].copy()
    df = df[df["year"] == int(year)].copy()
    df = df[df["round_date"] < ts].copy()

    if df.empty:
        return pd.DataFrame({"dg_id": list(dg_ids), "ytd_sg_total": np.nan})

    out = (
        df.groupby("dg_id", as_index=False)["sg_total"]
          .mean()
          .rename(columns={"sg_total": "ytd_sg_total"})
    )
    return out

def build_course_history_field_table(
    *,
    course_num: int,
    base_ids: list[int],
    rounds_2024p: pd.DataFrame,
    ev_2017_2023: pd.DataFrame,
    cutoff_dt: Optional[pd.Timestamp],
    season_year: int,
    years_back: int = 9,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if course_num is None:
        return (pd.DataFrame(), pd.DataFrame())

    course_num = int(course_num)
    base_ids = [int(x) for x in base_ids] if base_ids else []

    end_year = int(season_year) - 1
    target_years = set(range(end_year - years_back + 1, end_year + 1))

    def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _ensure_finish_text(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        fin_col = _pick_first_col(df, ["fin_text", "finish_text"])
        if fin_col is not None:
            df[fin_col] = df[fin_col].astype(str).fillna("")
            return df, fin_col

        if "finish_num" in df.columns:
            df["finish_num"] = pd.to_numeric(df["finish_num"], errors="coerce")
            df["fin_text"] = df["finish_num"].map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            df["fin_text"] = ""
        return df, "fin_text"

    cutoff_ts = pd.to_datetime(cutoff_dt) if cutoff_dt is not None and pd.notna(cutoff_dt) else None

    r = rounds_2024p.copy()

    for c in ["dg_id", "year", "event_id", "course_num", "course", "course_id", "round_num", "sg_total", "finish_num"]:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce")
    if "round_date" in r.columns:
        r["round_date"] = pd.to_datetime(r["round_date"], errors="coerce")

    course_col_r = _pick_first_col(r, ["course_num", "course", "course_id"])
    if course_col_r is None:
        r = r.iloc[0:0].copy()
    else:
        r = r.loc[r[course_col_r] == course_num].copy()

    if "dg_id" in r.columns and base_ids:
        r = r.loc[r["dg_id"].isin(base_ids)].copy()
    elif "dg_id" not in r.columns:
        r = r.iloc[0:0].copy()

    if not r.empty and cutoff_ts is not None:
        used_cutoff = False

        if "year" in r.columns and "event_id" in r.columns:
            ends_r = _event_end_table_roundlevel(rounds_2024p)
            rr = r.dropna(subset=["year", "event_id"]).copy()
            if not rr.empty:
                rr["year"] = rr["year"].astype(int)
                rr["event_id"] = rr["event_id"].astype(int)
                rr = rr.merge(ends_r, on=["year", "event_id"], how="left")
                rr["event_end"] = pd.to_datetime(rr["event_end"], errors="coerce")

                keep = rr["event_end"].notna() & (rr["event_end"] <= cutoff_ts)
                if keep.any():
                    r = rr.loc[keep].copy()
                    used_cutoff = True

        if not used_cutoff and "round_date" in r.columns:
            r = r.loc[r["round_date"].notna() & (r["round_date"] <= cutoff_ts)].copy()

    r, fin_col_r = _ensure_finish_text(r)

    r_year = pd.DataFrame()
    if not r.empty and "dg_id" in r.columns and "year" in r.columns:
        r_year = (
            r.groupby(["dg_id", "year"], as_index=False)
             .agg(
                 best_finish=(fin_col_r, _best_finish_text),
                 rounds=("round_num", "count") if "round_num" in r.columns else ("event_id", "size"),
                 sg_mean=("sg_total", "mean") if "sg_total" in r.columns else ("event_id", "size"),
             )
        )

    e = ev_2017_2023.copy()

    for c in ["dg_id", "year", "event_id", "course_num", "course", "course_id", "sg_total", "finish_num"]:
        if c in e.columns:
            e[c] = pd.to_numeric(e[c], errors="coerce")

    course_col_e = _pick_first_col(e, ["course_num", "course", "course_id"])
    if course_col_e is None:
        e = e.iloc[0:0].copy()
    else:
        e = e.loc[e[course_col_e] == course_num].copy()

    if "dg_id" in e.columns and base_ids:
        e = e.loc[e["dg_id"].isin(base_ids)].copy()
    elif "dg_id" not in e.columns:
        e = e.iloc[0:0].copy()

    if not e.empty and cutoff_ts is not None and "year" in e.columns and "event_id" in e.columns:
        ends_e = _event_end_table_eventlevel(ev_2017_2023)
        ee = e.dropna(subset=["year", "event_id"]).copy()
        if not ee.empty:
            ee["year"] = ee["year"].astype(int)
            ee["event_id"] = ee["event_id"].astype(int)
            ee = ee.merge(ends_e, on=["year", "event_id"], how="left")
            ee["event_end"] = pd.to_datetime(ee["event_end"], errors="coerce")

            has_end = ee["event_end"].notna()
            if has_end.any():
                ee = ee.loc[~has_end | (ee["event_end"] <= cutoff_ts)].copy()
            e = ee.copy()

    e, fin_col_e = _ensure_finish_text(e)

    e_year = pd.DataFrame()
    if not e.empty and "dg_id" in e.columns and "year" in e.columns:
        e_year = (
            e.groupby(["dg_id", "year"], as_index=False)
             .agg(
                 best_finish=(fin_col_e, _best_finish_text),
                 starts=("event_id", "nunique") if "event_id" in e.columns else (fin_col_e, "size"),
                 sg_mean=("sg_total", "mean") if "sg_total" in e.columns else (fin_col_e, "size"),
             )
        )
        e_year["rounds"] = (pd.to_numeric(e_year["starts"], errors="coerce").fillna(0).astype(int) * 4)

    combined = pd.concat(
        [r_year.assign(src="roundlevel"), e_year.assign(src="eventlevel")],
        ignore_index=True
    )

    if combined.empty:
        return (pd.DataFrame(), pd.DataFrame())

    combined["src_rank"] = combined["src"].map({"roundlevel": 0, "eventlevel": 1}).fillna(9)
    combined = (
        combined.sort_values(["dg_id", "year", "src_rank"])
                .drop_duplicates(["dg_id", "year"], keep="first")
                .copy()
    )

    years_present = sorted([y for y in combined["year"].dropna().astype(int).unique() if y in target_years])

    pivot = (
        combined[combined["year"].isin(years_present)]
        .pivot(index="dg_id", columns="year", values="best_finish")
        .reset_index()
    )

    agg_all = (
        combined.groupby("dg_id", as_index=False)
        .agg(
            rounds=("rounds", "sum"),
            true_sg=("sg_mean", "mean"),
        )
    )

    name_map = all_players[["dg_id", "player_name"]].drop_duplicates().copy()
    name_map["dg_id"] = pd.to_numeric(name_map["dg_id"], errors="coerce").astype("Int64")

    out_wide = (
        pivot.merge(agg_all, on="dg_id", how="left")
             .merge(name_map, on="dg_id", how="left")
    )

    out_wide["rounds"] = pd.to_numeric(out_wide["rounds"], errors="coerce").fillna(0).astype(int)
    out_wide["true_sg"] = pd.to_numeric(out_wide["true_sg"], errors="coerce")

    year_cols = [y for y in years_present if y in out_wide.columns]
    display_cols = ["dg_id", "player_name"] + year_cols + ["rounds", "true_sg"]
    out_wide = out_wide[display_cols].copy()

    out_wide = out_wide.rename(columns={
        "player_name": "PLAYER",
        "rounds": "ROUNDS",
        "true_sg": "SG",
    })

    out_wide = out_wide.sort_values("SG", ascending=False, na_position="last").reset_index(drop=True)

    horses = out_wide.copy()
    horses = horses.loc[horses["ROUNDS"] >= 8].head(12).copy()

    return out_wide, horses



def style_course_history_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Colors:
      - TRUE SG: green-high
      - year finish cells: gold for 1, green for top 10, RED for CUT/MC, GREY for DNP/WD/DQ
      - blank/None -> show "DNP"
    """
    if df is None or df.empty:
        return df.style

    show = df.copy()

    finish_cols = [c for c in show.columns if c not in {"PLAYER", "ROUNDS", "SG"}]

    if finish_cols:
        for c in finish_cols:
            s = show[c]

            s = s.astype("string")

            s_norm = s.str.strip().str.upper()
            missing_mask = s_norm.isna() | s_norm.isin(["", "NONE", "NAN", "<NA>"])

            show[c] = s.mask(missing_mask, "DNP")

    def _cell_style(v: str) -> str:
        s = str(v).strip().upper()

        if s == "DNP":
            return "background-color: rgba(255,255,255,0.03); color: rgba(255,255,255,0.45);"

        if s in {"CUT", "MC"}:
            return "background-color: rgba(220, 38, 38, 0.35); color: rgba(255,255,255,0.95); font-weight: 750;"

        if s in {"WD", "DQ"}:
            return "background-color: rgba(255,255,255,0.03); color: rgba(255,255,255,0.45);"

        key = _finish_sort_key(s)
        if key == 1:
            return "background-color: rgba(250, 204, 21, 0.50); font-weight: 800;"  # gold
        if key <= 3:
            return "background-color: rgba(34, 197, 94, 0.35); font-weight: 700;"
        if key <= 10:
            return "background-color: rgba(34, 197, 94, 0.22);"
        if key <= 25:
            return "background-color: rgba(34, 197, 94, 0.12);"
        return ""

    sty = show.style

    if "SG" in show.columns:
        sty = sty.background_gradient(subset=["SG"], cmap="RdYlGn")
        sty = sty.format({"SG": (lambda x: "" if pd.isna(x) else f"{float(x):+.2f}")})

    if finish_cols:
        sty = sty.applymap(_cell_style, subset=finish_cols)

    return sty




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

    tour_bucket_cols = ["50_100", "100_150", "150_200", "over_200"]
    weights = {c: float(pd.to_numeric(row.get(c), errors="coerce") or 0.0) for c in tour_bucket_cols}
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}

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

    count_cols = ["50_100_fw_shot_count", "100_150_fw_shot_count", "150_200_fw_shot_count", "over_200_fw_shot_count"]
    present = [c for c in count_cols if c in s_df.columns]
    if present:
        tmp = s_df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["approach_samples"] = tmp.sum(axis=1).astype(int)

    return base.merge(out, on="dg_id", how="left")


schedule = load_schedule()
fields = load_fields()
all_players = load_all_players()
ID_TO_IMG, NAME_TO_IMG = build_headshot_maps(all_players)
skills = load_player_skill()
buckets = load_approach_buckets()
rounds = load_rounds_minimal()

with st.sidebar:

    sched = schedule.copy()

    date_col = "start_date" if "start_date" in sched.columns else ("event_date" if "event_date" in sched.columns else None)
    if date_col:
        sched[date_col] = pd.to_datetime(sched[date_col], errors="coerce")
        sched = sched.sort_values(date_col, na_position="last")
    else:
        sched = sched.reset_index(drop=True)

    sched = sched.reset_index(drop=True)
    sched["__row_id"] = sched.index.astype(int)

    sched["__event_label"] = sched.get("event_name", "").astype(str)
    sched.loc[sched["__event_label"].str.strip().isin(["", "nan", "None"]), "__event_label"] = "Unknown event"

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



sched_row = selected_row

if "start_date" in sched_row.index and pd.notna(sched_row.get("start_date")):
    cutoff = pd.to_datetime(sched_row["start_date"]) - pd.Timedelta(days=1)
elif "event_date" in sched_row.index and pd.notna(sched_row.get("event_date")):
    cutoff = pd.to_datetime(sched_row["event_date"]) - pd.Timedelta(days=1)
else:
    cutoff = pd.Timestamp.today()

if event_id is None:
    st.info("This schedule row has no event_id yet, so field/odds/YTD-by-event can’t be shown for it.")



field_ev = pd.DataFrame()
field_ids: list[int] = []

if event_id is not None:
    f = fields.copy()
    f["event_id"] = pd.to_numeric(f["event_id"], errors="coerce")
    f["year"] = pd.to_numeric(f["year"], errors="coerce")
    f["dg_id"] = pd.to_numeric(f["dg_id"], errors="coerce")

    field_ev = f[
        (f["event_id"] == int(event_id)) &
        (f["year"] == SEASON_YEAR)
    ].dropna(subset=["dg_id"])

    field_ev = field_ev.drop_duplicates(subset=["dg_id"], keep="last")
    field_ids = field_ev["dg_id"].astype(int).tolist()

if only_in_field and not field_ids:
    st.warning(
        "No 2026 field uploaded yet for this event. Showing scouting universe instead."
    )
    only_in_field = False

f_univ = fields.copy()

for c in ["year", "event_id", "dg_id"]:
    if c in f_univ.columns:
        f_univ[c] = pd.to_numeric(f_univ[c], errors="coerce")

if "player_name" in f_univ.columns:
    f_univ["player_name"] = f_univ["player_name"].astype(str)

f_univ = f_univ.dropna(subset=["dg_id", "year"]).copy()
f_univ["dg_id"] = f_univ["dg_id"].astype(int)
f_univ["year"] = f_univ["year"].astype(int)

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

name_pool = f_univ[f_univ["dg_id"].isin(universe_ids)].copy()
name_pool["is_2026"] = (name_pool["year"] == 2026).astype(int)
name_pool = name_pool.sort_values(["dg_id", "is_2026", "year"], ascending=[True, False, False])

universe = name_pool.drop_duplicates(subset=["dg_id"], keep="first")[["dg_id", "player_name"]].copy()

if only_in_field:
    base_ids = field_ids
else:
    base_ids = universe["dg_id"].astype(int).unique().tolist()

base = universe[universe["dg_id"].isin(base_ids)].copy()


with st.spinner("Computing rolling stats (L40/L24/L12) from round-level data..."):
    rolling = compute_rolling_stats(rounds_df=rounds, as_of_date=cutoff, dg_ids=base_ids, windows=(40, 24, 12))
for w in (12, 24, 40):
    b = f"birdies_L{w}"
    e = f"eagles_or_better_L{w}"
    if b in rolling.columns and e in rolling.columns:
        rolling[f"birdies_or_better_L{w}"] = rolling[b].fillna(0) + rolling[e].fillna(0)

finishes = load_finishes()
ytd = compute_ytd_from_finishes(finishes, year=SEASON_YEAR)
if event_id is not None:
    fit = compute_approach_fit(base_ids, skills, buckets, event_id)
else:
    fit = pd.DataFrame({"dg_id": base_ids, "approach_fit_score": np.nan})


out = base.merge(rolling, on="dg_id", how="left") \
          .merge(ytd, on="dg_id", how="left") \
          .merge(fit, on="dg_id", how="left")
out["dg_id"] = pd.to_numeric(out["dg_id"], errors="coerce")
out = out.dropna(subset=["dg_id"]).copy()
out["dg_id"] = out["dg_id"].astype(int)
out = out.drop_duplicates(subset=["dg_id"], keep="first")
num_cols = out.select_dtypes(include=[np.number]).columns
out[num_cols] = out[num_cols].round(1)

odds_candidates = ["close_odds", "decimal_odds", "odds", "win_prob_est"]
odds_col = next((c for c in odds_candidates if c in field_ev.columns), None)
if odds_col:
    tmp = field_ev[["dg_id", odds_col]].copy()
    tmp["dg_id"] = pd.to_numeric(tmp["dg_id"], errors="coerce")
    out = out.merge(tmp, on="dg_id", how="left")

for c in ["ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cuts"]:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
if "ytd_made_cut_pct" in out.columns:
    out["ytd_made_cut_pct"] = pd.to_numeric(out["ytd_made_cut_pct"], errors="coerce")


for c in out.columns:
    if any(c.startswith(x) for x in ["sg_", "round_score"]):
        out[c] = pd.to_numeric(out[c], errors="coerce")
for c in [col for col in out.columns if col.startswith("sg_")]:
    out[c] = out[c].round(3)
for c in [col for col in out.columns if col.startswith("round_score_")]:
    out[c] = out[c].round(2)
if "approach_fit_score" in out.columns:
    out["approach_fit_score"] = pd.to_numeric(out["approach_fit_score"], errors="coerce").round(3)
if "ytd_made_cut_pct" in out.columns:
    out["ytd_made_cut_pct"] = out["ytd_made_cut_pct"].round(3)



WINDOW_LABELS = {
    "L12": "L12",
    "L24": "L24",
    "L40": "L40",
}

STAT_LABELS = {
    "sg_total": "Total",
    "sg_app": "App",
    "sg_putt": "Putt",
    "sg_ott": "OTT",
    "sg_arg": "ARG",
    "sg_t2g": "T2G",
    "round_score": "Score",
    "birdies_or_better": "Birdies+",
    "birdies": "Birdies",
    "eagles_or_better": "Eagles+",
}

def pretty_col(c: str, odds_col: str | None = None) -> str:
    if c == "player_name":
        return "Player"
    if odds_col and c == odds_col:
        return "Odds"

    if "_L" in c:
        base, w = c.rsplit("_L", 1)
        w = f"L{w}"
        base_label = STAT_LABELS.get(base, base.replace("_", " ").title())
        w_label = WINDOW_LABELS.get(w, w)
        return f"{base_label} — {w_label}"

    if c.startswith("ytd_"):
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

    rename_map = {c: pretty_col(c, odds_col=odds_col) for c in show.columns}
    show = show.rename(columns=rename_map)

    grad_cols = [rename_map.get(c, c) for c in gradient_cols if c in rename_map]

    low_is_good = [c for c in grad_cols if c.startswith("Round Score")]
    high_is_good = [c for c in grad_cols if c not in low_is_good]

    sty = show.style
    if high_is_good:
        sty = sty.background_gradient(subset=high_is_good, cmap="RdYlGn")
    if low_is_good:
        sty = sty.background_gradient(subset=low_is_good, cmap="RdYlGn_r")

    fmt = {c: "{:.1f}" for c in show.columns if c != "Player" and pd.api.types.is_numeric_dtype(show[c])}
    sty = sty.format(fmt)

    st.dataframe(sty, use_container_width=True, height=height)

def heat_table(df: pd.DataFrame, sg_cols, precision=2):
    show = df.copy()

    for c in sg_cols:
        if c in show.columns:
            show[c] = (
                show[c].astype(str)
                      .str.replace("+", "", regex=False)
                      .replace({"": None, "nan": None, "None": None})
            )
            show[c] = pd.to_numeric(show[c], errors="coerce")

    sty = show.style

    sty = sty.background_gradient(subset=[c for c in sg_cols if c in show.columns], cmap="RdYlGn", axis=None)

    fmt = {c: (lambda x: "" if pd.isna(x) else f"{x:+.{precision}f}") for c in sg_cols if c in show.columns}
    sty = sty.format(fmt)

    return sty

summary_top = out.copy()

_default_sort = "sg_total_L12" if "sg_total_L12" in summary_top.columns else None
if _default_sort:
    summary_top = summary_top.sort_values(_default_sort, ascending=False)

summary_top = summary_top.reset_index(drop=True)

TAB_NAMES = ["Strokes Gained", "Course History", "Approach Buckets", "H2H", "Deep Dive"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Strokes Gained"


active_tab = st.segmented_control(
    "",
    TAB_NAMES,
    default=st.session_state.active_tab,
    key="active_tab",
)

if active_tab == "Strokes Gained":
    st.subheader("Players (rolling L12/L24/L40)")

    SORT_LABELS = {
        "sg_total_L12": "SG Total - L12",
        "sg_total_L24": "SG Total - L24",
        "sg_total_L40": "SG Total - L40",
        "sg_app_L12": "SG Approach - L12",
        "sg_putt_L12": "SG Putting - L12",
    }

    sort_choices = [c for c in [
        "sg_total_L12", "sg_total_L24", "sg_total_L40",
        "sg_app_L12", "sg_putt_L12",
    ] if c in out.columns]

    sort_by = st.selectbox(
        "Sort by",
        options=sort_choices,
        index=0 if sort_choices else 0,
        format_func=lambda c: SORT_LABELS.get(c, c),
    )

    out_show = out.copy()
    if sort_by and sort_by in out_show.columns:
        out_show = out_show.sort_values(sort_by, ascending=False)

    name_col = "player_name"
    odds = odds_col if odds_col else None

    if st.session_state.get("only_in_field", False) and field_ids and "dg_id" in out_show.columns:
        out_show["dg_id"] = pd.to_numeric(out_show["dg_id"], errors="coerce")
        out_show = out_show[out_show["dg_id"].isin([int(x) for x in field_ids])]

    excluded_players = st.session_state.get("excluded_players", [])
    if excluded_players:
        out_show = out_show[~out_show[name_col].isin(excluded_players)]

    if "dg_id" in out_show.columns and not out_show.empty:
        tmp = out_show.copy()
        tmp["dg_id"] = pd.to_numeric(tmp["dg_id"], errors="coerce")
        tmp = tmp.dropna(subset=["dg_id"]).copy()
        tmp["dg_id"] = tmp["dg_id"].astype(int)
        if not tmp.empty:
            top1 = int(tmp.iloc[0]["dg_id"])
            top2 = int(tmp.iloc[1]["dg_id"]) if len(tmp) > 1 else top1
            st.session_state["weekly_top1_dg_id"] = top1
            st.session_state["weekly_top2_dg_id"] = top2

    total_cols = [name_col] + ([odds] if odds else []) + ["sg_total_L12", "sg_total_L24", "sg_total_L40"]
    total_grad = ["sg_total_L12", "sg_total_L24", "sg_total_L40"]
    render_table("Odds + SG Total", out_show, total_cols, total_grad, odds_col=odds_col, height=420)

    app_cols = [name_col, "sg_app_L12", "sg_app_L24", "sg_app_L40"]
    app_grad = ["sg_app_L12", "sg_app_L24", "sg_app_L40"]
    render_table("Approach", out_show, app_cols, app_grad, odds_col=None, height=420)

    putt_cols = [name_col, "sg_putt_L12", "sg_putt_L24", "sg_putt_L40"]
    putt_grad = ["sg_putt_L12", "sg_putt_L24", "sg_putt_L40"]
    render_table("Putting", out_show, putt_cols, putt_grad, odds_col=None, height=420)

    ott_cols = [name_col, "sg_ott_L12", "sg_ott_L24", "sg_ott_L40"]
    ott_grad = ["sg_ott_L12", "sg_ott_L24", "sg_ott_L40"]
    render_table("Off the Tee", out_show, ott_cols, ott_grad, odds_col=None, height=420)

    arg_cols = [name_col, "sg_arg_L12", "sg_arg_L24", "sg_arg_L40"]
    arg_grad = ["sg_arg_L12", "sg_arg_L24", "sg_arg_L40"]
    render_table("Around the Green", out_show, arg_cols, arg_grad, odds_col=None, height=420)

    t2g_cols = [name_col, "sg_t2g_L12", "sg_t2g_L24", "sg_t2g_L40"]
    t2g_grad = ["sg_t2g_L12", "sg_t2g_L24", "sg_t2g_L40"]
    render_table("Tee to Green", out_show, t2g_cols, t2g_grad, odds_col=None, height=420)

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

elif active_tab == "Course History":
    st.header("Course History")

    course_num = st.session_state.get("selected_course_num", None)
    if course_num is None:
        st.info("No course_num found for this schedule row.")
        st.stop()

    rounds_2024p = load_roundlevel_2024_present()
    ev_2017_2023 = load_eventlevel_2017_2023()

    cutoff_dt = get_pre_event_cutoff_date(sched, int(event_id)) if event_id is not None else None

    effective_base_ids: list[int] = []
    try:
        if base_ids:
            effective_base_ids = [int(x) for x in base_ids if pd.notna(x)]
    except Exception:
        effective_base_ids = []

    if not effective_base_ids:
        if all_players is None or all_players.empty or "dg_id" not in all_players.columns:
            st.error("all_players is missing or does not include a 'dg_id' column — cannot build Course History.")
            st.stop()

        effective_base_ids = (
            pd.to_numeric(all_players["dg_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    course_hist, horses = build_course_history_field_table(
        course_num=int(course_num),
        base_ids=effective_base_ids,
        rounds_2024p=rounds_2024p,
        ev_2017_2023=ev_2017_2023,
        cutoff_dt=cutoff_dt,
        season_year=SEASON_YEAR,
        years_back=9,
    )

    if horses is not None and not horses.empty:
        horses_show = horses.copy()

        if "SG" in horses_show.columns:
            horses_show["SG"] = pd.to_numeric(horses_show["SG"], errors="coerce")
        else:
            horses_show["SG"] = np.nan

        if "ROUNDS" in horses_show.columns:
            horses_show["ROUNDS"] = pd.to_numeric(horses_show["ROUNDS"], errors="coerce").fillna(0)
        else:
            horses_show["ROUNDS"] = 0

        top5 = (
            horses_show.sort_values(["SG", "ROUNDS"], ascending=[False, False], na_position="last")
            .head(5)
            .copy()
        )

        cols = st.columns(5, gap="large")
        for i, (_, r) in enumerate(top5.iterrows()):
            with cols[i]:
                dg_id = None
                if "dg_id" in top5.columns and pd.notna(r.get("dg_id")):
                    try:
                        dg_id = int(r.get("dg_id"))
                    except Exception:
                        dg_id = None

                name = str(r.get("PLAYER", "")) if pd.notna(r.get("PLAYER", "")) else ""

                img = None
                if dg_id is not None and "ID_TO_IMG" in globals():
                    img = ID_TO_IMG.get(dg_id)
                if img is None and "NAME_TO_IMG" in globals():
                    img = NAME_TO_IMG.get(name)

                if img is not None:
                    st.image(img, use_container_width=True)

                sg_val = r.get("SG")
                sg_txt = ""
                if pd.notna(sg_val):
                    try:
                        sg_txt = f"{float(sg_val):+.1f}"
                    except Exception:
                        sg_txt = ""
                st.caption(f"{name} {sg_txt}".strip())

    st.subheader("Course Horses")

    if horses is None or horses.empty:
        st.info("No reliable course-horse sample yet for this course.")
    else:
        horses_show = horses.copy()
        cols_h = ["PLAYER"] + [c for c in horses_show.columns if str(c).isdigit()] + ["ROUNDS", "SG"]
        cols_h = [c for c in cols_h if c in horses_show.columns]

        st.dataframe(
            style_course_history_table(horses_show[cols_h]),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    st.divider()

    as_of_date = None
    if "cutoff" in globals() and cutoff is not None:
        as_of_date = cutoff
    else:
        as_of_date = cutoff_dt

    if as_of_date is None:
        st.error("No as_of_date available (cutoff/cutoff_dt). Cannot compute YTD stats for Course History.")
        st.stop()

    ytd_sg = compute_ytd_sg_total(
        rounds_df=rounds,
        as_of_date=as_of_date,
        dg_ids=effective_base_ids,
        year=SEASON_YEAR,
    )

    if ytd_sg is None or ytd_sg.empty or "dg_id" not in ytd_sg.columns:
        ytd_sg = pd.DataFrame({"dg_id": pd.Series(dtype="Int64"), "ytd_sg_total": pd.Series(dtype="float")})

    ytd_sg = ytd_sg.copy()
    ytd_sg["dg_id"] = pd.to_numeric(ytd_sg["dg_id"], errors="coerce").astype("Int64")
    if "ytd_sg_total" in ytd_sg.columns:
        ytd_sg["ytd_sg_total"] = pd.to_numeric(ytd_sg["ytd_sg_total"], errors="coerce")
    else:
        ytd_sg["ytd_sg_total"] = np.nan

    ap = all_players.copy() if all_players is not None else pd.DataFrame()
    if not ap.empty and ("dg_id" in ap.columns) and ("player_name" in ap.columns):
        ap["dg_id"] = pd.to_numeric(ap["dg_id"], errors="coerce").astype("Int64")
        ap["player_name"] = ap["player_name"].astype(str)
        name_to_id_all = dict(zip(ap["player_name"], ap["dg_id"]))
    else:
        name_to_id_all = {}

    st.subheader("Finish History (by COURSE)")

    if course_hist is None or course_hist.empty:
        st.info("No course history found for the current player set at this course.")
    else:
        ch = course_hist.copy()

        if "dg_id" in ch.columns:
            ch["dg_id"] = pd.to_numeric(ch["dg_id"], errors="coerce").astype("Int64")
        else:
            if "PLAYER" not in ch.columns:
                st.error("course_hist is missing expected 'PLAYER' column.")
                st.stop()
            ch["dg_id"] = ch["PLAYER"].astype(str).map(name_to_id_all).astype("Int64")

        ch = ch.merge(ytd_sg[["dg_id", "ytd_sg_total"]], on="dg_id", how="left")

        if "SG" in ch.columns:
            ch["SG"] = pd.to_numeric(ch["SG"], errors="coerce")
        else:
            ch["SG"] = np.nan

        if "ROUNDS" in ch.columns:
            ch["ROUNDS"] = pd.to_numeric(ch["ROUNDS"], errors="coerce")
        else:
            ch["ROUNDS"] = np.nan

        ch["ytd_sg_total"] = pd.to_numeric(ch.get("ytd_sg_total"), errors="coerce")

        sort_cols = [c for c in ["ytd_sg_total", "SG", "ROUNDS", "PLAYER"] if c in ch.columns]
        asc = [False, False, False, True][: len(sort_cols)]
        ch = ch.sort_values(by=sort_cols, ascending=asc, na_position="last").reset_index(drop=True)

        cols = ["PLAYER"] + [c for c in ch.columns if str(c).isdigit()] + ["ROUNDS", "SG"]
        cols = [c for c in cols if c in ch.columns]

        st.dataframe(
            style_course_history_table(ch[cols]),
            use_container_width=True,
            hide_index=True,
            height=800,
        )

    st.divider()

    if "out" not in globals() or out is None or out.empty:
        st.info("YTD table source ('out') is empty or missing.")
    else:
        ytd_show = out.copy()

        if st.session_state.get("only_in_field", False):
            if "field_ids" in globals() and field_ids and "dg_id" in ytd_show.columns:
                ytd_show["dg_id"] = pd.to_numeric(ytd_show["dg_id"], errors="coerce")
                ytd_show = ytd_show[ytd_show["dg_id"].isin([int(x) for x in field_ids if pd.notna(x)])]

        excluded_players = st.session_state.get("excluded_players", [])
        if excluded_players and "player_name" in ytd_show.columns:
            ytd_show = ytd_show[~ytd_show["player_name"].isin(excluded_players)]

        if "dg_id" in ytd_show.columns:
            ytd_show["dg_id"] = pd.to_numeric(ytd_show["dg_id"], errors="coerce").astype("Int64")
        else:
            ytd_show["dg_id"] = pd.Series([pd.NA] * len(ytd_show), dtype="Int64")

        ytd_show = ytd_show.merge(ytd_sg[["dg_id", "ytd_sg_total"]], on="dg_id", how="left")
        ytd_show["ytd_sg_total"] = pd.to_numeric(ytd_show.get("ytd_sg_total"), errors="coerce")
        ytd_show = ytd_show.sort_values("ytd_sg_total", ascending=False, na_position="last")

        ytd_cols = ["player_name", "ytd_sg_total", "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cut_pct"]
        ytd_cols = [c for c in ytd_cols if c in ytd_show.columns]

        ytd_grad = [c for c in ["ytd_sg_total", "ytd_starts", "ytd_wins", "ytd_top10", "ytd_top25", "ytd_made_cut_pct"]
                    if c in ytd_show.columns]

        render_table("Year to Date", ytd_show, ytd_cols, ytd_grad, odds_col=None, height=520)


elif active_tab == "Approach Buckets":
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
    mix_items = [
        ("50–100", mix.get("50–100", np.nan)),
        ("100–150", mix.get("100–150", np.nan)),
        ("150–200", mix.get("150–200", np.nan)),
        ("Over 200", mix.get("Over 200", np.nan)),
    ]

    vals = [(lab, v) for lab, v in mix_items if pd.notna(v)]
    vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)

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


    col_map = {
        "50 – 100 SG": "50_100_fw_value",
        "100 – 150 SG": "100_150_fw_value",
        "150 – 200 SG": "150_200_fw_value",
        "Over 200 SG": "over_200_fw_value",
    }

    s = skills.copy()
    s["dg_id"] = pd.to_numeric(s.get("dg_id"), errors="coerce")
    s = s[s["dg_id"].isin([int(x) for x in base_ids])].copy()

    if "player_name" not in s.columns and "Player" in s.columns:
        s["player_name"] = s["Player"]

    keep_cols = ["player_name"] + [v for v in col_map.values() if v in s.columns]
    s = s[keep_cols].copy()

    rename_map = {"player_name": "Player"}
    rename_map.update({v: k for k, v in col_map.items() if v in s.columns})
    s = s.rename(columns=rename_map)

    val_cols = [c for c in s.columns if c.endswith("SG")]
    for c in val_cols:
        s[c] = pd.to_numeric(s[c], errors="coerce")

    SG_SCALE = 100.0
    if val_cols:
        s[val_cols] = s[val_cols] * SG_SCALE

    st.caption(f"Note: SG columns are scaled to SG per 100 shots (original values ×{int(SG_SCALE)}).")
    st.caption(
        "None indicates shots recorded in this distance bucket (often DP World Tour players or rookies).")

    default_sort = "100 – 150 SG" if "100 – 150 SG" in val_cols else (val_cols[0] if val_cols else None)

    sort_choice = st.selectbox(
        "Sort by",
        options=val_cols,
        index=(val_cols.index(default_sort) if default_sort else 0),
        key="tab2_sort_by",
    )

    if sort_choice and sort_choice in s.columns:
        s = s.sort_values(sort_choice, ascending=False)

    sty = s.style
    if val_cols:
        sty = sty.background_gradient(subset=val_cols, cmap="RdYlGn")

    sty = sty.format({c: "{:.1f}" for c in val_cols})

    st.dataframe(sty, use_container_width=True, height=800)


elif active_tab == "H2H":
    st.header("Head-to-Head Comparison")

    if event_id is None:
        st.info("No event_id for this schedule row yet.")
        st.stop()

    rounds_2024p = load_roundlevel_2024_present()
    ev_2017_2023 = load_eventlevel_2017_2023()

    cutoff_dt = get_pre_event_cutoff_date(sched, int(event_id))

    pool = summary_top[["dg_id", "player_name", "close_odds"]].dropna(
        subset=["dg_id", "player_name"]).drop_duplicates().copy()
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    pool["close_odds"] = pool.get("close_odds")  # keep column even if missing
    odds_by_id = dict(zip(pool["dg_id"], pool["close_odds"]))

    name_to_id = dict(zip(pool["player_name"], pool["dg_id"]))
    id_to_name = dict(zip(pool["dg_id"], pool["player_name"]))

    player_options = sorted(pool["player_name"].unique().tolist())

    weekly_top1 = st.session_state.get("weekly_top1_dg_id", None)
    weekly_top2 = st.session_state.get("weekly_top2_dg_id", None)

    top1_name = id_to_name.get(int(weekly_top1)) if weekly_top1 is not None else None
    top2_name = id_to_name.get(int(weekly_top2)) if weekly_top2 is not None else None

    if not top1_name or top1_name not in player_options:
        top1_name = player_options[0] if player_options else None
    if not top2_name or top2_name not in player_options:
        top2_name = player_options[1] if len(player_options) > 1 else top1_name

    if top1_name == top2_name and len(player_options) > 1:
        top2_name = next((n for n in player_options if n != top1_name), top2_name)

    if ("h2h_a" not in st.session_state) or (st.session_state.get("h2h_a") not in player_options):
        st.session_state["h2h_a"] = top1_name

    if ("h2h_b" not in st.session_state) or (st.session_state.get("h2h_b") not in player_options):
        st.session_state["h2h_b"] = top2_name

    if st.session_state.get("h2h_a") == st.session_state.get("h2h_b") and len(player_options) > 1:
        st.session_state["h2h_b"] = next((n for n in player_options if n != st.session_state["h2h_a"]),
                                         st.session_state["h2h_b"])



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
        name_a = st.selectbox(
            " ",
            player_options,
            key="h2h_a",
            label_visibility="collapsed",
        )

    with selB:
        st.markdown("### Player B")
        name_b = st.selectbox(
            " ",
            player_options,
            key="h2h_b",
            label_visibility="collapsed",
        )

    dg_a = int(name_to_id[name_a])
    dg_b = int(name_to_id[name_b])

    if dg_a == dg_b:
        st.warning("Pick two different players.")
        st.stop()

    odds_a = odds_by_id.get(dg_a, None)
    odds_b = odds_by_id.get(dg_b, None)

    kpis_a = [("Odds", str(odds_a))] if odds_a is not None and str(odds_a).strip() != "" else []
    kpis_b = [("Odds", str(odds_b))] if odds_b is not None and str(odds_b).strip() != "" else []

    heroA, heroB=st.columns(2, gap="large")

    with heroA:
        render_player_hero(
            dg_id=dg_a,
            player_name=name_a,
            all_players=all_players,
            ID_TO_IMG=ID_TO_IMG,
            NAME_TO_IMG=NAME_TO_IMG,
            odds=odds_a,
            headshot_width=110,
        )

    with heroB:
        render_player_hero(
            dg_id=dg_b,
            player_name=name_b,
            all_players=all_players,
            ID_TO_IMG=ID_TO_IMG,
            NAME_TO_IMG=NAME_TO_IMG,
            odds=odds_b,
            headshot_width=110,
        )

    st.subheader("Recent tournaments")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"#### {name_a}")
        tA = build_last_n_events_table(rounds_2024p, ev_2017_2023, dg_a, n=25, date_max=cutoff_dt)
        st.dataframe(
            tA[["Event", "Finish", "SG Total", "Year"]] if not tA.empty else tA,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown(f"#### {name_b}")
        tB = build_last_n_events_table(rounds_2024p, ev_2017_2023, dg_b, n=25, date_max=cutoff_dt)
        st.dataframe(
            tB[["Event", "Finish", "SG Total", "Year"]] if not tB.empty else tB,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

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

            smooth_window = st.slider("Smoothing (moving average window)", 1, 15, 5, key="smooth_w")

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

            fig.update_traces(line=dict(width=3))
            fig.update_traces(selector=dict(name=name_a), line=dict(color="orange"))
            fig.update_traces(selector=dict(name=name_b), line=dict(color="deepskyblue"))

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

    st.subheader("Performance by season")

    def _perf_by_season(dg_id: int) -> pd.DataFrame:
        rows = []

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

                starts = e.groupby("year")["event_id"].nunique().rename("starts")

                wins = pd.Series(0, index=starts.index)
                if "finish_num" in e.columns:
                    fn = pd.to_numeric(e["finish_num"], errors="coerce")
                    w = e.loc[fn == 1].groupby("year")["event_id"].nunique()
                    wins.loc[w.index] = w

                sg_col = "sg_total" if "sg_total" in e.columns else None
                avg_sg = e.groupby("year")[sg_col].mean() if sg_col else pd.Series(np.nan, index=starts.index)

                tmp = pd.DataFrame({
                    "year": starts.index.astype(int),
                    "starts": starts.values,
                    "wins": wins.values,
                    "avg_sg_per_event": avg_sg.reindex(starts.index).values,
                })
                rows.append(tmp)

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
        st.dataframe(_perf_by_season(dg_a), use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"#### {name_b}")
        st.dataframe(_perf_by_season(dg_b), use_container_width=True, hide_index=True)

    st.divider()

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

    for col in [name_a, name_b]:
        comp[col] = pd.to_numeric(comp[col], errors="coerce").round(1)


    def _highlight_better_value_cell(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        mask_a = df["winner"] == "◀"
        mask_b = df["winner"] == "▶"

        styles.loc[mask_a, name_a] = "background-color: rgba(255,255,255,0.18); font-weight: 700;"
        styles.loc[mask_b, name_b] = "background-color: rgba(255,255,255,0.18); font-weight: 700;"

        styles.loc[:, "winner"] = "text-align: center;"
        return styles


    sty = (
        comp.style
        .apply(_highlight_better_value_cell, axis=None)
        .format({name_a: "{:.1f}", name_b: "{:.1f}"})
    )

    lower_better = {"round_score", "bogies", "doubles_or_worse", "poor_shots", "prox_rgh", "prox_fw"}

    a_col = name_a
    b_col = name_b


    def _highlight_winner_row(row):
        stat = row["stat"]
        va = row[a_col]
        vb = row[b_col]

        styles = {c: "" for c in row.index}

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

elif active_tab == "Deep Dive":
    st.header("Player Deep Dive")

    if event_id is None:
        st.info("No event_id for this schedule row yet.")
        st.stop()

    rounds_2024p = load_roundlevel_2024_present()
    ev_2017_2023 = load_eventlevel_2017_2023()
    cutoff_dt = get_pre_event_cutoff_date(sched, int(event_id))

    cols_needed = ["dg_id", "player_name"]
    if "close_odds" in summary_top.columns:
        cols_needed.append("close_odds")

    pool = (
        summary_top[cols_needed]
        .dropna(subset=["dg_id", "player_name"])
        .drop_duplicates(subset=["dg_id"])
        .copy()
    )
    pool["dg_id"] = pd.to_numeric(pool["dg_id"], errors="coerce").astype("Int64")
    pool = pool.dropna(subset=["dg_id"]).copy()
    pool["dg_id"] = pool["dg_id"].astype(int)
    pool["player_name"] = pool["player_name"].astype(str)

    name_by_id = dict(zip(pool["dg_id"], pool["player_name"]))
    odds_by_id = dict(zip(pool["dg_id"], pool.get("close_odds", pd.Series([None] * len(pool)))))

    dg_options = pool["dg_id"].tolist()

    weekly_top1 = st.session_state.get("weekly_top1_dg_id", None)
    default_dg = int(weekly_top1) if weekly_top1 in dg_options else (dg_options[0] if dg_options else None)
    if default_dg is None:
        st.info("No players available.")
        st.stop()

    if ("dd_dg_id" not in st.session_state) or (st.session_state["dd_dg_id"] not in dg_options):
        st.session_state["dd_dg_id"] = default_dg

    DD_HERO_H = 300 # one knob


    def _fmt_int(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        try:
            return f"{int(x)}"
        except Exception:
            return str(x)


    def _fmt_pct(x, digits=0):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        try:
            return f"{100 * float(x):.{digits}f}%"
        except Exception:
            return "—"


    def _fmt_odds(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        try:
            return f"{float(x):.1f}"
        except Exception:
            return str(x)


    def _kpi_strip_html(kpis):
        cards = "".join(
            f'<div class="dd-kpi"><div class="dd-kpi-label">{label}</div><div class="dd-kpi-val">{val}</div></div>'
            for label, val in kpis
        )
        return f'<div class="dd-kpi-row">{cards}</div>'


    left, right = st.columns([8,4], gap="small")

    with left:
        dg_id_sel = st.selectbox(
            "Player",
            options=dg_options,
            index=dg_options.index(st.session_state["dd_dg_id"]),
            format_func=lambda dg: f"{name_by_id.get(int(dg), 'Unknown')}",
            key="dd_dg_id",
        )

        player_name_sel = name_by_id.get(int(dg_id_sel), "Unknown")
        odds_sel = odds_by_id.get(int(dg_id_sel), None)

        meta = _player_meta_from_all_players(all_players, int(dg_id_sel))
        owgr = meta.get("owgr", None)

        yrow = ytd.loc[ytd["dg_id"] == int(dg_id_sel)]
        yrow = yrow.iloc[0] if not yrow.empty else None

        ytd_starts = yrow["ytd_starts"] if yrow is not None else None
        ytd_cutpct = yrow["ytd_made_cut_pct"] if yrow is not None else None
        ytd_top10 = yrow["ytd_top10"] if yrow is not None else None

        kpis = [
            ("OWGR", _fmt_int(owgr)),
            ("Odds", _fmt_odds(odds_sel)),
            ("Starts", _fmt_int(ytd_starts)),
            ("Cut %", _fmt_pct(ytd_cutpct, digits=0)),
            ("Top 10", _fmt_int(ytd_top10)),
        ]
        st.markdown(_kpi_strip_html(kpis), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        img_url = get_headshot_url(int(dg_id_sel), player_name_sel, ID_TO_IMG, NAME_TO_IMG)
        show_headshot_cropped_card(img_url, height_px=DD_HERO_H)

    st.subheader("Course History")

    course_num = st.session_state.get("selected_course_num", None)

    if course_num is None:
        st.info("Course history unavailable (no course_num for the selected event).")
    else:
        out_parts = []

        r = rounds_2024p.copy()
        r["dg_id"] = pd.to_numeric(r.get("dg_id"), errors="coerce")
        r = r.loc[r["dg_id"] == int(dg_id_sel)].copy()

        if "course_num" not in r.columns:
            st.info("Course history unavailable for 2024+ (roundlevel has no course_num column).")
        else:
            r["course_num"] = pd.to_numeric(r["course_num"], errors="coerce")
            r = r.loc[r["course_num"] == int(course_num)].copy()

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

        if not out_parts:
            st.info("No course history found for this player at this course.")
        else:
            if not out_parts:
                st.info("No course history found for this player at this course.")
            else:
                course_hist = pd.concat(out_parts, ignore_index=True)
                course_hist["event_end"] = pd.to_datetime(course_hist.get("event_end"), errors="coerce")

                COURSE_SORT_LABELS = {
                    "SG Total": "SG Total",
                    "SG APP": "SG APP",
                    "SG OTT": "SG OTT",
                    "SG ARG": "SG ARG",
                    "SG PUTT": "SG PUTT",
                    "Year": "Year",
                }

                sort_choices = [c for c in ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT", "Year"] if
                                c in course_hist.columns]

                default_idx = sort_choices.index("Year") if "Year" in sort_choices else 0

                sort_by = st.selectbox(
                    "Sort by",
                    options=sort_choices,
                    index=default_idx,
                    format_func=lambda c: COURSE_SORT_LABELS.get(c, c),
                    key="dd_course_hist_sort_by",
                )

                ascending = (sort_by == "Year")  # year ascending usually reads better; flip if you want

                if sort_by in ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]:
                    course_hist[sort_by] = pd.to_numeric(course_hist[sort_by], errors="coerce")

                course_hist = course_hist.sort_values(
                    by=[sort_by, "event_end", "event_id"],
                    ascending=[ascending, False, False],
                    na_position="last",
                ).copy()

                ascending = False  # default

                if sort_by == "Year":
                    ascending = False

                course_hist = course_hist.sort_values(sort_by, ascending=ascending)

                for c in ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]:
                    if c in course_hist.columns:
                        course_hist[c] = pd.to_numeric(course_hist[c], errors="coerce").map(
                            lambda x: f"{x:+.2f}" if pd.notna(x) else ""
                        )

                show_cols = [c for c in ["Year", "Event", "Finish", "SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]
                             if c in course_hist.columns]
                sg_cols = ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]

                sty = heat_table(course_hist[show_cols], sg_cols=sg_cols, precision=2)
                st.dataframe(sty, use_container_width=True, hide_index=True)

    st.subheader("Last 60 Rounds")
    st.caption("Rounds without bars indicate non-PGA Tour events (SG breakdown not available).")

    metric_mode = st.selectbox(
        "Chart",
        ["Stacked SG components + SG Total line", "Single metric line (legacy)"],
        index=0,
        key="dd_chart_mode",
    )

    df = _last_n_rounds_pre_event(rounds_2024p, dg_id_sel, cutoff_dt, n=60)
    if df.empty:
        st.info("Not enough rounds pre-event for this player.")
    else:
        df = df.copy()
        df["round_index"] = range(1, len(df) + 1)

        for c in ["sg_total", "sg_app", "sg_ott", "sg_arg", "sg_putt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if metric_mode == "Stacked SG components + SG Total line":
            if "sg_total" not in df.columns:
                st.info("sg_total not available to chart.")
            else:
                base = df.copy()
                base["round_index"] = range(1, len(base) + 1)

                line_df = base.copy()
                line_df["sg_total"] = pd.to_numeric(line_df["sg_total"], errors="coerce")
                line_df = line_df.dropna(subset=["sg_total"]).copy()

                comps = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
                have_all_comps = all(c in base.columns for c in comps)

                long = None
                bar_df = pd.DataFrame()

                if have_all_comps:
                    bar_df = base.copy()
                    for c in comps:
                        bar_df[c] = pd.to_numeric(bar_df[c], errors="coerce")

                    bar_df = bar_df.dropna(subset=comps).copy()

                    if not bar_df.empty:
                        bar_df["sg_total_calc"] = bar_df[comps].sum(axis=1)

                        abs_sum = (
                                bar_df["sg_ott"].abs()
                                + bar_df["sg_app"].abs()
                                + bar_df["sg_arg"].abs()
                                + bar_df["sg_putt"].abs()
                        )
                        bar_df = bar_df.loc[abs_sum > 0].copy()

                fig = go.Figure()

                comps = [("SG OTT", "sg_ott"), ("SG APP", "sg_app"), ("SG ARG", "sg_arg"), ("SG PUTT", "sg_putt")]

                bar_df = base.copy()
                for _, c in comps:
                    bar_df[c] = pd.to_numeric(bar_df[c], errors="coerce")
                bar_df["sg_total"] = pd.to_numeric(bar_df["sg_total"], errors="coerce")

                need = [c for _, c in comps] + ["sg_total"]
                bar_df = bar_df.dropna(subset=need).copy()

                abs_sum = np.zeros(len(bar_df), dtype=float)
                for _, c in comps:
                    abs_sum += np.abs(bar_df[c].to_numpy(dtype=float))
                bar_df = bar_df.loc[abs_sum > 0].copy()
                abs_sum = abs_sum[abs_sum > 0]

                for label, c in comps:
                    w = np.abs(bar_df[c].to_numpy(dtype=float)) / abs_sum
                    bar_df[f"w_{c}"] = w
                    bar_df[f"seg_{c}"] = bar_df["sg_total"].to_numpy(dtype=float) * w

                x_all = list(range(1, len(base) + 1))
                x_bar = bar_df["round_index"].astype(int).tolist()

                COLOR_MAP = {
                    "SG OTT": "rgba(120, 180, 255, 0.55)",
                    "SG APP": "rgba(255, 170, 190, 0.55)",
                    "SG ARG": "rgba(190, 150, 255, 0.55)",
                    "SG PUTT": "rgba(255, 190, 120, 0.55)",
                }

                for label, c in comps:
                    fig.add_trace(go.Bar(
                        x=x_bar,
                        y=bar_df[f"seg_{c}"],
                        name=label,
                        marker=dict(color=COLOR_MAP[label]),
                        customdata=(bar_df[f"w_{c}"] * 100.0),
                        hovertemplate="%{fullData.name}: %{customdata:.0f}%<br>Seg: %{y:.2f}<extra></extra>"
                    ))

                line_df = base.copy()
                line_df["sg_total"] = pd.to_numeric(line_df["sg_total"], errors="coerce")
                line_df = line_df.dropna(subset=["sg_total"]).copy()

                fig.add_trace(go.Scatter(
                    x=line_df["round_index"].astype(int),
                    y=line_df["sg_total"],
                    mode="lines+markers",
                    name="SG Total",
                ))

                fig.update_layout(
                    barmode="relative",
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend_title_text="",
                )
                fig.update_xaxes(
                    type="linear",
                    tickmode="auto",
                    nticks=12,
                )
                fig.update_yaxes(zeroline=True)

                st.plotly_chart(fig, use_container_width=True)





        else:
            metric_options = [c for c in ["sg_total", "sg_t2g", "sg_app", "sg_ott", "sg_arg", "sg_putt", "round_score"]
                              if c in df.columns]
            if not metric_options:
                st.info("No metrics found in roundlevel_2024_present to chart.")
            else:
                metric = st.selectbox("Metric", metric_options, index=0, key="dd_metric")
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

    st.subheader("Tournament breakdown")

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

        for c in ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")

        show_cols = [c for c in ["Year", "Event", "Finish", "SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"] if c in out.columns]
        sg_cols = ["SG Total", "SG APP", "SG OTT", "SG ARG", "SG PUTT"]
        sty = heat_table(out[show_cols], sg_cols=sg_cols, precision=2)
        st.dataframe(sty, use_container_width=True, hide_index=True)

