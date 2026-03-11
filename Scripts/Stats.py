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
from elite_finish_tab import render_elite_finish_tab
from sg_production_tab import render_production_sg_tab
from course_history_proto import render_course_history_demo
from approach_skill_tab import render_approach_skill_tab
from h2h_visual_tab import render_h2h_visual_tab
from approach_skill_tab import load_approach_skill
from weather_tab import render_weather_tab

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
COURSE_FIT_PATH = INUSE_DIR / "course_fit_weights_predictive_2017_2026_distacc_relative.csv"
APPROACH_SKILL_PATH = INUSE_DIR / "approach_skill_all_periods.csv"
ALL_PLAYERS_PATH = INUSE_DIR / "All_players.xlsx"
SCHED_PATH       = INUSE_DIR / "OAD_2026_Schedule.xlsx"
SKILL_PATH       = INUSE_DIR / "app_skill.xlsx"
FIELDS_PATH      = INUSE_DIR / "Fields.xlsx"
FINISHES_PATH = INUSE_DIR / "Finishes.csv"
BUCKET_PATH_A = INUSE_DIR / "Approach_Buckets.xlsx"
BUCKET_PATH_B = INUSE_DIR / "Approach Buckets.xlsx"
BUCKET_PATH   = BUCKET_PATH_A if BUCKET_PATH_A.exists() else BUCKET_PATH_B
ROUNDS_PATH = INUSE_DIR / "combined_rounds_all_2017_2026.csv"
schedule_df = pd.read_excel(SCHED_PATH)

required = [
    ALL_PLAYERS_PATH, SCHED_PATH, SKILL_PATH, FIELDS_PATH, BUCKET_PATH,
    ROUNDS_PATH,
    FINISHES_PATH,
]

missing = [p for p in required if not p.exists()]

if missing:
    st.error("Missing required file(s):")
    for p in missing:
        st.code(str(p))
    st.stop()


@st.cache_data(show_spinner=False)
def load_rounds_all():
    df = pd.read_csv(ROUNDS_PATH, low_memory=False)

    for c in ['dg_id', 'event_id', 'year', 'finish_num']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    date_col = 'round_date' if 'round_date' in df.columns else 'event_completed'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    return df


# Call it
rounds_df = load_rounds_all()


# Load fields (optional, for predictions tab)
@st.cache_data(show_spinner=False)
def load_fields():
    if not FIELDS_PATH.exists():
        return None
    return pd.read_excel(FIELDS_PATH)


fields_df = load_fields()

SEASON_YEAR = 2026
APP_VERSION = "1.0.1"
st.title(f"TourStats - v{APP_VERSION}")

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

def build_last_n_events_table(
    rounds_df: pd.DataFrame,
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

    r = rounds_df.copy()
    if not r.empty and "dg_id" in r.columns:
        r["dg_id"] = pd.to_numeric(r["dg_id"], errors="coerce")
        r = r.loc[r["dg_id"] == int(dg_id)].copy()
        if not r.empty:
            ends_r = _event_end_table_roundlevel(rounds_df)
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

    if not parts:
        return pd.DataFrame(columns=["Event", "Finish", "SG Total", "Year", "event_id"])

    out = pd.concat(parts, ignore_index=True)
    if date_max is not None and pd.notna(date_max):
        out = out.loc[out["event_end"].notna() & (out["event_end"] <= pd.to_datetime(date_max))].copy()

    out = out.sort_values(["event_end", "year", "event_id"], ascending=[False, False, False]).head(int(n)).copy()

    out["SG Total"] = pd.to_numeric(out.get("SG_Total"), errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    out = out.rename(columns={"event_name": "Event", "year": "Year"})
    return out[["Event", "Finish", "SG Total", "Year", "event_id"]].reset_index(drop=True)


def _last_n_rounds_pre_event(rounds_df: pd.DataFrame, dg_id: int, cutoff_dt: Optional[pd.Timestamp], n: int = 40) -> pd.DataFrame:
    df = rounds_df.copy()
    df["dg_id"] = pd.to_numeric(df.get("dg_id"), errors="coerce")
    df = df.loc[df["dg_id"] == int(dg_id)].copy()
    if df.empty:
        return df

    ends = _event_end_table_roundlevel(rounds_df)
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
    rounds_df: pd.DataFrame,
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

    r = rounds_df.copy()

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
            ends_r = _event_end_table_roundlevel(rounds_df)
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


    combined = r_year.copy()

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



schedule = load_schedule()
fields = load_fields()
all_players = load_all_players()
ID_TO_IMG, NAME_TO_IMG = build_headshot_maps(all_players)

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
    rolling = compute_rolling_stats(rounds_df=rounds_df, as_of_date=cutoff, dg_ids=base_ids, windows=(40, 24, 12))
for w in (12, 24, 40):
    b = f"birdies_L{w}"
    e = f"eagles_or_better_L{w}"
    if b in rolling.columns and e in rolling.columns:
        rolling[f"birdies_or_better_L{w}"] = rolling[b].fillna(0) + rolling[e].fillna(0)

finishes = load_finishes()
ytd = compute_ytd_from_finishes(finishes, year=SEASON_YEAR)

out = base.merge(rolling, on="dg_id", how="left") \
          .merge(ytd, on="dg_id", how="left")
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
if "ytd_made_cut_pct" in out.columns:
    out["ytd_made_cut_pct"] = out["ytd_made_cut_pct"].round(3)



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

TAB_NAMES = [
    "Event Overview",
    "Field SG",
    "Course History",
    "Approach Skill",
    "H2H",
    "Player Deep Dive",
    "Event Archive",
    # "Contender Model",
]

if "active_tab" not in st.session_state or st.session_state.active_tab not in TAB_NAMES:
    st.session_state.active_tab = "Event Overview"


active_tab = st.segmented_control(
    "",
    TAB_NAMES,
    default=st.session_state.active_tab,
    key="active_tab",
)

if active_tab == "Field SG":
    render_production_sg_tab(
        rounds_df=rounds_df,
        field_ids=field_ids,
        all_players=all_players,
        id_to_img=ID_TO_IMG,
        name_to_img=NAME_TO_IMG,
        schedule_df=schedule_df,
        event_id=event_id,
        cutoff_dt=cutoff,
    )

elif active_tab == "Course History":
    render_course_history_demo(
        course_num=st.session_state.get("selected_course_num"),
        rounds_df=rounds_df,
        ev_2017_2023=None,
        all_players=all_players,
        field_ids=field_ids,
        cutoff_dt=get_pre_event_cutoff_date(sched, int(event_id)) if event_id is not None else None,
        season_year=SEASON_YEAR,
        build_course_history_func=build_course_history_field_table,
        id_to_img=ID_TO_IMG,
        name_to_img=NAME_TO_IMG
    )

elif active_tab == "Approach Skill":
    render_approach_skill_tab(
        approach_skill_path=INUSE_DIR / "approach_skill_all_periods.csv",
        approach_buckets_path=BUCKET_PATH,
        event_id=event_id,
        field_ids=field_ids,
        all_players=all_players,
    )

elif active_tab == "H2H":
    render_h2h_visual_tab(
        summary_top=summary_top,
        rounds_df=rounds_df,
        cutoff_dt=get_pre_event_cutoff_date(sched, int(event_id)) if event_id else None,
        all_players=all_players,
        ID_TO_IMG=ID_TO_IMG,
        NAME_TO_IMG=NAME_TO_IMG,
        render_player_hero=render_player_hero,
        build_last_n_events_table=build_last_n_events_table,
        _last_n_rounds_pre_event=_last_n_rounds_pre_event,
        course_fit_df=pd.read_csv(COURSE_FIT_PATH) if COURSE_FIT_PATH.exists() else None,
        course_num=st.session_state.get("selected_course_num"),
        approach_skill_df=load_approach_skill(INUSE_DIR / "approach_skill_all_periods.csv"),
        schedule_df=schedule_df,
        event_id=event_id,
        greens_ref_path=str(INUSE_DIR / "course_greens_reference.csv"),
    )

elif active_tab == "Player Deep Dive":
    from player_deep_dive_tab import render_player_deep_dive_tab
    render_player_deep_dive_tab(
        summary_top=summary_top,
        rounds_df=rounds_df,
        cutoff_dt=get_pre_event_cutoff_date(sched, int(event_id)) if event_id else None,
        all_players=all_players,
        ID_TO_IMG=ID_TO_IMG,
        NAME_TO_IMG=NAME_TO_IMG,
        render_player_hero=render_player_hero,
        build_last_n_events_table=build_last_n_events_table,
        _last_n_rounds_pre_event=_last_n_rounds_pre_event,
        _event_end_table_roundlevel=_event_end_table_roundlevel,
        get_headshot_url=get_headshot_url,
        show_headshot_cropped_card=show_headshot_cropped_card,
        heat_table=heat_table,
        ytd=ytd,
        course_fit_df=pd.read_csv(COURSE_FIT_PATH) if COURSE_FIT_PATH.exists() else None,
        course_num=st.session_state.get("selected_course_num"),
        approach_skill_df=load_approach_skill(APPROACH_SKILL_PATH),
        field_ids=field_ids,
        season_year=SEASON_YEAR,
        schedule_df=schedule_df,
        event_id=event_id,
        greens_ref_path=str(INUSE_DIR / "course_greens_reference.csv"),
    )

elif active_tab == "Event Archive":
    from event_browser_tab import render_event_browser_tab
    render_event_browser_tab(
        rounds_df=rounds_df,
        ID_TO_IMG=ID_TO_IMG,
        NAME_TO_IMG=NAME_TO_IMG,
    )

# elif active_tab == "Contender Model":
#     render_elite_finish_tab(rounds_df=rounds_df, fields_df=fields_df, event_id=event_id)

elif active_tab == "Event Overview":
    from overview_tab import render_overview_tab
    render_overview_tab(
        selected_row=selected_row,
        rounds_df=rounds_df,
        field_ev=field_ev,
        event_id=event_id,
        course_num=course_num,
        cutoff_dt=cutoff,
        course_fit_df=pd.read_csv(COURSE_FIT_PATH) if COURSE_FIT_PATH.exists() else None,
        id_to_img=ID_TO_IMG,
        weather_api_key=st.secrets.get("WEATHER_API_KEY", ""),
        schedule_df=schedule_df,
        tee_times_path=str(INUSE_DIR / "this_week_field.csv"),
    )