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
from documentation_tab import render_documentation_tab
from sg_production_tab import render_production_sg_tab
from course_history_proto import render_course_history_demo, SAME_VENUE_COURSE_NUMS
from approach_skill_tab import render_approach_skill_tab
from h2h_visual_tab import render_h2h_visual_tab
from approach_skill_tab import load_approach_skill
from weather_tab import render_weather_tab
from live_tab import render_live_tab, is_tournament_live
from field_adjusted_sg import compute_field_adjusted_sg

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
CLEAN_COMBINED_DIR = DATA_ROOT / "Clean" / "Combined"
schedule_df = pd.read_excel(SCHED_PATH)

required = [
    ALL_PLAYERS_PATH, SCHED_PATH, SKILL_PATH, FIELDS_PATH, BUCKET_PATH,
    FINISHES_PATH,
]

missing = [p for p in required if not p.exists()]
if not list(CLEAN_COMBINED_DIR.glob("combined_rounds_[0-9][0-9][0-9][0-9].csv")):
    missing.append(CLEAN_COMBINED_DIR / "combined_rounds_YYYY.csv")

if missing:
    st.error("Missing required file(s):")
    for p in missing:
        st.code(str(p))
    st.stop()


def _add_round_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute to_par, cum_to_par, round_position, round_position_text from raw round data."""
    for col in ["to_par", "cum_to_par", "round_position", "round_position_text"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.copy()
    df["to_par"] = df["round_score"] - df["course_par"]
    df = df.sort_values(["event_id", "season", "dg_id", "round_num"]).reset_index(drop=True)
    df["cum_to_par"] = df.groupby(["event_id", "season", "dg_id"])["to_par"].cumsum()

    missing_par = df.groupby(["event_id", "season"])["course_par"].apply(lambda x: x.isna().all())
    missing_keys = missing_par[missing_par].index
    if len(missing_keys) > 0:
        mask = df.set_index(["event_id", "season"]).index.isin(missing_keys)
        df.loc[mask, "cum_to_par"] = (
            df[mask].groupby(["event_id", "season", "dg_id"])["round_score"].cumsum().values
        )

    def rank_group(g):
        valid = g["cum_to_par"].notna()
        pos = pd.Series(np.nan, index=g.index)
        if valid.any():
            pos[valid] = g.loc[valid, "cum_to_par"].rank(method="min", ascending=True).astype(int)
        return pos

    df["round_position"] = (
        df.groupby(["event_id", "season", "round_num"], group_keys=False)
        .apply(lambda g: rank_group(g), include_groups=False)
    )
    df["round_position"] = pd.array(
        df["round_position"].where(df["round_position"].isna(), df["round_position"].astype("Int64")),
        dtype="Int64",
    )
    tie_counts = df.groupby(["event_id", "season", "round_num", "round_position"])["dg_id"].transform("count")
    df["round_position_text"] = df["round_position"].astype(str)
    df.loc[df["round_position"].isna(), "round_position_text"] = np.nan
    df.loc[tie_counts > 1, "round_position_text"] = "T" + df.loc[tie_counts > 1, "round_position"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_rounds_all():
    year_files = sorted(CLEAN_COMBINED_DIR.glob("combined_rounds_[0-9][0-9][0-9][0-9].csv"))
    dfs = [pd.read_csv(fp, low_memory=False) for fp in year_files]
    df = pd.concat(dfs, ignore_index=True)

    for c in ['dg_id', 'event_id', 'year', 'finish_num']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    date_col = 'round_date' if 'round_date' in df.columns else 'event_completed'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    df = _add_round_positions(df)

    return df


# Call it
rounds_df = load_rounds_all()


@st.cache_data(show_spinner=False)
def get_adjusted_rounds(rounds_df: pd.DataFrame, decay_lambda: float) -> pd.DataFrame:
    """Cached wrapper around compute_field_adjusted_sg.  Keyed on decay_lambda."""
    return compute_field_adjusted_sg(rounds_df, n_iter=8, decay_lambda=decay_lambda, min_rounds=15)


# Load fields (optional, for predictions tab)
@st.cache_data(show_spinner=False)
def load_fields():
    if not FIELDS_PATH.exists():
        return None
    return pd.read_excel(FIELDS_PATH)


fields_df = load_fields()

SEASON_YEAR = 2026
APP_VERSION = "1.0.3"
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

# ── Mobile / responsive CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@media (max-width: 768px) {
  /* Stack st.columns() layouts vertically */
  [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
  }
  [data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }

  /* KPI pill rows: wrap to 2 columns */
  .dd-kpi-row {
    flex-wrap: wrap !important;
  }
  .dd-kpi {
    flex: 1 1 calc(50% - 7px) !important;
    min-width: calc(50% - 7px) !important;
  }
}

@media (max-width: 480px) {
  /* KPI pills: full width on very small screens */
  .dd-kpi {
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
}

/* Course Horses: wrap to 2-per-row on tablet, 1 on phone */
@media (max-width: 768px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) {
    flex-direction: row !important;
    flex-wrap: wrap !important;
  }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) > [data-testid="stColumn"] {
    flex: 1 1 45% !important;
    min-width: 45% !important;
    max-width: 50% !important;
  }
}
@media (max-width: 480px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
  }
}

/* Compact segmented control — smaller font + tighter padding */
[data-testid="stSegmentedControl"] button {
  font-size: 9px !important;
  padding: 2px 5px !important;
  min-height: 0 !important;
  line-height: 1.2 !important;
}

/* Live leaderboard: horizontal scroll on mobile */
.live-leaderboard-wrap {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  width: 100%;
}
.live-leaderboard-wrap table {
  min-width: 640px;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────

def show_headshot_cropped_card(img_value: str | None, height_px: int = 250, player_name: str | None = None) -> None:
    s = str(img_value).strip().strip('"').strip("'").strip() if img_value else ""
    if not s or s.lower() in {"none", "nan"}:
        initials = ""
        if player_name:
            parts = player_name.replace(",", " ").split()
            initials = "".join(p[0].upper() for p in parts if p)[:2]
        st.markdown(
            f"<div class='dd-photo-card' style='height:{height_px}px;background:rgba(80,80,80,0.2);"
            f"border-radius:22px;display:flex;align-items:center;justify-content:center;"
            f"font-size:{height_px // 4}px;font-weight:700;color:rgba(255,255,255,0.2)'>{initials}</div>",
            unsafe_allow_html=True,
        )
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

def show_headshot(img_value: str | None, width: int = 90, player_name: str | None = None) -> None:
    s = str(img_value).strip() if img_value else ""
    if s and s.lower() not in {"none", "nan"}:
        p = Path(s)
        if p.exists():
            st.image(str(p), width=width)
            return
        p2 = REPO_ROOT / s
        if p2.exists():
            st.image(str(p2), width=width)
            return
        st.image(s, width=width)
        return
    # Placeholder when no image
    initials = ""
    if player_name:
        parts = player_name.replace(",", " ").split()
        initials = "".join(p[0].upper() for p in parts if p)[:2]
    st.markdown(
        f"<div style='width:{width}px;height:{width}px;background:rgba(80,80,80,0.25);"
        f"border-radius:8px;display:flex;align-items:center;justify-content:center;"
        f"font-size:{width // 3}px;font-weight:700;color:rgba(255,255,255,0.25)'>{initials}</div>",
        unsafe_allow_html=True,
    )

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
        show_headshot(url, width=headshot_width, player_name=player_name)
        return

    c_img, c_txt = st.columns([2, 8], vertical_alignment="center")
    with c_img:
        show_headshot(url, width=headshot_width, player_name=player_name)

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
    all_tours: bool = False,
) -> pd.DataFrame:
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    df = rounds_df.copy()

    if "tour" in df.columns and not all_tours:
        df = df[df["tour"] == "pga"].copy()

    df["dg_id"] = pd.to_numeric(df["dg_id"], errors="coerce")
    df = df[df["dg_id"].isin([int(x) for x in dg_ids])].copy()

    if "round_date" in df.columns and df["round_date"].notna().any():
        date_col = "round_date"
    elif "event_completed" in df.columns:
        date_col = "event_completed"
    else:
        raise ValueError("Rounds data must include round_date or event_completed.")

    # Include rows where date is unknown (NaT = pre-2022 data without round_date) OR before cutoff
    df = df[df[date_col].isna() | (df[date_col] < ts)].copy()
    if df.empty:
        return pd.DataFrame({"dg_id": list(dg_ids)})

    df = df.sort_values(["dg_id", date_col], ascending=[True, False], na_position="last")

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
    extra_course_nums: list[int] | None = None,
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
        _all_course_nums = {course_num}
        if extra_course_nums:
            _all_course_nums.update(extra_course_nums)
        r = r.loc[r[course_col_r].isin(_all_course_nums)].copy()

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

                # Include rows where event_end is unknown (2017–2021 lack round_date)
                # OR where event_end is known and before the cutoff
                keep = rr["event_end"].isna() | (rr["event_end"] <= cutoff_ts)
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

    if combined.empty or "year" not in combined.columns:
        return pd.DataFrame(), pd.DataFrame()

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
        # Keep current-week event as default until Sunday ~8pm (start_date + 3 days + 20 hours).
        # After that, flip to next week's event.
        now = pd.Timestamp.now()
        event_end_window = sched[date_col2] + pd.Timedelta(days=3, hours=20)
        is_upcoming = event_end_window >= now
        sched = sched.assign(__is_upcoming=is_upcoming)
        sched = sched.sort_values(["__is_upcoming", date_col2], ascending=[False, True]).drop(columns="__is_upcoming")

        upcoming = sched.loc[event_end_window >= now, "__row_id"].tolist()
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
        help="Select the tournament to analyze. Defaults to the current or next scheduled event.",
    )

    selected_row = sched.loc[sched["__row_id"].astype(int) == int(selected_row_id)].iloc[0]

    course_num_val = pd.to_numeric(selected_row.get("course_num"), errors="coerce")
    course_num = int(course_num_val) if pd.notna(course_num_val) else None
    st.session_state["selected_course_num"] = course_num

    img_url = selected_row.get("image", None)
    img_url = None if img_url is None else str(img_url).strip()
    if img_url and img_url.lower() not in {"nan", "none", "null", "<unset>", ""}:
        st.image(img_url, use_container_width=True)

    event_id_val = pd.to_numeric(selected_row.get("event_id"), errors="coerce")
    event_id = int(event_id_val) if pd.notna(event_id_val) else None

    # ── Data status ───────────────────────────────────────────────────────────
    import json as _json

    def _fmt_ts(ts: str) -> str:
        """'2026-03-15 21:11:16 ET' -> 'Mar 15, 9:11pm ET'"""
        try:
            _dt = pd.to_datetime(ts.replace(" ET", "").strip())
            return _dt.strftime("%-m/%-d %-I:%M%p").lower().replace("am", "am ET").replace("pm", "pm ET")
        except Exception:
            return ts

    _changelog_path  = INUSE_DIR / "data_changelog.json"
    _fieldstatus_path = INUSE_DIR / "field_status.json"

    # Last-updated per source type
    try:
        _events = _json.loads(_changelog_path.read_text()).get("events", []) if _changelog_path.exists() else []
        _last = {}
        for _ev in _events:
            _t = _ev.get("type", "")
            if _t not in _last:
                _last[_t] = _ev.get("timestamp", "")
    except Exception:
        _last = {}

    # Field status
    try:
        _fs = _json.loads(_fieldstatus_path.read_text()) if _fieldstatus_path.exists() else {}
    except Exception:
        _fs = {}

    # ── Data sources ──────────────────────────────────────────────────────────
    with st.expander("Data Sources", expanded=False):
        _SOURCE_ROWS = [
            ("rounds_refresh",     "Rounds"),
            ("field_odds_refresh", "Field / Odds"),
            ("live_odds_refresh",  "Live Odds"),
            ("approach_refresh",   "Approach Skill"),
        ]
        for _type_key, _label in _SOURCE_ROWS:
            _ts  = _last.get(_type_key, "")
            _disp = _fmt_ts(_ts) if _ts else "—"
            _color = "rgba(100,200,100,0.9)" if _ts else "rgba(120,120,120,0.4)"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<span style='font-size:13px;font-weight:600;color:rgba(210,210,210,0.85)'>{_label}</span>"
                f"<span style='font-size:11px;color:{_color};text-align:right'>{_disp}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Field status ──────────────────────────────────────────────────────────
    # Determine current/next week event_ids from the schedule (not the stale JSON keys)
    _sched_now = pd.Timestamp.now()
    _sched_copy = schedule_df.copy() if schedule_df is not None else pd.DataFrame()
    if not _sched_copy.empty:
        _date_c = "start_date" if "start_date" in _sched_copy.columns else ("event_date" if "event_date" in _sched_copy.columns else None)
        if _date_c:
            _sched_copy[_date_c] = pd.to_datetime(_sched_copy[_date_c], errors="coerce")
            _sched_copy["_end_w"] = _sched_copy[_date_c] + pd.Timedelta(days=3, hours=20)
            # In-progress: started and still within the end window
            _in_progress = _sched_copy.loc[
                (_sched_copy[_date_c] <= _sched_now) & (_sched_copy["_end_w"] >= _sched_now), "event_id"
            ].dropna().astype(int).tolist()
            _upcoming_sorted = _sched_copy.loc[_sched_copy[_date_c] > _sched_now].sort_values(_date_c)
            _all_upcoming = _upcoming_sorted["event_id"].dropna().astype(int).tolist()
            # "This Week" = in-progress event OR (if none) the next upcoming event
            if _in_progress:
                _current_eids = _in_progress
                _next_eids    = _all_upcoming
            else:
                _current_eids = _all_upcoming[:1]
                _next_eids    = _all_upcoming[1:]
        else:
            _current_eids, _next_eids = [], []
    else:
        _current_eids, _next_eids = [], []

    # Build lookup by event_name (DataGolf event_ids don't always match schedule event_ids)
    _fs_by_name = {
        str(v.get("event_name", "")).strip().lower(): v
        for v in _fs.values() if isinstance(v, dict) and v.get("event_name")
    }
    # Map schedule event_id -> field status entry via name match
    _sched_name_col = "event_name" if "event_name" in _sched_copy.columns else None

    def _lookup_fs(eids):
        if not eids or not _sched_name_col:
            return None
        row = _sched_copy.loc[_sched_copy["event_id"] == eids[0]]
        if row.empty:
            return None
        name = str(row.iloc[0][_sched_name_col]).strip().lower()
        return _fs_by_name.get(name)

    with st.expander("Field Status", expanded=False):
        for _week_label, _eids in [("This Week", _current_eids), ("Next Up", _next_eids[:1])]:
            _wf = _lookup_fs(_eids)
            if not _wf:
                st.markdown(
                    f"<div style='padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                    f"<span style='font-size:13px;font-weight:600;color:rgba(210,210,210,0.85)'>{_week_label}</span>"
                    f"<span style='font-size:11px;color:rgba(120,120,120,0.4);margin-left:10px'>Not yet loaded</span></div>",
                    unsafe_allow_html=True,
                )
                continue

            _ev_name  = _wf.get("event_name", "")
            _n        = _wf.get("player_count", "?")
            _updated  = _fmt_ts(_wf.get("last_updated", ""))
            _changes  = _wf.get("recent_changes", [])
            _latest_c = _changes[0] if _changes else {}
            _added    = _latest_c.get("added", [])
            _dropped  = _latest_c.get("withdrawn", [])

            _change_html = ""
            if _added:
                _names = ", ".join(_added[:5]) + ("…" if len(_added) > 5 else "")
                _change_html += f"<div style='font-size:11px;color:rgba(100,200,100,0.85);margin-top:4px'>↑ {len(_added)} added: {_names}</div>"
            if _dropped:
                _names = ", ".join(_dropped[:5]) + ("…" if len(_dropped) > 5 else "")
                _change_html += f"<div style='font-size:11px;color:rgba(255,120,100,0.85);margin-top:2px'>↓ {len(_dropped)} withdrawn: {_names}</div>"

            st.markdown(
                f"<div style='padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<span style='font-size:13px;font-weight:600;color:rgba(210,210,210,0.85)'>{_week_label}</span>"
                f"<span style='font-size:11px;color:rgba(100,200,100,0.9)'>{_updated}</span></div>"
                f"<div style='font-size:12px;color:rgba(170,170,170,0.7);margin-top:3px'>{_ev_name} · {_n} players</div>"
                f"{_change_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Sidebar weather summary ────────────────────────────────────────────
    _weather_api_key = st.secrets.get("WEATHER_API_KEY", "")
    _wx_q = ""
    for _col in ["weather_q", "weather_location", "location", "course_name"]:
        if _col in selected_row.index:
            _v = str(selected_row.get(_col, "")).strip()
            if _v.lower() not in {"nan", "none", "null", "", "<unset>"}:
                _wx_q = _v
                break

    _wx_start = None
    for _dc in ["start_date", "date_start", "event_date"]:
        if _dc in selected_row.index:
            _wx_start = pd.to_datetime(selected_row.get(_dc), errors="coerce")
            if pd.notna(_wx_start):
                break

    if _weather_api_key and _wx_q and _wx_start is not None and pd.notna(_wx_start):
        _today       = pd.Timestamp.now().normalize()
        _r1_away     = (_wx_start.normalize() - _today).days
        _r4_away     = _r1_away + 3

        st.markdown(
            "<div style='font-size:11px;font-weight:700;letter-spacing:0.08em;"
            "color:rgba(130,130,130,0.6);text-transform:uppercase;margin:18px 0 10px'>"
            "Weather</div>",
            unsafe_allow_html=True,
        )

        if _r4_away < -1:
            st.markdown(
                "<div style='font-size:11px;color:rgba(130,130,130,0.5)'>Event concluded.</div>",
                unsafe_allow_html=True,
            )
        elif _r1_away > 6:
            st.markdown(
                f"<div style='font-size:11px;color:rgba(130,130,130,0.5)'>"
                f"Forecast available in ~{_r1_away - 6} days.</div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                from weather_tab import _fetch_forecast
                _days_fetch = max(1, min(_r4_away + 2, 7))
                _wx_data    = _fetch_forecast(_weather_api_key, _wx_q, _days_fetch)
                _wx_loc        = _wx_data.get("location", {}).get("name", _wx_q)
                _wx_updated_raw = _wx_data.get("current", {}).get("last_updated", "")
                _wx_updated_str = ""
                if _wx_updated_raw:
                    try:
                        _wx_updated_str = pd.to_datetime(_wx_updated_raw).strftime("%-m/%-d %-I:%M%p").lower()
                    except Exception:
                        _wx_updated_str = _wx_updated_raw

                st.markdown(
                    f"<div style='font-size:11px;color:rgba(130,130,130,0.5);margin-bottom:8px'>"
                    f"{_wx_loc}"
                    f"{'  ·  <span style=\"font-size:10px\">' + _wx_updated_str + '</span>' if _wx_updated_str else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                _WIND_COLORS = [(25, "#ef4444"), (18, "#f97316"), (12, "#fbbf24"), (0, "#22c55e")]
                _RAIN_COLORS = [(60, "#60a5fa"), (30, "#93c5fd"), (0, "rgba(150,150,150,0.5)")]

                def _wind_color(mph):
                    for thresh, col in _WIND_COLORS:
                        if mph >= thresh:
                            return col
                    return "#22c55e"

                def _rain_color(pct):
                    for thresh, col in _RAIN_COLORS:
                        if pct >= thresh:
                            return col
                    return "rgba(150,150,150,0.5)"

                _round_labels = ["R1", "R2", "R3", "R4"]
                for _ri, _rd in enumerate([_wx_start + pd.Timedelta(days=i) for i in range(4)]):
                    _rd_str  = _rd.strftime("%Y-%m-%d")
                    _fd_list = _wx_data.get("forecast", {}).get("forecastday", [])
                    _fd      = next((d for d in _fd_list if d.get("date") == _rd_str), None)
                    if _fd is None:
                        continue
                    _day      = _fd.get("day", {})
                    _temp     = _day.get("maxtemp_f")
                    _wind     = _day.get("maxwind_mph")
                    _rain     = _day.get("daily_chance_of_rain", 0)
                    _cond     = (_day.get("condition") or {}).get("text", "")
                    _label    = _round_labels[_ri]
                    _date_fmt = _rd.strftime("%-m/%-d")
                    _wc       = _wind_color(_wind or 0)
                    _rc       = _rain_color(_rain or 0)
                    _temp_str = f"{_temp:.0f}°F" if _temp is not None else "—"
                    _wind_str = f"{_wind:.0f} mph" if _wind is not None else "—"
                    _rain_str = f"{int(_rain)}%" if _rain is not None else "—"

                    st.markdown(
                        f"<div style='padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<span style='font-size:12px;font-weight:700;color:rgba(210,210,210,0.9)'>"
                        f"{_label} <span style='font-weight:400;color:rgba(150,150,150,0.6);font-size:10px'>{_date_fmt}</span></span>"
                        f"<span style='font-size:11px;color:rgba(160,160,160,0.6)'>{_cond}</span>"
                        f"</div>"
                        f"<div style='display:flex;gap:10px;margin-top:3px'>"
                        f"<span style='font-size:11px;color:rgba(200,200,200,0.7)'>🌡 {_temp_str}</span>"
                        f"<span style='font-size:11px;color:{_wc};font-weight:600'>💨 {_wind_str}</span>"
                        f"<span style='font-size:11px;color:{_rc}'>🌧 {_rain_str}</span>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as _wx_err:
                st.caption(f"Weather data unavailable — {_wx_err}")

    # ── Field-Adjusted SG toggle ───────────────────────────────────────────
    st.divider()
    use_adjusted_sg = st.toggle(
        "Field-Adjusted SG",
        value=False,
        key="use_adjusted_sg",
        help=(
            "Adjusts all SG metrics for field strength across tours. "
            "Rounds from PGA, DP World, and LIV are weighted by the quality of the field played, "
            "anchored to a typical PGA Tour event. "
            "Useful for comparing players across tours on a level scale."
        ),
    )
    if use_adjusted_sg:
        decay_lambda = st.slider(
            "Recency decay",
            min_value=0.00,
            max_value=0.10,
            value=0.03,
            step=0.01,
            key="sg_decay_lambda",
            help=(
                "How fast older rounds lose influence on the field-quality estimate. "
                "0 = equal weight all time  |  0.03 = half-life ~23 rounds (default)  |  0.10 = half-life ~7 rounds"
            ),
        )
    else:
        decay_lambda = 0.03  # unused but keeps the cached call consistent


sched_row = selected_row

if "start_date" in sched_row.index and pd.notna(sched_row.get("start_date")):
    cutoff = pd.to_datetime(sched_row["start_date"]) - pd.Timedelta(days=1)
elif "event_date" in sched_row.index and pd.notna(sched_row.get("event_date")):
    cutoff = pd.to_datetime(sched_row["event_date"]) - pd.Timedelta(days=1)
else:
    cutoff = pd.Timestamp.today()

if event_id is None:
    st.markdown(
        "<div style=’background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);"
        "border-radius:12px;padding:32px 24px;text-align:center;margin-top:40px’>"
        "<div style=’font-size:18px;font-weight:700;color:rgba(200,200,200,0.7);margin-bottom:8px’>"
        "No event data yet</div>"
        "<div style=’font-size:13px;color:rgba(130,130,130,0.5)’>"
        "This event doesn’t have an ID assigned yet. Field, odds, and player stats will appear once the event goes live."
        "</div></div>",
        unsafe_allow_html=True,
    )



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

    # Synthetic field fallback for future events with no real field data yet
    if not field_ids and schedule is not None and not schedule.empty:
        try:
            _MAJOR_IDS = {14, 26, 33, 100}
            _SIG_IDS   = {5, 7, 9, 11, 12, 23, 34, 480, 541}
            _sched_row = schedule[
                pd.to_numeric(schedule.get("event_id", pd.Series(dtype=float)), errors="coerce") == int(event_id)
            ].head(1)
            _etype = str(_sched_row["event_type"].iloc[0]).upper() if not _sched_row.empty and "event_type" in _sched_row.columns else "REGULAR"

            _rdf = pd.read_csv(
                INUSE_DIR / "combined_roundlevel_2024_present.csv",
                usecols=["dg_id", "player_name", "tour", "event_id", "round_date"],
                low_memory=False,
            )
            _rdf["round_date"] = pd.to_datetime(_rdf["round_date"], errors="coerce")
            _cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)

            if _etype == "MAJOR":
                _pool = _rdf[
                    (_rdf["event_id"].isin(_MAJOR_IDS)) &
                    (_rdf["round_date"] >= _cutoff)
                ].drop_duplicates("dg_id")[["dg_id", "player_name"]]
            elif _etype == "SIGNATURE":
                _pool = _rdf[
                    (_rdf["tour"].str.lower() == "pga") &
                    (_rdf["event_id"].isin(_SIG_IDS)) &
                    (_rdf["round_date"] >= _cutoff)
                ].drop_duplicates("dg_id")[["dg_id", "player_name"]]
            else:
                _pga_recent = _rdf[
                    (_rdf["tour"].str.lower() == "pga") &
                    (_rdf["round_date"] >= _cutoff)
                ]
                _ecounts = _pga_recent.groupby("dg_id")["event_id"].nunique()
                _qualified = _ecounts[_ecounts >= 10].index
                _pool = _pga_recent[_pga_recent["dg_id"].isin(_qualified)].drop_duplicates("dg_id")[["dg_id", "player_name"]]

            if not _pool.empty:
                field_ids = _pool["dg_id"].astype(int).tolist()
                field_ev = _pool.copy()
                field_ev["event_id"] = event_id
                field_ev["year"] = SEASON_YEAR
                field_ev["close_odds"] = pd.NA
        except Exception as _e:
            st.sidebar.warning(f"Synthetic field error: {_e}")

    # Fallback: if close_odds is missing, pull from this_week_odds.csv
    # Only apply when the selected event matches this week's field — odds have no event_id
    # so applying them to a different/future event would show the wrong tournament's odds.
    _this_week_event_id = None
    _tw_field_path = INUSE_DIR / "this_week_field.csv"
    if _tw_field_path.exists():
        try:
            _tw_ids = pd.to_numeric(
                pd.read_csv(_tw_field_path, usecols=["event_id"])["event_id"], errors="coerce"
            ).dropna().unique()
            _this_week_event_id = int(_tw_ids[0]) if len(_tw_ids) > 0 else None
        except Exception:
            pass

    _odds_csv_path = INUSE_DIR / "this_week_odds.csv"
    if (
        _odds_csv_path.exists()
        and "close_odds" in field_ev.columns
        and field_ev["close_odds"].isna().all()
        and event_id is not None
        and _this_week_event_id is not None
        and int(event_id) == _this_week_event_id
    ):
        try:
            _odds_csv = pd.read_csv(_odds_csv_path)
            _odds_csv["dg_id"] = pd.to_numeric(_odds_csv["dg_id"], errors="coerce")
            # Pick best available bookmaker column
            _bookie_prefs = ["draftkings", "betmgm", "datagolf_baseline", "bovada", "betonline"]
            _bookie_col = next((c for c in _bookie_prefs if c in _odds_csv.columns and _odds_csv[c].notna().any()), None)
            if _bookie_col:
                _odds_merge = _odds_csv[["dg_id", _bookie_col]].rename(columns={_bookie_col: "_tmp_odds"}).dropna(subset=["dg_id"])
                field_ev = field_ev.merge(_odds_merge, on="dg_id", how="left")
                field_ev["close_odds"] = field_ev["close_odds"].fillna(field_ev["_tmp_odds"])
                field_ev = field_ev.drop(columns=["_tmp_odds"])
        except Exception:
            pass

only_in_field = bool(field_ids)  # default True when field is available, else show full universe

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
# Exclude DP World Tour events (IDs >= 10000, e.g. 2026105) — only PGA/LIV events have small IDs
ids_2026_all   = set(
    f_univ.loc[
        (f_univ["year"] == 2026) & (f_univ["event_id"].fillna(0).astype(int) < 10000),
        "dg_id"
    ].astype(int).unique().tolist()
)

universe_ids = sorted(ids_2026_all | ids_2025_4plus)

name_pool = f_univ[f_univ["dg_id"].isin(universe_ids)].copy()
name_pool["is_2026"] = (name_pool["year"] == 2026).astype(int)
name_pool = name_pool.sort_values(["dg_id", "is_2026", "year"], ascending=[True, False, False])

universe = name_pool.drop_duplicates(subset=["dg_id"], keep="first")[["dg_id", "player_name"]].copy()

# Fill blank player_name from rounds_df (covers players missing from Fields.xlsx)
if "player_name" in rounds_df.columns:
    _rnd_names = (
        rounds_df.dropna(subset=["player_name"])
        .assign(_pn=lambda d: d["player_name"].astype(str).str.strip())
        .query("_pn != '' and _pn != 'nan' and _pn != 'None'")
        .groupby("dg_id")["_pn"].first()
    )
    _blank = universe["player_name"].isna() | universe["player_name"].astype(str).str.strip().isin(["", "nan", "None"])
    universe.loc[_blank, "player_name"] = universe.loc[_blank, "dg_id"].map(_rnd_names)

if only_in_field:
    base_ids = field_ids
else:
    base_ids = universe["dg_id"].astype(int).unique().tolist()

base = universe[universe["dg_id"].isin(base_ids)].copy()


with st.spinner("Computing rolling stats (L40/L24/L12) from round-level data..."):
    if use_adjusted_sg:
        # Compute or retrieve cached field-adjusted rounds
        _adj_rounds = get_adjusted_rounds(rounds_df, decay_lambda=decay_lambda)
        # Replace raw sg_* columns with adj_sg_* so every consumer is unaware of the swap
        _adj_sg_cols = {c: c.replace("adj_sg_", "sg_") for c in _adj_rounds.columns if c.startswith("adj_sg_")}
        _sg_originals = list(_adj_sg_cols.values())  # e.g. ["sg_total", "sg_putt", ...]
        active_rounds_df = (
            _adj_rounds
            .drop(columns=[c for c in _sg_originals if c in _adj_rounds.columns])
            .rename(columns=_adj_sg_cols)
        )
        rolling = compute_rolling_stats(
            rounds_df=active_rounds_df, as_of_date=cutoff, dg_ids=base_ids,
            windows=(40, 24, 12), all_tours=True,
        )
    else:
        active_rounds_df = rounds_df
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
_odds_available = False
if odds_col is None and event_id is not None:
    _event_label_no_odds = selected_row.get("event_name", "this event") if "event_name" in selected_row.index else "this event"
    st.sidebar.caption(f"No odds column found for **{_event_label_no_odds}**.")
if odds_col:
    tmp = field_ev[["dg_id", odds_col]].copy()
    tmp["dg_id"] = pd.to_numeric(tmp["dg_id"], errors="coerce")
    out = out.merge(tmp, on="dg_id", how="left")
    _odds_available = out[odds_col].notna().any()
    if not _odds_available and event_id is not None:
        _event_label = selected_row.get("event_name", "this event") if "event_name" in selected_row.index else "this event"
        st.sidebar.info(f"Pre-tournament odds not yet released for **{_event_label}**. Odds will appear here once the API provider updates lines.")

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

# Fill any remaining blank player_name from rounds_df
if "player_name" in summary_top.columns and "player_name" in rounds_df.columns:
    _rnd_names = (
        rounds_df.dropna(subset=["player_name"])
        .assign(_pn=lambda d: d["player_name"].astype(str).str.strip())
        .query("_pn != '' and _pn != 'nan' and _pn != 'None'")
        .groupby("dg_id")["_pn"].first()
    )
    _blank = summary_top["player_name"].isna() | summary_top["player_name"].astype(str).str.strip().isin(["", "nan", "None"])
    summary_top.loc[_blank, "player_name"] = summary_top.loc[_blank, "dg_id"].map(_rnd_names)

_default_sort = "sg_total_L12" if "sg_total_L12" in summary_top.columns else None
if _default_sort:
    summary_top = summary_top.sort_values(_default_sort, ascending=False)

summary_top = summary_top.reset_index(drop=True)

# Load this week's and next week's field files
_this_week_field_path = INUSE_DIR / "this_week_field.csv"
_next_week_field_path = INUSE_DIR / "next_week_field.csv"
_this_week_field_df = pd.read_csv(_this_week_field_path) if _this_week_field_path.exists() else None
_next_week_field_df = pd.read_csv(_next_week_field_path) if _next_week_field_path.exists() else None
_live_active = is_tournament_live(_this_week_field_df) if _this_week_field_df is not None else False

# Determine which field file matches the selected event (for tee times / weather)
def _field_df_for_event(eid):
    """Return the field DataFrame whose event_id matches eid.
    Prefer this_week over next_week when both match — it has current tee times."""
    candidates = []
    for fdf in [_next_week_field_df, _this_week_field_df]:
        if fdf is None or fdf.empty:
            continue
        ids = pd.to_numeric(fdf["event_id"], errors="coerce").dropna().astype(int).unique()
        if eid in ids:
            tee_count = fdf["r1_teetime"].notna().sum() if "r1_teetime" in fdf.columns else 0
            candidates.append((tee_count, fdf))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return _this_week_field_df

def _field_path_for_event(eid):
    """Return the CSV path whose event_id matches eid.
    Prefer this_week over next_week when both match the same event —
    this_week is refreshed more frequently and has current tee times."""
    candidates = []
    for path, fdf in [(_next_week_field_path, _next_week_field_df), (_this_week_field_path, _this_week_field_df)]:
        if fdf is None or fdf.empty:
            continue
        ids = pd.to_numeric(fdf["event_id"], errors="coerce").dropna().astype(int).unique()
        if eid in ids:
            # Count how many r1_teetime values are non-null — more = better
            tee_count = fdf["r1_teetime"].notna().sum() if "r1_teetime" in fdf.columns else 0
            candidates.append((tee_count, str(path)))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return str(_this_week_field_path)

# Also check schedule directly — keeps Live visible even if field file has rolled over to next week
if not _live_active:
    _now_check = pd.Timestamp.now()
    for _, _sr in schedule_df.iterrows():
        _sd = pd.to_datetime(_sr.get("start_date"), errors="coerce")
        if pd.notna(_sd) and _sd <= _now_check <= _sd + pd.Timedelta(days=4):
            _live_active = True
            break

_live_label = "🔴 Live" if _live_active else "⚫ Live"

_MODEL_LAB_AVAILABLE = (Path(__file__).parent / "model_lab_tab.py").exists()
_OAD_GT_AVAILABLE = (Path(__file__).parent / "oad_game_theory_tab.py").exists()

TAB_NAMES = [
    "Overview",
    "Field",
    "Model",
    "Course",
    "Approach",
    "H2H",
    "Deep Dive",
    _live_label,
    "Archive",
] + (["Lab"] if _MODEL_LAB_AVAILABLE else []) + (["OAD"] if _OAD_GT_AVAILABLE else []) + ["Guide"]

# Apply any pending tab navigation (set by tab modules before rerun)
if "_pending_tab" in st.session_state:
    _pending = st.session_state.pop("_pending_tab")
    if _pending in TAB_NAMES:
        st.session_state.active_tab = _pending

if "active_tab" not in st.session_state or st.session_state.active_tab not in TAB_NAMES:
    st.session_state.active_tab = "Overview"


active_tab = st.segmented_control(
    "",
    TAB_NAMES,
    default=st.session_state.active_tab,
    key="active_tab",
)

# ── Event context bar ─────────────────────────────────────────────────────────
_ctx_name   = str(selected_row.get("event_name", "")).strip()
_ctx_course = ""
for _cc in ["course_name", "course", "location"]:
    _cv = str(selected_row.get(_cc, "")).strip()
    if _cv.lower() not in {"nan", "none", "null", "", "<unset>"}:
        _ctx_course = _cv
        break
_ctx_date = ""
for _dc in ["start_date", "event_date"]:
    _dv = selected_row.get(_dc)
    _ts = pd.to_datetime(_dv, errors="coerce")
    if pd.notna(_ts):
        _ctx_date = _ts.strftime("%b %-d")
        _ctx_date_end = (_ts + pd.Timedelta(days=3)).strftime("%-d, %Y")
        _ctx_date = f"{_ctx_date}–{_ctx_date_end}"
        break
_ctx_purse = ""
for _pc in ["total_purse", "purse", "prize_fund"]:
    _pv = selected_row.get(_pc)
    try:
        _pf = float(str(_pv).replace("$", "").replace(",", ""))
        _ctx_purse = f"${_pf/1_000_000:.1f}M"
        break
    except Exception:
        pass
_ctx_parts = [p for p in [_ctx_course, _ctx_date, _ctx_purse] if p]
_ctx_meta  = " · ".join(_ctx_parts)
if _ctx_name:
    st.markdown(
        f"<div style='font-size:12px;color:rgba(160,160,160,0.65);padding:4px 2px 10px;'>"
        f"<span style='font-weight:600;color:rgba(200,200,200,0.8)'>{_ctx_name}</span>"
        + (f"<span style='margin-left:10px'>{_ctx_meta}</span>" if _ctx_meta else "")
        + "</div>",
        unsafe_allow_html=True,
    )
# ─────────────────────────────────────────────────────────────────────────────

if event_id is None and active_tab not in {"Archive", "Guide", _live_label}:
    st.stop()

if active_tab == "Field":
    render_production_sg_tab(
        rounds_df=active_rounds_df,
        field_ids=field_ids,
        all_players=all_players,
        id_to_img=ID_TO_IMG,
        name_to_img=NAME_TO_IMG,
        schedule_df=schedule_df,
        field_df=_field_df_for_event(event_id) if event_id is not None else _this_week_field_df,
        event_id=event_id,
        cutoff_dt=cutoff,
        course_fit_df=pd.read_csv(COURSE_FIT_PATH) if COURSE_FIT_PATH.exists() else None,
        course_num=course_num_val if pd.notna(course_num_val) else None,
    )

elif active_tab == "Course":
    _cn = st.session_state.get("selected_course_num")
    _extra_cns = []
    if _cn is not None:
        _cn_int = int(_cn)
        for k, aliases in SAME_VENUE_COURSE_NUMS.items():
            if _cn_int == k:
                _extra_cns = aliases
            elif _cn_int in aliases:
                _extra_cns = [k] + [a for a in aliases if a != _cn_int]
    render_course_history_demo(
        course_num=_cn,
        rounds_df=active_rounds_df,
        ev_2017_2023=None,
        all_players=all_players,
        field_ids=field_ids,
        cutoff_dt=get_pre_event_cutoff_date(sched, int(event_id)) if event_id is not None else None,
        season_year=SEASON_YEAR,
        build_course_history_func=build_course_history_field_table,
        id_to_img=ID_TO_IMG,
        name_to_img=NAME_TO_IMG,
        event_id=event_id,
        extra_course_nums=_extra_cns,
    )

elif active_tab == "Approach":
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
        rounds_df=active_rounds_df,
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

elif active_tab == "Deep Dive":
    from player_deep_dive_tab import render_player_deep_dive_tab
    render_player_deep_dive_tab(
        summary_top=summary_top,
        rounds_df=active_rounds_df,
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

elif active_tab == "Archive":
    from event_browser_tab import render_event_browser_tab
    render_event_browser_tab(
        rounds_df=active_rounds_df,
        ID_TO_IMG=ID_TO_IMG,
        NAME_TO_IMG=NAME_TO_IMG,
    )

elif active_tab == "Guide":
    render_documentation_tab()

elif active_tab == "Lab" and _MODEL_LAB_AVAILABLE:
    from model_lab_tab import render_model_lab_tab
    render_model_lab_tab(
        rounds_df=active_rounds_df,
        field_ids=field_ids,
        all_players=all_players,
        cutoff_dt=cutoff,
        summary_top=summary_top,
        event_id=event_id,
        schedule_df=schedule_df,
    )

elif active_tab == "OAD" and _OAD_GT_AVAILABLE:
    from oad_game_theory_tab import render_oad_game_theory_tab
    render_oad_game_theory_tab()

# elif active_tab == "Model":
#     render_elite_finish_tab(rounds_df=rounds_df, fields_df=fields_df, event_id=event_id)

elif active_tab == "Model":
    from sg_production_tab import render_elite_finish_analysis
    render_elite_finish_analysis(
        rounds_df=active_rounds_df,
        field_ids=field_ids,
        all_players=all_players,
        cutoff_dt=cutoff,
        summary_top=summary_top,
    )

elif active_tab == "Overview":
    from overview_tab import render_overview_tab
    render_overview_tab(
        selected_row=selected_row,
        rounds_df=active_rounds_df,
        field_ev=field_ev,
        event_id=event_id,
        course_num=course_num,
        cutoff_dt=cutoff,
        course_fit_df=pd.read_csv(COURSE_FIT_PATH) if COURSE_FIT_PATH.exists() else None,
        id_to_img=ID_TO_IMG,
        weather_api_key=st.secrets.get("WEATHER_API_KEY", ""),
        schedule_df=schedule_df,
        tee_times_path=_field_path_for_event(event_id) if event_id is not None else str(_this_week_field_path),
        all_players=all_players,
    )

elif active_tab == _live_label:
    if not _live_active:
        st.markdown(
            "<div style='background:rgba(255,180,0,0.08);border:1px solid rgba(255,180,0,0.25);"
            "border-radius:10px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:12px'>"
            "<span style='font-size:20px'>⚫</span>"
            "<div><div style='font-size:13px;font-weight:700;color:rgba(255,200,80,0.9)'>No active tournament</div>"
            "<div style='font-size:12px;color:rgba(180,180,180,0.6);margin-top:2px'>"
            "Showing last available live data. Once R4 results are finalized they will appear in Event Archive.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    render_live_tab(
        field_df=_this_week_field_df,  # live tab always uses current week
        id_to_img=ID_TO_IMG,
    )