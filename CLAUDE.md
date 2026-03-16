# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
source .venv/bin/activate
streamlit run Scripts/Stats.py
```

Opens at `http://localhost:8501`. Python 3.13, virtual env at `.venv/`.

## Data Refresh Scripts

```bash
# Pull live DraftKings odds from DataGolf (run during tournaments)
python Scripts/refresh_live_odds.py

# Regenerate course greens reference (grass types, course metadata)
python Scripts/export_course_greens_reference.py
```

## Architecture

This is a **Streamlit golf analytics dashboard** for PGA/DP World/LIV tour analysis, used primarily for tournament betting and player evaluation.

### Entry Point: `Scripts/Stats.py`

The main script (~1550 lines) handles:
1. **Data loading** with `@st.cache_data` — loads ~250MB of CSV/Excel data once on startup
2. **Rolling stat computation** — calculates L12/L24/L40 (trailing 12/24/40 rounds) for all SG metrics
3. **`summary_top` dataframe** — central merged dataframe combining odds, YTD stats, finishes, and rolling performance per player
4. **Tab routing** via Streamlit segmented control → delegates to module-level `render_*_tab()` functions

### Tab Modules (`Scripts/`)

Each tab is a self-contained module that receives prepared dataframes and renders via Streamlit:

| Module | Tab |
|--------|-----|
| `overview_tab.py` | Event Overview |
| `sg_production_tab.py` | Field SG (strokes gained breakdown) |
| `course_history_proto.py` | Course History |
| `approach_skill_tab.py` | Approach Skill |
| `h2h_visual_tab.py` | Head-to-Head comparisons |
| `player_deep_dive_tab.py` | Player Deep Dive |
| `event_browser_tab.py` | Event Archive |
| `live_tab.py` | Live (conditional — shown during active tournaments) |
| `weather_tab.py` | Weather |
| `grass_putting_deepdive.py` | Grass type putting analysis |

### Key Data Files (`Data/in Use/`)

| File | Purpose |
|------|---------|
| `combined_roundlevel_2024_present.csv` | Recent round-level data with full SG stats |
| `combined_rounds_all_2017_2026.csv` | Master round history (68 MB) |
| `All_players.xlsx` | Player master list (dg_id, name, headshot URLs) |
| `OAD_2026_Schedule.xlsx` | 2026 schedule with course metadata |
| `Fields.xlsx` | Tournament fields, finish results, predictions |
| `approach_skill_all_periods.csv` | Approach skill by distance bucket |
| `course_fit_weights_predictive_*.parquet` | ML course fit weights |
| `this_week_field.csv` | Current week's field and tee times |
| `this_week_odds.csv` / `this_week_live_odds.csv` | Betting odds |

Raw tour data lives in `Data/Raw/{PGA,EURO,LIV}/`; cleaned combined rounds in `Data/Clean/Combined/`.

### Data Flow

```
Data/in Use/*.csv + *.xlsx
  → Stats.py loads & caches
  → compute rolling L12/L24/L40 per player
  → merge odds + YTD + finishes → summary_top
  → route to tab renderer → Streamlit output
```

### Config & Secrets

API keys resolved in priority order by `src/config.py`:
1. Environment variables
2. Streamlit secrets (`.streamlit/secrets.toml`)
3. `.env` file

Required keys: `DATAGOLF_API_KEY`, `WEATHER_API_KEY`

### Key Data Concepts

- **dg_id**: DataGolf player ID — the primary player key across all datasets
- **SG metrics**: `sg_putt`, `sg_arg`, `sg_app`, `sg_ott`, `sg_t2g`, `sg_total`
- **Rolling windows**: `L12`, `L24`, `L40` (trailing round counts)
- **Grass types**: Poa Annua, Bermuda, Bentgrass, Fescue, Ryegrass — used to segment putting/approach analysis
- **Live mode**: `is_tournament_live()` check gates the Live tab; `Data/live/` stores live snapshots
