from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List

# ============================================================
# PROJECT ROOT + DATA PATHS
# ============================================================

# This assumes: PROJECT_ROOT / Scripts / config.py (this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_ROOT: Path = PROJECT_ROOT / "Data"
IN_USE_DIR: Path = DATA_ROOT / "in Use"

SCRIPTS_DIR: Path = PROJECT_ROOT / "Scripts"

# Core combined rounds file we've already built
COMBINED_ROUNDS_ALL_PATH: Path = IN_USE_DIR / "combined_rounds_all_2017_2026.csv"

# Event skill file (avg field strength etc.)
EVENT_SKILL_PATH: Path = IN_USE_DIR / "event_skill.xlsx"

# Odds + results workbook (Excel)
ODDS_AND_RESULTS_PATH: Path = IN_USE_DIR / "Odds_and_Results.xlsx"

# ============================================================
# PATHS EXPECTED BY Scripts/data_io.py
# ============================================================

# data_io.py expects this exact name:
COMBINED_ROUNDS_PATH: Path = IN_USE_DIR / "combined_rounds_all_2017_2026.csv"

# If you want to keep your existing name too, keep it as an alias:
COMBINED_ROUNDS_ALL_PATH: Path = COMBINED_ROUNDS_PATH

# Templates expected by data_io.py (these are format strings)
COURSE_FIT_TEMPLATE: Path = IN_USE_DIR / "course_fit_{season}_dg_style_5attr.csv"
PLAYER_SKILL_TEMPLATE: Path = IN_USE_DIR / "player_skills_{season}.csv"
OAD_TEMPLATE: Path = IN_USE_DIR / "OAD_{season}.xlsx"
PRESEASON_TEMPLATE: Path = IN_USE_DIR / "preseason_{season}.csv"

# New: files used by weekly_view.py (don’t hard-code absolute paths there)
YTD_TRACKER_PATH: Path = IN_USE_DIR / "ytd_tracker.csv"
COURSE_SENSITIVITY_PATH: Path = DATA_ROOT / "Clean" / "Processed" / "course_sensitivity_table.csv"


# OAD schedules (Excel)
OAD_2024_PATH: Path = IN_USE_DIR / "OAD_2024.xlsx"
OAD_2025_PATH: Path = IN_USE_DIR / "OAD_2025.xlsx"
OAD_2026_PATH: Path = IN_USE_DIR / "OAD_2026.xlsx"

OAD_PATHS = {
    2024: OAD_2024_PATH,
    2025: OAD_2025_PATH,
    2026: OAD_2026_PATH,
}


# Where we'll drop derived files for modeling (course fit, baselines, preseason, etc.)
DERIVED_DIR: Path = IN_USE_DIR / "derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SEASONS / OAD CONTEXT
# ============================================================

# The season you are actually playing (real OAD decisions)
LIVE_OAD_SEASON: int = 2026

# Seasons we will backtest against with this codebase
BACKTEST_SEASONS: List[int] = [2024, 2025]


# ============================================================
# FILE REGISTRY (OPTIONAL – FILL AS NEEDED)
# ============================================================

@dataclass(frozen=True)
class SeasonFiles:
    """Paths for season-specific inputs."""
    odds_path: Path            # all markets odds for the season
    master_results_path: Path  # season master results (with earnings, finish, etc.)
    oad_schedule_path: Path    # your OAD schedule (event_id_fixed, rank, purse, etc.)
    preseason_path: Path | None = None  # optional preseason shortlist file


SEASON_FILES: Dict[int, SeasonFiles] = {
    # Example template – you can wire these later if you want
    # 2024: SeasonFiles(
    #     odds_path=IN_USE_DIR / "2024_all_markets_odds.csv",
    #     master_results_path=IN_USE_DIR / "2024_master_results.csv",
    #     oad_schedule_path=OAD_2024_PATH,
    #     preseason_path=IN_USE_DIR / "preseason_2024.csv",
    # ),
    # 2025: SeasonFiles(
    #     odds_path=IN_USE_DIR / "2025_all_markets_odds.csv",
    #     master_results_path=IN_USE_DIR / "2025_master_results.csv",
    #     oad_schedule_path=OAD_2025_PATH,
    #     preseason_path=IN_USE_DIR / "preseason_2025.csv",
    # ),
}


# ============================================================
# EVENT TIERS / TYPES
# ============================================================

class EventTier(str, Enum):
    MAJOR = "major"
    SIGNATURE = "signature"
    REGULAR = "regular"
    LIV = "liv"


@dataclass(frozen=True)
class TierConfig:
    """Configuration for how we label event tiers."""
    major_event_ids: List[int]
    signature_event_ids: List[int]
    liv_event_ids: List[int]


# Placeholder: we will fill this with real IDs once we wire schedule/master.
TIER_CONFIG = TierConfig(
    major_event_ids=[],
    signature_event_ids=[],
    liv_event_ids=[],
)


def infer_event_tier(event_id: int, event_name: str | None = None) -> EventTier:
    """
    Basic tier inference helper. For now: mostly ID-based, with a
    lightweight name fallback we can refine later.
    """
    if event_id in TIER_CONFIG.major_event_ids:
        return EventTier.MAJOR
    if event_id in TIER_CONFIG.signature_event_ids:
        return EventTier.SIGNATURE
    if event_id in TIER_CONFIG.liv_event_ids:
        return EventTier.LIV

    # Weak text fallback until we wire real mappings
    if event_name:
        name = event_name.lower()
        if (
            "masters" in name
            or "open championship" in name
            or "u.s. open" in name
            or "us open" in name
            or "pga championship" in name
        ):
            return EventTier.MAJOR

    return EventTier.REGULAR


# ============================================================
# MODEL / FEATURE CONSTANTS
# ============================================================

# Default odds to give a player when no historical odds are available
DEFAULT_DECIMAL_ODDS: float = 1000.0

# Recent-form windows (last N rounds) for helper tables
RECENT_WINDOWS: tuple[int, int, int] = (40, 24, 12)

# Minimum rounds for including a player in baselines / recent-form stats
MIN_ROUNDS_BASELINE: int = 20

# For course-fit importance shrinkage (n_obs / (n_obs + shrink_k))
COURSE_FIT_SHRINK_K: float = 400.0

# Rolling / decay hyperparameters per stat (in rounds) – used by baselines & course fit
HALF_LIVES_ROUNDS = {
    "driving_dist": 25,
    "driving_acc": 25,
    "sg_app": 60,
    "sg_arg": 60,
    "sg_putt": 120,
    "sg_total": 80,
}


# ============================================================
# SANITY HELPERS
# ============================================================

def assert_file_exists(path: Path, label: str | None = None) -> None:
    """Convenience check for critical files."""
    if not path.exists():
        prefix = f"[{label}] " if label else ""
        raise FileNotFoundError(f"{prefix}Expected file does not exist: {path}")