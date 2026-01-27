from pathlib import Path

BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

COMBINED_ROUNDS_PATH = BASE_DATA_DIR / "in Use" / "combined_rounds_all_2017_2026.csv"
COURSE_FIT_TEMPLATE = BASE_DATA_DIR / "in Use" / "course_fit_{season}_dg_style_5attr.csv"
EVENT_SKILL_PATH = BASE_DATA_DIR / "in Use" / "event_skill.xlsx"
OAD_TEMPLATE = BASE_DATA_DIR / "in Use" / "OAD_{season}.xlsx"
ODDS_AND_RESULTS_PATH = BASE_DATA_DIR / "in Use" / "Odds_and_Results.xlsx"
PRESEASON_TEMPLATE = BASE_DATA_DIR / "in Use" / "preseason_{season}.csv"

# NEW:
PRESEASON_SHORTLIST_TEMPLATE = BASE_DATA_DIR / "in Use" / "preseason_shortlist_{season}.csv"

PLAYER_SKILL_TEMPLATE = BASE_DATA_DIR / "in Use" / "player_skill_5attr_hist_to_{season}.csv"
