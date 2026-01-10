# Scripts/oad_schedule.py

from pathlib import Path
import pandas as pd

from Scripts.Archive.config import OAD_PATHS


def _clean_money(series: pd.Series) -> pd.Series:
    """
    Turn strings like '8,400,000' or '$8,400,000' into floats.
    """
    return (
        series.astype(str)
              .str.replace(r"[^0-9.\-]", "", regex=True)
              .replace("", "0")
              .astype(float)
    )


def load_oad_for_season(season: int) -> pd.DataFrame:
    """
    Load the OAD calendar for a given season (2024 / 2025).

    Expected columns in the Excel file (case-insensitive):
      - start_date
      - event_name
      - purse        (string with commas / $)
      - event_id_fixed
      - course_name (optional)
      - course_num_fixed or course_num (optional)
      - rank (optional)

    Returns a DataFrame with:
      - year
      - event_id
      - event_name
      - event_date
      - purse
      - course_num
      - rank

    NOTE: we do NOT filter by year here. The file itself is season-specific.
    """
    if season not in OAD_PATHS:
        raise ValueError(f"No OAD calendar path configured for season {season}.")

    path: Path = OAD_PATHS[season]
    df = pd.read_excel(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"start_date", "event_name", "purse", "event_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"OAD file for season {season} is missing columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    # Dates
    df["event_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["year"] = df["event_date"].dt.year.astype("Int64")

    # Event id
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")

    # Purse
    df["purse"] = _clean_money(df["purse"])

    # Course id: accept course_num_fixed or course_num
    course_col = None
    for cand in ("course_num", "course_num"):
        if cand in df.columns:
            course_col = cand
            break

    if course_col is not None:
        df["course_num"] = pd.to_numeric(df[course_col], errors="coerce").astype("Int64")
    else:
        df["course_num"] = pd.NA

    # Rank (optional)
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    else:
        df["rank"] = pd.NA

    return df[["year", "event_id", "event_name", "event_date", "purse", "course_num", "rank"]].copy()
