import numpy as np
import pandas as pd

TIER_WEIGHTS_DEFAULT = {
    "major":     {"sg12": 1.00, "ev": 0.00, "oad": 0.00},
    "signature": {"sg12": 0.65, "ev": 0.15, "oad": 0.20},
    "regular":   {"sg12": 0.60, "ev": 0.30, "oad": 0.10},
}

def _normalize_tier(tier: str) -> str:
    s = str(tier).strip().lower()
    if s in ("majors", "major"):
        return "major"
    if s in ("signature", "signature event", "sig"):
        return "signature"
    if s in ("regular", ""):
        return "regular"
    return s  # fall back

def _pct_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank in [0,1]. Higher is better. NaNs stay NaN.
    """
    s = pd.to_numeric(series, errors="coerce")
    n = s.notna().sum()
    if n <= 1:
        return pd.Series([np.nan] * len(s), index=s.index)
    r = s.rank(method="average", ascending=True)
    return (r - 1) / (n - 1)

def add_final_rank_score(
    summary: pd.DataFrame,
    tier: str,
    weights: dict | None = None,
    col_sg12: str = "sg_total_L12",
    col_ev: str = "ev_current_adj",
    col_oad: str = "oad_score",
) -> pd.DataFrame:
    """
    Adds:
      - pct_sg_total_L12
      - pct_ev_current_adj
      - pct_oad_score
      - final_rank_score
    Returns a copy.
    """
    wmap = weights or TIER_WEIGHTS_DEFAULT
    t = _normalize_tier(tier)
    w = wmap.get(t, wmap.get("regular"))

    out = summary.copy()

    # compute percentiles within the field for this event
    if col_sg12 in out.columns:
        out[f"pct_{col_sg12}"] = _pct_rank(out[col_sg12])
    else:
        out[f"pct_{col_sg12}"] = np.nan

    if col_ev in out.columns:
        out[f"pct_{col_ev}"] = _pct_rank(out[col_ev])
    else:
        out[f"pct_{col_ev}"] = np.nan

    if col_oad in out.columns:
        out[f"pct_{col_oad}"] = _pct_rank(out[col_oad])
    else:
        out[f"pct_{col_oad}"] = np.nan

    # final blend (NaNs -> 0 so missing one component doesn't nuke the score)
    out["final_rank_score"] = (
        w["sg12"] * out[f"pct_{col_sg12}"].fillna(0.0) +
        w["ev"]   * out[f"pct_{col_ev}"].fillna(0.0) +
        w["oad"]  * out[f"pct_{col_oad}"].fillna(0.0)
    )

    return out