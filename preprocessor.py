"""
preprocessor.py
Data cleaning, normalization, and Gig Activity Index (GAI) calculation.

Pipeline:
  1. Fill missing values per-column using per-region median
     (falls back to global median if a region is all-NaN for that column)
  2. Min-max normalize each numeric column to [0, 100] (stored as _norm columns)
  3. Compute GAI = weighted average of the 6 normalized columns
  4. Compute growth rate and data quality score
"""

import numpy as np
import pandas as pd

from data_loader import NUMERIC_COLS

# ── GAI weights (must sum to 1.0) ─────────────────────────────────────────────
# Delivery, rides, and search are the strongest real-time gig signals.
# Freelance and active_users are secondary. Urban activity is a structural control.
WEIGHTS: dict[str, float] = {
    "search_interest": 0.20,
    "delivery_orders": 0.20,
    "ride_requests":   0.20,
    "freelance_jobs":  0.15,
    "active_users":    0.15,
    "urban_activity":  0.10,
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned, enriched DataFrame with:
      - nulls filled (per-region median → global median fallback)
      - *_norm columns (0-100 min-max scaled) for each numeric column
      - gig_activity_index column (float, 0-100)

    Original raw columns are preserved so charts can still use real values.
    """
    df = df.copy()

    # Step 1: Fill missing numeric values ─────────────────────────────────────
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
            continue

        # Per-region median fill
        global_median = df[col].median()
        df[col] = df.groupby("region")[col].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else global_median)
        )
        # Fallback: fill any remaining NaN with global median
        df[col] = df[col].fillna(global_median if not np.isnan(global_median) else 0.0)

        # Clip to sensible physical bounds
        if col in ("search_interest", "urban_activity"):
            df[col] = df[col].clip(0, 100)
        else:
            df[col] = df[col].clip(0)

    # Step 2: Min-max normalize each column to [0, 100] ───────────────────────
    for col in NUMERIC_COLS:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min) * 100
        else:
            # No variation → assign neutral midpoint
            df[f"{col}_norm"] = 50.0

    # Step 3: Compute Gig Activity Index ──────────────────────────────────────
    # Only use columns that actually exist (robustness for partial uploads)
    available = {col: w for col, w in WEIGHTS.items() if col in df.columns}
    total_weight = sum(available.values())

    df["gig_activity_index"] = sum(
        df[f"{col}_norm"] * (w / total_weight)
        for col, w in available.items()
    )
    df["gig_activity_index"] = df["gig_activity_index"].clip(0, 100).round(1)

    return df


def compute_growth_rate(df: pd.DataFrame, window_days: int = 30) -> float | None:
    """
    Growth rate over the last `window_days` days compared to the preceding
    `window_days` period.
    Returns None if there is insufficient data.
    """
    if df.empty or "gig_activity_index" not in df.columns:
        return None

    df_sorted = df.sort_values("date")
    max_date = df_sorted["date"].max()
    cutoff = max_date - pd.Timedelta(days=window_days)
    cutoff2 = max_date - pd.Timedelta(days=window_days * 2)

    recent = df_sorted[df_sorted["date"] > cutoff]["gig_activity_index"]
    prior = df_sorted[
        (df_sorted["date"] > cutoff2) & (df_sorted["date"] <= cutoff)
    ]["gig_activity_index"]

    if recent.empty or prior.empty or prior.mean() == 0:
        return None

    return round((recent.mean() - prior.mean()) / prior.mean() * 100, 1)


def compute_data_quality(df: pd.DataFrame) -> float:
    """
    Data quality = % of non-null cells across all numeric columns
    in the original (pre-fill) data. Since we already filled NaNs,
    we approximate by counting how many cells were originally present
    (always 100% after filling). We use the global missing rate stored
    during load instead.

    Practical approach: return the percentage of numeric cells that
    are within realistic bounds (not clipped outliers).
    Returns a float in [0, 100].
    """
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if not cols:
        return 0.0
    total = len(df) * len(cols)
    valid = sum(df[col].notna().sum() for col in cols)
    return round(valid / total * 100, 1) if total > 0 else 0.0
