"""
data_loader.py
Handles loading the CSV dataset from either:
  - The default bundled file (../athar_gig_economy_dataset.csv)
  - An uploaded file object (from st.file_uploader)

Uses @st.cache_data so re-renders from filter interactions never re-read disk.
"""

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Expected columns ──────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "search_interest",
    "delivery_orders",
    "ride_requests",
    "freelance_jobs",
    "active_users",
    "urban_activity",
]
EXPECTED_COLS = {"date", "region"} | set(NUMERIC_COLS)

# Default dataset path: one level above this file
DEFAULT_CSV = Path(__file__).parent / "athar_gig_economy_dataset.csv"


@st.cache_data(show_spinner="Loading dataset…")
def load_default() -> pd.DataFrame:
    """Load the bundled default CSV dataset."""
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(
            f"Default dataset not found at {DEFAULT_CSV}. "
            "Please upload a CSV file."
        )
    return _parse(pd.read_csv(DEFAULT_CSV))


def load_uploaded(file_obj) -> pd.DataFrame:
    """
    Load an uploaded CSV file (BytesIO from st.file_uploader).
    Raises ValueError with a clear message if required columns are missing.
    """
    raw = pd.read_csv(file_obj)
    # Normalize column names to lower-snake-case
    raw.columns = raw.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    missing = EXPECTED_COLS - set(raw.columns)
    if missing:
        raise ValueError(
            f"Uploaded file is missing required columns: {sorted(missing)}\n"
            f"Found columns: {sorted(raw.columns.tolist())}"
        )
    return _parse(raw)


def _parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common parsing step applied to any loaded DataFrame:
    - Normalize column names
    - Parse date column
    - Strip region whitespace
    - Drop rows with unparseable dates
    """
    df = df.copy()
    # Normalize names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Parse dates; rows where date can't be parsed are dropped
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        st.toast(f"⚠️ Dropped {n_dropped} rows with unparseable dates.", icon="⚠️")

    # Clean region
    df["region"] = df["region"].astype(str).str.strip().str.title()

    # Cast numeric columns to float (coerce bad strings to NaN)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)
