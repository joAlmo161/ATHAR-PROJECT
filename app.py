"""
app.py  —  ATHAR: Digital Observatory for Gig Economy Activity
============================================================
Entry point for the Streamlit application.

Run with:
    streamlit run streamlit_app/app.py          (from project root)
  or
    streamlit run app.py                         (from inside streamlit_app/)

Structure:
  Section 1  — Landing header
  Section 2  — Data source (upload or default)
  Section 3  — Sidebar filters
  Section 4  — KPI cards
  Section 5  — Line chart (time series)
  Section 6  — Bar chart + Donut chart (side by side)
  Section 7  — Raw data preview + download
"""

import sys
import os

# ── Make sure local modules are importable when run from project root ─────────
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import re

from data_loader import load_default, load_uploaded, NUMERIC_COLS
from preprocessor import preprocess, compute_growth_rate, compute_data_quality
from charts import make_line_chart, make_bar_chart, make_donut_chart
from insights import line_chart_insight, bar_chart_insight, donut_chart_insight, overall_summary


def format_insight(text: str, title: str = "Insight") -> str:
    """Convert simple markdown-style insight text into a clean HTML card."""
    text = str(text).replace("💡", "").strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return f'<div class="insight-box"><span class="insight-title">{title}</span><span class="insight-text">{text}</span></div>'

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATHAR — Gig Economy Observatory",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Landing header gradient */
.athar-header {
    background: linear-gradient(135deg, #0d2e4e 0%, #1a6faf 60%, #2ea8a0 100%);
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.athar-header h1 { color: #ffffff; margin: 0; font-size: 2.8rem; letter-spacing: 2px; }
.athar-header .subtitle { color: #a8d8f0; margin: 0.4rem 0 0; font-size: 1.15rem; }
.athar-header .description { color: #c8e8f8; margin: 0.6rem 0 0; font-size: 0.92rem; line-height: 1.6; }
/* KPI card metric styling */
div[data-testid="metric-container"] {
    background: #f7fafd;
    border: 1px solid #d0e4f5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
/* Insight text */
.insight-box {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-left: 4px solid #2ea8a0;
    padding: 1rem 1.1rem;
    border-radius: 14px;
    margin-top: 0.75rem;
    margin-bottom: 1.15rem;
    color: #f8fafc;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}
.insight-box b {
    color: #ffffff;
    font-weight: 700;
}
.insight-title {
    display: block;
    color: #7dd3fc;
    font-weight: 700;
    margin-bottom: 0.35rem;
    font-size: 0.95rem;
}
.insight-text {
    color: #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Landing header
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="athar-header">
    <h1>📡 ATHAR &nbsp;|&nbsp; أثر</h1>
    <p class="subtitle">Digital Observatory for Gig Economy Activity in Saudi Arabia</p>
    <p class="description">
        ATHAR estimates gig economy activity using <strong>indirect digital signals</strong>
        — search trends, delivery demand, ride requests, and freelance activity —
        instead of relying solely on official data sources.
        Explore the dashboard below to uncover regional trends and activity patterns
        across five major Saudi cities.
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Data source selection
# ════════════════════════════════════════════════════════════════════════════════

with st.expander("📂 Upload your own CSV dataset (optional)", expanded=False):
    st.markdown(
        "Upload a CSV with the following columns: "
        "`date`, `region`, `search_interest`, `delivery_orders`, "
        "`ride_requests`, `freelance_jobs`, `active_users`, `urban_activity`."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="If uploaded, this replaces the default dataset.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # reset pointer after preview read
            st.success(f"✅ File accepted: **{uploaded_file.name}** — {len(preview_df):,} rows, {len(preview_df.columns)} columns")
            st.subheader("Preview (first 10 rows)")
            st.dataframe(preview_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            uploaded_file = None

# Load data ───────────────────────────────────────────────────────────────────
source_label = "Default dataset"
raw_df = None

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        raw_df = load_uploaded(uploaded_file)
        source_label = f"Uploaded: {uploaded_file.name}"
    except ValueError as e:
        st.error(f"❌ Upload error: {e}")
        st.info("Falling back to the default dataset.")
    except Exception as e:
        st.error(f"❌ Unexpected error loading upload: {e}")

if raw_df is None:
    try:
        raw_df = load_default()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Sidebar filters
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Filters")
    st.caption(f"📄 {source_label} · {len(raw_df):,} rows")
    st.divider()

    # Region filter
    all_regions = sorted(raw_df["region"].dropna().unique().tolist())
    selected_regions = st.multiselect(
        "🗺️ Region",
        options=all_regions,
        default=all_regions,
        help="Select one or more regions to display.",
    )

    # Date range filter
    min_date = raw_df["date"].min().date()
    max_date = raw_df["date"].max().date()

    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.divider()
    st.markdown(
        "**Gig Activity Index (GAI)** is a weighted composite of:\n"
        "- Search interest\n"
        "- Delivery orders\n"
        "- Ride requests\n"
        "- Freelance jobs\n"
        "- Active users\n"
        "- Urban activity"
    )
    st.divider()
    st.caption("ATHAR v2.0 · University Project · 2026")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = raw_df.copy()

if selected_regions:
    filtered = filtered[filtered["region"].isin(selected_regions)]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered = filtered[
        (filtered["date"].dt.date >= start_d) &
        (filtered["date"].dt.date <= end_d)
    ]

if filtered.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust the region or date range.")
    st.stop()

# Preprocess the filtered slice ───────────────────────────────────────────────
df = preprocess(filtered)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — KPI cards
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Key Performance Indicators")

avg_gai = round(df["gig_activity_index"].mean(), 1)
growth_rate = compute_growth_rate(df)
before_quality = compute_data_quality(filtered)
after_quality = compute_data_quality(df)
record_count = len(df)

# Show overall summary line
st.markdown(
    format_insight(overall_summary(df, growth_rate), "Current View Summary"),
    unsafe_allow_html=True,
)
st.markdown("")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="🎯 Gig Activity Index",
        value=f"{avg_gai} / 100",
        help="Weighted composite of all 6 digital signals, normalized to 0–100.",
    )

with kpi2:
    if growth_rate is not None:
        delta_str = f"{growth_rate:+.1f}%"
        st.metric(
            label="📈 30-Day Growth Rate",
            value=delta_str,
            delta=delta_str,
            delta_color="normal",
            help="Compares average GAI in the last 30 days vs. the 30 days before that.",
        )
    else:
        st.metric(
            label="📈 30-Day Growth Rate",
            value="N/A",
            help="Not enough data to compute growth rate.",
        )

with kpi3:
    st.metric(
        label="⚠️ Quality Before Cleaning",
        value=f"{before_quality:.1f}%",
        help="Data quality before preprocessing and cleaning.",
    )

with kpi4:
    st.metric(
        label="✅ Quality After Cleaning",
        value=f"{after_quality:.1f}%",
        help="Data quality after preprocessing and handling missing values.",
    )

with kpi5:
    st.metric(
        label="📋 Records",
        value=f"{record_count:,}",
        help="Number of rows in the current filtered view.",
    )
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Line chart: GAI over time
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("📈 Gig Activity Index Over Time")
st.plotly_chart(make_line_chart(df), use_container_width=True)
st.markdown(
    format_insight(line_chart_insight(df), "Trend Insight"),
    unsafe_allow_html=True,
)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Bar chart + Donut chart
# ════════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("🗺️ Regional Comparison")
    st.plotly_chart(make_bar_chart(df), use_container_width=True)
    st.markdown(
        format_insight(bar_chart_insight(df), "Regional Insight"),
        unsafe_allow_html=True,
    )

with col_right:
    st.subheader("🔢 Signal Contribution")
    st.plotly_chart(make_donut_chart(df), use_container_width=True)
    st.markdown(
        format_insight(donut_chart_insight(df), "Signal Insight"),
        unsafe_allow_html=True,
    )

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Raw data preview and download
# ════════════════════════════════════════════════════════════════════════════════

with st.expander("🔍 View & Download Data", expanded=False):
    display_cols = (
        ["date", "region"]
        + [c for c in NUMERIC_COLS if c in df.columns]
        + ["gig_activity_index"]
    )
    preview = df[display_cols].sort_values("date", ascending=False)

    st.dataframe(preview.head(500), use_container_width=True)
    st.caption(f"Showing up to 500 of {len(df):,} records in the current filter.")

    csv_bytes = preview.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="athar_filtered_data.csv",
        mime="text/csv",
    )
