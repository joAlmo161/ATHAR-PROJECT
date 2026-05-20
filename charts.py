"""
charts.py
All Plotly figure builders for the ATHAR dashboard.
Each function is pure: it takes a preprocessed DataFrame and returns a go.Figure.
Charts update automatically when filters change because Streamlit re-calls them.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from preprocessor import WEIGHTS

# Consistent colour palette across charts
REGION_COLORS = {
    "Riyadh":  "#1a6faf",
    "Jeddah":  "#2ea8a0",
    "Dammam":  "#e07b2a",
    "Makkah":  "#9b59b6",
    "Madinah": "#27ae60",
}
DEFAULT_COLOR = "#1a6faf"
CHART_TEMPLATE = "plotly_white"


# ── 1. Line chart — GAI over time ─────────────────────────────────────────────

def make_line_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing Gig Activity Index over time.
    Each region gets its own trace when multiple regions are present.
    """
    if df.empty:
        return _empty_figure("No data available for the selected filters.")

    # Aggregate: mean GAI per date × region
    agg = (
        df.groupby(["date", "region"], as_index=False)["gig_activity_index"]
        .mean()
        .rename(columns={"gig_activity_index": "GAI"})
    )

    regions = agg["region"].unique()

    # Assign colours — fall back to a Plotly sequence for unknown regions
    color_map = {r: REGION_COLORS.get(r, DEFAULT_COLOR) for r in regions}

    fig = px.line(
        agg,
        x="date",
        y="GAI",
        color="region",
        color_discrete_map=color_map,
        markers=True,
        title="Gig Activity Index Over Time",
        labels={"GAI": "Gig Activity Index (0–100)", "date": "Date", "region": "Region"},
        template=CHART_TEMPLATE,
    )

    # Mark the global peak
    peak_row = agg.loc[agg["GAI"].idxmax()]
    fig.add_annotation(
        x=peak_row["date"],
        y=peak_row["GAI"],
        text=f"Peak: {peak_row['GAI']:.1f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-36,
        font=dict(size=11, color="#c0392b"),
        arrowcolor="#c0392b",
    )

    fig.update_layout(
        height=400,
        legend_title_text="Region",
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    return fig


# ── 2. Bar chart — regional comparison ────────────────────────────────────────

def make_bar_chart(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart comparing average GAI across regions.
    Bars are sorted from lowest to highest so the leader is at the top.
    """
    if df.empty:
        return _empty_figure("No data available.")

    regional = (
        df.groupby("region", as_index=False)["gig_activity_index"]
        .mean()
        .rename(columns={"gig_activity_index": "GAI"})
        .sort_values("GAI", ascending=True)
    )

    colors = [REGION_COLORS.get(r, DEFAULT_COLOR) for r in regional["region"]]

    fig = go.Figure(
        go.Bar(
            x=regional["GAI"],
            y=regional["region"],
            orientation="h",
            marker_color=colors,
            text=regional["GAI"].round(1).astype(str),
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Average Gig Activity Index by Region",
        xaxis=dict(title="Gig Activity Index (0–100)", range=[0, 110]),
        yaxis=dict(title=""),
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(t=50, b=30),
    )
    return fig


# ── 3. Donut chart — signal contribution ──────────────────────────────────────

def make_donut_chart(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart showing the weighted contribution of each signal to the GAI
    in the current filtered view.
    Components with a zero mean in this view are excluded.
    """
    from data_loader import NUMERIC_COLS

    if df.empty:
        return _empty_figure("No data available.")

    # Compute the actual weighted contribution for each signal
    contributions = {}
    for col, weight in WEIGHTS.items():
        if col in df.columns:
            norm_col = f"{col}_norm"
            if norm_col in df.columns:
                mean_val = df[norm_col].mean()
            else:
                mean_val = df[col].mean()
            contributions[col] = mean_val * weight

    # Remove zero or NaN contributions
    contributions = {k: v for k, v in contributions.items() if v and v > 0}

    if not contributions:
        return _empty_figure("No signal data available for this view.")

    total = sum(contributions.values())
    labels = [k.replace("_", " ").title() for k in contributions]
    values = [round(v / total * 100, 1) for v in contributions.values()]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            marker=dict(
                colors=[
                    "#1a6faf", "#2ea8a0", "#e07b2a",
                    "#9b59b6", "#27ae60", "#e74c3c"
                ]
            ),
        )
    )
    fig.update_layout(
        title="Signal Contribution to Gig Activity Index",
        template=CHART_TEMPLATE,
        height=370,
        legend=dict(orientation="v", x=1.0, y=0.5),
        margin=dict(t=50, b=10),
        annotations=[
            dict(text="GAI<br>Signals", x=0.5, y=0.5, font_size=13, showarrow=False)
        ],
    )
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
