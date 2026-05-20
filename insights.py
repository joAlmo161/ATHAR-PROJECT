"""
insights.py
Generates data-driven storytelling text shown below each chart.
All functions accept the preprocessed, filtered DataFrame and return
a markdown string ready for st.markdown().
Updates automatically whenever filters change.
"""

import pandas as pd


def line_chart_insight(df: pd.DataFrame) -> str:
    """
    Insight for the line chart:
    - Which region leads overall
    - When activity peaked
    - Whether the trend is up or down over the period
    """
    if df.empty or "gig_activity_index" not in df.columns:
        return "_No data available to generate insights._"

    # Overall leader region
    region_avg = df.groupby("region")["gig_activity_index"].mean()
    top_region = region_avg.idxmax()
    top_val = region_avg.max()

    # Peak date (by mean across all regions on that day)
    daily = df.groupby("date")["gig_activity_index"].mean()
    peak_date = daily.idxmax()
    peak_val = daily.max()

    # Trend: compare first vs last third of the time window
    dates_sorted = df.sort_values("date")["date"]
    n = len(dates_sorted)
    third = max(n // 3, 1)
    first_avg = df.loc[df["date"].isin(dates_sorted.iloc[:third]), "gig_activity_index"].mean()
    last_avg = df.loc[df["date"].isin(dates_sorted.iloc[-third:]), "gig_activity_index"].mean()

    if last_avg > first_avg * 1.02:
        trend_text = "📈 Activity is **trending upward** over the selected period."
    elif last_avg < first_avg * 0.98:
        trend_text = "📉 Activity is **trending downward** over the selected period."
    else:
        trend_text = "➡️ Activity is **relatively stable** over the selected period."

    return (
        f"**{top_region}** leads with an average index of **{top_val:.1f}**. "
        f"Peak activity of **{peak_val:.1f}** was recorded on "
        f"**{peak_date.strftime('%B %d, %Y')}**. {trend_text}"
    )


def bar_chart_insight(df: pd.DataFrame) -> str:
    """
    Insight for the regional bar chart:
    - Gap between top and bottom regions
    - Which signal drives the top region
    """
    if df.empty or "gig_activity_index" not in df.columns:
        return "_No data available to generate insights._"

    region_avg = df.groupby("region")["gig_activity_index"].mean().sort_values(ascending=False)

    if len(region_avg) < 2:
        top = region_avg.index[0]
        top_val = region_avg.iloc[0]
        return f"**{top}** is the only region in the current view with an index of **{top_val:.1f}**."

    top_region = region_avg.index[0]
    top_val = region_avg.iloc[0]
    bot_region = region_avg.index[-1]
    bot_val = region_avg.iloc[-1]
    gap_pct = round((top_val - bot_val) / bot_val * 100, 1) if bot_val > 0 else 0

    return (
        f"**{top_region}** ({top_val:.1f}) leads, outperforming "
        f"**{bot_region}** ({bot_val:.1f}) by **{gap_pct}%**. "
        f"Larger cities tend to show higher gig activity due to greater "
        f"population density and digital adoption."
    )


def donut_chart_insight(df: pd.DataFrame) -> str:
    """
    Insight for the donut chart:
    - Which signal contributes most / least
    """
    from preprocessor import WEIGHTS

    if df.empty:
        return "_No data available to generate insights._"

    contributions = {}
    for col, weight in WEIGHTS.items():
        norm_col = f"{col}_norm"
        if norm_col in df.columns:
            contributions[col] = df[norm_col].mean() * weight
        elif col in df.columns:
            contributions[col] = df[col].mean() * weight

    contributions = {k: v for k, v in contributions.items() if v and v > 0}
    if not contributions:
        return "_No signal data available._"

    top_signal = max(contributions, key=contributions.get)
    low_signal = min(contributions, key=contributions.get)
    total = sum(contributions.values())
    top_pct = round(contributions[top_signal] / total * 100, 1)
    low_pct = round(contributions[low_signal] / total * 100, 1)

    return (
        f"**{top_signal.replace('_', ' ').title()}** is the strongest driver, "
        f"accounting for **{top_pct}%** of the index weight. "
        f"**{low_signal.replace('_', ' ').title()}** contributes the least "
        f"at **{low_pct}%**."
    )


def overall_summary(df: pd.DataFrame, growth_rate: float | None) -> str:
    """
    A short headline summary shown in the KPI section.
    """
    if df.empty:
        return ""

    avg_gai = df["gig_activity_index"].mean() if "gig_activity_index" in df.columns else 0

    if avg_gai >= 70:
        level = "🟢 **High** gig activity"
    elif avg_gai >= 45:
        level = "🟡 **Moderate** gig activity"
    else:
        level = "🔴 **Low** gig activity"

    growth_str = ""
    if growth_rate is not None:
        arrow = "↑" if growth_rate >= 0 else "↓"
        growth_str = f" | 30-day change: **{arrow} {abs(growth_rate):.1f}%**"

    return f"{level} detected in the selected view (avg. index: **{avg_gai:.1f}**){growth_str}."
