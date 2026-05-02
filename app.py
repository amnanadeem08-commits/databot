"""
Interactive Streamlit dashboard for customer analytics.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def load_dashboard_data(cleaned_path: Path) -> pd.DataFrame:
    """
    Load cleaned customer data for dashboard usage.

    Why: Centralized loading keeps parsing and errors consistent.
    """
    try:
        df = pd.read_csv(cleaned_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Cleaned file is empty: {cleaned_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Unable to parse cleaned file: {cleaned_path}") from exc

    if "Subscription Date" in df.columns:
        df["Subscription Date"] = pd.to_datetime(df["Subscription Date"], errors="coerce")

    return df


def infer_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer categorical fields suitable for sidebar filters.

    Why: Auto-detection keeps the dashboard flexible for schema changes.
    """
    exclude_cols = {"Customer Id", "full_name", "subscription_date_raw", "Email", "Website"}
    candidates = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            candidates.append(col)
    return candidates


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply sidebar date and categorical filters to return current and prior slices.

    Why: Parallel current/prior slices are required for KPI delta calculations.
    """
    st.sidebar.header("Filters")

    filtered = df.copy()
    date_col = "Subscription Date"

    # WHAT: Build date range controls from valid subscription dates.
    # WHY: Time windows are core to trend and delta interpretation.
    if date_col in filtered.columns and filtered[date_col].notna().any():
        min_date = filtered[date_col].min().date()
        max_date = filtered[date_col].max().date()
        selected_range = st.sidebar.date_input(
            "Subscription Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date, end_date = min_date, max_date

        filtered = filtered[
            (filtered[date_col].dt.date >= start_date) & (filtered[date_col].dt.date <= end_date)
        ]

        days_span = max((end_date - start_date).days + 1, 1)
        prior_end = pd.to_datetime(start_date) - pd.Timedelta(days=1)
        prior_start = prior_end - pd.Timedelta(days=days_span - 1)
        prior_period_df = df[
            (df[date_col] >= prior_start) & (df[date_col] <= prior_end)
        ].copy()
    else:
        prior_period_df = pd.DataFrame(columns=df.columns)

    categorical_cols = infer_categorical_columns(filtered if not filtered.empty else df)
    for col in categorical_cols:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        if not options:
            continue
        selected = st.sidebar.multiselect(f"{col}", options=options, default=options)
        filtered = filtered[filtered[col].astype(str).isin(selected)]
        if not prior_period_df.empty:
            prior_period_df = prior_period_df[prior_period_df[col].astype(str).isin(selected)]

    return filtered, prior_period_df


def calculate_kpis(current_df: pd.DataFrame, prior_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate KPI values and deltas for st.metric cards.

    Why: KPI cards are executive-friendly and highlight movement over time.
    """

    def safe_delta(current: float, prior: float) -> float:
        if prior == 0:
            return float("nan")
        return (current - prior) / prior

    cur_total = float(current_df["Customer Id"].nunique()) if "Customer Id" in current_df else float(current_df.shape[0])
    prv_total = float(prior_df["Customer Id"].nunique()) if "Customer Id" in prior_df else float(prior_df.shape[0])

    cur_new = float(current_df.shape[0])
    prv_new = float(prior_df.shape[0])

    cur_contact = float(
        (
            current_df.get("email_valid_flag", pd.Series([False] * len(current_df))).fillna(False).astype(bool)
            & current_df.get("phone_present_flag", pd.Series([False] * len(current_df))).fillna(False).astype(bool)
        ).mean()
    ) if not current_df.empty else 0.0
    prv_contact = float(
        (
            prior_df.get("email_valid_flag", pd.Series([False] * len(prior_df))).fillna(False).astype(bool)
            & prior_df.get("phone_present_flag", pd.Series([False] * len(prior_df))).fillna(False).astype(bool)
        ).mean()
    ) if not prior_df.empty else 0.0

    cur_geo = 0.0
    prv_geo = 0.0
    if "Country" in current_df and not current_df.empty:
        cur_geo = float(current_df["Country"].value_counts(normalize=True).head(5).sum())
    if "Country" in prior_df and not prior_df.empty:
        prv_geo = float(prior_df["Country"].value_counts(normalize=True).head(5).sum())

    return {
        "Unique Customers": {"value": cur_total, "delta": safe_delta(cur_total, prv_total)},
        "Subscriptions (Selected Period)": {"value": cur_new, "delta": safe_delta(cur_new, prv_new)},
        "Contactability Rate": {"value": cur_contact, "delta": safe_delta(cur_contact, prv_contact)},
        "Top 5 Country Concentration": {"value": cur_geo, "delta": safe_delta(cur_geo, prv_geo)},
    }


def render_kpi_row(kpis: Dict[str, Dict[str, float]]) -> None:
    """
    Render KPI cards with value and delta indicators.

    Why: Compact metric summaries reduce cognitive load for busy stakeholders.
    """
    cols = st.columns(4)
    for idx, (kpi_name, payload) in enumerate(kpis.items()):
        value = payload["value"]
        delta = payload["delta"]
        if "Rate" in kpi_name or "Concentration" in kpi_name:
            value_str = f"{value * 100:.1f}%"
            delta_str = f"{delta * 100:.1f}%" if not np.isnan(delta) else "N/A"
        else:
            value_str = f"{int(value):,}"
            delta_str = f"{delta * 100:.1f}%" if not np.isnan(delta) else "N/A"
        cols[idx].metric(label=kpi_name, value=value_str, delta=delta_str)


def build_monthly_subscription_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly subscription counts for trend and decline diagnostics.

    Why: Month-level granularity is the clearest lens for acquisition shifts.
    """
    if "Subscription Date" not in df.columns:
        return pd.DataFrame(columns=["year_month", "subscriptions"])

    # WHAT: Convert dates to month buckets and count records.
    # WHY: We need comparable month-over-month totals for decline quantification.
    trend_df = (
        df.dropna(subset=["Subscription Date"])
        .assign(year_month=lambda d: d["Subscription Date"].dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False)
        .size()
        .rename(columns={"size": "subscriptions"})
        .sort_values("year_month")
    )
    return trend_df


def quantify_mom_decline(trend_df: pd.DataFrame) -> Dict[str, float]:
    """
    Quantify month-over-month movement using the last two available months.

    Why: Explicit quantification makes trend commentary decision-ready.
    """
    if trend_df.shape[0] < 2:
        return {
            "current_month_subscriptions": 0.0,
            "prior_month_subscriptions": 0.0,
            "absolute_change": 0.0,
            "percent_change": float("nan"),
        }

    latest = trend_df.iloc[-1]
    prior = trend_df.iloc[-2]
    current_val = float(latest["subscriptions"])
    prior_val = float(prior["subscriptions"])
    absolute_change = current_val - prior_val
    percent_change = ((current_val - prior_val) / prior_val) if prior_val > 0 else float("nan")

    return {
        "current_month_subscriptions": current_val,
        "prior_month_subscriptions": prior_val,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
    }


def build_mom_breakdown(df: pd.DataFrame, breakdown_col: str) -> pd.DataFrame:
    """
    Break down latest month-over-month change by a chosen segment dimension.

    Why: Segment-level decomposition reveals where decline is concentrated.
    """
    if "Subscription Date" not in df.columns or breakdown_col not in df.columns:
        return pd.DataFrame(columns=[breakdown_col, "current_month", "prior_month", "abs_change", "pct_change"])

    base = df.dropna(subset=["Subscription Date"]).copy()
    if base.empty:
        return pd.DataFrame(columns=[breakdown_col, "current_month", "prior_month", "abs_change", "pct_change"])

    base["month_period"] = base["Subscription Date"].dt.to_period("M")
    months = sorted(base["month_period"].unique())
    if len(months) < 2:
        return pd.DataFrame(columns=[breakdown_col, "current_month", "prior_month", "abs_change", "pct_change"])

    current_month = months[-1]
    prior_month = months[-2]

    current_counts = (
        base.loc[base["month_period"] == current_month, breakdown_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename("current_month")
    )
    prior_counts = (
        base.loc[base["month_period"] == prior_month, breakdown_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename("prior_month")
    )

    combined = pd.concat([current_counts, prior_counts], axis=1).fillna(0)
    combined["abs_change"] = combined["current_month"] - combined["prior_month"]
    combined["pct_change"] = np.where(
        combined["prior_month"] > 0,
        combined["abs_change"] / combined["prior_month"],
        np.nan,
    )
    return combined.reset_index().rename(columns={"index": breakdown_col}).sort_values("abs_change")


def render_decline_insights(df: pd.DataFrame) -> None:
    """
    Render month-over-month decline quantification and decomposition visuals.

    Why: This section directly answers where and why acquisition slowed.
    """
    st.subheader("Month-over-Month Subscription Decline")
    trend_df = build_monthly_subscription_trend(df=df)
    if trend_df.empty:
        st.info("No valid monthly trend available for decline analysis.")
        return

    decline = quantify_mom_decline(trend_df=trend_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Month Subscriptions", f"{int(decline['current_month_subscriptions']):,}")
    c2.metric(
        "MoM Absolute Change",
        f"{int(decline['absolute_change']):,}",
        delta=f"{decline['percent_change'] * 100:.1f}%"
        if not np.isnan(decline["percent_change"])
        else "N/A",
    )
    c3.metric("Prior Month Subscriptions", f"{int(decline['prior_month_subscriptions']):,}")

    trend_chart = px.line(
        trend_df,
        x="year_month",
        y="subscriptions",
        title="Monthly Subscription Trend (MoM Context)",
        markers=True,
        color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(trend_chart, use_container_width=True)

    st.markdown("**Breakdown of latest MoM change by Country and Company segment**")
    country_breakdown = build_mom_breakdown(df=df, breakdown_col="Country")
    company_breakdown = build_mom_breakdown(df=df, breakdown_col="Company")

    b1, b2 = st.columns(2)
    if not country_breakdown.empty:
        country_decline = country_breakdown.head(10)
        fig_country_decline = px.bar(
            country_decline,
            x="Country",
            y="abs_change",
            title="Largest Country-Level Declines (Top 10)",
            color="abs_change",
            color_continuous_scale="Blues",
            hover_data=["current_month", "prior_month", "pct_change"],
        )
        b1.plotly_chart(fig_country_decline, use_container_width=True)
        b1.dataframe(country_decline, use_container_width=True, hide_index=True)
    else:
        b1.info("Country-level MoM breakdown unavailable.")

    if not company_breakdown.empty:
        company_decline = company_breakdown.head(10)
        fig_company_decline = px.bar(
            company_decline,
            x="Company",
            y="abs_change",
            title="Largest Segment-Level Declines (Top 10 Companies)",
            color="abs_change",
            color_continuous_scale="Blues",
            hover_data=["current_month", "prior_month", "pct_change"],
        )
        b2.plotly_chart(fig_company_decline, use_container_width=True)
        b2.dataframe(company_decline, use_container_width=True, hide_index=True)
    else:
        b2.info("Company-level MoM breakdown unavailable.")


def render_charts(df: pd.DataFrame) -> None:
    """
    Render interactive Plotly charts aligned to KPI questions.

    Why: Interactivity helps users move from headline KPI to root-cause exploration.
    """
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # WHAT: Build monthly subscription trend.
    # WHY: Time trend is the main signal for acquisition momentum.
    trend_df = build_monthly_subscription_trend(df=df)
    fig_line = px.line(
        trend_df,
        x="year_month",
        y="subscriptions",
        title="Subscription Trend Over Time",
        markers=True,
        color_discrete_sequence=["#1f77b4"],
    )

    country_df = (
        df["Country"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .rename_axis("Country")
        .reset_index(name="customers")
    )
    fig_country = px.bar(
        country_df,
        x="Country",
        y="customers",
        title="Top 10 Countries by Customers",
        color="Country",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    company_df = (
        df["Company"]
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .rename_axis("Company")
        .reset_index(name="customers")
    )
    fig_company = px.bar(
        company_df,
        x="Company",
        y="customers",
        title="Top 15 Companies by Customers",
        color="customers",
        color_continuous_scale="Blues",
    )

    scatter_df = (
        df.groupby(["Country", "Company"], as_index=False)
        .size()
        .rename(columns={"size": "customer_count"})
        .sort_values("customer_count", ascending=False)
        .head(200)
    )
    fig_scatter = px.scatter(
        scatter_df,
        x="Country",
        y="Company",
        size="customer_count",
        color="customer_count",
        title="Country-Company Concentration (Top Pairs)",
        color_continuous_scale="Viridis",
        hover_data=["customer_count"],
    )

    heatmap_df = (
        df.dropna(subset=["Subscription Date"])
        .assign(month=lambda d: d["Subscription Date"].dt.month, country=lambda d: d["Country"].fillna("Unknown"))
        .groupby(["country", "month"], as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )
    fig_heatmap = px.density_heatmap(
        heatmap_df,
        x="month",
        y="country",
        z="customers",
        title="Country vs Month Subscription Heatmap",
        color_continuous_scale="Cividis",
    )

    top_row_col1, top_row_col2 = st.columns(2)
    top_row_col1.plotly_chart(fig_line, use_container_width=True)
    top_row_col2.plotly_chart(fig_country, use_container_width=True)

    mid_row_col1, mid_row_col2 = st.columns(2)
    mid_row_col1.plotly_chart(fig_company, use_container_width=True)
    mid_row_col2.plotly_chart(fig_heatmap, use_container_width=True)

    st.plotly_chart(fig_scatter, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    """
    Render detailed table and CSV download action.

    Why: Operational users often need row-level drill-down and export.
    """
    st.subheader("Customer Detail Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_customers.csv",
        mime="text/csv",
    )


def render_footer(df: pd.DataFrame) -> None:
    """
    Render data freshness and row-count status footer.

    Why: Users should always know how current and scoped the view is.
    """
    latest_date = (
        df["Subscription Date"].max().strftime("%Y-%m-%d")
        if "Subscription Date" in df.columns and df["Subscription Date"].notna().any()
        else "N/A"
    )
    st.caption(
        f"Data freshness (max subscription date): {latest_date} | "
        f"Rows in current view: {len(df):,} | Rendered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def main() -> None:
    """
    Build and run the Streamlit dashboard.

    Why: Main entrypoint orchestrates all dashboard components in one flow.
    """
    st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
    st.title("Customer Analytics Dashboard")

    base_dir = Path(__file__).resolve().parent
    cleaned_path = base_dir / "data" / "cleaned" / "customers_cleaned.csv"

    try:
        df = load_dashboard_data(cleaned_path=cleaned_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Dashboard failed to load data: {exc}")
        st.stop()

    filtered_df, prior_df = apply_filters(df=df)
    kpis = calculate_kpis(current_df=filtered_df, prior_df=prior_df)

    with st.container():
        render_kpi_row(kpis=kpis)

    with st.container():
        render_decline_insights(df=filtered_df)

    with st.container():
        render_charts(df=filtered_df)

    with st.container():
        render_data_table(df=filtered_df)

    render_footer(df=filtered_df)


if __name__ == "__main__":
    main()
