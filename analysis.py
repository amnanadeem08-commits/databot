"""
Exploratory analysis and KPI computation for cleaned customer data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_clean_data(cleaned_path: Path) -> pd.DataFrame:
    """
    Load cleaned customer dataset and parse date columns.

    Why: Analysis depends on date-aware metrics and consistent schema.
    """
    try:
        df = pd.read_csv(cleaned_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Cleaned data file is empty: {cleaned_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse cleaned data CSV: {cleaned_path}") from exc

    if "Subscription Date" in df.columns:
        df["Subscription Date"] = pd.to_datetime(df["Subscription Date"], errors="coerce")

    return df


def build_eda_outputs(df: pd.DataFrame) -> Dict[str, object]:
    """
    Create EDA snapshots: descriptive stats, category distributions, and correlations.

    Why: EDA surfaces baseline patterns before KPI interpretation.
    """
    # WHAT: Run describe across both numeric and object fields.
    # WHY: Business users need both count ranges and category coverage.
    describe_output = df.describe(include="all").fillna("").to_dict()

    value_counts_output: Dict[str, Dict[str, int]] = {}
    for col in ["Country", "City", "Company", "subscription_year"]:
        if col in df.columns:
            top_counts = df[col].fillna("Unknown").astype(str).value_counts().head(20)
            value_counts_output[col] = {str(k): int(v) for k, v in top_counts.items()}

    corr_df = df.select_dtypes(include=["number"]).copy()
    correlation_output: Dict[str, Dict[str, float]] = {}
    if not corr_df.empty and corr_df.shape[1] >= 2:
        correlation_output = corr_df.corr(numeric_only=True).fillna(0.0).round(4).to_dict()

    return {
        "describe": describe_output,
        "value_counts_top20": value_counts_output,
        "correlation_matrix": correlation_output,
    }


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute customer-domain KPIs used in analysis and dashboard cards.

    Why: KPIs translate raw data into decision metrics for stakeholders.
    """
    total_customers = int(df["Customer Id"].nunique()) if "Customer Id" in df.columns else int(df.shape[0])
    total_records = int(df.shape[0])

    valid_contactability_rate = float(
        (
            df.get("email_valid_flag", pd.Series([False] * len(df))).fillna(False).astype(bool)
            & df.get("phone_present_flag", pd.Series([False] * len(df))).fillna(False).astype(bool)
        ).mean()
    )

    top_country_share = 0.0
    top_5_country_share = 0.0
    if "Country" in df.columns and total_records > 0:
        country_dist = df["Country"].fillna("Unknown").value_counts(normalize=True)
        top_country_share = float(country_dist.head(1).sum())
        top_5_country_share = float(country_dist.head(5).sum())

    top_company_share = 0.0
    top_10_company_share = 0.0
    if "Company" in df.columns and total_records > 0:
        company_dist = df["Company"].fillna("Unknown").value_counts(normalize=True)
        top_company_share = float(company_dist.head(1).sum())
        top_10_company_share = float(company_dist.head(10).sum())

    recent_subscriptions = 0
    growth_vs_prior_period = 0.0
    if "Subscription Date" in df.columns:
        valid_dates = df.dropna(subset=["Subscription Date"]).copy()
        if not valid_dates.empty:
            latest_month = valid_dates["Subscription Date"].max().to_period("M")
            prior_month = latest_month - 1
            current_count = int((valid_dates["Subscription Date"].dt.to_period("M") == latest_month).sum())
            prior_count = int((valid_dates["Subscription Date"].dt.to_period("M") == prior_month).sum())
            recent_subscriptions = current_count
            growth_vs_prior_period = float(
                ((current_count - prior_count) / prior_count) if prior_count > 0 else np.nan
            )

    return {
        "total_records": float(total_records),
        "total_unique_customers": float(total_customers),
        "recent_month_subscriptions": float(recent_subscriptions),
        "growth_vs_prior_month": float(growth_vs_prior_period)
        if not np.isnan(growth_vs_prior_period)
        else float("nan"),
        "top_country_share": float(top_country_share),
        "top_5_country_share": float(top_5_country_share),
        "top_company_share": float(top_company_share),
        "top_10_company_share": float(top_10_company_share),
        "valid_contactability_rate": float(valid_contactability_rate),
    }


def generate_top_insights(df: pd.DataFrame, kpis: Dict[str, float]) -> List[str]:
    """
    Produce top five business insights in plain language.

    Why: Decision-makers need narrative findings, not only metrics.
    """
    insights: List[str] = []

    total_customers = int(kpis["total_unique_customers"])
    insights.append(
        f"The customer base currently includes {total_customers} unique records, "
        "providing a sufficient footprint for segmentation by geography and company."
    )

    if "Country" in df.columns and not df.empty:
        top_country = df["Country"].fillna("Unknown").value_counts().idxmax()
        top_country_pct = kpis["top_country_share"] * 100
        insights.append(
            f"Customer concentration is highest in {top_country}, which represents about {top_country_pct:.1f}% "
            "of all records, indicating potential market dependence on one geography."
        )

    if "Company" in df.columns and not df.empty:
        top_company = df["Company"].fillna("Unknown").value_counts().idxmax()
        top_company_pct = kpis["top_company_share"] * 100
        insights.append(
            f"The largest company segment is {top_company} at roughly {top_company_pct:.1f}% of customers, "
            "helping prioritize partnership or account-management focus."
        )

    contactability_pct = kpis["valid_contactability_rate"] * 100
    insights.append(
        f"Data readiness is strong: approximately {contactability_pct:.1f}% of records appear contactable "
        "using both valid email format and phone presence."
    )

    if "Subscription Date" in df.columns:
        growth = kpis["growth_vs_prior_month"]
        if np.isnan(growth):
            insights.append(
                "Month-over-month subscription growth is not computable because the prior month has no records, "
                "so trend interpretation should rely on longer windows."
            )
        else:
            insights.append(
                f"Latest-month subscriptions changed by {growth * 100:.1f}% versus the prior month, "
                "which can be used as an early signal for acquisition momentum."
            )
    else:
        insights.append(
            "Subscription trend analysis is limited because no valid date column is available after cleaning."
        )

    return insights[:5]


def write_outputs(
    output_dir: Path,
    eda: Dict[str, object],
    kpis: Dict[str, float],
    insights: List[str],
) -> Dict[str, Path]:
    """
    Save analysis artifacts for BI consumption and traceability.

    Why: Persisted artifacts support downstream dashboards and audits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eda_path = output_dir / "eda_summary.json"
    kpi_path = output_dir / "kpis.json"
    insights_json_path = output_dir / "insights.json"
    insights_txt_path = output_dir / "insights.txt"

    try:
        eda_path.write_text(json.dumps(eda, indent=2, default=str), encoding="utf-8")
        kpi_path.write_text(json.dumps(kpis, indent=2), encoding="utf-8")
        insights_json_path.write_text(json.dumps({"insights": insights}, indent=2), encoding="utf-8")
        insights_txt_path.write_text(
            "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(insights)), encoding="utf-8"
        )
    except OSError as exc:
        raise OSError(f"Failed to write analysis outputs to: {output_dir}") from exc

    return {
        "eda_summary": eda_path,
        "kpis": kpi_path,
        "insights_json": insights_json_path,
        "insights_txt": insights_txt_path,
    }


def run_analysis(cleaned_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Run end-to-end analysis and persist artifacts.

    Why: Keeps analysis execution reproducible with one command.
    """
    df = load_clean_data(cleaned_path=cleaned_path)
    eda = build_eda_outputs(df=df)
    kpis = compute_kpis(df=df)
    insights = generate_top_insights(df=df, kpis=kpis)
    return write_outputs(output_dir=output_dir, eda=eda, kpis=kpis, insights=insights)


def main() -> None:
    """
    CLI entrypoint for generating analysis deliverables.

    Why: Enables script-style execution in local or CI environments.
    """
    base_dir = Path(__file__).resolve().parent
    cleaned_path = base_dir / "data" / "cleaned" / "customers_cleaned.csv"
    output_dir = base_dir / "outputs"

    written = run_analysis(cleaned_path=cleaned_path, output_dir=output_dir)
    for key, path in written.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
