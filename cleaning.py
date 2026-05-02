"""
Data cleaning pipeline for customer subscription records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    Load the raw customer dataset from CSV.

    Why: Centralizing ingestion makes path handling and error reporting consistent.
    """
    try:
        return pd.read_csv(input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw data file not found: {input_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw data file is empty: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed for: {input_path}") from exc


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim and standardize casing for key textual attributes.

    Why: Consistent text improves grouping accuracy for BI metrics.
    """
    normalized = df.copy()
    title_cols = ["First Name", "Last Name", "Company", "City", "Country"]

    for col in title_cols:
        if col in normalized.columns:
            normalized[col] = (
                normalized[col]
                .astype(str)
                .str.strip()
                .replace({"": np.nan})
                .str.title()
            )

    for col in ["Email", "Website"]:
        if col in normalized.columns:
            normalized[col] = (
                normalized[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": np.nan})
            )

    for col in ["Phone 1", "Phone 2", "Customer Id"]:
        if col in normalized.columns:
            normalized[col] = normalized[col].astype(str).str.strip().replace({"": np.nan})

    return normalized


def parse_subscription_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert subscription date into datetime and create parse-quality flags.

    Why: Time-based KPIs require a reliable date field.
    """
    parsed = df.copy()
    if "Subscription Date" not in parsed.columns:
        raise KeyError("Missing required column: Subscription Date")

    parsed["subscription_date_raw"] = parsed["Subscription Date"].astype(str).str.strip()
    parsed["Subscription Date"] = pd.to_datetime(parsed["subscription_date_raw"], errors="coerce")
    parsed["subscription_date_invalid_flag"] = parsed["Subscription Date"].isna()
    return parsed


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add data quality flags for email, phone, and website fields.

    Why: These flags provide business-friendly reliability KPIs downstream.
    """
    flagged = df.copy()
    email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    website_pattern = re.compile(r"^[a-z0-9][a-z0-9\.-]+\.[a-z]{2,}$")

    flagged["email_valid_flag"] = flagged["Email"].fillna("").astype(str).str.match(email_pattern)
    flagged["phone_present_flag"] = (
        flagged["Phone 1"].fillna("").astype(str).str.contains(r"\d")
        | flagged["Phone 2"].fillna("").astype(str).str.contains(r"\d")
    )
    flagged["website_valid_flag"] = flagged["Website"].fillna("").astype(str).str.match(website_pattern)
    return flagged


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply explicit missing-value treatment strategy by column type.

    Why: Retaining rows while clearly labeling unknown values preserves analytic coverage.
    """
    cleaned = df.copy()
    string_fill_cols = ["First Name", "Last Name", "Company", "City", "Country"]

    for col in string_fill_cols:
        if col in cleaned.columns:
            # WHAT: Fill business dimensions with "Unknown" when blank.
            # WHY: Segment-level KPIs should keep records instead of dropping customers.
            cleaned[col] = cleaned[col].fillna("Unknown")

    if "Customer Id" in cleaned.columns:
        # WHAT: Drop rows with missing customer identifier.
        # WHY: Customer Id is the record key used for deduplication and analysis.
        cleaned = cleaned.dropna(subset=["Customer Id"])

    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records, prioritizing unique customer identity.

    Why: Duplicate customers can inflate counts and mislead growth metrics.
    """
    deduped = df.copy()
    if "Customer Id" in deduped.columns:
        deduped = deduped.sort_values("Subscription Date").drop_duplicates(
            subset=["Customer Id"], keep="last"
        )
    deduped = deduped.drop_duplicates()
    return deduped


def cap_outliers_iqr(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Cap numeric outliers using the IQR rule.

    Why: Protects aggregate metrics when future data introduces extreme values.
    """
    capped = df.copy()
    outlier_summary: Dict[str, Dict[str, float]] = {}
    numeric_cols = capped.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        series = capped[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((capped[col] < lower) | (capped[col] > upper)).sum())
        capped[col] = capped[col].clip(lower=lower, upper=upper)
        outlier_summary[col] = {"lower_bound": lower, "upper_bound": upper, "rows_capped": outliers}

    return capped, outlier_summary


def create_engineered_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns used by analysis and dashboard modules.

    Why: Shared engineered fields avoid repeated logic in downstream scripts.
    """
    engineered = df.copy()
    engineered["full_name"] = (
        engineered["First Name"].fillna("").astype(str).str.strip()
        + " "
        + engineered["Last Name"].fillna("").astype(str).str.strip()
    ).str.strip()
    engineered["email_domain"] = engineered["Email"].fillna("").astype(str).str.split("@").str[-1]
    engineered["website_domain"] = engineered["Website"].fillna("").astype(str)

    if "Subscription Date" in engineered.columns:
        engineered["subscription_year"] = engineered["Subscription Date"].dt.year
        engineered["subscription_month"] = engineered["Subscription Date"].dt.month
        engineered["year_month"] = engineered["Subscription Date"].dt.to_period("M").astype(str)

    return engineered


def save_outputs(
    df: pd.DataFrame, outlier_summary: Dict[str, Dict[str, float]], output_dir: Path
) -> Tuple[Path, Path]:
    """
    Save cleaned data and cleaning metadata artifacts.

    Why: Persisted outputs make the pipeline reproducible for BI and audits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "customers_cleaned.csv"
    report_path = output_dir / "cleaning_report.json"

    try:
        df.to_csv(cleaned_path, index=False)
        report = {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "null_counts": {col: int(val) for col, val in df.isna().sum().items()},
            "duplicate_rows": int(df.duplicated().sum()),
            "outlier_capping": outlier_summary,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to write cleaned outputs to: {output_dir}") from exc

    return cleaned_path, report_path


def run_cleaning_pipeline(input_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Execute complete cleaning flow from raw ingestion to saved outputs.

    Why: A single orchestration function supports script and module usage.
    """
    df = load_raw_data(input_path=input_path)
    df = normalize_text_columns(df=df)
    df = parse_subscription_date(df=df)
    df = add_quality_flags(df=df)
    df = handle_missing_values(df=df)
    df = remove_duplicates(df=df)
    df, outlier_summary = cap_outliers_iqr(df=df)
    df = create_engineered_fields(df=df)
    return save_outputs(df=df, outlier_summary=outlier_summary, output_dir=output_dir)


def main() -> None:
    """
    Entrypoint for CLI execution.

    Why: Enables `python cleaning.py` with predictable filesystem behavior.
    """
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "customers-1000.csv"
    output_dir = base_dir / "data" / "cleaned"

    cleaned_path, report_path = run_cleaning_pipeline(input_path=input_path, output_dir=output_dir)
    print(f"Cleaned data written to: {cleaned_path}")
    print(f"Cleaning report written to: {report_path}")


if __name__ == "__main__":
    main()
