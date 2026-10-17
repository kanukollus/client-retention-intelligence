"""
src/data_processing.py
=======================
Graylogic Technologies — IT Client Engagement & Account Retention Risk Intelligence
Feature engineering and aggregation pipeline:
  • Loads raw CSVs (Clients metadata + Monthly Ledger)
  • Computes RFM and operational risk features per account:
      - Trailing 3-month & 6-month average billed amount
      - Spend Velocity Ratio: (Spend_Trailing_3M + 1) / (Spend_Prior_3M + 1)
      - Payment Friction Trend: Trailing 3M avg lag - Prior 3M avg lag
      - Escalation Count: Total formal disputes in trailing 6 months
      - Average CSAT / NPS in trailing 6 months
      - Support ticket intensity & trends
  • Merges aggregated features with client firmographic metadata
  • Encodes categoricals, handles missing values, and prepares modeling matrices
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Reference evaluation date ("today" for the portfolio analysis)
REFERENCE_DATE = pd.Timestamp("2024-01-01")


def _rolling_mean(series: pd.Series, n: int, default: float = 0.0) -> float:
    """Compute mean of the last n elements in series; returns default if empty."""
    tail = series.tail(n)
    return float(tail.mean()) if len(tail) > 0 else default


def engineer_client_features(client_id: str, ledger: pd.DataFrame) -> dict:
    """
    Given a client's full chronological ledger, compute account-level RFM
    and operational early-warning signals using the trailing history.
    """
    df = ledger[ledger["client_id"] == client_id].copy()
    df["billing_month"] = pd.to_datetime(df["billing_month"])
    df = df.sort_values("billing_month").reset_index(drop=True)

    if df.empty:
        return {}

    billing = df["billed_amount"]
    lag     = df["payment_lag_days"]
    esc     = df["escalations_or_disputes"]
    csat    = df["client_nps_or_csat"]
    tickets = df["support_tickets_raised"]

    # ── 1. Monetary & Spend Velocity ──────────────────────────────────────────
    spend_t3 = _rolling_mean(billing, 3)                     # Trailing 3-month avg
    spend_t6 = _rolling_mean(billing, 6)                     # Trailing 6-month avg
    spend_p3 = _rolling_mean(billing.iloc[:-3], 3) if len(billing) > 3 else spend_t3
    spend_p6 = _rolling_mean(billing.iloc[:-6], 6) if len(billing) > 6 else spend_t6

    # Spend Velocity Ratio: (Spend_Trailing_3M + 1) / (Spend_Prior_3M + 1)
    spend_velocity_ratio = (spend_t3 + 1.0) / (spend_p3 + 1.0)

    peak_monthly_spend   = float(billing.max())
    lifetime_spend       = float(billing.sum())
    current_monthly_spend = float(billing.iloc[-1])
    months_recorded      = len(df)

    # ── 2. Payment Friction & Liquidity Lag ───────────────────────────────────
    lag_t3 = _rolling_mean(lag, 3)                           # Trailing 3-month avg lag
    lag_p3 = _rolling_mean(lag.iloc[:-3], 3) if len(lag) > 3 else lag_t3
    payment_friction_trend = lag_t3 - lag_p3                 # Trailing 3M minus Prior 3M

    avg_lag_t6      = _rolling_mean(lag, 6)
    max_payment_lag = float(lag.max())
    current_lag     = float(lag.iloc[-1])

    # ── 3. Operational Escalations & Disputes ─────────────────────────────────
    escalation_count_6m   = int(esc.tail(6).sum())           # Total formal disputes in trailing 6m
    escalation_count_3m   = int(esc.tail(3).sum())
    avg_escalations_t6    = _rolling_mean(esc, 6)
    has_recent_escalation = int(escalation_count_3m > 0)
    current_escalations   = int(esc.iloc[-1])

    # Support tickets
    avg_tickets_t6 = _rolling_mean(tickets, 6)
    total_tickets_6m = float(tickets.tail(6).sum())

    # ── 4. CSAT / NPS Quality Tracking ────────────────────────────────────────
    avg_csat_t6  = _rolling_mean(csat, 6)                   # Average CSAT/NPS in trailing 6m
    csat_t3      = _rolling_mean(csat, 3)
    csat_p3      = _rolling_mean(csat.iloc[:-3], 3) if len(csat) > 3 else csat_t3
    csat_trend   = csat_t3 - csat_p3
    min_csat_obs = float(csat.min())
    current_csat = float(csat.iloc[-1])

    # ── 5. Recency ────────────────────────────────────────────────────────────
    last_month   = df["billing_month"].max()
    recency_days = (REFERENCE_DATE - last_month).days

    return {
        "client_id":                 client_id,
        # Spend Features
        "spend_trailing_3m":         round(spend_t3, 2),
        "spend_trailing_6m":         round(spend_t6, 2),
        "spend_velocity_ratio":      round(spend_velocity_ratio, 4),
        "peak_monthly_spend":        round(peak_monthly_spend, 2),
        "lifetime_spend":            round(lifetime_spend, 2),
        "current_monthly_spend":     round(current_monthly_spend, 2),
        "months_recorded":           months_recorded,
        # Payment Friction Features
        "payment_lag_t3":            round(lag_t3, 2),
        "payment_friction_trend":    round(payment_friction_trend, 2),
        "avg_lag_t6":                round(avg_lag_t6, 2),
        "max_payment_lag":           round(max_payment_lag, 2),
        "current_lag":               round(current_lag, 2),
        # Escalations & Support
        "escalation_count_6m":       escalation_count_6m,
        "escalation_count_3m":       escalation_count_3m,
        "avg_escalations_t6":        round(avg_escalations_t6, 4),
        "has_recent_escalation":     has_recent_escalation,
        "current_escalations":       current_escalations,
        "avg_tickets_t6":            round(avg_tickets_t6, 2),
        "total_tickets_6m":          round(total_tickets_6m, 1),
        # CSAT / NPS
        "avg_csat_t6":               round(avg_csat_t6, 4),
        "csat_trend":                round(csat_trend, 4),
        "min_csat_observed":         round(min_csat_obs, 2),
        "current_csat":              round(current_csat, 2),
        # Recency
        "recency_days":              int(recency_days),
    }


def load_and_engineer(
    clients_path: str = "data/clients.csv",
    ledger_path:  str = "data/monthly_ledger.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full data processing & feature engineering pipeline:
      1. Loads raw CSVs
      2. Computes per-client RFM and operational features
      3. Merges with client metadata
      4. Creates target variable 'churned'
      5. One-hot encodes categorical variables
      6. Handles missing values
    Returns:
      (processed_df, raw_ledger_df)
    """
    clients = pd.read_csv(clients_path, parse_dates=["account_start_date", "churn_date"])
    ledger  = pd.read_csv(ledger_path,  parse_dates=["billing_month"])

    # Aggregate features per client
    features = [engineer_client_features(cid, ledger) for cid in clients["client_id"].unique()]
    features_df = pd.DataFrame(features)

    # Merge metadata with engineered features
    merged = clients.merge(features_df, on="client_id", how="left")

    # Account tenure / age in days from contract inception
    merged["account_age_days"] = (
        pd.Timestamp("2024-01-01") - merged["account_start_date"]
    ).dt.days.clip(lower=0)

    # Binary classification target: 1 = Churned, 0 = Active
    merged["churned"] = (merged["account_status"] == "Churned / Inactive").astype(int)

    # Preserve original categorical columns for UI display/filtering,
    # and append one-hot encoded dummy columns for model inputs
    cat_cols = ["industry_vertical", "engagement_type", "primary_service_line", "account_tier"]
    dummies  = pd.get_dummies(merged[cat_cols], drop_first=False)
    merged_encoded = pd.concat([merged, dummies], axis=1)

    # Ensure all numeric columns are clean and impute median if any NaN exists
    numeric_cols = merged_encoded.select_dtypes(include=[np.number]).columns
    merged_encoded[numeric_cols] = merged_encoded[numeric_cols].fillna(
        merged_encoded[numeric_cols].median()
    )

    return merged_encoded, ledger


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ordered list of model input feature names (excluding IDs, metadata, and targets)."""
    exclude = {
        "client_id", "account_start_date", "churn_date", "account_status",
        "churned", "billing_month", "predicted_churn_prob", "risk_tier",
        "industry_vertical", "engagement_type", "primary_service_line", "account_tier"
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


def get_ledger_for_client(client_id: str, ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Return chronological monthly ledger slice for a specific client."""
    df = ledger_df[ledger_df["client_id"] == client_id].copy()
    df["billing_month"] = pd.to_datetime(df["billing_month"])
    return df.sort_values("billing_month").reset_index(drop=True)


if __name__ == "__main__":
    print("Testing data processing pipeline …")
    df, ledger = load_and_engineer()
    features = get_feature_columns(df)
    print(f"✓ Processed dataset shape: {df.shape}")
    print(f"✓ Engineered features count: {len(features)}")
    print(f"✓ Target distribution (churned): {df['churned'].value_counts().to_dict()}")
    print("✓ Sample features:", features[:8])
