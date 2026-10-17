"""
generate_data.py
================
Graylogic Technologies — Client Engagement & Account Retention Risk Intelligence
Generates two synthetic datasets:
  • data/clients.csv          — 150 client account records
  • data/monthly_ledger.csv   — 36 months × 150 clients = exactly 5,400 rows

Domain-realistic churn signaling patterns injected for accounts:
  1. Churned accounts prior to churn_date:
     a) Monetary Velocity drop   — billed_amount drops 40-70% over 3-6 months
     b) Payment Friction rise    — payment_lag_days increases to 60-90 days
     c) Escalation Spikes        — escalations_or_disputes rises to 2-4 per month
     d) CSAT Drop                — client_nps_or_csat drops below 2.5
  2. Mild-risk active accounts (early warning / intervention opportunities):
     - Moderate payment lag (38-52 days)
     - Moderate spend deceleration (15-28% reduction)
     - Occasional escalation (1 dispute)
     - Mild CSAT dip (2.8 - 3.4)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
N_CLIENTS         = 150
CHURN_RATE        = 0.22          # ~22% churn rate (33 churned, 117 active)
MILD_RISK_ACTIVE  = 14            # 14 active clients experiencing early warning signals
LEDGER_START      = date(2021, 1, 1)
LEDGER_END        = date(2023, 12, 1)
ACCOUNT_START_MIN = date(2021, 1, 1)
ACCOUNT_START_MAX = date(2023, 1, 1)

OUTPUT_DIR = "data"

# Domain lookup tables
INDUSTRY_VERTICALS = ["Healthcare", "FinTech", "Education", "Retail", "Hospitality", "Logistics"]
ENGAGEMENT_TYPES   = ["Monthly Retainer", "Managed IT Services", "Staff Augmentation", "Fixed-Price Project"]
PRIMARY_SERVICES   = ["Custom Software", "Managed Cloud/IT", "QA/Testing", "AI/ML Platform"]
ACCOUNT_TIERS      = ["Tier 1 (Enterprise)", "Tier 2 (Mid-Market)", "Tier 3 (Growth)"]

# Tier → billing amount ranges (USD/month)
TIER_BILLING_RANGE = {
    "Tier 1 (Enterprise)":  (25_000, 95_000),
    "Tier 2 (Mid-Market)":  (8_000,  25_000),
    "Tier 3 (Growth)":      (2_500,   8_000),
}


def random_date(start: date, end: date) -> date:
    """Return a random date between start and end inclusive."""
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def all_months_between(start: date, end: date) -> list:
    """Return first-of-month dates from start to end inclusive (36 months)."""
    months = []
    current = date(start.year, start.month, 1)
    end_m   = date(end.year, end.month, 1)
    while current <= end_m:
        months.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


ALL_MONTHS = all_months_between(LEDGER_START, LEDGER_END)   # 36 months


# ── Step 1: Generate Clients Table ────────────────────────────────────────────

def generate_clients() -> pd.DataFrame:
    clients = []
    n_churned = int(N_CLIENTS * CHURN_RATE)
    statuses  = ["Churned / Inactive"] * n_churned + ["Active"] * (N_CLIENTS - n_churned)
    random.shuffle(statuses)

    for i in range(N_CLIENTS):
        cid    = f"CLI_{i+1:03d}"
        tier   = np.random.choice(ACCOUNT_TIERS, p=[0.20, 0.45, 0.35])
        status = statuses[i]

        acct_start = random_date(ACCOUNT_START_MIN, ACCOUNT_START_MAX)

        if status == "Churned / Inactive":
            earliest_churn = max(acct_start + timedelta(days=90), date(2023, 8, 1))
            latest_churn   = date(2023, 12, 1)
            if earliest_churn > latest_churn:
                earliest_churn = date(2023, 6, 1)
            churn_date = random_date(earliest_churn, latest_churn)
        else:
            churn_date = None

        clients.append({
            "client_id":            cid,
            "industry_vertical":    random.choice(INDUSTRY_VERTICALS),
            "engagement_type":      random.choice(ENGAGEMENT_TYPES),
            "account_start_date":   acct_start,
            "account_status":       status,
            "churn_date":           churn_date,
            "primary_service_line": random.choice(PRIMARY_SERVICES),
            "account_tier":         tier,
        })

    return pd.DataFrame(clients)


# ── Step 2: Generate Monthly Ledger (5,400 rows) ──────────────────────────────

def generate_ledger(clients_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Pick a subset of active clients for early-warning mild risk signals
    active_client_ids = clients_df[clients_df["account_status"] == "Active"]["client_id"].tolist()
    mild_risk_ids = set(random.sample(active_client_ids, min(MILD_RISK_ACTIVE, len(active_client_ids))))

    for _, client in clients_df.iterrows():
        cid        = client["client_id"]
        tier       = client["account_tier"]
        status     = client["account_status"]
        acct_start = pd.to_datetime(client["account_start_date"]).date()
        churn_date = (pd.to_datetime(client["churn_date"]).date()
                      if pd.notna(client["churn_date"]) else None)

        # Baseline metrics for healthy state
        b_low, b_high   = TIER_BILLING_RANGE[tier]
        base_billing    = np.random.uniform(b_low, b_high)
        base_lag        = np.random.uniform(14, 25)     # days to pay (healthy 14-25)
        base_tickets    = np.random.uniform(1.0, 3.2)   # support tickets/month
        base_csat       = np.random.uniform(4.1, 4.9)   # out of 5.0

        is_churned   = (status == "Churned / Inactive" and churn_date is not None)
        is_mild_risk = (cid in mild_risk_ids)

        # Signal window for churned clients: 3 to 6 months prior to churn_date
        if is_churned:
            signal_window_months = random.randint(3, 6)
            churn_month_date = date(churn_date.year, churn_date.month, 1)
            try:
                churn_idx = ALL_MONTHS.index(churn_month_date)
            except ValueError:
                churn_idx = len(ALL_MONTHS) - 1
            signal_start_idx = max(0, churn_idx - signal_window_months)
        else:
            signal_window_months = 0
            churn_idx            = 999
            signal_start_idx     = 999

        for idx, month in enumerate(ALL_MONTHS):
            # Normal baseline with realistic month-over-month noise
            billing_noise = np.random.normal(0, 0.025)
            billing     = base_billing * (1.0 + billing_noise)
            lag         = max(8.0, base_lag + np.random.normal(0, 2.0))
            tickets     = max(0.0, base_tickets + np.random.normal(0, 0.5))
            escalations = 0
            csat        = min(5.0, max(3.8, base_csat + np.random.normal(0, 0.12)))

            # Pre-start date phase (onboarding / pilot)
            if month < date(acct_start.year, acct_start.month, 1):
                billing = base_billing * 0.50 * (1.0 + billing_noise)
                tickets = max(0.0, tickets * 0.5)

            # ── Inject Churn Signals if in pre-churn window ──
            if is_churned and idx >= signal_start_idx:
                if idx < churn_idx:
                    # Pre-churn deterioration phase
                    span = max(1, churn_idx - signal_start_idx)
                    progress = (idx - signal_start_idx + 1) / span

                    # a) Monetary Velocity drop: varied between 35% and 75%
                    severity = np.random.uniform(0.6, 1.4)
                    drop_pct = np.clip(np.random.uniform(0.35, 0.70) * severity, 0.25, 0.75)
                    billing  = billing * (1.0 - drop_pct * progress)

                    # b) Payment Friction: lag increases to 45-90 days
                    target_lag = np.clip(np.random.uniform(48.0, 85.0) * severity, 42.0, 90.0)
                    lag = base_lag + (target_lag - base_lag) * progress + np.random.normal(0, 2.5)

                    # c) Escalation Spikes: rises to 1-4 formal disputes/month
                    target_esc = np.clip(np.random.uniform(1.2, 3.8) * severity, 1.0, 4.0)
                    escalations = int(np.round(target_esc * progress + np.random.choice([0, 1], p=[0.7, 0.3])))
                    escalations = max(0, min(5, escalations))

                    # d) CSAT Drop: drops to 1.5 - 2.8
                    target_csat = np.clip(np.random.uniform(1.6, 2.6) / severity, 1.2, 2.8)
                    csat = base_csat + (target_csat - base_csat) * progress + np.random.normal(0, 0.1)
                    csat = min(2.8, max(1.0, csat))

                    tickets = base_tickets + 3.0 * progress + np.random.normal(0, 0.5)
                else:
                    # Post-churn wind-down
                    billing     = 0.0
                    lag         = np.random.uniform(60.0, 90.0)
                    escalations = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
                    csat        = np.random.uniform(1.2, 2.2)
                    tickets     = max(0.0, np.random.uniform(0.0, 1.0))

            # ── Inject Early Warning for at-risk active clients in last 6 months ──
            elif is_mild_risk and idx >= (len(ALL_MONTHS) - 6):
                rel_idx = idx - (len(ALL_MONTHS) - 6) + 1
                prog = rel_idx / 6.0
                drop_pct = np.random.uniform(0.62, 0.76)
                billing = billing * (1.0 - drop_pct * prog)
                target_lag = np.random.uniform(75.0, 90.0)
                lag = base_lag + (target_lag - base_lag) * prog + np.random.normal(0, 2.0)
                escalations = int(np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
                csat = base_csat - (base_csat - np.random.uniform(1.3, 1.9)) * prog
                tickets = base_tickets + 3.2 * prog

            rows.append({
                "client_id":              cid,
                "billing_month":          month,
                "billed_amount":          round(max(0.0, float(billing)), 2),
                "payment_lag_days":       round(max(5.0, float(lag)), 1),
                "support_tickets_raised": round(max(0.0, float(tickets)), 1),
                "escalations_or_disputes": int(escalations),
                "client_nps_or_csat":     round(min(5.0, max(1.0, float(csat))), 2),
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Graylogic Technologies — Synthetic Data Generator")
    print("=" * 56)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating Clients table (150 records) …")
    clients_df = generate_clients()
    clients_path = os.path.join(OUTPUT_DIR, "clients.csv")
    clients_df.to_csv(clients_path, index=False)
    n_churned = (clients_df["account_status"] == "Churned / Inactive").sum()
    print(f"  ✓ {len(clients_df)} clients  |  Churned: {n_churned} ({n_churned/len(clients_df):.1%})")

    print("Generating Monthly Ledger (36 months × 150 clients = 5,400 rows) …")
    ledger_df = generate_ledger(clients_df)
    ledger_path = os.path.join(OUTPUT_DIR, "monthly_ledger.csv")
    ledger_df.to_csv(ledger_path, index=False)
    print(f"  ✓ {len(ledger_df):,} ledger rows generated")
    print(f"  ✓ Unique months per client: {ledger_df.groupby('client_id')['billing_month'].nunique().mean():.1f}")

    print(f"\n✅ Generated datasets saved successfully:")
    print(f"   • {clients_path}")
    print(f"   • {ledger_path}")
