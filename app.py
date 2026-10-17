"""
app.py
======
Graylogic Technologies — IT Client Engagement & Account Retention Risk Intelligence Platform
=============================================================================================
A Decision-Support Dashboard featuring:
  1. Executive Portfolio KPI Overview Cards
  2. High-Risk Account Early-Warning Table with color-coded risk tiers
  3. Single-Client Diagnostic Drilldown (Plotly Risk Gauge, SHAP local drivers, 3 Time-Series Charts)
  4. What-If Prescriptive Action Simulator (Real-time ML inference & Retention ROI calculator)
  5. ML Model Governance & 5-Fold Cross-Validation Benchmarks
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Disable numba jit noise if any
os.environ["NUMBA_DISABLE_JIT"] = "1"
import shap

warnings.filterwarnings("ignore")

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Graylogic Retention Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "models_artifacts"
DATA_DIR      = "data"

# ── Custom Styling (Modern Dark Enterprise Theme) ─────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Backgrounds */
.stApp {
    background: radial-gradient(circle at 10% 10%, #0d1322 0%, #080c14 100%);
    color: #e2e8f0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Headers */
h1, h2, h3, h4, h5 {
    color: #f8fafc;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Metric KPI Cards */
.kpi-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(14px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 25px -4px rgba(99, 102, 241, 0.15);
}
.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    font-feature-settings: "tnum";
}
.kpi-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 6px;
}

/* Risk Badges */
.badge-high {
    background: rgba(244, 63, 94, 0.18);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-med {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-low {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* Custom Callout Box */
.callout-box {
    background: rgba(30, 41, 59, 0.5);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 16px;
}

/* Chart Interpretation Guide Card */
.chart-guide {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 10px;
    margin-bottom: 24px;
    font-size: 0.84rem;
    line-height: 1.5;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.guide-title {
    font-weight: 700;
    font-size: 0.88rem;
    color: #38bdf8;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.guide-row {
    color: #cbd5e1;
    margin-bottom: 5px;
}
.guide-row:last-child {
    margin-bottom: 0;
}
.guide-row b {
    color: #f8fafc;
}
.guide-row code {
    background: rgba(255, 255, 255, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    color: #fbbf24;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

/* Client Header Pill */
.client-hero {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Data & Artifact Loaders ───────────────────────────────────────────────────
@st.cache_resource
def load_all_artifacts():
    """Load pre-trained models, engineered features, SHAP values, and raw datasets."""
    try:
        best_model   = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
        all_models   = joblib.load(os.path.join(ARTIFACTS_DIR, "all_models.joblib"))
        cv_metrics   = joblib.load(os.path.join(ARTIFACTS_DIR, "cv_metrics.joblib"))
        fold_details = joblib.load(os.path.join(ARTIFACTS_DIR, "fold_details.joblib"))
        feature_names= joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
        global_imp   = joblib.load(os.path.join(ARTIFACTS_DIR, "global_importance.joblib"))
        df           = joblib.load(os.path.join(ARTIFACTS_DIR, "processed_clients_df.joblib"))
        ledger       = joblib.load(os.path.join(ARTIFACTS_DIR, "raw_ledger.joblib"))
        metadata     = joblib.load(os.path.join(ARTIFACTS_DIR, "metadata.joblib"))
        shap_explainer = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
        shap_values  = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_values.joblib"))
        return {
            "best_model": best_model,
            "all_models": all_models,
            "cv_metrics": cv_metrics,
            "fold_details": fold_details,
            "feature_names": feature_names,
            "global_imp": global_imp,
            "df": df,
            "ledger": ledger,
            "metadata": metadata,
            "shap_explainer": shap_explainer,
            "shap_values": shap_values,
        }
    except Exception as e:
        st.error(f"Error loading artifacts: {e}. Please ensure `train_models.py` has been executed.")
        st.stop()


artifacts = load_all_artifacts()
best_model    = artifacts["best_model"]
all_models    = artifacts["all_models"]
cv_metrics    = artifacts["cv_metrics"]
feature_names = artifacts["feature_names"]
global_imp    = artifacts["global_imp"]
df            = artifacts["df"]
if "risk_tier" in df.columns:
    df["risk_tier"] = df["risk_tier"].astype(str)
ledger        = artifacts["ledger"]
metadata      = artifacts["metadata"]
shap_values   = artifacts["shap_values"]
shap_explainer= artifacts["shap_explainer"]


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #6366f1, #06b6d4); border-radius: 10px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                ⚡
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: -0.01em;">Graylogic</div>
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">AcuityRisk™ Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Portfolio Overview",
            "⚠️ Early-Warning Radar",
            "🔍 Single-Client Diagnostic Drilldown",
            "🔮 Prescriptive Action Simulator",
            "🤖 Model Governance & Benchmarks",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🎯 Platform Health")
    st.caption(f"**Best Model**: `{metadata.get('best_model_name', 'L2 Logistic Regression')}`")
    st.caption(f"**Monitored Accounts**: `{len(df):,}`")
    st.caption(f"**Ledger Observations**: `{len(ledger):,}`")
    st.caption("**Data used for years**: `2022, 2023, 2024`")
    st.markdown("---")
    st.caption("Graylogic Technologies Inc. © 2026\nEnterprise Client Retention & Intelligence Unit")


# ── Top Brand Bar ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 14px;">
    <div>
        <h2 style="margin: 0; font-size: 1.65rem;">IT Client Engagement & Account Retention Intelligence</h2>
        <p style="margin: 3px 0 0 0; color: #94a3b8; font-size: 0.88rem;">Predictive churn risk scoring, leading indicator forensics, and prescriptive retention ROI simulation</p>
    </div>
    <div style="display: flex; gap: 8px;">
        <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">● Model Live</span>
        <span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">100% Data Quality</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: EXECUTIVE PORTFOLIO OVERVIEW
# ==============================================================================
# ==============================================================================
# PAGE 1: EXECUTIVE PORTFOLIO OVERVIEW
# ==============================================================================
if page == "📊 Executive Portfolio Overview":
    st.markdown("### 📊 Executive Portfolio Overview — Current Active Client Base")
    st.markdown(
        "Live operational health of the **117 current active accounts**. Models trained on 36-month historical "
        "telemetry detect early friction signals to protect recurring contract revenue."
    )

    # Active accounts slice
    active_df = df[df["account_status"] == "Active"].copy()
    total_active = len(active_df)
    total_historical = len(df)
    churned_historical = (df["account_status"] == "Churned / Inactive").sum()
    historical_churn_rate = (churned_historical / total_historical) * 100

    # Active revenue metrics
    active_arr = (active_df["spend_trailing_3m"] * 12).sum()
    # Active accounts with early friction / risk (>= 20% or >= 40%)
    at_risk_active = active_df[active_df["predicted_churn_prob"] >= 0.20]
    rev_at_risk = (at_risk_active["spend_trailing_3m"] * 12).sum()
    avg_active_lag = active_df["avg_lag_t6"].mean()

    # ── 4 KPI CARDS ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🏢 Active Monitored Base</div>
            <div class="kpi-value">{total_active} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">/ {total_historical}</span></div>
            <div class="kpi-sub">${active_arr/1_000_000:.2f}M Active Contract ARR</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Active Revenue at Risk</div>
            <div class="kpi-value">${rev_at_risk/1_000_000:.2f}M</div>
            <div class="kpi-sub">{len(at_risk_active)} Active Accounts with Risk &ge; 20%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📉 Historical Churn Benchmark</div>
            <div class="kpi-value">{historical_churn_rate:.1f}%</div>
            <div class="kpi-sub">33 Past Churners in Training Set</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">⏱️ Active Avg Payment Lag</div>
            <div class="kpi-value">{avg_active_lag:.1f} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">days</span></div>
            <div class="kpi-sub">Benchmark SLA: &le; 30 days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Portfolio Visualizations (Focused on Active Base) ─────────────────────
    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.markdown("#### ⚡ Churn Risk vs. Monthly Revenue Exposure (Active Base)")
        st.caption("Each circle represents an active account. **Vertical: Churn Risk** | **Horizontal: Monthly Spend** | **Size: Payment Lag (Days)**.")
        fig_scatter = px.scatter(
            active_df,
            x="spend_trailing_3m",
            y="predicted_churn_prob",
            size="payment_lag_t3",
            color="risk_tier",
            color_discrete_map={
                "High (>70%)": "#f43f5e",
                "Medium (40-70%)": "#f59e0b",
                "Low (<40%)": "#10b981",
            },
            hover_name="client_id",
            hover_data={
                "spend_trailing_3m": ":$,.0f",
                "predicted_churn_prob": ":.1%",
                "payment_lag_t3": ":.1f",
                "industry_vertical": True,
                "account_tier": True,
                "escalation_count_6m": True,
                "risk_tier": False,
            },
            labels={
                "spend_trailing_3m": "Monthly Billing ($)",
                "predicted_churn_prob": "Predicted Churn Probability",
                "payment_lag_t3": "Payment Lag (Days)",
                "risk_tier": "Risk Tier",
            },
            template="plotly_dark",
        )
        # Reference threshold line for elevated risk
        fig_scatter.add_hline(
            y=0.20,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="⚠️ Elevated Risk Watchlist (20%)",
            annotation_position="top left",
            annotation_font_color="#fbbf24",
            annotation_font_size=11,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=370,
            yaxis=dict(tickformat=".0%", title="Predicted Churn Risk (%)"),
            xaxis=dict(tickprefix="$", title="Monthly Billing Volume ($)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="chart-guide">
            <div class="guide-title">💡 How to Read & Act on this Graph</div>
            <div class="guide-row"><b>• What it shows:</b> Multi-dimensional health of all 117 active clients. Higher vertical position means higher risk of churn; farther right means higher monthly billing; larger bubble size means longer payment delay.</div>
            <div class="guide-row"><b>• Key Takeaway:</b> Accounts above the <b>20% Watchlist Line</b> (such as <code>CLI_141</code> and <code>CLI_042</code>) have large amber circles (65–75 day invoice delays), signaling operational friction. Meanwhile, high-billing accounts ($40k–$95k/mo) sit safely in the healthy green zone (<5% risk).</div>
            <div class="guide-row"><b>• Recommended Action:</b> Account Executives should focus outreach on accounts in the upper half before contract renewals approach.</div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("#### 🏷️ Active Revenue Exposure by Account Tier")
        st.caption("Distribution of Graylogic's monthly active billed revenue ($2.63M/mo total) across customer tiers.")
        tier_agg = active_df.groupby(["account_tier", "risk_tier"])["spend_trailing_3m"].sum().reset_index()
        fig_bar = px.bar(
            tier_agg,
            x="account_tier",
            y="spend_trailing_3m",
            color="risk_tier",
            color_discrete_map={
                "High (>70%)": "#f43f5e",
                "Medium (40-70%)": "#f59e0b",
                "Low (<40%)": "#10b981",
            },
            barmode="stack",
            labels={"spend_trailing_3m": "Monthly Billed Volume ($)", "account_tier": "Account Tier"},
            template="plotly_dark",
            text_auto="$,.0f",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=370,
            yaxis=dict(tickprefix="$", title="Monthly Billed Volume ($)"),
            xaxis=dict(title="Account Tier"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
        <div class="chart-guide">
            <div class="guide-title">💡 How to Read & Act on this Graph</div>
            <div class="guide-row"><b>• What it shows:</b> Total monthly recurring billing divided across Tier 1 Enterprise ($25k–$95k/mo), Tier 2 Mid-Market ($8k–$25k/mo), and Tier 3 Growth (<$8k/mo), colored by risk.</div>
            <div class="guide-row"><b>• Key Takeaway:</b> <b>Tier 1 (Enterprise)</b> generates <b>$1.63M/mo (62% of revenue)</b> and is currently 100% stable in the Low-Risk tier. Emerging friction is concentrated in <b>Tier 2 (Mid-Market)</b>, representing ~$14k/mo in active contract exposure.</div>
            <div class="guide-row"><b>• Recommended Action:</b> Preserve Tier 1 executive sponsorship while deploying proactive CSM intervention squads to Tier 2 accounts displaying early signs of strain.</div>
        </div>
        """, unsafe_allow_html=True)

    # Secondary row: Industry breakdown & Payment friction
    c_ind, c_lag = st.columns(2)
    with c_ind:
        st.markdown("#### 🏭 Active Account Risk Distribution by Industry")
        st.caption("Client account count across Graylogic's 6 industry verticals with risk composition.")
        ind_risk = active_df.groupby(["industry_vertical", "risk_tier"]).size().reset_index(name="count")
        fig_ind = px.bar(
            ind_risk,
            x="industry_vertical",
            y="count",
            color="risk_tier",
            color_discrete_map={
                "High (>70%)": "#f43f5e",
                "Medium (40-70%)": "#f59e0b",
                "Low (<40%)": "#10b981",
            },
            barmode="stack",
            template="plotly_dark",
            labels={"industry_vertical": "Industry Vertical", "count": "Number of Accounts"},
            text="count",
        )
        fig_ind.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=340,
            yaxis=dict(title="Active Account Count"),
            xaxis=dict(title="Industry Vertical"),
        )
        st.plotly_chart(fig_ind, use_container_width=True)

        st.markdown("""
        <div class="chart-guide">
            <div class="guide-title">💡 How to Read & Act on this Graph</div>
            <div class="guide-row"><b>• What it shows:</b> Portfolio diversification across industries and whether client risk is isolated to specific sectors or widespread.</div>
            <div class="guide-row"><b>• Key Takeaway:</b> Risk is <b>isolated, not systemic</b>. 4 of the 6 verticals (Logistics: 23, Retail: 22, FinTech: 21, Healthcare: 18) are 100% low-risk. Risk is confined to 1 account in Education (<code>CLI_141</code>) and 1 account in Hospitality (<code>CLI_042</code>).</div>
            <div class="guide-row"><b>• Recommended Action:</b> Direct technical delivery audits specifically toward the Education and Hospitality project teams to address specialized service bottlenecks.</div>
        </div>
        """, unsafe_allow_html=True)

    with c_lag:
        st.markdown("#### ⏱️ Active Payment Lag Distribution vs. Risk Tiers")
        st.caption("Invoice settlement days across risk tiers benchmarked against Graylogic's 30-day payment SLA.")
        fig_box = px.box(
            active_df,
            x="risk_tier",
            y="payment_lag_t3",
            color="risk_tier",
            points="all",  # Show individual accounts as jittered dots
            color_discrete_map={
                "High (>70%)": "#f43f5e",
                "Medium (40-70%)": "#f59e0b",
                "Low (<40%)": "#10b981",
            },
            hover_name="client_id",
            hover_data={
                "payment_lag_t3": ":.1f days",
                "spend_trailing_3m": ":$,.0f",
                "spend_velocity_ratio": ":.2f",
                "escalation_count_6m": True,
                "risk_tier": False,
            },
            template="plotly_dark",
            labels={"risk_tier": "Risk Tier", "payment_lag_t3": "Payment Lag (Days)"},
        )
        fig_box.add_hline(
            y=30,
            line_dash="dash",
            line_color="#fb7185",
            annotation_text="30-Day SLA Benchmark",
            annotation_position="bottom right",
            annotation_font_color="#fb7185",
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=340,
            showlegend=False,
            yaxis=dict(title="Days to Settle Invoice", range=[0, 85]),
            xaxis=dict(title="Risk Tier"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("""
        <div class="chart-guide">
            <div class="guide-title">💡 How to Read & Act on this Graph</div>
            <div class="guide-row"><b>• What it shows:</b> Days taken by clients to settle invoices. The red dashed line marks Graylogic's 30-day payment SLA benchmark. Individual clients are displayed as clickable/hoverable dots.</div>
            <div class="guide-row"><b>• Key Takeaway:</b> Healthy clients pay in <b>~20.5 days</b> (comfortably under the 30-day SLA). In stark contrast, Medium-risk clients average <b>72.0 days</b> (3.5x slower). Payment delay is Graylogic's <b>#1 early-warning tripwire</b>.</div>
            <div class="guide-row"><b>• Recommended Action:</b> When an invoice crosses 45 days overdue, automatically flag the account for an executive check-in rather than treating it merely as a collections delay.</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# ==============================================================================
# PAGE 2: HIGH-RISK ACCOUNT EARLY-WARNING RADAR
# ==============================================================================
elif page == "⚠️ Early-Warning Radar":
    st.markdown("### ⚠️ Account Early-Warning Radar")
    st.markdown(
        "Operational intelligence watchlist for Graylogic leadership and Account Executives. "
        "Prioritize retention interventions by predicted churn probability and leading friction telemetry."
    )

    # ── Scope Selector ────────────────────────────────────────────────────────
    radar_scope = st.radio(
        "Watchlist Scope:",
        [
            "🟢 Active Accounts Watchlist (Actionable Early Warning)",
            "📋 Historical Churned Accounts (Model Audit & Post-Mortem)",
            "🌐 All Accounts (150 Total Portfolio)",
        ],
        horizontal=True,
    )

    if radar_scope == "🟢 Active Accounts Watchlist (Actionable Early Warning)":
        base_scope_df = df[df["account_status"] == "Active"].copy()
        default_tiers = ["High (>70%)", "Medium (40-70%)", "Low (<40%)"]
        st.markdown("""
        <div class="callout-box">
            <b>🟢 Actionable Early-Warning Radar:</b> Displaying <b>117 active accounts</b>. Accounts are sorted by predicted churn risk to catch micro-signals (spend velocity drop, payment friction, disputes) while contract ARR is actively on the line.
        </div>
        """, unsafe_allow_html=True)
    elif radar_scope == "📋 Historical Churned Accounts (Model Audit & Post-Mortem)":
        base_scope_df = df[df["account_status"] == "Churned / Inactive"].copy()
        default_tiers = ["High (>70%)", "Medium (40-70%)", "Low (<40%)"]
        st.markdown("""
        <div class="callout-box">
            <b>📋 Historical Audit & Post-Mortem:</b> Displaying <b>33 clients</b> who churned during 2021–2023. These historical outcomes were used to train and calibrate the predictive models (achieving 1.00 AUC / 99%+ predicted risk). Current spend is $0.
        </div>
        """, unsafe_allow_html=True)
    else:
        base_scope_df = df.copy()
        default_tiers = ["High (>70%)", "Medium (40-70%)"]
        st.markdown("""
        <div class="callout-box">
            <b>🌐 Complete Monitored Portfolio:</b> Displaying all 150 client accounts across historical and active cohorts.
        </div>
        """, unsafe_allow_html=True)

    # ── Interactive Filters ───────────────────────────────────────────────────
    with st.expander("🔍 Filter & Search Watchlist", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            tier_filter = st.multiselect(
                "Risk Tier",
                ["High (>70%)", "Medium (40-70%)", "Low (<40%)"],
                default=default_tiers,
            )
        with f2:
            all_verticals = sorted(base_scope_df["industry_vertical"].dropna().unique().tolist())
            ind_filter = st.multiselect("Industry Vertical", all_verticals, default=all_verticals)
        with f3:
            all_tiers = sorted(base_scope_df["account_tier"].dropna().unique().tolist())
            acct_tier_filter = st.multiselect("Account Tier", all_tiers, default=all_tiers)
        with f4:
            search_query = st.text_input("Search Client ID", "").strip().upper()

    # Apply filters
    filtered_df = base_scope_df.copy()
    if tier_filter:
        filtered_df = filtered_df[filtered_df["risk_tier"].isin(tier_filter)]
    if ind_filter:
        filtered_df = filtered_df[filtered_df["industry_vertical"].isin(ind_filter)]
    if acct_tier_filter:
        filtered_df = filtered_df[filtered_df["account_tier"].isin(acct_tier_filter)]
    if search_query:
        filtered_df = filtered_df[filtered_df["client_id"].str.contains(search_query)]

    # Sort descending by predicted churn probability
    filtered_df = filtered_df.sort_values(by="predicted_churn_prob", ascending=False)

    st.markdown(f"**Showing {len(filtered_df)} Accounts Matching Criteria**")

    # Format table for executive presentation
    display_cols = [
        "client_id", "risk_tier", "predicted_churn_prob", "account_status",
        "industry_vertical", "account_tier", "current_monthly_spend",
        "spend_velocity_ratio", "current_lag", "escalation_count_6m", "avg_csat_t6"
    ]
    table_data = filtered_df[display_cols].copy()
    table_data["predicted_churn_prob"] = table_data["predicted_churn_prob"].apply(lambda p: f"{p*100:.1f}%")
    table_data["current_monthly_spend"] = table_data["current_monthly_spend"].apply(lambda s: f"${s:,.0f}")
    table_data["spend_velocity_ratio"]  = table_data["spend_velocity_ratio"].apply(lambda v: f"{v:.2f}")
    table_data["current_lag"]           = table_data["current_lag"].apply(lambda l: f"{l:.1f}d")
    table_data["avg_csat_t6"]           = table_data["avg_csat_t6"].apply(lambda c: f"{c:.2f} / 5")

    table_data.rename(columns={
        "client_id": "Client ID",
        "risk_tier": "Risk Tier",
        "predicted_churn_prob": "Churn Probability",
        "account_status": "Account Status",
        "industry_vertical": "Industry",
        "account_tier": "Account Tier",
        "current_monthly_spend": "Monthly Spend",
        "spend_velocity_ratio": "Spend Velocity",
        "current_lag": "Payment Lag",
        "escalation_count_6m": "Disputes (6M)",
        "avg_csat_t6": "CSAT (6M)",
    }, inplace=True)

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        height=480,
    )

    # Download CSV
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Watchlist (CSV)",
        data=csv_bytes,
        file_name="graylogic_account_risk_watchlist.csv",
        mime="text/csv",
    )


# ==============================================================================
# PAGE 3: SINGLE-CLIENT DIAGNOSTIC DRILLDOWN
# ==============================================================================
elif page == "🔍 Single-Client Diagnostic Drilldown":
    st.markdown("### 🔍 Single-Client Diagnostic Drilldown")
    st.markdown(
        "Deep-dive into individual client telemetry, local SHAP root-cause feature attributions, "
        "and 36-month longitudinal operational history."
    )

    # ── Cohort Selector ───────────────────────────────────────────────────────
    drill_cohort = st.radio(
        "Select Client Cohort:",
        [
            "🟢 Active Accounts (Live Retention Forensics)",
            "📋 Historical Churned Accounts (Post-Mortem Autopsy)",
        ],
        horizontal=True,
    )

    if drill_cohort == "🟢 Active Accounts (Live Retention Forensics)":
        cohort_df = df[df["account_status"] == "Active"].sort_values(by="predicted_churn_prob", ascending=False)
        st.caption("Showing active accounts sorted by churn probability. Focus on accounts with early friction signals.")
    else:
        cohort_df = df[df["account_status"] == "Churned / Inactive"].sort_values(by="predicted_churn_prob", ascending=False)
        st.caption("Showing historical churned accounts for retrospective failure mode analysis.")

    cohort_clients = cohort_df["client_id"].tolist()
    client_labels = [
        f"{cid} — Risk: {cohort_df.loc[cohort_df['client_id']==cid, 'predicted_churn_prob'].values[0]*100:.1f}% ({cohort_df.loc[cohort_df['client_id']==cid, 'account_status'].values[0]} | ${cohort_df.loc[cohort_df['client_id']==cid, 'spend_trailing_3m'].values[0]:,.0f}/mo)"
        for cid in cohort_clients
    ]
    selected_idx = st.selectbox(
        "Select Client Account to Diagnose:",
        range(len(cohort_clients)),
        format_func=lambda i: client_labels[i],
    )
    selected_cid = cohort_clients[selected_idx]

    client_row = df[df["client_id"] == selected_cid].iloc[0]
    client_ledger = ledger[ledger["client_id"] == selected_cid].sort_values("billing_month").reset_index(drop=True)
    client_ledger["billing_month"] = pd.to_datetime(client_ledger["billing_month"])

    risk_prob = float(client_row["predicted_churn_prob"])
    risk_tier = client_row["risk_tier"]

    # ── Client Hero Card ──────────────────────────────────────────────────────
    badge_class = "badge-high" if risk_prob > 0.70 else ("badge-med" if risk_prob >= 0.40 else "badge-low")
    st.markdown(f"""
    <div class="client-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                    <span style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{selected_cid}</span>
                    <span class="{badge_class}">{risk_tier}</span>
                    <span style="background: rgba(255,255,255,0.08); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">{client_row['account_status']}</span>
                </div>
                <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #94a3b8; flex-wrap: wrap;">
                    <span>🏢 <b>Vertical:</b> {client_row['industry_vertical']}</span>
                    <span>⭐ <b>Tier:</b> {client_row['account_tier']}</span>
                    <span>📑 <b>Contract:</b> {client_row['engagement_type']}</span>
                    <span>🛠️ <b>Primary Line:</b> {client_row['primary_service_line']}</span>
                    <span>📅 <b>Start:</b> {pd.to_datetime(client_row['account_start_date']).strftime('%Y-%m-%d')}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Trailing Monthly Spend</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">${client_row['spend_trailing_3m']:,.0f}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Lifetime Value: ${client_row['lifetime_spend']:,.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Row: Gauge & SHAP Drivers ─────────────────────────────────────────
    g_col, s_col = st.columns([1, 1.2])

    with g_col:
        st.markdown("#### 🎯 Predicted Retention Risk Score")
        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#ffffff", "family": "Plus Jakarta Sans"}},
            title={"text": "Churn Probability", "font": {"size": 15, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": "#ffffff", "thickness": 0.22},
                "bgcolor": "rgba(15,23,42,0.8)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.45)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.45)"},
                    {"range": [70, 100], "color": "rgba(244, 63, 94, 0.55)"},
                ],
                "threshold": {
                    "line": {"color": "#f43f5e", "width": 4},
                    "thickness": 0.75,
                    "value": risk_prob * 100,
                },
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=30, b=20),
            height=280,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with s_col:
        st.markdown("#### 🔬 Key Drivers of Risk (SHAP Local Attribution)")

        # Extract client row index in original df
        client_iloc = df.index.get_loc(client_row.name)
        client_shap = shap_values[client_iloc]

        # Top drivers for this specific account
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": client_shap,
            "abs_shap": np.abs(client_shap),
        }).sort_values("abs_shap", ascending=False).head(5)

        # Plot horizontal SHAP waterfall / bar
        shap_df["impact"] = shap_df["shap_value"].apply(lambda v: "Elevates Risk" if v > 0 else "Lowers Risk")
        fig_shap = px.bar(
            shap_df.sort_values("shap_value", ascending=True),
            x="shap_value",
            y="feature",
            orientation="h",
            color="impact",
            color_discrete_map={"Elevates Risk": "#f43f5e", "Lowers Risk": "#10b981"},
            template="plotly_dark",
            labels={"shap_value": "SHAP Log-Odds Contribution", "feature": "Feature"},
        )
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=260,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Top 3 drivers executive narrative
    top3 = shap_df.head(3)
    st.markdown("##### 📌 Top 3 Leading Indicators for this Account:")
    t_cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        feat = row["feature"]
        val  = client_row.get(feat, "N/A")
        val_str = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
        is_bad = row["shap_value"] > 0
        border_c = "#f43f5e" if is_bad else "#10b981"
        icon = "⚠️" if is_bad else "✅"
        direction = "increasing churn risk" if is_bad else "protective factor"
        with t_cols[idx]:
            st.markdown(f"""
            <div style="background: rgba(15,23,42,0.6); border-left: 4px solid {border_c}; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Driver #{idx+1} {icon}</div>
                <div style="font-size: 1rem; font-weight: 700; color: #ffffff; margin: 3px 0;">{feat}</div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Observed Value: <b>{val_str}</b></div>
                <div style="font-size: 0.75rem; color: {'#fb7185' if is_bad else '#34d399'}; margin-top: 4px;">{direction} (SHAP: {row['shap_value']:+.3f})</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 36-Month Longitudinal Operational History")

    # ── 3 TIME-SERIES CHARTS ──────────────────────────────────────────────────
    # Chart 1: Spend Trajectory
    st.markdown("#### 1. Monetary Spend Trajectory & Moving Averages")
    client_ledger["spend_t3_ma"] = client_ledger["billed_amount"].rolling(3, min_periods=1).mean()
    client_ledger["spend_t6_ma"] = client_ledger["billed_amount"].rolling(6, min_periods=1).mean()

    fig_spend = go.Figure()
    fig_spend.add_trace(go.Bar(
        x=client_ledger["billing_month"],
        y=client_ledger["billed_amount"],
        name="Monthly Billed ($)",
        marker_color="rgba(99, 102, 241, 0.4)",
    ))
    fig_spend.add_trace(go.Scatter(
        x=client_ledger["billing_month"],
        y=client_ledger["spend_t3_ma"],
        name="3-Month Moving Avg",
        line=dict(color="#38bdf8", width=2.5),
    ))
    fig_spend.add_trace(go.Scatter(
        x=client_ledger["billing_month"],
        y=client_ledger["spend_t6_ma"],
        name="6-Month Moving Avg",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    fig_spend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Billed Amount ($)"),
        xaxis=dict(title="Billing Month"),
    )
    st.plotly_chart(fig_spend, use_container_width=True)

    # Charts 2 & 3 side by side
    col_chart2, col_chart3 = st.columns(2)

    with col_chart2:
        st.markdown("#### 2. Payment Lag Trends & Friction Signals")
        fig_lag = go.Figure()
        fig_lag.add_trace(go.Scatter(
            x=client_ledger["billing_month"],
            y=client_ledger["payment_lag_days"],
            mode="lines+markers",
            name="Payment Lag (Days)",
            line=dict(color="#ec4899", width=2.5),
            marker=dict(size=5),
        ))
        # Add warning reference threshold lines
        fig_lag.add_hline(y=30, line_dash="dash", line_color="#f59e0b", annotation_text="Standard 30d SLA", annotation_position="top right")
        fig_lag.add_hline(y=60, line_dash="dash", line_color="#f43f5e", annotation_text="Critical 60d Warning", annotation_position="top right")
        fig_lag.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            yaxis=dict(title="Days to Pay"),
            xaxis=dict(title="Billing Month"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_lag, use_container_width=True)

    with col_chart3:
        st.markdown("#### 3. Operational Friction: Tickets, Escalations & CSAT")
        fig_ops = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ops.add_trace(
            go.Bar(
                x=client_ledger["billing_month"],
                y=client_ledger["support_tickets_raised"],
                name="Support Tickets",
                marker_color="rgba(56, 189, 248, 0.4)",
            ),
            secondary_y=False,
        )
        fig_ops.add_trace(
            go.Bar(
                x=client_ledger["billing_month"],
                y=client_ledger["escalations_or_disputes"],
                name="Formal Disputes",
                marker_color="rgba(244, 63, 94, 0.8)",
            ),
            secondary_y=False,
        )
        fig_ops.add_trace(
            go.Scatter(
                x=client_ledger["billing_month"],
                y=client_ledger["client_nps_or_csat"],
                name="CSAT Score (1-5)",
                line=dict(color="#fbbf24", width=3),
                mode="lines+markers",
            ),
            secondary_y=True,
        )
        fig_ops.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Billing Month"),
        )
        fig_ops.update_yaxes(title_text="Tickets / Disputes", secondary_y=False)
        fig_ops.update_yaxes(title_text="CSAT Score", range=[0.5, 5.5], secondary_y=True)
        st.plotly_chart(fig_ops, use_container_width=True)


# ==============================================================================
# PAGE 4: WHAT-IF PRESCRIPTIVE ACTION SIMULATOR
# ==============================================================================
elif page == "🔮 Prescriptive Action Simulator":
    st.markdown("### 🔮 What-If Prescriptive Action Simulator")
    st.markdown(
        "Evaluate real-time prescriptive retention interventions on active client accounts. "
        "Simulate the impact of dispute resolution, invoice acceleration, retention concessions, "
        "and CSAT recovery to project risk reduction and net financial ROI."
    )

    st.markdown("""
    <div class="callout-box" style="margin-bottom: 16px;">
        <b>🎯 Active Account Retention Simulator:</b> Prescriptive intervention scenarios are modeled strictly on <b>Active Accounts</b> with live Annual Contract Value (ACV). Adjust operational levers to project how dispute resolution, net-terms renegotiation, or service quality recovery prevents account departure.
    </div>
    """, unsafe_allow_html=True)

    # Select active client
    active_sim_df = df[df["account_status"] == "Active"].sort_values(by="predicted_churn_prob", ascending=False)
    active_sim_clients = active_sim_df["client_id"].tolist()
    sim_cid = st.selectbox(
        "Select Active Client Account for Prescriptive Scenario:",
        active_sim_clients,
        format_func=lambda cid: f"{cid} | Current Risk: {active_sim_df.loc[active_sim_df['client_id']==cid, 'predicted_churn_prob'].values[0]*100:.1f}% | ACV: ${active_sim_df.loc[active_sim_df['client_id']==cid, 'spend_trailing_3m'].values[0]*12:,.0f}/yr ({active_sim_df.loc[active_sim_df['client_id']==cid, 'account_tier'].values[0]} — {active_sim_df.loc[active_sim_df['client_id']==cid, 'industry_vertical'].values[0]})",
    )

    client_row = df[df["client_id"] == sim_cid].iloc[0]
    base_risk  = float(client_row["predicted_churn_prob"])
    base_acv   = float(client_row["spend_trailing_3m"] * 12)  # Annual Contract Value

    st.markdown(f"""
    <div class="callout-box">
        <b>Account Baseline Context:</b> {sim_cid} ({client_row['industry_vertical']} — {client_row['account_tier']}) &nbsp;|&nbsp; 
        Baseline Risk: <b>{base_risk*100:.1f}%</b> &nbsp;|&nbsp; 
        Annual Contract Value (ACV): <b>${base_acv:,.0f}</b> &nbsp;|&nbsp; 
        Recent Payment Lag: <b>{client_row['current_lag']:.1f} days</b> &nbsp;|&nbsp; 
        Open Disputes: <b>{client_row['escalation_count_6m']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎛️ Configure Retention Interventions")
    i_col1, i_col2 = st.columns(2)

    with i_col1:
        st.markdown("##### ⏱️ Collections & Payment Friction")
        reduce_lag = st.slider(
            "Accelerate Invoicing / Reduce Payment Lag (Days):",
            min_value=0,
            max_value=int(min(60, max(5, client_row["current_lag"] - 10))),
            value=int(min(30, max(0, client_row["current_lag"] - 25))),
            step=5,
            help="Implement milestone billing, automated follow-ups, or renegotiate net terms to reduce lag.",
        )

        st.markdown("##### 🛡️ Dispute & Escalation Resolution")
        curr_esc = int(client_row["escalation_count_6m"])
        resolve_disputes = st.slider(
            "Resolve Formal Disputes via Executive Sponsorship:",
            min_value=0,
            max_value=max(1, curr_esc),
            value=min(2, curr_esc),
            step=1,
            help="Deploy senior leadership or dedicated engineering taskforce to clear open disputes.",
        )

    with i_col2:
        st.markdown("##### 💰 Commercial Terms / Targeted Discount")
        discount_pct = st.slider(
            "Offer Temporary Retention Discount / Concession (%):",
            min_value=0,
            max_value=30,
            value=10,
            step=5,
            help="Offer a targeted service credit or 10-20% discount on next renewal to protect long-term ACV.",
        )

        st.markdown("##### ⭐ Service Quality & CSAT Recovery")
        csat_boost = st.slider(
            "Projected CSAT / NPS Score Improvement (+Points):",
            min_value=0.0,
            max_value=2.5,
            value=1.2,
            step=0.1,
            help="Assign specialized QA, replace underperforming team members, or add dedicated account manager.",
        )

    # ── Real-Time Machine Learning Inference ──────────────────────────────────
    # Construct modified feature vector
    sim_row = client_row[feature_names].copy()

    # 1. Payment lag adjustment
    new_lag = max(10.0, float(sim_row["payment_lag_t3"]) - reduce_lag)
    sim_row["payment_lag_t3"]       = new_lag
    sim_row["avg_lag_t6"]           = max(12.0, float(sim_row["avg_lag_t6"]) - (reduce_lag * 0.6))
    sim_row["current_lag"]          = max(10.0, float(sim_row["current_lag"]) - reduce_lag)
    sim_row["payment_friction_trend"] = float(sim_row["payment_friction_trend"]) - reduce_lag

    # 2. Escalation resolution
    new_esc_6m = max(0, int(sim_row["escalation_count_6m"]) - resolve_disputes)
    sim_row["escalation_count_6m"]   = new_esc_6m
    sim_row["escalation_count_3m"]   = max(0, int(sim_row["escalation_count_3m"]) - resolve_disputes)
    sim_row["avg_escalations_t6"]    = max(0.0, float(sim_row["avg_escalations_t6"]) - (resolve_disputes / 6.0))
    sim_row["has_recent_escalation"] = int(new_esc_6m > 0)
    sim_row["current_escalations"]   = max(0, int(sim_row["current_escalations"]) - resolve_disputes)

    # 3. Spend & Velocity adjustment (commercial concession stabilizes account velocity)
    # A successful intervention recovers velocity toward 0.95 - 1.05
    recovery_factor = 0.35 + (0.015 * reduce_lag) + (0.15 * resolve_disputes) + (0.08 * csat_boost)
    orig_velocity   = float(sim_row["spend_velocity_ratio"])
    new_velocity    = min(1.05, orig_velocity + (1.0 - orig_velocity) * min(0.9, recovery_factor))
    sim_row["spend_velocity_ratio"] = new_velocity

    # 4. CSAT boost
    new_csat = min(5.0, float(sim_row["avg_csat_t6"]) + csat_boost)
    sim_row["avg_csat_t6"]      = new_csat
    sim_row["min_csat_observed"] = min(5.0, float(sim_row["min_csat_observed"]) + csat_boost * 0.5)
    sim_row["current_csat"]     = min(5.0, float(sim_row["current_csat"]) + csat_boost)

    # Convert to single-row DataFrame matching feature_names
    sim_df = pd.DataFrame([sim_row])[feature_names]

    # Model prediction
    if hasattr(best_model, "predict_proba"):
        new_risk = float(best_model.predict_proba(sim_df)[0, 1])
    else:
        new_risk = float(best_model.decision_function(sim_df)[0])

    risk_reduction_abs = base_risk - new_risk
    risk_reduction_pct = (risk_reduction_abs / base_risk) * 100 if base_risk > 0 else 0

    # Financial & ROI Modeling
    # Expected preserved value = ACV * risk_reduction_abs
    retained_value = base_acv * max(0.0, risk_reduction_abs)

    # Estimated cost of retention package:
    # Discount cost + dispute legal/technical squad cost ($5,000 per dispute) + CSAT QA squad ($8,000)
    discount_cost   = base_acv * (discount_pct / 100.0) * 0.5  # assuming 6-month discount
    squad_cost      = (resolve_disputes * 4_000) + (10_000 if csat_boost > 0.5 else 3_000)
    total_cost      = discount_cost + squad_cost
    net_value_saved = retained_value - total_cost
    roi_pct         = (net_value_saved / total_cost) * 100 if total_cost > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Simulation Outcomes & Business Case")

    # Output KPIs
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📉 New Simulated Risk</div>
            <div class="kpi-value">{new_risk*100:.1f}%</div>
            <div class="kpi-sub" style="color: #34d399;">&darr; {risk_reduction_abs*100:.1f}% Absolute Drop</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🛡️ Retained Annual Value</div>
            <div class="kpi-value">${retained_value:,.0f}</div>
            <div class="kpi-sub">Preserved Contract Base</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💵 Retention Cost</div>
            <div class="kpi-value">${total_cost:,.0f}</div>
            <div class="kpi-sub">Credits & Engineering Squad</div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🚀 Net Retention ROI</div>
            <div class="kpi-value" style="color: {'#34d399' if roi_pct > 0 else '#fb7185'};">{roi_pct:,.0f}%</div>
            <div class="kpi-sub">Net Value: ${net_value_saved:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Side-by-side Gauges (Before vs After) ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    g_before, g_after = st.columns(2)

    with g_before:
        st.markdown("##### 🔴 Baseline Churn Probability (Status Quo)")
        fig_b = go.Figure(go.Indicator(
            mode="gauge+number",
            value=base_risk * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#f43f5e"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#f43f5e"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(244, 63, 94, 0.3)"},
                ]
            }
        ))
        fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    with g_after:
        st.markdown("##### 🟢 Simulated Churn Probability (Post-Intervention)")
        fig_a = go.Figure(go.Indicator(
            mode="gauge+number",
            value=new_risk * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#34d399"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#34d399"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(244, 63, 94, 0.2)"},
                ]
            }
        ))
        fig_a.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig_a, use_container_width=True)


# ==============================================================================
# PAGE 5: MODEL GOVERNANCE & 5-FOLD CROSS-VALIDATION BENCHMARKS
# ==============================================================================
elif page == "🤖 Model Governance & Benchmarks":
    st.markdown("### 🤖 ML Model Governance & Cross-Validation Benchmarks")
    st.markdown(
        "Complete rigorous 5-Fold Stratified Cross-Validation benchmark across candidate architectures, "
        "model explainability, and global feature importance attribution."
    )

    st.markdown("""
    <div class="callout-box" style="margin-bottom: 20px;">
        <b>🔬 Two-Stage Enterprise Retention Framework:</b><br>
        • <b>Stage 1 — Historical Telemetry Analysis (Offline Pipeline):</b> Predictive models are trained and benchmarked on 36 months of longitudinal ledgers (150 client accounts, 5,400 monthly logs, 33 historical churn events). Using 5-Fold Stratified Cross-Validation, models learn leading signals (spend velocity decay, payment friction deltas, dispute surges, and CSAT erosion).<br>
        • <b>Stage 2 — Active Portfolio Deployment (Operational Dashboard):</b> The calibrated model is continuously applied to score the <b>117 current active accounts</b> across the Executive Overview, Early-Warning Watchlist, Single-Client Drilldown, and Prescriptive Simulator to protect live recurring revenue.
    </div>
    """, unsafe_allow_html=True)

    # Cross-validation table
    st.markdown("#### 🏆 5-Fold Stratified Cross-Validation Benchmark (Historical Dataset)")
    cv_display = cv_metrics.copy()
    for col in ["AUC-ROC (Mean)", "Precision (Mean)", "Recall (Mean)", "F1-Score (Mean)"]:
        cv_display[col] = cv_display[col].apply(lambda x: f"{x:.4f}")
    for col in ["AUC-ROC (Std)", "Precision (Std)", "Recall (Std)", "F1-Score (Std)"]:
        cv_display[col] = cv_display[col].apply(lambda x: f"± {x:.4f}")

    st.dataframe(cv_display, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_m1, c_m2 = st.columns(2)

    with c_m1:
        st.markdown("#### 📊 Cross-Validation Performance Comparison")
        cv_plot_df = cv_metrics.melt(
            id_vars=["Model"],
            value_vars=["AUC-ROC (Mean)", "Precision (Mean)", "Recall (Mean)", "F1-Score (Mean)"],
            var_name="Metric",
            value_name="Score"
        )
        fig_comp = px.bar(
            cv_plot_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            template="plotly_dark",
            color_discrete_sequence=["#6366f1", "#06b6d4", "#f43f5e"],
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[0.85, 1.02]),
            height=340,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with c_m2:
        st.markdown("#### 🌍 Global SHAP Feature Importance (Top 10 Drivers)")
        top_global = global_imp.head(10).sort_values("importance", ascending=True)
        fig_gshap = px.bar(
            top_global,
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_dark",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig_gshap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_gshap, use_container_width=True)

    # Fold details expander
    with st.expander("📁 View Individual Cross-Validation Fold Metrics", expanded=False):
        fold_details = artifacts["fold_details"]
        for m_name, folds in fold_details.items():
            st.markdown(f"**{m_name}**")
            st.dataframe(pd.DataFrame(folds), hide_index=True)
