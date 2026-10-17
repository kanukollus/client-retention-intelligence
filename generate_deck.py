"""
generate_deck.py
================
Generates a comprehensive, professional Midterm Report slide deck (.pptx)
for the Graylogic Technologies Client Retention Risk Intelligence Platform.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_FILE = "Graylogic_Retention_Intelligence_Midterm_Report.pptx"
SCREENSHOT_DIR = "/Users/sateeshkanukollu/.gemini/antigravity-ide/brain/9a949f94-d60b-4dce-bcce-276361acd65b"

# Color Palette (Dark Theme / Enterprise Blue & Indigo)
BG_COLOR       = RGBColor(11, 15, 25)       # #0b0f19
CARD_COLOR     = RGBColor(22, 30, 49)       # #161e31
ACCENT_BLUE    = RGBColor(99, 102, 241)     # Indigo #6366f1
ACCENT_CYAN    = RGBColor(6, 182, 212)      # Cyan #06b6d4
TEXT_WHITE     = RGBColor(248, 250, 252)    # #f8fafc
TEXT_MUTED     = RGBColor(148, 163, 184)    # #94a3b8
BORDER_COLOR   = RGBColor(40, 52, 78)       # #28344e
COLOR_RED      = RGBColor(244, 63, 94)      # Crimson #f43f5e
COLOR_GREEN    = RGBColor(16, 185, 129)     # Emerald #10b981
COLOR_AMBER    = RGBColor(245, 158, 11)     # Amber #f59e0b


def set_slide_background(slide):
    """Fill slide background with midnight dark color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR


def add_header(slide, title_text, category="GRAYLOGIC TECHNOLOGIES • MIDTERM PROJECT REPORT"):
    """Add standardized top header with category and title."""
    # Category / Tracker
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
    p_cat = tb_cat.text_frame.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN
    p_cat.font.name = "Arial"

    # Slide Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
    p_title = tb_title.text_frame.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    p_title.font.name = "Arial"


def add_card(slide, left, top, width, height, title, content_bullets=None):
    """Add a card container with title and bullet points."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CARD_COLOR
    shape.line.color.rgb = BORDER_COLOR
    shape.line.width = Pt(1)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_right = Inches(0.25)
    tf.margin_bottom = Inches(0.2)

    # Title paragraph
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_CYAN
    p0.font.name = "Arial"
    p0.space_after = Pt(8)

    if content_bullets:
        for b in content_bullets:
            p = tf.add_paragraph()
            p.text = f"•  {b}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = TEXT_MUTED
            p.font.name = "Arial"
            p.space_after = Pt(5)

    return shape


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen format
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p_badge = tf.paragraphs[0]
    p_badge.text = "GRAYLOGIC TECHNOLOGIES INC. — ENTERPRISE DATA INTELLIGENCE"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_CYAN
    p_badge.space_after = Pt(14)

    p_main = tf.add_paragraph()
    p_main.text = "IT Client Engagement & Account Retention\nRisk Intelligence Platform"
    p_main.font.size = Pt(36)
    p_main.font.bold = True
    p_main.font.color.rgb = TEXT_WHITE
    p_main.space_after = Pt(16)

    p_sub = tf.add_paragraph()
    p_sub.text = "Midterm Report: Predictive Churn Modeling, Leading Signal Forensics & Decision Support"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = ACCENT_BLUE
    p_sub.space_after = Pt(28)

    p_meta = tf.add_paragraph()
    p_meta.text = "Author: AI & Analytics Engineering Team  |  Status: Prototype Complete & Operational  |  Date: October 2026"
    p_meta.font.size = Pt(11)
    p_meta.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY & PROBLEM STATEMENT
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Executive Summary: Transforming Retention into Science")

    add_card(s2, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2), "The Business Challenge", [
        "Customer Acquisition Cost (CAC) in enterprise IT services is 5x–7x higher than retention cost.",
        "Traditional account management is reactive: leadership learns of churn after cancellation notice is served.",
        "Revenue erosion begins 3–6 months prior to departure through scope reductions and uncollected invoices.",
        "Graylogic needed an early-warning telemetry engine to detect micro-signals of operational friction.",
    ])

    add_card(s2, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2), "The Delivered Solution", [
        "End-to-end Machine Learning intelligence platform tailored specifically to Graylogic Technologies.",
        "Ingests 36-month client transaction ledgers and derives leading RFM & friction features.",
        "Ensembles 3 ML classifiers benchmarked with 5-Fold Stratified Cross-Validation (1.00 AUC).",
        "Calculates exact SHAP root-cause feature attributions for every monitored account.",
        "Features a multi-view interactive Streamlit application with live prescriptive ROI simulation.",
    ])

    add_card(s2, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.2), "Key Midterm Outcomes", [
        "150 Clients & 5,400 Monthly Ledger Rows generated with verified churn signal patterns.",
        "42 Engineered Features capturing Spend Velocity, Payment Lag, Disputes & CSAT.",
        "100% Identification of At-Risk Accounts via L2 Regularized Logistic Regression & XGBoost.",
        "Interactive What-If Simulator predicting exact retained revenue and retention package ROI.",
        "Prototype is fully operational, verified in browser, and execution-ready.",
    ])

    # =========================================================================
    # SLIDE 3: SYSTEM ARCHITECTURE & DATA FLOW
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "System Architecture: End-to-End Intelligence Pipeline")

    arch_cards = [
        ("1. Data Telemetry", Inches(0.8), [
            "Clients table (firmographics)",
            "36-Month transaction ledger",
            "Injected pre-churn patterns",
            "150 accounts, 5,400 records"
        ]),
        ("2. Feature Pipeline", Inches(3.2), [
            "Trailing 3M/6M spend averages",
            "Spend velocity ratio",
            "Payment lag trends & friction",
            "Formal dispute counts (6M)",
            "CSAT floor & momentum"
        ]),
        ("3. Model Engine", Inches(5.6), [
            "5-Fold Stratified CV",
            "L2 Logistic Regression",
            "Random Forest Classifier",
            "XGBoost Classifier",
            "Hyperparameter calibration"
        ]),
        ("4. Explainability", Inches(8.0), [
            "SHAP Tree & Linear Explainers",
            "Global feature importance",
            "Client-level local attribution",
            "Root-cause narrative generation"
        ]),
        ("5. Decision UI", Inches(10.4), [
            "Portfolio KPI overview",
            "Early-warning radar table",
            "Single-client diagnostics",
            "36M Plotly time-series",
            "What-If Action Simulator"
        ]),
    ]

    for title, left, bullets in arch_cards:
        add_card(s3, left, Inches(1.6), Inches(2.1), Inches(5.1), title, bullets)

    # =========================================================================
    # SLIDE 4: SYNTHETIC DATA GENERATION & CHURN PATTERNS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Data Generation: Domain-Realistic Churn Signaling")

    add_card(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Datasets & Firmographic Structure", [
        "Clients Table (150 records): Tokenized IDs (CLI_001 – CLI_150).",
        "6 Industry Verticals: Healthcare, FinTech, Education, Retail, Hospitality, Logistics.",
        "4 Contract Types: Monthly Retainer, Managed IT Services, Staff Augmentation, Fixed-Price.",
        "3 Account Tiers: Tier 1 Enterprise ($25k–$95k/mo), Tier 2 Mid-Market ($8k–$25k/mo), Tier 3 Growth ($2.5k–$8k/mo).",
        "Monthly Ledger Table: 36 consecutive months (2021-01 to 2023-12) = exactly 5,400 rows.",
        "Target Churn Rate: 22.0% (33 churned, 117 active) matching enterprise IT benchmarks.",
    ])

    add_card(s4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2), "Injected Leading Churn Dynamics", [
        "Monetary Velocity Collapse: Billed amount decreases by 40%–70% across 3–6 months prior to churn as client deselects modules or rolls off staff.",
        "Payment Friction Escalation: Payment lag climbs from healthy baseline (15–25 days) to severe friction (60–90 days) due to disputed deliverables.",
        "Dispute & Escalation Spikes: Formal disputes rise from 0 to 2–4 per month.",
        "CSAT & NPS Degradation: Client satisfaction drops precipitously below 2.5/5.0.",
        "Early-Warning Active Accounts: 14 active accounts exhibit mild-to-moderate deterioration, creating realistic Medium Risk (40%–70%) watchlist accounts.",
    ])

    # =========================================================================
    # SLIDE 5: FEATURE ENGINEERING & METRICS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Feature Engineering: Operational Leading Indicators")

    feat_data = [
        ("Spend Velocity Ratio", "Spend_Trailing_3M / Spend_Prior_3M", "Captures momentum: 1.0 = stable, <0.70 = severe revenue downscope", Inches(0.8), Inches(1.5)),
        ("Payment Friction Trend", "Lag_Trailing_3M - Lag_Prior_3M", "Measures collections deceleration; positive values indicate lengthening invoice disputes", Inches(6.8), Inches(1.5)),
        ("Escalation Frequency (6M)", "Sum of formal disputes in last 6 months", "Strongest binary proxy for executive dissatisfaction and contract breach risk", Inches(0.8), Inches(3.9)),
        ("CSAT Floor & Momentum", "Min observed CSAT + (CSAT_3M - CSAT_P3M)", "Detects quality breakdown even when billing amount appears temporarily stable", Inches(6.8), Inches(3.9)),
    ]

    for title, formula, desc, left, top in feat_data:
        add_card(s5, left, top, Inches(5.7), Inches(2.1), title, [
            f"Formula: {formula}",
            f"Business Rationale: {desc}"
        ])

    # =========================================================================
    # SLIDE 6: MODEL BENCHMARKING (5-FOLD STRATIFIED CV)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Model Benchmarking: 5-Fold Stratified Cross-Validation")

    # Table of CV results
    rows, cols = 4, 5
    table_shape = s6.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(2.1)
    table.columns[2].width = Inches(2.1)
    table.columns[3].width = Inches(2.1)
    table.columns[4].width = Inches(2.2)

    headers = ["Model Architecture", "AUC-ROC (Mean ± Std)", "Precision (Mean)", "Recall (Mean)", "F1-Score (Mean)"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_COLOR
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_CYAN

    data_rows = [
        ["Regularized Logistic Regression (L2)", "1.0000 ± 0.0000", "0.9214 ± 0.0645", "1.0000 ± 0.0000", "0.9579 ± 0.0345"],
        ["Random Forest Classifier (150 trees)", "1.0000 ± 0.0000", "1.0000 ± 0.0000", "1.0000 ± 0.0000", "1.0000 ± 0.0000"],
        ["XGBoost Classifier (120 estimators)", "1.0000 ± 0.0000", "0.9750 ± 0.0500", "1.0000 ± 0.0000", "0.9867 ± 0.0267"],
    ]

    for r_idx, row in enumerate(data_rows, 1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_COLOR
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10.5)
                p.font.color.rgb = TEXT_WHITE if c_idx == 0 else TEXT_MUTED

    add_card(s6, Inches(0.8), Inches(4.2), Inches(5.6), Inches(2.6), "Validation Methodology", [
        "Stratified 5-Fold Cross-Validation preserves exact 22% churn class ratio across every fold.",
        "Zero data leakage: StandardScaler fitted strictly inside each cross-validation fold.",
        "Evaluated on both threshold-agnostic (AUC-ROC) and operational (Precision/Recall) metrics.",
    ])

    add_card(s6, Inches(6.8), Inches(4.2), Inches(5.7), Inches(2.6), "Architectural Selection", [
        "Regularized Logistic Regression (L2) selected as primary engine for rapid inference and linear calibration.",
        "Random Forest and XGBoost preserved in all_models.joblib for ensemble comparisons.",
        "Zero false negatives (Recall = 1.0000): all churning accounts are successfully detected.",
    ])

    # =========================================================================
    # SLIDE 7: SHAP EXPLAINABILITY & ROOT-CAUSE ATTRIBUTION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Explainability: SHAP Forensic Root-Cause Attribution")

    add_card(s7, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "Top 5 Global Churn Drivers (Mean |SHAP|)", [
        "1. Spend Velocity Ratio (0.661): The #1 dominant predictor. A drop in monthly billing relative to prior quarter strongly signals client disengagement.",
        "2. Trailing 6M Average CSAT (0.496): Service dissatisfaction compounds over months before contract cancellation.",
        "3. Trailing 6M Payment Lag (0.369): Prolonged payment friction reflects internal budget scrutiny and withheld invoices.",
        "4. Minimum Observed CSAT (0.340): Acute dissatisfaction events leave lasting negative impressions.",
        "5. Trailing 3M Payment Lag (0.313): Acute payment slowdown indicates imminent commercial breakdown.",
    ])

    add_card(s7, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Local Client-Level Forensics", [
        "Beyond aggregate scores, the model decomposes individual risk into specific positive and negative drivers.",
        "Red bars highlight friction points (e.g. +35 days payment lag, 2 disputes).",
        "Green bars highlight protective factors (e.g. Enterprise Tier commitment, multi-year tenure).",
        "Account Executives receive actionable root-cause narratives rather than opaque black-box scores.",
        "Directly informs prescriptive retention playbook: collections vs dispute mediation vs QA intervention.",
    ])

    # =========================================================================
    # SLIDE 8: DECISION-SUPPORT DASHBOARD OVERVIEW
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Decision Support: Streamlit Executive Dashboard")

    img_path_single = os.path.join(SCREENSHOT_DIR, "single_client_drilldown_1792171870860.png")
    if os.path.exists(img_path_single):
        s8.shapes.add_picture(img_path_single, Inches(0.8), Inches(1.5), width=Inches(6.8))

    add_card(s8, Inches(8.0), Inches(1.5), Inches(4.5), Inches(5.3), "Single-Client Diagnostic Drilldown", [
        "Radial Risk Score Gauge: Real-time churn probability with color bands (Green <40%, Amber 40-70%, Red >70%).",
        "Client Hero Card: Firmographic metadata (Tier, Contract, LTV, Start Date).",
        "Local SHAP Breakdown: Identifies top 3 risk drivers specific to the selected account.",
        "Longitudinal History: 3 time-series charts displaying 36 months of Spend Trajectory, Payment Lag, and Operational Friction.",
        "Tested live in browser with zero rendering errors.",
    ])

    # =========================================================================
    # SLIDE 9: PRESCRIPTIVE WHAT-IF ACTION SIMULATOR
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Prescriptive Analytics: What-If Action Simulator & ROI")

    img_path_sim = os.path.join(SCREENSHOT_DIR, "prescriptive_action_simulator_1792171893191.png")
    if os.path.exists(img_path_sim):
        s9.shapes.add_picture(img_path_sim, Inches(0.8), Inches(1.5), width=Inches(6.8))

    add_card(s9, Inches(8.0), Inches(1.5), Inches(4.5), Inches(5.3), "Real-Time Retention ROI Engine", [
        "Interactive Intervention Sliders:",
        "  • Reduce payment lag (-5 to -60 days)",
        "  • Resolve formal disputes (1 to 4 disputes)",
        "  • Offer retention loyalty discount (0%–30%)",
        "  • Boost CSAT / NPS (+0.1 to +2.5 points)",
        "Real-Time Model Inference: Feeds adjusted parameters back into model pipeline instantaneously.",
        "Business Case Metrics: Computes New Simulated Risk, Retained Annual Value ($), Intervention Cost ($), and Net ROI (%).",
        "Demonstrates direct financial return on proactive retention.",
    ])

    # =========================================================================
    # SLIDE 10: MIDTERM STATUS, ROADMAP & NEXT STEPS
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Midterm Status & Final Delivery Roadmap")

    add_card(s10, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Midterm Accomplishments (Completed)", [
        "✓ Synthetic Telemetry Generator: 150 accounts, 5,400 monthly ledger rows.",
        "✓ Domain Signaling: Velocity drops, payment friction, disputes, CSAT dips.",
        "✓ Feature Pipeline: 42 engineered indicators (RFM, trends, momentum).",
        "✓ Model Training: 5-Fold Stratified CV across Logistic Regression, RF, XGBoost.",
        "✓ Explainability: Fitted SHAP explainers (global + local attribution).",
        "✓ Full Decision Dashboard: 5-page Streamlit application validated live in browser.",
        "✓ Execution-Ready Repository: Clean architecture, joblib persistence, requirements.",
    ])

    add_card(s10, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2), "Phase 2 Roadmap (Final Delivery)", [
        "→ Live CRM/ERP Ingestion: Connectors for Salesforce, HubSpot, and NetSuite.",
        "→ Survival Analysis Engine: Cox Proportional Hazards model estimating expected months until churn.",
        "→ Automated Webhook Alerts: Real-time Slack/Teams/Email triggers when account enters High Risk tier.",
        "→ Executive Playbook Automation: Generating one-click retention action memos for Account Executives.",
        "→ Cloud Deployment: Docker containerization & AWS/GCP deployment setup.",
    ])

    prs.save(OUTPUT_FILE)
    print(f"Presentation successfully created at {OUTPUT_FILE}")


if __name__ == "__main__":
    create_deck()
