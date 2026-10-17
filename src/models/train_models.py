"""
src/models/train_models.py
===========================
Graylogic Technologies — IT Client Retention Risk Intelligence
Model Training, 5-Fold Stratified Cross-Validation, and SHAP Explainability Pipeline

Models Evaluated:
  1. Regularized Logistic Regression (L2 penalty with StandardScaler)
  2. Random Forest Classifier
  3. XGBoost Classifier

Outputs Saved to `models_artifacts/`:
  • best_model.joblib
  • all_models.joblib
  • cv_metrics.joblib
  • shap_explainer.joblib
  • shap_values.joblib
  • feature_names.joblib
  • metadata.joblib
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Disable numba jit warning noise if any
os.environ["NUMBA_DISABLE_JIT"] = "1"
import shap

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_processing import load_and_engineer, get_feature_columns

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "models_artifacts"
SEED = 42


def get_candidate_models() -> dict:
    """Return dictionary of the 3 candidate models."""
    return {
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=0.15,
                solver="lbfgs",
                max_iter=1000,
                random_state=SEED,
                class_weight="balanced"
            )),
        ]),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            class_weight="balanced",
            random_state=SEED,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=SEED,
        ),
    }


def evaluate_5fold_cv(models: dict, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Perform 5-Fold Stratified Cross-Validation reporting:
      - AUC-ROC
      - Precision
      - Recall
      - F1-Score
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results_records = []
    fold_details = {name: [] for name in models}

    for model_name, model in models.items():
        aucs, precisions, recalls, f1s = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit clone/instance on fold
            model.fit(X_train, y_train)

            # Predict probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.decision_function(X_val)

            y_pred = (y_prob >= 0.5).astype(int)

            auc  = roc_auc_score(y_val, y_prob)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec  = recall_score(y_val, y_pred, zero_division=0)
            f1   = f1_score(y_val, y_pred, zero_division=0)

            aucs.append(auc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            fold_details[model_name].append({
                "fold": fold,
                "auc_roc": auc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            })

        results_records.append({
            "Model": model_name,
            "AUC-ROC (Mean)": np.mean(aucs),
            "AUC-ROC (Std)":  np.std(aucs),
            "Precision (Mean)": np.mean(precisions),
            "Precision (Std)":  np.std(precisions),
            "Recall (Mean)":    np.mean(recalls),
            "Recall (Std)":     np.std(recalls),
            "F1-Score (Mean)":  np.mean(f1s),
            "F1-Score (Std)":   np.std(f1s),
        })

    cv_summary_df = pd.DataFrame(results_records).sort_values(by="AUC-ROC (Mean)", ascending=False)
    return cv_summary_df, fold_details


def compute_shap_explainability(best_model, X: pd.DataFrame, feature_names: list) -> tuple:
    """
    Calculate SHAP feature importance values for the top-performing model.
    Handles XGBoost, RandomForest, and Pipeline models.
    """
    print("Calculating SHAP feature importance values …")

    # Extract underlying estimator if model is inside a Pipeline
    estimator = best_model
    if isinstance(best_model, Pipeline):
        # Transform X through the pipeline's preprocessing steps
        if len(best_model.steps) > 1:
            preprocessor = Pipeline(best_model.steps[:-1])
            X_trans = preprocessor.transform(X)
            estimator = best_model.steps[-1][1]
            X_eval = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
        else:
            estimator = best_model.steps[-1][1]
            X_eval = X
    else:
        X_eval = X

    # Choose explainer based on model family
    if isinstance(estimator, (XGBClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(estimator)
        raw_shap_values = explainer.shap_values(X_eval)
        # Random forest returns list of [neg_class, pos_class]
        if isinstance(raw_shap_values, list) and len(raw_shap_values) == 2:
            shap_matrix = raw_shap_values[1]
        else:
            shap_matrix = raw_shap_values
    else:
        # Linear or general explainer
        explainer = shap.Explainer(estimator, X_eval)
        explanation = explainer(X_eval)
        shap_matrix = explanation.values

    # Mean absolute SHAP per feature (global importance)
    global_importance = np.abs(shap_matrix).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": global_importance,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("  ✓ SHAP calculations complete. Top 5 drivers:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    • {row['feature']:<28}: {row['importance']:.4f}")

    return explainer, shap_matrix, importance_df


def train_and_export_pipeline():
    """Execute complete training, evaluation, SHAP explainability, and export."""
    print("Graylogic Technologies — Model Training Pipeline")
    print("=" * 56)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load engineered features
    print("Loading data and engineering features …")
    processed_df, raw_ledger = load_and_engineer()
    feature_names = get_feature_columns(processed_df)

    X = processed_df[feature_names].copy()
    y = processed_df["churned"].copy()

    print(f"  Dataset: {len(X)} clients, {len(feature_names)} features")
    print(f"  Target: {y.sum()} Churned, {len(y) - y.sum()} Active ({y.mean():.1%} churn rate)")

    # 2. 5-Fold Stratified Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation across candidate models …")
    candidate_models = get_candidate_models()
    cv_summary, fold_details = evaluate_5fold_cv(candidate_models, X, y)

    print("\n" + "=" * 65)
    print("5-FOLD CROSS-VALIDATION BENCHMARK RESULTS")
    print("=" * 65)
    for _, row in cv_summary.iterrows():
        print(f"\nModel: {row['Model']}")
        print(f"  AUC-ROC:   {row['AUC-ROC (Mean)']:.4f} ± {row['AUC-ROC (Std)']:.4f}")
        print(f"  Precision: {row['Precision (Mean)']:.4f} ± {row['Precision (Std)']:.4f}")
        print(f"  Recall:    {row['Recall (Mean)']:.4f} ± {row['Recall (Std)']:.4f}")
        print(f"  F1-Score:  {row['F1-Score (Mean)']:.4f} ± {row['F1-Score (Std)']:.4f}")

    best_model_name = cv_summary.iloc[0]["Model"]
    print("\n" + "=" * 65)
    print(f"🏆 Top-Performing Model: {best_model_name}")
    print("=" * 65)

    # 3. Retrain all models on full data, save top-performing model
    print(f"\nFitting all models on full dataset …")
    fitted_models = {}
    for name, model in candidate_models.items():
        model.fit(X, y)
        fitted_models[name] = model

    best_model = fitted_models[best_model_name]

    # Predict probabilities for all accounts
    if hasattr(best_model, "predict_proba"):
        all_probs = best_model.predict_proba(X)[:, 1]
    else:
        all_probs = best_model.decision_function(X)

    processed_df["predicted_churn_prob"] = np.round(all_probs, 4)
    processed_df["risk_tier"] = pd.cut(
        processed_df["predicted_churn_prob"],
        bins=[-0.01, 0.40, 0.70, 1.01],
        labels=["Low (<40%)", "Medium (40-70%)", "High (>70%)"]
    )

    # 4. SHAP Explainability
    explainer, shap_matrix, importance_df = compute_shap_explainability(
        best_model, X, feature_names
    )

    # 5. Export artifacts to disk using joblib
    print(f"\nPersisting artifacts to `{ARTIFACTS_DIR}/` …")

    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
    joblib.dump(fitted_models, os.path.join(ARTIFACTS_DIR, "all_models.joblib"))
    joblib.dump(cv_summary, os.path.join(ARTIFACTS_DIR, "cv_metrics.joblib"))
    joblib.dump(fold_details, os.path.join(ARTIFACTS_DIR, "fold_details.joblib"))
    joblib.dump(explainer, os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
    joblib.dump(shap_matrix, os.path.join(ARTIFACTS_DIR, "shap_values.joblib"))
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
    joblib.dump(importance_df, os.path.join(ARTIFACTS_DIR, "global_importance.joblib"))
    joblib.dump(processed_df, os.path.join(ARTIFACTS_DIR, "processed_clients_df.joblib"))
    joblib.dump(raw_ledger, os.path.join(ARTIFACTS_DIR, "raw_ledger.joblib"))

    metadata = {
        "best_model_name": best_model_name,
        "n_clients": len(processed_df),
        "n_features": len(feature_names),
        "features": feature_names,
        "cv_summary": cv_summary.to_dict(orient="records"),
        "best_metrics": cv_summary.iloc[0].to_dict(),
    }
    joblib.dump(metadata, os.path.join(ARTIFACTS_DIR, "metadata.joblib"))

    print("✅ All artifacts saved successfully:")
    for f in os.listdir(ARTIFACTS_DIR):
        print(f"   • {ARTIFACTS_DIR}/{f}")


if __name__ == "__main__":
    train_and_export_pipeline()
