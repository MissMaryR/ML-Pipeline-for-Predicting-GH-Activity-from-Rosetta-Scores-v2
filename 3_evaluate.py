#!/usr/bin/env python3
"""
3_evaluate.py
-------------
- Loads all 4 trained models and their metadata automatically
- For each model produces:
    - PR curve (full dataset + test set)
    - ROC curve (full dataset + test set)
    - Confusion matrix (full dataset + test set)
    - Predicted probability distribution (full dataset + test set)
- All plots saved to output/evaluate/<model_name>/full/ and /test/
"""

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
CSV_SPLIT = 'Book1_aggregated_with_split.csv'
BASE_OUTDIR = 'output/evaluate'

# ============================================================
# LOAD SPLIT DATA
# ============================================================
print("\n📂 Loading split data...")
df = pd.read_csv(CSV_SPLIT)
df['split'] = df['split'].astype(str).str.lower()
df['active'] = df['active'].astype(int)

df_train = df[df['split'] == 'train'].copy()
df_test  = df[df['split'] == 'test'].copy()

print(f"  Full n={len(df)} | Train n={len(df_train)} | Test n={len(df_test)}")

# ============================================================
# FIND ALL TRAINED MODELS
# ============================================================
model_files = sorted(glob.glob('model_*.joblib'))
if not model_files:
    print("  ❌ No model_*.joblib files found.")
    print("  Run 1a, 1b, 1c, 1d first.")
    exit(1)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_model_metadata(model_path):
    """Derive features/threshold/metrics paths from model path."""
    suffix = model_path.replace('model_', '').replace('.joblib', '')
    features_path  = f'features_{suffix}.txt'
    threshold_path = f'threshold_{suffix}.txt'
    metrics_path   = f'metrics_{suffix}.json'

    with open(features_path) as f:
        features = [line.strip() for line in f if line.strip()]
    with open(threshold_path) as f:
        threshold = float(f.read().strip())
    with open(metrics_path) as f:
        metrics = json.load(f)

    return features, threshold, metrics

def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + '.png', dpi=300)
    fig.savefig(path + '.pdf')
    plt.close(fig)

def plot_pr_curve(y_true, y_prob, threshold, title, outpath):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()

    # Find operating point
    if len(thresholds) > 0:
        i = int(np.argmin(np.abs(thresholds - threshold)))
        p_thr, r_thr = precision[i], recall[i]
    else:
        p_thr, r_thr = precision[-1], recall[-1]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
    ax.hlines(prevalence, 0, 1, linestyles='dashed', linewidth=1.5,
              label=f'Baseline (prevalence = {prevalence:.2f})')
    ax.scatter([r_thr], [p_thr], s=80, zorder=3,
               label=f'Threshold = {threshold:.3f}\nP={p_thr:.3f}, R={r_thr:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.02, 0.02,
            f"n={len(y_true)} (pos={int(y_true.sum())}, neg={int((1-y_true).sum())})",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom')
    plt.tight_layout()
    save(fig, outpath)

def plot_roc_curve(y_true, y_prob, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    save(fig, outpath)

def plot_confusion_matrix(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inactive', 'Active'])

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    save(fig, outpath)

def plot_prob_distribution(df_data, y_prob_col, threshold, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df_data.loc[df_data['active'] == 0, y_prob_col],
        bins=30, color='steelblue', label='True Inactive',
        kde=True, stat='density', alpha=0.5, ax=ax
    )
    sns.histplot(
        df_data.loc[df_data['active'] == 1, y_prob_col],
        bins=30, color='darkorange', label='True Active',
        kde=True, stat='density', alpha=0.5, ax=ax
    )
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.3f}')
    ax.set_xlabel('Predicted Probability of Active')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    save(fig, outpath)

def print_metrics(y_true, y_pred, y_prob, label):
    print(f"    {label}:")
    print(f"      Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"      Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"      F1:        {f1_score(y_true, y_pred):.3f}")
    print(f"      PR-AUC:    {average_precision_score(y_true, y_prob):.3f}")

# ============================================================
# EVALUATE EACH MODEL
# ============================================================
for model_path in model_files:
    suffix = model_path.replace('model_', '').replace('.joblib', '')
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating: {suffix}")
    print(f"{'='*60}")

    # Load model and metadata
    try:
        model = joblib.load(model_path)
        features, threshold, metrics = load_model_metadata(model_path)
    except FileNotFoundError as e:
        print(f"  ⚠️  Skipping — missing file: {e}")
        continue

    model_name = metrics['model']
    print(f"  Model: {model_name} | Features: {len(features)} | Threshold: {threshold:.4f}")

    outdir_full = os.path.join(BASE_OUTDIR, suffix, 'full')
    outdir_test = os.path.join(BASE_OUTDIR, suffix, 'test')
    os.makedirs(outdir_full, exist_ok=True)
    os.makedirs(outdir_test, exist_ok=True)

    # --- Check all features exist ---
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features in CSV: {missing} — skipping.")
        continue

    # --- Predictions ---
    X_full  = df[features]
    y_full  = df['active'].values
    y_prob_full = model.predict_proba(X_full)[:, 1]
    y_pred_full = (y_prob_full >= threshold).astype(int)
    df['pred_prob'] = y_prob_full

    X_test  = df_test[features]
    y_test  = df_test['active'].values
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)
    df_test_copy = df_test.copy()
    df_test_copy['pred_prob'] = y_prob_test

    print_metrics(y_full, y_pred_full, y_prob_full, 'Full dataset')
    print_metrics(y_test, y_pred_test, y_prob_test, 'Test set')

    # --- PR Curves ---
    plot_pr_curve(
        y_full, y_prob_full, threshold,
        f'{model_name} — PR Curve (Full Dataset)',
        os.path.join(outdir_full, 'pr_curve')
    )
    plot_pr_curve(
        y_test, y_prob_test, threshold,
        f'{model_name} — PR Curve (Test Set)',
        os.path.join(outdir_test, 'pr_curve')
    )

    # --- ROC Curves ---
    plot_roc_curve(
        y_full, y_prob_full,
        f'{model_name} — ROC Curve (Full Dataset)',
        os.path.join(outdir_full, 'roc_curve')
    )
    plot_roc_curve(
        y_test, y_prob_test,
        f'{model_name} — ROC Curve (Test Set)',
        os.path.join(outdir_test, 'roc_curve')
    )

    # --- Confusion Matrices ---
    plot_confusion_matrix(
        y_full, y_pred_full,
        f'{model_name} — Confusion Matrix (Full Dataset)',
        os.path.join(outdir_full, 'confusion_matrix')
    )
    plot_confusion_matrix(
        y_test, y_pred_test,
        f'{model_name} — Confusion Matrix (Test Set)',
        os.path.join(outdir_test, 'confusion_matrix')
    )

    # --- Probability Distributions ---
    plot_prob_distribution(
        df, 'pred_prob', threshold,
        f'{model_name} — Predicted Probabilities (Full Dataset)',
        os.path.join(outdir_full, 'prob_distribution')
    )
    plot_prob_distribution(
        df_test_copy, 'pred_prob', threshold,
        f'{model_name} — Predicted Probabilities (Test Set)',
        os.path.join(outdir_test, 'prob_distribution')
    )

    print(f"  ✅ All plots saved to output/evaluate/{suffix}/")

print(f"\n🎉 Done! All models evaluated.")
print(f"   Plots saved to: {BASE_OUTDIR}/")
print(f"   Run 4_feature_plots.py next.")
