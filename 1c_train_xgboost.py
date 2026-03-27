#!/usr/bin/env python3
"""
1c_train_xgboost.py
-------------------
- Reads Book1_aggregated_with_split.csv (created by 1a_train_logreg.py)
- Ranks features using XGBoost gain-based feature importances
- Evaluates models adding one feature at a time
- Prompts you to choose how many features to use
- Saves model, features, threshold, and metrics
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, average_precision_score
)
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
CSV_SPLIT       = 'Book1_aggregated_with_split.csv'
MODEL_OUT       = 'model_xgboost.joblib'
FEATURES_OUT    = 'features_xgboost.txt'
THRESHOLD_OUT   = 'threshold_xgboost.txt'
METRICS_OUT     = 'metrics_xgboost.json'
EXCEL_OUT       = 'metrics_xgboost.xlsx'
PLOT_OUT        = 'feature_importance_xgboost.png'

RANDOM_STATE    = 42
N_SPLITS        = 5
DROP_COLS       = ['active', 'GH3', 'oligo', 'description', 'split']

# ============================================================
# LOAD DATA
# ============================================================
print("\n📂 Loading split data...")
df = pd.read_csv(CSV_SPLIT)
df['split'] = df['split'].astype(str).str.lower()

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
y = df['active'].astype(int)

train_mask = df['split'] == 'train'
test_mask  = df['split'] == 'test'

X_train = X.loc[train_mask].copy()
X_test  = X.loc[test_mask].copy()
y_train = y.loc[train_mask].copy()
y_test  = y.loc[test_mask].copy()

print(f"  Train n={len(X_train)} (pos={int(y_train.sum())}) | Test n={len(X_test)} (pos={int(y_test.sum())})")

# ============================================================
# FEATURE RANKING via XGBoost gain importance
# ============================================================
print("\n🔎 Ranking features with XGBoost gain-based importances...")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

ranking_clf = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='logloss',
    tree_method='hist',
)
ranking_clf.fit(X_train_res, y_train_res)

# Use gain importance — more reliable than weight/cover for feature selection
importances = ranking_clf.get_booster().get_score(importance_type='gain')
# Fill 0 for features not used by any tree
all_importances = {feat: importances.get(feat, 0.0) for feat in X.columns}
feature_ranking = pd.Series(all_importances).sort_values(ascending=False)
ordered_features = feature_ranking.index.tolist()

print(f"\n  Top 15 features by gain importance:")
for feat, val in feature_ranking.head(15).items():
    print(f"    {feat:<40} gain = {val:.4f}")

# ============================================================
# EVALUATE FEATURES ONE BY ONE
# ============================================================
print("\n📊 Evaluating features one by one (OOF CV)...")

def make_xgboost():
    return XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method='hist',
    )

def evaluate_oof(X_tr, y_tr, X_te, y_te, features):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_tr))
    pr_aucs = []

    for tr_idx_cv, val_idx_cv in skf.split(X_tr, y_tr):
        X_f = X_tr.iloc[tr_idx_cv][features]
        y_f = y_tr.iloc[tr_idx_cv]
        X_v = X_tr.iloc[val_idx_cv][features]
        y_v = y_tr.iloc[val_idx_cv]

        X_f_res, y_f_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_f, y_f)

        clf = make_xgboost()
        clf.fit(X_f_res, y_f_res)
        probs = clf.predict_proba(X_v)[:, 1]
        oof_probs[val_idx_cv] = probs
        pr_aucs.append(average_precision_score(y_v, probs))

    precision, recall, thresholds = precision_recall_curve(y_tr.values, oof_probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-6)
    best_idx = int(np.argmax(f1s))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred_oof = (oof_probs >= best_thresh).astype(int)

    # Train on full train set, evaluate on test
    X_tr_res, y_tr_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr[features], y_tr)
    clf_final = make_xgboost()
    clf_final.fit(X_tr_res, y_tr_res)
    test_probs = clf_final.predict_proba(X_te[features])[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    return {
        'cv_precision':   precision_score(y_tr.values, y_pred_oof),
        'cv_recall':      recall_score(y_tr.values, y_pred_oof),
        'cv_f1':          f1_score(y_tr.values, y_pred_oof),
        'cv_pr_auc':      float(np.mean(pr_aucs)),
        'test_precision': precision_score(y_te, test_preds),
        'test_recall':    recall_score(y_te, test_preds),
        'test_f1':        f1_score(y_te, test_preds),
        'test_pr_auc':    average_precision_score(y_te, test_probs),
        'threshold':      best_thresh,
    }

results = []
for i in range(1, len(ordered_features) + 1):
    features = ordered_features[:i]
    metrics = evaluate_oof(X_train, y_train, X_test, y_test, features)
    metrics['num_features'] = i
    metrics['features'] = ','.join(features)
    results.append(metrics)
    print(f"  n={i:3d} | CV PR-AUC={metrics['cv_pr_auc']:.3f} | Test PR-AUC={metrics['test_pr_auc']:.3f} | Test F1={metrics['test_f1']:.3f}")

results_df = pd.DataFrame(results)
results_df.to_excel(EXCEL_OUT, index=False, engine='openpyxl')
print(f"\n  ✅ Full metrics saved to {EXCEL_OUT}")

# ============================================================
# PROMPT USER TO CHOOSE NUMBER OF FEATURES
# ============================================================
print("\n📋 Top 10 by Test F1:")
print(
    results_df[['num_features', 'cv_pr_auc', 'test_pr_auc', 'test_f1']]
    .sort_values('test_f1', ascending=False)
    .head(10)
    .to_string(index=False)
)

while True:
    try:
        chosen_n = int(input(f"\n🔢 How many features for the final XGBoost model? (1–{len(ordered_features)}): "))
        if 1 <= chosen_n <= len(ordered_features):
            break
        print(f"  ⚠️ Enter a number between 1 and {len(ordered_features)}")
    except ValueError:
        print("  ⚠️ Invalid input — enter an integer.")

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print(f"\n🏋️  Training final XGBoost model with {chosen_n} features...")
selected_row = results_df[results_df['num_features'] == chosen_n].iloc[0]
best_features = selected_row['features'].split(',')
best_threshold = selected_row['threshold']

X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
    X_train[best_features], y_train
)
final_model = make_xgboost()
final_model.fit(X_train_res, y_train_res)

# ============================================================
# SAVE MODEL + METADATA
# ============================================================
joblib.dump(final_model, MODEL_OUT)

with open(FEATURES_OUT, 'w') as f:
    f.write('\n'.join(best_features))

with open(THRESHOLD_OUT, 'w') as f:
    f.write(str(best_threshold))

final_metrics = {
    'model':          'XGBoost',
    'num_features':   chosen_n,
    'features':       best_features,
    'threshold':      best_threshold,
    'cv_pr_auc':      selected_row['cv_pr_auc'],
    'cv_f1':          selected_row['cv_f1'],
    'cv_precision':   selected_row['cv_precision'],
    'cv_recall':      selected_row['cv_recall'],
    'test_pr_auc':    selected_row['test_pr_auc'],
    'test_f1':        selected_row['test_f1'],
    'test_precision': selected_row['test_precision'],
    'test_recall':    selected_row['test_recall'],
}
with open(METRICS_OUT, 'w') as f:
    json.dump(final_metrics, f, indent=2)

print(f"\n  ✅ Model saved:     {MODEL_OUT}")
print(f"  ✅ Features saved:  {FEATURES_OUT}")
print(f"  ✅ Threshold saved: {THRESHOLD_OUT}")
print(f"  ✅ Metrics saved:   {METRICS_OUT}")

# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================
final_importances = final_model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': best_features,
    'Gain': [final_importances.get(f, 0.0) for f in best_features]
}).sort_values('Gain', ascending=True)

plt.figure(figsize=(10, max(4, len(best_features) * 0.5)))
plt.barh(importance_df['Feature'], importance_df['Gain'])
plt.xlabel('Feature Importance (Gain)')
plt.title(f'XGBoost Feature Importances (n={chosen_n})')
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300)
plt.close()
print(f"  ✅ Feature importance plot saved: {PLOT_OUT}")

print(f"\n🎉 Done! Run 1d next.")
