#!/usr/bin/env python3
"""
1a_train_logreg.py
------------------
- Reads 8_Book1_final.csv
- Creates the train/test split (used by ALL subsequent model scripts)
- Ranks features using L1 logistic regression coefficients
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, average_precision_score
)
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
CSV_INPUT       = '8_Book1_final.csv'
CSV_SPLIT_OUT   = 'Book1_aggregated_with_split.csv'   # used by 1b, 1c, 1d
MODEL_OUT       = 'model_logreg.joblib'
FEATURES_OUT    = 'features_logreg.txt'
THRESHOLD_OUT   = 'threshold_logreg.txt'
METRICS_OUT     = 'metrics_logreg.json'
EXCEL_OUT       = 'metrics_logreg.xlsx'
PLOT_OUT        = 'feature_importance_logreg.png'

RANDOM_STATE    = 42
N_SPLITS        = 5
L1_C            = 0.5
DROP_COLS       = ['active', 'GH3', 'oligo', 'description']

# ============================================================
# LOAD DATA
# ============================================================
print("\n📂 Loading data...")
df = pd.read_csv(CSV_INPUT)
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
y = df['active'].astype(int)

print(f"  Total samples: {len(df)} | Active: {int(y.sum())} | Inactive: {int((1-y).sum())}")

# ============================================================
# TRAIN/TEST SPLIT — saved for all other model scripts
# ============================================================
print("\n✂️  Creating train/test split (80/20 stratified)...")
all_indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

# Save split CSV for other model scripts
df['split'] = 'train'
df.loc[test_idx, 'split'] = 'test'
df.to_csv(CSV_SPLIT_OUT, index=False)
print(f"  ✅ Split CSV saved to {CSV_SPLIT_OUT}")
print(f"  Train n={len(X_train)} (pos={int(y_train.sum())}) | Test n={len(X_test)} (pos={int(y_test.sum())})")

# ============================================================
# FEATURE RANKING via L1
# ============================================================
print("\n🔎 Ranking features with L1 logistic regression...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

ranking_clf = LogisticRegression(
    penalty='l1',
    C=L1_C,
    solver='liblinear',
    random_state=RANDOM_STATE,
    class_weight='balanced',
    max_iter=1000
)
ranking_clf.fit(X_train_res, y_train_res)

coefs = ranking_clf.coef_[0]
feature_ranking = pd.Series(np.abs(coefs), index=X.columns).sort_values(ascending=False)
ordered_features = feature_ranking.index.tolist()

nonzero = feature_ranking[feature_ranking > 0]
print(f"\n  Non-zero features ({len(nonzero)}):")
for feat, val in nonzero.items():
    print(f"    {feat:<40} |coef| = {val:.4f}")

# ============================================================
# EVALUATE FEATURES ONE BY ONE
# ============================================================
print("\n📊 Evaluating features one by one (OOF CV)...")

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

        clf = LogisticRegression(
            penalty='l1', C=L1_C, solver='liblinear',
            random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000
        )
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
    clf_final = LogisticRegression(
        penalty='l1', C=L1_C, solver='liblinear',
        random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000
    )
    clf_final.fit(X_tr_res, y_tr_res)
    test_probs = clf_final.predict_proba(X_te[features])[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    return {
        'cv_precision':  precision_score(y_tr.values, y_pred_oof),
        'cv_recall':     recall_score(y_tr.values, y_pred_oof),
        'cv_f1':         f1_score(y_tr.values, y_pred_oof),
        'cv_pr_auc':     float(np.mean(pr_aucs)),
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
        chosen_n = int(input(f"\n🔢 How many features for the final LogReg model? (1–{len(ordered_features)}): "))
        if 1 <= chosen_n <= len(ordered_features):
            break
        print(f"  ⚠️ Enter a number between 1 and {len(ordered_features)}")
    except ValueError:
        print("  ⚠️ Invalid input — enter an integer.")

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print(f"\n🏋️  Training final LogReg model with {chosen_n} features...")
selected_row = results_df[results_df['num_features'] == chosen_n].iloc[0]
best_features = selected_row['features'].split(',')
best_threshold = selected_row['threshold']

X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
    X_train[best_features], y_train
)
final_model = LogisticRegression(
    penalty='l1', C=L1_C, solver='liblinear',
    random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000
)
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
    'model':          'LogReg',
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

print(f"\n  ✅ Model saved:    {MODEL_OUT}")
print(f"  ✅ Features saved: {FEATURES_OUT}")
print(f"  ✅ Threshold saved:{THRESHOLD_OUT}")
print(f"  ✅ Metrics saved:  {METRICS_OUT}")

# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================
coefs_final = final_model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': best_features,
    'Coefficient': coefs_final
}).sort_values('Coefficient', key=np.abs, ascending=True)

plt.figure(figsize=(10, max(4, len(best_features) * 0.5)))
plt.barh(importance_df['Feature'], importance_df['Coefficient'])
plt.xlabel('Coefficient')
plt.title(f'L1 LogReg Feature Coefficients (n={chosen_n})')
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300)
plt.close()
print(f"  ✅ Feature importance plot saved: {PLOT_OUT}")

print(f"\n🎉 Done! Run 1b, 1c, 1d next — they will read {CSV_SPLIT_OUT} automatically.")
