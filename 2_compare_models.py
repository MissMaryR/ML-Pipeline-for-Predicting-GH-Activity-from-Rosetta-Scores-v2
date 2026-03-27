#!/usr/bin/env python3
"""
2_compare_models.py
-------------------
- Loads metrics_*.json from all 4 trained models automatically
- Produces comparison bar charts for Test PR-AUC, CV PR-AUC, Test F1
- Both single metric and grouped side-by-side plots
- Saves all plots to output/comparison/
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
OUTDIR = 'output/comparison'
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# LOAD ALL METRICS FILES AUTOMATICALLY
# ============================================================
print("\n📂 Loading model metrics...")
metrics_files = glob.glob('metrics_*.json')

if not metrics_files:
    print("  ❌ No metrics_*.json files found.")
    print("  Make sure you have run 1a, 1b, 1c, and 1d first.")
    exit(1)

rows = []
for path in sorted(metrics_files):
    with open(path) as f:
        data = json.load(f)
    rows.append(data)
    print(f"  ✅ Loaded: {path} ({data['model']}, {data['num_features']} features)")

df = pd.DataFrame(rows).sort_values('test_pr_auc', ascending=False).reset_index(drop=True)

print(f"\n📊 Summary:")
print(df[['model', 'num_features', 'cv_pr_auc', 'test_pr_auc', 'cv_f1', 'test_f1',
          'test_precision', 'test_recall']].to_string(index=False))

# ============================================================
# HELPER: save figure
# ============================================================
def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  ✅ Saved: {name}.png / .pdf")

def add_value_labels(ax, values, offset=0.02):
    for i, v in enumerate(values):
        ax.text(i, v + offset, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

# ============================================================
# PLOT 1: Test PR-AUC only
# ============================================================
print("\n📈 Generating plots...")
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.bar(df['model'], df['test_pr_auc'])
add_value_labels(ax, df['test_pr_auc'])
ax.set_ylim(0, 1.1)
ax.set_ylabel('Test PR-AUC (Average Precision)')
ax.set_title('Model Comparison — Test PR-AUC')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
save(fig, 'comparison_test_prauc')

# ============================================================
# PLOT 2: CV PR-AUC only
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.bar(df['model'], df['cv_pr_auc'])
add_value_labels(ax, df['cv_pr_auc'])
ax.set_ylim(0, 1.1)
ax.set_ylabel('CV PR-AUC (mean)')
ax.set_title('Model Comparison — CV PR-AUC')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
save(fig, 'comparison_cv_prauc')

# ============================================================
# PLOT 3: Test F1 only
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.bar(df['model'], df['test_f1'])
add_value_labels(ax, df['test_f1'])
ax.set_ylim(0, 1.1)
ax.set_ylabel('Test F1 Score')
ax.set_title('Model Comparison — Test F1')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
save(fig, 'comparison_test_f1')

# ============================================================
# PLOT 4: Test vs CV PR-AUC grouped
# ============================================================
x = np.arange(len(df))
w = 0.38

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.bar(x - w/2, df['test_pr_auc'], width=w, label='Test')
ax.bar(x + w/2, df['cv_pr_auc'],   width=w, label='CV (mean)')
ax.set_xticks(x)
ax.set_xticklabels(df['model'], rotation=25, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('PR-AUC (Average Precision)')
ax.set_title('Model Comparison — Test vs CV PR-AUC')
ax.legend(frameon=False)
plt.tight_layout()
save(fig, 'comparison_test_vs_cv_prauc')

# ============================================================
# PLOT 5: Precision, Recall, F1 grouped (test set)
# ============================================================
metrics_to_plot = ['test_precision', 'test_recall', 'test_f1']
labels = ['Precision', 'Recall', 'F1']
x = np.arange(len(df))
w = 0.25

fig, ax = plt.subplots(figsize=(7.0, 4.5))
for i, (col, label) in enumerate(zip(metrics_to_plot, labels)):
    offset = (i - 1) * w
    ax.bar(x + offset, df[col], width=w, label=label)

ax.set_xticks(x)
ax.set_xticklabels(df['model'], rotation=25, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Test Precision / Recall / F1')
ax.legend(frameon=False)
plt.tight_layout()
save(fig, 'comparison_precision_recall_f1')

# ============================================================
# PLOT 6: Number of features used per model
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.bar(df['model'], df['num_features'])
for i, v in enumerate(df['num_features']):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Number of Features')
ax.set_title('Features Selected per Model')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
save(fig, 'comparison_num_features')

# ============================================================
# SAVE SUMMARY TABLE
# ============================================================
summary_path = os.path.join(OUTDIR, 'model_comparison_summary.csv')
df.to_csv(summary_path, index=False)
print(f"  ✅ Summary table saved: {summary_path}")

print(f"\n🎉 Done! All plots saved to {OUTDIR}/")
print(f"   Run 3_evaluate.py next.")
