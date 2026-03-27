#!/usr/bin/env python3
"""
4_feature_plots.py
------------------
- Loads all 4 trained models and their metadata automatically
- For each model produces violin/box plots for each selected feature
  colored by GH identity, shaped by oligo, outlined by model correctness
- Plots split into full dataset and test set
- Saves all plots to output/feature_plots/<model_name>/full/ and /test/
- Saves a shared legend image
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
import matplotlib.lines as mlines
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
CSV_SPLIT  = 'Book1_aggregated_with_split.csv'
BASE_OUTDIR = 'output/feature_plots'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Marker shapes per oligo — update if your oligo names differ
OLIGO_MARKERS = {
    'CL3': 'o',
    'CR3': 'D',
    'H2B': 's',
    'H3B': '^',
    'XY3': 'X',
}
DEFAULT_MARKER = 'o'

# ============================================================
# LOAD DATA
# ============================================================
print("\n📂 Loading split data...")
df = pd.read_csv(CSV_SPLIT)
df['split']  = df['split'].astype(str).str.lower()
df['active'] = df['active'].astype(int)
df['GH3']    = df['GH3'].astype(str)
df['oligo']  = df['oligo'].astype(str)

df_test = df[df['split'] == 'test'].copy()

# Assign a consistent color per GH3 across all models
unique_gh3 = sorted(df['GH3'].unique())
palette = sns.color_palette('tab20', n_colors=len(unique_gh3))
gh3_colors = {gh3: palette[i] for i, gh3 in enumerate(unique_gh3)}

unique_oligos = sorted(df['oligo'].unique())

# ============================================================
# HELPERS
# ============================================================
def load_model_metadata(model_path):
    suffix = model_path.replace('model_', '').replace('.joblib', '')
    with open(f'features_{suffix}.txt') as f:
        features = [line.strip() for line in f if line.strip()]
    with open(f'threshold_{suffix}.txt') as f:
        threshold = float(f.read().strip())
    with open(f'metrics_{suffix}.json') as f:
        metrics = json.load(f)
    return features, threshold, metrics

def jitter_x(x, strength=0.08):
    return np.asarray(x) + np.random.uniform(-strength, strength, size=len(x))

def plot_feature(df_plot, feature, model_name, split_label, outpath):
    fig, ax = plt.subplots(figsize=(8, 10))

    # Violin
    sns.violinplot(
        data=df_plot, x='active', y=feature,
        palette='Pastel1', inner=None, cut=2, ax=ax
    )
    # Box
    sns.boxplot(
        data=df_plot, x='active', y=feature,
        width=0.2, showcaps=True,
        boxprops={'zorder': 2},
        showfliers=False, color='white', ax=ax
    )

    # Scatter with jitter
    x_jit = jitter_x(df_plot['active'].values)
    for idx, (_, row) in enumerate(df_plot.iterrows()):
        ax.scatter(
            x_jit[idx], row[feature],
            color=gh3_colors.get(row['GH3'], 'gray'),
            edgecolors=row['outline_color'],
            marker=OLIGO_MARKERS.get(row['oligo'], DEFAULT_MARKER),
            s=150,
            linewidth=2.3,
            alpha=0.9,
            zorder=3
        )

    # Axis formatting
    feat_vals = df_plot[feature].astype(float)
    pad = 2.5 * feat_vals.std() if feat_vals.std() > 0 else 1.0
    ax.set_ylim(feat_vals.min() - pad, feat_vals.max() + pad)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Inactive (0)', 'Active (1)'])
    ax.set_xlabel('Activity')
    ax.set_ylabel(feature)
    ax.set_title(
        f"{model_name} — {feature}\n"
        f"({split_label} | fill=GH, shape=oligo, edge=correctness)"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath + '.png', dpi=300)
    plt.close(fig)

def save_legend(outdir):
    """Save a shared legend for GH3 colors, oligo shapes, and correctness edges."""
    fig, ax = plt.subplots(figsize=(6, max(4, (len(unique_gh3) + len(unique_oligos) + 2) * 0.35)))
    handles = []

    # GH3 fill colors
    for gh3 in unique_gh3:
        handles.append(mlines.Line2D(
            [], [], marker='o', color='w',
            markerfacecolor=gh3_colors[gh3],
            markeredgecolor='black',
            markersize=10,
            label=f'GH: {gh3}'
        ))

    # Oligo shapes
    for oligo in unique_oligos:
        shape = OLIGO_MARKERS.get(oligo, DEFAULT_MARKER)
        handles.append(mlines.Line2D(
            [], [], marker=shape, color='black',
            markerfacecolor='white',
            markeredgewidth=1.5,
            markersize=10,
            linestyle='None',
            label=f'Oligo: {oligo}'
        ))

    # Correctness edges
    handles.append(mlines.Line2D(
        [], [], marker='o', color='w',
        markerfacecolor='white',
        markeredgecolor='green',
        markeredgewidth=2.3,
        markersize=10,
        label='Correct prediction'
    ))
    handles.append(mlines.Line2D(
        [], [], marker='o', color='w',
        markerfacecolor='white',
        markeredgecolor='red',
        markeredgewidth=2.3,
        markersize=10,
        label='Incorrect prediction'
    ))

    ax.legend(handles=handles, loc='center', fontsize=10, frameon=False)
    ax.axis('off')
    plt.tight_layout()
    legend_path = os.path.join(outdir, 'legend.png')
    fig.savefig(legend_path, dpi=300)
    plt.close(fig)
    print(f"  ✅ Legend saved: {legend_path}")

# ============================================================
# FIND ALL TRAINED MODELS
# ============================================================
model_files = sorted(glob.glob('model_*.joblib'))
if not model_files:
    print("  ❌ No model_*.joblib files found. Run 1a–1d first.")
    exit(1)

# ============================================================
# PLOT PER MODEL
# ============================================================
for model_path in model_files:
    suffix = model_path.replace('model_', '').replace('.joblib', '')
    print(f"\n{'='*60}")
    print(f"📊 Feature plots for: {suffix}")
    print(f"{'='*60}")

    try:
        model = joblib.load(model_path)
        features, threshold, metrics = load_model_metadata(model_path)
    except FileNotFoundError as e:
        print(f"  ⚠️  Skipping — missing file: {e}")
        continue

    model_name = metrics['model']
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing} — skipping.")
        continue

    outdir_full = os.path.join(BASE_OUTDIR, suffix, 'full')
    outdir_test = os.path.join(BASE_OUTDIR, suffix, 'test')
    os.makedirs(outdir_full, exist_ok=True)
    os.makedirs(outdir_test, exist_ok=True)

    # Add predictions and correctness to full df
    df['pred_prob']     = model.predict_proba(df[features])[:, 1]
    df['pred_label']    = (df['pred_prob'] >= threshold).astype(int)
    df['correct']       = df['pred_label'] == df['active']
    df['outline_color'] = df['correct'].map({True: 'green', False: 'red'})

    df_test_plot = df[df['split'] == 'test'].copy()

    # --- Full dataset plots ---
    print(f"  Generating full dataset plots ({len(features)} features)...")
    for feature in features:
        plot_feature(
            df, feature, model_name, 'Full Dataset',
            os.path.join(outdir_full, feature)
        )

    # --- Test set plots ---
    print(f"  Generating test set plots ({len(features)} features)...")
    for feature in features:
        plot_feature(
            df_test_plot, feature, model_name, 'Test Set',
            os.path.join(outdir_test, feature)
        )

    print(f"  ✅ Plots saved to output/feature_plots/{suffix}/")

# ============================================================
# SHARED LEGEND
# ============================================================
print("\n🖼️  Saving shared legend...")
save_legend(BASE_OUTDIR)

print(f"\n🎉 Done! All feature plots saved to {BASE_OUTDIR}/")
