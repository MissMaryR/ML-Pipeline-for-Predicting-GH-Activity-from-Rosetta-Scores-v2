# GH–Oligosaccharide Binding Activity Prediction Pipeline

A machine learning pipeline for predicting GH–oligosaccharide binding activity from Rosetta docking scores. Trains and evaluates four models independently, compares their performance, and produces publication-ready plots.

---

## Overview

This pipeline takes the cleaned and normalized docking data from the data preparation pipeline (`8_Book1_final.csv`) and:

1. Trains four independent classifiers, each selecting its own best features
2. Compares model performance across PR-AUC, F1, precision, and recall
3. Generates evaluation plots (PR curves, ROC curves, confusion matrices, probability distributions)
4. Generates feature-level violin/box plots showing model predictions per GH–oligo pair

---

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib matplotlib seaborn openpyxl
```

---

## Input

This pipeline reads from:

| File | Created by |
|------|------------|
| `8_Book1_final.csv` | `1_clean_data.py` (data preparation pipeline) |

`1a_train_logreg.py` must be run first — it creates `Book1_aggregated_with_split.csv` which all other model scripts depend on.

---

## Run Order

```bash
python3 1a_train_logreg.py        # Run FIRST — creates the train/test split
python3 1b_train_extratrees.py
python3 1c_train_xgboost.py
python3 1d_train_randomforest.py
python3 2_compare_models.py
python3 3_evaluate.py
python3 4_feature_plots.py
```

---

## Scripts

### `1a_train_logreg.py` — L1 Logistic Regression
- Reads `8_Book1_final.csv` and creates the stratified 80/20 train/test split
- Ranks features by L1 coefficient magnitude
- Evaluates adding one feature at a time using 5-fold OOF cross-validation with SMOTE
- Prints a top 10 table then prompts you to choose how many features to use
- Trains final model and saves all outputs

**Must be run before any other model script.**

### `1b_train_extratrees.py` — Extra Trees Classifier
- Reads the split CSV created by `1a`
- Ranks features using ExtraTrees native `feature_importances_` (mean decrease in impurity)
- Same OOF evaluation and interactive feature selection as `1a`
- Trains final model and saves all outputs

### `1c_train_xgboost.py` — XGBoost
- Reads the split CSV created by `1a`
- Ranks features using XGBoost gain-based importance (most informative split gain)
- Same OOF evaluation and interactive feature selection as `1a`
- Trains final model and saves all outputs

### `1d_train_randomforest.py` — Random Forest
- Reads the split CSV created by `1a`
- Ranks features using RandomForest native `feature_importances_`
- Same OOF evaluation and interactive feature selection as `1a`
- Trains final model and saves all outputs

### `2_compare_models.py` — Model Comparison
- Automatically loads all `metrics_*.json` files — no hardcoding needed
- Produces 6 comparison plots saved to `output/comparison/`
- Also saves a `model_comparison_summary.csv`

### `3_evaluate.py` — Model Evaluation
- Automatically loads all `model_*.joblib` files
- For each model produces 4 plots × 2 splits (full dataset + test set)
- Saves to `output/evaluate/<model_name>/full/` and `/test/`

### `4_feature_plots.py` — Feature Violin/Box Plots
- Automatically loads all `model_*.joblib` files
- For each model and each selected feature produces a violin + box + scatter plot
- Points colored by GH identity, shaped by oligo, outlined green/red by prediction correctness
- Saves to `output/feature_plots/<model_name>/full/` and `/test/`
- Saves one shared legend to `output/feature_plots/legend.png`

---

## Outputs

### Per model (1a–1d)
Each training script saves the following files:

| File | Description |
|------|-------------|
| `model_<name>.joblib` | Trained model |
| `features_<name>.txt` | Selected feature names (one per line) |
| `threshold_<name>.txt` | Decision threshold optimized on OOF pooled F1 |
| `metrics_<name>.json` | CV and test metrics (loaded by scripts 2–4) |
| `metrics_<name>.xlsx` | Full per-feature-count metrics table |
| `feature_importance_<name>.png` | Feature importance bar chart |

### `2_compare_models.py` → `output/comparison/`

| File | Description |
|------|-------------|
| `comparison_test_prauc` | Test PR-AUC per model |
| `comparison_cv_prauc` | CV PR-AUC per model |
| `comparison_test_f1` | Test F1 per model |
| `comparison_test_vs_cv_prauc` | Test vs CV PR-AUC grouped |
| `comparison_precision_recall_f1` | Precision / Recall / F1 grouped |
| `comparison_num_features` | Number of features selected per model |
| `model_comparison_summary.csv` | All metrics in one table |

All plots saved as `.png` and `.pdf`.

### `3_evaluate.py` → `output/evaluate/<model_name>/full/` and `/test/`

| File | Description |
|------|-------------|
| `pr_curve` | Precision-Recall curve with operating point marked |
| `roc_curve` | ROC curve with AUC |
| `confusion_matrix` | Confusion matrix (Inactive / Active) |
| `prob_distribution` | Predicted probability histogram by true class |

All plots saved as `.png` and `.pdf`.

### `4_feature_plots.py` → `output/feature_plots/<model_name>/full/` and `/test/`

One plot per selected feature per model per split. Each plot shows:
- Violin + box plot of feature distribution by activity
- Scatter points colored by GH identity
- Marker shape by oligo
- Point outline green (correct prediction) or red (incorrect prediction)

Plus `output/feature_plots/legend.png` — shared legend for all plots.

---

## Design Decisions

**Why independent feature selection per model?**
Each model type has its own native importance measure — L1 coefficients for logistic regression, impurity-based importance for tree models, gain for XGBoost. Forcing all models to use the same features would disadvantage models whose best features differ. The goal is to find which model performs best when given every advantage, not to make a constrained fair comparison.

**Why the same train/test split?**
The split is created once by `1a_train_logreg.py` and reused by all other scripts. This ensures that differences in test performance between models reflect genuine model differences rather than differences in what data ended up in the test set.

**Why SMOTE inside each CV fold?**
Applying SMOTE before cross-validation would leak synthetic samples into the validation set, inflating CV metrics. Applying it inside each fold ensures the validation set always contains only real samples.

**Why pooled OOF threshold?**
Rather than picking a threshold per fold and averaging, all out-of-fold predictions are pooled and the threshold that maximizes F1 on the full pooled set is used. This is more stable and directly optimizes the metric we care about.

---

## Customizing for Your Data

**Oligo marker shapes** — in `4_feature_plots.py`, update `OLIGO_MARKERS` to match your oligo names:
```python
OLIGO_MARKERS = {
    'oligo1': 'o',
    'oligo2': 'D',
    'oligo3': 's',
}
```

**Regularization strength** — in `1a_train_logreg.py`, adjust `L1_C`:
```python
L1_C = 0.5   # smaller = stronger regularization, more features driven to zero
```

**Train/test split ratio** — in `1a_train_logreg.py`:
```python
train_test_split(..., test_size=0.20, ...)  # change 0.20 to adjust split
```

**Number of CV folds** — in any training script:
```python
N_SPLITS = 5   # increase for more stable estimates on larger datasets
```

---

## Notes

- All 4 models may select different features — this is expected and intentional
- Scripts 2, 3, and 4 automatically detect all trained models via `glob` — if you only train 2 models, they will only evaluate those 2
- Each plot is saved as both `.png` (for viewing) and `.pdf` (for papers/presentations)
- Re-running `1a` will overwrite the train/test split — if you want to compare runs, back up `Book1_aggregated_with_split.csv` first
