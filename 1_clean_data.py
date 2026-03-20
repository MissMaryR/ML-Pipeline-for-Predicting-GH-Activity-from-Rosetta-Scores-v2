#!/usr/bin/env python3

import pandas as pd
import numpy as np
import openpyxl
import csv
import os

# ============================================================
# STEP 0: Collect all glycan_top_fullscores.csv into Book1.xlsx
# ============================================================
def step0_collect():
    print("\n📂 STEP 0: Collecting CSVs into Book1.xlsx...")
    base_dir = os.getcwd()
    all_rows = []
    headers = None

    for folder1 in os.listdir(base_dir):
        folder1_path = os.path.join(base_dir, folder1)
        if not os.path.isdir(folder1_path):
            continue
        for folder2 in os.listdir(folder1_path):
            folder2_path = os.path.join(folder1_path, folder2)
            if not os.path.isdir(folder2_path):
                continue
            csv_path = os.path.join(folder2_path, "Top_PDBs", "glycan_top_fullscores.csv")
            if not os.path.exists(csv_path):
                continue
            print(f"  ✅ Found: {csv_path}")
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                if headers is None:
                    headers = list(reader.fieldnames) + ['GH3', 'oligo']
                for row in reader:
                    row['GH3'] = folder1
                    row['oligo'] = folder2
                    all_rows.append(row)

    if not all_rows:
        print("  ❌ No glycan_top_fullscores.csv files found.")
        return False

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(headers)
    for row in all_rows:
        ws.append([row.get(h, 'NA') for h in headers])

    wb.save("Book1.xlsx")
    print(f"  📊 Saved {len(all_rows)} rows to Book1.xlsx")
    return True


# ============================================================
# STEP 1: Swap SR_1 and SR_2 if needed
# ============================================================
def step1_swap():
    print("\n🔄 STEP 1: Swapping SR_1/SR_2 if needed...")
    df = pd.read_excel("Book1.xlsx")

    sr1_cols = [col for col in df.columns if col.startswith('SR_1_')]
    sr2_cols = [col for col in df.columns if col.startswith('SR_2_')]

    assert len(sr1_cols) == len(sr2_cols), "Mismatch in SR_1_ and SR_2_ columns"
    assert all(s1.replace("SR_1_", "") == s2.replace("SR_2_", "") for s1, s2 in zip(sr1_cols, sr2_cols))

    def maybe_swap_sr(row):
        if row['SR_2'] <= row['SR_1']:
            row['SR_1'], row['SR_2'] = row['SR_2'], row['SR_1']
            for col1, col2 in zip(sr1_cols, sr2_cols):
                row[col1], row[col2] = row[col2], row[col1]
        return row

    df = df.apply(maybe_swap_sr, axis=1)
    df.to_excel("Book1_swapped.xlsx", index=False)
    print("  ✅ Saved Book1_swapped.xlsx")


# ============================================================
# STEP 2: Drop all-zero columns, print highly correlated pairs
# ============================================================
def step2_clean():
    print("\n🧹 STEP 2: Dropping zero columns and reporting correlations...")
    df = pd.read_excel("Book1_swapped.xlsx")
    numeric_df = df.select_dtypes(include='number')

    # Drop columns that are 100% zero
    zero_cols = [col for col in numeric_df.columns if (numeric_df[col] == 0).all()]
    df.drop(columns=zero_cols, inplace=True)
    print(f"\n  Dropped {len(zero_cols)} fully zero columns:")
    for col in zero_cols:
        print(f"    - {col}")

    df.to_excel("Book1_swapped.xlsx", index=False)
    print(f"\n  ✅ Saved updated Book1_swapped.xlsx")

    # Recompute numeric after dropping
    numeric_df = df.select_dtypes(include='number')

    # Print highly correlated pairs (>= 95%)
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (col1, col2, corr)
        for col1 in upper.columns
        for col2, corr in upper[col1].items()
        if corr >= 0.95
    ]

    print(f"\n  🧬 Highly correlated column pairs (>= 95% similarity):")
    if high_corr_pairs:
        for f1, f2, score in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"    {f1} ↔ {f2} = {score:.2f}")
    else:
        print("    None found.")


# ============================================================
# STEP 3: Drop correlated columns and recheck
# ============================================================
def step3_drop_correlated():
    print("\n✂️  STEP 3: Dropping correlated columns...")
    df = pd.read_excel("Book1_swapped.xlsx")

    columns_to_drop = [
    'fa_rep.1', 'hbond_sc.1', 'total_score.1', 'total_score_A',
    'SR_3_interf_E_1_2', 'if_A_fa_rep', 'SR_3_all_cst',
    'fa_atr', 'fa_sol', 'ref', 'tot_total_pos_charges', 'hbond_bb_sc',
    'tot_hbond_pm'
    ]

    dropped = [col for col in columns_to_drop if col in df.columns]
    not_found = [col for col in columns_to_drop if col not in df.columns]

    df.drop(columns=dropped, inplace=True)
    df.to_excel("Book1_cleaned.xlsx", index=False)

    print(f"  ✅ Dropped {len(dropped)} columns:")
    for col in dropped:
        print(f"    - {col}")
    if not_found:
        print(f"  ⚠️  Not found (already removed): {not_found}")
    print(f"\n  ✅ Saved Book1_cleaned.xlsx")


# ============================================================
# STEP 4: Redundancy recheck on cleaned file
# ============================================================
def step4_recheck():
    print("\n🔍 STEP 4: Rechecking correlations on cleaned file...")
    df = pd.read_excel("Book1_cleaned.xlsx")
    numeric_df = df.select_dtypes(include='number')

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (col1, col2, corr)
        for col1 in upper.columns
        for col2, corr in upper[col1].items()
        if corr >= 0.95
    ]

    print(f"\n  🧬 Remaining highly correlated pairs (>= 95%):")
    if high_corr_pairs:
        for f1, f2, score in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"    {f1} ↔ {f2} = {score:.2f}")
    else:
        print("    ✅ None found — data looks clean!")

    zero_cols = [col for col in numeric_df.columns if (numeric_df[col] == 0).all()]
    print(f"\n  🧹 Remaining fully zero columns:")
    if zero_cols:
        for col in zero_cols:
            print(f"    - {col}")
    else:
        print("    ✅ None found!")


# ============================================================
# STEP 5: Normalize
# ============================================================
def safe_div(numerator, denominator):
    return numerator / denominator.clip(lower=1e-8)

def step5_normalize():
    print("\n📐 STEP 5: Normalizing...")
    df = pd.read_excel("Book1_cleaned.xlsx")
    df.columns = df.columns.str.strip()

    residue_counts = {
        '39': 1001, '41': 801, '47': 994, '76': 857,
        'B3': 740, 'B6': 817, 'B9': 945,
        'I1': 829, 'I2': 823, 'W0': 1080
    }
    df['residue_count'] = df['GH3'].map(residue_counts)

    global_score_cols = [
        'total_score', 'fa_rep', 'hbond_sc', 'tot_pstat_pm',
        'tot_burunsat_pm', 'tot_NLconts_pm', 'tot_nlsurfaceE_pm',
        'tot_total_charge', 'tot_total_neg_charges',
        'fa_dun', 'fa_intra_rep', 'fa_intra_sol_xover4',
        'omega', 'p_aa_pp', 'pro_close', 'rama_prepro',
        'yhh_planarity', 'lk_ball_wtd'
    ]

    raw_to_drop = []
    for col in global_score_cols:
        if col in df.columns:
            df[f"{col}_per_res"] = df[col] / df['residue_count'].replace(0, np.nan)
            raw_to_drop.append(col)

    df.drop(columns=raw_to_drop, inplace=True)
    print(f"  🗑️  Dropped {len(raw_to_drop)} raw columns after normalizing")

    sr_sets = {
        'SR_1': ['SR_1_total_score', 'SR_1_fa_rep', 'SR_1_hbond_sc', 'SR_1_all_cst', 'SR_1_hbond_pm', 'SR_1_pstat_pm', 'SR_1_nlpstat_pm'],
        'SR_2': ['SR_2_total_score', 'SR_2_fa_rep', 'SR_2_hbond_sc', 'SR_2_hbond_pm', 'SR_2_burunsat_pm', 'SR_2_pstat_pm', 'SR_2_nlpstat_pm'],
        'SR_3': ['SR_3_total_score', 'SR_3_fa_rep', 'SR_3_hbond_sc', 'SR_3_dsasa_1_2', 'SR_3_hbond_pm', 'SR_3_burunsat_pm']
    }

    sr_to_remove = []
    for sr_key, sr_cols in sr_sets.items():
        if sr_key in df.columns:
            for col in sr_cols:
                if col in df.columns:
                    df[f"{col}_per_{sr_key}"] = df[col] / df[sr_key].replace(0, np.nan)
                    sr_to_remove.append(col)
            sr_to_remove.append(sr_key)
        else:
            print(f"  ⚠️ '{sr_key}' column missing — skipping.")

    df.drop(columns=[col for col in sr_to_remove if col in df.columns], inplace=True)

    leftover = [col for col in df.columns if col.startswith('SR_3') and '_per_' not in col]
    if leftover:
        print(f"  ⚠️ Dropping leftover SR_3 columns: {leftover}")
        df.drop(columns=leftover, inplace=True)

    df.drop(columns='residue_count', inplace=True, errors='ignore')
    df.to_excel("Book1_normalized.xlsx", index=False)
    print("  ✅ Saved Book1_normalized.xlsx")

# ============================================================
# STEP 6: Redundancy recheck on normalized file
# ============================================================
def step6_recheck():
    print("\n🔍 STEP 6: Rechecking correlations on normalized file...")
    df = pd.read_excel("Book1_normalized.xlsx")
    numeric_df = df.select_dtypes(include='number')

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (col1, col2, corr)
        for col1 in upper.columns
        for col2, corr in upper[col1].items()
        if corr >= 0.95
    ]

    print(f"\n  🧬 Remaining highly correlated pairs (>= 95%):")
    if high_corr_pairs:
        for f1, f2, score in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"    {f1} ↔ {f2} = {score:.2f}")
    else:
        print("    ✅ None found — data looks clean!")

    zero_cols = [col for col in numeric_df.columns if (numeric_df[col] == 0).all()]
    print(f"\n  🧹 Remaining fully zero columns:")
    if zero_cols:
        for col in zero_cols:
            print(f"    - {col}")
    else:
        print("    ✅ None found!")




# ============================================================
# STEP 7: Drop post-normalization correlated columns and recheck
# ============================================================
def step7_drop_and_recheck():
    print("\n✂️  STEP 7: Dropping post-normalization correlated columns...")
    df = pd.read_excel("Book1_normalized.xlsx")

    columns_to_drop = [
        'fa_dun_per_res',
        'tot_total_neg_charges_per_res'
    ]

    dropped = [col for col in columns_to_drop if col in df.columns]
    not_found = [col for col in columns_to_drop if col not in df.columns]

    df.drop(columns=dropped, inplace=True)
    df.to_excel("Book1_normalized.xlsx", index=False)

    print(f"  ✅ Dropped {len(dropped)} columns:")
    for col in dropped:
        print(f"    - {col}")
    if not_found:
        print(f"  ⚠️  Not found (already removed): {not_found}")

    # Recheck
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (col1, col2, corr)
        for col1 in upper.columns
        for col2, corr in upper[col1].items()
        if corr >= 0.95
    ]

    print(f"\n  🧬 Remaining highly correlated pairs (>= 95%):")
    if high_corr_pairs:
        for f1, f2, score in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"    {f1} ↔ {f2} = {score:.2f}")
    else:
        print("    ✅ None found — data looks clean!")

    zero_cols = [col for col in numeric_df.columns if (numeric_df[col] == 0).all()]
    print(f"\n  🧹 Remaining fully zero columns:")
    if zero_cols:
        for col in zero_cols:
            print(f"    - {col}")
    else:
        print("    ✅ None found!")


# ============================================================
# STEP 8: Aggregate by GH3 + oligo
# ============================================================
def step8_aggregate():
    print("\n📊 STEP 8: Aggregating by GH3 + oligo...")
    df = pd.read_excel("Book1_normalized.xlsx")
    df['GH3'] = df['GH3'].astype(str)
    df['oligo'] = df['oligo'].astype(str)
    agg_df = df.groupby(['GH3', 'oligo']).mean(numeric_only=True).reset_index()
    agg_df.to_csv("Book1_aggregated.csv", index=False)
    print(f"  ✅ Saved Book1_aggregated.csv ({len(agg_df)} rows)")


# ============================================================
# STEP 9: Add active/inactive labels
# ============================================================
def step9_label_active():
    print("\n🏷️  STEP 9: Labeling active/inactive...")
    df = pd.read_csv("Book1_aggregated.csv")
    df['GH3'] = df['GH3'].astype(str).str.strip()
    df['oligo'] = df['oligo'].astype(str).str.strip()

    active_pairs = {
        'CL3': {'B9', 'B3', '76', '41'},
        'CR3': {'B9', 'B3', 'I2', '41', '76', '47'},
        'XY3': {'B9'},
        'H2B': {'B9'},
        'H3B': {'B9', 'B3'},
    }

    def is_active(row):
        oligo = row['oligo']
        gh3 = row['GH3']
        return 1 if active_pairs.get(oligo) and gh3 in active_pairs[oligo] else 0

    df['active'] = df.apply(is_active, axis=1)

    active_count = int(df['active'].sum())
    inactive_count = len(df) - active_count
    print(f"  ✅ Labeled {active_count} active and {inactive_count} inactive entries")
    print(f"\n  Active pairs found:")
    for _, row in df[df['active'] == 1].iterrows():
        print(f"    {row['oligo']} - {row['GH3']}")

    df.to_csv("Book1_aggregated.csv", index=False)
    print(f"\n  💾 Saved updated Book1_aggregated.csv")
    
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if not step0_collect():
        exit(1)
    step1_swap()
    step2_clean()
    step3_drop_correlated()
    step4_recheck()
    step5_normalize()
    step6_recheck()
    step7_drop_and_recheck()
    step8_aggregate()
    step9_label_active()
    print("\n🎉 All steps complete!")
