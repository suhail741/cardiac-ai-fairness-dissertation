import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from tqdm import tqdm

# --- 1. SETTINGS ---
GT_PATH = "/data/se26/nnUNet/test_independent/labelsTs"
AI_RAW_FILE = "individual_ai_results.csv"

# --- 2. LOAD & CALCULATE GROUND TRUTH ---
print("Step 1: Loading Ground Truth...")
subject_vols = {}

def get_vol(path):
    try:
        nii = nib.load(path)
        voxel_vol = np.prod(nii.header.get_zooms()[:3]) / 1000.0
        return np.sum(nii.get_fdata() == 1) * voxel_vol
    except:
        return np.nan

files = [f for f in os.listdir(GT_PATH) if f.endswith(".nii.gz")]
for f in tqdm(files):
    sid = f.split('_')[0]
    vol = get_vol(os.path.join(GT_PATH, f))
    if sid not in subject_vols: subject_vols[sid] = {'GT_EDV': 0, 'GT_ESV': 0}
    if "_ED" in f: subject_vols[sid]['GT_EDV'] = vol
    elif "_ES" in f: subject_vols[sid]['GT_ESV'] = vol

gt_df = pd.DataFrame.from_dict(subject_vols, orient='index').reset_index().rename(columns={'index': 'sid'})

# --- 3. MERGE & CLEAN BIAS ---
print("Step 2: Merging and Cleaning Data...")
ai_df = pd.read_csv(AI_RAW_FILE)
ai_df['sid'] = ai_df['sid'].astype(str)
gt_df['sid'] = gt_df['sid'].astype(str)

merged_df = ai_df.merge(gt_df, on='sid')

# Calculate Biomarkers safely
# Use np.where to avoid division by zero
merged_df['GT_EF'] = np.where(merged_df['GT_EDV'] > 0, 
                             ((merged_df['GT_EDV'] - merged_df['GT_ESV']) / merged_df['GT_EDV']) * 100, 
                             np.nan)

merged_df['EDV_Bias'] = merged_df['EDV'] - merged_df['GT_EDV']
merged_df['ESV_Bias'] = merged_df['ESV'] - merged_df['GT_ESV']
merged_df['EF_Bias']  = merged_df['EF']  - merged_df['GT_EF']

# CRITICAL: Remove any NaNs or Infs that break the Kruskal-Wallis test
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['EDV_Bias', 'ESV_Bias', 'EF_Bias'])

# --- 4. KRUSKAL-WALLIS TEST (GLOBAL) ---
print("\n" + "="*75)
print("TEST 1: GLOBAL KRUSKAL-WALLIS (All 631 Patients)")
print("="*75)

tasks = merged_df['Task'].unique()
metrics = ['EDV_Bias', 'ESV_Bias', 'EF_Bias']

for metric in metrics:
    groups = [merged_df[merged_df['Task'] == t][metric].values for t in tasks]
    stat, p = kruskal(*groups)
    print(f"{metric:<10} | H-stat: {stat:>7.2f} | P-value: {p:.4e}")

# --- 5. THE "ANDY SPECIAL": SUBGROUP KRUSKAL-WALLIS ---
# This checks if the models are different WITHIN a specific group (e.g., just Females)
print("\n" + "="*75)
print("TEST 2: SUBGROUP KRUSKAL-WALLIS (Checking for Hidden Bias)")
print("="*75)

for sub in sorted(merged_df['Sex'].unique()):
    print(f"\n--- Group: {sub} ---")
    sub_data = merged_df[merged_df['Sex'] == sub]
    for metric in metrics:
        groups = [sub_data[sub_data['Task'] == t][metric].values for t in tasks]
        stat, p = kruskal(*groups)
        res = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"{metric:<10} | P-value: {p:.4e} | Result: {res}")

print("\n" + "="*75)