import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Safe for server use
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from tqdm import tqdm

# --- 1. SETTINGS ---
GT_PATH = "/data/se26/nnUNet/test_independent/labelsTs"
AI_RAW_FILE = "individual_ai_results.csv"
METADATA = "test_metadata.csv"

# --- 2. CALCULATE GROUND TRUTH PER PATIENT ---
print("Step 1: Calculating Ground Truth volumes for all patients...")
subject_vols = {}

def get_vol(path):
    nii = nib.load(path)
    voxel_vol = np.prod(nii.header.get_zooms()[:3]) / 1000.0
    return np.sum(nii.get_fdata() == 1) * voxel_vol

files = [f for f in os.listdir(GT_PATH) if f.endswith(".nii.gz")]
for f in tqdm(files):
    sid = f.split('_')[0]
    vol = get_vol(os.path.join(GT_PATH, f))
    if sid not in subject_vols: subject_vols[sid] = {'GT_EDV': 0, 'GT_ESV': 0}
    if "_ED" in f: subject_vols[sid]['GT_EDV'] = vol
    elif "_ES" in f: subject_vols[sid]['GT_ESV'] = vol

gt_df = pd.DataFrame.from_dict(subject_vols, orient='index').reset_index().rename(columns={'index': 'sid'})
gt_df['GT_EF'] = ((gt_df['GT_EDV'] - gt_df['GT_ESV']) / gt_df['GT_EDV']) * 100

# --- 3. MERGE WITH INDIVIDUAL AI RESULTS ---
print("Step 2: Merging AI results with Ground Truth...")
ai_df = pd.read_csv(AI_RAW_FILE)

# Convert sid to string to ensure matching works
ai_df['sid'] = ai_df['sid'].astype(str)
gt_df['sid'] = gt_df['sid'].astype(str)

merged_df = ai_df.merge(gt_df, on='sid')

# --- 4. CALCULATE THE BIAS (ERROR) ---
merged_df['EDV_Bias'] = merged_df['EDV'] - merged_df['GT_EDV']
merged_df['ESV_Bias'] = merged_df['ESV'] - merged_df['GT_ESV']
merged_df['EF_Bias']  = merged_df['EF']  - merged_df['GT_EF']

# --- 5. NORMALITY TESTING AND PLOTTING ---
metrics = ['EDV_Bias', 'ESV_Bias', 'EF_Bias']
units = ['mL', 'mL', '%']

print("\n" + "="*65)
print(f"{'Metric':<10} | {'Shapiro Stat':<12} | {'P-Value':<12} | {'Decision'}")
print("-" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metrics):
    data = merged_df[metric].dropna()
    
    # Shapiro-Wilk Test
    stat, p = shapiro(data)
    
    # Decisions based on the p-value
    decision = "USE T-TEST" if p > 0.05 else "USE WILCOXON"
    print(f"{metric:<10} | {stat:<12.4f} | {p:<12.4e} | {decision}")
    
    # Plot Distribution
    sns.histplot(data, kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f"{metric} Distribution\nShapiro P: {p:.2e}")
    axes[i].set_xlabel(f"Bias ({units[i]})")
    axes[i].axvline(0, color='red', linestyle='--', label='Zero Bias')

plt.tight_layout()
plt.savefig('normality_results.png')
print("="*65)
print(f"\nSuccess! Analyzed {len(merged_df)} predictions.")
print("Results saved to 'normality_results.png'.")