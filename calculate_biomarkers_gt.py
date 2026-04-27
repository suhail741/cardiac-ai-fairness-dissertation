import os
import nibabel as nib
import numpy as np
import csv
import pandas as pd
import json
from tqdm import tqdm

# --- CONFIG ---
PRED_BASE_PATH = "/data/se26/nnUNet/nnUNet_results/predictions_independent"
GT_PATH = "/data/se26/nnUNet/test_independent/labelsTs"
METADATA_FILE = "test_metadata.csv"
TASKS = {
    201: "Age Balanced", 202: "Old Only", 203: "Young Only",
    204: "Sex Balanced", 205: "Female Only", 206: "Male Only"
}

# --- 1. LOAD METADATA ---
test_meta = {}
with open(METADATA_FILE, mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = str(row['sid']).strip()
        test_meta[sid] = {'sex': row['sex'], 'age': float(row['age_y'])}

# --- 2. VOLUME FUNCTION ---
def get_lv_volume_ml(nii_path):
    try:
        nii = nib.load(nii_path)
        data = nii.get_fdata()
        voxel_vol_ml = np.prod(nii.header.get_zooms()[:3]) / 1000.0
        return np.sum(data == 1) * voxel_vol_ml
    except:
        return np.nan

# --- 3. LOAD GROUND TRUTH ---
print("Step 1: Loading Ground Truth volumes...")
gt_vols = {}
gt_files = sorted([f for f in os.listdir(GT_PATH) if f.endswith(".nii.gz")])
for f in tqdm(gt_files, desc="Processing GT"):
    sid = f.split('_')[0]
    vol = get_lv_volume_ml(os.path.join(GT_PATH, f))
    if sid not in gt_vols: gt_vols[sid] = {'GT_EDV': 0, 'GT_ESV': 0}
    if "_ED" in f: gt_vols[sid]['GT_EDV'] = vol
    elif "_ES" in f: gt_vols[sid]['GT_ESV'] = vol

for sid in gt_vols:
    ed = gt_vols[sid]['GT_EDV']
    gt_vols[sid]['GT_EF'] = ((ed - gt_vols[sid]['GT_ESV']) / ed * 100) if ed > 0 else np.nan

# --- 4. THE MASTER LOOP ---
individual_records = [] 

for tid, tname in TASKS.items():
    print(f"\nProcessing Task {tid}: {tname}...")
    task_path = os.path.join(PRED_BASE_PATH, f"Task{tid}_results")
    
    # --- TAILORED DICE LOOKUP BASED ON YOUR KEYS ---
    json_path = os.path.join(task_path, "summary.json")
    dice_lookup = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
                results_list = data.get('results', {}).get('all', [])
                
                for entry in results_list:
                    # Based on your keys: 'test' is the filename, '1' is the metrics
                    test_file = entry.get('test')
                    if test_file:
                        # Extract S000031_ED from path
                        case_id = os.path.basename(test_file).replace(".nii.gz", "")
                        # Dice is inside the '1' key
                        dice_val = entry.get('1', {}).get('Dice', np.nan)
                        dice_lookup[case_id] = dice_val
            except Exception as e:
                print(f"⚠️ Error parsing {json_path}: {e}")

    # Process files
    subject_vols = {}
    files = [f for f in os.listdir(task_path) if f.endswith(".nii.gz")]
    
    for f in tqdm(files, desc=f"Reading {tname} Predictions"):
        sid = f.split('_')[0]
        vol = get_lv_volume_ml(os.path.join(task_path, f))
        if sid not in subject_vols: 
            subject_vols[sid] = {'EDV': 0, 'ESV': 0, 'Dice_ED': np.nan, 'Dice_ES': np.nan}
        
        clean_f = f.replace(".nii.gz", "")
        d_val = dice_lookup.get(clean_f, np.nan)
        
        if "_ED" in f: 
            subject_vols[sid]['EDV'] = vol
            subject_vols[sid]['Dice_ED'] = d_val
        elif "_ES" in f: 
            subject_vols[sid]['ESV'] = vol
            subject_vols[sid]['Dice_ES'] = d_val

    # --- PART 1: Capture Individual Records ---
    for sid, vols in subject_vols.items():
        if sid not in test_meta or sid not in gt_vols: continue
        meta = test_meta[sid]; gt = gt_vols[sid]
        
        if vols['EDV'] > 0 and vols['ESV'] > 0:
            ef = ((vols['EDV'] - vols['ESV']) / vols['EDV']) * 100
            d_vals = [v for v in [vols['Dice_ED'], vols['Dice_ES']] if not np.isnan(v)]
            avg_dice = np.mean(d_vals) if d_vals else np.nan
            
            individual_records.append([
                sid, tname, meta['sex'], 
                "Young" if meta['age'] <= 54 else "Old", 
                vols['EDV'], vols['ESV'], ef,
                vols['EDV'] - gt['GT_EDV'], 
                vols['ESV'] - gt['GT_ESV'], 
                ef - gt['GT_EF'],
                avg_dice
            ])

# --- 5. SAVE ---
df_raw = pd.DataFrame(individual_records, columns=[
    'sid', 'Task', 'Sex', 'AgeGroup', 'EDV', 'ESV', 'EF', 
    'EDV_Bias', 'ESV_Bias', 'EF_Bias', 'Dice'
])
df_raw.to_csv('individual_ai_results.csv', index=False)
print(f"\n✅ All Done! Final check: {df_raw['Dice'].count()} Dice values saved.")