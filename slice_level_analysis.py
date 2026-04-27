import pandas as pd
import nibabel as nib
import numpy as np
import os
from scipy import stats

df = pd.read_csv('individual_ai_results.csv')
CORRUPTED = ['S000292','S000659','S001169','S002310','S002413','S003670']

GT_PATH   = '/data/se26/nnUNet/test_independent/labelsTs'
PRED_PATH = '/data/se26/nnUNet/nnUNet_results/predictions_independent/Task201_results'

age_balanced = df[
    (df['Task'] == 'Age Balanced') &
    (~df['sid'].isin(CORRUPTED))
][['sid','AgeGroup']].drop_duplicates()

young_basal, old_basal = [], []
young_mid,   old_mid   = [], []
young_apex,  old_apex  = [], []

excluded_orientation = 0
excluded_basal       = 0
included_basal       = 0

for _, row in age_balanced.iterrows():
    sid       = row['sid']
    age_group = row['AgeGroup']

    for frame in ['ED']:
        gt_file   = os.path.join(GT_PATH,   f'{sid}_{frame}.nii.gz')
        pred_file = os.path.join(PRED_PATH, f'{sid}_{frame}.nii.gz')
        if not all(os.path.exists(f) for f in [gt_file, pred_file]):
            continue

        img  = nib.load(gt_file)
        ornt = nib.orientations.io_orientation(img.affine)
        code = nib.orientations.ornt2axcodes(ornt)

        if code != ('L', 'A', 'S'):
            excluded_orientation += 1
            continue

        gt   = np.round(img.get_fdata()).astype(int)
        pred = np.round(nib.load(pred_file).get_fdata()).astype(int)

        valid_slices = []
        for z in range(gt.shape[2]):
            count = np.sum(gt[:,:,z] == 1)
            if count >= 100:
                valid_slices.append((z, count))

        if len(valid_slices) < 5:
            continue

        sorted_by_count = sorted(valid_slices, key=lambda x: x[1], reverse=True)
        mid_slices      = sorted_by_count[:2]
        mid_z_indices   = set([z for z, _ in mid_slices])
        avg_mid_vox     = np.mean([v for _, v in mid_slices])

        candidate_basal = valid_slices[:2]
        avg_basal_vox   = np.mean([v for _, v in candidate_basal])
        basal_z_indices = set([z for z, _ in candidate_basal])

        apex_slices    = valid_slices[-2:]
        apex_z_indices = set([z for z, _ in apex_slices])

        if avg_basal_vox >= avg_mid_vox or basal_z_indices & mid_z_indices:
            excluded_basal += 1
        else:
            included_basal += 1
            for z, _ in candidate_basal:
                gt_lv   = (gt[:,:,z] == 1)
                pred_lv = (pred[:,:,z] == 1)
                intersection = np.sum(gt_lv & pred_lv)
                total = np.sum(gt_lv) + np.sum(pred_lv)
                dice = (2*intersection)/total if total > 0 else 0.0
                if age_group == 'Young':
                    young_basal.append(dice)
                else:
                    old_basal.append(dice)

        if not mid_z_indices & apex_z_indices:
            for z, _ in mid_slices:
                gt_lv   = (gt[:,:,z] == 1)
                pred_lv = (pred[:,:,z] == 1)
                intersection = np.sum(gt_lv & pred_lv)
                total = np.sum(gt_lv) + np.sum(pred_lv)
                dice = (2*intersection)/total if total > 0 else 0.0
                if age_group == 'Young':
                    young_mid.append(dice)
                else:
                    old_mid.append(dice)

        if not apex_z_indices & mid_z_indices:
            for z, _ in apex_slices:
                gt_lv   = (gt[:,:,z] == 1)
                pred_lv = (pred[:,:,z] == 1)
                intersection = np.sum(gt_lv & pred_lv)
                total = np.sum(gt_lv) + np.sum(pred_lv)
                dice = (2*intersection)/total if total > 0 else 0.0
                if age_group == 'Young':
                    young_apex.append(dice)
                else:
                    old_apex.append(dice)

print(f'Orientation filter : excluded {excluded_orientation} non-LAS patients')
print(f'Basal verification : {included_basal} confirmed, {excluded_basal} excluded')
print()

n_basal = min(len(young_basal), len(old_basal))
n_mid   = min(len(young_mid),   len(old_mid))
n_apex  = min(len(young_apex),  len(old_apex))

np.random.seed(42)
yb = np.random.choice(young_basal, n_basal, replace=False)
ob = np.array(old_basal[:n_basal])
ym = np.random.choice(young_mid,   n_mid,   replace=False)
om = np.array(old_mid[:n_mid])
ya = np.random.choice(young_apex,  n_apex,  replace=False)
oa = np.array(old_apex[:n_apex])

print('CONFIRMED BASAL SLICES (LAS orientation only)')
print(f'  N per group : {n_basal}')
print(f'  Young median: {np.median(yb):.4f}')
print(f'  Old median  : {np.median(ob):.4f}')
print(f'  Gap         : {np.median(yb) - np.median(ob):.4f}')
stat, p = stats.mannwhitneyu(yb, ob, alternative='greater')
print(f'  p-value     : {p:.6f}')

print()
print('MID-VENTRICULAR SLICES (maximum voxel count per patient)')
print(f'  N per group : {n_mid}')
print(f'  Young median: {np.median(ym):.4f}')
print(f'  Old median  : {np.median(om):.4f}')
print(f'  Gap         : {np.median(ym) - np.median(om):.4f}')
stat, p = stats.mannwhitneyu(ym, om, alternative='greater')
print(f'  p-value     : {p:.6f}')

print()
print('APICAL SLICES (last 2 valid slices per patient)')
print(f'  N per group : {n_apex}')
print(f'  Young median: {np.median(ya):.4f}')
print(f'  Old median  : {np.median(oa):.4f}')
print(f'  Gap         : {np.median(ya) - np.median(oa):.4f}')
stat, p = stats.mannwhitneyu(ya, oa, alternative='greater')
print(f'  p-value     : {p:.6f}')
