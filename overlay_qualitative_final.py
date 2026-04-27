import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED_PATH  = "/data/se26/nnUNet/nnUNet_results/predictions_independent/Task201_results"
IMG_PATH   = "/data/se26/nnUNet/test_independent/imagesTs"
GT_PATH    = "/data/se26/nnUNet/test_independent/labelsTs"
OUTPUT_DIR = "overlay_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

patients = [
    ("S002179", "ED", 1,  "M", 0.940, "Under-segmentation"),
    ("S000589", "ED", 1,  "F", 0.938, "Over-segmentation"),
    ("S002113", "ED", 1,  "M", 0.923, "Boundary Mismatch"),
]

def get_mid_z(gt):
    valid_slices = []
    for z in range(gt.shape[2]):
        count = np.sum(gt[:,:,z] == 1)
        if count >= 100:
            valid_slices.append((z, count))
    sorted_slices = sorted(valid_slices, key=lambda x: x[1], reverse=True)
    return sorted_slices[0][0]

def get_slice_dice(gt, pred, z):
    gt_lv   = (gt[:,:,z] == 1)
    pred_lv = (pred[:,:,z] == 1)
    intersection = np.sum(gt_lv & pred_lv)
    total = np.sum(gt_lv) + np.sum(pred_lv)
    return (2*intersection)/total if total > 0 else 0.0

def make_row(axes, img, gt, pred, z, row_label, dice):
    img_slice   = img[:, :, z]
    img_display = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    gt_lv   = (gt[:, :, z] == 1).astype(float)
    pred_lv = (pred[:, :, z] == 1).astype(float)

    for ax in axes:
        ax.imshow(img_display.T, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_facecolor('black')

    axes[0].set_title(f'{row_label}\nMRI Only',
                      color='white', fontsize=10, fontweight='bold', pad=6)

    if np.sum(gt_lv) > 0:
        axes[1].contour(gt_lv.T, levels=[0.5], colors=['#00FF00'], linewidths=[2])
    axes[1].plot([], [], color='#00FF00', linewidth=2, label='Clinician')
    axes[1].set_title(f'{row_label}\nGround Truth',
                      color='white', fontsize=10, fontweight='bold', pad=6)
    axes[1].legend(loc='lower right', fontsize=8, framealpha=0.5,
                   facecolor='black', labelcolor='white')

    if np.sum(gt_lv) > 0:
        axes[2].contour(gt_lv.T,   levels=[0.5], colors=['#00FF00'], linewidths=[2])
    if np.sum(pred_lv) > 0:
        axes[2].contour(pred_lv.T, levels=[0.5], colors=['#FF4444'], linewidths=[2])
    axes[2].plot([], [], color='#00FF00', linewidth=2, label='Clinician')
    axes[2].plot([], [], color='#FF4444', linewidth=2, label='AI')
    axes[2].set_title(f'{row_label}\nAI vs Clinician | Dice={dice:.3f}',
                      color='white', fontsize=10, fontweight='bold', pad=6)
    axes[2].legend(loc='lower right', fontsize=8, framealpha=0.5,
                   facecolor='black', labelcolor='white')

for sid, frame, basal_z, sex, overall_dice, failure_type in patients:
    img  = nib.load(os.path.join(IMG_PATH,  f"{sid}_{frame}_0000.nii.gz")).get_fdata()
    gt   = np.round(nib.load(os.path.join(GT_PATH,   f"{sid}_{frame}.nii.gz")).get_fdata()).astype(int)
    pred = np.round(nib.load(os.path.join(PRED_PATH, f"{sid}_{frame}.nii.gz")).get_fdata()).astype(int)

    mid_z      = get_mid_z(gt)
    basal_dice = get_slice_dice(gt, pred, basal_z)
    mid_dice   = get_slice_dice(gt, pred, mid_z)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('black')

    make_row(axes[0], img, gt, pred, mid_z,
             f'Mid-ventricular (z={mid_z})', mid_dice)
    make_row(axes[1], img, gt, pred, basal_z,
             f'Basal (z={basal_z}) — {failure_type}', basal_dice)

    fig.suptitle(
        f"Patient {sid} ({sex}, Old) | Frame: {frame} | "
        f"Overall Dice={overall_dice:.3f} | "
        f"Mid Dice={mid_dice:.3f} | Basal Dice={basal_dice:.3f}",
        color='white', fontsize=12, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR,
        f"{sid}_{frame}_{failure_type.replace(' ','_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {out_path}")

print(f"Done. Saved to '{OUTPUT_DIR}/'")
