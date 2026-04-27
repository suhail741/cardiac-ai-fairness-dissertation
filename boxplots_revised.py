import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
CSV_FILE   = "individual_ai_results.csv"
OUTPUT_DIR = "boxplots_revised"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD AND PREPARE DATA ---
df = pd.read_csv(CSV_FILE)
df['GT_EDV'] = df['EDV'] - df['EDV_Bias']
df['GT_ESV'] = df['ESV'] - df['ESV_Bias']
df['EDV_ml'] = df['EDV_Bias'].abs()
df['ESV_ml'] = df['ESV_Bias'].abs()
df['EF_Abs'] = df['EF_Bias'].abs()
df['EDV_%']  = (df['EDV_ml'] / df['GT_EDV'].replace(0, np.nan)) * 100
df['ESV_%']  = (df['ESV_ml'] / df['GT_ESV'].replace(0, np.nan)) * 100

CORRUPTED = ['S000292', 'S000659', 'S001169', 'S002310', 'S002413', 'S003670']
df = df[~df['sid'].isin(CORRUPTED)]

# --- STYLE ---
sns.set_theme(style="whitegrid")
PALETTE_SEX = {"M": "#3498db", "F": "#e74c3c"}
PALETTE_AGE = {"Young": "#27ae60", "Old": "#8e44ad"}

SEX_MODEL_ORDER = ["Female Only", "Male Only", "Sex Balanced"]
AGE_MODEL_ORDER = ["Old Only", "Young Only", "Age Balanced"]

def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {path}")

def whisker_top(data):
    """Calculate the top whisker position (Q3 + 1.5*IQR, capped at max)"""
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    return min(upper_fence, data.max())

def add_sig(ax, x, y, label):
    color  = 'black' if label != 'ns' else '#888888'
    size   = 13 if label in ['*', '**', '***'] else 10
    weight = 'bold' if label != 'ns' else 'normal'
    ax.text(x, y, label, ha='center', va='bottom',
            fontsize=size, color=color, fontweight=weight)

# ─────────────────────────────────────────────
# FIGURE 1 — EDV_ml, Male vs Female, Sex Models
# ─────────────────────────────────────────────
df_sex = df[df['Task'].isin(SEX_MODEL_ORDER)].copy()

fig, ax = plt.subplots(figsize=(11, 6))

sns.boxplot(
    data=df_sex, x='Task', y='EDV_ml', hue='Sex',
    order=SEX_MODEL_ORDER, hue_order=['M', 'F'],
    palette=PALETTE_SEX, showfliers=False, ax=ax, width=0.6
)

# Get y max across all groups to set axis limit
all_tops = []
for task in SEX_MODEL_ORDER:
    td = df_sex[df_sex['Task'] == task]
    all_tops.append(whisker_top(td[td['Sex'] == 'M']['EDV_ml'].dropna()))
    all_tops.append(whisker_top(td[td['Sex'] == 'F']['EDV_ml'].dropna()))
y_max = max(all_tops)
ax.set_ylim(-0.5, y_max * 1.25)

for i, task in enumerate(SEX_MODEL_ORDER):
    td = df_sex[df_sex['Task'] == task]
    top = max(
        whisker_top(td[td['Sex'] == 'M']['EDV_ml'].dropna()),
        whisker_top(td[td['Sex'] == 'F']['EDV_ml'].dropna())
    )
    add_sig(ax, i, top + y_max * 0.03, '*')

ax.set_title('EDV Absolute Error by Sex\nApparent Male Disadvantage',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Training Model', fontsize=11)
ax.set_ylabel('EDV Absolute Error (ml)', fontsize=11)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Male', 'Female'], title='Sex', fontsize=10, framealpha=0.9)
ax.text(0.99, 0.01, '* p<0.05', transform=ax.transAxes,
        fontsize=9, color='grey', ha='right', va='bottom')

plt.tight_layout()
save_plot(fig, "fig1_EDV_ml_sex.png")

# ─────────────────────────────────────────────
# FIGURE 2 — EDV_%, Male vs Female, Sex Models
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

sns.boxplot(
    data=df_sex, x='Task', y='EDV_%', hue='Sex',
    order=SEX_MODEL_ORDER, hue_order=['M', 'F'],
    palette=PALETTE_SEX, showfliers=False, ax=ax, width=0.6
)

all_tops = []
for task in SEX_MODEL_ORDER:
    td = df_sex[df_sex['Task'] == task]
    all_tops.append(whisker_top(td[td['Sex'] == 'M']['EDV_%'].dropna()))
    all_tops.append(whisker_top(td[td['Sex'] == 'F']['EDV_%'].dropna()))
y_max = max(all_tops)
ax.set_ylim(-0.2, y_max * 1.3)

for i, task in enumerate(SEX_MODEL_ORDER):
    td = df_sex[df_sex['Task'] == task]
    top = max(
        whisker_top(td[td['Sex'] == 'M']['EDV_%'].dropna()),
        whisker_top(td[td['Sex'] == 'F']['EDV_%'].dropna())
    )
    add_sig(ax, i, top + y_max * 0.04, 'ns')

ax.set_title('EDV Normalised Percentage Error by Sex\nBias Disappears After Normalisation',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Training Model', fontsize=11)
ax.set_ylabel('EDV Error (%)', fontsize=11)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Male', 'Female'], title='Sex', fontsize=10, framealpha=0.9)
ax.text(0.99, 0.01, 'ns = not significant', transform=ax.transAxes,
        fontsize=9, color='grey', ha='right', va='bottom')

plt.tight_layout()
save_plot(fig, "fig2_EDV_pct_sex.png")

# ─────────────────────────────────────────────
# FIGURE 3 — Dice, Young vs Old, Age Models
# ─────────────────────────────────────────────
df_age = df[df['Task'].isin(AGE_MODEL_ORDER)].copy()

fig, ax = plt.subplots(figsize=(11, 6))

sns.boxplot(
    data=df_age, x='Task', y='Dice', hue='AgeGroup',
    order=AGE_MODEL_ORDER, hue_order=['Young', 'Old'],
    palette=PALETTE_AGE, showfliers=False, ax=ax, width=0.6
)

ax.axhline(y=0.93, color='red', linestyle='--', linewidth=1.5, zorder=5)
ax.set_ylim(0.75, 1.01)

for i in range(3):
    add_sig(ax, i, 0.984, '***')

ax.set_title('Dice Score by Age Group\nPersistent Bias Across All Models',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Training Model', fontsize=11)
ax.set_ylabel('Dice Score', fontsize=11)
handles, _ = ax.get_legend_handles_labels()
ax.legend(
    handles + [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5)],
    ['Young', 'Old', 'Old median (0.93)'],
    title='Age Group', fontsize=10, loc='lower right', framealpha=0.9
)
ax.text(0.99, 0.01, '*** p<0.001', transform=ax.transAxes,
        fontsize=9, color='grey', ha='right', va='bottom')

plt.tight_layout()
save_plot(fig, "fig3_Dice_age.png")

# ─────────────────────────────────────────────
# FIGURE 4 — EF_Abs, Young vs Old, Age Models
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

sns.boxplot(
    data=df_age, x='Task', y='EF_Abs', hue='AgeGroup',
    order=AGE_MODEL_ORDER, hue_order=['Young', 'Old'],
    palette=PALETTE_AGE, showfliers=False, ax=ax, width=0.6
)

sig_ef = ['ns', '*', 'ns']
all_tops = []
for task in AGE_MODEL_ORDER:
    td = df_age[df_age['Task'] == task]
    all_tops.append(whisker_top(td[td['AgeGroup'] == 'Young']['EF_Abs'].dropna()))
    all_tops.append(whisker_top(td[td['AgeGroup'] == 'Old']['EF_Abs'].dropna()))
y_max = max(all_tops)
ax.set_ylim(-0.2, y_max * 1.3)

for i, task in enumerate(AGE_MODEL_ORDER):
    td = df_age[df_age['Task'] == task]
    top = max(
        whisker_top(td[td['AgeGroup'] == 'Young']['EF_Abs'].dropna()),
        whisker_top(td[td['AgeGroup'] == 'Old']['EF_Abs'].dropna())
    )
    add_sig(ax, i, top + y_max * 0.04, sig_ef[i])

ax.set_title('Ejection Fraction Absolute Error by Age Group\nBalancing Rescues EF Accuracy',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Training Model', fontsize=11)
ax.set_ylabel('EF Absolute Error (percentage points)', fontsize=11)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Young', 'Old'], title='Age Group', fontsize=10, framealpha=0.9)
ax.text(0.99, 0.01, '* p<0.05    ns = not significant',
        transform=ax.transAxes, fontsize=9, color='grey', ha='right', va='bottom')

plt.tight_layout()
save_plot(fig, "fig4_EF_Abs_age.png")

# ─────────────────────────────────────────────
# FIGURE 5 — EDV_%, Young vs Old, Age Models
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

sns.boxplot(
    data=df_age, x='Task', y='EDV_%', hue='AgeGroup',
    order=AGE_MODEL_ORDER, hue_order=['Young', 'Old'],
    palette=PALETTE_AGE, showfliers=False, ax=ax, width=0.6
)

sig_edv = ['ns', '*', '**']
all_tops = []
for task in AGE_MODEL_ORDER:
    td = df_age[df_age['Task'] == task]
    all_tops.append(whisker_top(td[td['AgeGroup'] == 'Young']['EDV_%'].dropna()))
    all_tops.append(whisker_top(td[td['AgeGroup'] == 'Old']['EDV_%'].dropna()))
y_max = max(all_tops)
ax.set_ylim(-0.2, y_max * 1.3)

for i, task in enumerate(AGE_MODEL_ORDER):
    td = df_age[df_age['Task'] == task]
    top = max(
        whisker_top(td[td['AgeGroup'] == 'Young']['EDV_%'].dropna()),
        whisker_top(td[td['AgeGroup'] == 'Old']['EDV_%'].dropna())
    )
    add_sig(ax, i, top + y_max * 0.04, sig_edv[i])

ax.set_title('EDV Normalised Percentage Error by Age Group\nPartial Improvement via Balancing',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Training Model', fontsize=11)
ax.set_ylabel('EDV Error (%)', fontsize=11)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Young', 'Old'], title='Age Group', fontsize=10, framealpha=0.9)
ax.text(0.99, 0.01, '** p<0.01    * p<0.05    ns = not significant',
        transform=ax.transAxes, fontsize=9, color='grey', ha='right', va='bottom')

plt.tight_layout()
save_plot(fig, "fig5_EDV_pct_age.png")

print(f"\n🎉 All 5 boxplots saved to '{OUTPUT_DIR}/'")