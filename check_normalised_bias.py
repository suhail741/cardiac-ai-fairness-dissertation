import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv("individual_ai_results.csv") 
except FileNotFoundError:
    print("❌ Error: 'individual_ai_results.csv' not found.")
    exit()

# --- 2. CALCULATE NORMALIZED METRICS ---
df['GT_EDV'] = df['EDV'] - df['EDV_Bias']
df['GT_ESV'] = df['ESV'] - df['ESV_Bias']

df['EDV_ml'] = df['EDV_Bias'].abs()
df['ESV_ml'] = df['ESV_Bias'].abs()
df['EF_Abs'] = df['EF_Bias'].abs()

df['EDV_%'] = (df['EDV_ml'] / df['GT_EDV'].replace(0, np.nan)) * 100
df['ESV_%'] = (df['ESV_ml'] / df['GT_ESV'].replace(0, np.nan)) * 100

# Order requested: Dice -> EF -> ml metrics -> % metrics
metrics = ['Dice', 'EF_Abs', 'EDV_ml', 'ESV_ml', 'EDV_%', 'ESV_%']

def run_mwu(group_a, group_b):
    ga, gb = group_a.dropna(), group_b.dropna()
    if len(ga) > 5 and len(gb) > 5:
        stat, p = mannwhitneyu(ga, gb, alternative='two-sided')
        return p
    return np.nan

results = []

# Terminal Formatting
header = f"{'ANALYSIS':<15} | {'MODEL':<15} | {'GRP A':<8} | {'GRP B':<8} | {'METRIC':<14} | {'P-VALUE':<10} | {'SIG'}"
print("\n" + "="*105)
print(header)
print("="*105)

last_analysis, last_model = None, None

def handle_terminal_gaps(current_analysis, current_model):
    global last_analysis, last_model
    if last_analysis is not None and current_analysis != last_analysis:
        print("\n\n" + "#" * 105) 
    elif last_model is not None and current_model != last_model:
        print("\n" + "-" * 105)  
    last_analysis, last_model = current_analysis, current_model

# TASKS
sex_tasks = [205, 206, 204]
age_tasks = [202, 203, 201]
stress_tests = [
    ("Old Only", "M Old", "F Young"),
    ("Young Only", "F Young", "M Old"),
    ("Female Only", "F Old", "M Young"),
    ("Male Only", "M Young", "F Old")
]

def process_loop(task_list, mode):
    for tid in task_list:
        if mode == "sex":
            tname = {205:"Female Only", 206:"Male Only", 204:"Sex Balanced"}[tid]
            ga_name, gb_name = 'Male', 'Female'
        else:
            tname = {202:"Old Only", 203:"Young Only", 201:"Age Balanced"}[tid]
            ga_name, gb_name = 'Young', 'Old'
            
        handle_terminal_gaps(f"{mode.capitalize()} Disparity", tname)
        task_df = df[df['Task'] == tname]
        
        for m in metrics:
            if mode == "sex":
                ga, gb = task_df[task_df['Sex'] == 'M'][m], task_df[task_df['Sex'] == 'F'][m]
            else:
                ga, gb = task_df[task_df['AgeGroup'] == 'Young'][m], task_df[task_df['AgeGroup'] == 'Old'][m]
                
            p = run_mwu(ga, gb)
            sig = "TRUE" if p < 0.05 else "FALSE"
            p_str = f"{p:>10.4e}" if not np.isnan(p) else "N/A"
            print(f"{mode.capitalize()+' Disp':<15} | {tname:<15} | {ga_name:<8} | {gb_name:<8} | {m:<14} | {p_str} | {sig}")
            results.append([f"{mode.capitalize()} Disparity", tname, ga_name, gb_name, m, 
                            round(ga.median(), 3), round(gb.median(), 3), p_str, sig])

process_loop(sex_tasks, "sex")
process_loop(age_tasks, "age")

for tname, l_a, l_b in stress_tests:
    handle_terminal_gaps("Intersectional", tname)
    task_df = df[df['Task'] == tname]
    def get_subset(label):
        s, a = label.split()
        return task_df[(task_df['Sex'] == s) & (task_df['AgeGroup'] == a)]
    
    for m in metrics:
        ga, gb = get_subset(l_a)[m], get_subset(l_b)[m]
        p = run_mwu(ga, gb)
        sig = "TRUE" if p < 0.05 else "FALSE"
        p_str = f"{p:>10.4e}" if not np.isnan(p) else "N/A"
        print(f"{'Intersectional':<15} | {tname:<15} | {l_a:<8} | {l_b:<8} | {m:<14} | {p_str} | {sig}")
        results.append(['Intersectional', tname, l_a, l_b, m, round(ga.median(), 3), round(gb.median(), 3), p_str, sig])

# --- 4. EXPORT WITH GAPS ---
csv_data = []
prev_analysis, prev_model = None, None

for res in results:
    curr_analysis, curr_model = res[0], res[1]
    
    # Insert double gap for new Analysis section
    if prev_analysis is not None and curr_analysis != prev_analysis:
        csv_data.append([None] * 9)
        csv_data.append([None] * 9)
    # Insert single gap for new Model within same section
    elif prev_model is not None and curr_model != prev_model:
        csv_data.append([None] * 9)
    
    csv_data.append(res)
    prev_analysis, prev_model = curr_analysis, curr_model

output_file = "final_normalised_disparity_results_with_gaps.csv"
pd.DataFrame(csv_data, columns=['Analysis','Model','Group_A','Group_B','Metric','Med_A','Med_B','P-Value','Significant']).to_csv(output_file, index=False)

print(f"\n✅ Done! Check your CSV for the reordered metrics and visual gaps: {output_file}")