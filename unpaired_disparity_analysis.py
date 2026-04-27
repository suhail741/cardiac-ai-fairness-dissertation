import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv("individual_ai_results.csv") 
except FileNotFoundError:
    print("❌ Error: 'individual_ai_results.csv' not found. Run the Master Loop first.")
    exit()

# Ensure Absolute Error columns exist for volume/clinical metrics
df['EF_Abs_Error']  = df['EF_Bias'].abs()
df['EDV_Abs_Error'] = df['EDV_Bias'].abs()
df['ESV_Abs_Error'] = df['ESV_Bias'].abs()

# Metrics to analyze: Absolute clinical errors + Raw Dice
metrics = ['EF_Abs_Error', 'EDV_Abs_Error', 'ESV_Abs_Error', 'Dice']

def run_mwu(group_a, group_b):
    if len(group_a) > 5 and len(group_b) > 5:
        stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
        return p
    return np.nan

results = []

# Terminal Header
header = f"{'ANALYSIS':<15} | {'MODEL':<15} | {'GRP A':<8} | {'GRP B':<8} | {'METRIC':<14} | {'P-VALUE':<10} | {'SIG'}"
print("\n" + "="*95)
print(header)
print("="*95)

# --- HELPER TO ADD GAPS ---
last_analysis = None
last_model = None

def handle_gaps(current_analysis, current_model):
    global last_analysis, last_model
    if last_analysis is not None and current_analysis != last_analysis:
        print("\n\n" + "#" * 95) 
    elif last_model is not None and current_model != last_model:
        print("\n" + "-" * 95)  
    last_analysis = current_analysis
    last_model = current_model

# --- 2. DEFINE THE TEST GROUPS ---
sex_tasks = [205, 206, 204]
age_tasks = [202, 203, 201]
stress_tests = [
    ("Old Only", "M Old", "F Young"),
    ("Young Only", "F Young", "M Old"),
    ("Female Only", "F Old", "M Young"),
    ("Male Only", "M Young", "F Old")
]

# --- 3. RUN ANALYSES ---
def process_loop(task_list, mode):
    for tid in task_list:
        if mode == "sex":
            tname = {205:"Female Only", 206:"Male Only", 204:"Sex Balanced"}[tid]
            ga_name, gb_name = 'Male', 'Female'
        else:
            tname = {202:"Old Only", 203:"Young Only", 201:"Age Balanced"}[tid]
            ga_name, gb_name = 'Young', 'Old'
            
        handle_gaps(f"{mode.capitalize()} Disparity", tname)
        task_df = df[df['Task'] == tname]
        
        for m in metrics:
            if mode == "sex":
                ga, gb = task_df[task_df['Sex'] == 'M'][m].dropna(), task_df[task_df['Sex'] == 'F'][m].dropna()
            else:
                ga, gb = task_df[task_df['AgeGroup'] == 'Young'][m].dropna(), task_df[task_df['AgeGroup'] == 'Old'][m].dropna()
                
            p = run_mwu(ga, gb)
            sig = "TRUE" if p < 0.05 else "FALSE"
            
            p_str = f"{p:>10.4e}" if not np.isnan(p) else "N/A"
            print(f"{mode.capitalize()+' Disp':<15} | {tname:<15} | {ga_name:<8} | {gb_name:<8} | {m:<14} | {p_str} | {sig}")
            
            # Record results (Median is more robust for Dice/Bias)
            results.append([f"{mode.capitalize()} Disparity", tname, ga_name, gb_name, m, 
                            round(ga.median(), 3), round(gb.median(), 3), p_str, sig])

process_loop(sex_tasks, "sex")
process_loop(age_tasks, "age")

# Intersectional Loop
for tname, label_a, label_b in stress_tests:
    handle_gaps("Intersectional", tname)
    task_df = df[df['Task'] == tname]
    def get_subset(label):
        s, a = label.split()
        return task_df[(task_df['Sex'] == s) & (task_df['AgeGroup'] == a)]
    
    for m in metrics:
        ga, gb = get_subset(label_a)[m].dropna(), get_subset(label_b)[m].dropna()
        p = run_mwu(ga, gb)
        sig = "TRUE" if p < 0.05 else "FALSE"
        p_str = f"{p:>10.4e}" if not np.isnan(p) else "N/A"
        print(f"{'Intersectional':<15} | {tname:<15} | {label_a:<8} | {label_b:<8} | {m:<14} | {p_str} | {sig}")
        results.append(['Intersectional', tname, label_a, label_b, m, 
                        round(ga.median(), 3), round(gb.median(), 3), p_str, sig])

print("="*95)

# --- 4. EXPORT ---
csv_data = []
prev_analysis, prev_model = None, None

for res in results:
    curr_analysis, curr_model = res[0], res[1]
    if prev_analysis is not None and curr_analysis != prev_analysis:
        csv_data.append([None] * 9)
        csv_data.append([None] * 9)
    elif prev_model is not None and curr_model != prev_model:
        csv_data.append([None] * 9)
    
    csv_data.append(res)
    prev_analysis, prev_model = curr_analysis, curr_model

final_df = pd.DataFrame(csv_data, columns=['Analysis', 'Model', 'Group_A', 'Group_B', 'Metric', 'Med_A', 'Med_B', 'P-Value', 'Significant'])
final_df.to_csv("unpaired_disparity_results.csv", index=False)

print(f"\n✅ Analysis Complete! Results saved to 'unpaired_disparity_results.csv'")