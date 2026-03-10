import pandas as pd
import numpy as np
import os

def bin_metric(df, col_name, bins, labels):
    if col_name not in df.columns:
        return
        
    # convert to numeric first
    numeric_series = pd.to_numeric(df[col_name], errors='coerce')
    
    # bin
    new_col = f"{col_name}_Binned"
    df[new_col] = pd.cut(numeric_series, bins=bins, labels=labels)
    
    # keep as memory-efficient category type, explicitly fill nans
    df[new_col] = df[new_col].cat.add_categories('Not Recorded')
    df[new_col] = df[new_col].fillna('Not Recorded')


def process_phase_4_5():
    out_dir = r"./data/processed"
    in_path = os.path.join(out_dir, "phase4_merged_demographics_data.parquet")
    print(f"Loading merged Anadata from {in_path}...")
    df = pd.read_parquet(in_path)
    
    print("\n--- Binning Clinical Vitals & Metrics ---")
    
    # spo2: <90 critical, 90-94 low, 95-100 normal
    bin_metric(df, 'SPO2', bins=[-np.inf, 89.9, 94.9, 100, np.inf], 
               labels=['Critical (<90)', 'Low (90-94)', 'Normal (95-100)', 'Over 100 (Error)'])
               
    # bmi: <18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, >=30 obese
    bin_metric(df, 'BMI', bins=[-np.inf, 18.49, 24.99, 29.99, np.inf], 
               labels=['Underweight (<18.5)', 'Normal (18.5-24.9)', 'Overweight (25-29.9)', 'Obese (>=30)'])
               
    # heart rate (nabız): <60 bradycardia, 60-100 normal, >100 tachycardia
    bin_metric(df, 'HEART_RATE', bins=[-np.inf, 59.9, 100, np.inf], 
               labels=['Bradycardia (<60)', 'Normal (60-100)', 'Tachycardia (>100)'])
               
    # systolic bp (kb-s): <90 hypotension, 90-120 normal, 120-140 prehtn, >140 htn
    bin_metric(df, 'SYSTOLIC_BP', bins=[-np.inf, 89.9, 120, 140, np.inf], 
               labels=['Hypotension (<90)', 'Normal (90-120)', 'Pre-Hypertension (120-140)', 'Hypertension (>140)'])
               
    # diastolic bp (kb-d): <60 hypotension, 60-80 normal, 80-90 prehtn, >90 htn
    bin_metric(df, 'DIASTOLIC_BP', bins=[-np.inf, 59.9, 80, 90, np.inf], 
               labels=['Hypotension (<60)', 'Normal (60-80)', 'Pre-Hypertension (80-90)', 'Hypertension (>90)'])
               
    # pain score (ağrı skoru): 0 none, 1-3 mild, 4-6 moderate, 7-10 severe
    bin_metric(df, 'PAIN_SCORE', bins=[-np.inf, 0.1, 3.1, 6.1, np.inf], 
               labels=['None (0)', 'Mild (1-3)', 'Moderate (4-6)', 'Severe (7-10)'])
               
    # visit frequency (toplam_gelis_sayisi): 1 single, 2-5 infreq, 6-10 freq, >10 very freq
    bin_metric(df, 'TOTAL_VISIT_COUNT', bins=[-np.inf, 1.1, 5.1, 10.1, np.inf], 
               labels=['Single Visit (1)', 'Infrequent (2-5)', 'Frequent (6-10)', 'Very Frequent (>10)'])

    print("\n--- Calculating & Binning Comorbidity Burden ---")
    comorbid_cols = ['Hipertansiyon Hastada', 'Kalp Damar Hastada', 'Diyabet Hastada']
    existing_cols = [c for c in comorbid_cols if c in df.columns]
    
    if existing_cols:
        # sum boolean indicators (assuming they are encoded as '1', '1.0', or floats)
        burden = pd.Series(0, index=df.index)
        for c in existing_cols:
            is_present = df[c].astype(str).str.contains('1', na=False).astype(int)
            burden += is_present
            
        df['Comorbidity_Burden_Score'] = burden
        # 0: none, 1: mild, 2: moderate, 3: severe
        df['Comorbidity_Burden_Binned'] = pd.cut(burden, bins=[-np.inf, 0.1, 1.1, 2.1, np.inf],
                                                 labels=['None (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)'])
        # already has no nans because we accumulated ints, but doing it safely anyway
        if df['Comorbidity_Burden_Binned'].isna().any():
            df['Comorbidity_Burden_Binned'] = df['Comorbidity_Burden_Binned'].cat.add_categories('Not Recorded').fillna('Not Recorded')

    # labs reference bands:
    # since we dropped explicit refmin/refmax during the phase 4 pivot to avoid cartesian explosions,
    # mapping lab metrics back against individual exact bands is computationally prohibitive
    # we will instead calculate statistical quantiles for the top 50 lab features as a proxy for "normal/low/high"
    print("\n--- Quantiling Lab Results as Clinical Proxy ---")
    lab_cols = [c for c in df.columns if c.startswith('Lab_')]
    for lc in lab_cols:
        new_col = f"{lc}_Binned"
        # 0-25%: low, 25-75%: normal, 75-100%: high
        try:
            df[new_col] = pd.qcut(df[lc], q=[0, 0.25, 0.75, 1.0], labels=['Statistically Low', 'Statistically Normal', 'Statistically High'])
            df[new_col] = df[new_col].cat.add_categories('Not Recorded').fillna('Not Recorded')
        except Exception:
            # dropdown fails if too many identical values (e.g. all 0s)
            df[new_col] = pd.Categorical(['Not Recorded'] * len(df))

    out_path = os.path.join(out_dir, "phase4_5_binned_demographics_data.parquet")
    print(f"\nPhase 4.5 Complete. Saving {df.shape[1]} features to {out_path}...")
    df.to_parquet(out_path)

if __name__ == "__main__":
    process_phase_4_5()
