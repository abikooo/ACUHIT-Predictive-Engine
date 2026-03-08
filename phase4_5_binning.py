import pandas as pd
import numpy as np
import os

def bin_metric(df, col_name, bins, labels):
    if col_name not in df.columns:
        return
        
    # Convert to numeric first
    numeric_series = pd.to_numeric(df[col_name], errors='coerce')
    
    # Bin
    new_col = f"{col_name}_Binned"
    df[new_col] = pd.cut(numeric_series, bins=bins, labels=labels)
    
    # Keep as memory-efficient Category type, explicitly fill NaNs
    df[new_col] = df[new_col].cat.add_categories('Not Recorded')
    df[new_col] = df[new_col].fillna('Not Recorded')


def process_phase_4_5():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    in_path = os.path.join(out_dir, "phase4_merged_anadata.parquet")
    print(f"Loading merged Anadata from {in_path}...")
    df = pd.read_parquet(in_path)
    
    print("\n--- Binning Clinical Vitals & Metrics ---")
    
    # SPO2: <90 Critical, 90-94 Low, 95-100 Normal
    bin_metric(df, 'SPO2', bins=[-np.inf, 89.9, 94.9, 100, np.inf], 
               labels=['Critical (<90)', 'Low (90-94)', 'Normal (95-100)', 'Over 100 (Error)'])
               
    # BMI: <18.5 Underweight, 18.5-24.9 Normal, 25-29.9 Overweight, >=30 Obese
    bin_metric(df, 'BMI', bins=[-np.inf, 18.49, 24.99, 29.99, np.inf], 
               labels=['Underweight (<18.5)', 'Normal (18.5-24.9)', 'Overweight (25-29.9)', 'Obese (>=30)'])
               
    # Heart Rate (Nabız): <60 Bradycardia, 60-100 Normal, >100 Tachycardia
    bin_metric(df, 'Nabız', bins=[-np.inf, 59.9, 100, np.inf], 
               labels=['Bradycardia (<60)', 'Normal (60-100)', 'Tachycardia (>100)'])
               
    # Systolic BP (KB-S): <90 Hypotension, 90-120 Normal, 120-140 PreHTN, >140 HTN
    bin_metric(df, 'KB-S', bins=[-np.inf, 89.9, 120, 140, np.inf], 
               labels=['Hypotension (<90)', 'Normal (90-120)', 'Pre-Hypertension (120-140)', 'Hypertension (>140)'])
               
    # Diastolic BP (KB-D): <60 Hypotension, 60-80 Normal, 80-90 PreHTN, >90 HTN
    bin_metric(df, 'KB-D', bins=[-np.inf, 59.9, 80, 90, np.inf], 
               labels=['Hypotension (<60)', 'Normal (60-80)', 'Pre-Hypertension (80-90)', 'Hypertension (>90)'])
               
    # Pain Score (Ağrı skoru): 0 None, 1-3 Mild, 4-6 Moderate, 7-10 Severe
    bin_metric(df, 'Ağrı skoru', bins=[-np.inf, 0.1, 3.1, 6.1, np.inf], 
               labels=['None (0)', 'Mild (1-3)', 'Moderate (4-6)', 'Severe (7-10)'])
               
    # Visit Frequency (TOPLAM_GELIS_SAYISI): 1 Single, 2-5 Infreq, 6-10 Freq, >10 Very Freq
    bin_metric(df, 'TOPLAM_GELIS_SAYISI', bins=[-np.inf, 1.1, 5.1, 10.1, np.inf], 
               labels=['Single Visit (1)', 'Infrequent (2-5)', 'Frequent (6-10)', 'Very Frequent (>10)'])

    print("\n--- Calculating & Binning Comorbidity Burden ---")
    comorbid_cols = ['Hipertansiyon Hastada', 'Kalp Damar Hastada', 'Diyabet Hastada']
    existing_cols = [c for c in comorbid_cols if c in df.columns]
    
    if existing_cols:
        # Sum boolean indicators (assuming they are encoded as '1', '1.0', or floats)
        burden = pd.Series(0, index=df.index)
        for c in existing_cols:
            is_present = df[c].astype(str).str.contains('1', na=False).astype(int)
            burden += is_present
            
        df['Comorbidity_Burden_Score'] = burden
        # 0: None, 1: Mild, 2: Moderate, 3: Severe
        df['Comorbidity_Burden_Binned'] = pd.cut(burden, bins=[-np.inf, 0.1, 1.1, 2.1, np.inf],
                                                 labels=['None (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)'])
        # Already has no NaNs because we accumulated ints, but doing it safely anyway
        if df['Comorbidity_Burden_Binned'].isna().any():
            df['Comorbidity_Burden_Binned'] = df['Comorbidity_Burden_Binned'].cat.add_categories('Not Recorded').fillna('Not Recorded')

    # Labs Reference Bands:
    # Since we dropped explicit REFMIN/REFMAX during the Phase 4 pivot to avoid cartesian explosions,
    # mapping Lab metrics back against individual exact bands is computationally prohibitive
    # We will instead calculate statistical quantiles for the top 50 lab features as a proxy for "Normal/Low/High"
    print("\n--- Quantiling Lab Results as Clinical Proxy ---")
    lab_cols = [c for c in df.columns if c.startswith('Lab_')]
    for lc in lab_cols:
        new_col = f"{lc}_Binned"
        # 0-25%: Low, 25-75%: Normal, 75-100%: High
        try:
            df[new_col] = pd.qcut(df[lc], q=[0, 0.25, 0.75, 1.0], labels=['Statistically Low', 'Statistically Normal', 'Statistically High'])
            df[new_col] = df[new_col].cat.add_categories('Not Recorded').fillna('Not Recorded')
        except Exception:
            # Dropdown fails if too many identical values (e.g. all 0s)
            df[new_col] = pd.Categorical(['Not Recorded'] * len(df))

    out_path = os.path.join(out_dir, "phase4_5_binned_anadata.parquet")
    print(f"\nPhase 4.5 Complete. Saving {df.shape[1]} features to {out_path}...")
    df.to_parquet(out_path)

if __name__ == "__main__":
    process_phase_4_5()
