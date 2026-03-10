import pandas as pd
import glob
import os
import json

def process_phase_2():
    out_dir = r"./data/processed"
    
    print("Loading datasets...")
    # load all demographics_data files
    demographics_data_files = glob.glob(os.path.join(out_dir, "master_demographics_data.parquet", "*.parquet"))
    df_ana = pd.concat([pd.read_parquet(f) for f in demographics_data_files], ignore_index=True)
    initial_ana_len = len(df_ana)
    print(f"Loaded Anadata: {initial_ana_len} rows")

    # to check linkage, we just need to know which patient_ids exist in lab and prescription
    # loading just the patient_id columns to save memory
    lab_files = glob.glob(os.path.join(out_dir, "master_lab.parquet", "*.parquet"))
    prescription_files = glob.glob(os.path.join(out_dir, "master_prescription.parquet", "*.parquet"))
    
    print("Extracting unique PATIENT_IDs from Lab and Recete...")
    lab_patient_ids = set()
    for f in lab_files:
        df_l = pd.read_parquet(f, columns=['PATIENT_ID'])
        lab_patient_ids.update(df_l['PATIENT_ID'].dropna().unique())
        
    prescription_patient_ids = set()
    for f in prescription_files:
        df_r = pd.read_parquet(f, columns=['PATIENT_ID'])
        prescription_patient_ids.update(df_r['PATIENT_ID'].dropna().unique())

    print("\nCalculating Missingness...")
    
    # we define missingness based on "nan", "nan", "none" strings or actual nulls
    # because they were cast to strings in phase 1
    null_vals = ["nan", "NaN", "None", ""]
    
    # vitals columns
    vital_cols = ['Boy', 'Kilo', 'BMI', 'SPO2', 'HEART_RATE', 'SYSTOLIC_BP', 'DIASTOLIC_BP']
    
    # clinical columns
    clinical_cols = ['CHIEF_COMPLAINT', 'MEDICAL_HISTORY', 'Examination_Note', 'Treatment_Note', 'Followup_Note']
    
    # lifestyle columns
    lifestyle_cols = ['Sigara', 'Alkol', 'Madde']

    # 1. per-group binary flags
    df_ana['is_vitals_missing'] = df_ana[vital_cols].isin(null_vals).all(axis=1).astype(int)
    df_ana['is_clinical_notes_missing'] = df_ana[clinical_cols].isin(null_vals).all(axis=1).astype(int)
    df_ana['is_lifestyle_missing'] = df_ana[lifestyle_cols].isin(null_vals).all(axis=1).astype(int)

    # 2. overall missingindex percentage
    # let's assess missingness across all columns except the structural ones
    structural_cols = ['PATIENT_ID', 'ENCOUNTER_ID', 'ENCOUNTER_ID2', 'ENCOUNTER_DATE', 'SERVISADI']
    assessment_cols = [c for c in df_ana.columns if c not in structural_cols and not c.startswith('is_')]
    
    df_ana['MissingIndex'] = df_ana[assessment_cols].isin(null_vals).mean(axis=1)

    print("\nFiltering Ghost Records...")
    # cond 1: missingness > 90%
    cond_high_missing = df_ana['MissingIndex'] > 0.90
    
    # cond 2: zero lab linkage
    cond_no_lab = ~df_ana['PATIENT_ID'].isin(lab_patient_ids)
    
    # cond 3: zero prescription linkage
    cond_no_prescription = ~df_ana['PATIENT_ID'].isin(prescription_patient_ids)

    # ghost mask
    is_ghost = cond_high_missing & cond_no_lab & cond_no_prescription
    ghost_count = is_ghost.sum()
    
    df_ana_filtered = df_ana[~is_ghost].copy()
    
    print(f"Total Ghost Records Identified & Dropped: {ghost_count}")
    print(f"Anadata Final Shape: {df_ana_filtered.shape}")
    
    # parquet strictly requires consistent types per column
    # the 'ağrı skoru' or others might have mixed string/float due to concat
    print("Enforcing strict string types for object columns before Parquet export...")
    for col in df_ana_filtered.select_dtypes(include=['object']).columns:
        df_ana_filtered[col] = df_ana_filtered[col].astype(str)

    # save the filtered demographics_data
    out_path = os.path.join(out_dir, "phase2_demographics_data.parquet")
    df_ana_filtered.to_parquet(out_path)
    print(f"\nSaved filtered dataset to {out_path}")
    
    # export summary for the agent
    summary = {
        "Initial Anadata": int(initial_ana_len),
        "Final Anadata": int(len(df_ana_filtered)),
        "Ghosts Dropped": int(ghost_count),
        "Ghosts Percentage": round((ghost_count / initial_ana_len) * 100, 2)
    }
    with open(os.path.join(out_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    process_phase_2()
