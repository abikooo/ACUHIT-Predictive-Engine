import pandas as pd
import glob
import os
import json

def process_phase_2():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    
    print("Loading datasets...")
    # Load all anadata files
    anadata_files = glob.glob(os.path.join(out_dir, "master_anadata.parquet", "*.parquet"))
    df_ana = pd.concat([pd.read_parquet(f) for f in anadata_files], ignore_index=True)
    initial_ana_len = len(df_ana)
    print(f"Loaded Anadata: {initial_ana_len} rows")

    # To check linkage, we just need to know which HASTA_IDs exist in Lab and Recete
    # Loading just the HASTA_ID columns to save memory
    lab_files = glob.glob(os.path.join(out_dir, "master_lab.parquet", "*.parquet"))
    recete_files = glob.glob(os.path.join(out_dir, "master_recete.parquet", "*.parquet"))
    
    print("Extracting unique HASTA_IDs from Lab and Recete...")
    lab_hasta_ids = set()
    for f in lab_files:
        df_l = pd.read_parquet(f, columns=['HASTA_ID'])
        lab_hasta_ids.update(df_l['HASTA_ID'].dropna().unique())
        
    recete_hasta_ids = set()
    for f in recete_files:
        df_r = pd.read_parquet(f, columns=['HASTA_ID'])
        recete_hasta_ids.update(df_r['HASTA_ID'].dropna().unique())

    print("\nCalculating Missingness...")
    
    # We define missingness based on "nan", "NaN", "None" strings or actual nulls
    # because they were cast to strings in Phase 1
    null_vals = ["nan", "NaN", "None", ""]
    
    # Vitals columns
    vital_cols = ['Boy', 'Kilo', 'BMI', 'SPO2', 'Nabız', 'KB-S', 'KB-D']
    
    # Clinical columns
    clinical_cols = ['YAKINMA', 'ÖYKÜ', 'Muayene Notu', 'Tedavi Notu', 'Kontrol Notu']
    
    # Lifestyle columns
    lifestyle_cols = ['Sigara', 'Alkol', 'Madde']

    # 1. Per-Group Binary Flags
    df_ana['is_vitals_missing'] = df_ana[vital_cols].isin(null_vals).all(axis=1).astype(int)
    df_ana['is_clinical_notes_missing'] = df_ana[clinical_cols].isin(null_vals).all(axis=1).astype(int)
    df_ana['is_lifestyle_missing'] = df_ana[lifestyle_cols].isin(null_vals).all(axis=1).astype(int)

    # 2. Overall MissingIndex percentage
    # Let's assess missingness across ALL columns except the structural ones
    structural_cols = ['HASTA_ID', 'SQ_EPISODE', 'RF_EPISODE2', 'EPISODE_TARIH', 'SERVISADI']
    assessment_cols = [c for c in df_ana.columns if c not in structural_cols and not c.startswith('is_')]
    
    df_ana['MissingIndex'] = df_ana[assessment_cols].isin(null_vals).mean(axis=1)

    print("\nFiltering Ghost Records...")
    # Cond 1: Missingness > 90%
    cond_high_missing = df_ana['MissingIndex'] > 0.90
    
    # Cond 2: Zero lab linkage
    cond_no_lab = ~df_ana['HASTA_ID'].isin(lab_hasta_ids)
    
    # Cond 3: Zero recete linkage
    cond_no_recete = ~df_ana['HASTA_ID'].isin(recete_hasta_ids)

    # Ghost Mask
    is_ghost = cond_high_missing & cond_no_lab & cond_no_recete
    ghost_count = is_ghost.sum()
    
    df_ana_filtered = df_ana[~is_ghost].copy()
    
    print(f"Total Ghost Records Identified & Dropped: {ghost_count}")
    print(f"Anadata Final Shape: {df_ana_filtered.shape}")
    
    # Parquet strictly requires consistent types per column
    # The 'Ağrı skoru' or others might have mixed string/float due to concat
    print("Enforcing strict string types for object columns before Parquet export...")
    for col in df_ana_filtered.select_dtypes(include=['object']).columns:
        df_ana_filtered[col] = df_ana_filtered[col].astype(str)

    # Save the filtered anadata
    out_path = os.path.join(out_dir, "phase2_anadata.parquet")
    df_ana_filtered.to_parquet(out_path)
    print(f"\nSaved filtered dataset to {out_path}")
    
    # Export summary for the agent
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
