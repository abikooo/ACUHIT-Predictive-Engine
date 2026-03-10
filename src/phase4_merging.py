import pandas as pd
import glob
import os

def process_phase_4():
    out_dir = r"./data/processed"
    
    # 1. load demographics_data base (for the final output)
    ana_path = os.path.join(out_dir, "phase3_temporal_demographics_data.parquet")
    print(f"Loading base Anadata matrix from {ana_path}...")
    df_ana_base = pd.read_parquet(ana_path)
    
    # 2. load demographics_data dates (for lab chronological merging)
    # we need patient_id, sq_episode, and encounter_date
    ana_dates_path = os.path.join(out_dir, "phase3_demographics_data.parquet")
    print(f"Loading Anadata dates mapping from {ana_dates_path}...")
    df_ana_dates = pd.read_parquet(ana_dates_path, columns=['PATIENT_ID', 'ENCOUNTER_ID', 'ENCOUNTER_DATE'])
    
    # filter out missing/string 'nan' ids to avoid cartesian merge explosion
    invalid_ids = ['nan', 'NaN', 'None', '', 'NULL']
    df_ana_dates = df_ana_dates[~df_ana_dates['PATIENT_ID'].astype(str).isin(invalid_ids)].dropna(subset=['PATIENT_ID', 'ENCOUNTER_DATE'])
    
    # sort demographics_data dynamically for merge_asof operation
    df_ana_dates = df_ana_dates.sort_values('ENCOUNTER_DATE')
    
    # we will build two dataframes: prescription_features and lab_features
    # both indexed/keyed by sq_episode
    
    # --- prescription parsing ---
    print("\n--- Aggregating Recete Data ---")
    rec_paths = glob.glob(os.path.join(out_dir, "master_prescription.parquet", "cleaned_*.parquet"))
    rec_counts = []
    
    for f in rec_paths:
        df_r = pd.read_parquet(f)
        if 'ENCOUNTER_ID' not in df_r.columns:
            # fallback if the column is missing or named differently
            continue
            
        # total prescriptions per episode
        counts = df_r.groupby('ENCOUNTER_ID').size().rename('Total_Prescriptions')
        
        # drug flags (find the column that looks like medication name)
        drug_col = next((c for c in df_r.columns if 'LA' in c.upper() or 'ADI' in c.upper()), None)
        parol_counts = pd.Series(0, index=counts.index, name='Has_Parol')
        if drug_col:
            has_parol = df_r[drug_col].astype(str).str.upper().str.contains('PAROL', na=False)
            parol_counts = df_r[has_parol].groupby('ENCOUNTER_ID').size().rename('Has_Parol')
            
        chunk_features = pd.concat([counts, parol_counts], axis=1).fillna(0)
        rec_counts.append(chunk_features)
        
    if rec_counts:
        prescription_features = pd.concat(rec_counts)
        # sum across chunks (in case rf_episode was split across files, though unlikely)
        prescription_features = prescription_features.groupby('ENCOUNTER_ID').sum()
        # ensure parol flag is binary
        prescription_features['Has_Parol'] = (prescription_features['Has_Parol'] > 0).astype(int)
    else:
        prescription_features = pd.DataFrame(columns=['Total_Prescriptions', 'Has_Parol'])

    # --- lab parsing ---
    print("\n--- Aggregating Lab Data ---")
    lab_paths = glob.glob(os.path.join(out_dir, "master_lab.parquet", "cleaned_*.parquet"))
    
    # first pass: find top 50 sub_codes to avoid memory explosion when pivoting
    print("Finding top 50 prevalent lab test codes...")
    subcode_counts = pd.Series(dtype=int)
    for f in lab_paths:
        df_l = pd.read_parquet(f, columns=['SUB_CODE'])
        if 'SUB_CODE' in df_l.columns:
            counts = df_l['SUB_CODE'].value_counts()
            subcode_counts = subcode_counts.add(counts, fill_value=0)
            
    top_50_subcodes = set()
    if not subcode_counts.empty:
        top_50_subcodes = set(subcode_counts.nlargest(50).index)
    print(f"Top 50 Lab codes identified.")
    
    # second pass: extract features for top 50 within +/- 7 days
    lab_records = []
    num_processed = 0
    total_labs = len(lab_paths)
    for f in lab_paths:
        num_processed += 1
        print(f"  Filtering {num_processed}/{total_labs}: {os.path.basename(f)}...")
        df_l = pd.read_parquet(f)
        if 'PATIENT_ID' not in df_l.columns or 'REP_DATE' not in df_l.columns or 'SUB_CODE' not in df_l.columns:
            continue
            
        # filter for top 50 and drop invalid/null
        df_l = df_l[df_l['SUB_CODE'].isin(top_50_subcodes)]
        df_l = df_l[~df_l['PATIENT_ID'].astype(str).isin(invalid_ids)].dropna(subset=['PATIENT_ID', 'REP_DATE'])
        if df_l.empty:
            continue
            
        # sort lab for merge_asof
        df_l = df_l.sort_values('REP_DATE')
            
        # use merge_asof to map each lab row to exactly one nearest clinical episode within 7 days
        # this prevents cartesian explosion if a patient has 5,000 episodes and 1,000 labs
        merged = pd.merge_asof(
            df_l, 
            df_ana_dates,
            left_on='REP_DATE',
            right_on='ENCOUNTER_DATE',
            by='PATIENT_ID',
            direction='nearest',
            tolerance=pd.Timedelta('7D')
        )
        
        valid = merged.dropna(subset=['ENCOUNTER_ID'])
        
        if valid.empty:
            continue
            
        # keep only necessary columns for pivoting
        valid_records = valid[['ENCOUNTER_ID', 'SUB_CODE', 'RESULT_NUMERIC']]
        lab_records.append(valid_records)
        
    print("Pivoting valid lab records into features...")
    if lab_records:
        lab_all = pd.concat(lab_records, ignore_index=True)
        # average results if multiple tests of the same type exist in the 7-day window
        lab_features = lab_all.pivot_table(index='ENCOUNTER_ID', columns='SUB_CODE', values='RESULT_NUMERIC', aggfunc='mean')
        # prefix columns with lab_
        lab_features.columns = [f"Lab_{c}" for c in lab_features.columns]
    else:
        lab_features = pd.DataFrame()

    # --- final merge ---
    print("\n--- Merging Features into Base Anadata ---")
    
    # base demographics_data has sq_episode
    # merge prescription
    df_final = pd.merge(df_ana_base, prescription_features, left_on='ENCOUNTER_ID', right_index=True, how='left')
    
    # fill nas for prescription features with 0 (since meaning is "no prescriptions")
    if 'Total_Prescriptions' in df_final.columns:
        df_final['Total_Prescriptions'] = df_final['Total_Prescriptions'].fillna(0).astype(int)
    if 'Has_Parol' in df_final.columns:
        df_final['Has_Parol'] = df_final['Has_Parol'].fillna(0).astype(int)
        
    # merge labs
    df_final = pd.merge(df_final, lab_features, left_on='ENCOUNTER_ID', right_index=True, how='left')
    
    print(f"Final Merged Anadata shape: {df_final.shape}")
    
    # export
    out_path = os.path.join(out_dir, "phase4_merged_demographics_data.parquet")
    df_final.to_parquet(out_path)
    print(f"Saved Phase 4 final matrix to {out_path}")

if __name__ == "__main__":
    process_phase_4()
