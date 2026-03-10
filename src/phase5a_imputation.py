import pandas as pd
import numpy as np
import os

def process_phase_5a():
    out_dir = r"./data/processed"
    in_path = os.path.join(out_dir, "phase4_5_binned_demographics_data.parquet")
    print(f"Loading Phase 4.5 binned matrix from {in_path}...")
    df = pd.read_parquet(in_path)
    
    print("\n--- Phase 5a: Structured Imputation ---")
    
    # exclude columns that are already handled by binning (because their missingness is "not recorded")
    # also exclude free-text clinical notes which are reserved for phase 5b
    binned_cols = [col for col in df.columns if col.endswith('_Binned') or col == 'Age_Group']
    nlp_cols = ['CHIEF_COMPLAINT', 'MEDICAL_HISTORY', 'Examination_Note', 'Treatment_Note', 'Followup_Note', 'Past_History_Note', 'Family_History_Note']
    protected_cols = ['PATIENT_ID', 'ENCOUNTER_ID', 'ENCOUNTER_DATE', 'Is_Dead', 'Days_To_Death', 'OLUM', 'DEATH_DATE']
    missing_flags = [c for c in df.columns if c.startswith('is_') or 'Missing' in c]
    
    exclude_cols = set(binned_cols + nlp_cols + protected_cols + missing_flags)
    impute_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Total features targeted for structured imputation: {len(impute_cols)}")
    
    # separate numeric and categorical for fallback imputation
    numeric_cols = df[impute_cols].select_dtypes(include=[np.number]).columns
    categorical_cols = df[impute_cols].select_dtypes(exclude=[np.number]).columns
    
    # 1. median imputation for strict continuous untouched numeric fields
    if len(numeric_cols) > 0:
        print(f"  Imputing Medians for {len(numeric_cols)} numeric columns...")
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        
    # 2. mode / constant imputation for untouched categoricals
    if len(categorical_cols) > 0:
        print(f"  Imputing 'Unknown' explicit flags for {len(categorical_cols)} categorical columns...")
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        # cast to standard strings to avoid pyarrow memory overheads from mixed categoricals
        for col in categorical_cols:
            df[col] = df[col].astype(str)

    out_path = os.path.join(out_dir, "phase5a_structured_demographics_data.parquet")
    print(f"\nPhase 5a Complete. Exporting structured matrix to {out_path}...")
    df.to_parquet(out_path)

if __name__ == "__main__":
    process_phase_5a()
