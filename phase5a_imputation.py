import pandas as pd
import numpy as np
import os

def process_phase_5a():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    in_path = os.path.join(out_dir, "phase4_5_binned_anadata.parquet")
    print(f"Loading Phase 4.5 binned matrix from {in_path}...")
    df = pd.read_parquet(in_path)
    
    print("\n--- Phase 5a: Structured Imputation ---")
    
    # Exclude columns that are already handled by binning (because their missingness is "Not Recorded")
    # Also exclude free-text clinical notes which are reserved for Phase 5b
    binned_cols = [col for col in df.columns if col.endswith('_Binned') or col == 'Age_Group']
    nlp_cols = ['YAKINMA', 'ÖYKÜ', 'Muayene Notu', 'Tedavi Notu', 'Kontrol Notu', 'Özgeçmiş Notu', 'Soygeçmiş Notu']
    protected_cols = ['HASTA_ID', 'SQ_EPISODE', 'EPISODE_TARIH', 'Is_Dead', 'Days_To_Death', 'OLUM', 'OLUMTARIH']
    missing_flags = [c for c in df.columns if c.startswith('is_') or 'Missing' in c]
    
    exclude_cols = set(binned_cols + nlp_cols + protected_cols + missing_flags)
    impute_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Total features targeted for structured imputation: {len(impute_cols)}")
    
    # Separate numeric and categorical for fallback imputation
    numeric_cols = df[impute_cols].select_dtypes(include=[np.number]).columns
    categorical_cols = df[impute_cols].select_dtypes(exclude=[np.number]).columns
    
    # 1. Median Imputation for strict continuous untouched numeric fields
    if len(numeric_cols) > 0:
        print(f"  Imputing Medians for {len(numeric_cols)} numeric columns...")
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        
    # 2. Mode / Constant Imputation for untouched categoricals
    if len(categorical_cols) > 0:
        print(f"  Imputing 'Unknown' explicit flags for {len(categorical_cols)} categorical columns...")
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        # Cast to standard strings to avoid pyarrow memory overheads from mixed categoricals
        for col in categorical_cols:
            df[col] = df[col].astype(str)

    out_path = os.path.join(out_dir, "phase5a_structured_anadata.parquet")
    print(f"\nPhase 5a Complete. Exporting structured matrix to {out_path}...")
    df.to_parquet(out_path)

if __name__ == "__main__":
    process_phase_5a()
