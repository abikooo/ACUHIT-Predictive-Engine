import pandas as pd
import os
import numpy as np

def process_phase_3_temporal():
    """
    Sub-phases 3d, 3e, 3f for Temporal Feature Engineering.
    Converts raw datetimes into usable relative numerical and categorical ML features.
    """
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    ana_path = os.path.join(out_dir, "phase3_anadata.parquet")
    
    print(f"Loading Anadata from {ana_path}...")
    df_ana = pd.read_parquet(ana_path)
    
    # 3d: Relative Time Computation
    print("3d: Computing relative elapsed times...")
    
    # Anchor dates: EPISODE_TARIH is the anchor for most visit-level metrics
    anchor = df_ana['EPISODE_TARIH']
    
    def compute_days(dt_col):
        # Difference in days, returned as float. Handles NaT returning NaN automatically.
        return (anchor - df_ana[dt_col]).dt.total_seconds() / (24 * 3600)

    # Calculate Days Since metrics
    if 'MIN_TANI_TARIH' in df_ana.columns:
        df_ana['Days_Since_First_Diagnosis'] = compute_days('MIN_TANI_TARIH')
        # Negative days (diagnosis after visit) are anomalies or data entry errors, cap at 0
        df_ana['Days_Since_First_Diagnosis'] = df_ana['Days_Since_First_Diagnosis'].clip(lower=0)

    if 'MAX_TANI_TARIH' in df_ana.columns:
        df_ana['Days_Since_Last_Diagnosis'] = compute_days('MAX_TANI_TARIH')
        df_ana['Days_Since_Last_Diagnosis'] = df_ana['Days_Since_Last_Diagnosis'].clip(lower=0)
        
    if 'DOGUMTARIHI' in df_ana.columns:
        df_ana['Age_At_Visit_Days'] = compute_days('DOGUMTARIHI')
        df_ana['Age_At_Visit_Years'] = df_ana['Age_At_Visit_Days'] / 365.25

    if 'YAKINMA_BASLANGIC_ZAMANI' in df_ana.columns:
        df_ana['Days_Since_Complaint'] = compute_days('YAKINMA_BASLANGIC_ZAMANI')
        
    if 'BASLANGIC_ZAMANI' in df_ana.columns:
        df_ana['Days_Since_Onset'] = compute_days('BASLANGIC_ZAMANI')

    # Construct Survival Targets (Do not use this as an input feature for ML!)
    if 'OLUMTARIH' in df_ana.columns:
        # Days from this visit to death
        df_ana['Days_To_Death'] = (df_ana['OLUMTARIH'] - anchor).dt.total_seconds() / (24 * 3600)
        # Mortality flag (1 if dead, 0 if alive)
        df_ana['Is_Dead'] = df_ana['OLUMTARIH'].notna().astype(int)

    # 3e: Temporal Binning
    print("3e: Binning elapsed times into clinical categories...")
    
    # Define bins mimicking clinical timelines
    bins_days = [-np.inf, 30, 90, 365, 365*5, 365*10, np.inf]
    labels_days = ['<1 Month', '1-3 Months', '3-12 Months', '1-5 Years', '5-10 Years', '>10 Years']
    
    if 'Days_Since_First_Diagnosis' in df_ana.columns:
        df_ana['Time_Since_Diagnosis_Bin'] = pd.cut(df_ana['Days_Since_First_Diagnosis'], 
                                                    bins=bins_days, labels=labels_days)
        # Add explicit string category for missing values
        df_ana['Time_Since_Diagnosis_Bin'] = df_ana['Time_Since_Diagnosis_Bin'].astype(str)
        df_ana.loc[df_ana['Days_Since_First_Diagnosis'].isna(), 'Time_Since_Diagnosis_Bin'] = 'Not Recorded'
        
    if 'Age_At_Visit_Years' in df_ana.columns:
        bins_age = [-np.inf, 18, 35, 50, 65, 80, np.inf]
        labels_age = ['Pediatric (<18)', 'Young Adult (18-35)', 'Adult (36-50)', 'Older Adult (51-65)', 'Senior (66-80)', 'Elderly (>80)']
        df_ana['Age_Group'] = pd.cut(df_ana['Age_At_Visit_Years'], bins=bins_age, labels=labels_age)
        df_ana['Age_Group'] = df_ana['Age_Group'].astype(str)
        df_ana.loc[df_ana['Age_At_Visit_Years'].isna(), 'Age_Group'] = 'Not Recorded'

    # 3f: Drop Raw Datetimes
    print("3f: Dropping raw datetime columns to protect feature matrix...")
    datetime_cols = ['EPISODE_TARIH', 'DOGUMTARIHI', 'OLUMTARIH', 'TANITARIH', 'MIN_TANI_TARIH', 'MAX_TANI_TARIH', 'YAKINMA_BASLANGIC_ZAMANI', 'BASLANGIC_ZAMANI']
    
    cols_to_drop = [c for c in datetime_cols if c in df_ana.columns]
    df_ana_final = df_ana.drop(columns=cols_to_drop)
    
    print(f"Dropped exactly {len(cols_to_drop)} raw datetime columns: {cols_to_drop}")

    # Export
    out_path = os.path.join(out_dir, "phase3_temporal_anadata.parquet")
    df_ana_final.to_parquet(out_path)
    print(f"\nSaved final temporal matrix to {out_path}")
    print(f"Final Anadata shape: {df_ana_final.shape}")

if __name__ == "__main__":
    process_phase_3_temporal()
