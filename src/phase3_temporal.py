import pandas as pd
import os
import numpy as np

def process_phase_3_temporal():
    """
    sub-phases 3d, 3e, 3f for temporal feature engineering.
    converts raw datetimes into usable relative numerical and categorical ml features.
    """
    out_dir = r"./data/processed"
    ana_path = os.path.join(out_dir, "phase3_demographics_data.parquet")
    
    print(f"Loading Anadata from {ana_path}...")
    df_ana = pd.read_parquet(ana_path)
    
    # 3d: relative time computation
    print("3d: Computing relative elapsed times...")
    
    # anchor dates: encounter_date is the anchor for most visit-level metrics
    anchor = df_ana['ENCOUNTER_DATE']
    
    def compute_days(dt_col):
        # difference in days, returned as float. handles nat returning nan automatically.
        return (anchor - df_ana[dt_col]).dt.total_seconds() / (24 * 3600)

    # calculate days since metrics
    if 'MIN_DIAGN_DATE' in df_ana.columns:
        df_ana['Days_Since_First_Diagnosis'] = compute_days('MIN_DIAGN_DATE')
        # negative days (diagnosis after visit) are anomalies or data entry errors, cap at 0
        df_ana['Days_Since_First_Diagnosis'] = df_ana['Days_Since_First_Diagnosis'].clip(lower=0)

    if 'MAX_DIAGN_DATE' in df_ana.columns:
        df_ana['Days_Since_Last_Diagnosis'] = compute_days('MAX_DIAGN_DATE')
        df_ana['Days_Since_Last_Diagnosis'] = df_ana['Days_Since_Last_Diagnosis'].clip(lower=0)
        
    if 'BIRTH_DATE' in df_ana.columns:
        df_ana['Age_At_Visit_Days'] = compute_days('BIRTH_DATE')
        df_ana['Age_At_Visit_Years'] = df_ana['Age_At_Visit_Days'] / 365.25

    if 'COMPLAINT_START_DATE' in df_ana.columns:
        df_ana['Days_Since_Complaint'] = compute_days('COMPLAINT_START_DATE')
        
    if 'START_DATE' in df_ana.columns:
        df_ana['Days_Since_Onset'] = compute_days('START_DATE')

    # construct survival targets (do not use this as an input feature for ml!)
    if 'DEATH_DATE' in df_ana.columns:
        # days from this visit to death
        df_ana['Days_To_Death'] = (df_ana['DEATH_DATE'] - anchor).dt.total_seconds() / (24 * 3600)
        # mortality flag (1 if dead, 0 if alive)
        df_ana['Is_Dead'] = df_ana['DEATH_DATE'].notna().astype(int)

    # 3e: temporal binning
    print("3e: Binning elapsed times into clinical categories...")
    
    # define bins mimicking clinical timelines
    bins_days = [-np.inf, 30, 90, 365, 365*5, 365*10, np.inf]
    labels_days = ['<1 Month', '1-3 Months', '3-12 Months', '1-5 Years', '5-10 Years', '>10 Years']
    
    if 'Days_Since_First_Diagnosis' in df_ana.columns:
        df_ana['Time_Since_Diagnosis_Bin'] = pd.cut(df_ana['Days_Since_First_Diagnosis'], 
                                                    bins=bins_days, labels=labels_days)
        # add explicit string category for missing values
        df_ana['Time_Since_Diagnosis_Bin'] = df_ana['Time_Since_Diagnosis_Bin'].astype(str)
        df_ana.loc[df_ana['Days_Since_First_Diagnosis'].isna(), 'Time_Since_Diagnosis_Bin'] = 'Not Recorded'
        
    if 'Age_At_Visit_Years' in df_ana.columns:
        bins_age = [-np.inf, 18, 35, 50, 65, 80, np.inf]
        labels_age = ['Pediatric (<18)', 'Young Adult (18-35)', 'Adult (36-50)', 'Older Adult (51-65)', 'Senior (66-80)', 'Elderly (>80)']
        df_ana['Age_Group'] = pd.cut(df_ana['Age_At_Visit_Years'], bins=bins_age, labels=labels_age)
        df_ana['Age_Group'] = df_ana['Age_Group'].astype(str)
        df_ana.loc[df_ana['Age_At_Visit_Years'].isna(), 'Age_Group'] = 'Not Recorded'

    # 3f: drop raw datetimes
    print("3f: Dropping raw datetime columns to protect feature matrix...")
    datetime_cols = ['ENCOUNTER_DATE', 'BIRTH_DATE', 'DEATH_DATE', 'DIAGNOSIS_DATE', 'MIN_DIAGN_DATE', 'MAX_DIAGN_DATE', 'COMPLAINT_START_DATE', 'START_DATE']
    
    cols_to_drop = [c for c in datetime_cols if c in df_ana.columns]
    df_ana_final = df_ana.drop(columns=cols_to_drop)
    
    print(f"Dropped exactly {len(cols_to_drop)} raw datetime columns: {cols_to_drop}")

    # export
    out_path = os.path.join(out_dir, "phase3_temporal_demographics_data.parquet")
    df_ana_final.to_parquet(out_path)
    print(f"\nSaved final temporal matrix to {out_path}")
    print(f"Final Anadata shape: {df_ana_final.shape}")

if __name__ == "__main__":
    process_phase_3_temporal()
