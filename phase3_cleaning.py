import pandas as pd
import glob
import os
import re

def parse_turkish_date(date_series):
    """
    Vectorized parsing of Turkish dates.
    """
    months_tr = {
        'Ocak': '01', 'Şubat': '02', 'Mart': '03', 'Nisan': '04', 
        'Mayıs': '05', 'Haziran': '06', 'Temmuz': '07', 'Ağustos': '08', 
        'Eylül': '09', 'Ekim': '10', 'Kasım': '11', 'Aralık': '12',
        'Subat': '02', 'Mayis': '05', 'Agustos': '08', 'Eylul': '09', 'Kasim': '11', 'Aralik': '12'
    }
    
    # Fast path: copy series, replace NaN strings with empty
    s = date_series.copy().astype(str)
    s = s.replace(["nan", "NaN", "None"], "")
    
    # Vectorized string replace for all Turkish months
    for tr, num in months_tr.items():
        # Case insensitive replace to be safe
        s = s.str.replace(tr, num, case=False, regex=False)
        
    # Convert to datetime using standard pandas parser
    return pd.to_datetime(s, errors='coerce')


def extract_numeric_lab_result(val):
    if pd.isna(val) or val in ["nan", "NaN", "None", ""]:
        return pd.NA, pd.NA
        
    val = str(val).strip().upper()
    
    # Text results to binary
    if "POZITIF" in val or "POZİTİF" in val:
        return 1.0, "BINARY"
    if "NEGATIF" in val or "NEGATİF" in val:
        return 0.0, "BINARY"
        
    # Instrument error codes
    if "---" in val or "***" in val or "ERROR" in val:
        return pd.NA, "ERROR"
        
    # Remove inequality signs ("<0.1" -> "0.1", ">5.0" -> "5.0")
    val = val.replace("<", "").replace(">", "").replace("=", "").strip()
    
    # Turkish comma formatting: "1,250" -> "1250" or "4,5" -> "4.5"
    # To handle safely, we check if there are multiple parts
    # e.g., "1.250,50" -> 1250.50
    # In Turkish locale, comma is decimal, dot is thousands. But it could be mixed.
    # We will remove spaces and use regex to pull out the first valid float
    val = val.replace(" ", "")
    
    # Simple regex to grab digits and an optional standard period/comma decimal
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", val.replace(",", "."))
    if match:
        try:
            return float(match.group(0)), "NUMERIC"
        except:
            pass
            
    return pd.NA, "UNPARSABLE"

def process_phase_3():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"

    print("--- Cleaning Anadata ---")
    ana_path = os.path.join(out_dir, "phase2_anadata.parquet")
    df_ana = pd.read_parquet(ana_path)
    
    # IDs inside Anadata
    df_ana['HASTA_ID'] = df_ana['HASTA_ID'].astype(str)
    df_ana['SQ_EPISODE'] = df_ana['SQ_EPISODE'].astype(str)
    
    # Dates inside Anadata
    date_cols = ['EPISODE_TARIH', 'DOGUMTARIHI', 'OLUMTARIH', 'TANITARIH', 'MIN_TANI_TARIH', 'MAX_TANI_TARIH', 'YAKINMA_BASLANGIC_ZAMANI', 'BASLANGIC_ZAMANI']
    for col in date_cols:
        if col in df_ana.columns:
            print(f"  Parsing {col}...")
            df_ana[col] = parse_turkish_date(df_ana[col])
            
    # Save back Anadata
    out_ana_clean = os.path.join(out_dir, "phase3_anadata.parquet")
    df_ana.to_parquet(out_ana_clean)
    print(f"Saved {out_ana_clean}")

    print("\n--- Cleaning Lab Data ---")
    lab_paths = glob.glob(os.path.join(out_dir, "master_lab.parquet", "*.parquet"))
    lab_dfs = []
    
    for f in lab_paths:
        df_l = pd.read_parquet(f)
        
        # ID
        df_l['HASTA_ID'] = df_l['HASTA_ID'].astype(str)
        
        # Parse Dates
        print(f"  Parsing lab dates for {os.path.basename(f)}...")
        df_l['REP_DATE'] = parse_turkish_date(df_l['REP_DATE'])
        
        # Safe Results Extraction
        print(f"  Extracting numeric lab results for {os.path.basename(f)}...")
        # Apply the fast logic
        extracted = df_l['RESULT'].apply(extract_numeric_lab_result)
        # Use simple list comprehensions instead of pd apply for safer type conversion
        numeric_vals = [x[0] if not pd.isna(x[0]) else float('nan') for x in extracted]
        type_vals = [x[1] for x in extracted]
        
        df_l['RESULT_NUMERIC'] = pd.Series(numeric_vals, dtype='float64')
        df_l['RESULT_TYPE'] = pd.Series(type_vals, dtype='string')
        
        # Export individual file out
        out_f = os.path.join(out_dir, "master_lab.parquet", "cleaned_" + os.path.basename(f))
        df_l.to_parquet(out_f)
        lab_dfs.append(df_l)

    print("\n--- Cleaning Recete Data ---")
    rec_paths = glob.glob(os.path.join(out_dir, "master_recete.parquet", "*.parquet"))
    
    for f in rec_paths:
        df_r = pd.read_parquet(f)
        
        # IDs
        df_r['HASTA_ID'] = df_r['HASTA_ID'].astype(str)
        if 'RF_EPISODE' in df_r.columns:
            df_r['RF_EPISODE'] = df_r['RF_EPISODE'].astype(str)
            
        # Parse dates
        print(f"  Parsing recete dates for {os.path.basename(f)}...")
        if 'RECETE_TARIH' in df_r.columns:
            df_r['RECETE_TARIH'] = parse_turkish_date(df_r['RECETE_TARIH'])
            
        out_f = os.path.join(out_dir, "master_recete.parquet", "cleaned_" + os.path.basename(f))
        df_r.to_parquet(out_f)

    print("\nPhase 3 Data Type & Temporal Cleaning Complete.")

if __name__ == "__main__":
    process_phase_3()
