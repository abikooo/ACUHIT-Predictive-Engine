import pandas as pd
import glob
import os

def consolidate_files():
    base_dir = r"./data/raw/ACUHIT 2"
    out_dir = r"./data/processed"
    os.makedirs(out_dir, exist_ok=True)

    demographics_data_cats = ['Cancer_Anadata', 'Check_Up_Anadata', 'Ex_Anadata']
    lab_cats = ['Cancer_Lab', 'Check_Up_Lab', 'Ex_Lab']
    prescription_cats = ['Cancer_Recete', 'Check_Up_Recete', 'Ex_Recete']

    all_files = glob.glob(os.path.join(base_dir, "*", "*", "*.csv"))
    print(f"Total CSVs found: {len(all_files)}")

    # create destination directories
    for p in ["master_demographics_data.parquet", "master_lab.parquet", "master_prescription.parquet"]:
        os.makedirs(os.path.join(out_dir, p), exist_ok=True)

    print("\nProcessing files in stream mode...")
    
    demographics_data_count = 0
    lab_count = 0
    prescription_count = 0

    for i, f in enumerate(all_files):
        cat = os.path.basename(os.path.dirname(f))
        
        try:
            try:
                df = pd.read_csv(f, low_memory=False)
                if len(df.columns) == 1 and ';' in df.columns[0]:
                    df = pd.read_csv(f, sep=";", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(f, low_memory=False, encoding='iso-8859-9')
                if len(df.columns) == 1 and ';' in df.columns[0]:
                    df = pd.read_csv(f, sep=";", low_memory=False, encoding='iso-8859-9')

            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str)
                
            fname = os.path.basename(f).replace(".csv", ".parquet")

            if cat in demographics_data_cats:
                dest = os.path.join(out_dir, "master_demographics_data.parquet", f"{demographics_data_count}_{fname}")
                df.to_parquet(dest)
                demographics_data_count += 1
            elif cat in lab_cats:
                dest = os.path.join(out_dir, "master_lab.parquet", f"{lab_count}_{fname}")
                df.to_parquet(dest)
                lab_count += 1
            elif cat in prescription_cats:
                dest = os.path.join(out_dir, "master_prescription.parquet", f"{prescription_count}_{fname}")
                df.to_parquet(dest)
                prescription_count += 1
            else:
                print(f"Unknown category: {cat} for file {f}")
                
            if i % 10 == 0:
                print(f"Processed {i}/{len(all_files)} files...")
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"\nPhase 1 Consolidation Complete! Wrote {demographics_data_count} demographics_data, {lab_count} lab, {prescription_count} prescription files.")

    print("\nPhase 1 Consolidation Complete.")

if __name__ == "__main__":
    consolidate_files()
