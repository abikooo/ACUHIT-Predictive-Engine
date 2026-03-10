import pandas as pd
import glob
import os

base_dir = r"./data/raw/ACUHIT 2"
pattern = os.path.join(base_dir, "*", "*", "*.csv")
all_files = glob.glob(pattern)

categories = {}
for f in all_files:
    category = os.path.basename(os.path.dirname(f))
    if category not in categories:
        categories[category] = []
    categories[category].append(f)

output_file = r"./analysis_results.md"

with open(output_file, "w", encoding="utf-8") as out:
    out.write("# Data Analysis Results\n\n")
    out.write(f"**Total CSV files found**: {len(all_files)}\n\n")
    out.write(f"**Categories found**: {list(categories.keys())}\n\n")

    for cat, files in categories.items():
        out.write(f"## Category: {cat} (Total files: {len(files)})\n\n")
        
        sample_file = files[0]
        out.write(f"**Sample File**: `{os.path.basename(sample_file)}`\n\n")
        
        try:
            # try comma
            df = pd.read_csv(sample_file, low_memory=False, nrows=1000, encoding_errors='replace')
            if len(df.columns) == 1 and ';' in df.columns[0]:
                df = pd.read_csv(sample_file, sep=";", low_memory=False, nrows=1000, encoding_errors='replace')
                
            out.write(f"**Columns**: `{df.columns.tolist()}`\n\n")
            out.write(f"**Shape of sample read**: `{df.shape}`\n\n")
            
            out.write("**Data Types:**\n")
            for col, dtype in zip(df.columns, df.dtypes):
                out.write(f"- `{col}`: `{dtype}`\n")
            out.write("\n")
                
            out.write("**Null Content (%):**\n")
            nulls = (df.isnull().sum() / len(df) * 100).round(2)
            for col, null_pct in nulls.items():
                if null_pct > 0:
                    out.write(f"- `{col}`: `{null_pct}%`\n")
            out.write("\n")
                    
            out.write("**Head (2 rows):**\n")
            out.write("```text\n")
            out.write(df.head(2).to_string())
            out.write("\n```\n\n")
            
        except Exception as e:
            out.write(f"**Error reading file**: {e}\n\n")
