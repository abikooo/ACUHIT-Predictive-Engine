"""
pipeline v2: duckdb + polars
-----------------------------
consolidated phase 3 through 5a in a single, fast script.
target: < 10 minutes on 6m rows.
"""
import polars as pl
import duckdb
import numpy as np
import os
import time
import re

OUT_DIR = r"./data/processed"

def timer(label, t0):
    elapsed = time.time() - t0
    print(f"  [{label}] {elapsed:.1f}s elapsed")
    return time.time()

def parse_lab_result(val: str) -> tuple:
    """parse a single messy lab result string into (numeric_value, result_type)."""
    if val is None or val in ("nan", "NaN", "None", ""):
        return (None, None)
    v = val.strip().upper()
    if "POSITIVE" in v or "POSITIVE" in v:
        return (1.0, "BINARY")
    if "NEGATIVE" in v or "NEGATIVE" in v:
        return (0.0, "BINARY")
    if "---" in v or "***" in v or "ERROR" in v:
        return (None, "ERROR")
    v = v.replace("<", "").replace(">", "").replace("=", "").strip().replace(" ", "")
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", v.replace(",", "."))
    if m:
        try:
            return (float(m.group(0)), "NUMERIC")
        except:
            pass
    return (None, "UNPARSABLE")


def run_pipeline():
    t_start = time.time()
    t = t_start

    # phase 3a-3c: load demographics_data, cast ids, parse dates (polars vectorized)
    print("=== Phase 3a-3c: Load & Clean Anadata ===")
    df_ana = pl.read_parquet(os.path.join(OUT_DIR, "phase2_demographics_data.parquet"))
    t = timer("Load Anadata", t)

    # cast ids to string
    df_ana = df_ana.with_columns([
        pl.col("PATIENT_ID").cast(pl.Utf8),
        pl.col("ENCOUNTER_ID").cast(pl.Utf8),
    ])

    # parse date columns — polars handles mixed formats via str.to_datetime
    date_cols = ['ENCOUNTER_DATE', 'BIRTH_DATE', 'DEATH_DATE', 'DIAGNOSIS_DATE',
                 'MIN_DIAGN_DATE', 'MAX_DIAGN_DATE', 'COMPLAINT_START_DATE', 'START_DATE']
    for col in date_cols:
        if col in df_ana.columns:
            df_ana = df_ana.with_columns(
                pl.col(col).cast(pl.Utf8)
                .str.replace_all("(?i)Ocak", "01")
                .str.replace_all("(?i)Şubat|Subat", "02")
                .str.replace_all("(?i)Mart", "03")
                .str.replace_all("(?i)Nisan", "04")
                .str.replace_all("(?i)Mayıs|Mayis", "05")
                .str.replace_all("(?i)Haziran", "06")
                .str.replace_all("(?i)Temmuz", "07")
                .str.replace_all("(?i)Ağustos|Agustos", "08")
                .str.replace_all("(?i)Eylül|Eylul", "09")
                .str.replace_all("(?i)Ekim", "10")
                .str.replace_all("(?i)Kasım|Kasim", "11")
                .str.replace_all("(?i)Aralık|Aralik", "12")
                .str.to_datetime(strict=False)
                .alias(col)
            )
    t = timer("Parse dates", t)

    # phase 3d-3f: relative time features, temporal binning, drop datetimes
    print("\n=== Phase 3d-3f: Temporal Features ===")
    anchor = pl.col("ENCOUNTER_DATE")

    temporal_exprs = []
    if "BIRTH_DATE" in df_ana.columns:
        temporal_exprs.append(
            ((anchor - pl.col("BIRTH_DATE")).dt.total_days() / 365.25).alias("Age_At_Visit_Years")
        )
    if "MIN_DIAGN_DATE" in df_ana.columns:
        temporal_exprs.append(
            (anchor - pl.col("MIN_DIAGN_DATE")).dt.total_days().clip(0).alias("Days_Since_First_Diagnosis")
        )
    if "MAX_DIAGN_DATE" in df_ana.columns:
        temporal_exprs.append(
            (anchor - pl.col("MAX_DIAGN_DATE")).dt.total_days().clip(0).alias("Days_Since_Last_Diagnosis")
        )
    if "COMPLAINT_START_DATE" in df_ana.columns:
        temporal_exprs.append(
            (anchor - pl.col("COMPLAINT_START_DATE")).dt.total_days().alias("Days_Since_Complaint")
        )
    if "DEATH_DATE" in df_ana.columns:
        temporal_exprs.append(
            (pl.col("DEATH_DATE") - anchor).dt.total_days().alias("Days_To_Death")
        )
        temporal_exprs.append(
            pl.col("DEATH_DATE").is_not_null().cast(pl.Int8).alias("Is_Dead")
        )

    if temporal_exprs:
        df_ana = df_ana.with_columns(temporal_exprs)

    # temporal binning
    bins_diag = [0, 30, 90, 365, 1825, 3650]
    labels_diag = ["<1 Month", "1-3 Months", "3-12 Months", "1-5 Years", "5-10 Years", ">10 Years"]
    if "Days_Since_First_Diagnosis" in df_ana.columns:
        expr = pl.lit("Not Recorded")
        for i in range(len(bins_diag) - 1, 0, -1):
            expr = pl.when(pl.col("Days_Since_First_Diagnosis") <= bins_diag[i]).then(pl.lit(labels_diag[i-1])).otherwise(expr)
        expr = pl.when(pl.col("Days_Since_First_Diagnosis") > bins_diag[-1]).then(pl.lit(labels_diag[-1])).otherwise(expr)
        expr = pl.when(pl.col("Days_Since_First_Diagnosis").is_null()).then(pl.lit("Not Recorded")).otherwise(expr)
        df_ana = df_ana.with_columns(expr.alias("Time_Since_Diagnosis_Bin"))

    # age binning
    if "Age_At_Visit_Years" in df_ana.columns:
        df_ana = df_ana.with_columns(
            pl.when(pl.col("Age_At_Visit_Years").is_null()).then(pl.lit("Not Recorded"))
            .when(pl.col("Age_At_Visit_Years") < 18).then(pl.lit("Pediatric (<18)"))
            .when(pl.col("Age_At_Visit_Years") < 35).then(pl.lit("Young Adult (18-35)"))
            .when(pl.col("Age_At_Visit_Years") < 50).then(pl.lit("Adult (36-50)"))
            .when(pl.col("Age_At_Visit_Years") < 65).then(pl.lit("Older Adult (51-65)"))
            .when(pl.col("Age_At_Visit_Years") < 80).then(pl.lit("Senior (66-80)"))
            .otherwise(pl.lit("Elderly (>80)"))
            .alias("Age_Group")
        )

    # drop raw datetimes
    drop_cols = [c for c in date_cols if c in df_ana.columns]
    df_ana = df_ana.drop(drop_cols)
    t = timer("Temporal features + binning", t)

    # phase 4: relational merging via duckdb engine
    print("\n=== Phase 4: Relational Merging (DuckDB) ===")
    con = duckdb.connect()

    # register demographics_data
    con.register("demographics_data", df_ana.to_arrow())
    
    # --- prescription aggregation ---
    print("  Aggregating Recete...")
    import glob
    rec_paths = glob.glob(os.path.join(OUT_DIR, "master_prescription.parquet", "*.parquet"))
    if rec_paths:
        rec_frames = []
        for rp in rec_paths:
            try:
                rdf = pl.read_parquet(rp)
                ep_col = "ENCOUNTER_ID" if "ENCOUNTER_ID" in rdf.columns else None
                drug_col = next((c for c in rdf.columns if 'la' in c.lower() or 'adi' in c.lower() or 'adı' in c.lower()), None)
                if ep_col:
                    exprs = [pl.col(ep_col).cast(pl.Utf8).alias("ENCOUNTER_ID")]
                    if drug_col:
                        exprs.append(pl.col(drug_col).cast(pl.Utf8).alias("DRUG_NAME"))
                    rdf_clean = rdf.select(exprs).filter(
                        ~pl.col("ENCOUNTER_ID").is_in(["nan", "NaN", "None", ""])
                    )
                    rec_frames.append(rdf_clean)
            except Exception as e:
                print(f"    Skipping {rp}: {e}")
        
        if rec_frames:
            rec_all = pl.concat(rec_frames)
            if "DRUG_NAME" in rec_all.columns:
                prescription_agg = rec_all.group_by("ENCOUNTER_ID").agg([
                    pl.len().alias("Total_Prescriptions"),
                    pl.col("DRUG_NAME").str.to_uppercase().str.contains("PAROL").sum().gt(0).cast(pl.Int8).alias("Has_Parol")
                ])
            else:
                prescription_agg = rec_all.group_by("ENCOUNTER_ID").agg([
                    pl.len().alias("Total_Prescriptions"),
                ])
            
            # pure polars left join
            df_ana = df_ana.join(prescription_agg, left_on="ENCOUNTER_ID", right_on="ENCOUNTER_ID", how="left")
            if "Total_Prescriptions" in df_ana.columns:
                df_ana = df_ana.with_columns(pl.col("Total_Prescriptions").fill_null(0))
            if "Has_Parol" in df_ana.columns:
                df_ana = df_ana.with_columns(pl.col("Has_Parol").fill_null(0))
    t = timer("Recete merge", t)

    # --- lab aggregation ---
    print("  Aggregating Lab (top 50 codes, 7-day window)...")
    # load demographics_data dates separately for the temporal join
    df_ana_dates = pl.read_parquet(
        os.path.join(OUT_DIR, "phase2_demographics_data.parquet"),
        columns=["PATIENT_ID", "ENCOUNTER_ID", "ENCOUNTER_DATE"]
    )
    # parse encounter_date for dates lookup
    df_ana_dates = df_ana_dates.with_columns([
        pl.col("PATIENT_ID").cast(pl.Utf8),
        pl.col("ENCOUNTER_ID").cast(pl.Utf8),
        pl.col("ENCOUNTER_DATE").cast(pl.Utf8).str.to_datetime(strict=False).alias("ENCOUNTER_DATE"),
    ]).filter(
        (~pl.col("PATIENT_ID").is_in(["nan", "NaN", "None", ""])) & pl.col("ENCOUNTER_DATE").is_not_null()
    )
    
    lab_path_str = os.path.join(OUT_DIR, "master_lab.parquet", "*.parquet").replace("\\", "/")

    # find top 50 lab codes via duckdb engine (fast scan)
    top_codes = con.execute(f"""
        SELECT SUB_CODE, COUNT(*) as cnt
        FROM read_parquet('{lab_path_str}')
        WHERE SUB_CODE IS NOT NULL
        GROUP BY SUB_CODE
        ORDER BY cnt DESC
        LIMIT 50
    """).pl()["sub_code"].to_list()
    t = timer("find top 50 lab codes", t)

    # read labs, parse results, join to nearest episode within 7 days
    # process per-file to avoid loading all 156 files into memory at once
    lab_files = glob.glob(os.path.join(out_dir, "master_lab.parquet", "*.parquet"))
    print(f"  processing {len(lab_files)} lab files (chunked join_asof)...")
    
    # pre-sort dates once outside the loop
    dates_sorted = df_ana_dates.sort(["patient_id", "encounter_date"])
    
    joined_chunks = []
    for i, lf in enumerate(lab_files):
        try:
            ldf = pl.read_parquet(lf, columns=["patient_id", "sub_code", "result", "rep_date"])
            ldf = ldf.filter(pl.col("sub_code").is_in(top_codes))
            ldf = ldf.with_columns([
                pl.col("patient_id").cast(pl.utf8),
                pl.col("result").cast(pl.utf8)
                    .str.replace_all(r"[<>=]", "")
                    .str.replace_all(",", ".")
                    .str.strip_chars()
                    .cast(pl.float64, strict=false)
                    .alias("result_numeric"),
                pl.col("rep_date").cast(pl.utf8).str.to_datetime(strict=false).alias("rep_date"),
            ]).filter(
                (~pl.col("patient_id").is_in(["nan", "nan", "none", ""])) & pl.col("rep_date").is_not_null()
            )
            if len(ldf) == 0:
                continue
            
            # sort this chunk and join_asof against the full dates
            ldf_sorted = ldf.sort(["patient_id", "rep_date"])
            chunk_joined = ldf_sorted.join_asof(
                dates_sorted,
                left_on="rep_date",
                right_on="encounter_date",
                by="patient_id",
                strategy="nearest",
                tolerance="7d",
            ).filter(pl.col("sq_episode").is_not_null())
            
            if len(chunk_joined) > 0:
                joined_chunks.append(chunk_joined.select(["sq_episode", "sub_code", "result_numeric"]))
        except exception:
            pass
        
        if (i + 1) % 20 == 0:
            print(f"    processed {i+1}/{len(lab_files)} lab files...")
    
    t = timer("lab chunked asof join", t)
    
    lab_joined = pl.concat(joined_chunks) if joined_chunks else pl.dataframe()

    if joined_chunks and len(lab_joined) > 0:
        lab_pivot = lab_joined.select(["sq_episode", "sub_code", "result_numeric"]).group_by(
            ["sq_episode", "sub_code"]
        ).agg(
            pl.col("result_numeric").mean()
        ).pivot(on="sub_code", index="sq_episode", values="result_numeric")
        # prefix columns
        lab_pivot = lab_pivot.rename({c: f"lab_{c}" for c in lab_pivot.columns if c != "sq_episode"})

        # pure polars left join
        df_ana = df_ana.join(lab_pivot, on="sq_episode", how="left")
    t = timer("lab pivot + join", t)

    # phase 4.5: clinical binning (polars when/then)
    print("\n=== phase 4.5: clinical binning ===")
    
    def bin_vital(col_name, thresholds, labels):
        """build a polars when/then chain for clinical binning."""
        if col_name not in df_ana.columns:
            return None
        expr = pl.when(pl.col(col_name).cast(pl.Float64, strict=False).is_null()).then(pl.lit("Not Recorded"))
        numeric = pl.col(col_name).cast(pl.Float64, strict=False)
        for i, (lo, hi) in enumerate(zip([-float('inf')] + thresholds, thresholds + [float('inf')])):
            if lo == -float('inf'):
                expr = expr.when(numeric <= hi).then(pl.lit(labels[i]))
            elif hi == float('inf'):
                expr = expr.when(numeric > lo).then(pl.lit(labels[i]))
            else:
                expr = expr.when((numeric > lo) & (numeric <= hi)).then(pl.lit(labels[i]))
        return expr.alias(f"{col_name}_Binned")
    
    bin_exprs = [e for e in [
        bin_vital("SPO2", [89.9, 94.9, 100], ["Critical (<90)", "Low (90-94)", "Normal (95-100)", "Over 100"]),
        bin_vital("BMI", [18.49, 24.99, 29.99], ["Underweight", "Normal", "Overweight", "Obese"]),
        bin_vital("HEART_RATE", [59.9, 100], ["Bradycardia (<60)", "Normal (60-100)", "Tachycardia (>100)"]),
        bin_vital("SYSTOLIC_BP", [89.9, 120, 140], ["Hypotension", "Normal", "Pre-HTN", "HTN"]),
        bin_vital("DIASTOLIC_BP", [59.9, 80, 90], ["Hypotension", "Normal", "Pre-HTN", "HTN"]),
        bin_vital("PAIN_SCORE", [0.1, 3.1, 6.1], ["None", "Mild", "Moderate", "Severe"]),
        bin_vital("TOTAL_VISIT_COUNT", [1.1, 5.1, 10.1], ["Single", "Infrequent", "Frequent", "Very Frequent"]),
    ] if e is not None]
    
    if bin_exprs:
        df_ana = df_ana.with_columns(bin_exprs)
    t = timer("Clinical binning", t)

    # phase 5a: structured imputation (polars fill_null)
    print("\n=== Phase 5a: Structured Imputation ===")
    nlp_cols = ['CHIEF_COMPLAINT', 'MEDICAL_HISTORY', 'Examination_Note', 'Treatment_Note', 'Followup_Note', 'Past_History_Note', 'Family_History_Note']
    protected = {'PATIENT_ID', 'ENCOUNTER_ID', 'Is_Dead', 'Days_To_Death'}
    binned = {c for c in df_ana.columns if c.endswith('_Binned') or c == 'Age_Group'}
    exclude = protected | binned | set(nlp_cols)

    # numeric columns: fill with median
    num_cols = [c for c in df_ana.columns if c not in exclude and df_ana[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32)]
    if num_cols:
        df_ana = df_ana.with_columns([pl.col(c).fill_null(pl.col(c).median()) for c in num_cols])

    # string columns: fill with "unknown"
    str_cols = [c for c in df_ana.columns if c not in exclude and df_ana[c].dtype == pl.Utf8]
    if str_cols:
        df_ana = df_ana.with_columns([pl.col(c).fill_null(pl.lit("Unknown")) for c in str_cols])
    t = timer("Imputation", t)

    # save
    out_path = os.path.join(OUT_DIR, "pipeline_v2_output.parquet")
    df_ana.write_parquet(out_path)
    
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline v2 COMPLETE")
    print(f"  Rows: {df_ana.shape[0]:,}")
    print(f"  Columns: {df_ana.shape[1]}")
    print(f"  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_pipeline()
