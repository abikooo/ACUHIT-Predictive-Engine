"""
phase 5c: target engineering + train/test split
------------------------------------------------
100% lazyframe approach to avoid segmentation faults.
uses streaming collect for the final output.
engineers: mortality_label, los_proxy (composite), early_return.
"""
import polars as pl
import numpy as np
import os
import time
import json

OUT_DIR = r"./data/processed"

def run():
    t_start = time.time()
    
    # 1. load deceased ids (proxy from ex_demographics_data raw files)
    print("=== Step 0: Loading mortality proxies ===")
    deceased_path = '/tmp/deceased_ids.json'
    deceased_ids = []
    if os.path.exists(deceased_path):
        with open(deceased_path, 'r') as f:
            deceased_ids = json.load(f)
    print(f"  Loaded {len(deceased_ids)} deceased patient IDs from proxy.")

    # step 1: process phase 2 for early_return and mortality dates
    print("=== Step 1: Building temporal targets (window functions) ===")
    
    # lazy scan of phase 2
    lf_dates = pl.scan_parquet(
        os.path.join(OUT_DIR, "phase2_demographics_data.parquet"),
        low_memory=True
    ).select(["ENCOUNTER_ID", "PATIENT_ID", "ENCOUNTER_DATE", "TOTAL_VISIT_COUNT", "ALL_DIAGNOSES"]).with_columns([
        pl.col("ENCOUNTER_ID").cast(pl.Utf8),
        pl.col("PATIENT_ID").cast(pl.Utf8),
        pl.col("ENCOUNTER_DATE").cast(pl.Utf8).str.to_datetime(strict=False),
    ]).filter(
        pl.col("ENCOUNTER_DATE").is_not_null() 
        & (~pl.col("PATIENT_ID").is_in(["nan", "NaN", "None", ""]))
    ).sort(["PATIENT_ID", "ENCOUNTER_DATE"])
    
    # compute early_return
    lf_dates = lf_dates.with_columns([
        pl.col("ENCOUNTER_DATE").shift(-1).over("PATIENT_ID").alias("next_visit_date")
    ]).with_columns([
        ((pl.col("next_visit_date") - pl.col("ENCOUNTER_DATE")).dt.total_days() <= 30).fill_null(False).cast(pl.Int8).alias("early_return")
    ])

    # compute days_to_death proxy
    # for deceased ids, find max(encounter_date)
    lf_deceased_dates = lf_dates.filter(pl.col("PATIENT_ID").is_in(deceased_ids)).group_by("PATIENT_ID").agg(
        pl.col("ENCOUNTER_DATE").max().alias("death_date_proxy")
    )
    
    lf_dates = lf_dates.join(lf_deceased_dates, on="PATIENT_ID", how="left").with_columns([
        (pl.col("death_date_proxy") - pl.col("ENCOUNTER_DATE")).dt.total_days().alias("Days_To_Death")
    ])

    # terminal icd-10 codes regex (to augment mortality_label)
    terminal_regex = r"(Z51\.5|Z76\.0|I46|R99|J96\.9)"
    
    # finalize target engineering in step 1
    lf_targets = lf_dates.with_columns([
        # mortality_label: 1 if in deceased_ids or contains terminal diagnosis
        pl.when(pl.col("PATIENT_ID").is_in(deceased_ids) | pl.col("ALL_DIAGNOSES").fill_null("").str.contains(terminal_regex))
        .then(1).otherwise(0).cast(pl.Int8).alias("mortality_label")
    ]).with_columns([
        # los_proxy: composite
        pl.when(pl.col("mortality_label") == 1).then(
            pl.when(pl.col("Days_To_Death") <= 30).then(pl.lit("Short (<30 days till death)"))
            .when(pl.col("Days_To_Death") <= 180).then(pl.lit("Medium (1-6 months till death)"))
            .otherwise(pl.lit("Long (>6 months till death)"))
        ).otherwise(
            pl.when(pl.col("TOTAL_VISIT_COUNT").cast(pl.Int64) <= 3).then(pl.lit("Short (1-3 visits)"))
            .when(pl.col("TOTAL_VISIT_COUNT").cast(pl.Int64) <= 10).then(pl.lit("Medium (4-10 visits)"))
            .otherwise(pl.lit("Long (11+ visits)"))
        ).alias("LOS_proxy")
    ]).select(["ENCOUNTER_ID", "early_return", "mortality_label", "LOS_proxy"])
    
    # step 2: main feature matrix + tf-idf join
    print("\n=== Step 2: Joining Main Matrix + TF-IDF + Targets ===")
    
    # main matrix
    lf_main = pl.scan_parquet(os.path.join(OUT_DIR, "pipeline_v2_output.parquet"))
    
    # tf-idf matrix
    lf_tfidf = pl.scan_parquet(os.path.join(OUT_DIR, "phase5b_tfidf.parquet"))
    
    # joint assembly
    lf_final = lf_main.join(lf_tfidf, on="ENCOUNTER_ID", how="left").join(lf_targets, on="ENCOUNTER_ID", how="inner")
    
    # drop raw string columns to save memory in final file
    drop_cols = ["ALL_DIAGNOSES", "Treatment_Note", "Followup_Note", "Past_History_Note", "Family_History_Note", "CHIEF_COMPLAINT", "MEDICAL_HISTORY", "Examination_Note"]
    cols_to_drop = [c for c in drop_cols if c in lf_final.columns]
    lf_final = lf_final.drop(cols_to_drop)

    # step 3: collect and save
    print("\n=== Step 3: Executing and saving final_feature_matrix.parquet ===")
    out_path = os.path.join(OUT_DIR, "final_feature_matrix.parquet")
    
    # using engine="streaming" as recommended in polars >= 1.25.0
    df = lf_final.collect(engine="streaming")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    
    # --- compute dhs using polars native (memory safe) ---
    print("\n  Computing DHS Formula...")
    
    def polars_norm_cap(col_expr):
        c = col_expr.fill_null(0.0)
        p99 = c.quantile(0.99)
        c_capped = pl.when(c > p99).then(p99).otherwise(c)
        c_min = c_capped.min()
        c_max = c_capped.max()
        return pl.when(c_max == c_min).then(pl.lit(0.0)).otherwise((c_capped - c_min) / (c_max - c_min))
        
    vs_exprs = []
    if "SPO2" in df.columns: vs_exprs.append(polars_norm_cap(100.0 - pl.col("SPO2").cast(pl.Float64)))
    if "HEART_RATE" in df.columns: vs_exprs.append(polars_norm_cap(pl.col("HEART_RATE").cast(pl.Float64)))
    if "PAIN_SCORE" in df.columns: vs_exprs.append(polars_norm_cap(pl.col("PAIN_SCORE").cast(pl.Float64)))
    vital_score = pl.sum_horizontal(vs_exprs) if vs_exprs else pl.lit(0.0)

    lab_cols = [pl.col(c).cast(pl.Float64) for c in df.columns if c.startswith("Lab_")]
    lab_score = polars_norm_cap(pl.sum_horizontal(lab_cols)) if lab_cols else pl.lit(0.0)

    comorb_score = polars_norm_cap(pl.col("TOTAL_VISIT_COUNT").cast(pl.Float64)) if "TOTAL_VISIT_COUNT" in df.columns else pl.lit(0.0)

    tfidf_cols = [pl.col(c).cast(pl.Float64) for c in df.columns if c.startswith("TFIDF_")]
    nlp_score = polars_norm_cap(pl.sum_horizontal(tfidf_cols)) if tfidf_cols else pl.lit(0.0)

    visit_score = polars_norm_cap(pl.col("GELIS_SAYISI").cast(pl.Float64)) if "GELIS_SAYISI" in df.columns else pl.lit(0.0)

    rx_score = polars_norm_cap(pl.col("Total_Prescriptions").cast(pl.Float64)) if "Total_Prescriptions" in df.columns else pl.lit(0.0)

    df = df.with_columns([
        ((vital_score + lab_score + comorb_score + nlp_score + visit_score + rx_score) / 6.0 * 100.0).alias("DHS")
    ])

    mean_dhs_alive = df.filter(pl.col("mortality_label") == 0).select(pl.col("DHS").mean()).item()
    mean_dhs_dead = df.filter(pl.col("mortality_label") == 1).select(pl.col("DHS").mean()).item()
    print(f"  Validation - Mean DHS [Alive]: {mean_dhs_alive:.2f} | [Deceased]: {mean_dhs_dead:.2f}")
    
    df.write_parquet(out_path)
    
    # statistics
    print("\nTarget Distributions:")
    print("mortality_label:")
    print(df["mortality_label"].value_counts())
    print("\nLOS_proxy:")
    print(df["LOS_proxy"].value_counts().sort("count", descending=True))
    print("\nearly_return:")
    print(df["early_return"].value_counts())
    
    # step 4: 80/20 stratified train/test split on mortality_label
    print("\n=== Step 4: Stratified Train/Test Split (80/20) ===")
    
    np.random.seed(42)
    pos_df = df.filter(pl.col("mortality_label") == 1)
    neg_df = df.filter(pl.col("mortality_label") == 0)
    
    pos_idx = np.random.permutation(len(pos_df))
    neg_idx = np.random.permutation(len(neg_df))
    
    pos_split = int(0.8 * len(pos_df))
    neg_split = int(0.8 * len(neg_df))
    
    train_df = pl.concat([
        pos_df[pos_idx[:pos_split].tolist()],
        neg_df[neg_idx[:neg_split].tolist()]
    ]).sample(fraction=1.0, seed=42)
    
    test_df = pl.concat([
        pos_df[pos_idx[pos_split:].tolist()],
        neg_df[neg_idx[neg_split:].tolist()]
    ]).sample(fraction=1.0, seed=42)
    
    train_df.write_parquet(os.path.join(OUT_DIR, "train.parquet"))
    test_df.write_parquet(os.path.join(OUT_DIR, "test.parquet"))
    
    print(f"  Train: {train_df.shape[0]:,} rows")
    print(f"  Test:  {test_df.shape[0]:,} rows")
    
    total = time.time() - t_start
    print(f"\nALL COMPLETE in {total:.1f}s")

if __name__ == "__main__":
    run()
