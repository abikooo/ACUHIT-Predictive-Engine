"""
Phase 5c: Target Engineering + Train/Test Split
------------------------------------------------
100% LazyFrame approach to avoid Segmentation Faults.
Uses streaming collect for the final output.
Engineers: mortality_label, LOS_proxy (composite), early_return.
"""
import polars as pl
import numpy as np
import os
import time
import json

OUT_DIR = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"

def run():
    t_start = time.time()
    
    # 1. Load Deceased IDs (Proxy from Ex_Anadata raw files)
    print("=== Step 0: Loading mortality proxies ===")
    deceased_path = '/tmp/deceased_ids.json'
    deceased_ids = []
    if os.path.exists(deceased_path):
        with open(deceased_path, 'r') as f:
            deceased_ids = json.load(f)
    print(f"  Loaded {len(deceased_ids)} deceased patient IDs from proxy.")

    # =========================================================================
    # STEP 1: Process Phase 2 for early_return and mortality dates
    # =========================================================================
    print("=== Step 1: Building temporal targets (window functions) ===")
    
    # Lazy scan of Phase 2
    lf_dates = pl.scan_parquet(
        os.path.join(OUT_DIR, "phase2_anadata.parquet"),
        low_memory=True
    ).select(["SQ_EPISODE", "HASTA_ID", "EPISODE_TARIH", "TOPLAM_GELIS_SAYISI", "TUM_EPS_TANILAR"]).with_columns([
        pl.col("SQ_EPISODE").cast(pl.Utf8),
        pl.col("HASTA_ID").cast(pl.Utf8),
        pl.col("EPISODE_TARIH").cast(pl.Utf8).str.to_datetime(strict=False),
    ]).filter(
        pl.col("EPISODE_TARIH").is_not_null() 
        & (~pl.col("HASTA_ID").is_in(["nan", "NaN", "None", ""]))
    ).sort(["HASTA_ID", "EPISODE_TARIH"])
    
    # Compute early_return
    lf_dates = lf_dates.with_columns([
        pl.col("EPISODE_TARIH").shift(-1).over("HASTA_ID").alias("next_visit_date")
    ]).with_columns([
        ((pl.col("next_visit_date") - pl.col("EPISODE_TARIH")).dt.total_days() <= 30).fill_null(False).cast(pl.Int8).alias("early_return")
    ])

    # Compute Days_To_Death proxy
    # For deceased IDs, find max(EPISODE_TARIH)
    lf_deceased_dates = lf_dates.filter(pl.col("HASTA_ID").is_in(deceased_ids)).group_by("HASTA_ID").agg(
        pl.col("EPISODE_TARIH").max().alias("death_date_proxy")
    )
    
    lf_dates = lf_dates.join(lf_deceased_dates, on="HASTA_ID", how="left").with_columns([
        (pl.col("death_date_proxy") - pl.col("EPISODE_TARIH")).dt.total_days().alias("Days_To_Death")
    ])

    # Terminal ICD-10 codes regex (to augment mortality_label)
    terminal_regex = r"(Z51\.5|Z76\.0|I46|R99|J96\.9)"
    
    # Finalize Target Engineering in Step 1
    lf_targets = lf_dates.with_columns([
        # mortality_label: 1 if in deceased_ids OR contains terminal diagnosis
        pl.when(pl.col("HASTA_ID").is_in(deceased_ids) | pl.col("TUM_EPS_TANILAR").fill_null("").str.contains(terminal_regex))
        .then(1).otherwise(0).cast(pl.Int8).alias("mortality_label")
    ]).with_columns([
        # LOS_proxy: Composite
        pl.when(pl.col("mortality_label") == 1).then(
            pl.when(pl.col("Days_To_Death") <= 30).then(pl.lit("Short (<30 days till death)"))
            .when(pl.col("Days_To_Death") <= 180).then(pl.lit("Medium (1-6 months till death)"))
            .otherwise(pl.lit("Long (>6 months till death)"))
        ).otherwise(
            pl.when(pl.col("TOPLAM_GELIS_SAYISI").cast(pl.Int64) <= 3).then(pl.lit("Short (1-3 visits)"))
            .when(pl.col("TOPLAM_GELIS_SAYISI").cast(pl.Int64) <= 10).then(pl.lit("Medium (4-10 visits)"))
            .otherwise(pl.lit("Long (11+ visits)"))
        ).alias("LOS_proxy")
    ]).select(["SQ_EPISODE", "early_return", "mortality_label", "LOS_proxy"])
    
    # =========================================================================
    # STEP 2: Main Feature Matrix + TF-IDF Join
    # =========================================================================
    print("\n=== Step 2: Joining Main Matrix + TF-IDF + Targets ===")
    
    # Main matrix
    lf_main = pl.scan_parquet(os.path.join(OUT_DIR, "pipeline_v2_output.parquet"))
    
    # TF-IDF matrix
    lf_tfidf = pl.scan_parquet(os.path.join(OUT_DIR, "phase5b_tfidf.parquet"))
    
    # Joint assembly
    lf_final = lf_main.join(lf_tfidf, on="SQ_EPISODE", how="left").join(lf_targets, on="SQ_EPISODE", how="inner")
    
    # Drop raw string columns to save memory in final file
    drop_cols = ["TUM_EPS_TANILAR", "Tedavi Notu", "Kontrol Notu", "Özgeçmiş Notu", "Soygeçmiş Notu", "YAKINMA", "ÖYKÜ", "Muayene Notu"]
    cols_to_drop = [c for c in drop_cols if c in lf_final.columns]
    lf_final = lf_final.drop(cols_to_drop)

    # =========================================================================
    # STEP 3: Collect and Save
    # =========================================================================
    print("\n=== Step 3: Executing and saving final_feature_matrix.parquet ===")
    out_path = os.path.join(OUT_DIR, "final_feature_matrix.parquet")
    
    # Using engine="streaming" as recommended in Polars >= 1.25.0
    df = lf_final.collect(engine="streaming")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    
    # --- Compute DHS Using Polars Native (Memory Safe) ---
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
    if "Nabız" in df.columns: vs_exprs.append(polars_norm_cap(pl.col("Nabız").cast(pl.Float64)))
    if "Ağrı skoru" in df.columns: vs_exprs.append(polars_norm_cap(pl.col("Ağrı skoru").cast(pl.Float64)))
    vital_score = pl.sum_horizontal(vs_exprs) if vs_exprs else pl.lit(0.0)

    lab_cols = [pl.col(c).cast(pl.Float64) for c in df.columns if c.startswith("Lab_")]
    lab_score = polars_norm_cap(pl.sum_horizontal(lab_cols)) if lab_cols else pl.lit(0.0)

    comorb_score = polars_norm_cap(pl.col("TOPLAM_GELIS_SAYISI").cast(pl.Float64)) if "TOPLAM_GELIS_SAYISI" in df.columns else pl.lit(0.0)

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
    
    # Statistics
    print("\nTarget Distributions:")
    print("mortality_label:")
    print(df["mortality_label"].value_counts())
    print("\nLOS_proxy:")
    print(df["LOS_proxy"].value_counts().sort("count", descending=True))
    print("\nearly_return:")
    print(df["early_return"].value_counts())
    
    # =========================================================================
    # STEP 4: 80/20 Stratified Train/Test Split on mortality_label
    # =========================================================================
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
