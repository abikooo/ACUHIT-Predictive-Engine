import duckdb
import json
import os

def run():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    pv2_path = os.path.join(out_dir, "pipeline_v2_output.parquet")
    p2_path = os.path.join(out_dir, "phase2_anadata.parquet")
    out_path = os.path.join(out_dir, "final_feature_matrix.parquet")

    con = duckdb.connect()
    
    # Configure out-of-core memory management to prevent OOM on 6M row join
    tmp_path = os.path.join(out_dir, "duckdb_tmp")
    os.makedirs(tmp_path, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp_path}'")
    con.execute("PRAGMA memory_limit='12GB'")

    # 1. Inspect pipeline_v2_output
    cols_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pv2_path}')").df()
    cols = cols_df['column_name'].tolist()

    lab_cols = [c for c in cols if c.startswith('Lab_')]
    
    # 2. Find top 10 most populated Lab_* columns
    counts = {}
    for c in lab_cols:
        cnt = con.execute(f"SELECT COUNT(\"{c}\") FROM read_parquet('{pv2_path}')").fetchone()[0]
        counts[c] = cnt
    
    top_10 = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)[:10]

    # 3. Calculate mean and std
    stats = {}
    for c in top_10:
        try:
            m, s = con.execute(f"SELECT AVG(TRY_CAST(\"{c}\" AS DOUBLE)), STDDEV_POP(TRY_CAST(\"{c}\" AS DOUBLE)) FROM read_parquet('{pv2_path}')").fetchone()
        except Exception as e:
            print(f"Stats parse error on {c}: {e}")
            m, s = 0.0, 1.0
            
        m = m if m is not None else 0.0
        s = s if s is not None and s > 0 else 1.0
        stats[c] = (m, s)

    # Build RMS expr
    rms_parts = []
    for c in top_10:
        m, s = stats[c]
        rms_parts.append(f"COALESCE(POWER((TRY_CAST(\"{c}\" AS DOUBLE) - {m}) / {s}, 2), 0)")
    
    rms_expr = f"SQRT( ({' + '.join(rms_parts)}) / 10.0 )" if rms_parts else "0.0"

    # 4. ICD10 logic
    icd10_case_str = """CASE
        WHEN substr(TANIKODU, 1, 1) IN ('A', 'B') THEN 'Infectious'
        WHEN substr(TANIKODU, 1, 1) = 'C' THEN 'Neoplasms'
        WHEN substr(TANIKODU, 1, 1) = 'D' THEN 'Blood'
        WHEN substr(TANIKODU, 1, 1) = 'E' THEN 'Endocrine'
        WHEN substr(TANIKODU, 1, 1) = 'F' THEN 'Mental'
        WHEN substr(TANIKODU, 1, 1) = 'G' THEN 'Nervous'
        WHEN substr(TANIKODU, 1, 1) = 'H' THEN 'Sensory'
        WHEN substr(TANIKODU, 1, 1) = 'I' THEN 'Circulatory'
        WHEN substr(TANIKODU, 1, 1) = 'J' THEN 'Respiratory'
        WHEN substr(TANIKODU, 1, 1) = 'K' THEN 'Digestive'
        WHEN substr(TANIKODU, 1, 1) = 'L' THEN 'Skin'
        WHEN substr(TANIKODU, 1, 1) = 'M' THEN 'Musculoskeletal'
        WHEN substr(TANIKODU, 1, 1) = 'N' THEN 'Genitourinary'
        WHEN substr(TANIKODU, 1, 1) = 'O' THEN 'Pregnancy'
        WHEN substr(TANIKODU, 1, 1) = 'R' THEN 'Symptoms'
        WHEN substr(TANIKODU, 1, 1) = 'S' THEN 'Injury'
        WHEN substr(TANIKODU, 1, 1) = 'Z' THEN 'Health_Contact'
        ELSE 'Other'
    END"""

    # 5. Missing columns (EPISODE_TARIH, Is_Dead) handling
    # We will join phase2_anadata explicitly if EPISODE_TARIH is missing
    # We'll use deceased_ids if Is_Dead is missing
    deceased_path = '/tmp/deceased_ids.json'
    deceased_ids = []
    if os.path.exists(deceased_path):
        with open(deceased_path, 'r') as f:
            deceased_ids = set(json.load(f))
    deceased_ids_str = ",".join(f"'{i}'" for i in deceased_ids) if deceased_ids else "''"

    has_episode_tarih = "EPISODE_TARIH" in cols
    has_is_dead = "Is_Dead" in cols

    comorb_expr = """(
        COALESCE(TRY_CAST("Hipertansiyon Hastada" AS INTEGER), 0) + 
        COALESCE(TRY_CAST("Kalp Damar Hastada" AS INTEGER), 0) + 
        COALESCE(TRY_CAST("Diyabet Hastada" AS INTEGER), 0) + 
        COALESCE(TRY_CAST("Kan Hastalıkları Hastada" AS INTEGER), 0)
    )"""

    query = f"""
    COPY (
        WITH p2 AS (
            SELECT SQ_EPISODE, HASTA_ID, try_cast(EPISODE_TARIH as TIMESTAMP) AS EPISODE_TARIH
            FROM read_parquet('{p2_path}')
        ),
        win_calc AS (
            SELECT SQ_EPISODE, 
                   EPISODE_TARIH,
                   LEAD(EPISODE_TARIH) OVER (PARTITION BY HASTA_ID ORDER BY EPISODE_TARIH) AS next_ep_date
            FROM p2
        ),
        final_features AS (
            SELECT pv.*,
                   CASE WHEN pv.HASTA_ID IN ({deceased_ids_str}) THEN 1 ELSE 0 END AS mortality_label,
                   CASE WHEN date_diff('day', w.EPISODE_TARIH, w.next_ep_date) <= 30 THEN 1 ELSE 0 END AS early_return_30d,
                   {icd10_case_str} AS ICD10_Chapter,
                   {comorb_expr} AS Comorbidity_Count,
                   {rms_expr} AS Lab_Deviation_RMS
            FROM read_parquet('{pv2_path}') pv
            LEFT JOIN win_calc w ON pv.SQ_EPISODE = w.SQ_EPISODE
        )
        SELECT * FROM final_features
    ) TO '{out_path}' (FORMAT PARQUET);
    """
    
    # Needs a small fix for timestamp casting. If EPISODE_TARIH was not timestamp, p2 already casts it.
    query = query.replace("strpt(p2.EPISODE_TARIH)", "p2.EPISODE_TARIH")

    print("Running DuckDB Query...")
    con.execute(query)

    # 6. Print row count and column count
    res = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    final_cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_path}')").df()
    print(f"DONE. Row count: {res}, Column count: {len(final_cols)}")

if __name__ == "__main__":
    run()
