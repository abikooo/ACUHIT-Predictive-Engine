import duckdb
import json
import os

def run():
    out_dir = r"./data/processed"
    pv2_path = os.path.join(out_dir, "pipeline_v2_output.parquet")
    p2_path = os.path.join(out_dir, "phase2_demographics_data.parquet")
    out_path = os.path.join(out_dir, "final_feature_matrix.parquet")

    con = duckdb.connect()
    
    # configure out-of-core memory management to prevent oom on 6m row join
    tmp_path = os.path.join(out_dir, "duckdb_tmp")
    os.makedirs(tmp_path, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp_path}'")
    con.execute("PRAGMA memory_limit='12GB'")

    # 1. inspect pipeline_v2_output
    cols_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pv2_path}')").df()
    cols = cols_df['column_name'].tolist()

    lab_cols = [c for c in cols if c.startswith('Lab_')]
    
    # 2. find top 10 most populated lab_* columns
    counts = {}
    for c in lab_cols:
        cnt = con.execute(f"SELECT COUNT(\"{c}\") FROM read_parquet('{pv2_path}')").fetchone()[0]
        counts[c] = cnt
    
    top_10 = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)[:10]

    # 3. calculate mean and std
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

    # build rms expr
    rms_parts = []
    for c in top_10:
        m, s = stats[c]
        rms_parts.append(f"COALESCE(POWER((TRY_CAST(\"{c}\" AS DOUBLE) - {m}) / {s}, 2), 0)")
    
    rms_expr = f"SQRT( ({' + '.join(rms_parts)}) / 10.0 )" if rms_parts else "0.0"

    # 4. icd10 logic
    icd10_case_str = """case
        when substr(tanikodu, 1, 1) in ('a', 'b') then 'infectious'
        when substr(tanikodu, 1, 1) = 'c' then 'neoplasms'
        when substr(tanikodu, 1, 1) = 'd' then 'blood'
        when substr(tanikodu, 1, 1) = 'e' then 'endocrine'
        when substr(tanikodu, 1, 1) = 'f' then 'mental'
        when substr(tanikodu, 1, 1) = 'g' then 'nervous'
        when substr(tanikodu, 1, 1) = 'h' then 'sensory'
        when substr(tanikodu, 1, 1) = 'i' then 'circulatory'
        when substr(tanikodu, 1, 1) = 'j' then 'respiratory'
        when substr(tanikodu, 1, 1) = 'k' then 'digestive'
        when substr(tanikodu, 1, 1) = 'l' then 'skin'
        when substr(tanikodu, 1, 1) = 'm' then 'musculoskeletal'
        when substr(tanikodu, 1, 1) = 'n' then 'genitourinary'
        when substr(tanikodu, 1, 1) = 'o' then 'pregnancy'
        when substr(tanikodu, 1, 1) = 'r' then 'symptoms'
        when substr(tanikodu, 1, 1) = 's' then 'injury'
        when substr(tanikodu, 1, 1) = 'z' then 'health_contact'
        else 'other'
    end"""

    # 5. missing columns (encounter_date, is_dead) handling
    # we will join phase2_demographics_data explicitly if encounter_date is missing
    # we'll use deceased_ids if is_dead is missing
    deceased_path = '/tmp/deceased_ids.json'
    deceased_ids = []
    if os.path.exists(deceased_path):
        with open(deceased_path, 'r') as f:
            deceased_ids = set(json.load(f))
    deceased_ids_str = ",".join(f"'{i}'" for i in deceased_ids) if deceased_ids else "''"

    has_encounter_date = "ENCOUNTER_DATE" in cols
    has_is_dead = "Is_Dead" in cols

    comorb_expr = """(
        coalesce(try_cast("hipertansiyon hastada" as integer), 0) + 
        coalesce(try_cast("kalp damar hastada" as integer), 0) + 
        coalesce(try_cast("diyabet hastada" as integer), 0) + 
        coalesce(try_cast("kan hastalıkları hastada" as integer), 0)
    )"""

    query = f"""
    COPY (
        WITH p2 AS (
            SELECT ENCOUNTER_ID, PATIENT_ID, try_cast(ENCOUNTER_DATE as TIMESTAMP) AS ENCOUNTER_DATE
            FROM read_parquet('{p2_path}')
        ),
        win_calc AS (
            SELECT ENCOUNTER_ID, 
                   ENCOUNTER_DATE,
                   LEAD(ENCOUNTER_DATE) OVER (PARTITION BY PATIENT_ID ORDER BY ENCOUNTER_DATE) AS next_ep_date
            FROM p2
        ),
        final_features AS (
            SELECT pv.*,
                   CASE WHEN pv.PATIENT_ID IN ({deceased_ids_str}) THEN 1 ELSE 0 END AS mortality_label,
                   CASE WHEN date_diff('day', w.ENCOUNTER_DATE, w.next_ep_date) <= 30 THEN 1 ELSE 0 END AS early_return_30d,
                   {icd10_case_str} AS ICD10_Chapter,
                   {comorb_expr} AS Comorbidity_Count,
                   {rms_expr} AS Lab_Deviation_RMS
            FROM read_parquet('{pv2_path}') pv
            LEFT JOIN win_calc w ON pv.ENCOUNTER_ID = w.ENCOUNTER_ID
        )
        SELECT * FROM final_features
    ) TO '{out_path}' (FORMAT PARQUET);
    """
    
    # needs a small fix for timestamp casting. if encounter_date was not timestamp, p2 already casts it.
    query = query.replace("strpt(p2.ENCOUNTER_DATE)", "p2.ENCOUNTER_DATE")

    print("Running DuckDB Query...")
    con.execute(query)

    # 6. print row count and column count
    res = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    final_cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_path}')").df()
    print(f"DONE. Row count: {res}, Column count: {len(final_cols)}")

if __name__ == "__main__":
    run()
