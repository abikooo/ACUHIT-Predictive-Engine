import duckdb
import os

def run():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    f_path = os.path.join(out_dir, "final_feature_matrix.parquet")
    tmp_path = os.path.join(out_dir, "duckdb_dhs_tmp")
    os.makedirs(tmp_path, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{tmp_path}'")
    con.execute("PRAGMA memory_limit='12GB'")

    print("Executing DHS Computation Query via DuckDB...")

    query = f"""
    COPY (
        WITH raw_data AS (
            SELECT * FROM read_parquet('{f_path}')
        ),
        scored_components AS (
            SELECT *,
                -- Vitals Score (V)
                (
                    CASE 
                        WHEN "SPO2_Binned" = 'Normal (95-100)' THEN 0.0
                        WHEN "SPO2_Binned" = 'Low (90-94)' THEN 0.5
                        WHEN "SPO2_Binned" = 'Critical (<90)' THEN 1.0
                        ELSE 0.3
                    END +
                    CASE 
                        WHEN "Nabız_Binned" = 'Normal (60-100)' THEN 0.0
                        WHEN "Nabız_Binned" = 'Tachycardia (>100)' THEN 0.5
                        WHEN "Nabız_Binned" = 'Severe_Tachycardia' THEN 1.0  -- Assuming Severe is Tachycardia for this dataset based on bins
                        WHEN "Nabız_Binned" = 'Bradycardia (<60)' THEN 0.4
                        ELSE 0.3
                    END +
                    CASE 
                        WHEN "KB-S_Binned" = 'Normal' THEN 0.0
                        WHEN "KB-S_Binned" = 'Pre-HTN' THEN 0.2
                        WHEN "KB-S_Binned" = 'HTN' THEN 0.5
                        WHEN "KB-S_Binned" = 'Crisis' THEN 1.0
                        ELSE 0.3
                    END +
                    CASE 
                        WHEN "KB-D_Binned" = 'Normal' THEN 0.0
                        WHEN "KB-D_Binned" = 'Pre-HTN' THEN 0.2
                        WHEN "KB-D_Binned" = 'HTN' THEN 0.5
                        WHEN "KB-D_Binned" = 'Crisis' THEN 1.0
                        ELSE 0.3
                    END +
                    CASE 
                        WHEN "BMI_Binned" = 'Normal' THEN 0.0
                        WHEN "BMI_Binned" = 'Overweight' THEN 0.2
                        WHEN "BMI_Binned" = 'Obese' THEN 0.5
                        WHEN "BMI_Binned" = 'Obese_II' THEN 0.8
                        WHEN "BMI_Binned" = 'Underweight' THEN 0.4
                        ELSE 0.3
                    END
                ) / 5.0 AS V_Score,

                -- Lab Score (L): RMS clipped at 57.589 and normalized
                LEAST("Lab_Deviation_RMS" / 57.58918092836535, 1.0) AS L_Score,

                -- Comorbidity Score (C): Count / 4.0
                "Comorbidity_Count" / 4.0 AS C_Score,

                -- Temporal Score (T): log1p(visits) / log1p(99th percentile: 213.0)
                LEAST(LN(1.0 + TRY_CAST("TOPLAM_GELIS_SAYISI" AS DOUBLE)) / LN(1.0 + 213.0), 1.0) AS T_Score,

                -- Medication Score (M): Prescriptions / 99th percentile: 126.0
                LEAST(TRY_CAST("Total_Prescriptions" AS DOUBLE) / 126.0, 1.0) AS M_Score
                
            FROM raw_data
        ),
        dhs_computed AS (
            SELECT * EXCLUDE (V_Score, L_Score, C_Score, T_Score, M_Score),
                   (V_Score * 0.30 + L_Score * 0.25 + C_Score * 0.20 + T_Score * 0.15 + M_Score * 0.10) * 100.0 AS DHS
            FROM scored_components
        ),
        final_with_risk AS (
            SELECT *,
                   CASE 
                        WHEN DHS < 25.0 THEN 'Low'
                        WHEN DHS < 50.0 THEN 'Medium'
                        WHEN DHS < 75.0 THEN 'High'
                        ELSE 'Critical'
                   END AS Risk_Profile
            FROM dhs_computed
        )
        SELECT * FROM final_with_risk
    ) TO '{f_path}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    print("DHS Computed and saved successfully.")

    # Validation Queries
    print("\n" + "="*50)
    print("VALIDATION CHECKS")
    print("="*50)

    # 1. Mean DHS by mortality
    print("\n1. Mean DHS by Mortality Label:")
    res_mortality = con.execute(f"SELECT mortality_label, AVG(DHS) AS Mean_DHS FROM read_parquet('{f_path}') GROUP BY mortality_label ORDER BY mortality_label").df()
    print(res_mortality.to_string(index=False))

    # 2. Mean DHS by Risk Profile
    print("\n2. Mean DHS per Risk Tier:")
    res_tier = con.execute(f"SELECT Risk_Profile, AVG(DHS) AS Mean_DHS, COUNT(*) AS Patient_Visits FROM read_parquet('{f_path}') GROUP BY Risk_Profile ORDER BY 2").df()
    print(res_tier.to_string(index=False))

    # 3. Mean DHS by ICD10 Chapter
    print("\n3. Mean DHS per ICD10 Chapter:")
    res_icd10 = con.execute(f"SELECT ICD10_Chapter, AVG(DHS) AS Mean_DHS, COUNT(*) AS Visit_Count FROM read_parquet('{f_path}') GROUP BY ICD10_Chapter ORDER BY Mean_DHS DESC").df()
    print(res_icd10.to_string(index=False))

if __name__ == "__main__":
    run()
