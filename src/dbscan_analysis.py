import duckdb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def run():
    out_dir = r"./data/processed"
    f_path = os.path.join(out_dir, "final_feature_matrix.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='12GB'")

    print("Extracting Stratified Sample (100K max) via DuckDB...")
    
    # use order by random() to ensure we strictly get up to 50k from each without scanning the whole 70m rows into memory
    query = f"""
    WITH alive AS (
        SELECT DHS, 
               CAST(TANI_YASI AS DOUBLE) AS Age_At_Visit_Years, 
               CAST(Comorbidity_Count AS DOUBLE) AS Comorbidity_Count, 
               CAST(Lab_Deviation_RMS AS DOUBLE) AS Lab_Deviation_RMS, 
               CAST(Total_Prescriptions AS DOUBLE) AS Total_Prescriptions, 
               CAST(mortality_label AS DOUBLE) AS mortality_label
        FROM read_parquet('{f_path}')
        WHERE mortality_label = 0
        USING SAMPLE 50000 PERCENT (Bernoulli) 
        -- Fallback if dataset is huge, 'USING SAMPLE' is faster than random() sorting
        -- But since we want exactly 50K, we can use reservoir sampling
    )
    """

    # better approach for exact 50k is reservoir sampling 
    query = f"""
    SELECT DHS, 
           CAST(TANI_YASI AS DOUBLE) AS Age_At_Visit_Years, 
           CAST(Comorbidity_Count AS DOUBLE) AS Comorbidity_Count, 
           CAST(Lab_Deviation_RMS AS DOUBLE) AS Lab_Deviation_RMS, 
           CAST(Total_Prescriptions AS DOUBLE) AS Total_Prescriptions, 
           CAST(mortality_label AS DOUBLE) AS mortality_label
    FROM read_parquet('{f_path}')
    WHERE mortality_label = 0
    USING SAMPLE reservoir(50000)
    UNION ALL
    SELECT DHS, 
           CAST(TANI_YASI AS DOUBLE) AS Age_At_Visit_Years, 
           CAST(Comorbidity_Count AS DOUBLE) AS Comorbidity_Count, 
           CAST(Lab_Deviation_RMS AS DOUBLE) AS Lab_Deviation_RMS, 
           CAST(Total_Prescriptions AS DOUBLE) AS Total_Prescriptions, 
           CAST(mortality_label AS DOUBLE) AS mortality_label
    FROM read_parquet('{f_path}')
    WHERE mortality_label = 1
    USING SAMPLE reservoir(50000)
    """

    df = con.execute(query).df()
    print(f"Sample loaded: {df.shape[0]} rows.")

    # drop any nulls just in case
    df = df.dropna()

    features = ['DHS', 'Age_At_Visit_Years', 'Comorbidity_Count', 'Lab_Deviation_RMS', 'Total_Prescriptions']
    X = df[features].values

    print("\nScaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running DBSCAN (eps=0.5, min_samples=50)...")
    dbscan = DBSCAN(eps=0.5, min_samples=50, n_jobs=-1)
    df['Cluster'] = dbscan.fit_predict(X_scaled)
    print("DBSCAN complete.")

    print("Calculating Silhouette Score (Subsampled)...")
    sil_sample = min(5000, len(df))
    idx = np.random.choice(len(df), sil_sample, replace=False)
    if len(set(df['Cluster'].iloc[idx])) > 1:
        score = silhouette_score(X_scaled[idx], df['Cluster'].iloc[idx])
        print(f"Silhouette Score (subsampled N={sil_sample}): {score:.4f}")
    else:
        print("Silhouette Score: N/A (Only 1 cluster/noise found in sample)")

    # cluster summaries
    print("\nComputing Cluster Summaries...")
    summary = df.groupby('Cluster').agg(
        Size=('DHS', 'count'),
        Mean_DHS=('DHS', 'mean'),
        Mortality_Rate=('mortality_label', 'mean'),
        Mean_Age=('Age_At_Visit_Years', 'mean'),
        Mean_Comorbidity=('Comorbidity_Count', 'mean')
    ).reset_index()

    # label clusters (ignore noise -1)
    valid_clusters = summary[summary['Cluster'] != -1].copy()
    
    if len(valid_clusters) > 0:
        # sort by combination of mortality rate and mean dhs
        # we'll normalize both to 0-1 and add them to find the "severity" rank
        norm_mort = valid_clusters['Mortality_Rate'] / (valid_clusters['Mortality_Rate'].max() + 1e-9)
        norm_dhs = valid_clusters['Mean_DHS'] / (valid_clusters['Mean_DHS'].max() + 1e-9)
        valid_clusters['Severity_Score'] = norm_mort + norm_dhs
        
        valid_clusters = valid_clusters.sort_values(by='Severity_Score', ascending=False)
        
        labels = ['Critical', 'High', 'Medium', 'Low']
        
        assign_labels = []
        for i in range(len(valid_clusters)):
            if i < len(labels):
                assign_labels.append(labels[i])
            else:
                assign_labels.append('Low') # fallback if more than 4 clusters
                
        valid_clusters['Clinical_Tier'] = assign_labels
        
        # merge back
        summary = summary.merge(valid_clusters[['Cluster', 'Clinical_Tier']], on='Cluster', how='left')
    
    summary['Clinical_Tier'] = summary['Clinical_Tier'].fillna('Unclassified (Noise)')
    summary = summary.sort_values(by='Mean_DHS', ascending=False)
    
    csv_path = os.path.join(r"./", "cluster_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nCluster summary saved to: {csv_path}")
    print(summary.to_string(index=False))

    # scatter plot
    print("\nGenerating Scatter Plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x='Age_At_Visit_Years', 
        y='DHS', 
        hue='Cluster', 
        palette='tab10', 
        alpha=0.6, 
        edgecolor=None,
        s=10
    )
    plt.title('DBSCAN Clusters: DHS vs Age')
    plt.xlabel('Age at Visit (Years)')
    plt.ylabel('Dynamic Health Score (DHS)')
    plt.legend(title='DBSCAN Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(r"./", "dbscan_clusters.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to: {plot_path}")

    print("\n" + "="*60)
    print("DHS SCORE - CALCULATION EXPLANATION AND INTERPRETATION")
    print("="*60)
    print("The Dynamic Health Score (DHS) ranges from 0 to 100 for each visit.")
    print("It is composed of 5 clinical sub-scores, each normalized 0-1 and weighted:")
    print("1. [V] Vitals Score (30% weight)")
    print("   -> (SPO2_Risk + HR_Risk + SBP_Risk + DBP_Risk + BMI_Risk) / 5")
    print("   -> Extreme vitals map to 1.0, normal to 0.0, unrecorded to 0.3.")
    print("2. [L] Lab Score (25% weight)")
    print("   -> Root Mean Square (RMS) of z-scores for top 10 populated labs.")
    print("   -> Capped at 99th percentile (~57.6) and normalized to 0-1. Higher deviation = higher risk.")
    print("3. [C] Comorbidity Burden (20% weight)")
    print("   -> Sum of exactly 4 major conditions (HTN, Diabetes, Cardio, Blood) divided by 4.")
    print("4. [T] Temporal Trajectory Score (15% weight)")
    print("   -> log(1+Visits) / log(1+99th_percentile_visits).")
    print("   -> Frequent returners score closer to 1.0.")
    print("5. [M] Medication Score (10% weight)")
    print("   -> Total_Prescriptions / 99th_percentile (~126). Polypharmacy indicator.")
    print("\nFinal Math:")
    print("DHS = (V * 0.30  +  L * 0.25  +  C * 0.20  +  T * 0.15  +  M * 0.10) * 100")
    print("="*60)

if __name__ == "__main__":
    run()
