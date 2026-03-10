"""
phase 5b v2: tf-idf nlp baseline
---------------------------------
fast nlp feature extraction using tf-idf on the yakinma (chief complaint) column.
runs in minutes instead of hours compared to berturk on cpu.
"""
import polars as pl
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np

def process_phase_5b_tfidf():
    t_start = time.time()
    out_dir = r"./data/processed"
    
    # load only the columns we need
    print("Loading text columns...")
    df = pl.read_parquet(
        os.path.join(out_dir, "phase2_demographics_data.parquet"),
        columns=["ENCOUNTER_ID", "CHIEF_COMPLAINT"]
    )
    df = df.with_columns([
        pl.col("ENCOUNTER_ID").cast(pl.Utf8),
        pl.col("CHIEF_COMPLAINT").cast(pl.Utf8).fill_null("").alias("CHIEF_COMPLAINT"),
    ])
    
    texts = df["CHIEF_COMPLAINT"].to_list()
    print(f"Loaded {len(texts):,} texts. Fitting TF-IDF...")
    
    # tf-idf with a cap on features for memory efficiency
    vectorizer = TfidfVectorizer(
        max_features=200,       # top 200 terms
        min_df=50,              # must appear in at least 50 docs
        max_df=0.95,            # ignore terms appearing in >95% of docs
        strip_accents='unicode',
        sublinear_tf=True,
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    elapsed = time.time() - t_start
    print(f"TF-IDF fitted in {elapsed:.1f}s. Shape: {tfidf_matrix.shape}")
    
    # convert sparse to dense polars dataframe
    feature_names = [f"TFIDF_{fn}" for fn in vectorizer.get_feature_names_out()]
    dense = tfidf_matrix.toarray().astype(np.float32)
    
    df_tfidf = pl.DataFrame({
        "ENCOUNTER_ID": df["ENCOUNTER_ID"],
        **{name: dense[:, i] for i, name in enumerate(feature_names)}
    })
    
    out_path = os.path.join(out_dir, "phase5b_tfidf.parquet")
    df_tfidf.write_parquet(out_path)
    
    total = time.time() - t_start
    print(f"\nPhase 5b TF-IDF Complete in {total:.1f}s ({total/60:.1f} min)")
    print(f"  Features: {len(feature_names)}")
    print(f"  Output: {out_path}")

if __name__ == "__main__":
    process_phase_5b_tfidf()
