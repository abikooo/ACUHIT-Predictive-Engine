"""
Phase 5b v2: TF-IDF NLP Baseline
---------------------------------
Fast NLP feature extraction using TF-IDF on the YAKINMA (Chief Complaint) column.
Runs in minutes instead of hours compared to BERTurk on CPU.
"""
import polars as pl
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np

def process_phase_5b_tfidf():
    t_start = time.time()
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    
    # Load only the columns we need
    print("Loading text columns...")
    df = pl.read_parquet(
        os.path.join(out_dir, "phase2_anadata.parquet"),
        columns=["SQ_EPISODE", "YAKINMA"]
    )
    df = df.with_columns([
        pl.col("SQ_EPISODE").cast(pl.Utf8),
        pl.col("YAKINMA").cast(pl.Utf8).fill_null("").alias("YAKINMA"),
    ])
    
    texts = df["YAKINMA"].to_list()
    print(f"Loaded {len(texts):,} texts. Fitting TF-IDF...")
    
    # TF-IDF with a cap on features for memory efficiency
    vectorizer = TfidfVectorizer(
        max_features=200,       # Top 200 terms
        min_df=50,              # Must appear in at least 50 docs
        max_df=0.95,            # Ignore terms appearing in >95% of docs
        strip_accents='unicode',
        sublinear_tf=True,
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    elapsed = time.time() - t_start
    print(f"TF-IDF fitted in {elapsed:.1f}s. Shape: {tfidf_matrix.shape}")
    
    # Convert sparse to dense Polars DataFrame
    feature_names = [f"TFIDF_{fn}" for fn in vectorizer.get_feature_names_out()]
    dense = tfidf_matrix.toarray().astype(np.float32)
    
    df_tfidf = pl.DataFrame({
        "SQ_EPISODE": df["SQ_EPISODE"],
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
