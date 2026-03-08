"""
Phase 5b: BERTurk NLP Embedding Extraction
-------------------------------------------
Runs INDEPENDENTLY from the structured pipeline.
Reads the Phase 3 cleaned Anadata (which still has the free-text columns),
extracts embeddings using BERTurk, and saves them as a separate Parquet file
keyed by SQ_EPISODE for later joining in Phase 5c.

Requirements: pip install transformers torch
"""
import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 64
MAX_LENGTH = 128
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

def get_embeddings(texts, tokenizer, model, device):
    """Batch encode a list of texts into [CLS] embeddings."""
    # Replace NaN/empty with a placeholder
    clean_texts = [str(t) if pd.notna(t) and str(t) not in ['nan', 'NaN', 'None', ''] else '' for t in texts]
    
    encoded = tokenizer(
        clean_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    # Use [CLS] token embedding (first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings


def process_phase_5b():
    out_dir = r"c:/Users/seiil/Desktop/ACUHIT_SCRATCH/data/processed"
    
    # Load the Phase 3 Anadata which still has free-text columns
    ana_path = os.path.join(out_dir, "phase3_temporal_anadata.parquet")
    print(f"Loading Anadata with text columns from {ana_path}...")
    
    # Only load the columns we need to save memory
    text_cols = ['YAKINMA', 'ÖYKÜ', 'Muayene Notu']
    id_col = 'SQ_EPISODE'
    df = pd.read_parquet(ana_path, columns=[id_col] + text_cols)
    
    print(f"Loaded {len(df)} rows. Initializing BERTurk model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # We will concatenate the three text fields into one combined clinical note
    # This gives the model the full clinical context per visit
    print("Combining text fields into single clinical note...")
    df['combined_text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.strip()
    
    # Process in batches
    total = len(df)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings = []
    
    print(f"Extracting embeddings in {num_batches} batches of {BATCH_SIZE}...")
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch_texts = df['combined_text'].iloc[start:end].tolist()
        
        embs = get_embeddings(batch_texts, tokenizer, model, device)
        all_embeddings.append(embs)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {end}/{total} rows ({(end/total)*100:.1f}%)...")
    
    print("Stacking embeddings...")
    embeddings_matrix = np.vstack(all_embeddings)
    
    # Create embedding DataFrame
    emb_dim = embeddings_matrix.shape[1]
    emb_cols = [f"NLP_Emb_{j}" for j in range(emb_dim)]
    df_emb = pd.DataFrame(embeddings_matrix, columns=emb_cols)
    df_emb[id_col] = df[id_col].values
    
    # Save
    out_path = os.path.join(out_dir, "phase5b_nlp_embeddings.parquet")
    df_emb.to_parquet(out_path)
    print(f"\nPhase 5b Complete. Saved {emb_dim}-dim embeddings for {total} episodes to {out_path}")

if __name__ == "__main__":
    process_phase_5b()
