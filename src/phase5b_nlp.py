"""
phase 5b: berturk nlp embedding extraction
-------------------------------------------
runs independently from the structured pipeline.
reads the phase 3 cleaned demographics_data (which still has the free-text columns),
extracts embeddings using berturk, and saves them as a separate parquet file
keyed by sq_episode for later joining in phase 5c.

requirements: pip install transformers torch
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
    """batch encode a list of texts into [cls] embeddings."""
    # replace nan/empty with a placeholder
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
    
    # use [cls] token embedding (first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings


def process_phase_5b():
    out_dir = r"./data/processed"
    
    # load the phase 3 demographics_data which still has free-text columns
    ana_path = os.path.join(out_dir, "phase3_temporal_demographics_data.parquet")
    print(f"Loading Anadata with text columns from {ana_path}...")
    
    # only load the columns we need to save memory
    text_cols = ['CHIEF_COMPLAINT', 'MEDICAL_HISTORY', 'Examination_Note']
    id_col = 'ENCOUNTER_ID'
    df = pd.read_parquet(ana_path, columns=[id_col] + text_cols)
    
    print(f"Loaded {len(df)} rows. Initializing BERTurk model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # we will concatenate the three text fields into one combined clinical note
    # this gives the model the full clinical context per visit
    print("Combining text fields into single clinical note...")
    df['combined_text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.strip()
    
    # process in batches
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
    
    # create embedding dataframe
    emb_dim = embeddings_matrix.shape[1]
    emb_cols = [f"NLP_Emb_{j}" for j in range(emb_dim)]
    df_emb = pd.DataFrame(embeddings_matrix, columns=emb_cols)
    df_emb[id_col] = df[id_col].values
    
    # save
    out_path = os.path.join(out_dir, "phase5b_nlp_embeddings.parquet")
    df_emb.to_parquet(out_path)
    print(f"\nPhase 5b Complete. Saved {emb_dim}-dim embeddings for {total} episodes to {out_path}")

if __name__ == "__main__":
    process_phase_5b()
