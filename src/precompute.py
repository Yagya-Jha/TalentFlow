import json
import argparse
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import docx
except ImportError:
    pass

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')

def load_jd(jd_path: str) -> str:
    if jd_path.endswith('.docx'):
        doc = docx.Document(jd_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        text = '\n'.join(full_text)
    else:
        with open(jd_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    return normalize_text(text).lower()

def safe_get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default

def get_candidate_text(c: dict) -> str:
    parts = []
    profile = c.get('profile', {})
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('current_title', ''))
    
    for skill in c.get('skills', []):
        parts.append(skill.get('name', ''))
        
    for exp in c.get('career_history', []):
        parts.append(exp.get('description', ''))
        
    return normalize_text(" ".join(p for p in parts if p)).lower()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--jd', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print("Loading base model (sentence-transformers/all-MiniLM-L6-v2)...", flush=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Loading CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2) for cache...", flush=True)
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("Loading JD...", flush=True)
    jd_text = load_jd(args.jd)
    jd_embedding = model.encode([jd_text], show_progress_bar=False)[0]
    
    print("Streaming candidates and computing embeddings...", flush=True)
    candidate_ids = []
    embeddings = []
    
    batch_texts = []
    batch_ids = []
    chunk_size = 256  # Small chunk to prevent tokenization memory explosion
    total_processed = 0
    
    with open(args.candidates, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            c = json.loads(line)
            batch_ids.append(c.get('candidate_id'))
            batch_texts.append(get_candidate_text(c))
            
            if len(batch_texts) >= chunk_size:
                # Encode small chunk
                batch_embeddings = model.encode(batch_texts, batch_size=32, show_progress_bar=False)
                embeddings.append(batch_embeddings)
                candidate_ids.extend(batch_ids)
                
                total_processed += len(batch_texts)
                if total_processed % 5120 == 0:
                    print(f"  Encoded {total_processed} candidates...", flush=True)
                    
                batch_texts = []
                batch_ids = []
                
    # Process remaining
    if batch_texts:
        batch_embeddings = model.encode(batch_texts, batch_size=32, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        candidate_ids.extend(batch_ids)
        total_processed += len(batch_texts)
        print(f"  Encoded {total_processed} candidates...", flush=True)
        
    print("Stacking embeddings matrix...", flush=True)
    embeddings_matrix = np.vstack(embeddings)
    
    print(f"Saving to {args.out}...", flush=True)
    np.savez(args.out, candidate_ids=np.array(candidate_ids), embeddings=embeddings_matrix, jd_embedding=jd_embedding)
    print("Precomputation complete!", flush=True)

if __name__ == "__main__":
    main()
