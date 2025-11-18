import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re

# === CONFIG ===
EMBEDDINGS_DIR = "embeddings"
TAXONOMY_FILES = {
    "Order": "Order.txt",
    "Family": "Family.txt",
    "Genus": "Genus.txt",
    "Species": "Species.txt"
}
MODEL = "nomic-ai/nomic-embed-text-v1"
TOP_K = 5
OUTPUT_DIR = "taxonomy_index_hybrid"
LITERAL_BOOST = 0.15
SIM_THRESHOLD = 0.4

# === CONTEXT SETTINGS ===
CONTEXT_WINDOW = 1       # number of neighboring chunks before/after to include
SNIPPET_MAX_LEN = 2000   # total snippet length limit (after joining context)

# === MULTICELLULARITY KEYWORDS ===
MULTICELLULAR_KEYWORDS = [
    "multicellular", "unicellular", "single-celled", "colony", "biofilm",
    "filament", "chain", "aggregate", "cluster", "rosette", "floc", "sheath",
    "mycelium", "mat", "microcolony", "syncytium", "fruiting body",
    "quorum sensing", "autoinducer", "quorum quenching", "extracellular matrix",
    "exopolysaccharide", "EPS", "extracellular polymeric substance",
    "surface adhesion", "biofilm matrix", "cooperation",
    "social behavior", "coordinated motility", "swarming",
    "gliding motility", "type IV pili", "slime trails",
    "filamentous growth", "branching", "hyphae", "aerial hyphae",
    "spore formation", "fruiting structure", "differentiation",
    "developmental cycle", "tissue-like organization", "cell-cell signaling"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🔍 Loading embedding model: {MODEL}")
embedder = SentenceTransformer(MODEL, trust_remote_code=True)
print("✅ Model loaded successfully.\n")

# === LOAD EMBEDDINGS ===
print("📚 Loading book embeddings...")
book_chunks = []

for file in tqdm(os.listdir(EMBEDDINGS_DIR), desc="Reading embeddings"):
    if not file.endswith(".json"):
        continue
    path = os.path.join(EMBEDDINGS_DIR, file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        book_chunks.append((
            file.replace(".txt.json", "").replace(".json", ""),
            entry["chunk_index"],
            np.array(entry["embedding"], dtype=np.float32),
            entry["text"]
        ))

print(f"✅ Loaded {len(book_chunks)} chunks total.\n")

chunk_embeddings = np.vstack([x[2] for x in book_chunks])
chunk_embeddings /= np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)


# === FUNCTIONS ===
def make_name_embedding(name, embedder):
    """Create taxonomy-level embedding with morphological keyword augmentation."""
    templates = [
        name,
        f"{name} bacteria",
        f"genus {name}",
        f"{name} species",
        f"{name} morphology",
        f"{name} is a bacterium"
    ]
    keyword_phrases = [f"{name} {kw}" for kw in MULTICELLULAR_KEYWORDS]
    templates.extend(keyword_phrases)

    embs = embedder.encode(templates, convert_to_numpy=True)
    return embs.mean(axis=0)


def extract_contextual_snippet(book_chunks, target_idx, window=CONTEXT_WINDOW, max_len=SNIPPET_MAX_LEN):
    """Return merged snippet from the target chunk ± window neighboring chunks."""
    book, chunk_idx, _, _ = book_chunks[target_idx]
    # Collect chunks from the same book
    same_book = [c for c in book_chunks if c[0] == book]
    same_book.sort(key=lambda x: x[1])

    # Find index of this chunk
    chunk_indices = [c[1] for c in same_book]
    pos = chunk_indices.index(chunk_idx)

    # Include neighbors
    start = max(0, pos - window)
    end = min(len(same_book), pos + window + 1)
    merged_text = " ".join(c[3].replace("\n", " ").strip() for c in same_book[start:end])
    merged_text = re.sub(r"\s+", " ", merged_text.strip())

    # Cut to full sentence boundaries if too long
    if len(merged_text) > max_len:
        cut = merged_text[:max_len]
        last_period = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if last_period > 200:
            cut = cut[:last_period + 1]
        merged_text = cut.strip() + " ..."
    return merged_text


def hybrid_search(name, name_emb, book_chunks, chunk_embeddings, top_k=TOP_K, literal_boost=LITERAL_BOOST):
    """Combine semantic similarity and literal occurrence."""
    q = name_emb / (np.linalg.norm(name_emb) + 1e-9)
    sims = chunk_embeddings.dot(q)

    for i, (_, _, _, text) in enumerate(book_chunks):
        if name.lower() in text.lower():
            sims[i] += literal_boost

    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(i, book_chunks[i], float(sims[i])) for i in top_idx if sims[i] > SIM_THRESHOLD]


# === MAIN LOOP ===
for level, filename in TAXONOMY_FILES.items():
    if not os.path.exists(filename):
        print(f"⚠️ File {filename} not found, skipping.")
        continue

    print(f"\n🔎 Processing taxonomy level: {level}")
    with open(filename, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"   Found {len(names)} names.")

    results = []

    for name in tqdm(names, desc=f"Searching {level}"):
        name_emb = make_name_embedding(name, embedder)
        hits = hybrid_search(name, name_emb, book_chunks, chunk_embeddings)

        for idx, (book, chunk_idx, _, _), sim in hits:
            snippet = extract_contextual_snippet(book_chunks, idx)
            results.append({
                "level": level,
                "name": name,
                "book": book,
                "chunk_index": chunk_idx,
                "similarity": round(sim, 4),
                "snippet": snippet
            })

    out_path = os.path.join(OUTPUT_DIR, f"{level}_hybrid_matches.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(results)} results for {level} → {out_path}")

print("\n✅ All taxonomy levels processed successfully.")
