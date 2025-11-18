import os
import json
import time
import re
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
BOOKS_DIR = "/home/nfs/jzygmuntowicz/books/extracted_text"  # folder z plikami .txt
OUTPUT_DIR = "embeddings"                                   # gdzie zapisać wyniki
MODEL = "nomic-ai/nomic-embed-text-v1"                      # model do embeddingów
CHUNK_SIZE = 1000                                            # długość chunku (znaki)
OVERLAP = 200                                                # nakładanie (znaki)
BATCH_SIZE = 32                                              # liczba chunków w batchu
DEBUG_PRINT_N = 3                                            # ile chunków wypisać do debugowania

# === PREPARATION ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🔍 Loading embedding model: {MODEL} ...")
embedder = SentenceTransformer(MODEL, trust_remote_code=True)
print("✅ Model loaded successfully.")


# === UTILS ===
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split text into overlapping chunks based on words, not characters.
    This prevents cutting sentences in half and improves semantic consistency.
    """
    words = re.findall(r"\S+\s*", text)
    chunks = []
    start = 0

    while start < len(words):
        end = start
        length = 0
        while end < len(words) and length + len(words[end]) < chunk_size:
            length += len(words[end])
            end += 1
        chunk = "".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        # overlap proportional to chunk size, but on word level
        start = max(end - int(overlap / 5), end)

    return chunks


def safe_to_numpy(x):
    """Convert tensor/list/numpy to numpy.ndarray safely."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        try:
            return np.array(x)
        except Exception:
            return np.asarray(x, dtype=np.float32)
    return np.asarray(x)


def mean_pool_embeddings(emb):
    """Mean pooling over tokens → 1D vector per chunk."""
    emb_np = safe_to_numpy(emb)
    if emb_np.ndim == 1:
        return emb_np
    return np.mean(emb_np, axis=0)


# === MAIN LOOP ===
for filename in os.listdir(BOOKS_DIR):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(BOOKS_DIR, filename)
    print(f"\n📘 Processing {filename}...")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"   ✂️ Split into {len(chunks)} overlapping chunks (word-based, size={CHUNK_SIZE}, overlap={OVERLAP}).")

    embeddings = []
    times, norms = [], []
    bad_counts = {"nan": 0, "all_zero": 0}

    total_chunks = len(chunks)
    batch_iter = range(0, total_chunks, BATCH_SIZE)

    for b in tqdm(batch_iter, desc=f"Embedding {filename}", unit="batch"):
        batch = chunks[b:b + BATCH_SIZE]
        t0 = time.perf_counter()

        try:
            raw_batch = embedder.encode(
                batch,
                output_value='token_embeddings',
                convert_to_numpy=False,
                batch_size=BATCH_SIZE,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"   ❌ Batch {b // BATCH_SIZE + 1} failed: {e}")
            continue

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        for j, raw in enumerate(raw_batch):
            mean_emb = mean_pool_embeddings(raw).astype(np.float32)

            if np.isnan(mean_emb).any():
                bad_counts["nan"] += 1
            if np.allclose(mean_emb, 0.0):
                bad_counts["all_zero"] += 1

            norm = np.linalg.norm(mean_emb)
            norms.append(norm)

            chunk_idx = b + j + 1
            if chunk_idx <= DEBUG_PRINT_N:
                print(f"[{filename}] DEBUG chunk {chunk_idx}: batch_time={elapsed:.4f}s, dim={mean_emb.shape}, norm={norm:.4f}")
                print(f"      first elems: {mean_emb[:6].tolist()}")

            embeddings.append({
                "chunk_index": chunk_idx,
                "text": batch[j],
                "embedding": mean_emb.tolist()
            })

    # === SUMMARY ===
    avg_time = float(np.mean(times)) if times else 0.0
    median_time = float(np.median(times)) if times else 0.0
    print("\n--- SUMMARY ---")
    print(f"📚 Chunks processed: {len(embeddings)} (batch size={BATCH_SIZE})")
    print(f"⏱️  Average batch time: {avg_time:.4f}s (median {median_time:.4f}s)")
    print(f"📏 Embedding norms: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}, "
          f"min={np.min(norms):.4f}, max={np.max(norms):.4f}")
    print(f"⚠️  NaN embeddings: {bad_counts['nan']}, all-zero embeddings: {bad_counts['all_zero']}")
    print("---------------------------")

    # === SAVE ===
    out_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(embeddings, out_f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved embeddings to {out_path}")

print("\n✅ All books processed successfully.")
