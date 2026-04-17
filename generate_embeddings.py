"""
Generates real semantic embedding test sets for BenchmarkVectorDB.

Each model natively produces the target dimensionality:
  384  -> sentence-transformers/all-MiniLM-L6-v2
  512  -> BAAI/bge-small-en-v1.5
  768  -> sentence-transformers/all-mpnet-base-v2
 1024  -> sentence-transformers/all-roberta-large-v1

Text source: Wikipedia (English), streamed so it never loads the full
dataset into memory. Falls back to ag_news, then synthetic sentences.

Usage:
  python generate_embeddings.py              # 100k corpus, 1k queries
  python generate_embeddings.py --n 1000000 # 1M corpus
  python generate_embeddings.py --n 500000 --queries 2000
  python generate_embeddings.py --force      # re-generate existing files

Output: embeddings/corpus_{dim}.bin  and  embeddings/queries_{dim}.bin
File format: [int32 n][int32 d][float32 x n*d]  (little-endian)

Install deps:
  pip install sentence-transformers datasets
"""

import argparse
import os
import struct

MODELS = {
    384:  "sentence-transformers/all-MiniLM-L6-v2",
    512:  "BAAI/bge-small-en-v1.5",
    768:  "sentence-transformers/all-mpnet-base-v2",
    1024: "sentence-transformers/all-roberta-large-v1",
}

ENCODE_BATCH = 512   # sentences per GPU batch

# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def stream_texts(n: int) -> list[str]:
    """Stream up to n texts from Wikipedia, falling back to ag_news."""
    texts = []

    # Try Wikipedia first (large enough for 1M+)
    try:
        from datasets import load_dataset
        print(f"  Streaming Wikipedia (need {n:,} texts)...")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for item in ds:
            text = item["text"][:512].strip()
            if text:
                texts.append(text)
            if len(texts) % 50_000 == 0 and len(texts) > 0:
                print(f"  {len(texts):,} / {n:,}", flush=True)
            if len(texts) >= n:
                break
        if len(texts) >= n:
            print(f"  Loaded {n:,} texts from Wikipedia.")
            return texts[:n]
        print(f"  Wikipedia gave {len(texts):,} texts, trying ag_news for the rest...")
    except Exception as e:
        print(f"  Wikipedia unavailable ({e}), trying ag_news...")

    # ag_news fallback
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="train")
        for item in ds:
            texts.append(item["text"])
            if len(texts) >= n:
                break
        if len(texts) >= n:
            print(f"  Loaded {n:,} texts from ag_news.")
            return texts[:n]
        print(f"  ag_news gave {len(texts):,} texts, padding synthetically...")
    except Exception as e:
        print(f"  ag_news unavailable ({e}), using synthetic sentences.")

    # Synthetic padding
    base = [
        "The company reported record profits this quarter.",
        "Scientists discovered a new species in the Amazon rainforest.",
        "The championship game ended in a dramatic overtime victory.",
        "New legislation was passed to address climate change.",
        "Researchers developed a breakthrough cancer treatment.",
        "The stock market experienced significant volatility today.",
        "A major earthquake struck the coastal region.",
        "The film won several awards at the international festival.",
        "Unemployment rates fell to their lowest level in decades.",
        "The team successfully launched a new satellite into orbit.",
    ]
    while len(texts) < n:
        for s in base:
            texts.append(f"{s} Entry {len(texts)}.")
            if len(texts) >= n:
                break
    return texts[:n]


# ---------------------------------------------------------------------------
# Streaming file writer
# ---------------------------------------------------------------------------

def encode_and_save(path: str, model, texts: list[str], d: int):
    """Encode texts in batches and write directly to disk to keep RAM low."""
    n = len(texts)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        written = 0
        for start in range(0, n, ENCODE_BATCH):
            batch = texts[start : start + ENCODE_BATCH]
            vecs = model.encode(
                batch,
                batch_size=ENCODE_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("<f4")
            f.write(vecs.tobytes())
            written += len(batch)
            pct = written / n * 100
            print(f"\r  {written:>{len(str(n))}}/{n:,}  ({pct:.1f}%)", end="", flush=True)
    print(f"\r  Saved {n:,} x {d}  ->  {path}          ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=100_000, help="Corpus size (default 100000)")
    parser.add_argument("--queries", type=int, default=1_000,   help="Query count  (default 1000)")
    parser.add_argument("--force",   action="store_true",       help="Re-generate existing files")
    args = parser.parse_args()

    num_corpus  = args.n
    num_queries = args.queries
    total       = num_corpus + num_queries

    print("=" * 60)
    print(f"  Corpus: {num_corpus:,}  |  Queries: {num_queries:,}")
    print("=" * 60)

    # Device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load texts once and share across all models
    print(f"\nLoading {total:,} texts...")
    all_texts     = stream_texts(total)
    corpus_texts  = all_texts[:num_corpus]
    query_texts   = all_texts[num_corpus:]

    from sentence_transformers import SentenceTransformer

    for dim, model_name in MODELS.items():
        corpus_path  = f"embeddings/corpus_{dim}.bin"
        queries_path = f"embeddings/queries_{dim}.bin"

        if not args.force and os.path.exists(corpus_path) and os.path.exists(queries_path):
            print(f"\n--- dim={dim}: files exist, skipping (use --force to regenerate) ---")
            continue

        print(f"\n=== dim={dim} | {model_name} ===")
        print("  Loading model...")
        model = SentenceTransformer(model_name, device=device)

        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim != dim:
            print(f"  WARNING: model produces {actual_dim}-dim, expected {dim}. Skipping.")
            continue

        print(f"  Encoding {num_corpus:,} corpus texts...")
        encode_and_save(corpus_path, model, corpus_texts, dim)

        print(f"  Encoding {num_queries:,} query texts...")
        encode_and_save(queries_path, model, query_texts, dim)

    print("\nDone. Run: java BenchmarkVectorDB [num_vectors]")


if __name__ == "__main__":
    main()
