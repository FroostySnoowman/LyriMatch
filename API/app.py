import os
from typing import List, Dict

import numpy as np
import pandas as pd
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch

# ----------------------------
# Configuration
# ----------------------------
PARQUET_PATH = r"D:\Spotify project\Data\song_embeddings\song_embeddings.parquet"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # 1024-dim; matches your dataset
TOP_K = 5

# ----------------------------
# Globals (loaded once)
# ----------------------------
app = Flask(__name__)

_vectors = None           # np.ndarray [N, 1024], float32, L2-normalized
_metadata = None          # pd.DataFrame with columns: id, name, album_name
_index = None             # faiss.Index (cosine via inner product)
_model = None             # SentenceTransformer


# ----------------------------
# Initialization
# ----------------------------
def load_data(parquet_path: str):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Expect vector_0 ... vector_1023
    vector_cols = [f"vector_{i}" for i in range(1024)]
    for c in vector_cols:
        if c not in df.columns:
            raise ValueError(f"Missing expected vector column: {c}")

    vectors = df[vector_cols].to_numpy(dtype=np.float32)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    metadata = df[["id", "name", "album_name"]].reset_index(drop=True)
    return vectors, metadata


def build_cosine_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    # After L2-normalization, inner product equals cosine similarity
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(x)
    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)
    return index


def load_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    # Shorter sequence speeds up inference while keeping 1024-d outputs.
    model.max_seq_length = 64
    return model


def embed_text(text: str) -> np.ndarray:
    """
    Returns a (1, D) float32 L2-normalized embedding for the input lyrics.
    Single pass; model will truncate long inputs to max_seq_length.
    """
    emb = _model.encode(
        [text.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=1,
        show_progress_bar=False,
    ).astype(np.float32)
    return emb  # already L2-normalized by SentenceTransformer


def init():
    global _vectors, _metadata, _index, _model

    _vectors, _metadata = load_data(PARQUET_PATH)
    _index = build_cosine_index(_vectors)
    _model = load_embedder(EMBED_MODEL)

    # Sanity check: dimensions must match
    test_vec = embed_text("hello world")
    if test_vec.shape[1] != _vectors.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: model outputs {test_vec.shape[1]}, "
            f"dataset is {_vectors.shape[1]}"
        )


# ----------------------------
# API
# ----------------------------
@app.route("/search", methods=["POST"])
def search():
    """
    POST JSON: { "lyrics": "<string>" }
    Returns top-5 matches with cosine_sim.
    """
    data = request.get_json(silent=True) or {}
    lyrics = data.get("lyrics", "")

    if not isinstance(lyrics, str) or not lyrics.strip():
        return jsonify({"error": "Missing or empty 'lyrics'"}), 400

    query_vec = embed_text(lyrics)  # (1, D), normalized
    scores, indices = _index.search(query_vec, TOP_K)

    results: List[Dict] = []
    for rank, (i, s) in enumerate(zip(indices[0], scores[0]), start=1):
        row = _metadata.iloc[int(i)]
        results.append(
            {
                "rank": rank,
                "name": row["name"],
                "album_name": row["album_name"],
                "id": row["id"],
                "cosine_sim": float(s),
            }
        )

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # Use all CPU threads for FAISS search
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    init()
    # Run Flask
    # Set host to "0.0.0.0" if you want to access from other machines
    app.run(host="127.0.0.1", port=5000, debug=False)
