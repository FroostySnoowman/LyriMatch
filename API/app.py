# app.py
#
# Minimal Flask API:
#   POST /search   { "lyrics": "<text>" }              ->  top-5 nearest songs as JSON
#   POST /addSong  { "title": "<title>", "lyrics": "" } ->  add song & embedding to parquet + index
#
# Requirements:
#   pip install flask faiss-cpu sentence-transformers pandas pyarrow torch

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch

# ----------------------------
# Configuration
# ----------------------------

# Directory that contains songembeddings.parquet (or nothing)
SONG_EMBEDDINGS_DIR = r"API\Data\song_embeddings"
SONG_EMBEDDINGS_FILE = os.path.join(SONG_EMBEDDINGS_DIR, "songembeddings.parquet")

EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 5

# ----------------------------
# Globals (loaded once)
# ----------------------------
app = Flask(__name__)

_vectors: np.ndarray | None = None    # np.ndarray [N, D], float32, L2-normalized
_metadata: pd.DataFrame | None = None # columns: id, name, album_name
_index: faiss.IndexFlatIP | None = None
_model: SentenceTransformer | None = None


# ----------------------------
# Helpers for parquet + index
# ----------------------------
def ensure_embeddings_dir():
    os.makedirs(SONG_EMBEDDINGS_DIR, exist_ok=True)


def infer_vector_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer vector_0 ... vector_(D-1) columns from an existing parquet.
    """
    vec_cols = [c for c in df.columns if c.startswith("vector_")]
    if not vec_cols:
        raise ValueError("No vector_* columns found in embeddings parquet.")
    # sort by the integer index after "vector_"
    vec_cols = sorted(vec_cols, key=lambda c: int(c.split("_")[1]))
    return vec_cols


def load_data(path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load all embeddings from a single parquet file at `path`.

    Expects columns:
      - id, name, album_name
      - vector_0 ... vector_(D-1)
    If the file does not exist, returns empty vectors & metadata.
    """
    if not os.path.exists(path):
        print(f"No existing embeddings file at {path}. Starting with empty dataset.")
        # Empty placeholders; actual dim will be handled in init()
        vectors = np.zeros((0, 0), dtype=np.float32)
        metadata = pd.DataFrame(columns=["id", "name", "album_name"])
        return vectors, metadata

    print(f"Loading embeddings from: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} songs from parquet")

    vec_cols = infer_vector_columns(df)
    vectors = df[vec_cols].to_numpy(dtype=np.float32)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    metadata = df[["id", "name", "album_name"]].reset_index(drop=True)
    return vectors, metadata


def save_data(path: str, vectors: np.ndarray, metadata: pd.DataFrame):
    """
    Save the full dataset to parquet at `path`.

    Columns:
      - id, name, album_name
      - vector_0 ... vector_(D-1)
    """
    ensure_embeddings_dir()

    if vectors.ndim != 2:
        raise ValueError(f"Expected vectors to be 2D, got shape {vectors.shape}")

    n, d = vectors.shape
    vector_cols = [f"vector_{i}" for i in range(d)]

    # Ensure metadata length matches number of vectors
    if len(metadata) != n:
        raise ValueError(
            f"Metadata length {len(metadata)} != number of vectors {n}"
        )

    df_meta = metadata.reset_index(drop=True).copy()
    df_vecs = pd.DataFrame(vectors, columns=vector_cols)
    df = pd.concat([df_meta, df_vecs], axis=1)
    df.to_parquet(path, index=False)
    print(f"Saved {n} songs to {path}")


def build_cosine_index(vectors: np.ndarray, dim: int | None = None) -> faiss.IndexFlatIP:
    """
    Build a cosine-similarity FAISS index.

    If vectors is empty, build an empty index of dimension `dim`
    (which must be provided).
    """
    if vectors.shape[0] == 0:
        if dim is None:
            raise ValueError("Cannot build empty index without embedding dimension.")
        index = faiss.IndexFlatIP(dim)
        return index

    x = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(x)
    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)
    return index


def load_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 2048
    return model


def embed_text(text: str) -> np.ndarray:
    """
    Returns a (1, D) float32 L2-normalized embedding for the input lyrics
    using BAAI/bge-m3.
    """
    global _model
    if _model is None:
        raise RuntimeError("Embedding model not loaded.")

    text = text.strip()
    if not text:
        raise ValueError("Empty text passed to embed_text")

    emb = _model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=1,
        show_progress_bar=False,
    ).astype(np.float32)

    emb = np.ascontiguousarray(emb, dtype=np.float32)
    return emb


def next_song_id() -> int:
    """
    Return the next integer ID based on _metadata.
    """
    global _metadata
    if _metadata is None or _metadata.empty:
        return 1
    return int(_metadata["id"].max()) + 1


# ----------------------------
# Initialization
# ----------------------------
def init():
    global _vectors, _metadata, _index, _model

    ensure_embeddings_dir()

    # 1) Load embedding model first so we know the embedding dimension
    _model = load_embedder(EMBED_MODEL)
    emb_dim = _model.get_sentence_embedding_dimension()

    # 2) Load existing data (or empty)
    _vectors, _metadata = load_data(SONG_EMBEDDINGS_FILE)

    # If we had no existing vectors, initialize an empty array with correct dim
    if _vectors.size == 0:
        _vectors = np.zeros((0, emb_dim), dtype=np.float32)
        _metadata = pd.DataFrame(columns=["id", "name", "album_name"])

    # 3) Build FAISS index
    _index = build_cosine_index(_vectors, dim=emb_dim)

    # 4) Sanity check: embedding dim must match
    test_vec = embed_text("hello world")
    if test_vec.shape[1] != _vectors.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: model outputs {test_vec.shape[1]}, "
            f"dataset is {_vectors.shape[1]}"
        )

    print(
        f"Initialization complete. Songs loaded: {_vectors.shape[0]}, "
        f"Embedding dim: {_vectors.shape[1]}"
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
    global _index, _metadata

    data = request.get_json(silent=True) or {}
    lyrics = data.get("lyrics", "")

    if not isinstance(lyrics, str) or not lyrics.strip():
        return jsonify({"error": "Missing or empty 'lyrics'"}), 400

    if _index is None or _metadata is None or _metadata.empty:
        return jsonify({"error": "No songs are indexed yet."}), 400

    query_vec = embed_text(lyrics)  # (1, D), normalized
    scores, indices = _index.search(query_vec, TOP_K)

    results: List[Dict] = []
    for rank, (i, s) in enumerate(zip(indices[0], scores[0]), start=1):
        if i < 0 or i >= len(_metadata):
            continue
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


@app.route("/addSong", methods=["POST"])
def add_song():
    """
    POST JSON: { "title": "<song title>", "lyrics": "<full lyrics>" }

    - Embeds the lyrics with BAAI/bge-m3
    - Adds the song to in-memory vectors, metadata, and FAISS index
    - Saves full dataset back to songembeddings.parquet

    Returns JSON with the new song's id and title.
    """
    global _vectors, _metadata, _index

    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    lyrics = data.get("lyrics", "")

    if not isinstance(title, str) or not title.strip():
        return jsonify({"error": "Missing or empty 'title'"}), 400
    if not isinstance(lyrics, str) or not lyrics.strip():
        return jsonify({"error": "Missing or empty 'lyrics'"}), 400

    # Embed lyrics
    try:
        vec = embed_text(lyrics)  # (1, D)
    except Exception as e:
        return jsonify({"error": f"Failed to embed lyrics: {e}"}), 500

    # Ensure globals are initialized
    if _vectors is None or _metadata is None or _index is None:
        return jsonify({"error": "Server not initialized yet."}), 500

    # Make sure dimensions match
    if _vectors.shape[1] != vec.shape[1]:
        return jsonify({
            "error": f"Embedding dimension mismatch. Existing: "
                     f"{_vectors.shape[1]}, new: {vec.shape[1]}"
        }), 500

    # Assign new ID
    new_id = next_song_id()

    # Append to in-memory vectors
    _vectors = np.vstack([_vectors, vec])

    # Append to metadata (title stored as 'name'; album_name left empty)
    new_row = {"id": new_id, "name": title.strip(), "album_name": ""}
    _metadata = pd.concat(
        [_metadata, pd.DataFrame([new_row])],
        ignore_index=True
    )

    # Add to FAISS index
    faiss.normalize_L2(vec)  # should already be normalized, but safe to call
    _index.add(vec)

    # Persist to parquet
    try:
        save_data(SONG_EMBEDDINGS_FILE, _vectors, _metadata)
    except Exception as e:
        return jsonify({
            "error": f"Song added in-memory but failed to save parquet: {e}",
            "id": new_id,
            "title": title.strip(),
        }), 500

    return jsonify({
        "status": "ok",
        "id": new_id,
        "title": title.strip(),
    }), 200


@app.route("/health", methods=["GET"])
def health():
    ready = _index is not None and _model is not None
    songs = int(_vectors.shape[0]) if _vectors is not None else 0
    return jsonify({
        "status": "ok" if ready else "initializing",
        "index_loaded": _index is not None,
        "model_loaded": _model is not None,
        "song_count": songs,
    })


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
