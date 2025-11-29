"""
experiments_tfidf_vs_embedding.py

Offline evaluation script for LyriMatch.

- Builds a simple TF-IDF baseline.
- Uses the same SentenceTransformer model as app.py for embedding-based search.
- Defines a small evaluation set based on artist identity.
- Computes Precision@k and Recall@k for both methods.
- Prints a few qualitative examples (query + top-k neighbors) so they can
  be copied into the report.

Run from the project root (where app.py lives):

    python experiments_tfidf_vs_embedding.py

Assumes data/lyrics_dataset.csv exists with columns:
    song_id, title, artist, lyrics
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random
from typing import List, Tuple, Dict, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This should exist after running build_lyrics_dataset_from_playlist.py
DATA_PATH = os.path.join(BASE_DIR, "data", "lyrics_dataset.csv")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_SEED = 42
TOP_K = 5       # for P@k / qualitative examples
EVAL_SIZE = 50  # number of query songs in the eval set (if available)

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run build_lyrics_dataset_from_playlist.py first."
        )
    df = pd.read_csv(path)
    expected_cols = {"song_id", "title", "artist", "lyrics"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=["lyrics"]).reset_index(drop=True)
    return df

def build_tfidf_model(df: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray]:
    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["lyrics"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix

def build_embedding_model(df: pd.DataFrame) -> Tuple[SentenceTransformer, np.ndarray, faiss.IndexFlatIP]:
    print("Building embedding matrix with SentenceTransformer...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = 512

    lyrics_list = df["lyrics"].tolist()
    all_embs = []

    # Manual batching (no multiprocessing, no progress bar) to avoid segfaults
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(lyrics_list), batch_size):
            batch = lyrics_list[start:start + batch_size]
            batch_emb = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=len(batch),
                show_progress_bar=False,
            ).astype(np.float32)
            all_embs.append(batch_emb)

    embeddings = np.vstack(all_embs)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Build cosine-similarity index (inner product with L2-normalized vectors)
    faiss.omp_set_num_threads(1)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    return model, embeddings, index

def get_relevant_by_artist(row: pd.Series, df: pd.DataFrame) -> Set[int]:
    """Return IDs of all songs by the same artist (excluding the query itself)."""
    artist = row["artist"]
    song_id = row["song_id"]
    same_artist = df[(df["artist"] == artist) & (df["song_id"] != song_id)]
    return set(same_artist["song_id"].tolist())

def precision_recall_at_k(pred_ids: List[int], true_ids: Set[int], k: int) -> Tuple[float, float]:
    pred_topk = pred_ids[:k]
    hits = sum(1 for sid in pred_topk if sid in true_ids)
    p_at_k = hits / k if k > 0 else 0.0
    r_at_k = hits / len(true_ids) if len(true_ids) > 0 else 0.0
    return p_at_k, r_at_k

def tfidf_neighbors(
    query_index: int,
    tfidf_matrix,
    df: pd.DataFrame,
    k: int = TOP_K
) -> List[int]:
    """
    Return indices of top-k neighbors using TF-IDF cosine similarity.
    query_index is the row index in df.
    """
    query_vec = tfidf_matrix[query_index]  # (1, vocab)
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]  # (N,)
    sims[query_index] = -1.0  # exclude self
    top_idx = np.argsort(-sims)[:k]
    return top_idx.tolist()

def embedding_neighbors(
    query_index: int,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    k: int = TOP_K
) -> List[int]:
    """
    Return indices of top-k neighbors using FAISS over embedding space.
    query_index is the row index in df.
    """
    query_vec = embeddings[query_index].reshape(1, -1)
    # search for k+1 neighbors because nearest is usually the query itself
    scores, idxs = index.search(query_vec, k + 1)
    idxs = idxs[0].tolist()
    # remove query_index if present
    idxs = [i for i in idxs if i != query_index]
    return idxs[:k]

def build_eval_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a small evaluation set of songs that have at least a few
    other songs by the same artist (so we have non-empty "relevant" sets).
    """
    group_sizes = df.groupby("artist")["song_id"].transform("size")
    candidates = df[group_sizes >= 4].reset_index(drop=True)  # need at least 3 other songs
    if candidates.empty:
        raise ValueError(
            "No artists with >= 4 songs in the dataset. "
            "Evaluation-by-artist won't work. Use a different heuristic or dataset."
        )
    n = len(candidates)
    size = min(EVAL_SIZE, n)
    random.seed(RANDOM_SEED)
    eval_indices = random.sample(range(n), size)
    eval_df = candidates.iloc[eval_indices].reset_index(drop=True)
    print(f"Evaluation set size: {len(eval_df)} (from {n} candidate songs)")
    return eval_df

def evaluate_model(
    df: pd.DataFrame,
    eval_df: pd.DataFrame,
    tfidf_matrix,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    k: int = TOP_K
) -> Dict[str, Tuple[float, float]]:
    """
    Compute average P@k and R@k for TF-IDF baseline and embedding model.
    Returns a dict:
        {"tfidf": (p_at_k, r_at_k), "embed": (p_at_k, r_at_k)}
    """
    ps_tfidf, rs_tfidf = [], []
    ps_emb, rs_emb = [], []

    # Map song_id -> row index in df for quick lookup
    id_to_idx = {sid: i for i, sid in enumerate(df["song_id"].tolist())}

    for _, row in eval_df.iterrows():
        song_id = row["song_id"]
        q_idx = id_to_idx[song_id]
        true_ids = get_relevant_by_artist(row, df)
        if not true_ids:
            continue

        # --- TF-IDF baseline ---
        tf_neighbors_idx = tfidf_neighbors(q_idx, tfidf_matrix, df, k=k)
        tf_neighbors_ids = [df.iloc[i]["song_id"] for i in tf_neighbors_idx]
        p_t, r_t = precision_recall_at_k(tf_neighbors_ids, true_ids, k)
        ps_tfidf.append(p_t); rs_tfidf.append(r_t)

        # --- Embedding model ---
        emb_neighbors_idx = embedding_neighbors(q_idx, embeddings, index, k=k)
        emb_neighbors_ids = [df.iloc[i]["song_id"] for i in emb_neighbors_idx]
        p_e, r_e = precision_recall_at_k(emb_neighbors_ids, true_ids, k)
        ps_emb.append(p_e); rs_emb.append(r_e)

    tfidf_p = sum(ps_tfidf) / len(ps_tfidf)
    tfidf_r = sum(rs_tfidf) / len(rs_tfidf)
    emb_p = sum(ps_emb) / len(ps_emb)
    emb_r = sum(rs_emb) / len(rs_emb)

    return {
        "tfidf": (tfidf_p, tfidf_r),
        "embed": (emb_p, emb_r),
    }

def print_qualitative_examples(
    df: pd.DataFrame,
    tfidf_matrix,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    num_examples: int = 3,
    k: int = TOP_K
) -> None:
    """
    Print a few query songs and their top-k neighbors according to each method.
    You can copy these into the report and add commentary.
    """
    print("\n=== Qualitative examples ===\n")
    n = len(df)
    random.seed(RANDOM_SEED + 1)
    example_indices = random.sample(range(n), min(num_examples, n))

    for j, q_idx in enumerate(example_indices, start=1):
        row = df.iloc[q_idx]
        print(f"Example {j}: Query song")
        print(f"  ID:    {row['song_id']}")
        print(f"  Title: {row['title']}")
        print(f"  Artist:{row['artist']}\n")

        # TF-IDF neighbors
        tf_neighbors_idx = tfidf_neighbors(q_idx, tfidf_matrix, df, k=k)
        print("  TF-IDF top-{} neighbors:".format(k))
        for rank, i in enumerate(tf_neighbors_idx, start=1):
            r = df.iloc[i]
            print(f"    {rank}. {r['title']} — {r['artist']} (id={r['song_id']})")
        print()

        # Embedding neighbors
        emb_neighbors_idx = embedding_neighbors(q_idx, embeddings, index, k=k)
        print("  Embedding top-{} neighbors:".format(k))
        for rank, i in enumerate(emb_neighbors_idx, start=1):
            r = df.iloc[i]
            print(f"    {rank}. {r['title']} — {r['artist']} (id={r['song_id']})")
        print("\n" + "-" * 60 + "\n")

def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Total songs: {len(df)}")

    vectorizer, tfidf_matrix = build_tfidf_model(df)
    model, embeddings, index = build_embedding_model(df)

    eval_df = build_eval_set(df)

    print("\nComputing P@{0} and R@{0} for both models...".format(TOP_K))
    scores = evaluate_model(df, eval_df, tfidf_matrix, embeddings, index, k=TOP_K)

    tfidf_p, tfidf_r = scores["tfidf"]
    emb_p, emb_r = scores["embed"]

    print("\n=== Quantitative results ===")
    print(f"TF-IDF baseline:     P@{TOP_K} = {tfidf_p:.3f}, R@{TOP_K} = {tfidf_r:.3f}")
    print(f"Embedding + FAISS:   P@{TOP_K} = {emb_p:.3f}, R@{TOP_K} = {emb_r:.3f}")

    print_qualitative_examples(df, tfidf_matrix, embeddings, index, num_examples=3, k=TOP_K)

if __name__ == "__main__":
    main()