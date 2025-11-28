# Minimal Flask API:
#   POST /search   { "lyrics": "<text>" }                ->  top-5 nearest songs as JSON
#   POST /add_song  { "title": "<title>", "lyrics": "" } ->  add song & embedding to parquet + index
#   GET /health                                          ->  health check endpoint

import os
from typing import List, Dict, Tuple

# reduce chances of OpenMP / tokenizer crashes on Apple Silicon
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import lyricsgenius as lg
import spotipy
import numpy as np
import pandas as pd
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import torch

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY", "")
PORT = int(os.getenv("PORT", "8080"))

# Spotify API Credentials
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

# Genius Lyrics API Key
genius = lg.Genius(
    GENIUS_API_KEY,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    remove_section_headers=True,
    timeout=15
)

# Directory that contains songembeddings.parquet (or nothing)
SONG_EMBEDDINGS_DIR = r"data\song_embeddings"
SONG_EMBEDDINGS_FILE = os.path.join(SONG_EMBEDDINGS_DIR, "songembeddings.parquet")

# Use a smaller, stable embedding model instead of BAAI/bge-m3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

app = Flask(__name__)
CORS(app)

_vectors: np.ndarray | None = None    # np.ndarray [N, D], float32, L2-normalized
_metadata: pd.DataFrame | None = None # columns: id, name, album_name
_index: faiss.IndexFlatIP | None = None
_model: SentenceTransformer | None = None

def ensure_embeddings_dir():
    os.makedirs(SONG_EMBEDDINGS_DIR, exist_ok=True)

def infer_vector_columns(df: pd.DataFrame) -> List[str]:
    vec_cols = [c for c in df.columns if c.startswith("vector_")]
    if not vec_cols:
        raise ValueError("No vector_* columns found in embeddings parquet.")
    vec_cols = sorted(vec_cols, key=lambda c: int(c.split("_")[1]))
    return vec_cols

def load_data(path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    if not os.path.exists(path):
        print(f"No existing embeddings file at {path}. Starting with empty dataset.")
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
    ensure_embeddings_dir()

    if vectors.ndim != 2:
        raise ValueError(f"Expected vectors to be 2D, got shape {vectors.shape}")

    n, d = vectors.shape
    vector_cols = [f"vector_{i}" for i in range(d)]

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
    torch.set_num_threads(1)
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 512
    return model

def embed_text(text: str) -> np.ndarray:
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

def song_exists(name: str) -> bool:
    """Check if a song with the given name already exists in the database."""
    global _metadata
    if _metadata is None or _metadata.empty:
        return False
    # Case-insensitive check for duplicate song names
    return name.strip().lower() in _metadata["name"].str.lower().values

def next_song_id() -> int:
    global _metadata
    if _metadata is None or _metadata.empty:
        return 1
    return int(_metadata["id"].max()) + 1

def get_playlist_tracks(playlist_url, song_limit=None):
    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    print(f"Using playlist ID: {playlist_id}")

    results = sp.playlist_items(playlist_id)
    tracks = []

    while results:
        for item in results["items"]:
            track = item["track"]
            if not track:
                continue

            title = track["name"]
            artists = [a["name"] for a in track["artists"]]
            tracks.append((title, artists))

            if song_limit is not None and len(tracks) >= song_limit:
                return tracks

        if results["next"]:
            results = sp.next(results)
        else:
            break

    return tracks

def init():
    global _vectors, _metadata, _index, _model

    ensure_embeddings_dir()

    _model = load_embedder(EMBED_MODEL)
    emb_dim = _model.get_sentence_embedding_dimension()

    _vectors, _metadata = load_data(SONG_EMBEDDINGS_FILE)

    if _vectors.size == 0:
        _vectors = np.zeros((0, emb_dim), dtype=np.float32)
        _metadata = pd.DataFrame(columns=["id", "name", "album_name"])

    _index = build_cosine_index(_vectors, dim=emb_dim)

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

@app.route("/search", methods=["POST"])
def search():
    global _index, _metadata

    data = request.get_json(silent=True) or {}
    lyrics = data.get("lyrics", "")

    if not isinstance(lyrics, str) or not lyrics.strip():
        return jsonify({"error": "Missing or empty 'lyrics'"}), 400

    if _index is None or _metadata is None or _metadata.empty:
        return jsonify({"error": "No songs are indexed yet."}), 400

    query_vec = embed_text(lyrics)
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

@app.route("/add_song", methods=["POST"])
def add_song():
    global _vectors, _metadata, _index

    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    lyrics = data.get("lyrics", "")

    if not isinstance(title, str) or not title.strip():
        return jsonify({"error": "Missing or empty 'title'"}), 400
    if not isinstance(lyrics, str) or not lyrics.strip():
        return jsonify({"error": "Missing or empty 'lyrics'"}), 400

    if _vectors is None or _metadata is None or _index is None:
        return jsonify({"error": "Server not initialized yet."}), 500

    # Check for duplicate song name
    if song_exists(title):
        return jsonify({"error": f"Song '{title.strip()}' already exists in the database."}), 409

    try:
        vec = embed_text(lyrics)
    except Exception as e:
        return jsonify({"error": f"Failed to embed lyrics: {e}"}), 500

    if _vectors.shape[1] != vec.shape[1]:
        return jsonify({
            "error": f"Embedding dimension mismatch. Existing: "
                     f"{_vectors.shape[1]}, new: {vec.shape[1]}"
        }), 500

    new_id = next_song_id()

    _vectors = np.vstack([_vectors, vec])

    new_row = {"id": new_id, "name": title.strip(), "album_name": ""}
    _metadata = pd.concat(
        [_metadata, pd.DataFrame([new_row])],
        ignore_index=True
    )

    faiss.normalize_L2(vec)
    _index.add(vec)

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

@app.route("/get_playlist_tracks", methods=["POST"])
def get_playlist_tracks_endpoint():
    data = request.get_json(silent=True) or {}
    playlist_url = data.get("playlist_url", "")
    song_limit = data.get("song_limit")

    if not isinstance(playlist_url, str) or not playlist_url.strip():
        return jsonify({"error": "Missing or empty 'playlist_url'"}), 400

    if song_limit is not None:
        try:
            song_limit = int(song_limit)
            if song_limit <= 0:
                song_limit = None
        except Exception:
            song_limit = None

    try:
        tracks = get_playlist_tracks(playlist_url, song_limit=song_limit)
    except SpotifyException as e:
        if e.http_status == 404:
            return jsonify({"error": "Playlist not found or not accessible via API."}), 400
        return jsonify({"error": f"Failed to read playlist: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to read playlist: {e}"}), 500

    if not tracks:
        return jsonify({"error": "No tracks found for this playlist."}), 400

    track_list = [
        {"title": title, "artists": artists}
        for title, artists in tracks
    ]

    return jsonify({
        "status": "ok",
        "tracks": track_list,
        "total": len(track_list),
    }), 200

@app.route("/add_song_from_search", methods=["POST"])
def add_song_from_search():
    global _vectors, _metadata, _index

    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    artist = data.get("artist", "")

    if not isinstance(title, str) or not title.strip():
        return jsonify({"error": "Missing or empty 'title'"}), 400

    if _vectors is None or _metadata is None or _index is None:
        return jsonify({"error": "Server not initialized yet."}), 500

    display_title = title.strip()
    display_artist = artist.strip()
    name = display_title if not display_artist else f"{display_title} - {display_artist}"

    # Check for duplicate song name
    if song_exists(name):
        return jsonify({"error": f"Song '{name}' already exists in the database."}), 409

    try:
        song = genius.search_song(title, artist)
    except Exception as e:
        return jsonify({"error": f"Failed to search for lyrics: {e}"}), 500

    if not song or not isinstance(song.lyrics, str) or not song.lyrics.strip():
        return jsonify({"error": f"No lyrics found for '{title}' by '{artist}'"}), 404

    lyrics = song.lyrics

    try:
        vec = embed_text(lyrics)
    except Exception as e:
        return jsonify({"error": f"Failed to embed lyrics: {e}"}), 500

    if _vectors.shape[1] != vec.shape[1]:
        return jsonify({
            "error": f"Embedding dimension mismatch. Existing: "
                     f"{_vectors.shape[1]}, new: {vec.shape[1]}"
        }), 500

    new_id = next_song_id()

    _vectors = np.vstack([_vectors, vec])

    new_row = {"id": new_id, "name": name, "album_name": ""}
    _metadata = pd.concat(
        [_metadata, pd.DataFrame([new_row])],
        ignore_index=True
    )

    faiss.normalize_L2(vec)
    _index.add(vec)

    try:
        save_data(SONG_EMBEDDINGS_FILE, _vectors, _metadata)
    except Exception as e:
        return jsonify({
            "error": f"Song added in-memory but failed to save parquet: {e}",
            "id": new_id,
            "title": display_title,
            "artist": display_artist,
        }), 500

    return jsonify({
        "status": "ok",
        "id": new_id,
        "title": display_title,
        "artist": display_artist,
    }), 200

@app.route("/add_playlist", methods=["POST"])
def add_playlist():
    global _vectors, _metadata, _index

    data = request.get_json(silent=True) or {}
    playlist_url = data.get("playlist_url", "")
    song_limit = data.get("song_limit")

    if not isinstance(playlist_url, str) or not playlist_url.strip():
        return jsonify({"error": "Missing or empty 'playlist_url'"}), 400

    if song_limit is not None:
        try:
            song_limit = int(song_limit)
            if song_limit <= 0:
                song_limit = None
        except Exception:
            song_limit = None

    if _vectors is None or _metadata is None or _index is None:
        return jsonify({"error": "Server not initialized yet."}), 500

    try:
        tracks = get_playlist_tracks(playlist_url, song_limit=song_limit)
    except SpotifyException as e:
        if e.http_status == 404:
            return jsonify({"error": "Playlist not found or not accessible via API."}), 400
        return jsonify({"error": f"Failed to read playlist: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to read playlist: {e}"}), 500

    if not tracks:
        return jsonify({"error": "No tracks found for this playlist."}), 400

    added = []

    for title, artists in tracks:
        primary_artist = artists[0] if artists else ""
        try:
            song = genius.search_song(title, primary_artist)
        except Exception:
            continue

        if not song or not isinstance(song.lyrics, str) or not song.lyrics.strip():
            continue

        lyrics = song.lyrics

        try:
            vec = embed_text(lyrics)
        except Exception as e:
            return jsonify({"error": f"Failed to embed lyrics for '{title}': {e}"}), 500

        if _vectors.shape[1] != vec.shape[1]:
            return jsonify({
                "error": f"Embedding dimension mismatch. Existing: "
                         f"{_vectors.shape[1]}, new: {vec.shape[1]}"
            }), 500

        new_id = next_song_id()

        _vectors = np.vstack([_vectors, vec])

        display_title = title.strip()
        display_artist = primary_artist.strip()
        name = display_title if not display_artist else f"{display_title} - {display_artist}"

        new_row = {"id": new_id, "name": name, "album_name": ""}
        _metadata = pd.concat(
            [_metadata, pd.DataFrame([new_row])],
            ignore_index=True
        )

        faiss.normalize_L2(vec)
        _index.add(vec)

        added.append({
            "id": new_id,
            "title": display_title,
            "artist": display_artist,
        })

    if not added:
        return jsonify({"error": "No lyrics were embedded for this playlist."}), 400

    try:
        save_data(SONG_EMBEDDINGS_FILE, _vectors, _metadata)
    except Exception as e:
        return jsonify({
            "error": f"Songs added in-memory but failed to save parquet: {e}",
            "added": added,
        }), 500

    return jsonify({
        "status": "ok",
        "added": added,
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

if __name__ == "__main__":
    # keep faiss single-threaded as well
    faiss.omp_set_num_threads(1)
    init()
    app.run(host="127.0.0.1", port=PORT, debug=False)