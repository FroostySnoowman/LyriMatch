# Minimal Flask API:
#   POST /search   { "lyrics": "<text>" }                ->  top-5 nearest songs as JSON
#   POST /add_song  { "title": "<title>", "lyrics": "" } ->  add song & embedding to parquet + index
#   GET /health                                          ->  health check endpoint

import os
import re
import difflib
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SONG_EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "song_embeddings")
SONG_EMBEDDINGS_FILE = os.path.join(SONG_EMBEDDINGS_DIR, "songembeddings.parquet")

# Use a smaller, stable embedding model instead of BAAI/bge-m3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

app = Flask(__name__)
CORS(app)

_vectors: np.ndarray | None = None
_metadata: pd.DataFrame | None = None
_index: faiss.IndexFlatIP | None = None
_model: SentenceTransformer | None = None
_song_keys_full: set[Tuple[str, str]] | None = None
_song_keys_title_only: set[str] | None = None

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

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
        metadata = pd.DataFrame(columns=["id", "name", "album_name", "lyrics"])
        return vectors, metadata

    print(f"Loading embeddings from: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} songs from parquet")

    vec_cols = infer_vector_columns(df)
    vectors = df[vec_cols].to_numpy(dtype=np.float32)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    # Support existing parquet files without lyrics column.
    meta_cols = ["id", "name", "album_name"]
    if "lyrics" in df.columns:
        meta_cols.append("lyrics")

    metadata = df[meta_cols].reset_index(drop=True)
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

def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value

def _normalize_for_match(text: str) -> str:
    """
    Lightweight normalization for lyric overlap:
    - lowercases
    - strips punctuation
    - joins tokens with single spaces
    """
    if not isinstance(text, str):
        text = str(text)
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return " ".join(tokens)

def _partial_ratio(a: str, b: str) -> float:
    """
    Approximate fuzzywuzzy.partial_ratio:
    finds the best-matching substring of the longer string against the shorter.
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a

    matcher = difflib.SequenceMatcher(None, a, b)
    blocks = matcher.get_matching_blocks()
    best = 0.0
    la = len(a)

    for block in blocks:
        start = max(block[1] - block[0], 0)
        end = start + la
        substring = b[start:end]
        r = difflib.SequenceMatcher(None, a, substring).ratio()
        if r > best:
            best = r

    return best

def lyric_overlap_score(query_text: str, lyrics_text: str) -> float:
    """
    Blend:
    - exact normalized substring match (strongest signal)
    - token coverage: fraction of query tokens that appear anywhere in the lyrics
    - partial fuzzy ratio: how well the query matches some window of the lyrics

    Exact or near-exact snippets from the same song will score very close to 1.0.
    """
    q_norm = _normalize_for_match(query_text)
    t_norm = _normalize_for_match(lyrics_text)

    if not q_norm or not t_norm:
        return 0.0

    # Exact normalized snippet inside normalized lyrics → treat as perfect match.
    if q_norm in t_norm:
        return 1.0

    q_tokens = q_norm.split()
    t_tokens = t_norm.split()
    if not q_tokens or not t_tokens:
        return 0.0

    t_set = set(t_tokens)
    common = sum(1 for tok in q_tokens if tok in t_set)
    coverage = common / len(q_tokens)

    partial = _partial_ratio(q_norm, t_norm)

    # Weight coverage higher so high-overlap snippets are close to 1.0
    return 0.7 * coverage + 0.3 * partial

def register_song_key(name: str, album_name: str = ""):
    """
    Register a song into the in-memory duplicate key sets.

    name is the stored 'name' column, album_name is the stored 'album_name'
    (often used as artist). Handles legacy rows where name='title - artist'
    and album_name may be empty.
    """
    global _song_keys_full, _song_keys_title_only
    if _song_keys_full is None:
        _song_keys_full = set()
    if _song_keys_title_only is None:
        _song_keys_title_only = set()

    n = _normalize_text(name)
    a = _normalize_text(album_name)

    if a:
        title = n
        artist = a
        suffix = f" - {artist}"
        if n.endswith(suffix):
            title = _normalize_text(n[: -len(suffix)])
        _song_keys_full.add((title, artist))
    else:
        if " - " in n:
            raw_title, raw_artist = n.split(" - ", 1)
            title = _normalize_text(raw_title)
            artist = _normalize_text(raw_artist)
            _song_keys_full.add((title, artist))
        else:
            title = n
            _song_keys_title_only.add(title)

def rebuild_song_keys():
    global _song_keys_full, _song_keys_title_only, _metadata
    _song_keys_full = set()
    _song_keys_title_only = set()
    if _metadata is None or _metadata.empty:
        return
    for _, row in _metadata.iterrows():
        name = row.get("name", "")
        album_name = row.get("album_name", "")
        register_song_key(name, album_name)

def song_exists(name: str, album_name: str = "") -> bool:
    """
    Check if a song with the given name and artist/album already exists.
    
    If album_name is provided, checks for exact match of both name and album.
    If album_name is empty, only checks if the name exists.
    
    This allows multiple songs with the same title but different artists/albums.
    """
    global _metadata, _song_keys_full, _song_keys_title_only
    if _metadata is None or _metadata.empty:
        return False

    if _song_keys_full is None or _song_keys_title_only is None:
        rebuild_song_keys()

    name_norm = _normalize_text(name)
    album_norm = _normalize_text(album_name)

    if album_norm:
        # Full (title, artist/album) duplicate check.
        key = (name_norm, album_norm)
        return key in _song_keys_full

    return name_norm in _song_keys_title_only

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
        _metadata = pd.DataFrame(columns=["id", "name", "album_name", "lyrics"])
    else:
        # Backfill lyrics column for older parquet files that don't have it.
        if "lyrics" not in _metadata.columns:
            _metadata["lyrics"] = ""

    # Build duplicate key sets from whatever metadata we loaded.
    rebuild_song_keys()

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

    # Retrieve a larger candidate set from FAISS, then re-rank by lyric overlap.
    ntotal = _index.ntotal if hasattr(_index, "ntotal") else len(_metadata)
    if ntotal <= 0:
        return jsonify({"error": "No songs are indexed yet."}), 400

    candidate_k = min(max(TOP_K * 25, 100), ntotal)
    scores, indices = _index.search(query_vec, candidate_k)

    has_lyrics = "lyrics" in _metadata.columns
    raw_results: List[Dict] = []

    for i, s in zip(indices[0], scores[0]):
        if i < 0 or i >= len(_metadata):
            continue
        row = _metadata.iloc[int(i)]

        lyrics_text = ""
        if has_lyrics:
            val = row.get("lyrics", "")
            if isinstance(val, str):
                lyrics_text = val

        overlap = lyric_overlap_score(lyrics, lyrics_text) if lyrics_text else 0.0

        # Combined match score: almost entirely lyric-driven; cosine nudges ties.
        raw_cos = float(s)
        cos_clamped = float(max(s, 0.0))
        match_score = 0.95 * overlap + 0.05 * cos_clamped

        raw_results.append(
            {
                "name": str(row["name"]),
                "album_name": str(row["album_name"]),
                "id": int(row["id"]),
                "raw_cosine_sim": raw_cos,
                "lyric_overlap": float(overlap),
                "match_score": float(match_score),
            }
        )

    if not raw_results:
        return jsonify({"results": []}), 200

    # Sort by combined match score.
    raw_results.sort(key=lambda r: r["match_score"], reverse=True)

    results: List[Dict] = []
    for rank, r in enumerate(raw_results[:TOP_K], start=1):
        results.append(
            {
                "rank": int(rank),
                "name": r["name"],
                "album_name": r["album_name"],
                "id": r["id"],
                "cosine_sim": r["match_score"],
                "lyric_overlap": r["lyric_overlap"],
                "match_score": r["match_score"],
                "raw_cosine_sim": r["raw_cosine_sim"],
            }
        )

    return jsonify({"results": results})

@app.route("/add_song", methods=["POST"])
def add_song():
    global _vectors, _metadata, _index

    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    lyrics = data.get("lyrics", "")
    album = data.get("album", "")  # Optional album/artist field

    if not isinstance(title, str) or not title.strip():
        return jsonify({"error": "Missing or empty 'title'"}), 400
    if not isinstance(lyrics, str) or not lyrics.strip():
        return jsonify({"error": "Missing or empty 'lyrics'"}), 400

    if _vectors is None or _metadata is None or _index is None:
        return jsonify({"error": "Server not initialized yet."}), 500

    # Check for duplicate song name (with album/artist if provided)
    if song_exists(title, album):
        if album:
            return jsonify({"error": f"Song '{title.strip()}' by '{album.strip()}' already exists in the database."}), 409
        else:
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

    new_row = {
        "id": new_id,
        "name": title.strip(),
        "album_name": album.strip() if album else "",
        "lyrics": lyrics.strip(),
    }
    _metadata = pd.concat(
        [_metadata, pd.DataFrame([new_row])],
        ignore_index=True
    )
    register_song_key(new_row["name"], new_row["album_name"])

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
        return jsonify({"error": "Failed to read playlist: {}".format(e)}), 500
    except Exception as e:
        return jsonify({"error": "Failed to read playlist: {}".format(e)}), 500

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
    
    # Check for duplicate: if we have artist info, check title+artist combo
    if song_exists(display_title, display_artist):
        if display_artist:
            return jsonify({"error": f"Song '{display_title}' by '{display_artist}' already exists in the database."}), 409
        else:
            return jsonify({"error": f"Song '{name}' already exists in the database."}), 409

    try:
        song = genius.search_song(display_title, display_artist)
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

    # Store artist in album_name field for consistency
    new_row = {
        "id": new_id,
        "name": name,
        "album_name": display_artist,
        "lyrics": lyrics.strip(),
    }
    _metadata = pd.concat(
        [_metadata, pd.DataFrame([new_row])],
        ignore_index=True
    )
    register_song_key(new_row["name"], new_row["album_name"])

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
        display_title = title.strip()
        display_artist = primary_artist.strip()

        # Skip if this (title, artist) pair is already present.
        if song_exists(display_title, display_artist):
            continue

        try:
            song = genius.search_song(display_title, display_artist)
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

        name = display_title if not display_artist else f"{display_title} - {display_artist}"

        new_row = {
            "id": new_id,
            "name": name,
            "album_name": "",
            "lyrics": lyrics.strip(),
        }
        _metadata = pd.concat(
            [_metadata, pd.DataFrame([new_row])],
            ignore_index=True
        )
        register_song_key(new_row["name"], new_row["album_name"])

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

@app.route("/browse_songs", methods=["GET"])
def browse_songs():
    global _metadata
    
    if _metadata is None or _metadata.empty:
        return jsonify({
            "songs": [],
            "total": 0,
            "page": 1,
            "per_page": 20,
            "total_pages": 0,
        }), 200
    
    # Get query parameters
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search_query = request.args.get("query", "").strip()
    sort_by = request.args.get("sort_by", "id")  # id, name, album_name
    sort_order = request.args.get("sort_order", "asc")  # asc, desc
    
    # Validate parameters
    page = max(1, page)
    per_page = max(1, min(100, per_page))  # Limit to 100 per page
    
    # Work with a copy to avoid modifying global metadata
    df = _metadata.copy()
    
    # Apply search filter if query provided
    if search_query:
        # Case-insensitive search in both name and album_name
        mask = (
            df["name"].str.lower().str.contains(search_query.lower(), na=False) |
            df["album_name"].str.lower().str.contains(search_query.lower(), na=False)
        )
        df = df[mask]
    
    # Apply sorting
    if sort_by in ["id", "name", "album_name"]:
        ascending = sort_order.lower() == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Calculate pagination
    total_songs = len(df)
    total_pages = (total_songs + per_page - 1) // per_page  # Ceiling division
    
    # Validate page number
    if page > total_pages and total_pages > 0:
        page = total_pages
    
    # Get page slice
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_df = df.iloc[start_idx:end_idx]
    
    # Convert to list of dicts
    songs = []
    for _, row in page_df.iterrows():
        songs.append({
            "id": int(row["id"]),
            "name": str(row["name"]),
            "album_name": str(row["album_name"]),
        })
    
    return jsonify({
        "songs": songs,
        "total": total_songs,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
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
    app.run(host="0.0.0.0", port=PORT, debug=False)