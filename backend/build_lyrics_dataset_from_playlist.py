"""
build_lyrics_dataset_from_playlist.py

Create a small lyrics dataset from one or more Spotify playlists.

The output is data/lyrics_dataset.csv with columns:
    song_id, title, artist, lyrics

This dataset is used ONLY for offline experiments (TF-IDF vs embeddings)
in experiments_tfidf_vs_embedding.py. It does not affect the Flask API.

Usage:

    python build_lyrics_dataset_from_playlist.py

You will be prompted for a playlist URL. You can run this script multiple
times; new songs will be appended if they are not already in the CSV.
"""

import os
import csv
from typing import List, Tuple

import lyricsgenius as lg
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY", "")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise RuntimeError("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment.")
if not GENIUS_API_KEY:
    raise RuntimeError("Missing GENIUS_API_KEY in environment.")

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
)

genius = lg.Genius(
    GENIUS_API_KEY,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    remove_section_headers=True,
    timeout=15,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "lyrics_dataset.csv")

def get_playlist_tracks(playlist_url: str, song_limit: int | None = None) -> List[Tuple[str, str]]:
    """
    Return a list of (title, primary_artist) from the Spotify playlist.
    """
    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    print(f"Using playlist ID: {playlist_id}")

    results = sp.playlist_items(playlist_id)
    tracks: List[Tuple[str, str]] = []

    while results:
        for item in results["items"]:
            track = item.get("track")
            if not track:
                continue
            title = track["name"]
            artists = track.get("artists") or []
            primary_artist = artists[0]["name"] if artists else ""
            tracks.append((title, primary_artist))

            if song_limit is not None and len(tracks) >= song_limit:
                return tracks

        if results["next"]:
            results = sp.next(results)
        else:
            break

    return tracks

def load_existing_ids(path: str) -> set[int]:
    """
    If lyrics_dataset.csv already exists, load existing song_id values so
    we don't duplicate them. song_id is just an integer counter.
    """
    if not os.path.exists(path):
        return set()
    ids: set[int] = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.add(int(row["song_id"]))
            except Exception:
                continue
    return ids

def next_song_id(existing_ids: set[int]) -> int:
    return (max(existing_ids) + 1) if existing_ids else 1

def main():
    print("=== Build lyrics_dataset.csv from a Spotify playlist ===\n")
    playlist_url = input("Enter Spotify playlist URL: ").strip()
    if not playlist_url:
        print("No playlist URL provided. Exiting.")
        return

    try:
        raw_limit = input("Optional: max number of songs to pull (press Enter for all): ").strip()
        song_limit = int(raw_limit) if raw_limit else None
    except Exception:
        song_limit = None

    print("\nFetching tracks from playlist...")
    tracks = get_playlist_tracks(playlist_url, song_limit=song_limit)
    print(f"Found {len(tracks)} tracks in playlist.")

    existing_ids = load_existing_ids(CSV_PATH)
    current_id = next_song_id(existing_ids)

    # Ensure CSV has header if being created for the first time
    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["song_id", "title", "artist", "lyrics"])
    if new_file:
        writer.writeheader()

    added = 0
    skipped = 0

    for title, artist in tracks:
        print(f"\nProcessing: {title} — {artist}")
        try:
            song = genius.search_song(title, artist)
        except Exception as e:
            print(f"  [WARN] Genius search failed: {e}")
            skipped += 1
            continue

        if not song or not isinstance(song.lyrics, str) or not song.lyrics.strip():
            print("  [WARN] No lyrics found, skipping.")
            skipped += 1
            continue

        lyrics = song.lyrics.strip()

        writer.writerow(
            {
                "song_id": current_id,
                "title": title.strip(),
                "artist": artist.strip(),
                "lyrics": lyrics,
            }
        )
        print(f"  [OK] Added with song_id={current_id}")
        current_id += 1
        added += 1

    f.close()
    print("\n=== Done ===")
    print(f"Added:   {added}")
    print(f"Skipped: {skipped}")
    print(f"Dataset saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()