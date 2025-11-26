import time
import requests

host = "127.0.0.1"
port = 5000
BASE_URL = f"http://{host}:{port}"

add_url = f"{BASE_URL}/add_song"
search_url = f"{BASE_URL}/search"

# --- 1) Add three simple "songs" ---

songs = [
    {
        "title": "Sunny Day",
        "lyrics": "I am walking in the sunshine and feeling so happy today."
    },
    {
        "title": "Rainy Night",
        "lyrics": "The rain is falling softly while I sit alone at night."
    },
    {
        "title": "Study Time",
        "lyrics": "I am sitting at my desk, studying and drinking coffee."
    },
]

print("Adding test songs...\n")

for song in songs:
    print(f"→ Adding '{song['title']}'")
    r = requests.post(add_url, json=song)
    if r.status_code != 200:
        raise SystemExit(f"❌ Failed to add song '{song['title']}': {r.status_code} {r.text}")
    print(f"   ✅ Added with response: {r.json()}")

# --- 2) Query with a similar line ---

query_lyrics = "I feel so happy walking in the bright sunshine."
payload = {"lyrics": query_lyrics}

print("\n----------------------------------------")
print(f"Calling {search_url} with query:\n{query_lyrics}\n")

t0 = time.perf_counter()
r = requests.post(search_url, json=payload)
elapsed = time.perf_counter() - t0

if r.status_code != 200:
    raise SystemExit(f"❌ Expected 200 OK, got {r.status_code}: {r.text}")

data = r.json()
results = data.get("results", [])

print(f"Request completed in {elapsed:.2f}s")
print("Top 5 results:")
for item in results:
    print(
        f"{item['rank']}. {item['name']} — {item.get('album_name', '')} "
        f"[ID: {item['id']}] (cosine_sim={item['cosine_sim']:.4f})"
    )
