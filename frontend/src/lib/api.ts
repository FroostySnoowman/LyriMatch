export type SongRecommendation = {
  title: string;
  artist: string;
  similarity: number;
  reason: string;
};

export type AnalyzeResponse = {
  recommendations: SongRecommendation[];
  analysis: string;
};

export type AddSongResponse = {
  status: string;
  id: number;
  title: string;
};

export type AddPlaylistResponse = {
  status: string;
  added: {
    id: number;
    title: string;
    artist: string;
  }[];
};

export type PlaylistTracksResponse = {
  status: string;
  tracks: {
    title: string;
    artists: string[];
  }[];
  total: number;
};

export type AddSongFromSearchResponse = {
  status: string;
  id: number;
  title: string;
  artist: string;
};

export type BrowseSong = {
  id: number;
  name: string;
  album_name: string;
};

export type BrowseSongsResponse = {
  songs: BrowseSong[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
};

// Use backend URL from Vite environment variable
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

/**
 * Sends lyrics to the Flask backend and returns analyzed results.
 */
export async function analyzeLyrics(lyrics: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ lyrics }),
  });

  if (!res.ok) {
    throw new Error(`API Error: ${res.status}`);
  }

  const data = await res.json();

  /**
   * The backend currently returns:
   * { results: [ { name, album_name, cosine_sim, ... }, ... ] }
   * But App.tsx expects:
   * { analysis: string, recommendations: SongRecommendation[] }
   */

  return {
    analysis: "", // Optional placeholder (can be filled later if backend supports it)
    recommendations: data.results.map((song: any) => ({
      title: song.name,
      artist: song.album_name,
      similarity: song.cosine_sim,
      reason: "Similar based on lyrics embedding.", // Placeholder text
    })),
  };
}

/**
 * Adds a new song with lyrics to the Flask backend database.
 */
export async function addSong(title: string, lyrics: string): Promise<AddSongResponse> {
  const res = await fetch(`${API_BASE}/add_song`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title, lyrics }),
  });

  if (!res.ok) {
    throw new Error(`API Error: ${res.status}`);
  }

  const data = await res.json();
  return data;
}

/**
 * Adds songs from a Spotify playlist to the Flask backend database.
 */
export async function addPlaylist(playlistUrl: string, songLimit?: number): Promise<AddPlaylistResponse> {
  const res = await fetch(`${API_BASE}/add_playlist`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ playlist_url: playlistUrl, song_limit: songLimit }),
  });

  if (!res.ok) {
    throw new Error(`API Error: ${res.status}`);
  }

  const data = await res.json();
  return data;
}

/**
 * Fetches the track list from a Spotify playlist without processing.
 */
export async function getPlaylistTracks(playlistUrl: string, songLimit?: number): Promise<PlaylistTracksResponse> {
  const res = await fetch(`${API_BASE}/get_playlist_tracks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ playlist_url: playlistUrl, song_limit: songLimit }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `API Error: ${res.status}`);
  }

  const data = await res.json();
  return data;
}

/**
 * Adds a song by searching for lyrics on Genius.
 */
export async function addSongFromSearch(title: string, artist: string): Promise<AddSongFromSearchResponse> {
  const res = await fetch(`${API_BASE}/add_song_from_search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title, artist }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `API Error: ${res.status}`);
  }

  const data = await res.json();
  return data;
}

/**
 * Browse all songs in the database with search, pagination, and sorting.
 */
export async function browseSongs(
  page: number = 1,
  perPage: number = 20,
  query: string = "",
  sortBy: "id" | "name" | "album_name" = "id",
  sortOrder: "asc" | "desc" = "asc"
): Promise<BrowseSongsResponse> {
  const params = new URLSearchParams({
    page: page.toString(),
    per_page: perPage.toString(),
    sort_by: sortBy,
    sort_order: sortOrder,
  });

  if (query) {
    params.append("query", query);
  }

  const res = await fetch(`${API_BASE}/browse_songs?${params.toString()}`);

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `API Error: ${res.status}`);
  }

  const data = await res.json();
  return data;
}