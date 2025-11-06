// src/lib/api.ts

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
