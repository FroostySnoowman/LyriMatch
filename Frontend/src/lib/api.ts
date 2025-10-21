export interface SongRecommendation {
  title: string;
  artist: string;
  similarity: number;
  reason: string;
}

export interface AnalysisResponse {
  recommendations: SongRecommendation[];
  analysis: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function analyzeLyrics(lyrics: string, userId: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ lyrics, user_id: userId }),
  });

  if (!response.ok) {
    throw new Error('Failed to analyze lyrics');
  }

  return response.json();
}

export async function getUserHistory(userId: string) {
  const response = await fetch(`${API_BASE_URL}/api/history/${userId}`);

  if (!response.ok) {
    throw new Error('Failed to fetch history');
  }

  return response.json();
}
