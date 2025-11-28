/*
  # Create lyric searches table for LyriMatch

  1. New Tables
    - `lyric_searches`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `lyrics` (text, the input lyrics)
      - `recommendations` (jsonb, the AI recommendations)
      - `analysis` (text, the AI analysis)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on `lyric_searches` table
    - Add policy for authenticated users to insert their own searches
    - Add policy for authenticated users to read their own searches
*/

CREATE TABLE IF NOT EXISTS lyric_searches (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  lyrics text NOT NULL,
  recommendations jsonb DEFAULT '[]'::jsonb,
  analysis text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE lyric_searches ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert own searches"
  ON lyric_searches
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can read own searches"
  ON lyric_searches
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_lyric_searches_user_id ON lyric_searches(user_id);
CREATE INDEX IF NOT EXISTS idx_lyric_searches_created_at ON lyric_searches(created_at DESC);