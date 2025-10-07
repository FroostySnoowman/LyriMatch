import { useState, FormEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Music2, Sparkles } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import UnifiedNavbar from '../components/UnifiedNavbar';
import TextArea from '../components/TextArea';
import Button from '../components/Button';
import LoadingSpinner from '../components/LoadingSpinner';
import { analyzeLyrics, SongRecommendation } from '../lib/api';

export default function App() {
  const { user } = useAuth();
  const [lyrics, setLyrics] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<{
    recommendations: SongRecommendation[];
    analysis: string;
  } | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!lyrics.trim()) return;

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const data = await analyzeLyrics(lyrics, user?.id || '');
      setResults(data);
    } catch (err) {
      setError('Failed to analyze lyrics. Please make sure your backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-black" />

      <UnifiedNavbar />

      <div className="relative z-10 pt-24 px-6 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          className="max-w-5xl mx-auto"
        >
          <div className="text-center mb-12">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-full mb-6"
            >
              <Sparkles size={16} className="text-gray-400" />
              <span className="text-sm text-gray-400">AI-Powered Analysis</span>
            </motion.div>

            <h1 className="text-5xl font-bold mb-4">
              Discover Songs from Lyrics
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Enter lyrics that resonate with you and let AI find songs with similar themes and emotions
            </p>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, duration: 0.4, ease: 'easeOut' }}
            className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-sm mb-8"
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              <TextArea
                value={lyrics}
                onChange={(e) => setLyrics(e.target.value)}
                placeholder="Enter your favorite lyrics here...&#10;&#10;Example:&#10;I've been trying to do it right&#10;I've been living a lonely life&#10;I've been sleeping here instead&#10;I've been sleeping in my bed"
                rows={8}
                required
              />

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm"
                >
                  {error}
                </motion.div>
              )}

              <Button
                type="submit"
                variant="primary"
                className="w-full"
                disabled={loading || !lyrics.trim()}
              >
                {loading ? (
                  <>
                    <LoadingSpinner />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Send size={20} />
                    Find Matches
                  </>
                )}
              </Button>
            </form>
          </motion.div>

          <AnimatePresence mode="wait">
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-20"
              >
                <LoadingSpinner />
                <p className="mt-6 text-gray-400">Analyzing your lyrics...</p>
              </motion.div>
            )}

            {results && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -40 }}
                className="space-y-8"
              >
                {results.analysis && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-sm"
                  >
                    <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                      <Sparkles size={24} />
                      Analysis
                    </h2>
                    <p className="text-gray-300 leading-relaxed">{results.analysis}</p>
                  </motion.div>
                )}

                <div>
                  <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                    <Music2 size={24} />
                    Recommended Songs
                  </h2>

                  <div className="grid gap-6">
                    {results.recommendations.map((song, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 * index }}
                        className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm hover:bg-white/10 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold">{song.title}</h3>
                            <p className="text-gray-400">{song.artist}</p>
                          </div>
                          <div className="flex items-center gap-2 px-3 py-1 bg-white/10 rounded-full">
                            <span className="text-sm font-medium">
                              {Math.round(song.similarity * 100)}% Match
                            </span>
                          </div>
                        </div>
                        <p className="text-gray-300 leading-relaxed">{song.reason}</p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}