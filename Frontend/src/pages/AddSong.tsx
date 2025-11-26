import { useState, FormEvent } from 'react';
import { motion } from 'framer-motion';
import { Plus, Music2, Sparkles } from 'lucide-react';
import UnifiedNavbar from '../components/UnifiedNavbar';
import Input from '../components/Input';
import TextArea from '../components/TextArea';
import Button from '../components/Button';
import LoadingSpinner from '../components/LoadingSpinner';
import { addSong } from '../lib/api';

export default function AddSong() {
  const [title, setTitle] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<{ id: number; title: string } | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !lyrics.trim()) return;

    setLoading(true);
    setError('');
    setSuccess(null);

    try {
      const data = await addSong(title, lyrics);
      setSuccess(data);
      setTitle('');
      setLyrics('');
    } catch (err) {
      setError('Failed to add song. Please make sure your backend is running.');
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
              <span className="text-sm text-gray-400">Expand the Database</span>
            </motion.div>

            <h1 className="text-5xl font-bold mb-4">
              Add a New Song
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Add your favorite song to our database to help improve future recommendations
            </p>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, duration: 0.4, ease: 'easeOut' }}
            className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-sm mb-8"
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              <Input
                label="Song Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter the song title..."
                required
              />

              <TextArea
                label="Lyrics"
                value={lyrics}
                onChange={(e) => setLyrics(e.target.value)}
                placeholder="Enter the full lyrics here...&#10;&#10;Example:&#10;I've been trying to do it right&#10;I've been living a lonely life&#10;I've been sleeping here instead&#10;I've been sleeping in my bed"
                rows={12}
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

              {success && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm"
                >
                  <div className="flex items-center gap-2">
                    <Music2 size={16} />
                    <span>
                      Successfully added "{success.title}" (ID: {success.id})
                    </span>
                  </div>
                </motion.div>
              )}

              <Button
                type="submit"
                variant="primary"
                className="w-full"
                disabled={loading || !title.trim() || !lyrics.trim()}
              >
                {loading ? (
                  <>
                    <LoadingSpinner />
                    Adding Song...
                  </>
                ) : (
                  <>
                    <Plus size={20} />
                    Add Song to Database
                  </>
                )}
              </Button>
            </form>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
