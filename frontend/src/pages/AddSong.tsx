import { useState, FormEvent } from 'react';
import { motion } from 'framer-motion';
import { Plus, Music2, Sparkles, ListMusic, CheckCircle, XCircle, Loader2, MinusCircle } from 'lucide-react';
import UnifiedNavbar from '../components/UnifiedNavbar';
import Input from '../components/Input';
import TextArea from '../components/TextArea';
import Button from '../components/Button';
import LoadingSpinner from '../components/LoadingSpinner';
import { addSong, getPlaylistTracks, addSongFromSearch } from '../lib/api';

type Mode = 'song' | 'playlist';

type SongProgress = {
  title: string;
  artist: string;
  status: 'pending' | 'processing' | 'success' | 'failed' | 'skipped';
  error?: string;
  id?: number;
};

export default function AddSong() {
  const [mode, setMode] = useState<Mode>('song');
  const [title, setTitle] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [playlistUrl, setPlaylistUrl] = useState('');
  const [songLimit, setSongLimit] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<{ id?: number; title?: string; count?: number } | null>(null);
  const [error, setError] = useState('');
  
  // Playlist progress tracking
  const [playlistProgress, setPlaylistProgress] = useState<SongProgress[]>([]);
  const [currentSongIndex, setCurrentSongIndex] = useState(0);
  const [isProcessingPlaylist, setIsProcessingPlaylist] = useState(false);

  const handleSongSubmit = async (e: FormEvent) => {
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

  const handlePlaylistSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!playlistUrl.trim()) return;

    setLoading(true);
    setError('');
    setSuccess(null);
    setPlaylistProgress([]);
    setCurrentSongIndex(0);
    setIsProcessingPlaylist(false);

    try {
      const limit = songLimit ? parseInt(songLimit) : undefined;
      
      // First, fetch the playlist tracks
      const tracksData = await getPlaylistTracks(playlistUrl, limit);
      
      // Initialize progress tracking
      const initialProgress: SongProgress[] = tracksData.tracks.map(track => ({
        title: track.title,
        artist: track.artists[0] || 'Unknown Artist',
        status: 'pending',
      }));
      
      setPlaylistProgress(initialProgress);
      setIsProcessingPlaylist(true);
      setLoading(false);
      
      // Process each song sequentially
      let successCount = 0;
      
      for (let i = 0; i < initialProgress.length; i++) {
        setCurrentSongIndex(i);
        
        // Update status to processing
        setPlaylistProgress(prev => {
          const updated = [...prev];
          updated[i] = { ...updated[i], status: 'processing' };
          return updated;
        });
        
        try {
          const result = await addSongFromSearch(
            initialProgress[i].title,
            initialProgress[i].artist
          );
          
          // Update status to success
          setPlaylistProgress(prev => {
            const updated = [...prev];
            updated[i] = {
              ...updated[i],
              status: 'success',
              id: result.id,
            };
            return updated;
          });
          
          successCount++;
        } catch (err: any) {
          // Check if it's a duplicate (409 status)
          const isDuplicate = err.message && err.message.includes('already exists');
          
          // Update status to skipped if duplicate, failed otherwise
          setPlaylistProgress(prev => {
            const updated = [...prev];
            updated[i] = {
              ...updated[i],
              status: isDuplicate ? 'skipped' : 'failed',
              error: err.message || 'Failed to add song',
            };
            return updated;
          });
        }
      }
      
      // Show final success message
      setSuccess({ count: successCount });
      setIsProcessingPlaylist(false);
      setPlaylistUrl('');
      setSongLimit('');
      
    } catch (err: any) {
      setError(err.message || 'Failed to fetch playlist. Please check the URL and make sure your backend is running.');
      setLoading(false);
      setIsProcessingPlaylist(false);
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
              Add New Music
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Add individual songs or entire playlists to help improve future recommendations
            </p>
          </div>

          {/* Mode Toggle */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.3, ease: 'easeOut' }}
            className="flex justify-center mb-8"
          >
            <div className="inline-flex bg-white/5 border border-white/10 rounded-full p-1">
              <button
                onClick={() => setMode('song')}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-full transition-all ${
                  mode === 'song'
                    ? 'bg-white text-black font-medium'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Music2 size={18} />
                Single Song
              </button>
              <button
                onClick={() => setMode('playlist')}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-full transition-all ${
                  mode === 'playlist'
                    ? 'bg-white text-black font-medium'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <ListMusic size={18} />
                Playlist
              </button>
            </div>
          </motion.div>

          <motion.div
            key={mode}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-sm mb-8"
          >
            {mode === 'song' ? (
              <form onSubmit={handleSongSubmit} className="space-y-6">
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

                {success && success.title && (
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
            ) : (
              <form onSubmit={handlePlaylistSubmit} className="space-y-6">
                <Input
                  label="Spotify Playlist URL"
                  value={playlistUrl}
                  onChange={(e) => setPlaylistUrl(e.target.value)}
                  placeholder="https://open.spotify.com/playlist/..."
                  required
                  disabled={isProcessingPlaylist}
                />

                <Input
                  label="Song Limit (Optional)"
                  type="number"
                  value={songLimit}
                  onChange={(e) => setSongLimit(e.target.value)}
                  placeholder="Leave empty to add all songs"
                  min="1"
                  disabled={isProcessingPlaylist}
                />

                {!isProcessingPlaylist && (
                  <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg text-blue-400 text-sm">
                    <p className="mb-2 font-medium">How it works:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>Paste a Spotify playlist URL</li>
                      <li>Songs will be fetched from Spotify</li>
                      <li>Lyrics will be retrieved from Genius</li>
                      <li>Each song will be embedded and added to the database</li>
                    </ul>
                  </div>
                )}

                {/* Progress Display */}
                {isProcessingPlaylist && playlistProgress.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                  >
                    {/* Progress Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">
                          Processing songs...
                        </span>
                        <span className="text-white font-medium">
                          {playlistProgress.filter(s => s.status === 'success').length} added
                          {playlistProgress.filter(s => s.status === 'skipped').length > 0 && (
                            <span className="text-yellow-400 ml-2">
                              · {playlistProgress.filter(s => s.status === 'skipped').length} skipped
                            </span>
                          )}
                          {playlistProgress.filter(s => s.status === 'failed').length > 0 && (
                            <span className="text-red-400 ml-2">
                              · {playlistProgress.filter(s => s.status === 'failed').length} failed
                            </span>
                          )}
                        </span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                          initial={{ width: 0 }}
                          animate={{
                            width: `${((playlistProgress.filter(s => s.status === 'success' || s.status === 'failed' || s.status === 'skipped').length) / playlistProgress.length) * 100}%`
                          }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>

                    {/* Song List */}
                    <div className="max-h-96 overflow-y-auto space-y-2 p-4 bg-black/20 rounded-lg border border-white/5">
                      {playlistProgress.map((song, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className={`flex items-center gap-3 p-3 rounded-lg ${
                            song.status === 'processing'
                              ? 'bg-blue-500/10 border border-blue-500/20'
                              : song.status === 'success'
                              ? 'bg-green-500/5 border border-green-500/10'
                              : song.status === 'skipped'
                              ? 'bg-yellow-500/5 border border-yellow-500/10'
                              : song.status === 'failed'
                              ? 'bg-red-500/5 border border-red-500/10'
                              : 'bg-white/5 border border-white/10'
                          }`}
                        >
                          <div className="flex-shrink-0">
                            {song.status === 'processing' && (
                              <Loader2 size={16} className="text-blue-400 animate-spin" />
                            )}
                            {song.status === 'success' && (
                              <CheckCircle size={16} className="text-green-400" />
                            )}
                            {song.status === 'skipped' && (
                              <MinusCircle size={16} className="text-yellow-400" />
                            )}
                            {song.status === 'failed' && (
                              <XCircle size={16} className="text-red-400" />
                            )}
                            {song.status === 'pending' && (
                              <div className="w-4 h-4 rounded-full bg-white/20" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-white truncate">
                              {song.title}
                            </p>
                            <p className="text-xs text-gray-400 truncate">
                              {song.artist}
                            </p>
                            {song.error && (
                              <p className={`text-xs mt-1 ${song.status === 'skipped' ? 'text-yellow-400' : 'text-red-400'}`}>
                                {song.error}
                              </p>
                            )}
                          </div>
                          {song.id && (
                            <span className="text-xs text-gray-500">
                              ID: {song.id}
                            </span>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {error && !isProcessingPlaylist && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm"
                  >
                    {error}
                  </motion.div>
                )}

                {success && success.count !== undefined && !isProcessingPlaylist && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm"
                  >
                    <div className="flex items-center gap-2">
                      <ListMusic size={16} />
                      <span>
                        Successfully added {success.count} song{success.count !== 1 ? 's' : ''} from the playlist!
                      </span>
                    </div>
                  </motion.div>
                )}

                <Button
                  type="submit"
                  variant="primary"
                  className="w-full"
                  disabled={loading || isProcessingPlaylist || !playlistUrl.trim()}
                >
                  {loading ? (
                    <>
                      <LoadingSpinner />
                      Fetching Playlist...
                    </>
                  ) : isProcessingPlaylist ? (
                    <>
                      <LoadingSpinner />
                      Processing {currentSongIndex + 1} / {playlistProgress.length}
                    </>
                  ) : (
                    <>
                      <Plus size={20} />
                      Add Playlist to Database
                    </>
                  )}
                </Button>
              </form>
            )}
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
