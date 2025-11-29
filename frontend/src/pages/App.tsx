import { useState, FormEvent, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Music2, Sparkles, Library, Search, ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import UnifiedNavbar from '../components/UnifiedNavbar';
import TextArea from '../components/TextArea';
import Input from '../components/Input';
import Button from '../components/Button';
import LoadingSpinner from '../components/LoadingSpinner';
import { analyzeLyrics, browseSongs, SongRecommendation, BrowseSongsResponse } from '../lib/api';

type Tab = 'search' | 'browse';

export default function App() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>('search');
  
  // Search tab state
  const [lyrics, setLyrics] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<{
    recommendations: SongRecommendation[];
    analysis: string;
  } | null>(null);
  const [error, setError] = useState('');

  // Browse tab state
  const [browseData, setBrowseData] = useState<BrowseSongsResponse | null>(null);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [browseError, setBrowseError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [perPage, setPerPage] = useState(20);
  const [sortBy, setSortBy] = useState<'id' | 'name' | 'album_name'>('id');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [debouncedQuery, setDebouncedQuery] = useState('');

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
      setCurrentPage(1); // Reset to first page on new search
    }, 500);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Fetch browse data when parameters change
  useEffect(() => {
    if (activeTab === 'browse') {
      fetchBrowseData();
    }
  }, [activeTab, currentPage, perPage, sortBy, sortOrder, debouncedQuery]);

  const fetchBrowseData = async () => {
    setBrowseLoading(true);
    setBrowseError('');

    try {
      const data = await browseSongs(currentPage, perPage, debouncedQuery, sortBy, sortOrder);
      setBrowseData(data);
    } catch (err: any) {
      setBrowseError(err.message || 'Failed to load songs');
    } finally {
      setBrowseLoading(false);
    }
  };

  const handleSearchSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!lyrics.trim()) return;

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const data = await analyzeLyrics(lyrics);
      setResults(data);
    } catch (err) {
      setError('Failed to analyze lyrics. Please make sure your backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const toggleSort = (field: 'id' | 'name' | 'album_name') => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
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
          className="max-w-6xl mx-auto"
        >
          <div className="text-center mb-12">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-full mb-6"
            >
              <Sparkles size={16} className="text-gray-400" />
              <span className="text-sm text-gray-400">AI-Powered Music Discovery</span>
            </motion.div>

            <h1 className="text-5xl font-bold mb-4">
              Discover Your Next Favorite Song
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Search by lyrics or browse our complete database of embedded songs
            </p>
          </div>

          {/* Tab Navigation */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.3, ease: 'easeOut' }}
            className="flex justify-center mb-8"
          >
            <div className="inline-flex bg-white/5 border border-white/10 rounded-full p-1">
              <button
                onClick={() => setActiveTab('search')}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-full transition-all ${
                  activeTab === 'search'
                    ? 'bg-white text-black font-medium'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Search size={18} />
                Search by Lyrics
              </button>
              <button
                onClick={() => setActiveTab('browse')}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-full transition-all ${
                  activeTab === 'browse'
                    ? 'bg-white text-black font-medium'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Library size={18} />
                Browse All Songs
              </button>
            </div>
          </motion.div>

          {/* Search Tab Content */}
          {activeTab === 'search' && (
            <motion.div
              key="search-tab"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
            >
              <motion.div
                className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-sm mb-8"
              >
                <form onSubmit={handleSearchSubmit} className="space-y-6">
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
          )}

          {/* Browse Tab Content */}
          {activeTab === 'browse' && (
            <motion.div
              key="browse-tab"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
            >
              {/* Search and Filters */}
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm mb-6">
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="flex-1">
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search songs by name or artist..."
                      className="w-full"
                    />
                  </div>
                  <div className="flex gap-2">
                    <select
                      value={perPage}
                      onChange={(e) => {
                        setPerPage(Number(e.target.value));
                        setCurrentPage(1);
                      }}
                      className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-white/50"
                    >
                      <option value="10">10 per page</option>
                      <option value="20">20 per page</option>
                      <option value="50">50 per page</option>
                      <option value="100">100 per page</option>
                    </select>
                  </div>
                </div>

                {browseData && (
                  <div className="mt-4 text-sm text-gray-400">
                    Showing {browseData.songs.length} of {browseData.total} songs
                  </div>
                )}
              </div>

              {/* Songs Table */}
              {browseLoading ? (
                <div className="flex flex-col items-center justify-center py-20">
                  <LoadingSpinner />
                  <p className="mt-6 text-gray-400">Loading songs...</p>
                </div>
              ) : browseError ? (
                <div className="p-8 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-center">
                  {browseError}
                </div>
              ) : browseData && browseData.songs.length > 0 ? (
                <>
                  <div className="bg-white/5 border border-white/10 rounded-2xl overflow-hidden backdrop-blur-sm">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-white/5 border-b border-white/10">
                          <tr>
                            <th
                              className="px-6 py-4 text-left text-sm font-semibold cursor-pointer hover:bg-white/5 transition-colors"
                              onClick={() => toggleSort('id')}
                            >
                              <div className="flex items-center gap-2">
                                ID
                                <ArrowUpDown size={14} className={sortBy === 'id' ? 'text-white' : 'text-gray-500'} />
                              </div>
                            </th>
                            <th
                              className="px-6 py-4 text-left text-sm font-semibold cursor-pointer hover:bg-white/5 transition-colors"
                              onClick={() => toggleSort('name')}
                            >
                              <div className="flex items-center gap-2">
                                Song Name
                                <ArrowUpDown size={14} className={sortBy === 'name' ? 'text-white' : 'text-gray-500'} />
                              </div>
                            </th>
                            <th
                              className="px-6 py-4 text-left text-sm font-semibold cursor-pointer hover:bg-white/5 transition-colors"
                              onClick={() => toggleSort('album_name')}
                            >
                              <div className="flex items-center gap-2">
                                Album/Artist
                                <ArrowUpDown size={14} className={sortBy === 'album_name' ? 'text-white' : 'text-gray-500'} />
                              </div>
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/10">
                          {browseData.songs.map((song, index) => (
                            <motion.tr
                              key={song.id}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: index * 0.05 }}
                              className="hover:bg-white/5 transition-colors"
                            >
                              <td className="px-6 py-4 text-sm text-gray-400">{song.id}</td>
                              <td className="px-6 py-4 text-sm font-medium">{song.name}</td>
                              <td className="px-6 py-4 text-sm text-gray-400">{song.album_name || '—'}</td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Pagination */}
                  {browseData.total_pages > 1 && (
                    <div className="mt-6 flex items-center justify-between">
                      <div className="text-sm text-gray-400">
                        Page {browseData.page} of {browseData.total_pages}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="secondary"
                          onClick={() => setCurrentPage(currentPage - 1)}
                          disabled={!browseData.has_prev}
                        >
                          <ChevronLeft size={20} />
                          Previous
                        </Button>
                        <Button
                          variant="secondary"
                          onClick={() => setCurrentPage(currentPage + 1)}
                          disabled={!browseData.has_next}
                        >
                          Next
                          <ChevronRight size={20} />
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="p-8 bg-white/5 border border-white/10 rounded-2xl text-center text-gray-400">
                  {searchQuery ? 'No songs found matching your search.' : 'No songs in the database yet.'}
                </div>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
}