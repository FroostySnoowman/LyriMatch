import { motion } from 'framer-motion';
import { Music, Brain, Sparkles, Database, Cpu, Network, Target, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Button from '../components/Button';
import AnimatedBackground from '../components/AnimatedBackground';
import UnifiedNavbar from '../components/UnifiedNavbar';

export default function HowItWorks() {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.05, delayChildren: 0.1 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: 'easeOut' },
    },
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      <AnimatedBackground />
      <UnifiedNavbar />

      <div className="relative z-10">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="max-w-7xl mx-auto px-6 pt-32 pb-32"
        >
          <motion.div variants={itemVariants} className="text-center mb-20">
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              How <span className="bg-gradient-to-r from-white via-gray-300 to-gray-500 bg-clip-text text-transparent">LyriMatch</span> Works
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Discover the technology and methodology behind our AI-powered music recommendation engine
            </p>
          </motion.div>

          <motion.div variants={itemVariants} className="mb-32">
            <div className="grid md:grid-cols-3 gap-12 mb-20 relative z-0">
              <motion.div
                className="relative z-10"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <div className="absolute -top-12 left-3 w-20 h-20 bg-white/15 blur-2xl rounded-full pointer-events-none" />
                <div className="absolute -top-8 left-6 w-14 h-14 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl shadow-xl ring-1 ring-black/10 z-20">
                  1
                </div>
                <div className="pt-12 p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm h-full relative z-10">
                  <Music className="mb-4 text-white" size={32} />
                  <h3 className="text-2xl font-semibold mb-3">Input Lyrics</h3>
                  <p className="text-gray-400 leading-relaxed mb-4">
                    You start by entering lyrics that resonate with you. These can be from any song, poem, or even your own creative writing.
                  </p>
                  <div className="bg-black/30 p-4 rounded-lg text-sm text-gray-300 font-mono">
                    "I've been trying to do it right<br />
                    I've been living a lonely life"
                  </div>
                </div>
              </motion.div>

              <motion.div
                className="relative z-10"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <div className="absolute -top-12 left-3 w-20 h-20 bg-white/15 blur-2xl rounded-full pointer-events-none" />
                <div className="absolute -top-8 left-6 w-14 h-14 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl shadow-xl ring-1 ring-black/10 z-20">
                  2
                </div>
                <div className="pt-12 p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm h-full relative z-10">
                  <Brain className="mb-4 text-white" size={32} />
                  <h3 className="text-2xl font-semibold mb-3">AI Processing</h3>
                  <p className="text-gray-400 leading-relaxed mb-4">
                    Our AI analyzes your lyrics using natural language processing to understand themes, emotions, and semantic meaning.
                  </p>
                  <div className="space-y-2">
                    <div className="bg-black/30 p-2 rounded text-xs text-gray-300">Emotion: Loneliness, Hope</div>
                    <div className="bg-black/30 p-2 rounded text-xs text-gray-300">Theme: Self-improvement</div>
                    <div className="bg-black/30 p-2 rounded text-xs text-gray-300">Style: Introspective</div>
                  </div>
                </div>
              </motion.div>

              <motion.div
                className="relative z-10"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
              >
                <div className="absolute -top-12 left-3 w-20 h-20 bg-white/15 blur-2xl rounded-full pointer-events-none" />
                <div className="absolute -top-8 left-6 w-14 h-14 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl shadow-xl ring-1 ring-black/10 z-20">
                  3
                </div>
                <div className="pt-12 p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm h-full relative z-10">
                  <Sparkles className="mb-4 text-white" size={32} />
                  <h3 className="text-2xl font-semibold mb-3">Get Recommendations</h3>
                  <p className="text-gray-400 leading-relaxed mb-4">
                    Receive curated song recommendations with similarity scores and detailed explanations of why each song matches.
                  </p>
                  <div className="bg-black/30 p-3 rounded">
                    <div className="text-sm font-semibold">Ho Hey - The Lumineers</div>
                    <div className="text-xs text-gray-400 mt-1">92% Match</div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="mb-32">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">The Technology</h2>
              <p className="text-xl text-gray-400">Advanced AI and machine learning power our recommendations</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)' }}
              >
                <Cpu className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Natural Language Processing</h3>
                <p className="text-gray-400 leading-relaxed">
                  We use state-of-the-art NLP models to understand the semantic meaning, sentiment, and linguistic patterns in your lyrics. Our models are trained on millions of songs across all genres and languages.
                </p>
              </motion.div>

              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)' }}
              >
                <Database className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Massive Music Database</h3>
                <p className="text-gray-400 leading-relaxed">
                  Our database contains millions of songs with pre-analyzed lyrics, metadata, and emotional signatures. This allows for instant, accurate matching against your input.
                </p>
              </motion.div>

              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)' }}
              >
                <Network className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Semantic Embeddings</h3>
                <p className="text-gray-400 leading-relaxed">
                  Both your input and our song database are converted into high-dimensional vector embeddings that capture meaning and context. We then calculate similarity in this semantic space.
                </p>
              </motion.div>

              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)' }}
              >
                <Target className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Multi-Factor Matching</h3>
                <p className="text-gray-400 leading-relaxed">
                  Our algorithm considers multiple factors: thematic similarity, emotional tone, linguistic style, metaphor usage, and cultural context to provide the most relevant matches.
                </p>
              </motion.div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mb-32 bg-white/5 border border-white/10 rounded-2xl p-12 backdrop-blur-sm"
          >
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">What Makes Us Different</h2>
              <p className="text-xl text-gray-400">Why LyriMatch outperforms traditional music discovery</p>
            </div>

            <div className="grid md:grid-cols-2 gap-12">
              <div>
                <h3 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center text-red-400">✕</div>
                  Traditional Methods
                </h3>
                <ul className="space-y-4 text-gray-400">
                  <li className="flex gap-3">
                    <span className="text-red-400 flex-shrink-0">•</span>
                    <span>Simple keyword matching misses context and meaning</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-red-400 flex-shrink-0">•</span>
                    <span>Genre-based filtering limits discovery opportunities</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-red-400 flex-shrink-0">•</span>
                    <span>Collaborative filtering requires existing listening history</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-red-400 flex-shrink-0">•</span>
                    <span>No explanation for why songs are recommended</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-green-500/20 rounded-full flex items-center justify-center text-green-400">✓</div>
                  LyriMatch Approach
                </h3>
                <ul className="space-y-4 text-gray-400">
                  <li className="flex gap-3">
                    <span className="text-green-400 flex-shrink-0">•</span>
                    <span>Semantic understanding captures true meaning and emotion</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-green-400 flex-shrink-0">•</span>
                    <span>Cross-genre discovery based on thematic similarity</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-green-400 flex-shrink-0">•</span>
                    <span>Works from day one with just a few words</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-green-400 flex-shrink-0">•</span>
                    <span>Detailed explanations for every recommendation</span>
                  </li>
                </ul>
              </div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="text-center bg-gradient-to-r from-white/10 to-white/5 border border-white/10 rounded-2xl p-16 backdrop-blur-sm"
          >
            <Sparkles className="mx-auto mb-6" size={48} />
            <h2 className="text-4xl md:text-5xl font-bold mb-6">Ready to Try It?</h2>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              Experience the power of AI-driven lyric analysis and discover your next favorite song
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="primary" size="lg" onClick={() => navigate('/signup')}>
                Start Matching
                <ArrowRight size={20} />
              </Button>
              <Button variant="secondary" size="lg" onClick={() => navigate('/examples')}>
                See Examples
              </Button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}