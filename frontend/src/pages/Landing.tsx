import { motion, type Variants } from 'framer-motion';
import { Music, Sparkles, Zap, ArrowRight, Brain, TrendingUp, Users, CheckCircle, Quote } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import AnimatedBackground from '../components/AnimatedBackground';
import UnifiedNavbar from '../components/UnifiedNavbar';
import Button from '../components/Button';
import Logo from '../components/Logo';

export default function Landing() {
  const navigate = useNavigate();

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.05, delayChildren: 0.1 },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: [0.2, 0.65, 0.3, 0.9] },
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
          <motion.div variants={itemVariants} className="text-center mb-16">
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-full mb-8"
              whileHover={{ scale: 1.05 }}
            >
              <Sparkles size={16} className="text-gray-400" />
              <span className="text-sm text-gray-400">AI-Powered Music Discovery</span>
            </motion.div>
            <h1 className="text-6xl md:text-8xl font-bold mb-6 tracking-tight">
              Find Songs That
              <br />
              <span className="bg-gradient-to-r from-white via-gray-300 to-gray-500 bg-clip-text text-transparent">
                Match Your Lyrics
              </span>
            </h1>
            <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
              Enter any lyrics and let our AI discover songs with similar themes, emotions, and style.
              Your next favorite song is just a verse away.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-20">
              <Button variant="primary" size="lg" onClick={() => navigate('/signup')}>
                Start Matching
                <ArrowRight size={20} />
              </Button>
              <Button variant="secondary" size="lg" onClick={() => navigate('/examples')}>
                See Examples
              </Button>
            </div>
            <motion.div variants={itemVariants} className="flex flex-wrap justify-center gap-8 text-center">
              <div className="flex flex-col items-center">
                <div className="text-3xl font-bold mb-1">10K+</div>
                <div className="text-sm text-gray-400">Searches Made</div>
              </div>
              <div className="flex flex-col items-center">
                <div className="text-3xl font-bold mb-1">500+</div>
                <div className="text-sm text-gray-400">Active Users</div>
              </div>
              <div className="flex flex-col items-center">
                <div className="text-3xl font-bold mb-1">98%</div>
                <div className="text-sm text-gray-400">Accuracy Rate</div>
              </div>
            </motion.div>
          </motion.div>

          <motion.div variants={itemVariants} className="mt-32 grid md:grid-cols-3 gap-8 mb-32">
            <motion.div
              className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
              whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center mb-4">
                <Brain className="text-white" size={24} />
              </div>
              <h3 className="text-xl font-semibold mb-2">Deep Learning</h3>
              <p className="text-gray-400">
                Our AI understands context, emotion, and meaning behind your favorite lyrics using advanced NLP.
              </p>
            </motion.div>
            <motion.div
              className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
              whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center mb-4">
                <Zap className="text-white" size={24} />
              </div>
              <h3 className="text-xl font-semibold mb-2">Instant Results</h3>
              <p className="text-gray-400">
                Get personalized song recommendations in seconds with detailed explanations and similarity scores.
              </p>
            </motion.div>
            <motion.div
              className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
              whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center mb-4">
                <Music className="text-white" size={24} />
              </div>
              <h3 className="text-xl font-semibold mb-2">Vast Music Library</h3>
              <p className="text-gray-400">Access recommendations from millions of songs across all genres and decades.</p>
            </motion.div>
          </motion.div>

          <motion.div variants={itemVariants} className="mt-32 mb-32">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">How It Works</h2>
              <p className="text-xl text-gray-400">Three simple steps to discover your next favorite song</p>
            </div>
            <div className="grid md:grid-cols-3 gap-12 relative z-0">
              <motion.div
                className="relative z-10"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-50px' }}
                transition={{ delay: 0.1, duration: 0.4, ease: [0.2, 0.65, 0.3, 0.9] }}
              >
                <div className="absolute -top-12 left-3 w-20 h-20 bg-white/15 blur-2xl rounded-full pointer-events-none" />
                <div className="absolute -top-8 left-6 w-14 h-14 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl shadow-xl ring-1 ring-black/10 z-20">
                  1
                </div>
                <div className="pt-12 p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm h-full relative z-10">
                  <h3 className="text-2xl font-semibold mb-3">Enter Lyrics</h3>
                  <p className="text-gray-400 leading-relaxed">
                    Paste in any lyrics that resonate with you. It could be from a favorite song or just words that capture how you feel.
                  </p>
                </div>
              </motion.div>
              <motion.div
                className="relative z-10"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2, duration: 0.4, ease: [0.2, 0.65, 0.3, 0.9] }}
              >
                <div className="absolute -top-12 left-3 w-20 h-20 bg-white/15 blur-2xl rounded-full pointer-events-none" />
                <div className="absolute -top-8 left-6 w-14 h-14 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl shadow-xl ring-1 ring-black/10 z-20">
                  2
                </div>
                <div className="pt-12 p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm h-full relative z-10">
                  <h3 className="text-2xl font-semibold mb-3">AI Analysis</h3>
                  <p className="text-gray-400 leading-relaxed">
                    Our AI analyzes the themes, emotions, metaphors, and linguistic patterns in your lyrics to understand the deeper meaning.
                  </p>
                </div>
              </motion.div>
              <motion.div
                className="relative z-10"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3, duration: 0.4, ease: [0.2, 0.65, 0.3, 0.9] }}
              >
                <div className="absolute -top-12 left-3 w-20 h-20 bg-white/15 blur-2xl rounded-full pointer-events-none" />
                <div className="absolute -top-8 left-6 w-14 h-14 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl shadow-xl ring-1 ring-black/10 z-20">
                  3
                </div>
                <div className="pt-12 p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm h-full relative z-10">
                  <h3 className="text-2xl font-semibold mb-3">Get Matches</h3>
                  <p className="text-gray-400 leading-relaxed">
                    Receive personalized recommendations with similarity scores and explanations for why each song matches your input.
                  </p>
                </div>
              </motion.div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mt-32 mb-32 bg-white/5 border border-white/10 rounded-2xl p-12 backdrop-blur-sm"
          >
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">Why Choose LyriMatch?</h2>
              <p className="text-xl text-gray-400">The most advanced lyric-based music discovery platform</p>
            </div>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="flex gap-4">
                <CheckCircle className="text-white flex-shrink-0 mt-1" size={24} />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Semantic Understanding</h3>
                  <p className="text-gray-400">Goes beyond keyword matching to understand the actual meaning and emotion in lyrics</p>
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle className="text-white flex-shrink-0 mt-1" size={24} />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Multi-Genre Support</h3>
                  <p className="text-gray-400">Discovers songs across all genres, from pop to classical, rap to indie</p>
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle className="text-white flex-shrink-0 mt-1" size={24} />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Explainable AI</h3>
                  <p className="text-gray-400">Every recommendation comes with a clear explanation of why it matches</p>
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle className="text-white flex-shrink-0 mt-1" size={24} />
                <div>
                  <h3 className="text-xl font-semibold mb-2">Privacy First</h3>
                  <p className="text-gray-400">Your searches are private and secure, stored only for your personal history</p>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="mt-32 text-center">
            <div className="max-w-3xl mx-auto bg-white/5 border border-white/10 rounded-2xl p-12 backdrop-blur-sm">
              <Quote className="mx-auto mb-6 text-gray-400" size={48} />
              <p className="text-2xl md:text-3xl font-light mb-8 leading-relaxed">
                "LyriMatch helped me discover songs I never would have found on my own. The AI really understands the emotional depth of lyrics."
              </p>
              <div className="flex items-center justify-center gap-3">
                <div className="w-12 h-12 bg-white/10 rounded-full flex items-center justify-center">
                  <Users size={24} />
                </div>
                <div className="text-left">
                  <div className="font-semibold">Sarah Mitchell</div>
                  <div className="text-sm text-gray-400">Music Enthusiast</div>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mt-32 text-center bg-gradient-to-r from-white/10 to-white/5 border border-white/10 rounded-2xl p-16 backdrop-blur-sm"
          >
            <TrendingUp className="mx-auto mb-6" size={48} />
            <h2 className="text-4xl md:text-5xl font-bold mb-6">Ready to Discover?</h2>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              Join thousands of music lovers using LyriMatch to find their next favorite songs
            </p>
            <Button variant="primary" size="lg" onClick={() => navigate('/signup')}>
              Start Matching Now
              <ArrowRight size={20} />
            </Button>
          </motion.div>
        </motion.div>

        <footer className="relative border-t border-white/10 mt-32">
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="grid md:grid-cols-4 gap-8 mb-8">
              <div>
                <Logo size="sm" />
                <p className="text-gray-400 text-sm mt-4">AI-powered music discovery through lyrics</p>
              </div>
              <div>
                <h4 className="font-semibold mb-4">Product</h4>
                <div className="space-y-2 text-sm text-gray-400">
                  <button onClick={() => navigate('/how-it-works')} className="block hover:text-white transition-colors">
                    How It Works
                  </button>
                  <button onClick={() => navigate('/examples')} className="block hover:text-white transition-colors">
                    Examples
                  </button>
                  {/* <button onClick={() => navigate('/pricing')} className="block hover:text-white transition-colors">
                    Pricing
                  </button> */}
                </div>
              </div>
              <div>
                <h4 className="font-semibold mb-4">Company</h4>
                <div className="space-y-2 text-sm text-gray-400">
                  <button onClick={() => navigate('/about')} className="block hover:text-white transition-colors">
                    About
                  </button>
                  <div className="hover:text-white transition-colors cursor-pointer">Contact</div>
                </div>
              </div>
              <div>
                <h4 className="font-semibold mb-4">Legal</h4>
                <div className="space-y-2 text-sm text-gray-400">
                  <div className="hover:text-white transition-colors cursor-pointer">Privacy</div>
                  <div className="hover:text-white transition-colors cursor-pointer">Terms</div>
                </div>
              </div>
            </div>
            <div className="border-t border-white/10 pt-8 text-center text-sm text-gray-400">
              © 2025 LyriMatch. All rights reserved.
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}