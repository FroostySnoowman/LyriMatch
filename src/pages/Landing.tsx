import { motion } from 'framer-motion';
import { Music, Sparkles, Zap, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Logo from '../components/Logo';
import Button from '../components/Button';
import { useAuth } from '../contexts/AuthContext';
import { useEffect } from 'react';

export default function Landing() {
  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    if (user) {
      navigate('/app');
    }
  }, [user, navigate]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 },
    },
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-black" />

      <div className="absolute inset-0 overflow-hidden">
        {[...Array(50)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      <div className="relative z-10">
        <motion.nav
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          className="px-6 py-6 flex items-center justify-between max-w-7xl mx-auto"
        >
          <Logo animated />
          <div className="flex gap-3">
            <Button variant="ghost" onClick={() => navigate('/signin')}>
              Sign In
            </Button>
            <Button variant="primary" onClick={() => navigate('/signup')}>
              Get Started
            </Button>
          </div>
        </motion.nav>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="max-w-6xl mx-auto px-6 pt-20 pb-32"
        >
          <motion.div variants={itemVariants} className="text-center mb-12">
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

            <div className="flex gap-4 justify-center">
              <Button
                variant="primary"
                size="lg"
                onClick={() => navigate('/signup')}
              >
                Start Matching
                <ArrowRight size={20} />
              </Button>
              <Button
                variant="secondary"
                size="lg"
                onClick={() => navigate('/signin')}
              >
                Sign In
              </Button>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mt-32 grid md:grid-cols-3 gap-8"
          >
            <motion.div
              className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
              whileHover={{ scale: 1.05, borderColor: 'rgba(255,255,255,0.3)' }}
            >
              <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center mb-4">
                <Music className="text-white" size={24} />
              </div>
              <h3 className="text-xl font-semibold mb-2">Lyric Analysis</h3>
              <p className="text-gray-400">
                Our AI understands context, emotion, and meaning behind your favorite lyrics.
              </p>
            </motion.div>

            <motion.div
              className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
              whileHover={{ scale: 1.05, borderColor: 'rgba(255,255,255,0.3)' }}
            >
              <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center mb-4">
                <Zap className="text-white" size={24} />
              </div>
              <h3 className="text-xl font-semibold mb-2">Instant Results</h3>
              <p className="text-gray-400">
                Get personalized song recommendations in seconds with detailed explanations.
              </p>
            </motion.div>

            <motion.div
              className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
              whileHover={{ scale: 1.05, borderColor: 'rgba(255,255,255,0.3)' }}
            >
              <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center mb-4">
                <Sparkles className="text-white" size={24} />
              </div>
              <h3 className="text-xl font-semibold mb-2">Discover New Music</h3>
              <p className="text-gray-400">
                Expand your musical horizons with AI-curated suggestions tailored to your taste.
              </p>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
