import { motion } from 'framer-motion';
import { Target, Heart, Zap, Users } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import AnimatedBackground from '../components/AnimatedBackground';
import UnifiedNavbar from '../components/UnifiedNavbar';
import Button from '../components/Button';

export default function About() {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
        delayChildren: 0.1,
      },
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
              About <span className="bg-gradient-to-r from-white via-gray-300 to-gray-500 bg-clip-text text-transparent">LyriMatch</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              We're revolutionizing music discovery by helping people find songs through the power of lyrics and AI
            </p>
          </motion.div>

          <motion.div variants={itemVariants} className="mb-32">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-12 backdrop-blur-sm">
              <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
              <p className="text-xl text-gray-300 leading-relaxed mb-6">
                LyriMatch was created to solve a simple problem: sometimes you remember a few lyrics or want to find songs with a specific emotional tone, but you don't know where to start. Traditional music discovery relies on genres, artists, or popularity, but we believe lyrics are the soul of a song.
              </p>
              <p className="text-xl text-gray-300 leading-relaxed">
                Our mission is to make music discovery more intuitive, personal, and meaningful by analyzing the actual words that resonate with you.
              </p>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="mb-32">
            <h2 className="text-4xl font-bold mb-12 text-center">Our Values</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
              >
                <Target className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Accuracy First</h3>
                <p className="text-gray-400 leading-relaxed">
                  We're committed to providing the most accurate and relevant song recommendations by continuously improving our AI models and algorithms.
                </p>
              </motion.div>

              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
              >
                <Heart className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">User-Centric</h3>
                <p className="text-gray-400 leading-relaxed">
                  Every feature we build is designed with you in mind. Your privacy, preferences, and music discovery journey are our top priorities.
                </p>
              </motion.div>

              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
              >
                <Zap className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Innovation</h3>
                <p className="text-gray-400 leading-relaxed">
                  We leverage cutting-edge AI and natural language processing to push the boundaries of what's possible in music discovery.
                </p>
              </motion.div>

              <motion.div
                className="p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-sm"
                whileHover={{ scale: 1.02, borderColor: 'rgba(255,255,255,0.3)', transition: { duration: 0.2 } }}
              >
                <Users className="mb-4 text-white" size={32} />
                <h3 className="text-2xl font-semibold mb-3">Community</h3>
                <p className="text-gray-400 leading-relaxed">
                  We're building a community of music lovers who appreciate the art of songwriting and the power of meaningful lyrics.
                </p>
              </motion.div>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="mb-32">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-12 backdrop-blur-sm">
              <h2 className="text-3xl font-bold mb-6">The Story</h2>
              <p className="text-lg text-gray-300 leading-relaxed mb-6">
                LyriMatch started as a class project but quickly grew into something more. We realized that while streaming services are great at recommending music based on listening history, they often miss the emotional and thematic connections that make a song truly resonate with someone.
              </p>
              <p className="text-lg text-gray-300 leading-relaxed mb-6">
                By focusing on lyrics, we're able to capture the essence of what makes music meaningful, whether it's a shared experience, a powerful metaphor, or just the perfect words at the right time.
              </p>
              <p className="text-lg text-gray-300 leading-relaxed">
                Today, we're proud to help thousands of users discover music that speaks to them on a deeper level.
              </p>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="text-center bg-gradient-to-r from-white/10 to-white/5 border border-white/10 rounded-2xl p-16 backdrop-blur-sm"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Join Our Journey
            </h2>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              Be part of the future of music discovery. Start finding songs that truly resonate with you.
            </p>
            <Button
              variant="primary"
              size="lg"
              onClick={() => navigate('/signup')}
            >
              Get Started
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
