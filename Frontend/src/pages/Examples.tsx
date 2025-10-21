import { motion } from 'framer-motion';
import { Music2, Heart, CloudRain, Zap, Smile, ArrowRight, Play } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Button from '../components/Button';
import AnimatedBackground from '../components/AnimatedBackground';
import UnifiedNavbar from '../components/UnifiedNavbar';

interface ExampleSearch {
  id: number;
  title: string;
  lyrics: string;
  icon: React.ReactNode;
  theme: string;
  recommendations: {
    title: string;
    artist: string;
    match: number;
    reason: string;
  }[];
}

export default function Examples() {
  const navigate = useNavigate();

  const examples: ExampleSearch[] = [
    {
      id: 1,
      title: 'Loneliness & Self-Discovery',
      lyrics: "I've been trying to do it right\nI've been living a lonely life\nI've been sleeping here instead\nI've been sleeping in my bed",
      icon: <Heart size={24} />,
      theme: 'Introspective, Folk',
      recommendations: [
        {
          title: 'Ho Hey',
          artist: 'The Lumineers',
          match: 94,
          reason: 'Both songs explore themes of loneliness and the journey toward human connection with a folk-rock sensibility.',
        },
        {
          title: 'Skinny Love',
          artist: 'Bon Iver',
          match: 91,
          reason: 'Similar introspective tone and raw emotional vulnerability about isolation and yearning for connection.',
        },
        {
          title: 'The Night We Met',
          artist: 'Lord Huron',
          match: 88,
          reason: 'Shares the melancholic, reflective mood and themes of longing and personal struggle.',
        },
      ],
    },
    {
      id: 2,
      title: 'Heartbreak & Moving On',
      lyrics: "Someone like you makes it hard to live without somebody else\nSomeone like you makes it easy to give, never think about myself",
      icon: <CloudRain size={24} />,
      theme: 'Emotional, Pop',
      recommendations: [
        {
          title: 'Someone Like You',
          artist: 'Adele',
          match: 96,
          reason: 'Direct thematic match exploring acceptance of lost love and the difficulty of moving forward.',
        },
        {
          title: 'When We Were Young',
          artist: 'Adele',
          match: 89,
          reason: 'Similar nostalgic reflection on past relationships and the person you used to be with someone.',
        },
        {
          title: 'All Too Well',
          artist: 'Taylor Swift',
          match: 87,
          reason: 'Shares the emotional depth and detailed reminiscence of a past relationship.',
        },
      ],
    },
    {
      id: 3,
      title: 'Empowerment & Confidence',
      lyrics: "I got this feeling inside my bones\nIt goes electric, wavy when I turn it on\nAll through my city, all through my home\nWe're flying up, no ceiling, when we're in our zone",
      icon: <Zap size={24} />,
      theme: 'Upbeat, Pop',
      recommendations: [
        {
          title: "Can't Stop the Feeling!",
          artist: 'Justin Timberlake',
          match: 95,
          reason: 'Perfect match with energetic, feel-good vibes and themes of unstoppable positive energy.',
        },
        {
          title: 'Confident',
          artist: 'Demi Lovato',
          match: 90,
          reason: 'Shares the powerful, self-assured energy and themes of personal empowerment.',
        },
        {
          title: 'Good as Hell',
          artist: 'Lizzo',
          match: 88,
          reason: 'Similar uplifting message and infectious energy celebrating self-confidence.',
        },
      ],
    },
    {
      id: 4,
      title: 'Nostalgia & Youth',
      lyrics: "We're just young dumb and broke\nBut we still got love to give\nWhile we're young, dumb, young, young dumb and broke",
      icon: <Smile size={24} />,
      theme: 'Youthful, R&B',
      recommendations: [
        {
          title: 'Young Dumb & Broke',
          artist: 'Khalid',
          match: 98,
          reason: 'Exact thematic match celebrating youth, financial struggles, and the richness of young love.',
        },
        {
          title: 'Location',
          artist: 'Khalid',
          match: 86,
          reason: 'Similar laid-back style and themes of young romance and carefree youth.',
        },
        {
          title: 'Best Day of My Life',
          artist: 'American Authors',
          match: 84,
          reason: 'Shares the optimistic, carefree attitude and celebration of living in the moment.',
        },
      ],
    },
  ];

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
              Example <span className="bg-gradient-to-r from-white via-gray-300 to-gray-500 bg-clip-text text-transparent">Searches</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              See how LyriMatch analyzes different types of lyrics and discovers perfect song matches
            </p>
          </motion.div>

          <div className="space-y-16">
            {examples.map((example, index) => (
              <motion.div
                key={example.id}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="bg-white/5 border border-white/10 rounded-2xl overflow-hidden backdrop-blur-sm"
              >
                <div className="p-8 border-b border-white/10">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      {example.icon}
                    </div>
                    <div className="flex-1">
                      <h2 className="text-3xl font-bold mb-2">{example.title}</h2>
                      <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/10 rounded-full text-sm">
                        <Music2 size={14} />
                        {example.theme}
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/30 p-6 rounded-lg">
                    <div className="flex items-center gap-2 mb-3 text-sm text-gray-400">
                      <Play size={14} />
                      <span>Input Lyrics</span>
                    </div>
                    <p className="text-gray-300 leading-relaxed font-light whitespace-pre-line">
                      {example.lyrics}
                    </p>
                  </div>
                </div>

                <div className="p-8">
                  <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
                    <Zap size={20} className="text-gray-400" />
                    AI-Generated Recommendations
                  </h3>

                  <div className="space-y-4">
                    {example.recommendations.map((rec, recIndex) => (
                      <motion.div
                        key={recIndex}
                        initial={{ opacity: 0, x: -20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: recIndex * 0.1 }}
                        className="bg-white/5 border border-white/10 rounded-xl p-6 hover:bg-white/10 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex-1">
                            <h4 className="text-xl font-semibold">{rec.title}</h4>
                            <p className="text-gray-400">{rec.artist}</p>
                          </div>
                          <div className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-full">
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                            <span className="text-sm font-medium">{rec.match}% Match</span>
                          </div>
                        </div>
                        <p className="text-gray-300 leading-relaxed">{rec.reason}</p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            variants={itemVariants}
            className="mt-20 text-center bg-gradient-to-r from-white/10 to-white/5 border border-white/10 rounded-2xl p-16 backdrop-blur-sm"
          >
            <Music2 className="mx-auto mb-6" size={48} />
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Try It Yourself
            </h2>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              These are just a few examples. Create your own searches and discover songs that match your unique taste
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
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
                onClick={() => navigate('/how-it-works')}
              >
                Learn More
              </Button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
