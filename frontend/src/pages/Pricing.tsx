import { motion } from 'framer-motion';
import { Check, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import AnimatedBackground from '../components/AnimatedBackground';
import UnifiedNavbar from '../components/UnifiedNavbar';
import Button from '../components/Button';

export default function Pricing() {
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

  const plans = [
    {
      name: 'Free',
      price: '$0',
      period: 'forever',
      description: 'Perfect for casual music discovery',
      features: [
        '5 lyric searches per day',
        'Basic recommendations',
        'Up to 3 songs per search',
        'Community support',
        'Search history (7 days)',
      ],
      cta: 'Get Started',
      highlighted: false,
    },
    {
      name: 'Pro',
      price: '$9.99',
      period: 'per month',
      description: 'For serious music enthusiasts',
      features: [
        'Unlimited lyric searches',
        'Advanced AI recommendations',
        'Up to 10 songs per search',
        'Priority support',
        'Full search history',
        'Export results',
        'API access',
        'No ads',
      ],
      cta: 'Start Free Trial',
      highlighted: true,
    },
    {
      name: 'Team',
      price: '$29.99',
      period: 'per month',
      description: 'For music professionals and teams',
      features: [
        'Everything in Pro',
        'Up to 5 team members',
        'Shared playlists',
        'Team analytics',
        'Custom integrations',
        'Dedicated account manager',
        'SLA guarantee',
      ],
      cta: 'Contact Sales',
      highlighted: false,
    },
  ];

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
              Simple, <span className="bg-gradient-to-r from-white via-gray-300 to-gray-500 bg-clip-text text-transparent">Transparent Pricing</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Choose the plan that works best for you. All plans include our core AI-powered music discovery features.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 mb-20">
            {plans.map((plan, index) => (
              <motion.div
                key={plan.name}
                variants={itemVariants}
                className={`relative p-8 rounded-2xl backdrop-blur-sm ${
                  plan.highlighted
                    ? 'bg-white/10 border-2 border-white/30 scale-105'
                    : 'bg-white/5 border border-white/10'
                }`}
                whileHover={{ scale: plan.highlighted ? 1.07 : 1.02, transition: { duration: 0.2 } }}
              >
                {plan.highlighted && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 px-4 py-1 bg-white text-black text-sm font-semibold rounded-full">
                    Most Popular
                  </div>
                )}

                <div className="mb-6">
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <p className="text-gray-400 text-sm">{plan.description}</p>
                </div>

                <div className="mb-6">
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold">{plan.price}</span>
                    <span className="text-gray-400">/ {plan.period}</span>
                  </div>
                </div>

                <Button
                  variant={plan.highlighted ? 'primary' : 'secondary'}
                  className="w-full mb-8"
                  onClick={() => navigate('/signup')}
                >
                  {plan.cta}
                  <ArrowRight size={16} />
                </Button>

                <div className="space-y-4">
                  {plan.features.map((feature, featureIndex) => (
                    <div key={featureIndex} className="flex items-start gap-3">
                      <Check size={20} className="text-white flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300 text-sm">{feature}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div variants={itemVariants} className="text-center mb-20">
            <h2 className="text-3xl font-bold mb-8">Frequently Asked Questions</h2>
            <div className="grid md:grid-cols-2 gap-6 text-left max-w-5xl mx-auto">
              <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
                <h3 className="text-lg font-semibold mb-2">Can I change plans anytime?</h3>
                <p className="text-gray-400 text-sm">
                  Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately.
                </p>
              </div>

              <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
                <h3 className="text-lg font-semibold mb-2">Is there a free trial?</h3>
                <p className="text-gray-400 text-sm">
                  Pro plans come with a 14-day free trial. No credit card required to start.
                </p>
              </div>

              <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
                <h3 className="text-lg font-semibold mb-2">What payment methods do you accept?</h3>
                <p className="text-gray-400 text-sm">
                  We accept all major credit cards, PayPal, and wire transfers for enterprise plans.
                </p>
              </div>

              <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
                <h3 className="text-lg font-semibold mb-2">Can I cancel anytime?</h3>
                <p className="text-gray-400 text-sm">
                  Absolutely. Cancel anytime with no questions asked. You'll retain access until the end of your billing period.
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="text-center bg-gradient-to-r from-white/10 to-white/5 border border-white/10 rounded-2xl p-16 backdrop-blur-sm"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Still Have Questions?
            </h2>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              Our team is here to help. Reach out and we'll get back to you within 24 hours.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="primary" size="lg" onClick={() => navigate('/signup')}>
                Start Free Trial
              </Button>
              <Button variant="secondary" size="lg">
                Contact Support
              </Button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
