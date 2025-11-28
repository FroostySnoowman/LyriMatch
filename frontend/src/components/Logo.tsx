import { Music } from 'lucide-react';
import { motion } from 'framer-motion';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
}

export default function Logo({ size = 'md', animated = false }: LogoProps) {
  const sizeClasses = {
    sm: 'text-xl',
    md: 'text-2xl',
    lg: 'text-4xl',
  };

  const iconSizes = {
    sm: 20,
    md: 28,
    lg: 40,
  };

  const LogoContent = (
    <div className={`flex items-center gap-2 ${sizeClasses[size]} font-bold`}>
      <Music size={iconSizes[size]} className="text-white" />
      <span className="text-white">Lyri<span className="text-gray-400">Match</span></span>
    </div>
  );

  if (animated) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {LogoContent}
      </motion.div>
    );
  }

  return LogoContent;
}
