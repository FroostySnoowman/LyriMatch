import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  type?: 'button' | 'submit';
  disabled?: boolean;
  className?: string;
}

export default function Button({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  type = 'button',
  disabled = false,
  className = '',
}: ButtonProps) {
  const baseClasses = 'font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2';

  const variantClasses = {
    primary: 'bg-white text-black hover:bg-gray-200 disabled:bg-gray-600 disabled:text-gray-400',
    secondary: 'bg-transparent border border-white text-white hover:bg-white hover:text-black disabled:border-gray-600 disabled:text-gray-600',
    ghost: 'bg-transparent text-white hover:bg-white/10 disabled:text-gray-600',
  };

  const sizeClasses = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg',
  };

  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
    >
      {children}
    </motion.button>
  );
}
