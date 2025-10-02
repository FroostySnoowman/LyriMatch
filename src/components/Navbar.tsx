import { motion } from 'framer-motion';
import { LogOut, User } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import Logo from './Logo';
import Button from './Button';

export default function Navbar() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-xl border-b border-gray-800"
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="cursor-pointer" onClick={() => navigate('/app')}>
          <Logo size="sm" />
        </div>

        {user && (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <User size={16} />
              <span>{user.email}</span>
            </div>
            <Button variant="ghost" size="sm" onClick={handleSignOut}>
              <LogOut size={16} />
              Sign Out
            </Button>
          </div>
        )}
      </div>
    </motion.nav>
  );
}
