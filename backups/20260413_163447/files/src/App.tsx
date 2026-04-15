import React, { useState } from 'react';
import Sidebar from './components/Sidebar/Sidebar';
import ChatBox from './components/Chat/ChatBox';
import EvalDashboard from './components/Eval/EvalDashboard';
import { Sun, Moon, Menu, X } from 'lucide-react';

const App: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState<'chat' | 'dashboard'>('chat');

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className={`flex h-screen w-full bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 ${isDarkMode ? 'dark' : ''}`}>
      {/* Mobile Sidebar Overlay */}
      {!isSidebarOpen && (
        <button
          onClick={() => setIsSidebarOpen(true)}
          className="fixed top-4 left-4 z-50 lg:hidden p-2 rounded-lg bg-slate-100 dark:bg-slate-800"
        >
          <Menu size={24} />
        </button>
      )}

      {/* Sidebar */}
      <div className={`${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-40 lg:relative lg:translate-x-0 transition-transform duration-300 ease-in-out`}>
        <Sidebar activeView={activeView} onChangeView={setActiveView} />
        <button
          onClick={() => setIsSidebarOpen(false)}
          className="absolute top-4 right-4 lg:hidden p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
        >
          <X size={20} />
        </button>
      </div>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <header className="h-16 border-b dark:border-slate-800 flex items-center justify-between px-6 bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm z-30">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Mathematics Agentic Chatbot
          </h1>
          
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            {isDarkMode ? <Sun size={20} className="text-yellow-400" /> : <Moon size={20} className="text-slate-600" />}
          </button>
        </header>

        <div className="flex-1 overflow-hidden relative">
          {activeView === 'chat' ? <ChatBox /> : <EvalDashboard />}
        </div>
      </main>
    </div>
  );
};

export default App;
