import React, { useState } from 'react';
import { useChatStore } from '../../store/useChatStore';
import { MessageSquare, Plus, Trash2, Edit3, Settings, LogOut, FileUp } from 'lucide-react';
import FileUpload from './FileUpload';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const Sidebar: React.FC = () => {
  const { conversations, currentConversationId, setCurrentConversation, createNewConversation } = useChatStore();
  const [isUploading, setIsUploading] = useState(false);

  return (
    <div className="flex flex-col h-full w-64 bg-slate-50 dark:bg-slate-950 border-r dark:border-slate-800 p-4 gap-4">
      <button
        onClick={() => createNewConversation()}
        className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 transition-colors w-full"
      >
        <Plus size={18} />
        New Chat
      </button>

      <div className="flex-1 overflow-y-auto space-y-2">
        {conversations.map((conv) => (
          <div
            key={conv.id}
            onClick={() => setCurrentConversation(conv.id)}
            className={cn(
              'flex items-center gap-2 rounded-lg p-2 text-sm cursor-pointer hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors group',
              currentConversationId === conv.id ? 'bg-slate-200 dark:bg-slate-800 text-blue-600 dark:text-blue-400' : 'text-slate-600 dark:text-slate-400'
            )}
          >
            <MessageSquare size={16} />
            <span className="flex-1 truncate">{conv.title}</span>
            <div className="hidden group-hover:flex gap-1">
              <button className="p-1 hover:text-slate-900 dark:hover:text-slate-100">
                <Edit3 size={14} />
              </button>
              <button className="p-1 hover:text-red-500">
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="border-t dark:border-slate-800 pt-4 space-y-2">
        <button 
          onClick={() => setIsUploading(!isUploading)}
          className="flex items-center gap-2 rounded-lg p-2 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors w-full"
        >
          <FileUp size={18} />
          Upload Document
        </button>
        {isUploading && <FileUpload onClose={() => setIsUploading(false)} />}
        
        <button className="flex items-center gap-2 rounded-lg p-2 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors w-full">
          <Settings size={18} />
          Settings
        </button>
        <button className="flex items-center gap-2 rounded-lg p-2 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors w-full">
          <LogOut size={18} />
          Logout
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
