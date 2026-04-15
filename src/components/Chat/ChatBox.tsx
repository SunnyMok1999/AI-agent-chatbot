import React, { useState, useEffect, useRef } from 'react';
import { useChatStore } from '../../store/useChatStore';
import MessageBubble from './MessageBubble';
import { Send, Loader2, Bot } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const ChatBox: React.FC = () => {
  const [input, setInput] = useState('');
  const { currentConversationId, conversations, isLoading, setLoading, createNewConversation, addMessage, updateAssistantMessage } = useChatStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const debugChatEnabled = import.meta.env.VITE_ENABLE_DEBUG_CHAT !== 'false';

  const currentConversation = conversations.find(c => c.id === currentConversationId);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [currentConversation?.messages]);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    let conversationId = currentConversationId;
    if (!conversationId) {
      conversationId = createNewConversation();
    }

    const userMessage = input.trim();
    setInput('');
    addMessage(conversationId, { role: 'user', content: userMessage });

    setLoading(true);

    try {
      if (debugChatEnabled) {
        const response = await fetch('/api/debug/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage, stream: false, conversation_id: conversationId }),
        });

        if (!response.ok) throw new Error('Failed to send message');
        const data = await response.json();

        addMessage(conversationId, {
          role: 'assistant',
          content: data?.answer || data?.content || 'No answer returned.',
          debug: {
            agent_outputs: data?.agent_outputs || {},
            retrieval_debug: data?.retrieval_debug || {},
            strict_validation: data?.strict_validation || {},
            upload_context: data?.upload_context || null,
          },
        });
      } else {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage, conversation_id: conversationId }),
        });

        if (!response.ok) throw new Error('Failed to send message');

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        addMessage(conversationId, { role: 'assistant', content: '' });

        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6));
                  if (data.type === 'chunk') {
                    const conversation = useChatStore.getState().conversations.find(c => c.id === conversationId);
                    const lastMsg = conversation?.messages[conversation.messages.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant') {
                      updateAssistantMessage(conversationId, lastMsg.id, data.content);
                    }
                  }
                } catch {
                  // Ignore parsing errors for non-JSON lines
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      addMessage(conversationId, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full bg-white dark:bg-slate-900 border-x dark:border-slate-800">
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 scroll-smooth"
      >
        {currentConversation?.messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && currentConversation?.messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="animate-spin text-primary" size={32} />
          </div>
        )}
        {!currentConversationId && (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-4">
            <Bot size={48} className="text-slate-300" />
            <p>Start a new conversation to begin chatting!</p>
          </div>
        )}
      </div>

      <form 
        onSubmit={handleSend}
        className="p-4 border-t dark:border-slate-800 bg-white dark:bg-slate-900"
      >
        <div className="flex gap-2 max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything about mathematics..."
            className="flex-1 rounded-full border dark:border-slate-700 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary dark:bg-slate-800 dark:text-white"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="rounded-full bg-blue-600 p-2 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isLoading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatBox;
