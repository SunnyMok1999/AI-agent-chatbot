import { create } from 'zustand';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  debug?: {
    agent_outputs?: Record<string, string>;
    retrieval_debug?: Record<string, any>;
    strict_validation?: Record<string, any>;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

interface ChatState {
  conversations: Conversation[];
  currentConversationId: string | null;
  isLoading: boolean;
  addMessage: (conversationId: string, message: Omit<Message, 'id' | 'timestamp'>) => string;
  setLoading: (loading: boolean) => void;
  createNewConversation: () => string;
  setCurrentConversation: (id: string) => void;
  updateAssistantMessage: (conversationId: string, messageId: string, chunk: string) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  currentConversationId: null,
  isLoading: false,

  setLoading: (loading) => set({ isLoading: loading }),

  createNewConversation: () => {
    const id = Math.random().toString(36).substring(7);
    const newConv: Conversation = {
      id,
      title: 'New Chat',
      messages: [],
    };
    set((state) => ({
      conversations: [newConv, ...state.conversations],
      currentConversationId: id,
    }));
    return id;
  },

  setCurrentConversation: (id) => set({ currentConversationId: id }),

  addMessage: (conversationId, message) => {
    const newMessage: Message = {
      ...message,
      id: Math.random().toString(36).substring(7),
      timestamp: new Date(),
    };
    set((state) => ({
      conversations: state.conversations.map((conv) =>
        conv.id === conversationId
          ? { ...conv, messages: [...conv.messages, newMessage] }
          : conv
      ),
    }));
    return newMessage.id;
  },

  updateAssistantMessage: (conversationId, messageId, chunk) => {
    set((state) => ({
      conversations: state.conversations.map((conv) =>
        conv.id === conversationId
          ? {
              ...conv,
              messages: conv.messages.map((msg) =>
                msg.id === messageId
                  ? { ...msg, content: msg.content + chunk }
                  : msg
              ),
            }
          : conv
      ),
    }));
  },
}));
