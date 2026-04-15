import React from 'react';
import { Message } from '../../store/useChatStore';
import { User, Bot } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const [showDebug, setShowDebug] = React.useState(false);

  const hasDebug = !isUser && (message.debug?.agent_outputs || message.debug?.retrieval_debug || message.debug?.strict_validation);

  return (
    <div
      className={cn(
        'flex w-full mb-4 gap-3',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      <div
        className={cn(
          'flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow',
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
        )}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>
      <div
        className={cn(
          'flex flex-col gap-1 max-w-[80%]',
          isUser ? 'items-end' : 'items-start'
        )}
      >
        <div
          className={cn(
            'rounded-lg px-4 py-2 text-sm shadow-sm',
            isUser
              ? 'bg-blue-600 text-white rounded-tr-none'
              : 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 rounded-tl-none'
          )}
        >
          {message.content}
        </div>

        {hasDebug && (
          <div className="w-full">
            <button
              type="button"
              onClick={() => setShowDebug((v) => !v)}
              className="text-[11px] text-blue-600 dark:text-blue-400 hover:underline"
            >
              {showDebug ? 'Hide debug details' : 'Show debug details'}
            </button>

            {showDebug && (
              <div className="mt-2 rounded-md border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-3 text-[11px] space-y-3">
                {message.debug?.strict_validation && (
                  <div>
                    <div className="font-semibold">Strict validation</div>
                    <pre className="whitespace-pre-wrap">{JSON.stringify(message.debug.strict_validation, null, 2)}</pre>
                  </div>
                )}

                {message.debug?.retrieval_debug && (
                  <div>
                    <div className="font-semibold">Retrieval / reranker</div>
                    <pre className="whitespace-pre-wrap">{JSON.stringify(message.debug.retrieval_debug, null, 2)}</pre>
                  </div>
                )}

                {message.debug?.agent_outputs && (
                  <div>
                    <div className="font-semibold">Agent outputs</div>
                    {Object.entries(message.debug.agent_outputs).map(([agent, output]) => (
                      <details key={agent} className="mt-1">
                        <summary className="cursor-pointer font-medium">{agent}</summary>
                        <div className="mt-1 whitespace-pre-wrap">{String(output)}</div>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        <span className="text-[10px] text-muted-foreground">
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
};

export default MessageBubble;
