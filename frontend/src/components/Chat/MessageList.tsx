import { useEffect, useRef, useState, useMemo } from 'react';
import type { ChatMessage } from '../../types';
import { MessageBubble } from './MessageBubble';

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return 'Good Morning! What shall we build today?';
  if (hour >= 12 && hour < 17) return 'Good Afternoon! Ready to create something?';
  if (hour >= 17 && hour < 21) return 'Good Evening! Let\'s get things done.';
  return 'Late Night Session \u2014 Let\'s make it count.';
}

interface MessageListProps {
  messages: ChatMessage[];
  isStreaming: boolean;
}

export function MessageList({ messages, isStreaming }: MessageListProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const greeting = useMemo(() => getGreeting(), []);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, autoScroll, isStreaming]);

  const handleScroll = () => {
    if (!listRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(isAtBottom);
  };

  if (messages.length === 0) {
    return (
      <div className="message-list">
        <div className="message-list-empty" style={{ flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '24px', fontWeight: 600, color: '#1e293b' }}>
            {greeting}
          </span>
          <span style={{ fontSize: '14px' }}>
            Type a message below to get started.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="message-list" ref={listRef} onScroll={handleScroll}>
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
    </div>
  );
}
