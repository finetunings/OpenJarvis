import { useState, useRef, useCallback, useEffect } from 'react';

const COLLAPSE_CHAR_THRESHOLD = 500;
const COLLAPSE_LINE_THRESHOLD = 6;

function shouldCollapse(text: string): boolean {
  return (
    text.length > COLLAPSE_CHAR_THRESHOLD ||
    text.split('\n').length > COLLAPSE_LINE_THRESHOLD
  );
}

function formatSize(text: string): string {
  const chars = text.length;
  const lines = text.split('\n').length;
  if (chars >= 1000) {
    return `${(chars / 1000).toFixed(1)}k chars, ${lines} line${lines !== 1 ? 's' : ''}`;
  }
  return `${chars} chars, ${lines} line${lines !== 1 ? 's' : ''}`;
}

interface InputAreaProps {
  onSend: (content: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}

export function InputArea({ onSend, onStop, isStreaming }: InputAreaProps) {
  // Pasted/long content stored separately as an "attachment"
  const [attachment, setAttachment] = useState('');
  // Text typed in the visible textarea
  const [typed, setTyped] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const fullMessage = attachment ? attachment + '\n' + typed : typed;

  const handleSend = useCallback(() => {
    if (!fullMessage.trim() || isStreaming) return;
    onSend(fullMessage);
    setAttachment('');
    setTyped('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [fullMessage, isStreaming, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setTyped(newValue);
    // Auto-resize textarea
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  };

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const pasted = e.clipboardData.getData('text');
      if (shouldCollapse(pasted)) {
        e.preventDefault();
        // Store long paste as an attachment pill
        setAttachment((prev) => (prev ? prev + '\n' + pasted : pasted));
      }
      // Short pastes go directly into the textarea as normal
    },
    [],
  );

  const handleClearAttachment = useCallback(() => {
    setAttachment('');
    textareaRef.current?.focus();
  }, []);

  const handleExpandAttachment = useCallback(() => {
    // Move attachment content back into the textarea
    setTyped((prev) => (attachment + (prev ? '\n' + prev : '')));
    setAttachment('');
    // Let React render, then resize
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.height =
          Math.min(textareaRef.current.scrollHeight, 200) + 'px';
      }
    }, 0);
  }, [attachment]);

  // Focus textarea after attachment changes
  useEffect(() => {
    textareaRef.current?.focus();
  }, [attachment]);

  return (
    <div className="input-area">
      {attachment && (
        <div className="input-attachment-row">
          <div className="pasted-pill">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
              <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
              <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
            </svg>
            <span className="pasted-pill-text">Pasted text</span>
            <span className="pasted-pill-size">{formatSize(attachment)}</span>
            <button
              className="pasted-pill-action"
              onClick={handleExpandAttachment}
              title="Expand to edit"
            >
              Edit
            </button>
            <button
              className="pasted-pill-action pasted-pill-remove"
              onClick={handleClearAttachment}
              title="Remove pasted text"
            >
              &times;
            </button>
          </div>
        </div>
      )}
      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={typed}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={
            attachment
              ? 'Add instructions for the pasted text...'
              : 'Type a message... (Shift+Enter for new line)'
          }
          rows={3}
          disabled={isStreaming}
        />
        {isStreaming ? (
          <button className="stop-btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!fullMessage.trim()}
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}
