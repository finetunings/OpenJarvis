import { useState, useRef, useCallback } from 'react';
import type { ChatMessage, StreamState, ToolCallInfo, TokenUsage } from '../types';
import { streamChat } from '../api/sse';
import * as storage from '../storage/conversations';

const INITIAL_STREAM_STATE: StreamState = {
  isStreaming: false,
  phase: '',
  elapsedMs: 0,
  activeToolCalls: [],
  content: '',
};

export function useChat(conversationId: string | null, model: string) {
  const [streamState, setStreamState] = useState<StreamState>(INITIAL_STREAM_STATE);
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (!conversationId) return [];
    const conv = storage.getConversation(conversationId);
    return conv?.messages || [];
  });

  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  // Reload messages when conversation changes
  const reloadMessages = useCallback(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }
    const conv = storage.getConversation(conversationId);
    setMessages(conv?.messages || []);
  }, [conversationId]);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setStreamState(INITIAL_STREAM_STATE);
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!conversationId || !content.trim()) return;

      // Add user message
      const userMsg: ChatMessage = {
        id: storage.generateMessageId(),
        role: 'user',
        content: content.trim(),
        timestamp: Date.now(),
      };
      storage.addMessage(conversationId, userMsg);

      // Build API messages BEFORE adding the assistant placeholder,
      // so the placeholder's empty content isn't sent to the backend.
      const conv = storage.getConversation(conversationId);
      const apiMessages = (conv?.messages || []).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Add placeholder assistant message (after building apiMessages)
      const assistantMsg: ChatMessage = {
        id: storage.generateMessageId(),
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      };
      storage.addMessage(conversationId, assistantMsg);

      // Update local state
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      // Start timer
      startTimeRef.current = Date.now();
      const timer = setInterval(() => {
        setStreamState((s) => ({
          ...s,
          elapsedMs: Date.now() - startTimeRef.current,
        }));
      }, 100);
      timerRef.current = timer;

      const controller = new AbortController();
      abortRef.current = controller;

      let accumulatedContent = '';
      let usage: TokenUsage | undefined;
      const toolCalls: ToolCallInfo[] = [];

      setStreamState({
        isStreaming: true,
        phase: 'Sending request...',
        elapsedMs: 0,
        activeToolCalls: [],
        content: '',
      });

      try {
        for await (const sseEvent of streamChat(
          { model, messages: apiMessages, stream: true },
          controller.signal,
        )) {
          const eventName = sseEvent.event;

          if (eventName === 'agent_turn_start') {
            setStreamState((s) => ({ ...s, phase: 'Agent thinking...' }));
          } else if (eventName === 'inference_start') {
            setStreamState((s) => ({ ...s, phase: 'Generating response...' }));
          } else if (eventName === 'inference_end') {
            // Just update phase
          } else if (eventName === 'tool_call_start') {
            try {
              const data = JSON.parse(sseEvent.data);
              const tc: ToolCallInfo = {
                id: storage.generateMessageId(),
                tool: data.tool,
                arguments: data.arguments || '',
                status: 'running',
              };
              toolCalls.push(tc);
              setStreamState((s) => ({
                ...s,
                phase: `Running ${data.tool}...`,
                activeToolCalls: [...toolCalls],
              }));
            } catch {}
          } else if (eventName === 'tool_call_end') {
            try {
              const data = JSON.parse(sseEvent.data);
              const tc = toolCalls.find((t) => t.tool === data.tool && t.status === 'running');
              if (tc) {
                tc.status = data.success ? 'success' : 'error';
                tc.latency = data.latency;
                tc.result = data.result;
              }
              setStreamState((s) => ({
                ...s,
                phase: 'Generating response...',
                activeToolCalls: [...toolCalls],
              }));
            } catch {}
          } else {
            // Content chunk (no event name or event: content)
            try {
              const data = JSON.parse(sseEvent.data);
              const delta = data.choices?.[0]?.delta;
              if (data.usage) {
                usage = data.usage;
              }
              if (delta?.content) {
                accumulatedContent += delta.content;
                setStreamState((s) => ({
                  ...s,
                  content: accumulatedContent,
                }));
                // Update messages in state
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === 'assistant') {
                    updated[updated.length - 1] = {
                      ...last,
                      content: accumulatedContent,
                      toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined,
                    };
                  }
                  return updated;
                });
              }
              if (data.choices?.[0]?.finish_reason === 'stop') break;
            } catch {}
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          accumulatedContent = accumulatedContent || 'Error: Failed to get response.';
        }
      } finally {
        // Show a message if streaming completed with no content
        if (!accumulatedContent) {
          accumulatedContent = 'No response was generated. Please try again.';
        }
        // Save final state
        storage.updateLastAssistantMessage(
          conversationId,
          accumulatedContent,
          toolCalls.length > 0 ? toolCalls : undefined,
          usage,
        );
        // Update local messages with usage
        if (usage) {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, usage };
            }
            return updated;
          });
        }
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        setStreamState(INITIAL_STREAM_STATE);
        abortRef.current = null;
      }
    },
    [conversationId, model],
  );

  return {
    messages,
    streamState,
    sendMessage,
    stopStreaming,
    reloadMessages,
  };
}
