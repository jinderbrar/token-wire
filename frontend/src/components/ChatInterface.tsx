import React, { useState, useRef, useEffect, useCallback } from 'react';
import { TokenStats, initialStats } from '../App';

const API_BASE = 'http://localhost:8000';

type Protocol = 'tokenWire' | 'baseline' | 'benchmark';

type Message = {
  id: string;
  role: 'user' | 'ai';
  text: string;
  protocol?: 'tokenWire' | 'baseline';
  stats?: TokenStats;
  isStreaming?: boolean;
};

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [protocol, setProtocol] = useState<Protocol>('tokenWire');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  const workerRef = useRef<Worker | null>(null);
  const baselineSocketRef = useRef<WebSocket | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const aiMessageIdRef = useRef<string | null>(null);

  useEffect(() => {
    // Fetch available models
    fetch(`${API_BASE}/api/models`)
      .then(r => r.json())
      .then(data => {
        const list: string[] = data.models || [];
        setModels(list);
        if (list.length > 0) {
          setSelectedModel(list[0]);
          // Initialize worker with first model's dictionary
          if (workerRef.current) {
            workerRef.current.postMessage({ type: 'INIT_DICTIONARY', model: list[0] });
          }
        }
      })
      .catch(console.error);

    workerRef.current = new Worker(
      new URL('../workers/inflator.worker.ts', import.meta.url),
      { type: 'module' }
    );
    // Initial dictionary load will happen after models are fetched

    workerRef.current.onmessage = (e) => {
      const data = e.data;
      if (data.type === 'TOKEN_DECODED') {
        const { text, stats } = data;
        const msgId = aiMessageIdRef.current;
        if (!msgId) return;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === msgId
              ? {
                ...m,
                text: m.text + text,
                isStreaming: true,
                stats: {
                  ...initialStats,
                  payloadBytes: stats.binaryBytes,
                  tokensPerSecond: stats.tps,
                  totalTokens: stats.tokens,
                  timeToFirstTokenMs: stats.ttft,
                  totalTimeMs: stats.totalMs,
                },
              }
              : m
          )
        );
      } else if (data.type === 'STREAM_DONE') {
        const msgId = aiMessageIdRef.current;
        setIsGenerating(false);
        if (msgId) {
          setMessages((prev) =>
            prev.map((m) => (m.id === msgId ? { ...m, isStreaming: false } : m))
          );
        }
      }
    };

    return () => workerRef.current?.terminate();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Wraps baseline WS into a promise so we can await it in benchmark mode
  const runBaselineStream = useCallback(
    (prompt: string, aiMsgId: string): Promise<void> => {
      return new Promise((resolve) => {
        if (baselineSocketRef.current) baselineSocketRef.current.close();
        const ws = new WebSocket('ws://localhost:8000/ws/baseline');
        baselineSocketRef.current = ws;

        let tokens = 0, totalBytes = 0, streamStart = 0;
        let ttft: number | null = null;

        ws.onopen = () => {
          streamStart = performance.now();
          ws.send(prompt);
        };

        ws.onmessage = (e) => {
          const now = performance.now();
          if (ttft === null) ttft = now - streamStart;

          const payload = e.data;
          totalBytes += new Blob([payload]).size;
          const msg = JSON.parse(payload);

          if (msg.done) {
            setMessages((prev) =>
              prev.map((m) => (m.id === aiMsgId ? { ...m, isStreaming: false } : m))
            );
            resolve();
            return;
          }

          tokens++;

          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsgId
                ? {
                  ...m,
                  text: m.text + msg.response,
                  isStreaming: true,
                  stats: {
                    ...initialStats,
                    payloadBytes: totalBytes,
                    tokensPerSecond: tokens / ((performance.now() - streamStart) / 1000),
                    timeToFirstTokenMs: ttft,
                    totalTokens: tokens,
                    totalTimeMs: performance.now() - streamStart,
                  },
                }
                : m
            )
          );
        };

        ws.onclose = () => resolve();
        ws.onerror = () => resolve();
      });
    },
    []
  );

  // Wraps tokenWire worker into a promise so we can await it in benchmark mode
  const runTokenWireStream = useCallback(
    (prompt: string, aiMsgId: string): Promise<void> => {
      return new Promise((resolve) => {
        if (!workerRef.current) { resolve(); return; }

        // Temporarily override onmessage for this run
        const prevHandler = workerRef.current.onmessage;
        workerRef.current.onmessage = (e) => {
          const data = e.data;
          if (data.type === 'TOKEN_DECODED') {
            const { text, stats } = data;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiMsgId
                  ? {
                    ...m,
                    text: m.text + text,
                    isStreaming: true,
                    stats: {
                      ...initialStats,
                      payloadBytes: stats.binaryBytes,
                      tokensPerSecond: stats.tps,
                      totalTokens: stats.tokens,
                      timeToFirstTokenMs: stats.ttft,
                      totalTimeMs: stats.totalMs,
                    },
                  }
                  : m
              )
            );
          } else if (data.type === 'STREAM_DONE') {
            setMessages((prev) =>
              prev.map((m) => (m.id === aiMsgId ? { ...m, isStreaming: false } : m))
            );
            // Restore original handler and resolve
            if (workerRef.current) workerRef.current.onmessage = prevHandler;
            resolve();
          }
        };

        workerRef.current.postMessage({
          type: 'START_STREAM',
          prompt,
          headless: false,
        });
      });
    },
    []
  );

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isGenerating) return;

    const userPrompt = input.trim();
    setInput('');
    setIsGenerating(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    const userMsgId = `u-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', text: userPrompt },
    ]);

    if (protocol === 'benchmark') {
      // 1. Baseline run
      const baselineMsgId = `ai-baseline-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        { id: baselineMsgId, role: 'ai', text: '', protocol: 'baseline', isStreaming: true },
      ]);
      aiMessageIdRef.current = baselineMsgId;
      await runBaselineStream(userPrompt, baselineMsgId);

      // 2. TokenWire run (immediately after)
      const tokenWireMsgId = `ai-tokenWire-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        { id: tokenWireMsgId, role: 'ai', text: '', protocol: 'tokenWire', isStreaming: true },
      ]);
      aiMessageIdRef.current = tokenWireMsgId;
      await runTokenWireStream(userPrompt, tokenWireMsgId);

      setIsGenerating(false);
    } else if (protocol === 'baseline') {
      const aiMsgId = `ai-baseline-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        { id: aiMsgId, role: 'ai', text: '', protocol: 'baseline', isStreaming: true },
      ]);
      aiMessageIdRef.current = aiMsgId;
      await runBaselineStream(userPrompt, aiMsgId);
      setIsGenerating(false);
    } else {
      // TokenWire only
      const aiMsgId = `ai-tokenWire-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        { id: aiMsgId, role: 'ai', text: '', protocol: 'tokenWire', isStreaming: true },
      ]);
      aiMessageIdRef.current = aiMsgId;
      workerRef.current?.postMessage({
        type: 'START_STREAM',
        prompt: userPrompt,
        headless: false,
      });
      // isGenerating cleared by STREAM_DONE handler
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  const copyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const protocolLabel = (p?: 'tokenWire' | 'baseline') =>
    p === 'tokenWire' ? 'TokenWire Binary' : 'Baseline JSON';

  return (
    <div className="chat-root">
      {/* Scrollable messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h2 className="chat-empty-title">TokenWire Engine</h2>
            <p className="chat-empty-sub">
              Stream text via binary TokenWire, legacy JSON, or run both side-by-side in Benchmark mode.
            </p>
          </div>
        )}

        <div className="chat-message-list">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-row chat-row--${msg.role}`}>
              {msg.role === 'ai' && (
                <div className="chat-ai-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
              )}

              <div className={`chat-bubble-wrap chat-bubble-wrap--${msg.role}`}>
                {msg.role === 'user' ? (
                  <div className="chat-bubble-user">{msg.text}</div>
                ) : (
                  <div className="chat-bubble-ai">
                    <div className="chat-ai-protocol">
                      {protocolLabel(msg.protocol)}
                    </div>
                    <div className="chat-ai-text">
                      {msg.text || (
                        <span className="chat-typing">
                          <span /><span /><span />
                        </span>
                      )}
                    </div>

                    {/* Stats action row — appears below each completed response */}
                    {!msg.isStreaming && msg.stats && (
                      <div className="chat-actions">
                        <div className="chat-stat">
                          <span className="chat-stat-label">TPS</span>
                          <span className="chat-stat-value chat-stat-value--tps">
                            {msg.stats.tokensPerSecond.toFixed(1)}
                          </span>
                        </div>
                        <div className="chat-stat-divider" />
                        <div className="chat-stat">
                          <span className="chat-stat-label">TTFT</span>
                          <span className="chat-stat-value chat-stat-value--ttft">
                            {msg.stats.timeToFirstTokenMs?.toFixed(0)}ms
                          </span>
                        </div>
                        <div className="chat-stat-divider" />
                        <div className="chat-stat">
                          <span className="chat-stat-label">Size</span>
                          <span className="chat-stat-value chat-stat-value--size">
                            {(msg.stats.payloadBytes / 1024).toFixed(2)} KB
                          </span>
                        </div>
                        <div className="chat-stat-divider" />
                        <div className="chat-stat">
                          <span className="chat-stat-label">Time</span>
                          <span className="chat-stat-value">
                            {(msg.stats.totalTimeMs / 1000).toFixed(2)}s
                          </span>
                        </div>
                        <div className="chat-stat-divider" />
                        <button
                          className="chat-action-btn"
                          onClick={() => copyText(msg.text, msg.id)}
                          title="Copy response"
                        >
                          {copiedId === msg.id ? (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                              <polyline points="20 6 9 17 4 12" />
                            </svg>
                          ) : (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                            </svg>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={endRef} style={{ height: 16 }} />
        </div>
      </div>

      {/* Fixed bottom input */}
      <div className="chat-input-wrap">
        <div className="chat-input-inner">
          <div className="chat-input-box">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Message TokenWire..."
              rows={1}
              disabled={isGenerating}
            />
            <button
              className={`chat-send-btn ${isGenerating || !input.trim() ? 'chat-send-btn--disabled' : ''}`}
              onClick={() => handleSubmit()}
              disabled={isGenerating || !input.trim()}
            >
              {isGenerating ? (
                <div className="chat-spinner" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M2 12l20-9-9 20-4-7-7-4z" />
                </svg>
              )}
            </button>
          </div>

          {/* Protocol selector */}
          <div className="chat-bottom-bar">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div className="chat-protocol-toggle">
                <button
                  className={`chat-proto-btn ${protocol === 'tokenWire' ? 'chat-proto-btn--active' : ''}`}
                  onClick={() => setProtocol('tokenWire')}
                >
                  <span className={`chat-proto-dot ${protocol === 'tokenWire' ? 'chat-proto-dot--active' : ''}`} />
                  TokenWire Binary
                </button>
                <button
                  className={`chat-proto-btn ${protocol === 'baseline' ? 'chat-proto-btn--base' : ''}`}
                  onClick={() => setProtocol('baseline')}
                >
                  Baseline JSON
                </button>
                <button
                  className={`chat-proto-btn ${protocol === 'benchmark' ? 'chat-proto-btn--bench' : ''}`}
                  onClick={() => setProtocol('benchmark')}
                >
                  <span className={`chat-proto-dot ${protocol === 'benchmark' ? 'chat-proto-dot--bench' : ''}`} />
                  Benchmark Both
                </button>
              </div>

              {/* Model selector */}
              {models.length > 0 && (
                <div className="chat-model-row">
                  <span className="chat-model-label">Model</span>
                  <select
                    className="chat-model-select"
                    value={selectedModel}
                    disabled={isGenerating}
                    onChange={async (e) => {
                      const m = e.target.value;
                      setSelectedModel(m);
                      // Load model in backend
                      await fetch(`${API_BASE}/api/model/load`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model_id: m }),
                      }).catch(console.error);
                      // Reload dictionary for new model
                      workerRef.current?.postMessage({ type: 'RELOAD_DICTIONARY', model: m });
                    }}
                  >
                    {models.map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
            <p className="chat-disclaimer">
              Local inference · Binary transport · Zero-copy decode
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
