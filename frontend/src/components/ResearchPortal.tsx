import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

type SerializableStats = {
  payloadBytes: number;
  totalTokens: number;
  timeToFirstTokenMs: number | null;
  tokensPerSecond: number;
  responseTotalMs: number;   // total wall-clock time for the full stream
};

type PromptResult = {
  prompt_id: string;
  category: string;
  prompt: string;
  tokenWire?: SerializableStats;
  baseline?: SerializableStats;
};

const fmt = {
  kb:  (b: number) => (b / 1024).toFixed(2) + ' KB',
  tps: (t: number) => t.toFixed(1),
  ms:  (m: number | null) => m != null ? m.toFixed(0) + ' ms' : '—',
  sec: (m: number) => (m / 1000).toFixed(2) + ' s',
};

export const ResearchPortal: React.FC = () => {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [status, setStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [results, setResults] = useState<PromptResult[]>([]);

  const workerRef = useRef<Worker | null>(null);
  const baselineSocketRef = useRef<WebSocket | null>(null);
  const isRunningRef = useRef(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/research/datasets`)
      .then((r) => r.json())
      .then((data) => {
        setDatasets(data.datasets || []);
        if (data.datasets?.length > 0) setSelectedDataset(data.datasets[0].id);
      })
      .catch(console.error);

    workerRef.current = new Worker(
      new URL('../workers/inflator.worker.ts', import.meta.url),
      { type: 'module' }
    );

    // Fetch available models and init dictionary with the first one
    fetch(`${API_BASE}/api/research/models`)
      .then((r) => r.json())
      .then((data) => {
        const models = data.models || [];
        if (models.length > 0 && workerRef.current) {
          workerRef.current.postMessage({ type: 'INIT_DICTIONARY', model: models[0] });
        }
      })
      .catch(console.error);

    return () => {
      workerRef.current?.terminate();
      baselineSocketRef.current?.close();
    };
  }, []);

  const runTokenWireHeadless = (prompt: string): Promise<SerializableStats> =>
    new Promise((resolve) => {
      if (!workerRef.current)
        return resolve({ payloadBytes: 0, totalTokens: 0, timeToFirstTokenMs: 0, tokensPerSecond: 0, responseTotalMs: 0 });

      const wallStart = performance.now();

      workerRef.current.onmessage = (e) => {
        if (e.data.type === 'STREAM_DONE') {
          const s = e.data.stats || { binaryBytes: 0, tokens: 0, ttft: 0, tps: 0 };
          resolve({
            payloadBytes: s.binaryBytes,
            totalTokens: s.tokens,
            timeToFirstTokenMs: s.ttft,
            tokensPerSecond: s.tps,
            responseTotalMs: performance.now() - wallStart,
          });
        }
      };

      workerRef.current.postMessage({ type: 'START_STREAM', prompt, headless: true });
    });

  const runBaselineHeadless = (prompt: string): Promise<SerializableStats> =>
    new Promise((resolve) => {
      baselineSocketRef.current?.close();
      const ws = new WebSocket('ws://localhost:8000/ws/baseline');
      baselineSocketRef.current = ws;

      let tokens = 0, totalBytes = 0, streamStart = 0;
      let ttft: number | null = null;

      ws.onopen = () => {
        streamStart = performance.now();
        ws.send(prompt);
      };

      ws.onmessage = (e) => {
        totalBytes += new Blob([e.data]).size;
        const msg = JSON.parse(e.data);
        if (msg.done) {
          const elapsed = performance.now() - streamStart;
          resolve({
            payloadBytes: totalBytes,
            totalTokens: tokens,
            timeToFirstTokenMs: ttft,
            tokensPerSecond: tokens / (elapsed / 1000),
            responseTotalMs: elapsed,
          });
          return;
        }
        tokens++;
        if (ttft === null) ttft = performance.now() - streamStart;
      };

      ws.onerror = () =>
        resolve({ payloadBytes: 0, totalTokens: 0, timeToFirstTokenMs: 0, tokensPerSecond: 0, responseTotalMs: 0 });
    });

  const startBenchmark = async () => {
    if (!selectedDataset) return;
    setStatus('running');
    isRunningRef.current = true;
    setResults([]);

    try {
      const resp = await fetch(`${API_BASE}/api/research/datasets/${selectedDataset}/prompts`);
      const data = await resp.json();
      const prompts = data.prompts || [];
      setProgress({ current: 0, total: prompts.length });

      const localResults: PromptResult[] = [];

      for (let i = 0; i < prompts.length; i++) {
        if (!isRunningRef.current) break;
        const p = prompts[i];

        await fetch(`${API_BASE}/api/research/cache/clear`, { method: 'POST' }).catch(() => { });
        const baselineStats = await runBaselineHeadless(p.prompt);

        await fetch(`${API_BASE}/api/research/cache/clear`, { method: 'POST' }).catch(() => { });
        const tokenWireStats = await runTokenWireHeadless(p.prompt);

        localResults.push({
          prompt_id: p.id,
          category: p.category,
          prompt: p.prompt,
          baseline: baselineStats,
          tokenWire: tokenWireStats,
        });

        setResults([...localResults]);
        setProgress({ current: i + 1, total: prompts.length });
      }
    } catch (e) {
      console.error(e);
    } finally {
      setStatus('completed');
      isRunningRef.current = false;
    }
  };

  const stopBenchmark = () => {
    isRunningRef.current = false;
    setStatus('idle');
  };

  const pct = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <div className="research-root">
      <div className="research-content">

        {/* ── Header ── */}
        <div className="research-header-card">
          <div className="research-header-glow" />

          <div className="research-header-left">
            <h1>
              <div className="research-header-icon">
                <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              Research Portal
            </h1>
            <p className="research-header-sub">Headless Evaluation · JSON vs TokenWire Binary</p>
          </div>

          <div className="research-header-right">
            <div className="research-select-wrap">
              <span className="research-select-label">Evaluation Dataset</span>
              <select
                className="research-select"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={status === 'running'}
              >
                <option value="" disabled>Choose a dataset…</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.filename} ({d.prompt_count} prompts)
                  </option>
                ))}
              </select>
            </div>

            {status === 'running' ? (
              <button className="research-abort-btn" onClick={stopBenchmark}>Abort Run</button>
            ) : (
              <button className="research-run-btn" onClick={startBenchmark} disabled={!selectedDataset}>
                Run Benchmark
              </button>
            )}
          </div>
        </div>

        {/* ── Empty state ── */}
        {!selectedDataset && status === 'idle' && (
          <div className="research-empty">
            <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="0.8">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p style={{ fontSize: 15, fontWeight: 600 }}>No Dataset Selected</p>
            <p style={{ fontSize: 13 }}>Choose a dataset above to begin headless evaluation across both protocols.</p>
          </div>
        )}

        {/* ── Progress ── */}
        {status === 'running' && (
          <div className="research-progress-card">
            <div className="research-progress-ring">
              <div className="research-progress-ring-bg" />
              <div className="research-progress-ring-fill" />
              <div className="research-progress-pct">{pct}%</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p className="research-progress-label">Evaluating Headlessly</p>
              <p className="research-progress-sub" style={{ marginTop: 4 }}>
                Prompt {progress.current} of {progress.total}
              </p>
            </div>
            <div className="research-progress-bar-wrap">
              <div className="research-progress-bar-fill" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}

        {/* ── Results: grouped table ── */}
        {results.length > 0 && (
          <div className="research-table-card">
            <div className="research-table-header">
              <span className="research-table-title">Execution Ledger</span>
              <span className="research-table-count">{results.length} prompt{results.length > 1 ? 's' : ''} · 2 protocols each</span>
            </div>

            <table className="research-table research-table--grouped">
              <thead>
                <tr>
                  <th style={{ width: '28%' }}>Prompt</th>
                  <th style={{ width: '10%' }}>Protocol</th>
                  <th className="center" style={{ width: '12%' }}>TTFT</th>
                  <th className="center" style={{ width: '10%' }}>TPS</th>
                  <th className="center" style={{ width: '12%' }}>Size</th>
                  <th className="center" style={{ width: '12%' }}>Total Time</th>
                  <th className="center" style={{ width: '10%' }}>Tokens</th>
                  <th className="center" style={{ width: '10%' }}>Δ Size</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => {
                  const baseBytes = r.baseline?.payloadBytes || 1;
                  const latBytes  = r.tokenWire?.payloadBytes  || 1;
                  const savings   = ((1 - latBytes / baseBytes) * 100).toFixed(1);
                  const isLast    = i === results.length - 1;

                  return (
                    <React.Fragment key={r.prompt_id}>
                      {/* ── Baseline row ── */}
                      <tr className={`research-grouped-row research-grouped-row--first ${isLast ? 'research-grouped-row--last-group' : ''}`}>
                        {/* Prompt cell spans both rows using rowSpan=2 */}
                        <td
                          className="research-prompt-cell"
                          rowSpan={2}
                          style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}
                        >
                          <span className="research-prompt-category">{r.category}</span>
                          <span className="research-prompt-text">{r.prompt.length > 60 ? r.prompt.slice(0, 60) + '…' : r.prompt}</span>
                        </td>

                        <td className="research-proto-cell research-proto-cell--json">JSON</td>
                        <td className="center mono">{fmt.ms(r.baseline?.timeToFirstTokenMs ?? null)}</td>
                        <td className="center mono">{r.baseline ? fmt.tps(r.baseline.tokensPerSecond) : '—'}</td>
                        <td className="center mono">{r.baseline ? fmt.kb(r.baseline.payloadBytes) : '—'}</td>
                        <td className="center mono">{r.baseline ? fmt.sec(r.baseline.responseTotalMs) : '—'}</td>
                        <td className="center mono">{r.baseline?.totalTokens ?? '—'}</td>
                        <td className="center">—</td>
                      </tr>

                      {/* ── TokenWire row ── */}
                      <tr className={`research-grouped-row research-grouped-row--second ${isLast ? 'research-grouped-row--last-group' : ''}`}>
                        <td className="research-proto-cell research-proto-cell--tokenWire">TokenWire Binary</td>
                        <td className="center mono">{fmt.ms(r.tokenWire?.timeToFirstTokenMs ?? null)}</td>
                        <td className="center mono">{r.tokenWire ? fmt.tps(r.tokenWire.tokensPerSecond) : '—'}</td>
                        <td className="center mono cyan">{r.tokenWire ? fmt.kb(r.tokenWire.payloadBytes) : '—'}</td>
                        <td className="center mono cyan">{r.tokenWire ? fmt.sec(r.tokenWire.responseTotalMs) : '—'}</td>
                        <td className="center mono">{r.tokenWire?.totalTokens ?? '—'}</td>
                        <td className="center">
                          <span className="research-savings-badge">-{savings}%</span>
                        </td>
                      </tr>
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
};
