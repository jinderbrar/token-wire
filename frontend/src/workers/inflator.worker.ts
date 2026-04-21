/**
 * TokenWire Inflator Worker
 *
 * Web Worker that handles:
 * - Dictionary loading and management
 * - WebSocket connection to TokenWire server
 * - Token ID to text decoding via O(1) dictionary lookup
 * - Performance metrics and instrumentation
 *
 * Instrumentation includes:
 * - Dictionary lookup time per token
 * - Total decode time
 * - Memory usage tracking
 * - Per-batch timing statistics
 */

// ─── Flat Binary Memory Maps ─────────────────────────────────────────────────

let offsets: Uint32Array | null = null;
let stringBlock: Uint8Array | null = null;
let maxTokenId = 0;
const textDecoder = new TextDecoder("utf-8");

// ─── Telemetry State ─────────────────────────────────────────────────────────

let tokensReceived = 0;
let startTime = 0;
let ttft: number | null = null;
let lastBytesArray: number = 0;

let socket: WebSocket | null = null;
let isHeadless = false;
let lastKnownStats: StreamStats | null = null;

// Session ID to prevent stale socket STREAM_DONE from racing with new prompts
let currentSessionId = 0;

// Current loaded model
let currentModel: string | null = null;

// ─── Instrumentation State ───────────────────────────────────────────────────

interface DecodeTiming {
  lookupTimeNs: number;      // Time spent in dictionary lookup (nanoseconds)
  decodeTimeNs: number;      // Time spent in text decoding (nanoseconds)
  tokenCount: number;        // Number of tokens in this batch
}

interface InstrumentationStats {
  totalLookupTimeMs: number;      // Total dictionary lookup time (ms)
  totalDecodeTimeMs: number;      // Total text decoding time (ms)
  avgLookupTimePerTokenUs: number; // Average lookup time per token (microseconds)
  avgDecodeTimePerTokenUs: number; // Average decode time per token (microseconds)
  batchTimings: DecodeTiming[];   // Per-batch timing data
  memoryUsageMB: number | null;   // Memory usage if available (MB)
  dictionarySizeBytes: number;    // Size of loaded dictionary
}

interface StreamStats {
  binaryBytes: number;
  tps: number;
  tokens: number;
  ttft: number | null;
  totalMs: number;
  instrumentation?: InstrumentationStats;
}

// Instrumentation accumulators
let instrumentationEnabled = true;
let totalLookupTimeNs = 0;
let totalDecodeTimeNs = 0;
let batchTimings: DecodeTiming[] = [];
let dictionarySizeBytes = 0;

// High-resolution timing helper (uses performance.now() which returns ms with sub-ms precision)
function getHighResTime(): number {
  return performance.now();
}

// Reset instrumentation for new stream
function resetInstrumentation(): void {
  totalLookupTimeNs = 0;
  totalDecodeTimeNs = 0;
  batchTimings = [];
}

// Get memory usage if available
function getMemoryUsage(): number | null {
  // Check for memory API (available in some browsers)
  if ((performance as any).memory) {
    const memory = (performance as any).memory;
    return memory.usedJSHeapSize / (1024 * 1024); // Convert to MB
  }
  return null;
}

// Build instrumentation stats
function buildInstrumentationStats(): InstrumentationStats {
  const totalLookupTimeMs = totalLookupTimeNs / 1_000_000;
  const totalDecodeTimeMs = totalDecodeTimeNs / 1_000_000;
  const totalTokens = tokensReceived || 1;

  return {
    totalLookupTimeMs,
    totalDecodeTimeMs,
    avgLookupTimePerTokenUs: (totalLookupTimeNs / totalTokens) / 1000,
    avgDecodeTimePerTokenUs: (totalDecodeTimeNs / totalTokens) / 1000,
    batchTimings: batchTimings.slice(), // Copy array
    memoryUsageMB: getMemoryUsage(),
    dictionarySizeBytes,
  };
}

// ─── Dictionary Management ───────────────────────────────────────────────────

function normalizeModelName(modelName: string): string {
  // Convert 'gemma3:4b' to 'gemma3_4b' for filesystem safety
  return modelName.replace(/:/g, '_').replace(/\//g, '_').replace(/\./g, '_');
}

async function loadDictionary(model?: string) {
  const modelName = model || 'qwen2.5-coder:1.5b';
  const normalizedName = normalizeModelName(modelName);

  console.log(`Fetching dictionary for model: ${modelName} (${normalizedName}.bin)`);

  try {
    // Try model-specific dictionary first
    let res = await fetch(`/dictionaries/${normalizedName}.bin`);

    // Fallback to legacy dictionary.bin if model-specific not found
    if (!res.ok) {
      console.log(`Model-specific dictionary not found, trying legacy dictionary.bin`);
      res = await fetch('/dictionary.bin');
    }

    if (!res.ok) {
      throw new Error(`No dictionary found for model: ${modelName}`);
    }

    const buffer = await res.arrayBuffer();
    dictionarySizeBytes = buffer.byteLength;

    // Check Header (Magic number + MaxTokenId)
    const magic = textDecoder.decode(new Uint8Array(buffer, 0, 4));
    if (magic !== 'TWRE') throw new Error("Invalid TokenWire dictionary magic: " + magic);

    const view = new DataView(buffer);
    maxTokenId = view.getUint32(4, true); // true = little-endian

    const offsetCount = maxTokenId + 2;
    const offsetsByteLength = offsetCount * 4;

    // 8 bytes in: safely aligned to 4-byte boundaries for Uint32Array
    offsets = new Uint32Array(buffer, 8, offsetCount);

    const stringsStart = 8 + offsetsByteLength;
    stringBlock = new Uint8Array(buffer, stringsStart);

    currentModel = modelName;
    console.log(`Successfully loaded dictionary for ${modelName}. Max Token ID: ${maxTokenId}, Size: ${(dictionarySizeBytes / 1024).toFixed(2)} KB`);
    postMessage({
      type: 'INITIALIZED',
      model: modelName,
      dictionaryAvailable: true,
      dictionaryStats: {
        maxTokenId,
        sizeBytes: dictionarySizeBytes,
        sizeKB: dictionarySizeBytes / 1024,
      }
    });
  } catch (err) {
    console.error("Dictionary load failed:", err);
    // Clear dictionary state
    offsets = null;
    stringBlock = null;
    maxTokenId = 0;
    currentModel = null;
    dictionarySizeBytes = 0;
    postMessage({ type: 'INITIALIZED', model: modelName, dictionaryAvailable: false, error: String(err) });
  }
}

// ─── Token Decoding with Instrumentation ─────────────────────────────────────

/**
 * Decode a single token ID to text using the dictionary.
 * Includes timing instrumentation.
 *
 * @param tokenId - The token ID to decode
 * @returns Decoded text string, or placeholder if not found
 */
function decodeToken(tokenId: number): { text: string; lookupTimeNs: number; decodeTimeNs: number } {
  const lookupStart = getHighResTime();

  // Dictionary lookup (O(1) via offset table)
  if (!offsets || !stringBlock || tokenId > maxTokenId) {
    const lookupEnd = getHighResTime();
    return {
      text: ` <${tokenId}> `,
      lookupTimeNs: (lookupEnd - lookupStart) * 1_000_000,
      decodeTimeNs: 0,
    };
  }

  const start = offsets[tokenId];
  const end = offsets[tokenId + 1];

  const lookupEnd = getHighResTime();
  const lookupTimeNs = (lookupEnd - lookupStart) * 1_000_000;

  if (start >= end) {
    return { text: '', lookupTimeNs, decodeTimeNs: 0 };
  }

  // Text decoding
  const decodeStart = getHighResTime();
  const slice = stringBlock.subarray(start, end);
  const text = textDecoder.decode(slice, { stream: true });
  const decodeEnd = getHighResTime();
  const decodeTimeNs = (decodeEnd - decodeStart) * 1_000_000;

  return { text, lookupTimeNs, decodeTimeNs };
}

/**
 * Decode a batch of token IDs from a binary buffer.
 * Includes timing instrumentation for the entire batch.
 *
 * @param buffer - ArrayBuffer containing 4-byte little-endian token IDs
 * @returns Decoded text and timing information
 */
function decodeBatch(buffer: ArrayBuffer): { text: string; timing: DecodeTiming } {
  const view = new DataView(buffer);
  const tokenCount = buffer.byteLength / 4;

  let finalRenderStr = "";
  let batchLookupTimeNs = 0;
  let batchDecodeTimeNs = 0;

  for (let i = 0; i < buffer.byteLength; i += 4) {
    const currentTokenId = view.getUint32(i, true);

    if (!isHeadless) {
      const result = decodeToken(currentTokenId);
      finalRenderStr += result.text;
      batchLookupTimeNs += result.lookupTimeNs;
      batchDecodeTimeNs += result.decodeTimeNs;
    }
  }

  const timing: DecodeTiming = {
    lookupTimeNs: batchLookupTimeNs,
    decodeTimeNs: batchDecodeTimeNs,
    tokenCount,
  };

  return { text: finalRenderStr, timing };
}

// ─── WebSocket Connection ────────────────────────────────────────────────────

function connectWebSocket(prompt: string) {
  // Capture session ID in closure - stale sockets skip their callbacks
  const mySessionId = ++currentSessionId;

  if (socket) socket.close();

  // Reset instrumentation for new stream
  resetInstrumentation();

  socket = new WebSocket('ws://localhost:8000/ws/stream');
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    // Only proceed if this is still the current session
    if (mySessionId !== currentSessionId) return;

    startTime = performance.now();
    tokensReceived = 0;
    lastBytesArray = 0;
    ttft = null;

    // Send control message natively as JSON (Text Frame)
    socket!.send(JSON.stringify({
      signal: "START_STREAM",
      prompt: prompt
    }));
  };

  socket.onmessage = async (e) => {
    const now = performance.now();
    if (ttft === null) ttft = now - startTime;

    if (mySessionId !== currentSessionId) return; // stale - discard
    const buffer = e.data;

    // Decode batch with instrumentation
    const { text: finalRenderStr, timing } = decodeBatch(buffer);

    // Accumulate instrumentation data
    if (instrumentationEnabled) {
      totalLookupTimeNs += timing.lookupTimeNs;
      totalDecodeTimeNs += timing.decodeTimeNs;
      batchTimings.push(timing);
    }

    tokensReceived += timing.tokenCount;

    const binarySize = buffer.byteLength;
    lastBytesArray += binarySize;

    const tps = tokensReceived / ((now - startTime) / 1000);

    // Build stats with instrumentation
    lastKnownStats = {
      binaryBytes: lastBytesArray,
      tps: tps,
      tokens: tokensReceived,
      ttft: ttft,
      totalMs: now - startTime,
      instrumentation: instrumentationEnabled ? buildInstrumentationStats() : undefined,
    };

    if (!isHeadless) {
      postMessage({
        type: 'TOKEN_DECODED',
        text: finalRenderStr,
        stats: lastKnownStats
      });
    }
  };

  socket.onclose = () => {
    if (mySessionId !== currentSessionId) return; // stale socket

    // Final stats with complete instrumentation
    if (lastKnownStats && instrumentationEnabled) {
      lastKnownStats.instrumentation = buildInstrumentationStats();
    }

    postMessage({ type: 'STREAM_DONE', stats: lastKnownStats });
  };

  socket.onerror = (e) => {
    if (mySessionId !== currentSessionId) return;
    console.error("TokenWire socket error", e);
    postMessage({ type: 'STREAM_DONE', stats: lastKnownStats });
  };
}

// ─── Message Handler ─────────────────────────────────────────────────────────

self.onmessage = async (e) => {
  switch (e.data.type) {
    case 'INIT_DICTIONARY':
      await loadDictionary(e.data.model);
      break;

    case 'RELOAD_DICTIONARY':
      await loadDictionary(e.data.model);
      break;

    case 'START_STREAM':
      isHeadless = e.data.headless || false;
      instrumentationEnabled = e.data.instrumentation !== false; // Default to true
      connectWebSocket(e.data.prompt);
      break;

    case 'GET_MEMORY_USAGE':
      postMessage({
        type: 'MEMORY_USAGE',
        memoryMB: getMemoryUsage(),
        dictionarySizeBytes,
      });
      break;

    case 'GET_INSTRUMENTATION':
      postMessage({
        type: 'INSTRUMENTATION_STATS',
        stats: buildInstrumentationStats(),
      });
      break;

    case 'ENABLE_INSTRUMENTATION':
      instrumentationEnabled = true;
      break;

    case 'DISABLE_INSTRUMENTATION':
      instrumentationEnabled = false;
      break;

    default:
      console.warn('Unknown message type:', e.data.type);
  }
};

// ─── Type Exports (for TypeScript consumers) ─────────────────────────────────

export type { DecodeTiming, InstrumentationStats, StreamStats };
