# TokenWire

**A lightweight binary token streaming prototype for local LLM inference.**

TokenWire streams raw token IDs over WebSockets and reconstructs text on the client using a pre-generated binary dictionary. It is designed to compare compact binary token transport against traditional JSON-based streaming.

[Read the research paper draft](./Binary_Token_Transmission_with_Client_Side_Detokenization_for_Language_Model_Inference.pdf)

---

## What it does

TokenWire explores a simple idea:

> Instead of sending decoded text chunks inside JSON payloads, send token IDs directly and let the client reconstruct the text.

This can reduce repeated network overhead and make token-level streaming easier to benchmark.

## Core features

- Binary token streaming over WebSockets
- Client-side token reconstruction
- Baseline JSON streaming for comparison
- Local LLM support through Ollama / llama.cpp
- Benchmark tooling for transport-level experiments
- React frontend for interactive testing

## How it works

```text
Local LLM
   ↓
Token IDs
   ↓
Binary WebSocket stream
   ↓
Client-side dictionary lookup
   ↓
Rendered text

```

## Quickstart

```bash
# Setup
python setup.py
# Run Backend
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
# Run Frontend
cd frontend && npm install && npm run dev

# Dictionary setup for particular model
# Saved to `frontend/public/dictionaries/`
python scripts/extract_vocab.py --model qwen2.5-coder:7b

# Bechmarking
cd scripts/node_benchmark && npm install && npx tsx cli.ts --dataset sample_1
```

---

## Research & Feedback
TokenWire is independent research into lower-overhead LLM streaming and client-side detokenization.If this work is useful to your own experiments, please consider starring the repo, sharing feedback, or opening an issue with benchmark results, questions, or replication notes. It helps improve the research and makes the project easier for others to discover.
