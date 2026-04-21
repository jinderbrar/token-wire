# TokenWire Colab Benchmark

Self-contained benchmark for comparing **TokenWire** (binary token streaming) vs **Baseline** (JSON text streaming).

## Quick Start

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `tokenwire_benchmark.ipynb`
3. Run all cells

**That's it!** The notebook is completely self-contained:
- All library code is embedded
- Model downloads automatically (~1GB)
- Dictionary builds from model at runtime
- No external files needed

### Features

- **Colab UI Forms** - Sliders and inputs for configuration
- **Collapsible Cells** - Library code hidden by default
- **GPU Support** - Auto-detects and uses T4/A100

## What It Measures

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token (ms) |
| **Bandwidth** | Bytes per token (4 vs ~60) |
| **Win Rate** | % of prompts where TokenWire is faster |

## Configuration Options

In cell 3, you can adjust:
- `model_url` - HuggingFace GGUF model URL
- `model_filename` - Local filename
- `max_tokens` - Tokens per generation (32-256)
- `num_prompts` - Number of test prompts (4-20)
- `cooldown_between_protocols` - Seconds between runs

## Output

- Summary statistics
- TTFT comparison charts
- Bandwidth savings visualization
- Per-prompt results table

---

*TokenWire - Binary token streaming for faster LLM inference*
