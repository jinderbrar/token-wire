import struct
import os
import argparse
import array
import sys

# Mapping from Ollama model names to HuggingFace tokenizer names
OLLAMA_TO_HUGGINGFACE = {
    # Qwen models
    "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    # Gemma models
    "gemma3:1b": "google/gemma-3-1b-it",
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3:12b": "google/gemma-3-12b-it",
    "gemma:2b": "google/gemma-2b-it",
    "gemma:7b": "google/gemma-7b-it",
    # Llama models
    "llama3.2:1b": "meta-llama/Llama-3.2-1B",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    # Mistral
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    # CodeLlama
    "codellama:7b": "codellama/CodeLlama-7b-hf",
}


def normalize_model_name(ollama_name: str) -> str:
    """Convert 'gemma3:4b' to 'gemma3_4b' for filesystem safety."""
    return ollama_name.replace(":", "_").replace("/", "_").replace(".", "_")


def get_huggingface_name(ollama_name: str) -> str:
    """Get HuggingFace tokenizer name from Ollama model name."""
    if ollama_name in OLLAMA_TO_HUGGINGFACE:
        return OLLAMA_TO_HUGGINGFACE[ollama_name]
    # If not in mapping, assume it's already a HuggingFace name
    print(f"Warning: '{ollama_name}' not in mapping, using as-is")
    return ollama_name


def extract_vocabulary(hf_model_name: str):
    """Extract vocabulary from HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
        print(f"Downloading/loading tokenizer for {hf_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        vocab = tokenizer.get_vocab()

        id_to_bytes = {}
        for raw_token, t_id in vocab.items():
            try:
                txt = tokenizer.convert_tokens_to_string([raw_token])
                b = txt.encode("utf-8")
            except Exception:
                b = raw_token.encode("utf-8")
            id_to_bytes[t_id] = b

        return id_to_bytes
    except Exception as e:
        print(f"Error loading tokenizer {hf_model_name}: {e}")
        raise


def build_dictionary_binary(vocab: dict) -> bytes:
    """Build binary dictionary from vocabulary dict."""
    max_token_id = max(vocab.keys())
    print(f"Max token ID: {max_token_id}")

    header = struct.pack("<4sI", b"TWRE", max_token_id)  # TokenWire format

    offsets = []
    string_blocks = []
    current_offset = 0

    for i in range(max_token_id + 1):
        offsets.append(current_offset)
        b = vocab.get(i, b"")
        string_blocks.append(b)
        current_offset += len(b)

    offsets.append(current_offset)

    offset_array = array.array("I", offsets)
    if offset_array.itemsize == 4 and sys.byteorder == 'big':
        offset_array.byteswap()
    offset_bytes = offset_array.tobytes()

    string_bytes = b"".join(string_blocks)

    return header + offset_bytes + string_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Extract tokenizer vocabulary and generate binary dictionary for TokenWire"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen2.5-coder:7b",
        help="Ollama model name (e.g., 'gemma3:4b', 'qwen2.5-coder:7b')"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Override HuggingFace model name directly (e.g., 'google/gemma-3-4b-it')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: frontend/public/dictionaries)"
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Also output to colab/dictionaries/"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported model mappings"
    )
    args = parser.parse_args()

    if args.list_models:
        print("Supported Ollama -> HuggingFace mappings:")
        for ollama, hf in sorted(OLLAMA_TO_HUGGINGFACE.items()):
            print(f"  {ollama:25} -> {hf}")
        return

    # Determine HuggingFace model name
    if args.hf_model:
        hf_model_name = args.hf_model
        normalized_name = normalize_model_name(args.model)
    else:
        hf_model_name = get_huggingface_name(args.model)
        normalized_name = normalize_model_name(args.model)

    # Determine output directory
    if args.output_dir:
        target_dir = args.output_dir
    else:
        target_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "dictionaries")

    os.makedirs(target_dir, exist_ok=True)

    # Extract vocabulary
    print(f"Extracting vocabulary for: {args.model}")
    print(f"Using HuggingFace tokenizer: {hf_model_name}")
    vocab = extract_vocabulary(hf_model_name)

    # Build binary
    binary_data = build_dictionary_binary(vocab)

    # Write output
    output_path = os.path.join(target_dir, f"{normalized_name}.bin")
    print(f"Writing dictionary to: {output_path}")

    with open(output_path, "wb") as f:
        f.write(binary_data)

    # Also write to colab if requested
    if args.colab:
        colab_dir = os.path.join(os.path.dirname(__file__), "..", "colab", "dictionaries")
        os.makedirs(colab_dir, exist_ok=True)
        colab_path = os.path.join(colab_dir, f"{normalized_name}.bin")
        print(f"Writing dictionary to: {colab_path}")
        with open(colab_path, "wb") as f:
            f.write(binary_data)

    print(f"Done! Dictionary size: {len(binary_data)} bytes")
    print(f"\nTo use this dictionary, select model '{args.model}' in the frontend.")


if __name__ == "__main__":
    main()
