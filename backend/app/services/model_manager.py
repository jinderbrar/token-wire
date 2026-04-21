"""
Model Manager for TokenWire Backend

Manages Llama model instances for both TokenWire and Baseline protocols.
Provides instrumented stream generators with detailed timing metrics.

This module maintains two independent Llama instances:
- _baseline_instance: Used by /ws/baseline (JSON token stream)
- _tokenwire_instance: Used by /ws/stream (binary token ID stream)

Each instance has its own KV cache to prevent cross-contamination
between baseline and TokenWire runs.
"""

import json
import os
import gc
import asyncio
import time
from typing import AsyncGenerator, Tuple, Optional
from app.core.config import settings
from app.services.metrics import StreamMetrics, log_metrics, create_metrics
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

MAX_TOKENS = 128


# ─── Model resolution helper ──────────────────────────────────────────────────


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve a model name like 'qwen2.5-coder:1.5b' to its GGUF blob path.

    Args:
        model_name: Model name in Ollama format (e.g., 'qwen2.5-coder:1.5b').

    Returns:
        Full path to the GGUF model blob file.

    Raises:
        RuntimeError: If model blob cannot be found.
    """
    parts = model_name.split(":")
    model_dir = parts[0]
    model_tag = parts[1] if len(parts) > 1 else "latest"

    manifest_path = os.path.expanduser(
        f"~/.ollama/models/manifests/registry.ollama.ai/library/{model_dir}/{model_tag}"
    )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"]
            blob_name = digest.replace(":", "-")
            return os.path.expanduser(f"~/.ollama/models/blobs/{blob_name}")

    raise RuntimeError(f"Could not find model blob for {model_name}")


def _load_llama(model_path: str, label: str) -> Llama:
    """
    Load a Llama model instance.

    Args:
        model_path: Path to the GGUF model file.
        label: Label for logging (e.g., 'BASELINE', 'TOKENWIRE').

    Returns:
        Loaded Llama instance.
    """
    logger.info(f"[{label}] Loading Llama instance from: {model_path}")
    return Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)


# ─── ModelManager ─────────────────────────────────────────────────────────────


class ModelManager:
    """
    Maintains TWO independent Llama instances:
      - _baseline_instance  -> used by /ws/baseline (JSON token stream)
      - _tokenwire_instance -> used by /ws/stream   (binary token ID stream)

    Each instance has its own KV cache, so there is no cross-contamination
    between a baseline run and a TokenWire run. Cache clearing between prompts
    is no longer necessary.
    """

    _baseline_instance: Llama | None = None
    _tokenwire_instance: Llama | None = None
    _current_model_name: str = getattr(settings, "LLM_MODEL_NAME", "qwen2.5-coder:1.5b")

    # ── Discovery ──────────────────────────────────────────────────────────────

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return only allowed models that are also installed in Ollama."""
        allowed = settings.ALLOWED_MODELS

        # Check which allowed models are actually installed
        installed = set()
        manifest_base = os.path.expanduser(
            "~/.ollama/models/manifests/registry.ollama.ai/library"
        )
        if os.path.exists(manifest_base):
            for model_dir in os.listdir(manifest_base):
                tags_dir = os.path.join(manifest_base, model_dir)
                if os.path.isdir(tags_dir):
                    for tag in os.listdir(tags_dir):
                        installed.add(f"{model_dir}:{tag}")

        # Return only allowed models that are installed
        return [m for m in allowed if m in installed]

    @classmethod
    def get_current_model(cls) -> str:
        """Get the currently loaded model name."""
        return cls._current_model_name

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def load_model(cls, model_name: str):
        """
        Load (or reload) BOTH instances with the given model.
        Frees existing instances first to reclaim memory.

        Args:
            model_name: Name of the model to load.

        Raises:
            ValueError: If model is not in the allowed list.
            RuntimeError: If model fails to load.
        """
        # Validate model is in allowed list
        if model_name not in settings.ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not in the allowed list. "
                f"Allowed models: {settings.ALLOWED_MODELS}"
            )

        if (
            cls._baseline_instance is not None
            and cls._tokenwire_instance is not None
            and cls._current_model_name == model_name
        ):
            logger.info(f"Model '{model_name}' already loaded as dual instances - skipping.")
            return

        logger.info(f"Loading dual Llama instances for model: {model_name}")

        # Free existing instances
        cls._free_instances()

        try:
            model_path = _resolve_model_path(model_name)
        except Exception as e:
            logger.error(f"Failed to resolve model path for '{model_name}': {e}")
            raise

        try:
            cls._baseline_instance = _load_llama(model_path, "BASELINE")
            cls._tokenwire_instance = _load_llama(model_path, "TOKENWIRE")
        except Exception as e:
            error_msg = str(e)
            if "key not found" in error_msg or "hyperparameters" in error_msg:
                raise RuntimeError(
                    f"Model '{model_name}' architecture is not supported by this version of llama-cpp-python. "
                    f"Try using a different model like 'qwen2.5-coder:7b'. Error: {error_msg}"
                )
            raise RuntimeError(f"Failed to load model '{model_name}': {error_msg}")

        cls._current_model_name = model_name
        logger.info(f"Dual instances ready for model: {model_name}")

    @classmethod
    def _free_instances(cls):
        """Free model instances and reclaim memory."""
        if cls._baseline_instance is not None:
            del cls._baseline_instance
            cls._baseline_instance = None
        if cls._tokenwire_instance is not None:
            del cls._tokenwire_instance
            cls._tokenwire_instance = None
        gc.collect()

    # ── Getters ────────────────────────────────────────────────────────────────

    @classmethod
    def get_baseline_llm(cls) -> Llama:
        """Get the baseline Llama instance, loading if necessary."""
        if cls._baseline_instance is None:
            cls.load_model(cls._current_model_name)
        return cls._baseline_instance  # type: ignore

    @classmethod
    def get_tokenWire_llm(cls) -> Llama:
        """Get the TokenWire Llama instance, loading if necessary."""
        if cls._tokenwire_instance is None:
            cls.load_model(cls._current_model_name)
        return cls._tokenwire_instance  # type: ignore

    # Keep for backwards compatibility (e.g. any code that calls get_llm())
    @classmethod
    def get_llm(cls) -> Llama:
        """Get default Llama instance (alias for get_baseline_llm)."""
        return cls.get_baseline_llm()

    # ── Cache reset (optional, kept for the HTTP endpoint) ────────────────────

    @classmethod
    def reset_cache(cls):
        """Reset KV caches on both instances independently."""
        if cls._baseline_instance is not None:
            cls._baseline_instance.reset()
            logger.info("Baseline KV cache reset.")
        if cls._tokenwire_instance is not None:
            cls._tokenwire_instance.reset()
            logger.info("TokenWire KV cache reset.")


# ─── Stream generators ────────────────────────────────────────────────────────


async def llama_tokenWire_generator(
    prompt: str,
    collect_metrics: bool = True
) -> AsyncGenerator[list[int], None]:
    """
    Yields batches of raw token IDs from the TOKENWIRE instance.

    Stops on:
      1) model EOS token
      2) MAX_TOKENS generated tokens

    Args:
        prompt: Input prompt string.
        collect_metrics: Whether to collect and log detailed metrics.

    Yields:
        Lists of token IDs (batched for efficiency).
    """
    llm = ModelManager.get_tokenWire_llm()
    eos_id = llm.token_eos()

    # Initialize metrics
    metrics: Optional[StreamMetrics] = None
    if collect_metrics:
        metrics = create_metrics(
            protocol="tokenwire",
            prompt_length=len(prompt),
            model_name=ModelManager.get_current_model()
        )
        metrics.start_prompt()

    count = 0

    try:
        # Tokenization phase
        if metrics:
            metrics.start_tokenization()

        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))

        if metrics:
            metrics.end_tokenization()

        batch: list[int] = []
        is_first = True

        for token_id in llm.generate(prompt_tokens, temp=0.0):
            token_id = int(token_id)

            if token_id == eos_id or count >= MAX_TOKENS:
                break

            # Record first token generation
            if is_first and metrics:
                metrics.record_first_token_generated()

            batch.append(token_id)
            count += 1

            # Flush first token immediately for tight TTFT
            if is_first:
                batch_bytes = len(batch) * 4  # 4 bytes per token ID
                yield batch

                if metrics:
                    metrics.record_first_token_sent()
                    metrics.record_token_batch(len(batch), batch_bytes)

                await asyncio.sleep(0)  # Flush to event loop
                batch = []
                is_first = False

            elif len(batch) >= 4:
                batch_bytes = len(batch) * 4
                yield batch

                if metrics:
                    metrics.record_token_batch(len(batch), batch_bytes)

                batch = []
                await asyncio.sleep(0)

        # Yield remaining tokens
        if batch:
            batch_bytes = len(batch) * 4
            yield batch

            if metrics:
                metrics.record_token_batch(len(batch), batch_bytes)

            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"TokenWire generator error: {e}")
        raise
    finally:
        if metrics:
            metrics.end_stream()
            log_metrics("TOKENWIRE", metrics)

        elapsed = metrics.total_generation_time_ms / 1000 if metrics else 0
        logger.info(f"[TOKENWIRE] Generation finished in {elapsed:.3f}s ({count} tokens)")
        llm.reset()


async def llama_tokenWire_generator_with_metrics(
    prompt: str
) -> AsyncGenerator[Tuple[list[int], Optional[StreamMetrics]], None]:
    """
    Yields batches of raw token IDs along with metrics on final yield.

    This variant provides access to the collected metrics for the caller.
    The metrics object is yielded as the second element of the tuple,
    and is only populated (non-None) on the final yield.

    Args:
        prompt: Input prompt string.

    Yields:
        Tuple of (token_ids, metrics). metrics is None until final yield.
    """
    llm = ModelManager.get_tokenWire_llm()
    eos_id = llm.token_eos()

    metrics = create_metrics(
        protocol="tokenwire",
        prompt_length=len(prompt),
        model_name=ModelManager.get_current_model()
    )
    metrics.start_prompt()

    count = 0

    try:
        metrics.start_tokenization()
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
        metrics.end_tokenization()

        batch: list[int] = []
        is_first = True

        for token_id in llm.generate(prompt_tokens, temp=0.0):
            token_id = int(token_id)

            if token_id == eos_id or count >= MAX_TOKENS:
                break

            if is_first:
                metrics.record_first_token_generated()

            batch.append(token_id)
            count += 1

            if is_first:
                batch_bytes = len(batch) * 4
                yield batch, None

                metrics.record_first_token_sent()
                metrics.record_token_batch(len(batch), batch_bytes)

                await asyncio.sleep(0)
                batch = []
                is_first = False

            elif len(batch) >= 4:
                batch_bytes = len(batch) * 4
                yield batch, None

                metrics.record_token_batch(len(batch), batch_bytes)
                batch = []
                await asyncio.sleep(0)

        # Yield remaining tokens with final metrics
        if batch:
            batch_bytes = len(batch) * 4
            metrics.record_token_batch(len(batch), batch_bytes)
            metrics.end_stream()
            yield batch, metrics
        else:
            metrics.end_stream()
            yield [], metrics

    except Exception as e:
        logger.error(f"TokenWire generator error: {e}")
        raise
    finally:
        log_metrics("TOKENWIRE", metrics)
        elapsed = metrics.total_generation_time_ms / 1000
        logger.info(f"[TOKENWIRE] Generation finished in {elapsed:.3f}s ({count} tokens)")
        llm.reset()


async def llama_baseline_generator(
    prompt: str,
    collect_metrics: bool = True
) -> AsyncGenerator[str, None]:
    """
    Yields decoded text pieces from the BASELINE instance.

    Uses the same stopping conditions as llama_tokenWire_generator:
      1) model EOS token
      2) MAX_TOKENS generated tokens

    Args:
        prompt: Input prompt string.
        collect_metrics: Whether to collect and log detailed metrics.

    Yields:
        Decoded text strings.
    """
    llm = ModelManager.get_baseline_llm()
    eos_id = llm.token_eos()

    # Initialize metrics
    metrics: Optional[StreamMetrics] = None
    if collect_metrics:
        metrics = create_metrics(
            protocol="baseline",
            prompt_length=len(prompt),
            model_name=ModelManager.get_current_model()
        )
        metrics.start_prompt()

    count = 0

    try:
        # Tokenization phase
        if metrics:
            metrics.start_tokenization()

        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))

        if metrics:
            metrics.end_tokenization()

        is_first = True

        for token_id in llm.generate(prompt_tokens, temp=0.0):
            token_id = int(token_id)

            if token_id == eos_id or count >= MAX_TOKENS:
                break

            # Record first token generation
            if is_first and metrics:
                metrics.record_first_token_generated()

            text = llm.detokenize([token_id]).decode("utf-8", errors="ignore")
            count += 1

            if text:
                # Calculate bytes for the JSON message (simulated)
                # In actual baseline, this would be: {"token": "text", "done": false}
                json_overhead = len('{"token":"","done":false}')
                message_bytes = len(text.encode('utf-8')) + json_overhead

                yield text

                if is_first and metrics:
                    metrics.record_first_token_sent()
                    is_first = False

                if metrics:
                    metrics.record_token(message_bytes)

                await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"Baseline generator error: {e}")
        raise
    finally:
        if metrics:
            metrics.end_stream()
            log_metrics("BASELINE", metrics)

        elapsed = metrics.total_generation_time_ms / 1000 if metrics else 0
        logger.info(f"[BASELINE] Generation finished in {elapsed:.3f}s ({count} tokens)")
        llm.reset()


async def llama_baseline_generator_with_metrics(
    prompt: str
) -> AsyncGenerator[Tuple[str, Optional[StreamMetrics]], None]:
    """
    Yields decoded text pieces along with metrics on final yield.

    This variant provides access to the collected metrics for the caller.
    The metrics object is yielded as the second element of the tuple,
    and is only populated (non-None) on the final yield.

    Args:
        prompt: Input prompt string.

    Yields:
        Tuple of (text, metrics). metrics is None until final yield.
    """
    llm = ModelManager.get_baseline_llm()
    eos_id = llm.token_eos()

    metrics = create_metrics(
        protocol="baseline",
        prompt_length=len(prompt),
        model_name=ModelManager.get_current_model()
    )
    metrics.start_prompt()

    count = 0
    last_text = ""

    try:
        metrics.start_tokenization()
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
        metrics.end_tokenization()

        is_first = True

        for token_id in llm.generate(prompt_tokens, temp=0.0):
            token_id = int(token_id)

            if token_id == eos_id or count >= MAX_TOKENS:
                break

            if is_first:
                metrics.record_first_token_generated()

            text = llm.detokenize([token_id]).decode("utf-8", errors="ignore")
            count += 1

            if text:
                json_overhead = len('{"token":"","done":false}')
                message_bytes = len(text.encode('utf-8')) + json_overhead

                last_text = text
                yield text, None

                if is_first:
                    metrics.record_first_token_sent()
                    is_first = False

                metrics.record_token(message_bytes)
                await asyncio.sleep(0)

        # Final yield with metrics
        metrics.end_stream()
        yield "", metrics

    except Exception as e:
        logger.error(f"Baseline generator error: {e}")
        raise
    finally:
        log_metrics("BASELINE", metrics)
        elapsed = metrics.total_generation_time_ms / 1000
        logger.info(f"[BASELINE] Generation finished in {elapsed:.3f}s ({count} tokens)")
        llm.reset()
