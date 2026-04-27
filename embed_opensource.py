"""Embed parsed interview transcripts using open-source HuggingFace models.

Produces identical output files (.npy and _ids.json) to embed_api.py, but
uses local SentenceTransformer models on GPU instead of paid API calls.
Designed to run on Northeastern Discovery HPC cluster.

Usage:
    python embed_opensource.py                              # all models x all conditions
    python embed_opensource.py --model qwen3-0.6b           # one model, all conditions
    python embed_opensource.py --condition C3               # all models, one condition
    python embed_opensource.py --model qwen3-0.6b --condition C3
    python embed_opensource.py --validate-only              # test model loading and exit
    python embed_opensource.py --model qwen3-0.6b --test-run  # 50 candidates only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PARSED_DIR = BASE_DIR / "parsed"
VECTORS_DIR = BASE_DIR / "vectors"

CHECKPOINT_INTERVAL = 100   # save progress every N candidates
MAX_ENCODE_RETRIES = 5
RETRY_WAIT = 5.0            # seconds between retries

ALL_CONDITIONS = ["C1a", "C1b", "C2a", "C2b", "C3", "C4"]

# Segment keys in fixed order for per-skill conditions
SEGMENT_KEYS_NO_INTRO = ["react", "javascript", "html_css"]
SEGMENT_KEYS_WITH_INTRO = ["intro", "react", "javascript", "html_css"]

# Model registry
MODELS = {
    "kalm-12b": {
        "hf_id": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "dim": 3840,
        "token_limit": 8192,
        "batch_size": 2,
        "checkpoint_interval": 50,
        "model_kwargs": {"torch_dtype": torch.bfloat16},
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-Embedding-8B",
        "dim": 4096,
        "token_limit": 8192,
        "batch_size": 4,
        "checkpoint_interval": 50,
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-Embedding-4B",
        "dim": 2560,
        "token_limit": 8192,
        "batch_size": 4,
        "checkpoint_interval": 50,
    },
    "jina-v5-small": {
        "hf_id": "jinaai/jina-embeddings-v5-text-small",
        "dim": 1024,
        "token_limit": 8192,
        "batch_size": 16,
        "checkpoint_interval": 100,
        "model_kwargs": {"default_task": "retrieval"},
    },
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "dim": 1024,
        "token_limit": 8192,
        "batch_size": 16,
        "checkpoint_interval": 100,
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_condition(condition):
    """Load parsed segment data for a given condition.

    Args:
        condition: Condition ID string (e.g. 'C1a', 'C3').

    Returns:
        Tuple of (app_ids, texts_per_candidate, segments_per_candidate).
        - app_ids: list of job_application_id strings.
        - texts_per_candidate: int (3 for C1a/C2a, 4 for C1b/C2b, 1 for C3/C4).
        - segments_per_candidate: list of lists — each inner list has
          `texts_per_candidate` strings in fixed order.
    """
    path = PARSED_DIR / f"segments_{condition}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    app_ids = []
    all_texts = []

    if condition in ("C3", "C4"):
        for rec in data:
            app_ids.append(rec["job_application_id"])
            all_texts.append([rec["text"]])
        return app_ids, 1, all_texts

    if condition in ("C1b", "C2b"):
        seg_keys = SEGMENT_KEYS_WITH_INTRO
    else:
        seg_keys = SEGMENT_KEYS_NO_INTRO

    for rec in data:
        app_ids.append(rec["job_application_id"])
        segs = rec["segments"]
        all_texts.append([segs.get(k, "") for k in seg_keys])

    return app_ids, len(seg_keys), all_texts


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------


def truncate_text(text, token_limit):
    """Truncate text to stay within a model's token limit.

    Uses a conservative word-to-token ratio of 1.5 to estimate token
    count, then truncates by words if necessary.  The 1.5 ratio accounts
    for technical text where subword tokenisation produces more tokens
    per whitespace-delimited word.

    Args:
        text: Input string.
        token_limit: Maximum tokens the model accepts.

    Returns:
        Truncated string (unchanged if already within limit).
    """
    word_limit = int(token_limit / 1.5)
    words = text.split()
    if len(words) <= word_limit:
        return text
    return " ".join(words[:word_limit])


def estimate_tokens(text):
    """Rough token count estimate: word count * 1.5.

    Args:
        text: Input string.

    Returns:
        Estimated token count as int.
    """
    return int(len(text.split()) * 1.5)


def compute_truncation_stats(all_texts, token_limit):
    """Compute truncation statistics for a list of text groups.

    Args:
        all_texts: List of lists of strings (one inner list per candidate).
        token_limit: Model's maximum token count.

    Returns:
        Dict with keys: total, truncated, max_tokens_seen, limit.
    """
    total = 0
    truncated = 0
    max_tokens = 0

    for candidate_texts in all_texts:
        for text in candidate_texts:
            total += 1
            est = estimate_tokens(text)
            max_tokens = max(max_tokens, est)
            if est > token_limit:
                truncated += 1

    return {
        "total_texts": total,
        "truncated": truncated,
        "max_tokens_seen": max_tokens,
        "limit": token_limit,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def checkpoint_path(model_key, condition):
    """Return the checkpoint file path for a model-condition pair.

    Args:
        model_key: Model registry key.
        condition: Condition ID string.

    Returns:
        Path object for the checkpoint file.
    """
    return VECTORS_DIR / f".checkpoint_{model_key}_{condition}.npz"


def load_checkpoint(model_key, condition):
    """Load a checkpoint if it exists.

    Args:
        model_key: Model registry key.
        condition: Condition ID string.

    Returns:
        Tuple of (embeddings_so_far, n_done) where embeddings_so_far is a
        numpy array and n_done is the number of completed candidates.
        Returns (None, 0) if no checkpoint exists.
    """
    cp = checkpoint_path(model_key, condition)
    if not cp.exists():
        return None, 0

    data = np.load(cp)
    embeddings = data["embeddings"]
    n_done = int(data["n_done"])
    log.info("  Resuming from candidate %d (checkpoint found)", n_done)
    return embeddings, n_done


def save_checkpoint(model_key, condition, embeddings, n_done):
    """Save a checkpoint to disk.

    Args:
        model_key: Model registry key.
        condition: Condition ID string.
        embeddings: Numpy array of embeddings computed so far.
        n_done: Number of completed candidates.
    """
    cp = checkpoint_path(model_key, condition)
    np.savez(cp, embeddings=embeddings, n_done=np.array(n_done))


def remove_checkpoint(model_key, condition):
    """Delete a checkpoint file if it exists.

    Args:
        model_key: Model registry key.
        condition: Condition ID string.
    """
    cp = checkpoint_path(model_key, condition)
    if cp.exists():
        cp.unlink()


# ---------------------------------------------------------------------------
# Encoding with retry
# ---------------------------------------------------------------------------


def encode_with_retry(model, texts, batch_size):
    """Encode texts with a SentenceTransformer model, retrying on failure.

    On CUDA OOM errors, frees GPU memory and halves the batch size before
    retrying.  This handles variable-length sequences where a batch of long
    texts can exceed VRAM even though the model itself fits comfortably.

    Args:
        model: Loaded SentenceTransformer instance.
        texts: List of strings to encode.
        batch_size: Initial batch size for model.encode().

    Returns:
        Numpy array of embeddings, shape (len(texts), dim).

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    current_batch_size = batch_size

    for attempt in range(1, MAX_ENCODE_RETRIES + 1):
        try:
            vectors = model.encode(
                texts,
                batch_size=current_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vectors
        except Exception as exc:
            exc_str = str(exc).lower()
            is_oom = "out of memory" in exc_str or "cuda" in exc_str

            if attempt == MAX_ENCODE_RETRIES:
                log.error(
                    "Encode failed after %d retries (batch_size=%d): %s",
                    MAX_ENCODE_RETRIES, current_batch_size, exc,
                )
                raise RuntimeError(
                    f"Encoding failed after {MAX_ENCODE_RETRIES} retries "
                    f"(batch_size={current_batch_size}): {exc}"
                ) from exc

            if is_oom:
                torch.cuda.empty_cache()
                new_batch_size = max(1, current_batch_size // 2)
                log.warning(
                    "CUDA OOM on attempt %d/%d. Freed cache, reducing "
                    "batch_size %d -> %d. Retrying in %.0fs...",
                    attempt, MAX_ENCODE_RETRIES,
                    current_batch_size, new_batch_size, RETRY_WAIT,
                )
                current_batch_size = new_batch_size
            else:
                log.warning(
                    "Encode error on attempt %d/%d (waiting %.0fs): %s",
                    attempt, MAX_ENCODE_RETRIES, RETRY_WAIT, exc,
                )
            time.sleep(RETRY_WAIT)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _validate_output(arr, model_key, condition):
    """Check embedding array for anomalies.

    Args:
        arr: Numpy array of embeddings.
        model_key: Model registry key.
        condition: Condition ID string.
    """
    cfg = MODELS[model_key]
    expected_dim = cfg["dim"]

    # Check dimension
    actual_dim = arr.shape[-1]
    if actual_dim != expected_dim:
        log.error(
            "  DIMENSION MISMATCH: %s/%s expected %d, got %d",
            model_key, condition, expected_dim, actual_dim,
        )

    # Check for NaN and inf values
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0:
        log.error(
            "  NaN VALUES: %s/%s has %d NaN values (%.1f%% of data)",
            model_key, condition, nan_count, nan_count / arr.size * 100,
        )
    if inf_count > 0:
        log.error(
            "  INF VALUES: %s/%s has %d inf values",
            model_key, condition, inf_count,
        )

    # Check for all-zero vectors
    if arr.ndim == 2:
        zero_rows = np.all(arr == 0, axis=1).sum()
    else:
        # (n, segs, dim) — check each segment vector
        reshaped = arr.reshape(-1, actual_dim)
        zero_rows = np.all(reshaped == 0, axis=1).sum()

    if zero_rows > 0:
        log.warning(
            "  ALL-ZERO VECTORS: %s/%s has %d zero vectors",
            model_key, condition, zero_rows,
        )

    if nan_count == 0 and inf_count == 0 and zero_rows == 0:
        log.info("  Validation passed: shape=%s, no issues", arr.shape)


# ---------------------------------------------------------------------------
# Truncation report
# ---------------------------------------------------------------------------


def save_truncation_report(report):
    """Save the truncation report to vectors/truncation_report.json.

    Merges with any existing report to allow incremental runs.

    Args:
        report: Nested dict {model_key: {condition: stats_dict}}.
    """
    report_path = VECTORS_DIR / "truncation_report.json"

    existing = {}
    if report_path.exists():
        with open(report_path) as f:
            existing = json.load(f)

    # Merge new data into existing
    for model_key, conditions in report.items():
        if model_key not in existing:
            existing[model_key] = {}
        existing[model_key].update(conditions)

    with open(report_path, "w") as f:
        json.dump(existing, f, indent=2)

    log.info("Saved truncation report to %s", report_path.name)


# ---------------------------------------------------------------------------
# Core embedding loop
# ---------------------------------------------------------------------------


def embed_condition(model_key, condition, st_model, test_run=False):
    """Embed all candidates for one model-condition pair.

    Flattens per-candidate text segments into a single list, encodes in
    batches with checkpointing, and saves the result as a .npy file.

    Args:
        model_key: Key into MODELS dict (e.g. 'qwen3-0.6b').
        condition: Condition ID string (e.g. 'C3').
        st_model: Pre-loaded SentenceTransformer instance.
        test_run: If True, limit to 50 candidates and use _test suffix.

    Returns:
        Dict with truncation statistics for this model-condition pair.
    """
    cfg = MODELS[model_key]
    dim = cfg["dim"]
    batch_size = cfg["batch_size"]
    checkpoint_interval = cfg.get("checkpoint_interval", CHECKPOINT_INTERVAL)

    log.info("Embedding %s / %s (dim=%d)", model_key, condition, dim)

    # Load data
    app_ids, segs_per_candidate, all_texts = load_condition(condition)

    # Limit for test run
    if test_run:
        limit = min(50, len(app_ids))
        app_ids = app_ids[:limit]
        all_texts = all_texts[:limit]
        log.info("  TEST RUN: limited to %d candidates", limit)

    n_total = len(app_ids)

    # Truncation stats (computed on full data before limiting)
    trunc_stats = compute_truncation_stats(all_texts, cfg["token_limit"])
    if trunc_stats["truncated"] > 0:
        log.warning(
            "  Truncation: %d / %d texts exceed %d token limit (max seen: %d)",
            trunc_stats["truncated"], trunc_stats["total_texts"],
            cfg["token_limit"], trunc_stats["max_tokens_seen"],
        )

    # Output file paths
    suffix = "_test" if test_run else ""
    npy_path = VECTORS_DIR / f"{model_key}_{condition}{suffix}.npy"
    ids_path = VECTORS_DIR / f"{model_key}_{condition}{suffix}_ids.json"

    if npy_path.exists() and ids_path.exists():
        log.info("  Output already exists, skipping: %s", npy_path.name)
        return trunc_stats

    # Load checkpoint or start fresh
    checkpoint_data, n_done = load_checkpoint(model_key, condition)

    if checkpoint_data is not None:
        if segs_per_candidate == 1:
            result = np.zeros((n_total, dim), dtype=np.float32)
        else:
            result = np.zeros(
                (n_total, segs_per_candidate, dim), dtype=np.float32
            )
        result[:n_done] = checkpoint_data[:n_done]
    else:
        if segs_per_candidate == 1:
            result = np.zeros((n_total, dim), dtype=np.float32)
        else:
            result = np.zeros(
                (n_total, segs_per_candidate, dim), dtype=np.float32
            )
        n_done = 0

    # Process candidates in chunks of CHECKPOINT_INTERVAL
    start_time = time.time()

    pbar = tqdm(
        total=n_total,
        initial=n_done,
        desc=f"{model_key}/{condition}",
        unit="cand",
    )

    candidate_idx = n_done
    while candidate_idx < n_total:
        # Collect a checkpoint-interval worth of texts
        chunk_end = min(candidate_idx + checkpoint_interval, n_total)
        chunk_texts = all_texts[candidate_idx:chunk_end]
        chunk_size = chunk_end - candidate_idx

        # Flatten all segment texts for this chunk
        flat_texts = []
        for cand_texts in chunk_texts:
            flat_texts.extend(cand_texts)

        # Truncate texts that exceed the model's token limit
        token_limit = cfg["token_limit"]
        flat_texts = [truncate_text(t, token_limit) for t in flat_texts]

        # Replace empty strings with a single space
        flat_texts = [t if t.strip() else " " for t in flat_texts]

        # Sort by length (longest first) so long texts are batched together
        # rather than mixed with short texts, reducing peak VRAM usage
        sort_indices = sorted(
            range(len(flat_texts)), key=lambda i: len(flat_texts[i]),
            reverse=True,
        )
        sorted_texts = [flat_texts[i] for i in sort_indices]

        # Encode in batches
        sorted_embeddings = encode_with_retry(
            st_model, sorted_texts, batch_size
        )

        # Unsort back to original order
        flat_embeddings = np.empty_like(sorted_embeddings)
        for new_pos, orig_pos in enumerate(sort_indices):
            flat_embeddings[orig_pos] = sorted_embeddings[new_pos]

        # Reshape and store
        if segs_per_candidate == 1:
            result[candidate_idx:chunk_end] = flat_embeddings
        else:
            result[candidate_idx:chunk_end] = flat_embeddings.reshape(
                chunk_size, segs_per_candidate, dim
            )

        candidate_idx = chunk_end
        pbar.update(chunk_size)

        # Free GPU memory between chunks
        torch.cuda.empty_cache()

        # Checkpoint
        if candidate_idx < n_total:
            save_checkpoint(model_key, condition, result, candidate_idx)

        # Log ETA
        elapsed = time.time() - start_time
        if candidate_idx > n_done and candidate_idx < n_total:
            rate = (candidate_idx - n_done) / elapsed
            remaining = (n_total - candidate_idx) / rate
            log.info(
                "  %d/%d done (%.1f cand/s, ~%.0fs remaining)",
                candidate_idx, n_total, rate, remaining,
            )

    pbar.close()

    # Save final outputs
    np.save(npy_path, result)
    with open(ids_path, "w") as f:
        json.dump(app_ids, f)
    remove_checkpoint(model_key, condition)

    elapsed = time.time() - start_time
    log.info(
        "  Saved %s  shape=%s  dtype=%s  (%.1fs)",
        npy_path.name, result.shape, result.dtype, elapsed,
    )

    # Validate
    _validate_output(result, model_key, condition)

    return trunc_stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_model(model_key, st_model=None):
    """Test that a model can be loaded and produces correct output shape.

    Args:
        model_key: Key into MODELS dict.
        st_model: Optional pre-loaded SentenceTransformer instance.

    Returns:
        True if validation passes, False otherwise.
    """
    cfg = MODELS[model_key]
    log.info("  Validating %s (%s)...", model_key, cfg["hf_id"])

    try:
        if st_model is None:
            t0 = time.time()
            load_kwargs = {"torch_dtype": torch.float16}
            load_kwargs.update(cfg.get("model_kwargs", {}))
            st_model = SentenceTransformer(
                cfg["hf_id"], trust_remote_code=True, device="cuda",
                model_kwargs=load_kwargs,
            )
            load_time = time.time() - t0
        else:
            load_time = 0.0

        vecs = st_model.encode(
            ["validation test"],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        actual_shape = vecs.shape
        expected_shape = (1, cfg["dim"])

        if actual_shape == expected_shape:
            log.info(
                "  PASS  %-16s  dim=%d  loaded in %.1fs",
                model_key, cfg["dim"], load_time,
            )
            return True
        else:
            log.error(
                "  FAIL  %-16s  expected shape %s, got %s",
                model_key, expected_shape, actual_shape,
            )
            return False
    except Exception as exc:
        log.error("  FAIL  %-16s  %s", model_key, exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with model, condition, validate_only,
        and test_run fields.
    """
    parser = argparse.ArgumentParser(
        description="Embed transcripts with open-source HuggingFace models."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Run a single model (default: all).",
    )
    parser.add_argument(
        "--condition",
        choices=ALL_CONDITIONS,
        default=None,
        help="Run a single condition (default: all).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Test model loading and exit without embedding.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Limit to 50 candidates; save with _test suffix.",
    )
    return parser.parse_args()


def main():
    """Entry point for the open-source embedding pipeline."""
    # Set HF cache paths BEFORE any HuggingFace imports
    os.environ["HF_HOME"] = "/projects/PalikotLab/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/projects/PalikotLab/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/projects/PalikotLab/hf_cache"

    args = parse_args()
    os.makedirs(VECTORS_DIR, exist_ok=True)

    # Determine which models and conditions to run
    model_keys = [args.model] if args.model else list(MODELS.keys())
    conditions = [args.condition] if args.condition else ALL_CONDITIONS

    # Validate-only mode: test a single model
    if args.validate_only:
        test_key = args.model if args.model else "qwen3-0.6b"
        log.info("Validating model: %s", test_key)
        ok = validate_model(test_key)
        if not ok:
            sys.exit(1)
        return

    # Run embedding pipeline
    truncation_report = {}
    total_combos = len(model_keys) * len(conditions)
    combo_num = 0

    for model_key in model_keys:
        truncation_report[model_key] = {}
        cfg = MODELS[model_key]
        log.info("Loading model: %s", cfg["hf_id"])
        load_kwargs = {"torch_dtype": torch.float16}
        load_kwargs.update(cfg.get("model_kwargs", {}))
        st_model = SentenceTransformer(
            cfg["hf_id"], trust_remote_code=True, device="cuda",
            model_kwargs=load_kwargs,
        )
        log.info("Model loaded.")
        for condition in conditions:
            combo_num += 1
            log.info(
                "=== [%d/%d] %s x %s ===",
                combo_num, total_combos, model_key, condition,
            )
            stats = embed_condition(
                model_key, condition, st_model, test_run=args.test_run
            )
            truncation_report[model_key][condition] = stats

    # Save truncation report
    save_truncation_report(truncation_report)

    log.info("Done. All embeddings saved to %s", VECTORS_DIR)


if __name__ == "__main__":
    main()
