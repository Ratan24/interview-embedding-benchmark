"""Embed parsed interview transcripts using paid API models.

Supports 5 embedding models across 4 providers (Google, OpenAI, Voyage,
Cohere).  Each model is run against all 6 parsing conditions, producing
.npy embedding arrays and metadata in vectors/.

Usage:
    python embed_api.py                             # all models x all conditions
    python embed_api.py --model gemini              # one model, all conditions
    python embed_api.py --condition C3              # all models, one condition
    python embed_api.py --model gemini --condition C3
    python embed_api.py --validate-only             # test API keys and exit
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PARSED_DIR = BASE_DIR / "parsed"
VECTORS_DIR = BASE_DIR / "vectors"

CHECKPOINT_INTERVAL = 100   # save progress every N candidates
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0       # seconds

ALL_CONDITIONS = ["C1a", "C1b", "C2a", "C2b", "C3", "C4"]

# Segment keys in fixed order for per-skill conditions
SEGMENT_KEYS_NO_INTRO = ["react", "javascript", "html_css"]
SEGMENT_KEYS_WITH_INTRO = ["intro", "react", "javascript", "html_css"]

# Model registry
MODELS = {
    "gemini": {
        "model_id": "gemini-embedding-001",
        "provider": "google",
        "dim": 3072,
        "token_limit": 2048,
        "batch_size": 30,
        "env_key": "GOOGLE_API_KEY",
    },
    "openai-large": {
        "model_id": "text-embedding-3-large",
        "provider": "openai",
        "dim": 3072,
        "token_limit": 8191,
        "batch_size": 100,
        "env_key": "OPENAI_API_KEY",
    },
    "openai-small": {
        "model_id": "text-embedding-3-small",
        "provider": "openai",
        "dim": 1536,
        "token_limit": 8191,
        "batch_size": 100,
        "env_key": "OPENAI_API_KEY",
    },
    "voyage": {
        "model_id": "voyage-3.5",
        "provider": "voyage",
        "dim": 1024,
        "token_limit": 32000,
        "batch_size": 128,
        "env_key": "VOYAGE_API_KEY",
    },
    "cohere": {
        "model_id": "embed-v4.0",
        "provider": "cohere",
        "dim": 1536,
        "token_limit": 128000,
        "batch_size": 96,
        "env_key": "COHERE_API_KEY",
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
# Provider clients (lazy-initialised)
# ---------------------------------------------------------------------------

_clients = {}


def _get_client(provider):
    """Return a cached API client for the given provider.

    Args:
        provider: One of 'google', 'openai', 'voyage', 'cohere'.

    Returns:
        Initialised API client object.
    """
    if provider in _clients:
        return _clients[provider]

    if provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "voyage":
        import voyageai
        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    elif provider == "cohere":
        import cohere
        client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider}")

    _clients[provider] = client
    return client


# ---------------------------------------------------------------------------
# Embedding dispatch
# ---------------------------------------------------------------------------


def _embed_batch_google(client, texts, model_id):
    """Embed a batch of texts with Google GenAI.

    Args:
        client: google.genai.Client instance.
        texts: List of strings to embed.
        model_id: Model identifier string.

    Returns:
        List of embedding vectors (list of floats each).
    """
    resp = client.models.embed_content(
        model=model_id,
        contents=texts,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )
    return [e.values for e in resp.embeddings]


def _embed_batch_openai(client, texts, model_id):
    """Embed a batch of texts with OpenAI.

    Args:
        client: openai.OpenAI instance.
        texts: List of strings to embed.
        model_id: Model identifier string.

    Returns:
        List of embedding vectors (list of floats each).
    """
    resp = client.embeddings.create(input=texts, model=model_id)
    # Response items have an .index field; sort by it to guarantee order
    sorted_data = sorted(resp.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def _embed_batch_voyage(client, texts, model_id):
    """Embed a batch of texts with Voyage AI.

    Args:
        client: voyageai.Client instance.
        texts: List of strings to embed.
        model_id: Model identifier string.

    Returns:
        List of embedding vectors (list of floats each).
    """
    result = client.embed(
        texts=texts,
        model=model_id,
        input_type="document",
    )
    return result.embeddings


def _embed_batch_cohere(client, texts, model_id):
    """Embed a batch of texts with Cohere.

    Args:
        client: cohere.Client instance.
        texts: List of strings to embed.
        model_id: Model identifier string.

    Returns:
        List of embedding vectors (list of floats each).
    """
    resp = client.embed(
        texts=texts,
        model=model_id,
        input_type="search_document",
        embedding_types=["float"],
    )
    return [list(v) for v in resp.embeddings.float_]


_EMBED_FNS = {
    "google": _embed_batch_google,
    "openai": _embed_batch_openai,
    "voyage": _embed_batch_voyage,
    "cohere": _embed_batch_cohere,
}


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


def _is_retryable(exc):
    """Check whether an API exception is transient and worth retrying.

    Rate-limit (429) and server errors (5xx) are retryable.
    Client errors like 400 (bad request) are not.

    Args:
        exc: The caught exception.

    Returns:
        True if the request should be retried.
    """
    exc_str = str(exc)
    # Check for HTTP status codes in common exception formats
    if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
        return True
    if any(f"{c}" in exc_str for c in range(500, 600)):
        return True
    # Connection / timeout errors are retryable
    for keyword in ("timeout", "connection", "Timeout", "Connection"):
        if keyword in exc_str:
            return True
    # OpenAI sometimes returns spurious 400 errors about context length
    # even when texts are within limits.  Since we pre-truncate all texts,
    # retry these — a genuinely over-limit text would fail all attempts.
    if "maximum context length" in exc_str:
        return True
    return False


def embed_batch_with_retry(model_key, texts):
    """Embed a batch of texts with retries and exponential backoff.

    Only retries on transient errors (rate limits, server errors).
    Permanent errors (400 bad request) raise immediately.

    Args:
        model_key: Key into MODELS dict (e.g. 'gemini').
        texts: List of strings to embed.

    Returns:
        List of embedding vectors.

    Raises:
        RuntimeError: If all retries are exhausted or a permanent error occurs.
    """
    cfg = MODELS[model_key]
    client = _get_client(cfg["provider"])
    embed_fn = _EMBED_FNS[cfg["provider"]]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return embed_fn(client, texts, cfg["model_id"])
        except Exception as exc:
            if not _is_retryable(exc):
                log.error(
                    "Permanent error for %s (not retrying): %s",
                    model_key, exc,
                )
                raise RuntimeError(
                    f"Embedding failed for {model_key} "
                    f"(permanent error): {exc}"
                ) from exc

            wait = INITIAL_BACKOFF * (2 ** (attempt - 1))
            if attempt == MAX_RETRIES:
                log.error(
                    "Failed after %d retries for %s: %s",
                    MAX_RETRIES, model_key, exc,
                )
                raise RuntimeError(
                    f"Embedding failed for {model_key} after {MAX_RETRIES} "
                    f"retries: {exc}"
                ) from exc
            log.warning(
                "Retry %d/%d for %s (waiting %.1fs): %s",
                attempt, MAX_RETRIES, model_key, wait, exc,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def validate_keys(model_keys):
    """Test each required API key with a minimal embedding call.

    Args:
        model_keys: List of model keys to validate.

    Returns:
        True if all keys are valid, False otherwise.
    """
    # Determine unique providers needed
    providers_needed = {}
    for mk in model_keys:
        cfg = MODELS[mk]
        p = cfg["provider"]
        if p not in providers_needed:
            providers_needed[p] = mk

    all_ok = True
    for provider, first_model_key in sorted(providers_needed.items()):
        cfg = MODELS[first_model_key]
        env_var = cfg["env_key"]
        key_val = os.environ.get(env_var, "")

        if not key_val:
            log.error("  FAIL  %-12s  %s is not set", provider, env_var)
            all_ok = False
            continue

        try:
            vecs = embed_batch_with_retry(first_model_key, ["hello world"])
            dim = len(vecs[0])
            log.info(
                "  PASS  %-12s  %s  dim=%d",
                provider, cfg["model_id"], dim,
            )
        except Exception as exc:
            log.error("  FAIL  %-12s  %s  %s", provider, env_var, exc)
            all_ok = False

    return all_ok


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
# Truncation tracking
# ---------------------------------------------------------------------------


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
# Core embedding loop
# ---------------------------------------------------------------------------


def embed_condition(model_key, condition):
    """Embed all candidates for one model-condition pair.

    Flattens per-candidate text segments into a single list, embeds in
    batches with checkpointing, and saves the result as a .npy file.

    Args:
        model_key: Key into MODELS dict (e.g. 'gemini').
        condition: Condition ID string (e.g. 'C3').

    Returns:
        Dict with truncation statistics for this model-condition pair.
    """
    cfg = MODELS[model_key]
    dim = cfg["dim"]
    batch_size = cfg["batch_size"]

    log.info("Embedding %s / %s (dim=%d)", model_key, condition, dim)

    # Load data
    app_ids, segs_per_candidate, all_texts = load_condition(condition)
    n_total = len(app_ids)

    # Truncation stats
    trunc_stats = compute_truncation_stats(all_texts, cfg["token_limit"])
    if trunc_stats["truncated"] > 0:
        log.warning(
            "  Truncation: %d / %d texts exceed %d token limit (max seen: %d)",
            trunc_stats["truncated"], trunc_stats["total_texts"],
            cfg["token_limit"], trunc_stats["max_tokens_seen"],
        )

    # Check for existing final output
    npy_path = VECTORS_DIR / f"{model_key}_{condition}.npy"
    ids_path = VECTORS_DIR / f"{model_key}_{condition}_ids.json"
    if npy_path.exists() and ids_path.exists():
        log.info("  Output already exists, skipping: %s", npy_path.name)
        return trunc_stats

    # Load checkpoint or start fresh
    checkpoint_data, n_done = load_checkpoint(model_key, condition)

    if checkpoint_data is not None:
        # Pre-allocate full array and copy checkpoint data
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
        chunk_end = min(candidate_idx + CHECKPOINT_INTERVAL, n_total)
        chunk_texts = all_texts[candidate_idx:chunk_end]
        chunk_size = chunk_end - candidate_idx

        # Flatten all segment texts for this chunk
        flat_texts = []
        for cand_texts in chunk_texts:
            flat_texts.extend(cand_texts)

        # Truncate texts that exceed the model's token limit
        token_limit = cfg["token_limit"]
        flat_texts = [truncate_text(t, token_limit) for t in flat_texts]

        # Embed in API batches
        flat_embeddings = []
        provider = cfg["provider"]
        for batch_start in range(0, len(flat_texts), batch_size):
            batch = flat_texts[batch_start:batch_start + batch_size]
            # Replace empty strings with a single space (APIs reject "")
            batch = [t if t.strip() else " " for t in batch]
            vecs = embed_batch_with_retry(model_key, batch)
            flat_embeddings.extend(vecs)
            # Throttle Google API to stay within TPM limits
            if provider == "google" and batch_start + batch_size < len(flat_texts):
                time.sleep(2.0)

        # Reshape and store
        flat_arr = np.array(flat_embeddings, dtype=np.float32)
        if segs_per_candidate == 1:
            result[candidate_idx:chunk_end] = flat_arr
        else:
            result[candidate_idx:chunk_end] = flat_arr.reshape(
                chunk_size, segs_per_candidate, dim
            )

        candidate_idx = chunk_end
        pbar.update(chunk_size)

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
        log.warning(
            "  DIMENSION MISMATCH: %s/%s expected %d, got %d",
            model_key, condition, expected_dim, actual_dim,
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
    else:
        log.info("  Validation passed: shape=%s, no zero vectors", arr.shape)


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
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with model, condition, and validate_only fields.
    """
    parser = argparse.ArgumentParser(
        description="Embed transcripts with paid API models."
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
        help="Test API keys and exit without embedding.",
    )
    return parser.parse_args()


def main():
    """Entry point for the embedding pipeline."""
    args = parse_args()

    # Load environment
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        log.info("Loaded .env from %s", env_path)
    else:
        log.warning("No .env file found at %s", env_path)

    os.makedirs(VECTORS_DIR, exist_ok=True)

    # Determine which models and conditions to run
    model_keys = [args.model] if args.model else list(MODELS.keys())
    conditions = [args.condition] if args.condition else ALL_CONDITIONS

    # Validate API keys
    log.info("Validating API keys...")
    if not validate_keys(model_keys):
        log.error("API key validation failed. Fix keys in .env and retry.")
        sys.exit(1)
    log.info("All API keys validated successfully.")

    if args.validate_only:
        return

    # Run embedding pipeline
    truncation_report = {}
    total_combos = len(model_keys) * len(conditions)
    combo_num = 0

    for model_key in model_keys:
        truncation_report[model_key] = {}
        for condition in conditions:
            combo_num += 1
            log.info(
                "=== [%d/%d] %s x %s ===",
                combo_num, total_combos, model_key, condition,
            )
            stats = embed_condition(model_key, condition)
            truncation_report[model_key][condition] = stats

    # Save truncation report
    save_truncation_report(truncation_report)

    log.info("Done. All embeddings saved to %s", VECTORS_DIR)


if __name__ == "__main__":
    main()
