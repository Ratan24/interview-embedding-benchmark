"""Embed truncated C3 transcripts at varying token limits using Voyage API.

Generates a set of .npy embedding files — one per token limit — to produce
a transcript-length vs accuracy curve.  Reuses the existing embed_api.py
infrastructure for batching, retries, and checkpointing.

Usage:
    python embed_truncated.py                # all token limits
    python embed_truncated.py --limit 1024   # single limit
    python embed_truncated.py --dry-run      # show stats, no API calls
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Reuse embed_api infrastructure
from embed_api import (
    MODELS,
    VECTORS_DIR,
    PARSED_DIR,
    embed_batch_with_retry,
    truncate_text,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_KEY = "voyage"
CONDITION = "C3"
TOKEN_LIMITS = [256, 512, 1024, 2048, 4096, 8192]  # "full" = no truncation (already exists)

CHECKPOINT_INTERVAL = 100

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


def load_c3_transcripts():
    """Load C3 transcripts from parsed/segments_C3.json.

    Returns:
        Tuple of (app_ids, texts) where texts is a list of strings.
    """
    path = PARSED_DIR / "segments_C3.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    app_ids = [rec["job_application_id"] for rec in data]
    texts = [rec["text"] for rec in data]
    return app_ids, texts


def truncate_to_token_limit(text, token_limit):
    """Truncate text to a specific token limit.

    Uses word count × 1.5 as token estimate (consistent with embed_api.py).

    Args:
        text: Raw transcript string.
        token_limit: Maximum tokens allowed.

    Returns:
        Truncated string.
    """
    word_limit = int(token_limit / 1.5)
    words = text.split()
    if len(words) <= word_limit:
        return text
    return " ".join(words[:word_limit])


# ---------------------------------------------------------------------------
# Core embedding loop
# ---------------------------------------------------------------------------


def embed_at_limit(app_ids, texts, token_limit):
    """Embed all transcripts truncated to a specific token limit.

    Args:
        app_ids: List of job_application_id strings.
        texts: List of full transcript strings.
        token_limit: Token limit to truncate to.

    Returns:
        numpy array of shape (n_candidates, dim).
    """
    cfg = MODELS[MODEL_KEY]
    dim = cfg["dim"]
    batch_size = cfg["batch_size"]
    n_total = len(texts)
    
    # Output paths
    suffix = f"trunc{token_limit}"
    npy_path = VECTORS_DIR / f"{MODEL_KEY}_{CONDITION}_{suffix}.npy"
    ids_path = VECTORS_DIR / f"{MODEL_KEY}_{CONDITION}_{suffix}_ids.json"

    # Skip if already done
    if npy_path.exists() and ids_path.exists():
        log.info("  Already exists: %s — skipping", npy_path.name)
        return np.load(npy_path)

    # Truncate all texts
    truncated = [truncate_to_token_limit(t, token_limit) for t in texts]

    # Stats
    orig_words = [len(t.split()) for t in texts]
    trunc_words = [len(t.split()) for t in truncated]
    n_truncated = sum(1 for o, t in zip(orig_words, trunc_words) if t < o)
    log.info(
        "  Token limit %d: %d/%d transcripts truncated (avg words: %.0f -> %.0f)",
        token_limit, n_truncated, n_total,
        sum(orig_words) / n_total, sum(trunc_words) / n_total,
    )

    # Also apply the model's own token limit (Voyage = 32000, much higher)
    model_token_limit = cfg["token_limit"]
    truncated = [truncate_text(t, model_token_limit) for t in truncated]

    # Embed in batches
    result = np.zeros((n_total, dim), dtype=np.float32)
    
    pbar = tqdm(
        total=n_total,
        desc=f"voyage/C3/{suffix}",
        unit="cand",
    )

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch = truncated[batch_start:batch_end]
        # Replace empty strings with a single space (APIs reject "")
        batch = [t if t.strip() else " " for t in batch]

        vecs = embed_batch_with_retry(MODEL_KEY, batch)
        result[batch_start:batch_end] = np.array(vecs, dtype=np.float32)
        pbar.update(batch_end - batch_start)

    pbar.close()

    # Save
    np.save(npy_path, result)
    with open(ids_path, "w") as f:
        json.dump(app_ids, f)

    log.info("  Saved %s  shape=%s", npy_path.name, result.shape)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed truncated C3 transcripts for length curve."
    )
    parser.add_argument(
        "--limit",
        type=int,
        choices=TOKEN_LIMITS,
        default=None,
        help="Run a single token limit (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show truncation stats without calling APIs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load environment
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        log.info("Loaded .env from %s", env_path)

    os.makedirs(VECTORS_DIR, exist_ok=True)

    # Load data
    app_ids, texts = load_c3_transcripts()
    log.info("Loaded %d C3 transcripts", len(texts))

    limits = [args.limit] if args.limit else TOKEN_LIMITS

    if args.dry_run:
        log.info("DRY RUN — showing truncation stats only:\n")
        for limit in limits:
            truncated = [truncate_to_token_limit(t, limit) for t in texts]
            orig_words = [len(t.split()) for t in texts]
            trunc_words = [len(t.split()) for t in truncated]
            n_trunc = sum(1 for o, t in zip(orig_words, trunc_words) if t < o)
            avg_orig = sum(orig_words) / len(texts)
            avg_trunc = sum(trunc_words) / len(texts)
            total_tokens = sum(int(w * 1.5) for w in trunc_words)
            log.info(
                "  limit=%5d: %d/%d truncated, avg words %.0f->%.0f, "
                "total tokens ~%dk",
                limit, n_trunc, len(texts), avg_orig, avg_trunc,
                total_tokens // 1000,
            )

        total_calls = len(texts) * len(limits)
        all_tokens = 0
        for limit in limits:
            truncated = [truncate_to_token_limit(t, limit) for t in texts]
            all_tokens += sum(int(len(t.split()) * 1.5) for t in truncated)
        cost = all_tokens / 1_000_000 * 0.06
        log.info("\n  Total API calls: %d", total_calls)
        log.info("  Total tokens: ~%.1fM", all_tokens / 1_000_000)
        log.info("  Estimated Voyage cost: $%.2f", cost)
        return

    # Run embeddings
    t0 = time.time()
    for limit in limits:
        log.info("Processing token limit: %d", limit)
        embed_at_limit(app_ids, texts, limit)

    elapsed = time.time() - t0
    log.info("All truncated embeddings complete in %.0fs", elapsed)
    log.info("Output files in %s:", VECTORS_DIR)
    for limit in limits:
        suffix = f"trunc{limit}"
        npy = VECTORS_DIR / f"{MODEL_KEY}_{CONDITION}_{suffix}.npy"
        log.info("  %s  exists=%s", npy.name, npy.exists())


if __name__ == "__main__":
    main()
