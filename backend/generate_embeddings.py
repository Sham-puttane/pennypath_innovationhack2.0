#!/usr/bin/env python3
"""
PennyPath — Generate Entity Embeddings
Embeds all 4,194 entities using Gemini embedding API with 6-key rotation.
Used for Q6 semantic search in query_graph.py.

Output: output/entity_embeddings.json
"""

import json
import sys
import time
import requests
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"

import os

def _load_api_keys():
    keys_str = os.environ.get("GEMINI_API_KEYS", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    single = os.environ.get("GRAPHRAG_API_KEY", "")
    if single:
        return [single]
    return ["YOUR_API_KEY_HERE"]

API_KEYS = _load_api_keys()

MODEL = "gemini-embedding-001"
BATCH_SIZE = 10  # Gemini supports batch embedding
MAX_RETRIES = 3


def embed_batch(texts: list[str], key_index: int, retry: int = 0) -> list[list[float]] | None:
    """Embed a batch of texts using Gemini embedding API."""
    key = API_KEYS[key_index % len(API_KEYS)]

    # Gemini batch embedding endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents?key={key}"

    requests_body = [
        {
            "model": f"models/{MODEL}",
            "content": {"parts": [{"text": t[:2000]}]},  # cap text length
        }
        for t in texts
    ]

    try:
        resp = requests.post(
            url,
            json={"requests": requests_body},
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            return [e["values"] for e in data["embeddings"]]
        elif resp.status_code == 429:
            if retry < MAX_RETRIES:
                wait = 2 ** (retry + 1)
                time.sleep(wait)
                return embed_batch(texts, key_index + 1, retry + 1)
            else:
                print(f"    Rate limited after {MAX_RETRIES} retries")
                return None
        else:
            print(f"    API error {resp.status_code}: {resp.text[:100]}")
            if retry < MAX_RETRIES:
                return embed_batch(texts, key_index + 1, retry + 1)
            return None
    except Exception as e:
        print(f"    Error: {e}")
        if retry < MAX_RETRIES:
            return embed_batch(texts, key_index + 1, retry + 1)
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("  Generating Entity Embeddings")
    print("=" * 60)

    entities = json.loads(
        (OUTPUT_DIR / "entities.json").read_text(encoding="utf-8")
    )
    print(f"  Entities to embed: {len(entities)}")

    # Check for existing partial progress
    progress_path = OUTPUT_DIR / "embeddings_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        embeddings = progress["embeddings"]
        start_idx = progress["last_index"]
        print(f"  Resuming from index {start_idx} ({len(embeddings)} already done)")
    else:
        embeddings = {}
        start_idx = 0

    # Prepare texts: "NAME: description"
    entity_texts = [
        f"{e['name']}: {e.get('description', '')[:500]}"
        for e in entities
    ]

    total = len(entity_texts)
    key_idx = 0
    success_count = len(embeddings)
    fail_count = 0
    start_time = time.time()

    for i in range(start_idx, total, BATCH_SIZE):
        batch = entity_texts[i : i + BATCH_SIZE]
        batch_names = [entities[j]["name"] for j in range(i, min(i + BATCH_SIZE, total))]

        result = embed_batch(batch, key_idx)
        key_idx += 1  # rotate key each batch

        if result and len(result) == len(batch):
            for name, emb in zip(batch_names, result):
                embeddings[name] = emb
            success_count += len(batch)
        else:
            fail_count += len(batch)
            print(f"    FAILED batch at index {i}")

        # Progress update every 50 batches
        if (i // BATCH_SIZE) % 50 == 0 or i + BATCH_SIZE >= total:
            elapsed = time.time() - start_time
            rate = success_count / max(elapsed, 1) * 60
            remaining = (total - i) / max(rate / 60, 0.1)
            print(
                f"  [{success_count}/{total}] "
                f"{elapsed:.0f}s elapsed, "
                f"~{rate:.0f} entities/min, "
                f"~{remaining:.0f}s remaining"
            )

            # Save progress checkpoint
            progress_path.write_text(
                json.dumps({"last_index": i, "embeddings": embeddings}, ensure_ascii=False),
                encoding="utf-8",
            )

        # Small delay between batches to avoid rate limits
        time.sleep(0.3)

    # Save final embeddings
    print(f"\n  Saving {len(embeddings)} embeddings...")

    # Save as JSON (name -> embedding vector)
    output_path = OUTPUT_DIR / "entity_embeddings.json"
    output_path.write_text(
        json.dumps(embeddings, ensure_ascii=False), encoding="utf-8"
    )
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"  Saved: entity_embeddings.json ({size_mb:.1f} MB)")

    # Also save as numpy for faster loading
    try:
        names = list(embeddings.keys())
        vectors = np.array([embeddings[n] for n in names], dtype=np.float32)
        np.savez_compressed(
            str(OUTPUT_DIR / "entity_embeddings.npz"),
            names=names,
            vectors=vectors,
        )
        npz_size = (OUTPUT_DIR / "entity_embeddings.npz").stat().st_size / 1_048_576
        print(f"  Saved: entity_embeddings.npz ({npz_size:.1f} MB)")
        print(f"  Embedding dimensions: {vectors.shape}")
    except Exception as e:
        print(f"  numpy save failed (non-critical): {e}")

    # Cleanup progress file
    if progress_path.exists():
        progress_path.unlink()

    print(f"\n  Done! {success_count} succeeded, {fail_count} failed")
    print(f"  Total time: {time.time() - start_time:.0f}s")

    # Also embed community reports
    print("\n  Embedding community reports...")
    reports_path = OUTPUT_DIR / "community_reports.json"
    if reports_path.exists():
        reports = json.loads(reports_path.read_text(encoding="utf-8"))
        report_texts = [r["summary"] for r in reports]
        report_embeddings = {}

        for i in range(0, len(report_texts), BATCH_SIZE):
            batch = report_texts[i : i + BATCH_SIZE]
            result = embed_batch(batch, key_idx)
            key_idx += 1
            if result:
                for j, emb in enumerate(result):
                    report_embeddings[str(i + j)] = emb
            time.sleep(0.3)

        (OUTPUT_DIR / "community_embeddings.json").write_text(
            json.dumps(report_embeddings, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Saved {len(report_embeddings)} community embeddings")


if __name__ == "__main__":
    main()
