#!/usr/bin/env python3
"""
Run GraphRAG indexing with API key rotation.
Patches the environment variable before each retry cycle.
GraphRAG has built-in caching, so it will skip already-completed chunks.
"""

import os
import sys
import time
import subprocess
import itertools

def _load_api_keys():
    keys_str = os.environ.get("GEMINI_API_KEYS", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    single = os.environ.get("GRAPHRAG_API_KEY", "")
    if single:
        return [single]
    return ["YOUR_API_KEY_HERE"]

API_KEYS = _load_api_keys()

MAX_CYCLES = 10  # max retry cycles before giving up
WAIT_BETWEEN = 30  # seconds between cycles

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    key_cycle = itertools.cycle(enumerate(API_KEYS))

    for cycle in range(1, MAX_CYCLES + 1):
        key_idx, key = next(key_cycle)
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle}/{MAX_CYCLES} — Using key {key_idx + 1}/{len(API_KEYS)} ({key[:12]}...)")
        print(f"{'='*60}\n")

        # Set the env var for this run
        env = os.environ.copy()
        env["GRAPHRAG_API_KEY"] = key

        result = subprocess.run(
            [sys.executable, "-m", "graphrag", "index", "--root", root],
            env=env,
            cwd=root,
            timeout=1800,  # 30 min max per cycle
        )

        if result.returncode == 0:
            print(f"\n  GraphRAG indexing COMPLETED successfully on cycle {cycle}!")
            return 0

        print(f"\n  Cycle {cycle} exited with code {result.returncode}")
        print(f"  Waiting {WAIT_BETWEEN}s before next cycle with next key...")
        time.sleep(WAIT_BETWEEN)

    print(f"\n  Exhausted {MAX_CYCLES} cycles. Check logs for progress.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
