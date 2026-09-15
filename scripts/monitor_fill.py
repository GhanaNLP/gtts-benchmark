#!/usr/bin/env python3
"""Poll job + dataset state until the fill pass settles.

Prints a compact line each poll. Exits 0 when all jobs reach a terminal
state (COMPLETED/ERROR/CANCELLED) or when --max-polls is hit.

Run: python3 scripts/monitor_fill.py --interval 600 --max-polls 12
"""

import argparse
import os
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

AUDIO_REPO = "ghananlpcommunity/gtts-benchmark-audio"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="6aa8ea",
                    help="job-id prefix to watch")
    ap.add_argument("--namespace", default="ghananlpcommunity")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--max-polls", type=int, default=12)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        if cached.exists():
            token = cached.read_text().strip()

    api = HfApi()
    target = 69 * 200

    for poll in range(args.max_polls):
        jobs = [j for j in api.list_jobs(namespace=args.namespace, token=token)
                if (j.id or "").startswith(args.prefix)]
        stages = Counter(j.status.stage for j in jobs)

        tree = api.list_repo_tree(AUDIO_REPO, repo_type="dataset",
                                  path_in_repo="audio", recursive=True)
        langs = Counter()
        for p in tree:
            parts = p.path.split("/")
            if len(parts) >= 3 and p.path.endswith(".wav"):
                langs[parts[1]] += 1

        cov = ", ".join(f"{k}={langs[k]}" for k in sorted(langs)) or "none"
        print(f"[{time.strftime('%H:%M:%S')} poll {poll+1}] "
              f"{' '.join(f'{s}:{n}' for s, n in stages.items())} | "
              f"wavs={sum(langs.values()):,}/{target*12:,} | {cov}")

        terminal = {"COMPLETED", "ERROR", "CANCELLED"}
        if jobs and all(j.status.stage in terminal for j in jobs):
            print("all terminal — done")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()