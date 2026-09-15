#!/usr/bin/env python3
"""Check/stream the jobs submitted for the gTTS-voices benchmark.

Run:  python3 scripts/hf_job_status.py              # one-line totals
      python3 scripts/hf_job_status.py --watch 60   # poll every 60s
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import huggingface_hub as hub


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default=os.environ.get("GTTS_HF_NAMESPACE", "ghananlpcommunity"))
    ap.add_argument("--watch", type=int, default=0, help="poll interval in seconds (0 = once)")
    ap.add_argument("--prefix", default="", help="only jobs whose id starts with this")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        if cached.exists():
            token = cached.read_text().strip()

    api = hub.HfApi()
    jobs = api.list_jobs(namespace=args.namespace, token=token or None)
    while True:
        jobs = [j for j in api.list_jobs(namespace=args.namespace, token=token or None)
                if not args.prefix or (j.id or "").startswith(args.prefix)]
        from collections import Counter
        totals = Counter(j.status.stage for j in jobs)
        print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} job(s): "
              + ", ".join(f"{s}:{n}" for s, n in totals.items()) or "none")
        if not args.watch:
            return
        if totals.get("COMPLETED", 0) == len(jobs) and len(jobs):
            print("all done")
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()