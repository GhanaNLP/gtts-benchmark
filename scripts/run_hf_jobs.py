#!/usr/bin/env python3
"""Submit the benchmark to HuggingFace Jobs.

One job per language for synthesis (all 69 gTTS voices inside), one per
language for scoring (all 69 voices judged by that language's ASR).  Jobs
are independent containers, so a failure costs one unit rather than the
whole run, and everything survives this machine being closed.

The scoring image is reused from nsanku-tts-benchmark (same judges).
The synthesis image is built by GhanaOpenAI/gtts-benchmark-images.

Run:  python3 scripts/run_hf_jobs.py --stage synth
      python3 scripts/run_hf_jobs.py --stage score --langs ewe,twi_asante
      python3 scripts/run_hf_jobs.py --stage all --dry-run
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, run_job

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_URL = "https://github.com/GhanaNLP/gtts-benchmark"
NAMESPACE = os.environ.get("GTTS_HF_NAMESPACE", "ghananlpcommunity")

# Synthesis: a tiny image (python:3.12-slim + gTTS + africa-g2p) built by
# GhanaOpenAI/gtts-benchmark-images.  Scoring: reuses nsanku's asr image
# since the judges (khaya / griot / omni) are identical.
NSANKU_REGISTRY = "ghcr.io/ghanaopenai/nsanku-tts-benchmark"
OUR_REGISTRY = "ghcr.io/ghanaopenai/gtts-benchmark"
IMAGES = {"synth": f"{OUR_REGISTRY}:synth", "score": f"{NSANKU_REGISTRY}:asr"}

# Synthesis is pure HTTP (no GPU), so cpu-upgrade is cheapest.  Khaya is a
# REST call (cpu).  Griot (153M) fits an A10G.  OmniASR (7B) needs an L40S.
FLAVORS = {
    "synth": "cpu-upgrade",
    "score_gpu": "a10g-small",
    "score_omni": "l40sx1",
    "score_khaya": "cpu-upgrade",
}


def _flavor_for(spec):
    """Pick the cheapest hardware that can run this judge."""
    if spec is None or spec.get("kind") == "khaya":
        return FLAVORS["score_khaya"]
    if spec.get("kind") == "omniasr":
        return FLAVORS["score_omni"]
    return FLAVORS["score_gpu"]          # griot


def build_command(stage, iso, ref, concurrency, force, device, image_is_score):
    unit = f"--stage {stage} --lang {iso}"
    if concurrency:
        unit += f" --concurrency {concurrency}"
    if force:
        unit += " --force"
    if device:
        unit += f" --device {device}"
    script = (
        "set -eux; "
        f"git clone --depth 1 --branch {ref} {REPO_URL} /app; "
        "cd /app; "
        f"python -m benchmark.job {unit}"
    )
    return ["bash", "-lc", script]


def submit(stage, iso, image, flavor, is_score, args, token):
    return run_job(
        image=image,
        command=build_command(stage, iso, args.ref, args.concurrency,
                              args.force, "cuda" if is_score and
                              flavor != FLAVORS["score_khaya"] else "cpu",
                              is_score),
        flavor=flavor,
        secrets={"HF_TOKEN": token,
                 "KHAYA_API_KEY": os.environ.get("KHAYA_API_KEY", "")},
        namespace=NAMESPACE,
        timeout="12h",
        token=token,
    )


def wait_batch(ids, token, poll=120, max_wait_h=13, label=""):
    """Block until every job in *ids* reaches a terminal state.

    Jobs share the Google endpoint and the dataset repo, so running them in
    waves keeps the request rate gentle and pushes non-conflicting.
    """
    from collections import Counter
    import time
    api = HfApi()
    deadline = time.time() + max_wait_h * 3600
    while time.time() < deadline:
        jobs = [j for j in api.list_jobs(namespace=NAMESPACE, token=token)
                if (j.id or "") in ids]
        stages = Counter(j.status.stage for j in jobs)
        done = sum(n for s, n in stages.items()
                   if s in ("COMPLETED", "ERROR", "CANCELLED"))
        print(f"  [{label}] {','.join(j.id[-5:] for j in jobs)} "
              f"{dict(stages)}", flush=True)
        if done == len(ids):
            return
        time.sleep(poll)
    raise SystemExit(
        f"timed out waiting for batch after {max_wait_h}h")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synth", "score", "all"], default="all")
    ap.add_argument("--langs", default="",
                    help="comma/space separated subset (default: all 12)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ref", default="main", help="git ref the jobs check out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=4,
                    help="max concurrently running jobs per wave")
    ap.add_argument("--no-wait", action="store_true",
                    help="submit all jobs without waiting for completion")
    args = ap.parse_args()

    from benchmark.asr import judge_for
    from benchmark.config import LANGS

    isos = ([s.strip() for s in args.langs.replace(",", " ").split() if s.strip()]
            or list(LANGS))
    token = os.environ.get("HF_TOKEN")

    planned = []
    if args.stage in ("all", "synth"):
        for iso in isos:
            planned.append(("synth", iso, IMAGES["synth"],
                            FLAVORS["synth"], False))
    if args.stage in ("all", "score"):
        for iso in isos:
            spec = judge_for(iso)
            planned.append(("score", iso, IMAGES["score"],
                            _flavor_for(spec), True))

    print(f"{len(planned)} job(s) to submit as {NAMESPACE} "
          f"(batch={args.batch})")

    if args.no_wait:
        for i, (stage, iso, image, flavor, is_score) in enumerate(planned):
            if args.dry_run:
                print(f"  [dry-run] {stage:6s} {iso}  ({flavor})")
                continue
            job = submit(stage, iso, image, flavor, is_score, args, token)
            print(f"  submitted {stage:6s} {iso}  ({flavor})  -> {job.id}")
            if args.stage == "synth" and i < len(planned) - 1:
                import time
                time.sleep(8)   # stagger submissions; don't open the Google
                                # endpoint with every job at the same instant
        return

    # Wave-scoped submission: wait for each batch before starting the next,
    # so the Google endpoint and dataset repo see a steady, bounded load.
    pending = list(enumerate(planned))
    wave = 1
    while pending:
        batch = pending[:args.batch]
        submit_ids = []
        for i, (stage, iso, image, flavor, is_score) in batch:
            label = f"{stage:6s} {iso}"
            print(f"  [{label}] wave {wave}", flush=True)
            if args.dry_run:
                continue
            job = submit(stage, iso, image, flavor, is_score, args, token)
            submit_ids.append(job.id)
            print(f"  submitted {label}  ({flavor})  -> {job.id}")
            import time
            time.sleep(8)   # stagger submissions within the wave

        if not args.dry_run and submit_ids:
            wait_batch(submit_ids, token,
                       label="wave %d (%s)"
                             % (wave, ",".join(iso for _, (_, iso, *_)
                                               in batch)))

        pending = pending[args.batch:]
        wave += 1
    print("all waves finished")


if __name__ == "__main__":
    main()
