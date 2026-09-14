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
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synth", "score", "all"], default="all")
    ap.add_argument("--langs", default="",
                    help="comma/space separated subset (default: all 12)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ref", default="main", help="git ref the jobs check out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import run_job
    from benchmark.asr import judge_for
    from benchmark.config import LANGS

    isos = ([s.strip() for s in args.langs.replace(",", " ").split() if s.strip()]
            or list(LANGS))
    token = os.environ.get("HF_TOKEN")
    secrets = {
        "HF_TOKEN": token,
        "KHAYA_API_KEY": os.environ.get("KHAYA_API_KEY", ""),
    }

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

    print(f"{len(planned)} job(s) to submit as {NAMESPACE}")
    for stage, iso, image, flavor, is_score in planned:
        label = f"{stage:6s}  {iso}"
        if args.dry_run:
            print(f"  [dry-run] {label}  ({flavor})")
            continue
        job = run_job(
            image=image,
            command=build_command(stage, iso, args.ref, args.concurrency,
                                  args.force, "cuda" if is_score and
                                  flavor != FLAVORS["score_khaya"] else "cpu",
                                  is_score),
            flavor=flavor,
            secrets=secrets,
            namespace=NAMESPACE,
            timeout="12h",
            token=token,
        )
        print(f"  submitted {label}  ({flavor})  -> {job.id}")


if __name__ == "__main__":
    main()
