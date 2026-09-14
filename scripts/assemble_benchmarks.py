#!/usr/bin/env python3
"""Assemble benchmarks/*.yaml from the per-language results HF Jobs produced.

Jobs write a results/{iso}.json into the audio dataset repo rather than
committing to git, so a job needs only an HF token.  This pulls those
results together into the YAMLs the leaderboard reads, which are then
committed here.

Run:  python3 scripts/assemble_benchmarks.py
      python3 scripts/assemble_benchmarks.py --repo ghananlpcommunity/gtts-benchmark-audio
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get(
        "GTTS_AUDIO_REPO", "ghananlpcommunity/gtts-benchmark-audio"))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from benchmark.asr import judge_for
    from benchmark.config import ISO_TO_NAME, NUM_SAMPLES
    from benchmark.evaluate import (
        BENCHMARK_DIR, _write_language_yaml, _write_voice_yaml, _write_matrix,
    )

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "voice").mkdir(parents=True, exist_ok=True)

    local = Path(snapshot_download(
        repo_id=args.repo, repo_type="dataset",
        allow_patterns=["results/*.json"],
        token=os.environ.get("HF_TOKEN") or None,
    ))
    files = sorted((local / "results").glob("*.json"))
    if not files:
        print("no results/*.json in the dataset repo yet")
        return

    # Build the nested dicts that _write_* expect.
    iso_voice = {}
    voice_iso = {}
    for path in files:
        iso = path.stem
        results = json.loads(path.read_text(encoding="utf-8"))
        if not results:
            continue
        for r in results:
            v = r.get("voice")
            res = {
                "cer": r.get("cer"),
                "wer": r.get("wer"),
                "score": r.get("cer"),
                "samples": r.get("samples", NUM_SAMPLES),
                "valid": r.get("valid", 0),
                "sample_clip": r.get("sample_clip"),
            }
            iso_voice.setdefault(iso, {})[v] = res
            voice_iso.setdefault(v, {})[iso] = res

    # CER files.
    for iso, vv in iso_voice.items():
        rows = sorted(vv.items(), key=lambda kv: kv[1]["cer"] or 1e9)
        # Re-use the YAML writer directly — it expects res dicts with
        # cer/wer/samples/valid but also sample_clip (which it skips).
        _write_language_yaml(iso, vv)

    for voice, ll in voice_iso.items():
        rows = sorted(ll.items(), key=lambda kv: kv[1]["cer"] or 1e9)
        _write_voice_yaml(voice, ll)

    _write_matrix(voice_iso)

    print(f"\nassembled {len(files)} language file(s) into benchmarks/")


if __name__ == "__main__":
    main()
