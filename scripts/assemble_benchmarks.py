#!/usr/bin/env python3
"""Assemble benchmarks/*.yaml from the per-language results HF Jobs produced.

Jobs write transcriptions/{iso}_{voice}.csv and results/{iso}.json into the
audio dataset repo rather than committing to git.  This script pulls those
together into the YAMLs the leaderboard reads.

Balanced sampling: clip coverage is uneven while the gTTS endpoint is rate
limiting, so a voice lost a few samples mid-run.  Ranking voices with
different clip counts is unfair, so by default only samples that at least
``GTTS_MIN_VOICES`` voices covered are kept, and a voice is ranked only if it
covers every one of those common samples.  Tune with e.g.:

    GTTS_MIN_VOICES=40 python3 scripts/assemble_benchmarks.py

Run:  python3 scripts/assemble_benchmarks.py
      python3 scripts/assemble_benchmarks.py --repo ghananlpcommunity/gtts-benchmark-audio
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_csv(csv_path):
    """sample_id -> {cer, wer} for rows with a real score."""
    out = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                cer = float(row["cer"])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                wer = float(row["wer"])
            except (TypeError, ValueError, KeyError):
                wer = None
            out[int(row["sample_id"])] = {"cer": cer, "wer": wer}
    return out


def _balancer(cells):
    """Pick, per language, the common samples + voices that cover them.

    Returns ``(keep_voices, common)`` — voices with < coverage of the common
    sample set are dropped so rankings are comparable.
    """
    min_voices = int(os.environ.get("GTTS_MIN_VOICES", "40"))
    from collections import Counter

    keep_voices, common = {}, {}
    for iso, per_voice in sorted(cells.items()):
        per_voice = {v: s for v, s in per_voice.items() if s}
        if not per_voice:
            continue
        votes = Counter()
        for s in per_voice.values():
            votes.update(s.keys())
        com = {i for i, c in votes.items() if c >= min_voices}
        if not com:
            continue
        good = {v: s for v, s in per_voice.items() if set(s) >= com}
        if len(good) < 2:
            continue
        keep_voices[iso] = good
        common[iso] = com
    return keep_voices, common


def _res_from(subset, sample_clip):
    cers = [s["cer"] for s in subset.values() if s["cer"] is not None]
    wers = [s["wer"] for s in subset.values() if s["wer"] is not None]
    if not cers:
        return None
    return {
        "cer": round(sum(cers) / len(cers), 4),
        "wer": round(sum(wers) / len(wers), 4) if wers else None,
        "score": round(sum(cers) / len(cers), 4),
        "samples": len(subset),
        "valid": len(cers),
        "sample_clip": sample_clip,
    }


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
        allow_patterns=["transcriptions/*.csv", "results/*.json"],
        token=os.environ.get("HF_TOKEN") or None,
    ))

    # Per-language voice -> sample_id -> scores.
    cells = {}
    for csv_path in sorted((local / "transcriptions").glob("*.csv")):
        stem = csv_path.stem  # {iso}_{voice}  (voice may contain '-')
        iso, _, voice = stem.rpartition("_")
        cells.setdefault(iso, {})[voice] = _load_csv(csv_path)

    keep_voices, common = _balancer(cells)

    # Fall back to results/*.json for languages with no transcriptions yet.
    for json_path in sorted((local / "results").glob("*.json")):
        iso = json_path.stem
        if iso in cells:
            continue
        results = json.loads(json_path.read_text(encoding="utf-8"))
        if not results:
            continue
        cells[iso] = {
            r["voice"]: {
                i: {"cer": r["cer"], "wer": r["wer"]}
                for i in range(int(r["valid"] or 0))
            }
            for r in results
        }
        keep_voices[iso] = cells[iso]
        common[iso] = set(range(int(results[0].get("valid", 0) or 0)))

    iso_voice = {}
    voice_iso = {}
    for iso in sorted(keep_voices):
        for voice, samples in keep_voices[iso].items():
            subset = {i: s for i, s in samples.items() if i in common[iso]}
            sample_clip = next(
                (i for i in sorted(subset) if subset[i]["cer"] is not None),
                None,
            )
            res = _res_from(subset, sample_clip)
            if res is None:
                print(f"  drop {iso}/{voice} — nothing valid in common set")
                continue
            if len(subset) < len(common[iso]):
                print(f"  drop {iso}/{voice} — "
                      f"{len(subset)}/{len(common[iso])} common samples")
            iso_voice.setdefault(iso, {})[voice] = res
            voice_iso.setdefault(voice, {})[iso] = res

    for iso, vv in iso_voice.items():
        _write_language_yaml(iso, vv)
    for voice, ll in voice_iso.items():
        _write_voice_yaml(voice, ll)
    _write_matrix(voice_iso)

    kept = sum(len(v) for v in iso_voice.values())
    print(f"\nassembled {len(iso_voice)} language(s), {kept} kept voice-cells "
          f"into benchmarks/  (isolated common sample per language: "
          f"{ {iso: len(common[iso]) for iso in sorted(common)} })")


if __name__ == "__main__":
    main()