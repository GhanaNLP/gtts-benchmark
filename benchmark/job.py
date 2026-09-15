"""Run one unit of benchmark work inside a HuggingFace Job.

A job is a fresh container with no shared filesystem, so the unit has to pull
what it needs and push what it produced:

    pull   existing clips / transcriptions for this language from the audio
           dataset repo, so a re-run resumes instead of starting over
    run    synthesis (all 69 voices for one language) or scoring (all 69
           voices, using that language's ASR judge)
    push   clips, transcriptions and a per-language result JSON back to the
           audio dataset repo

The YAMLs the leaderboard reads are assembled by scripts/assemble_benchmarks.py
and committed to GitHub, so a job needs only an HF token — no git credentials.

Run (inside a job):
    python -m benchmark.job --stage synth --lang twi_asante
    python -m benchmark.job --stage score --lang ewe --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("GTTS_WORK_DIR", "/data"))
AUDIO_REPO = os.environ.get("GTTS_AUDIO_REPO",
                             "ghananlpcommunity/gtts-benchmark-audio")
RESULTS_PREFIX = "results"


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def _prepare_dirs():
    """Point the benchmark at this container's scratch space."""
    audio = WORK / "audio"
    trans = WORK / "transcriptions"
    audio.mkdir(parents=True, exist_ok=True)
    trans.mkdir(parents=True, exist_ok=True)
    os.environ["GTTS_AUDIO_DIR"] = str(audio)
    os.environ["GTTS_TRANSCRIPTIONS_DIR"] = str(trans)
    os.environ.setdefault("HF_HOME", str(WORK / "hf-cache"))
    return audio, trans


def _pull(api, patterns, dest):
    """Fetch just the paths this unit needs, not the full dataset."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=AUDIO_REPO,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(dest),
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as e:
        print(f"  nothing to resume ({type(e).__name__}: {str(e)[:120]})")


def _push(api, folder, path_in_repo, message, attempts=5, base_delay=30.0):
    """Push a folder to the audio dataset repo, retrying HF-side 429s.

    All jobs push to the same dataset repo, so uploads can collide in HF's
    per-repo commit/concurrency queue.  Back off and retry instead of
    dropping the whole job.
    """
    if not any(Path(folder).rglob("*")):
        return
    last = None
    for attempt in range(attempts):
        try:
            api.upload_folder(
                folder_path=str(folder),
                path_in_repo=path_in_repo,
                repo_id=AUDIO_REPO,
                repo_type="dataset",
                commit_message=message,
            )
            return
        except Exception as e:  # noqa: BLE001 - any HF error: retry
            last = e
            delay = base_delay * (2 ** attempt)
            print(f"  push attempt {attempt + 1}/{attempts} failed "
                  f"({type(e).__name__}: {str(e)[:100]}); retrying in "
                  f"{delay:.0f}s")
            time.sleep(delay)
    raise last


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synth", "score"], required=True)
    ap.add_argument("--lang", required=True, help="ISO code of the language")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    audio_dir, trans_dir = _prepare_dirs()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark.config import DEVICE, CONCURRENCY
    from benchmark.evaluate import score_language, synthesize_language

    api = _api()
    conc = args.concurrency or CONCURRENCY
    device = args.device or DEVICE

    if args.stage == "synth":
        _pull(api, [f"audio/{args.lang}/*"], WORK)
        w, f = synthesize_language(args.lang, force=args.force,
                                   concurrency=conc)
        _push(api, audio_dir, "audio",
              f"Synthesis: {args.lang} ({w} ok, {f} failed)")
        return

    _pull(api, [f"audio/{args.lang}/*", f"transcriptions/{args.lang}_*"], WORK)
    results = score_language(args.lang, device=device, force=args.force)
    _push(api, trans_dir, "transcriptions", f"Transcriptions: {args.lang}")

    out = WORK / RESULTS_PREFIX
    out.mkdir(exist_ok=True)
    (out / f"{args.lang}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _push(api, out, RESULTS_PREFIX, f"Scores: {args.lang}")
    print(f"\nScored {len(results)} voice(s) for {args.lang}")


if __name__ == "__main__":
    main()
