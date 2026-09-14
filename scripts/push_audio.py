"""Push synthesised audio to GhanaOpenAI/gtts-benchmark-audio.

Uploads the whole ``audio/`` tree to a dataset repo so the HF Space can stream
demo clips.  Idempotent: huggingface_hub skips unchanged files when you pass
``deduplicate=True``.

Usage:
    python scripts/push_audio.py            # upload everything
    python scripts/push_audio.py --repo GhanaOpenAI/gtts-benchmark-audio
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="GhanaOpenAI/gtts-benchmark-audio")
    ap.add_argument("--audio-dir", default=str(ROOT / "audio"))
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        raise SystemExit(f"No audio dir at {audio_dir}")

    api.create_repo(
        repo_id=args.repo, repo_type="dataset", exist_ok=True,
        private=False,
        description="gTTS-voices x Ghanaian languages benchmark audio "
                    "(69 voices x 12 languages).",
    )
    print(f"Uploading {audio_dir} -> {args.repo} ...")
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(audio_dir),
        path_in_repo="",
        repo_type="dataset",
        ignore_patterns=["synth_errors.json", "*.txt", "**/texts.json"],
    )
    print("Done.")


if __name__ == "__main__":
    main()