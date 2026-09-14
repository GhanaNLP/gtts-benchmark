"""Full gTTS-voices benchmark pipeline.

Every gTTS voice reads each Ghanaian language's 200-sample pool (converted to
africa-g2p universal graphemes), the best per-language ASR judge transcribes
every clip, and each clip is scored as CER against the original sentence.

Usage:
    python -m benchmark.pipeline                                  # full run
    python -m benchmark.pipeline --voices en fr sw --langs ewe    # subset
    python -m benchmark.pipeline --stage synth --limit 3           # quick synth
    python -m benchmark.pipeline --stage score                     # only score
    python -m benchmark.pipeline --dry-run                         # show plan
"""

import argparse
import asyncio


def main():
    ap = argparse.ArgumentParser(description="gTTS-voices benchmark pipeline")
    ap.add_argument("--voices", nargs="*", help="gTTS voices (default: all 69)")
    ap.add_argument("--langs", nargs="*",
                    help="Ghanaian language codes (default: all 12)")
    ap.add_argument("--no-universal", action="store_true",
                    help="Skip africa-g2p universal conversion")
    ap.add_argument("--stage", choices=["all", "synth", "score"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="Re-run completed samples")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process this many samples per cell (smoke test)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help=f"gTTS requests in flight (default: from config)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from .config import CONCURRENCY, LANGS
    from .evaluate import (
        assemble,
        score_cell,
        synthesize_cell_async,
    )
    from .langs import voice_name, voices
    from .normalize import UniversalConverter
    from .synthesizer import GTTSModel

    vlist = args.voices or voices()
    isos = args.langs or list(LANGS)
    conc = args.concurrency or CONCURRENCY

    if args.dry_run:
        print(f"Plan: {len(vlist)} voices x {len(isos)} languages "
              f"x {args.limit or 200} samples = "
              f"{len(vlist) * len(isos):,} cells")
        print(f"  Voices   ({len(vlist)}): " + ", ".join(vlist))
        print(f"  Languages ({len(isos)}): " + ", ".join(isos))
        print(f"  Concurrency: {conc}")
        return

    print(f"== gTTS-voices benchmark: {len(vlist)} voices x {len(isos)} "
          f"languages == (concurrency={conc})")

    if args.stage != "score":
        asyncio.run(_run_synth(vlist, isos, conc, args))

    if args.stage != "synth":
        _run_score(vlist, isos, args)

    if args.stage != "synth":
        assemble()


async def _run_synth(vlist, isos, conc, args):
    from .evaluate import _lang_pool, synthesize_cell_async
    from .langs import voice_name
    from .normalize import UniversalConverter
    from .synthesizer import GTTSModel

    converter = UniversalConverter(enabled=not args.no_universal)

    # Pre-convert all pools (cheap, sequential) so the text cache is ready
    # before concurrent synthesis starts.
    for iso in isos:
        _lang_pool(iso, converter)

    # Synthesis: one voice at a time, all samples within each (voice, iso)
    # cell are concurrent.
    for voice in vlist:
        print(f"\n=== voice {voice_name(voice)} ({voice}) ===")
        gtts_voice = GTTSModel(voice)
        for iso in isos:
            try:
                await synthesize_cell_async(
                    voice, iso, converter,
                    force=args.force, gtts_voice=gtts_voice,
                    limit=args.limit, concurrency=conc,
                )
            except Exception as e:
                print(f"  ERROR synth {voice}->{iso}: {e}")


def _run_score(vlist, isos, args):
    from .asr import load_judge
    from .config import DEVICE
    from .evaluate import score_cell

    judges = {}
    try:
        for iso in isos:
            if iso in judges:
                continue
            print(f"  Loading ASR judge for {iso} ...")
            judges[iso] = load_judge(iso, device=DEVICE)

        for voice in vlist:
            for iso in isos:
                judge = judges.get(iso)
                if judge is None:
                    print(f"  ERROR: no judge for {iso}")
                    continue
                try:
                    score_cell(voice, iso, judge=judge,
                               force=args.force, limit=args.limit)
                except Exception as e:
                    print(f"  ERROR score {voice}->{iso}: {e}")
    finally:
        for judge in judges.values():
            try:
                judge.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()