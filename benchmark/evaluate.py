"""Orchestrate the gTTS-voices x Ghanaian-languages benchmark.

A "model" on the board is one gTTS voice (e.g. ``en``, ``fr-CA``, ``sw``).
Every voice is asked to read each Ghanaian language's sample pool, after the
text has been converted to africa-g2p's universal graphemes; every clip is
then transcribed by that language's best ASR judge and scored as CER against
the original sentence.

Cell = (voice, iso).  Clips live at ``audio/{iso}/{voice}/{idx:05d}.wav`` so
any cell can be synthesised or re-scored independently and incrementally.

Results:
    benchmarks/{iso}.yaml            voices ranked for one language
    benchmarks/voice/{voice}.yaml    languages ranked for one voice
    benchmarks/voice_matrix.json     voice x language CER matrix
    transcriptions/{iso}_{voice}.csv per-sample scores (auditable)
"""

import asyncio
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .asr import judge_for, load_judge
from .config import (
    AUDIO_DIR,
    BENCHMARK_DIR,
    CONCURRENCY,
    ISO_TO_NAME,
    LANGS,
    NUM_SAMPLES,
    TRANSCRIPTIONS_DIR,
)
from .dataset import load_text_samples
from .langs import GTTS_LANGS, voice_name, voices
from .metrics import compute_metrics
from .normalize import UniversalConverter
from .synthesizer import GTTSModel

ERRORS_FILE = "synth_errors.json"


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def clip_dir(iso, voice):
    return AUDIO_DIR / iso / voice


def bench_yaml(iso):
    return BENCHMARK_DIR / f"{iso}.yaml"


def voice_yaml(voice):
    return BENCHMARK_DIR / "voice" / f"{voice}.yaml"


def transcriptions_csv(iso, voice):
    return TRANSCRIPTIONS_DIR / f"{iso}_{voice}.csv"


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Per-language text pool (universally converted once, cached) ─────────────


def _lang_pool(iso, converter, force=False):
    """Load and universally convert the sample pool for *iso*.

    Conversion depends only on the source language, so it is computed once
    and cached in ``audio/{iso}/texts.json`` for reuse across all 69 voices.
    """
    info = LANGS[iso]
    cache = AUDIO_DIR / iso / "texts.json"

    if cache.exists() and not force:
        try:
            rows = json.loads(cache.read_text(encoding="utf-8"))
            if len(rows) >= NUM_SAMPLES:
                return rows
        except (OSError, json.JSONDecodeError):
            rows = None

    samples = load_text_samples(info["subset"], offset=0, limit=NUM_SAMPLES)
    rows = [
        {
            "index": s["index"],
            "text": s["text"],
            "converted": converter.convert_text(iso, s["text"]),
        }
        for s in samples
    ]
    (AUDIO_DIR / iso).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"    [{iso}] prepared {len(rows)} universally-converted texts")
    return rows


# ── Stage 1: synthesis of one (voice, iso) cell ─────────────────────────────


def synthesize_cell(voice, iso, converter=None, force=False, gtts_voice=None,
                    limit=None):
    """Have *voice* read *iso*'s converted pool. Incremental."""
    if converter is None:
        converter = UniversalConverter()

    rows = _lang_pool(iso, converter)
    if limit:
        rows = rows[:limit]
    if not rows:
        return (0, 0)

    out_dir = clip_dir(iso, voice)
    out_dir.mkdir(parents=True, exist_ok=True)

    if gtts_voice is None:
        gtts_voice = GTTSModel(voice)

    _write_text_index(iso, voice, rows)

    todo = [
        r for r in rows
        if force or not (out_dir / f"{r['index']:05d}.wav").exists()
    ]
    if not todo:
        return (0, 0)

    errors = _load_errors(out_dir)
    written = failed = 0
    t0 = time.time()
    for r in todo:
        key = str(r["index"])
        text = r.get("converted") or r["text"]
        try:
            wav = gtts_voice.synthesize(text)
            (out_dir / f"{r['index']:05d}.wav").write_bytes(wav)
            errors.pop(key, None)
            written += 1
        except Exception as e:
            errors[key] = str(e)[:200]
            failed += 1

    _save_errors(out_dir, errors)
    elapsed = time.time() - t0
    print(f"    {voice_name(voice)} ({voice}) -> {iso}: {written} wav, "
          f"{failed} failed ({elapsed:.0f}s)")
    return (written, failed)


async def synthesize_cell_async(voice, iso, converter=None, force=False,
                                gtts_voice=None, limit=None,
                                concurrency=CONCURRENCY):
    """Async version of synthesize_cell: ``concurrency`` samples are
    synthesised concurrently via a thread pool (gTTS is blocking).

    Returns ``(written, failed)``.
    """
    if converter is None:
        converter = UniversalConverter()

    rows = _lang_pool(iso, converter)
    if limit:
        rows = rows[:limit]
    if not rows:
        return (0, 0)

    out_dir = clip_dir(iso, voice)
    out_dir.mkdir(parents=True, exist_ok=True)

    if gtts_voice is None:
        gtts_voice = GTTSModel(voice)

    _write_text_index(iso, voice, rows)

    todo = [
        r for r in rows
        if force or not (out_dir / f"{r['index']:05d}.wav").exists()
    ]
    if not todo:
        return (0, 0)

    errors = _load_errors(out_dir)
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:

        async def _one(r):
            async with sem:
                key = str(r["index"])
                text = r.get("converted") or r["text"]
                try:
                    wav = await loop.run_in_executor(
                        pool, gtts_voice.synthesize, text)
                    (out_dir / f"{r['index']:05d}.wav").write_bytes(wav)
                    return (key, True)
                except Exception as e:
                    return (key, str(e)[:200])

        results = await asyncio.gather(*(_one(r) for r in todo))

    written = failed = 0
    for key, status in results:
        if status is True:
            errors.pop(key, None)
            written += 1
        else:
            errors[key] = status
            failed += 1

    _save_errors(out_dir, errors)
    elapsed = time.time() - t0
    print(f"    {voice_name(voice)} ({voice}) -> {iso}: {written} wav, "
          f"{failed} failed ({elapsed:.0f}s, "
          f"{written / max(elapsed, 0.1):.1f} req/s)")
    return (written, failed)


def _write_text_index(iso, voice, rows):
    path = AUDIO_DIR / iso / f"TEXTS_{voice}.txt"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{ISO_TO_NAME[iso]} ({iso}) read by voice "
                f"{voice_name(voice)} ({voice})\n\n")
        for r in rows:
            f.write(f"{r['index']:05d}  {r['text']}\n")
            if r.get("converted") and r["converted"] != r["text"]:
                f.write(f"       -> {r['converted']}\n")


# ── Stage 2: scoring of one (voice, iso) cell ───────────────────────────────


def score_cell(voice, iso, judge=None, force=False, limit=None):
    """Transcribe one cell's clips and score each against the original text."""
    rows = _load_pool(iso)
    if not rows:
        return {}
    if limit:
        rows = rows[:limit]

    audio_dir = clip_dir(iso, voice)
    if not audio_dir.is_dir():
        print(f"    {voice_name(voice)} -> {iso}: no clips — synthesise first")
        return {}

    print(f"  Scoring {voice_name(voice)} ({voice}) -> {ISO_TO_NAME[iso]} ({iso})")
    if judge is None:
        judge = load_judge(iso)

    entries = {} if force else _load_transcriptions(transcriptions_csv(iso, voice))
    text_by_index = {r["index"]: r for r in rows}
    entries = {k: v for k, v in entries.items() if int(k) in text_by_index}
    synth_errors = _load_errors(audio_dir)

    pending = []
    for index in sorted(text_by_index):
        key = str(index)
        if entries.get(key, {}).get("cer") is not None:
            continue
        wav_path = audio_dir / f"{index:05d}.wav"
        if not wav_path.exists():
            entries[key] = {
                "text": text_by_index[index]["text"],
                "converted": text_by_index[index].get("converted", ""),
                "cer": None,
                "error": synth_errors.get(key, "no audio synthesised"),
            }
            continue
        pending.append((index, wav_path))

    t0 = time.time()
    hyps = judge.transcribe_many([p.read_bytes() for _, p in pending])
    for (index, _), hyp in zip(pending, hyps):
        ref = text_by_index[index]
        scores = compute_metrics(ref["text"], hyp)
        entries[str(index)] = {
            "text": ref["text"],
            "converted": ref.get("converted", ""),
            "hyp": hyp,
            "cer": scores["cer"],
            "wer": scores["wer"],
        }
    elapsed = time.time() - t0

    _save_transcriptions(transcriptions_csv(iso, voice), entries)

    cers = [e["cer"] for e in entries.values() if e.get("cer") is not None]
    wers = [e["wer"] for e in entries.values() if e.get("wer") is not None]
    if not cers:
        print("    no valid output")
        return {}

    sample_clip = next(
        (k for k in sorted(entries, key=lambda x: int(x))
         if entries[k].get("cer") is not None),
        None,
    )
    mean_cer = round(sum(cers) / len(cers), 4)
    mean_wer = round(sum(wers) / len(wers), 4)
    print(f"    CER {mean_cer:.4f} WER {mean_wer:.4f} "
          f"({len(cers)} valid of {len(entries)}, {elapsed:.0f}s)")
    return {
        "voice": voice,
        "lang": iso,
        "cer": mean_cer,
        "wer": mean_wer,
        "score": mean_cer,
        "samples": len(entries),
        "valid": len(cers),
        "sample_clip": sample_clip,
    }


def _load_pool(iso):
    cache = AUDIO_DIR / iso / "texts.json"
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


# ── Aggregation ──────────────────────────────────────────────────────────────


def assemble():
    """Recompute per-language + per-voice YAMLs and the matrix from CSVs."""
    import glob

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "voice").mkdir(parents=True, exist_ok=True)

    cells = {}
    for csv_path in sorted(glob.glob(str(TRANSCRIPTIONS_DIR / "*.csv"))):
        stem = Path(csv_path).stem  # {iso}_{voice}
        iso, _, voice = stem.rpartition("_")
        if not iso or not voice:
            continue
        cells[(iso, voice)] = _csv_summary(csv_path)

    if not cells:
        print("No transcriptions found — run scoring first.")
        return

    iso_voice = {}
    voice_iso = {}
    for (iso, voice), res in cells.items():
        iso_voice.setdefault(iso, {})[voice] = res
        voice_iso.setdefault(voice, {})[iso] = res

    for iso, vv in iso_voice.items():
        _write_language_yaml(iso, vv)
    for voice, ll in voice_iso.items():
        _write_voice_yaml(voice, ll)
    _write_matrix(voice_iso)

    print(f"\nAggregated {len(cells)} cells -> {BENCHMARK_DIR}")


def _csv_summary(csv_path):
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cers, wers = [], []
    sample_clip = None
    for r in rows:
        try:
            if r.get("cer") not in (None, ""):
                cer = float(r["cer"])
                if sample_clip is None:
                    sample_clip = r["sample_id"]
                cers.append(cer)
            if r.get("wer") not in (None, ""):
                wers.append(float(r["wer"]))
        except ValueError:
            pass
    return {
        "cer": round(sum(cers) / len(cers), 4) if cers else None,
        "wer": round(sum(wers) / len(wers), 4) if wers else None,
        "samples": len(rows),
        "valid": len(cers),
        "sample_clip": sample_clip,
    }


def _judge_meta(iso):
    spec = judge_for(iso)
    return {"model": spec["model"], "cer_on_real_speech": spec["judge_cer"]} if spec else None


def _write_language_yaml(iso, voices_by_iso):
    rows = [
        {
            "model": v,
            "language": voice_name(v),
            **{k: res[k] for k in res if k != "sample_clip"},
        }
        for v, res in sorted(
            voices_by_iso.items(), key=lambda kv: kv[1]["cer"] or 1e9)
    ]
    out = {
        "iso_639_3": iso,
        "language": ISO_TO_NAME[iso],
        "num_samples": NUM_SAMPLES,
        "scoring": "cer",
        "judge": _judge_meta(iso),
        "updated": today(),
        "benchmarks": rows,
        "sample_clips": {v: res.get("sample_clip")
                         for v, res in voices_by_iso.items()},
    }
    with open(bench_yaml(iso), "w", encoding="utf-8") as f:
        yaml.dump(out, f, Dumper=_NoAliasDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def _write_voice_yaml(voice, iso_rows):
    rows = [
        {
            "model": voice,
            "language": ISO_TO_NAME.get(iso, iso),
            **res,
        }
        for iso, res in sorted(
            iso_rows.items(), key=lambda kv: kv[1]["cer"] or 1e9)
    ]
    out = {
        "ietf": voice,
        "voice": voice_name(voice),
        "num_samples": NUM_SAMPLES,
        "scoring": "cer",
        "updated": today(),
        "languages": rows,
    }
    with open(voice_yaml(voice), "w", encoding="utf-8") as f:
        yaml.dump(out, f, Dumper=_NoAliasDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def _write_matrix(voice_iso):
    isos = sorted(LANGS)
    vlist = sorted(voice_iso)
    matrix = {}
    clips = {}
    voice_avg = {}
    for v in vlist:
        avg = [res["cer"] for res in voice_iso[v].values() if res["cer"] is not None]
        voice_avg[v] = round(sum(avg) / len(avg), 4) if avg else None
        matrix[v] = {
            iso: voice_iso[v].get(iso, {}).get("cer") for iso in isos
        }
        clips[v] = {
            iso: voice_iso[v].get(iso, {}).get("sample_clip") for iso in isos
        }
    out = {
        "num_samples": NUM_SAMPLES,
        "updated": today(),
        "languages": isos,
        "voices": vlist,
        "voice_avg_cer": voice_avg,
        "matrix": matrix,
        "clips": clips,
    }
    with open(BENCHMARK_DIR / "voice_matrix.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Wrote matrix ({len(vlist)} voices x {len(isos)} languages)")


# ── Language-level orchestrators (used by HF Job containers) ────────────────


def synthesize_language(iso, force=False, concurrency=CONCURRENCY, limit=None,
                        on_cell=None):
    """Synthesise every voice for *iso*. Returns (written, failed) totals.

    ``on_cell(voice, written, failed)`` is invoked after each voice so a
    caller (HF Job) can push progress incrementally instead of waiting for
    the full language to finish.
    """
    from .normalize import UniversalConverter

    converter = UniversalConverter()
    _lang_pool(iso, converter, force=force)
    total_w = total_f = 0
    for voice in voices():
        try:
            w, f = asyncio.run(synthesize_cell_async(
                voice, iso, converter, force=force,
                concurrency=concurrency, limit=limit,
            ))
            total_w += w
            total_f += f
            if on_cell is not None:
                try:
                    on_cell(voice, w, f)
                except Exception as e:
                    print(f"  on_cell failed for {voice}: {e}")
        except Exception as e:
            print(f"  ERROR synth {voice}->{iso}: {e}")
            total_f += 1
    return total_w, total_f


def score_language(iso, device=None, force=False, limit=None, verbose=True):
    """Score every voice for *iso* with that language's judge.

    Returns a list of per-voice summary dicts (safe to JSON-serialise).
    """
    device = device or DEVICE
    judge = None
    summaries = []
    try:
        if verbose:
            print(f"  Loading ASR judge for {iso} ...")
        judge = load_judge(iso, device=device)
        for voice in voices():
            try:
                res = score_cell(voice, iso, judge=judge,
                                force=force, limit=limit)
                if res:
                    summaries.append(res)
            except Exception as e:
                print(f"  ERROR score {voice}->{iso}: {e}")
    finally:
        if judge is not None:
            try:
                judge.cleanup()
            except Exception:
                pass
    return summaries


# ── Internal helpers ─────────────────────────────────────────────────────────


def _load_errors(model_dir):
    path = model_dir / ERRORS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_errors(model_dir, errors):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / ERRORS_FILE).write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_transcriptions(csv_path):
    if not csv_path.exists():
        return {}
    entries = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            hyp = row.get("hypothesis") or ""
            entry = {"text": row.get("reference", ""),
                     "converted": row.get("converted", ""),
                     "hyp": hyp}
            if hyp.startswith("ERROR: "):
                entry.update({"cer": None, "error": hyp[len("ERROR: "):]})
            else:
                try:
                    entry["cer"] = float(row["cer"])
                    entry["wer"] = float(row["wer"])
                except (TypeError, ValueError, KeyError):
                    entry["cer"] = None
            entries[row["sample_id"]] = entry
    return entries


def _save_transcriptions(csv_path, entries):
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(entries.items(), key=lambda kv: int(kv[0]))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "reference", "converted",
                         "hypothesis", "wer", "cer"])
        for key, entry in rows:
            hyp = entry.get("hyp", "")
            if entry.get("error") is not None:
                hyp = f"ERROR: {entry['error']}"
            writer.writerow([
                key,
                entry.get("text", ""),
                entry.get("converted", ""),
                hyp,
                entry.get("wer", ""),
                entry.get("cer", ""),
            ])
    return csv_path