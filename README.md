# gTTS-voices × Ghanaian languages benchmark

**Which free Google Text-to-Speech (gTTS) voice is best at reading Ghanaian languages?**

Google's Text-to-Speech voices are trained on ~69 languages — none of them Ghanaian. Yet the
voices are free and instantly available, so for many low-resource speech applications teams
grab the closest voice and hope. This benchmark measures that systematically.

## Method

1. **Text** — 200 sentences per Ghanaian language from
   [`ghanaopenai/ghana-sentences`](https://huggingface.co/datasets/ghanaopenai/ghana-sentences)
   (12 languages: Dangme, Dagbani, Dagaare, Ewe, Fante, Ga, Gonja, Gurene, Nzema, Akuapem Twi,
   Asante Twi, Kasem).
2. **Universal graphemes** — each sentence is converted with [africa-g2p](https://github.com/AfriSpeech/africa-g2p)
   to the *universal grapheme set*: the shared majority African orthography. Every voice then
   attempts the same, identical spelling, so scores across voices are comparable.
3. **Synthesis** — all 69 gTTS voices (the full `gtts.lang.tts_langs()` set) read every pool,
   concurrently (gTTS is network-bound; default 8 in flight).
4. **Scoring** — the best ASR judge per language (Khaya / omniASR / griot — the same judges as
   [nsanku-tts-benchmark](https://github.com/GhanaOpenAI/nsanku-tts-benchmark)) transcribes every
   clip; each is scored as **CER** (and WER) against the *original* sentence. Lower is better.

The matrix is **69 voices × 12 languages × 200 samples**. Each cell is independently
incremental: re-running only synthesises/scores what is not already on disk.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                     # or: pip install -r requirements.txt
pip install africa-g2p==0.2.0        # >=0.2.0 needed for the universal converter
```

For the heavy ASR judges (omniasr → ewe/gur/nzi, griot → gaa/twi) install their
dependencies on a GPU box (see `nsanku-tts-benchmark/docker/`).

## Run

```bash
# Full matrix: 69 voices x 12 languages
python pipeline.py

# Smoke test (2 voices, 1 language, 3 samples)
python pipeline.py --voices en sw --langs twi_asante --limit 3

# Synthesis only / scoring only / re-run
python pipeline.py --stage synth
python pipeline.py --stage score
python pipeline.py --force

# Tuning
python pipeline.py --concurrency 16          # gTTS requests in flight
GTTS_NUM_SAMPLES=500 python pipeline.py      # grow to 500 samples incrementally
```

### Env vars

| var | default | meaning |
|---|---|---|
| `GTTS_NUM_SAMPLES` | 200 | samples per language (`--limit` overrides for testing) |
| `GTTS_CONCURRENCY` | 8 | concurrent gTTS requests |
| `GTTS_TLD` | com | Google TLD (useful if `com` is blocked) |
| `GTTS_UNIVERSAL` | 1 | set `0` to skip africa-g2p conversion (baseline) |
| `KHAYA_API_KEY` | — | required for the Khaya ASR judge |
| `GTTS_DEVICE` | cuda | device for griot/omniasr judges |

## Results

- `benchmarks/{iso}.yaml` — voices ranked per language (with the ASR judge and its own
  CER floor on real speech)
- `benchmarks/voice/{voice}.yaml` — languages ranked per voice
- `benchmarks/voice_matrix.json` — full voice × language CER matrix + demo-clip map
- `transcriptions/{iso}_{voice}.csv` — every clip's reference, universally-converted text,
  transcription, WER and CER (auditable, same format as nsanku)

## HF Space

[GhanaOpenAI/gtts-benchmark (Space)](https://huggingface.co/spaces/GhanaOpenAI/gtts-benchmark)
renders the leaderboard and matrix from the GitHub `benchmarks/` files and streams demo audio
from `GhanaOpenAI/gtts-benchmark-audio`.

Deploy:

```bash
python -c "from huggingface_hub import create_repo, upload_folder
create_repo('GhanaOpenAI/gtts-benchmark', repo_type='space', space_sdk='static', exist_ok=True)
upload_folder(repo_id='GhanaOpenAI/gtts-benchmark', repo_type='space', folder_path='space')"
python scripts/push_audio.py
```

## Repo layout

```
benchmark/            benchmark library (config, dataset, normalize, synth,
                      asr judges, evaluate, pipeline)
data/                 ASR judge registry (from nsanku)
audio/                synthesised clips (gitignored)
benchmarks/           results: per-language YAMLs, matrix JSON
transcriptions/       per-sample CSVs
space/                HF Space static app
scripts/              audio upload helper
```

## Why universal graphemes?

Ghanaian orthographies are largely phonemic, but each uses its own digraph/tone conventions
(`ɔ ɛ ɔɔ`, `ky/ty`, `hy/shy`...). africa-g2p normalises them into one common spelling
(`o ny a n k o p o n`). Feeding every gTTS voice the *same* grapheme sequence isolates the
quality of each voice's phone realisation — which is exactly the property we want to rank.

## Acknowledged caveats

- A voice's score is entangled with the judge's own language coverage (griots, etc.). The
  per-language YAML records each judge's CER on real speech as the floor.
- The universal set maps a few phonemes lossily by design (e.g. `/ɔ/` and `/o/`).
- gTTS is a network API; results depend on the Google endpoint's current voices.