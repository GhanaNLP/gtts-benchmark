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
   concurrently (gTTS is network-bound; default 24 in flight).
4. **Scoring** — the best ASR judge per language (Khaya / omniASR / griot — the same judges as
   [nsanku-tts-benchmark](https://github.com/GhanaNLP/nsanku-tts-benchmark)) transcribes every
   clip; each is scored as **CER** (and WER) against the *original* sentence. Lower is better.

The matrix is **69 voices × 12 languages × 200 samples**. Each cell is independently
incremental: re-running only synthesises/scores what is not already present.

## How the run is orchestrated (HuggingFace Jobs)

Like `nsanku-tts-benchmark`, synthesis and scoring run as **HuggingFace Jobs** under the
`ghananlpcommunity` namespace:

- **One synthesis job per language** — pulls existing clips for that language from the
  `ghananlpcommunity/gtts-benchmark-audio` dataset repo, synthesises all 69 voices, pushes
  clips back (resumable).
- **One scoring job per language** — pulls every voice's clips + existing transcriptions,
  transcribes all 69 voices with that language's judge, pushes `transcriptions/` and
  `results/{iso}.json`.
- Jobs are independent containers: a failure costs one unit, and the run survives this
  machine being closed. A job needs only an HF token (`HF_TOKEN`) and `KHAYA_API_KEY` for the
  Khaya-judged languages.
- **Images:** scoring reuses `ghcr.io/ghanaopenai/nsanku-tts-benchmark:asr` (identical judges)
  verbatim. Synthesis uses a tiny image, `ghcr.io/ghanaopenai/gtts-benchmark:synth`, built by
  [`GhanaOpenAI/gtts-benchmark-images`](https://github.com/GhanaOpenAI/gtts-benchmark-images)
  from `docker/Dockerfile.synth`.

### Submit the jobs

```bash
export HF_TOKEN=hf_...          # write access to ghananlpcommunity org (jobs + audio repo)
export KHAYA_API_KEY=...        # required for ada/dag/dga/fat/gjn/xsm scoring

python scripts/run_hf_jobs.py --stage synth --dry-run     # preview
python scripts/run_hf_jobs.py --stage synth               # 12 jobs
python scripts/run_hf_jobs.py --stage score --langs ewe,twi_asante
python scripts/run_hf_jobs.py --stage score               # 12 jobs
```

### Assemble and publish results

```bash
python scripts/assemble_benchmarks.py    # pulls results/*.json -> benchmarks/*
git add benchmarks && git commit -m "results" && git push
```

The HF Space reads `benchmarks/*.{yaml,json}` from this repo and the demo clips from the
audio dataset repo, so pushing the repo updates the leaderboard.

## Run a quick local smoke test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . python -m pip install "africa-g2p @ git+https://github.com/AfriSpeech/africa-g2p.git"
python pipeline.py --voices en sw --langs twi_asante --limit 3     # synth only, 3 samples
```

(Full local synthesis is possible too — `python pipeline.py` — but the clips then need
uploading before scoring, which the Jobs path does automatically; that's the supported flow.)

## Results

- `benchmarks/{iso}.yaml` — voices ranked per language (with the ASR judge and its own CER
  floor on real speech)
- `benchmarks/voice/{voice}.yaml` — languages ranked per voice
- `benchmarks/voice_matrix.json` — full voice × language CER matrix + demo-clip map
- `audit: transcriptions/{iso}_{voice}.csv` per clip in the audio dataset repo

## HF Space

[ghananlpcommunity/gtts-benchmark (Space)](https://huggingface.co/spaces/ghananlpcommunity/gtts-benchmark)
renders the leaderboard and matrix from the GitHub `benchmarks/` files and streams demo audio
from `ghananlpcommunity/gtts-benchmark-audio`.

Deploy (needs an HF token with write access to the org):

```bash
python -c "from huggingface_hub import create_repo, upload_folder
create_repo('ghananlpcommunity/gtts-benchmark', repo_type='space', space_sdk='static', exist_ok=True)
upload_folder(repo_id='ghananlpcommunity/gtts-benchmark', repo_type='space', folder_path='space')"
python scripts/push_audio.py
```

## Repo layout

```
benchmark/            benchmark library (config, dataset, normalize, synth,
                      asr judges, evaluate, job)
data/                 ASR judge registry (from nsanku)
audio/                synthesised clips (gitignored; lives in the HF dataset repo)
benchmarks/           results: per-language YAMLs, matrix JSON
transcriptions/       per-sample CSVs (gitignored; in the HF dataset repo)
docker/               synthesis image Dockerfile
scripts/              HF-job submitter, results assembler, audio uploader
space/                HF Space static app
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