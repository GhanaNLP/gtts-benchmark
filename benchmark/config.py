"""Configuration for the gTTS-voices x Ghanaian-languages benchmark."""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data"
AUDIO_DIR = Path(os.environ.get("GTTS_AUDIO_DIR", str(ROOT / "audio")))
BENCHMARK_DIR = Path(os.environ.get("GTTS_RESULTS_DIR", str(ROOT / "benchmarks")))
TRANSCRIPTIONS_DIR = Path(
    os.environ.get("GTTS_TRANSCRIPTIONS_DIR", str(ROOT / "transcriptions")))

# Text source: same corpus as nsanku-tts-benchmark.
GHANA_SENTENCES = "ghanaopenai/ghana-sentences"
NUM_SAMPLES = int(os.environ.get("GTTS_NUM_SAMPLES", "200"))

# How many gTTS requests to keep in flight during synthesis.
CONCURRENCY = int(os.environ.get("GTTS_CONCURRENCY", "4"))

# ── The 12 Ghanaian languages ─────────────────────────────────────────────
# Keyed by the ISO-like codes used across Ghana Open AI benchmarks.
LANGS = {
    "ada": {"name": "Dangme", "subset": "ada", "g2p": "ada"},
    "dag": {"name": "Dagbani", "subset": "dag", "g2p": "dag"},
    "dga": {"name": "Dagaare", "subset": "dga", "g2p": "dgd"},
    "ewe": {"name": "Ewe", "subset": "ewe", "g2p": "ewe"},
    "fat": {"name": "Fante", "subset": "fat", "g2p": "fat"},
    "gaa": {"name": "Ga", "subset": "gaa", "g2p": "gaa"},
    "gjn": {"name": "Gonja", "subset": "gjn", "g2p": "gjn"},
    "gur": {"name": "Gurene", "subset": "gur", "g2p": "gur"},
    "nzi": {"name": "Nzema", "subset": "nzi", "g2p": "nzi"},
    "twi_akuapem": {"name": "Akuapem Twi", "subset": "twi-aku", "g2p": "twi"},
    "twi_asante": {"name": "Asante Twi", "subset": "twi-asa", "g2p": "twi"},
    "xsm": {"name": "Kasem", "subset": "xsm", "g2p": "xsm"},
}

SUBSET_TO_ISO = {info["subset"]: iso for iso, info in LANGS.items()}
ISO_TO_NAME = {iso: info["name"] for iso, info in LANGS.items()}

# ── Paths ────────────────────────────────────────────────────────────────────
ASR_JUDGES = DATA_DIR / "asr_judges.json"

# ── Synthesis / scoring ──────────────────────────────────────────────────────
GTTS_TLD = os.environ.get("GTTS_TLD", "com")
GTTS_TIMEOUT = float(os.environ.get("GTTS_TIMEOUT", "5"))
UNIVERSAL_CONVERT = os.environ.get("GTTS_UNIVERSAL", "1") == "1"
SAMPLE_RATE = 24000

# Judges: device for local judge kinds (griot / omniasr).
DEVICE = os.environ.get("GTTS_DEVICE", "cuda")

# Env secrets
HF_TOKEN = os.environ.get("HF_TOKEN", "")
KHAYA_API_KEY = os.environ.get("KHAYA_API_KEY", "")