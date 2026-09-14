"""Load evaluation text samples from ghana-sentences.

Identical sampling behaviour to nsanku-tts-benchmark: rows are keyed by the
subset row index, cleaned with the same ``is_clean_sentence`` filter, so the
two benchmarks read the same sentences and can be compared directly.
"""

import os
import re

from datasets import load_dataset

from .config import GHANA_SENTENCES, NUM_SAMPLES

MIN_WORDS = 5
MAX_WORDS = 30

_REJECT_CHARS = re.compile(r"[\[\]{}<>|\\/=•·~^_*#@\d]")
_SENTENCE_END = (".", "!", "?")
_WORD_CHARS = re.compile(r"[^\W\d_]", re.UNICODE)


def is_clean_sentence(text):
    """Is *text* a self-contained sentence worth asking a TTS model to read?"""
    text = text.strip()
    if not text:
        return False

    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False

    if not text.endswith(_SENTENCE_END):
        return False
    if not text[0].isupper():
        return False

    letters = _WORD_CHARS.findall(text)
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False

    if _REJECT_CHARS.search(text):
        return False

    if ":" in text:
        return False

    return len(letters) / len(text) > 0.7


def load_text_samples(subset, offset=0, limit=NUM_SAMPLES, num_rows=None, clean=True):
    """Load *limit* speakable sentences for a ghana-sentences subset.

    Args:
        subset: subset name (twi-aku, dag, ...)
        offset: skip the first *offset* accepted samples.
        limit: maximum number of samples to return.
        num_rows: stop after scanning this many rows of the subset.
        clean: keep only rows passing :func:`is_clean_sentence`.

    Returns:
        list of {"text": str, "index": int} — index is the row index in the
        subset, used to dedupe across incremental runs.
    """
    limit = int(os.environ.get("GTTS_NUM_SAMPLES", NUM_SAMPLES)) if limit is None else limit
    ds = load_dataset(GHANA_SENTENCES, subset, split="train", streaming=True)
    samples = []
    accepted = 0
    for i, row in enumerate(ds):
        if num_rows is not None and i >= num_rows:
            break
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if clean and not is_clean_sentence(text):
            continue
        accepted += 1
        if accepted <= offset:
            continue
        samples.append({"text": text, "index": i})
        if len(samples) >= limit:
            break
    return samples