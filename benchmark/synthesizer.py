"""gTTS synthesis wrapper (thread-safe, retrying with backoff)."""

import io
import logging
import random
import time

from gtts import gTTS

from .config import GTTS_TIMEOUT, GTTS_TLD

logger = logging.getLogger(__name__)

_RATE_LIMIT_HINTS = ("429", "too many", "quota", "rate limit",
                     "max requests", "try again later", "403")


def _is_rate_limit(err):
    return any(h in str(err).lower() for h in _RATE_LIMIT_HINTS)


class GTTSModel:
    """Synthesise text with Google Text-to-Speech.

    Each instance is one "model row" on the board: a gTTS voice (e.g. ``en``,
    ``fr-CA``, ``sw``).  The voice reads every Ghanaian language's converted
    pool.  ``synthesize`` is safe to call from multiple threads.

    Retries are exponential with jitter.  Rate-limit responses get a longer,
    gentler backoff since gTTS can hit Google's endpoint hard under
    concurrency.
    """

    def __init__(self, ietf, tld=GTTS_TLD, timeout=GTTS_TIMEOUT,
                 retries=6, backoff=1.5, jitter=0.4):
        self.ietf = ietf
        self.tld = tld
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.jitter = jitter

    def synthesize(self, text, lang=None):
        """Synthesise *text* and return WAV bytes, retrying transient errors.

        ``lang`` defaults to this voice's own code.
        """
        lang = lang or self.ietf
        last = None
        for attempt in range(self.retries):
            try:
                buf = io.BytesIO()
                tts = gTTS(text=text, lang=lang, tld=self.tld,
                           timeout=self.timeout, lang_check=False)
                tts.write_to_fp(buf)
                return buf.getvalue()
            except Exception as e:
                last = e
                if attempt < self.retries - 1:
                    if _is_rate_limit(e):
                        base = max(self.backoff * 4, 8.0)
                    else:
                        base = self.backoff
                    delay = base * (2 ** attempt) * random.uniform(
                        1 - self.jitter, 1 + self.jitter)
                    time.sleep(delay)
        raise last

    def cleanup(self):
        pass

    def __repr__(self):
        return f"<GTTSModel {self.ietf}>"