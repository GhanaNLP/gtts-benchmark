"""gTTS synthesis wrapper (thread-safe, retrying)."""

import io
import logging
import time

from gtts import gTTS

from .config import GTTS_TIMEOUT, GTTS_TLD

logger = logging.getLogger(__name__)


class GTTSModel:
    """Synthesise text with Google Text-to-Speech.

    Each instance is one "model row" on the board: a gTTS voice (e.g. ``en``,
    ``fr-CA``, ``sw``).  The voice reads every Ghanaian language's converted
    pool.  ``synthesize`` is safe to call from multiple threads.
    """

    def __init__(self, ietf, tld=GTTS_TLD, timeout=GTTS_TIMEOUT,
                 retries=3, backoff=0.5):
        self.ietf = ietf
        self.tld = tld
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

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
                    time.sleep(self.backoff * (2 ** attempt))
        raise last

    def cleanup(self):
        pass

    def __repr__(self):
        return f"<GTTSModel {self.ietf}>"