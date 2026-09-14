"""Universal grapheme conversion via africa-g2p.

Each Ghanaian sample text is converted to africa-g2p's universal grapheme set
— the cross-language majority African orthography — before synthesis.  Any of
the 69 gTTS voices can then attempt to read it: the universal spelling is the
same for every voice, so a voice's score reflects how well it realises that
shared orthography.
"""

from .config import LANGS


class UniversalConverter:
    """Converts Ghanaian text -> universal graphemes."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._converters = {}

    def _get(self, iso):
        if iso in self._converters:
            return self._converters[iso]
        conv = None
        g2p_code = LANGS.get(iso, {}).get("g2p")
        if g2p_code:
            try:
                from africa_g2p import GraphemeConverter, UNIVERSAL

                conv = GraphemeConverter(g2p_code, UNIVERSAL)
            except Exception:
                conv = None
        self._converters[iso] = conv
        return conv

    def is_supported(self, iso):
        return self._get(iso) is not None

    def convert_text(self, iso, text):
        if not self.enabled:
            return text
        conv = self._get(iso)
        if conv is None:
            return text
        try:
            return conv.convert(text)
        except Exception:
            return text

    def convert_batch(self, iso, texts):
        if not self.enabled:
            return list(texts)
        conv = self._get(iso)
        if conv is None:
            return list(texts)
        out = []
        for t in texts:
            try:
                out.append(conv.convert(t))
            except Exception:
                out.append(t)
        return out