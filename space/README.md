# gTTS-voices benchmark (GhanaNLP / ghananlpcommunity)

Who is the best Google Text-to-Speech voice at reading Ghanaian languages?

Every gTTS voice (all 69 supported by the library) is asked to read 200
sentences in each of the 12 Ghanaian languages.  Before synthesis each
sentence is converted to **africa-g2p universal graphemes** — the shared
cross-language African orthography — so every voice attempts exactly the same
spelling.  The clips are then transcribed by the best ASR judge per language
(Khaya / omniASR / griot, identical to nsanku-tts-benchmark) and scored as
**character error rate (CER)** against the original sentence.  Lower CER wins.

The result answers a practical question for low-resource speech work: **which
free Google voice should you route Ghanaian text through?**

## Data flow

- **Results** are read from `benchmarks/` on
  [Github.com/GhanaNLP/gtts-benchmark](https://github.com/GhanaNLP/gtts-benchmark)
  (assembled from HF Job results by `scripts/assemble_benchmarks.py`).
- **Demo clips** stream from the
  `ghananlpcommunity/gtts-benchmark-audio` dataset repo, at
  `audio/{iso}/{voice}/{idx:05d}.wav`.