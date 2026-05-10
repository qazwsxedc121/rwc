# RWC — Random Word Collector

Mechanical word mixing device for creative LLM prompting.
Injects unexpected words from vector space to disrupt semantic inertia.

## Install

```bash
pip install -e .
```

Requires Python 3.10+, numpy, anthropic (for A/B testing).

## Quick start

```python
from rwc import WordMixer

mixer = WordMixer("path/to/vectors.txt")  # .txt or .txt.gz, GloVe or fastText

# Close semantic neighbors
results = mixer.stir("time", distance=0.0, n=5)

# Surprising but related (sweet spot: d=0.2~0.3)
results = mixer.stir("time", distance=0.2, n=5)

# Wild random jumps
results = mixer.stir("time", distance=0.5, n=5)

# Explore vocabulary
desc = mixer.describe("time")
print(desc["nearest"])
```

## Key features

- **Non-linear distance mapping** — d→d² keeps small distances near true neighbors
- **Default word quality filter** — rejects Latin text, digits, symbols, long compounds
- **Chinese-first** — built-in `chinese_word_filter`, CJK character validation
- **Multi-format support** — GloVe .txt, fastText .txt, gzipped .vec.gz
- **Multi-word phrases** — handles space-separated multi-word tokens
- **Fuzzy suggestions** — trigram index for fast "did you mean?" lookup
- **A/B test framework** — `rwc.ab_test` for scientific evaluation with blind LLM judging

## A/B testing

```bash
# Chinese
python -m rwc.ab_test --model-path vectors/tencent_light_zh.txt --rounds 20 --distance 0.2

# English
python -m rwc.ab_test --model-path vectors/glove.6B.300d.txt --rounds 20 --lang en

# Factorial (Temperature × RWC)
python -m rwc.factorial_test --model-path vectors/tencent_light_zh.txt --rounds 10
```

Uses DeepSeek Anthropic-compatible API. Requires `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL` in environment.

### Evaluation methodology

- **Minimal prompts** — single-word topic only, no scene-setting
- **Blind judging** — A/B labels randomized, judge has no condition knowledge
- **Five-dimension criteria** — precision, coherence, structure, emotional depth, intellectual weight
- **Stateless calls** — fresh API client per request, no conversation history

### Findings (T=1.0, fair judge, n≈100)

RWC achieves ~40-45% win rate against bare LLM responses.
The gap reflects a fundamental trade-off: mechanical randomness introduces
surprising imagery at the cost of coherence. The LLM judge, being an LLM,
prefers its own coherent output — which is exactly the inertia RWC is
designed to break.

## Getting word vectors

- English: [GloVe](https://nlp.stanford.edu/projects/glove/) (glove.6B.zip)
- Chinese: [Tencent AI Lab](https://ai.tencent.com/ailab/nlp/en/embedding.html) Chinese Word Vectors

Any GloVe-format or fastText-format .txt file works. Gzip-compressed files (.gz) are supported.

## API

### WordMixer

```python
mixer = WordMixer(model_path)
mixer.stir(topic, distance=0.5, n=5, word_filter=chinese_word_filter)
mixer.stir_batch(topics, distance=0.5, n=5)
mixer.describe(topic)
mixer.vocab_size
```

### PhraseMixer

```python
from rwc.phrase_mixer import PhraseMixer
pm = PhraseMixer(engine)
pm.surprise("荒原", template="nvn")       # → "荒原吞咽洋槐"
pm.surprises("荒原", n=5)                  # → multiple phrases
```

### Filters

```python
from rwc import chinese_word_filter
mixer.stir("time", word_filter=chinese_word_filter)  # default
mixer.stir("time", word_filter=None)                  # no filter
```

## License

MIT
