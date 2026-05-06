# RWC Design

## Motivation

LLMs have two structural biases that limit creative output:
1. Training on mainstream data distributions → tendency toward "safe" answers
2. Autoregressive next-token prediction → each step picks the most "reasonable" path, missing innovative connections that come from unrelated concepts

RWC is a **mechanical word mixing device** that forcibly injects unexpected word combinations into an LLM's thinking process, disrupting its deterministic inertia.

## Core Design

A Python library centered on a single class:

```python
from rwc import WordMixer

mixer = WordMixer("path/to/glove.6B.300d.txt")
results = mixer.stir("time", distance=0.8, n=5)
# -> ["screwdriver", "glacier", "lens", "ritual", "voltage"]
```

- **Input**: topic word + concept distance (0.0~1.0) + result count
- **Engine**: local word vector model (GloVe/Word2Vec), numpy-based cosine similarity
- **Output**: n words with their similarity scores, sampled from the distance-controlled similarity band
- **Modes of use**: SDK (primary), CLI and HTTP API as thin wrappers later

## Architecture

```
rwc/
├── rwc/
│   ├── __init__.py      # public API: from rwc import WordMixer
│   ├── engine.py        # vector loading, similarity, neighborhood search
│   ├── types.py         # type definitions + error types
│   └── data/            # word vector files (user-provided or auto-downloaded)
├── tests/
│   ├── test_engine.py
│   └── test_wordmixer.py
├── pyproject.toml
└── README.md
```

## Distance Parameter

| distance | behavior | sampled from |
|----------|----------|-------------|
| 0.0 | nearest N words | top of similarity ranking |
| 0.5 | random uniform | entire vocabulary |
| 1.0 | orthogonal/negative region | bottom of similarity ranking |

Default: 0.5 (random)

## Data Flow

```
Caller
  │  mixer.stir("time", distance=0.8, n=5)
  ▼
WordMixer.stir()
  │  1. Look up "time" vector in vocabulary
  │  2. Compute cosine similarity to all words (numpy matmul)
  │  3. Sample n words from the similarity band mapped by distance
  │  4. Return [WordResult, ...]
  ▼
list[WordResult]
```

## Performance

- 400K word vocabulary × dim 300: ~30-80ms via numpy BLAS (single query)
- Pre-normalized vectors at load time
- FAISS optional for scaling beyond ~1M words

## Error Handling

- Topic not in vocabulary → `WordNotInVocabularyError` with closest-match suggestions
- Model file not found → `ModelNotFoundError`
- Invalid distance → `ValueError`

## Testing

- Unit: cosine similarity correctness, distance band mapping
- Integration: real vector file (small fixture), verify stir result diversity
- Edge: missing words, n > vocabulary, distance extremes (0, 1)

## Out of Scope (MVP)

- CLI and HTTP API (build later on core library)
- Multi-language word vectors (configurable but not pre-packaged)
- Phrase-level mixing (n-grams)
- Persistence / save-history
