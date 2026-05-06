# RWC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python SDK that uses local word vectors to sample words at controlled semantic distances from a topic word.

**Architecture:** Single Python package `rwc`. Core math in `engine.py` (vector loading, cosine similarity, distance-band sampling). Public API `WordMixer` in `__init__.py` wraps the engine with input validation and user-friendly errors.

**Tech Stack:** Python 3.10+, numpy, scipy (optional: for fast cosine distance), pytest

---

### Task 1: Project skeleton and test fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/tiny_glove.txt` (test data, 10 words × 5 dims)
- Modify: `README.md` (brief usage)

**Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "rwc"
version = "0.1.0"
description = "Random Word Collector - mechanical word mixing device for creative LLM prompting"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[tool.setuptools.packages.find]
include = ["rwc*"]
```

**Step 2: Write test fixture (tiny_glove.txt)**

Create `tests/fixtures/` directory and `tiny_glove.txt` with 10 hand-crafted words × 5 dimensions in GloVe format:

```
dog 0.8 0.6 0.4 0.2 0.0
cat 0.7 0.5 0.5 0.1 0.1
animal 0.75 0.55 0.45 0.15 0.05
car -0.5 -0.3 0.6 0.7 0.8
vehicle -0.4 -0.2 0.5 0.6 0.7
truck -0.6 -0.4 0.55 0.65 0.75
house 0.1 0.9 -0.3 -0.5 -0.7
building 0.15 0.85 -0.25 -0.45 -0.65
tree -0.8 0.2 -0.8 -0.3 0.1
plant -0.7 0.3 -0.7 -0.2 0.15
```

Words designed so: dog/cat/animal cluster, car/vehicle/truck cluster, house/building cluster, tree/plant cluster. Cross-cluster similarity is low.

**Step 3: Write conftest.py**

```python
import pytest
from pathlib import Path

@pytest.fixture
def tiny_model_path():
    return Path(__file__).parent / "fixtures" / "tiny_glove.txt"

@pytest.fixture
def sample_vocab():
    return ["dog", "cat", "animal", "car", "vehicle", "truck", "house", "building", "tree", "plant"]
```

**Step 4: Verify project structure**

Run: `python -c "import numpy; print(numpy.__version__)"`
Expected: numpy version printed

**Step 5: Commit**

```bash
git add pyproject.toml tests/ README.md
git commit -m "feat: project skeleton with test fixtures"
```

---

### Task 2: types.py — data types and error classes

**Files:**
- Create: `rwc/__init__.py` (empty placeholder)
- Create: `rwc/types.py`
- Create: `tests/test_types.py`

**Step 1: Write the failing test**

```python
import pytest
from rwc.types import WordResult, WordNotInVocabularyError, ModelNotFoundError

def test_word_result_creation():
    r = WordResult(word="dog", similarity=0.85)
    assert r.word == "dog"
    assert r.similarity == 0.85

def test_word_result_immutability():
    r = WordResult(word="dog", similarity=0.85)
    with pytest.raises(Exception):
        r.word = "cat"

def test_word_not_in_vocabulary_error():
    err = WordNotInVocabularyError("xyz", suggestions=["abc", "def"])
    assert "xyz" in str(err)
    assert "abc" in str(err)

def test_model_not_found_error():
    err = ModelNotFoundError("/nonexistent/path.txt")
    assert "/nonexistent/path.txt" in str(err)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'rwc.types'"

**Step 3: Write minimal implementation**

`rwc/types.py`:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class WordResult:
    word: str
    similarity: float

class WordNotInVocabularyError(KeyError):
    def __init__(self, word: str, suggestions: list[str] | None = None):
        self.word = word
        self.suggestions = suggestions or []
        msg = f"'{word}' not in vocabulary."
        if self.suggestions:
            msg += f" Did you mean: {', '.join(self.suggestions[:5])}?"
        super().__init__(msg)

class ModelNotFoundError(FileNotFoundError):
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Model file not found: {path}")
```

`rwc/__init__.py`:
```python
"""RWC - Random Word Collector. A mechanical word mixing device."""

from rwc.types import WordResult, WordNotInVocabularyError, ModelNotFoundError

__all__ = ["WordResult", "WordNotInVocabularyError", "ModelNotFoundError"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add rwc/ tests/test_types.py
git commit -m "feat: add types and error classes"
```

---

### Task 3: engine.py — load word vectors

**Files:**
- Create: `rwc/engine.py`
- Create: `tests/test_engine.py`

**Step 1: Write the failing test**

```python
import numpy as np
import pytest
from rwc.engine import WordVectorEngine
from rwc.types import ModelNotFoundError

def test_load_vectors(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    assert engine.vocab_size == 10
    assert engine.vector_dim == 5
    assert "dog" in engine.word_to_index
    assert engine.word_to_index["dog"] is not None

def test_load_vectors_returns_normalized(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    norms = np.linalg.norm(engine.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_load_nonexistent_file():
    with pytest.raises(ModelNotFoundError):
        WordVectorEngine("/nonexistent/path.txt")

def test_vocab_list(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    words = engine.vocabulary()
    assert len(words) == 10
    assert "dog" in words
    assert "cat" in words
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

`rwc/engine.py`:
```python
import numpy as np
from pathlib import Path
from rwc.types import ModelNotFoundError

class WordVectorEngine:
    def __init__(self, model_path: str | Path):
        self._path = Path(model_path)
        if not self._path.exists():
            raise ModelNotFoundError(str(self._path))
        self._load()

    def _load(self):
        words = []
        vectors = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]])
                words.append(word)
                vectors.append(vec)

        self.vector_dim = vectors[0].shape[0]
        self.vectors = np.array(vectors)
        # Normalize
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / norms
        self.word_to_index = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(words)

    def vocabulary(self) -> list[str]:
        return list(self.word_to_index.keys())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add rwc/engine.py tests/test_engine.py
git commit -m "feat: add vector loading with normalization"
```

---

### Task 4: engine.py — cosine similarity and index lookup

**Files:**
- Modify: `rwc/engine.py`
- Modify: `tests/test_engine.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/test_engine.py`:

```python
def test_get_vector(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    vec = engine.get_vector("dog")
    assert vec.shape == (5,)
    assert np.allclose(np.linalg.norm(vec), 1.0, atol=1e-6)

def test_get_vector_missing(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    with pytest.raises(KeyError):
        engine.get_vector("xyz123")

def test_cosine_similarities_dog(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    results = engine.similarities("dog")
    # dog should be most similar to itself
    dog_idx = engine.word_to_index["dog"]
    assert results[dog_idx] > 0.999  # self-similarity ~1.0
    
    # cat and animal should be more similar to dog than car or house
    cat_idx = engine.word_to_index["cat"]
    car_idx = engine.word_to_index["car"]
    assert results[cat_idx] > results[car_idx]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_get_vector -v`
Expected: FAIL with "AttributeError: 'WordVectorEngine' object has no attribute 'get_vector'"

**Step 3: Write minimal implementation**

Add to `WordVectorEngine` in `engine.py`:

```python
    def get_vector(self, word: str) -> np.ndarray:
        if word not in self.word_to_index:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.vectors[self.word_to_index[word]]

    def similarities(self, topic: str) -> np.ndarray:
        """Return cosine similarities of topic word to all words."""
        topic_vec = self.get_vector(topic)
        # All vectors are normalized, so dot product = cosine similarity
        return self.vectors @ topic_vec
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add rwc/engine.py tests/test_engine.py
git commit -m "feat: add cosine similarity computation"
```

---

### Task 5: engine.py — distance-band sampling

**Files:**
- Modify: `rwc/engine.py`
- Modify: `tests/test_engine.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/test_engine.py`:

```python
def test_distance_band_mapping():
    """Test internal mapping from distance param to similarity percentile range."""
    # distance 0.0 → top (high similarity)
    low, high = WordVectorEngine._distance_to_percentile(0.0)
    assert high > 0.95  # should sample from top
    
    # distance 0.5 → full range
    low, high = WordVectorEngine._distance_to_percentile(0.5)
    assert low < 0.2
    assert high > 0.8
    
    # distance 1.0 → bottom (low similarity, including negative)
    low, high = WordVectorEngine._distance_to_percentile(1.0)
    assert low < 0.1  # includes the very bottom

def test_stir_returns_n_words(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    results = engine.stir("dog", distance=0.5, n=3)
    assert len(results) == 3
    for r in results:
        assert hasattr(r, 'word')
        assert hasattr(r, 'similarity')

def test_stir_excludes_topic_word(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    results = engine.stir("dog", distance=0.5, n=10)
    words = {r.word for r in results}
    assert "dog" not in words

def test_stir_distance_0_returns_closest(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    results = engine.stir("dog", distance=0.0, n=3)
    words = [r.word for r in results]
    # cat and animal should be in top results for dog
    assert "cat" in words or "animal" in words

def test_stir_distance_1_returns_farthest(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    results = engine.stir("dog", distance=1.0, n=3)
    words = [r.word for r in results]
    similarities = [r.similarity for r in results]
    # Results should have low similarity
    assert all(s < 0.5 for s in similarities)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_distance_band_mapping -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

Add to `WordVectorEngine` in `engine.py`:

```python
    @staticmethod
    def _distance_to_percentile(distance: float) -> tuple[float, float]:
        """Map 0.0-1.0 distance to (low_percentile, high_percentile) range."""
        # distance 0.0: top 10% (most similar)
        # distance 0.5: full range
        # distance 1.0: bottom 10% (least similar)
        low = distance * 0.9  # e.g. distance=1.0 → low=0.9 (top of bottom 10%)
        high = min(1.0, low + 0.1 + (1.0 - distance) * 0.9)
        # When distance=0.5: low=0.45, high=1.0 → full range
        # When distance=0.0: low=0.0, high=1.0 → also full range, but we'll weight
        return (low, high)

    def stir(self, topic: str, distance: float = 0.5, n: int = 5) -> list:
        from rwc.types import WordResult

        similarities = self.similarities(topic)
        # Exclude the topic word itself
        topic_idx = self.word_to_index[topic]
        mask = np.ones(self.vocab_size, dtype=bool)
        mask[topic_idx] = False

        # Get sorted indices by similarity
        sorted_indices = np.argsort(similarities)[::-1]  # descending
        # Remove topic from sorted list
        sorted_indices = [i for i in sorted_indices if i != topic_idx]

        low_pct, high_pct = self._distance_to_percentile(distance)
        total = len(sorted_indices)
        lo = int(low_pct * total)
        hi = max(lo + 1, int(high_pct * total))

        # Sample from the band
        candidates = sorted_indices[lo:hi]
        if len(candidates) == 0:
            candidates = sorted_indices

        chosen = np.random.choice(candidates, size=min(n, len(candidates)), replace=False)

        index_to_word = {v: k for k, v in self.word_to_index.items()}
        return [
            WordResult(word=index_to_word[i], similarity=float(similarities[i]))
            for i in chosen
        ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (12 tests)

**Step 5: Commit**

```bash
git add rwc/engine.py tests/test_engine.py
git commit -m "feat: add distance-band sampling in stir()"
```

---

### Task 6: engine.py — describe method

**Files:**
- Modify: `rwc/engine.py`
- Modify: `tests/test_engine.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_engine.py`:

```python
def test_describe(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    desc = engine.describe("dog")
    # Should return top similar words
    assert desc["topic"] == "dog"
    assert len(desc["nearest"]) > 0
    assert desc["nearest"][0].word in ["cat", "animal"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_describe -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

Add to `WordVectorEngine` in `engine.py`:

```python
    def describe(self, topic: str, n: int = 5) -> dict:
        """Return top N nearest words for debugging/validation."""
        from rwc.types import WordResult
        similarities = self.similarities(topic)
        topic_idx = self.word_to_index[topic]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != topic_idx][:n]
        index_to_word = {v: k for k, v in self.word_to_index.items()}
        return {
            "topic": topic,
            "nearest": [
                WordResult(word=index_to_word[i], similarity=float(similarities[i]))
                for i in sorted_indices
            ]
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (13 tests)

**Step 5: Commit**

```bash
git add rwc/engine.py tests/test_engine.py
git commit -m "feat: add describe() for vocabulary exploration"
```

---

### Task 7: WordMixer public API

**Files:**
- Modify: `rwc/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_wordmixer.py`:

```python
import pytest
from rwc import WordMixer
from rwc.types import WordResult, WordNotInVocabularyError, ModelNotFoundError

def test_wordmixer_init(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    assert mixer.model_path == tiny_model_path
    assert mixer.vocab_size == 10

def test_wordmixer_stir(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    results = mixer.stir("dog", distance=0.3, n=3)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, WordResult)
        assert r.word != "dog"

def test_wordmixer_stir_batch(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    results = mixer.stir_batch(["dog", "car"], distance=0.5, n=3)
    assert len(results) == 2
    assert len(results[0]) == 3
    assert len(results[1]) == 3

def test_wordmixer_invalid_topic(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    with pytest.raises(WordNotInVocabularyError) as exc:
        mixer.stir("xyz123", n=3)
    assert exc.value.suggestions

def test_wordmixer_invalid_distance(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    with pytest.raises(ValueError, match="distance"):
        mixer.stir("dog", distance=1.5)

def test_wordmixer_describe(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    desc = mixer.describe("dog")
    assert desc["topic"] == "dog"
    assert len(desc["nearest"]) > 0

def test_wordmixer_model_not_found():
    with pytest.raises(ModelNotFoundError):
        WordMixer("/nonexistent/path.txt")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wordmixer.py -v`
Expected: FAIL with "ImportError: cannot import name 'WordMixer'"

**Step 3: Write WordMixer**

Replace `rwc/__init__.py`:

```python
"""RWC - Random Word Collector. A mechanical word mixing device."""

from pathlib import Path
from rwc.engine import WordVectorEngine
from rwc.types import (
    WordResult,
    WordNotInVocabularyError,
    ModelNotFoundError,
)

class WordMixer:
    """Mechanical word mixing device for creative LLM prompting.

    Usage:
        mixer = WordMixer("path/to/glove.6B.300d.txt")
        results = mixer.stir("time", distance=0.8, n=5)
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelNotFoundError(str(self.model_path))
        self._engine = WordVectorEngine(self.model_path)

    @property
    def vocab_size(self) -> int:
        return self._engine.vocab_size

    def stir(self, topic: str, distance: float = 0.5, n: int = 5) -> list[WordResult]:
        if not (0.0 <= distance <= 1.0):
            raise ValueError(f"distance must be 0.0-1.0, got {distance}")

        topic = topic.lower().strip()
        if topic not in self._engine.word_to_index:
            suggestions = self._find_suggestions(topic)
            raise WordNotInVocabularyError(topic, suggestions)

        return self._engine.stir(topic, distance=distance, n=n)

    def stir_batch(self, topics: list[str], distance: float = 0.5, n: int = 5) -> list[list[WordResult]]:
        return [self.stir(t, distance=distance, n=n) for t in topics]

    def describe(self, topic: str) -> dict:
        topic = topic.lower().strip()
        if topic not in self._engine.word_to_index:
            suggestions = self._find_suggestions(topic)
            raise WordNotInVocabularyError(topic, suggestions)
        return self._engine.describe(topic)

    def _find_suggestions(self, word: str, max_suggestions: int = 5) -> list[str]:
        """Find closest words by edit distance (simple substring match for MVP)."""
        vocab = self._engine.vocabulary()
        scored = [(w, _simple_score(word, w)) for w in vocab]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, s in scored[:max_suggestions] if s > 0]


def _simple_score(a: str, b: str) -> int:
    """Simple character overlap score. Replace with edit distance later if needed."""
    a_set = set(a)
    b_set = set(b)
    intersection = a_set & b_set
    return len(intersection) / max(len(a_set | b_set), 1)


__all__ = [
    "WordMixer",
    "WordResult",
    "WordNotInVocabularyError",
    "ModelNotFoundError",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wordmixer.py -v`
Expected: PASS (7 tests)

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS (24 tests total)

**Step 6: Commit**

```bash
git add rwc/__init__.py tests/test_wordmixer.py
git commit -m "feat: WordMixer public API with validation"
```

---

### Task 8: Integration test and final verification

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
import pytest
from rwc import WordMixer

def test_full_flow_with_real_usage(tiny_model_path):
    """Simulate a real usage scenario: agent using RWC for creative prompting."""
    mixer = WordMixer(tiny_model_path)

    # Close-range: "dog" at distance=0.0 should return cat/animal
    close = mixer.stir("dog", distance=0.0, n=2)
    close_words = {r.word for r in close}
    assert len(close_words & {"cat", "animal"}) >= 1

    # Mid-range: should be diverse
    mid = mixer.stir("dog", distance=0.5, n=3)
    assert len(mid) == 3

    # Far-range: should pull from different clusters
    far = mixer.stir("dog", distance=1.0, n=3)
    far_words = {r.word for r in far}
    # These should be from non-animal clusters
    assert len(far_words & {"car", "vehicle", "truck", "house", "building", "tree", "plant"}) >= 1

def test_batch_stir(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    results = mixer.stir_batch(["dog", "car", "house"], distance=0.0, n=2)
    assert len(results) == 3
    # Each topic's nearest should be in the same cluster
    dog_cluster = {r.word for r in results[0]}
    assert len(dog_cluster & {"cat", "animal"}) >= 1
```

**Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (2 integration tests)

**Step 3: Run full suite one last time**

Run: `pytest tests/ -v`
Expected: PASS (26 tests total)

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full stir flow"
```

---

### Task 9: README

**Files:**
- Modify: `README.md`

**Step 1: Write README**

```markdown
# random word collector

Mechanical word mixing device for creative LLM prompting.

## Install

```bash
pip install -e .
```

Requires Python 3.10+, numpy.

## Usage

```python
from rwc import WordMixer

mixer = WordMixer("path/to/glove.6B.300d.txt")

# Close semantic neighbors
results = mixer.stir("time", distance=0.1, n=5)

# Wild random jumps
results = mixer.stir("time", distance=0.9, n=5)

# Batch
results = mixer.stir_batch(["time", "love", "machine"], distance=0.7, n=3)

# Explore
desc = mixer.describe("time")
print(desc["nearest"])
```

## Getting word vectors

Download GloVe vectors from https://nlp.stanford.edu/projects/glove/.
Any GloVe-format .txt file works.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add usage README"
```

---

## Implementation Order

```
Task 1: Project skeleton + fixtures  ──┐
Task 2: types.py                       ├─ Foundation
Task 3: engine.py - load vectors       │
Task 4: engine.py - similarities       ├─ Core math
Task 5: engine.py - stir sampling      │
Task 6: engine.py - describe           ──┘
Task 7: WordMixer public API           ──┐ Public face
Task 8: Integration tests              ──┘
Task 9: README                         ──  Polish
```

Each task builds on the previous. Tasks 1-3 can be done in any order relative to each other. Tasks 4-6 are strictly sequential within engine.py. Task 7 depends on all engine tasks. Task 8 depends on 7. Task 9 is independent.
