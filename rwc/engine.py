import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

from rwc.types import ModelNotFoundError, WordResult


def _char_trigrams(word: str) -> set[str]:
    padded = f"  {word}  "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class WordVectorEngine:
    def __init__(self, model_path: str | Path):
        self._path = Path(model_path)
        if not self._path.exists():
            raise ModelNotFoundError(str(self._path))
        self._load()

    def _open(self):
        if self._path.suffix == ".gz":
            return gzip.open(self._path, mode="rt", encoding="utf-8")
        return open(self._path, encoding="utf-8")

    def _load(self):
        words = []
        vectors = []
        known_dim = None

        with self._open() as f:
            first = f.readline().strip().split()
            if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
                known_dim = int(first[1])
            else:
                known_dim = self._count_trailing_floats(first)
                if known_dim > 0 and len(first) >= known_dim + 1:
                    word = ' '.join(first[:-known_dim])
                    vec = np.array([float(x) for x in first[-known_dim:]])
                    words.append(word)
                    vectors.append(vec)

            for line in f:
                parts = line.strip().split()
                if len(parts) < known_dim + 1:
                    continue
                word = ' '.join(parts[:-known_dim])
                vec = np.array([float(x) for x in parts[-known_dim:]])
                words.append(word)
                vectors.append(vec)

        self.vector_dim = vectors[0].shape[0]
        self.vectors = np.array(vectors)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / norms
        self.word_to_index = {w: i for i, w in enumerate(words)}
        self._index_to_word = {i: w for w, i in self.word_to_index.items()}
        self.vocab_size = len(words)
        self._trigram_index: dict[str, set[int]] = defaultdict(set)
        for i, word in enumerate(words):
            for trigram in _char_trigrams(word):
                self._trigram_index[trigram].add(i)

    @staticmethod
    def _count_trailing_floats(parts):
        count = 0
        for token in reversed(parts):
            try:
                float(token)
                count += 1
            except ValueError:
                break
        return count

    def vocabulary(self) -> list[str]:
        return list(self.word_to_index.keys())

    def find_similar_words(self, query: str, max_suggestions: int = 5) -> list[str]:
        query_trigrams = _char_trigrams(query)
        overlap: dict[int, int] = defaultdict(int)
        for trigram in query_trigrams:
            for idx in self._trigram_index.get(trigram, ()):
                overlap[idx] += 1
        if not overlap:
            return []
        scored = sorted(overlap.items(), key=lambda x: x[1], reverse=True)
        return [self._index_to_word[idx] for idx, _ in scored[:max_suggestions]]

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self.word_to_index:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.vectors[self.word_to_index[word]]

    def similarities(self, topic: str) -> np.ndarray:
        topic_vec = self.get_vector(topic)
        return self.vectors @ topic_vec

    @staticmethod
    def _distance_to_percentile(distance: float) -> tuple[float, float]:
        half_width = 0.003
        low = max(0.0, distance - half_width)
        high = min(1.0, distance + half_width)
        return (low, high)

    def stir(self, topic: str, distance: float = 0.5, n: int = 5) -> list[WordResult]:
        similarities = self.similarities(topic)
        topic_idx = self.word_to_index[topic]

        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != topic_idx]

        low_pct, high_pct = self._distance_to_percentile(distance)
        total = len(sorted_indices)
        lo = int(low_pct * total)
        hi = max(lo + 1, int(high_pct * total))

        candidates = sorted_indices[lo:hi]
        if len(candidates) == 0:
            candidates = sorted_indices

        chosen = np.random.choice(candidates, size=min(n, len(candidates)), replace=False)

        return [
            WordResult(word=self._index_to_word[i], similarity=float(similarities[i]))
            for i in chosen
        ]

    def describe(self, topic: str, n: int = 5) -> dict:
        similarities = self.similarities(topic)
        topic_idx = self.word_to_index[topic]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != topic_idx][:n]
        return {
            "topic": topic,
            "nearest": [
                WordResult(word=self._index_to_word[i], similarity=float(similarities[i]))
                for i in sorted_indices
            ]
        }
