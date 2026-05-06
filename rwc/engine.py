import numpy as np
from pathlib import Path
from rwc.types import ModelNotFoundError, WordResult


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
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / norms
        self.word_to_index = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(words)

    def vocabulary(self) -> list[str]:
        return list(self.word_to_index.keys())

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self.word_to_index:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.vectors[self.word_to_index[word]]

    def similarities(self, topic: str) -> np.ndarray:
        topic_vec = self.get_vector(topic)
        return self.vectors @ topic_vec

    @staticmethod
    def _distance_to_percentile(distance: float) -> tuple[float, float]:
        low = max(0.0, distance - 0.2)
        high = min(1.0, distance + 0.2)
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

        index_to_word = {v: k for k, v in self.word_to_index.items()}
        return [
            WordResult(word=index_to_word[i], similarity=float(similarities[i]))
            for i in chosen
        ]

    def describe(self, topic: str, n: int = 5) -> dict:
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
