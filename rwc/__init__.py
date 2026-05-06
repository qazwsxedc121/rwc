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
        vocab = self._engine.vocabulary()
        scored = [(w, _simple_score(word, w)) for w in vocab]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, s in scored[:max_suggestions] if s > 0]


def _simple_score(a: str, b: str) -> float:
    a_set = set(a)
    b_set = set(b)
    intersection = a_set & b_set
    union = a_set | b_set
    return len(intersection) / max(len(union), 1)


__all__ = [
    "WordMixer",
    "WordResult",
    "WordNotInVocabularyError",
    "ModelNotFoundError",
]
