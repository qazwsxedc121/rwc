"""RWC - Random Word Collector. A mechanical word mixing device."""

from __future__ import annotations

import re
from pathlib import Path

from rwc.engine import WordVectorEngine
from rwc.types import (
    WordResult,
    WordNotInVocabularyError,
    ModelNotFoundError,
)


def chinese_word_filter(word: str) -> bool:
    """Default quality filter for Chinese creative injection.

    Keeps only clean, short, readable Chinese words. Rejects:
      - Latin letters, digits, symbols, URLs
      - Words shorter than 1 or longer than 4 characters
      - Words containing non-CJK characters
    """
    if len(word) < 1 or len(word) > 4:
        return False
    if re.search(r"[a-zA-Z0-9]", word):
        return False
    if re.search(r"[<>\(\)\[\]\{\}@://\.\,\-\+\/\\\%\s]", word):
        return False
    for ch in word:
        cp = ord(ch)
        if not (0x3400 <= cp <= 0x9FFF or 0xE800 <= cp <= 0xF8FF
                or 0x2000 <= cp <= 0x206F
                or 0x3000 <= cp <= 0x303F
                or 0xFF00 <= cp <= 0xFFEF
                or 0x4E00 <= cp <= 0x9FFF):
            return False
    return True


def english_word_filter(word: str) -> bool:
    """Quality filter for English creative injection.

    Keeps only clean alphabetic words. Rejects:
      - Words shorter than 2 or longer than 12 characters
      - Pure uppercase acronyms (>2 chars)
      - Words with digits, symbols, or non-letter characters
    """
    if len(word) < 2 or len(word) > 12:
        return False
    if not re.match(r"^[a-zA-Z][a-zA-Z\-']*[a-zA-Z]$", word):
        return False
    if word.isupper() and len(word) > 2:
        return False
    return True


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

    def stir(
        self,
        topic: str,
        distance: float = 0.5,
        n: int = 5,
        *,
        word_filter: "Callable[[str], bool] | None" = chinese_word_filter,
    ) -> list[WordResult]:
        if not (0.0 <= distance <= 1.0):
            raise ValueError(f"distance must be 0.0-1.0, got {distance}")

        topic = topic.lower().strip()
        if topic not in self._engine.word_to_index:
            suggestions = self._find_suggestions(topic)
            raise WordNotInVocabularyError(topic, suggestions)

        return self._engine.stir(topic, distance=distance, n=n, word_filter=word_filter)

    def stir_batch(self, topics: list[str], distance: float = 0.5, n: int = 5) -> list[list[WordResult]]:
        return [self.stir(t, distance=distance, n=n) for t in topics]

    def describe(self, topic: str) -> dict:
        topic = topic.lower().strip()
        if topic not in self._engine.word_to_index:
            suggestions = self._find_suggestions(topic)
            raise WordNotInVocabularyError(topic, suggestions)
        return self._engine.describe(topic)

    def _find_suggestions(self, word: str, max_suggestions: int = 5) -> list[str]:
        return self._engine.find_similar_words(word, max_suggestions=max_suggestions)


__all__ = [
    "WordMixer",
    "WordResult",
    "WordNotInVocabularyError",
    "ModelNotFoundError",
    "chinese_word_filter",
    "english_word_filter",
]
