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
