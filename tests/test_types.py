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
