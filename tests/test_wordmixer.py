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
        mixer.stir("dogg", n=3)
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
