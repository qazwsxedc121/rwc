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
    dog_idx = engine.word_to_index["dog"]
    assert results[dog_idx] > 0.999

    cat_idx = engine.word_to_index["cat"]
    car_idx = engine.word_to_index["car"]
    assert results[cat_idx] > results[car_idx]


def test_distance_band_mapping():
    low, high = WordVectorEngine._distance_to_percentile(0.0)
    assert low == 0.0
    assert high < 0.3

    low, high = WordVectorEngine._distance_to_percentile(0.5)
    assert 0.2 < low < 0.4
    assert 0.6 < high < 0.8

    low, high = WordVectorEngine._distance_to_percentile(1.0)
    assert low > 0.7


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
    results = engine.stir("dog", distance=0.0, n=2)
    words = [r.word for r in results]
    assert "cat" in words or "animal" in words


def test_stir_distance_1_returns_farthest(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    results = engine.stir("dog", distance=1.0, n=3)
    similarities = [r.similarity for r in results]
    assert all(s < 0.5 for s in similarities)


def test_describe(tiny_model_path):
    engine = WordVectorEngine(tiny_model_path)
    desc = engine.describe("dog")
    assert desc["topic"] == "dog"
    assert len(desc["nearest"]) > 0
    assert desc["nearest"][0].word in ["cat", "animal"]
