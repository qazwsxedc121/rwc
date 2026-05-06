import pytest
from pathlib import Path


@pytest.fixture
def tiny_model_path():
    return Path(__file__).parent / "fixtures" / "tiny_glove.txt"


@pytest.fixture
def sample_vocab():
    return ["dog", "cat", "animal", "car", "vehicle", "truck", "house", "building", "tree", "plant"]
