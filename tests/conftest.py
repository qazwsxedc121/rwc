import pytest
from pathlib import Path


@pytest.fixture
def tiny_model_path():
    return Path(__file__).parent / "fixtures" / "tiny_glove.txt"
