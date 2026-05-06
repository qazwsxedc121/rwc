# random word collector

Mechanical word mixing device for creative LLM prompting.

## Install

```bash
pip install -e .
```

Requires Python 3.10+, numpy.

## Usage

```python
from rwc import WordMixer

mixer = WordMixer("path/to/glove.6B.300d.txt")

# Close semantic neighbors
results = mixer.stir("time", distance=0.1, n=5)

# Wild random jumps
results = mixer.stir("time", distance=0.9, n=5)

# Batch
results = mixer.stir_batch(["time", "love", "machine"], distance=0.7, n=3)

# Explore vocabulary
desc = mixer.describe("time")
print(desc["nearest"])
```

## Getting word vectors

Download GloVe vectors from https://nlp.stanford.edu/projects/glove/.
Any GloVe-format .txt file works.
