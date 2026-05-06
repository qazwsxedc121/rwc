from rwc import WordMixer


def test_full_flow_with_real_usage(tiny_model_path):
    """Simulate a real usage scenario: agent using RWC for creative prompting."""
    mixer = WordMixer(tiny_model_path)

    close = mixer.stir("dog", distance=0.0, n=2)
    close_words = {r.word for r in close}
    assert len(close_words & {"cat", "animal"}) >= 1

    mid = mixer.stir("dog", distance=0.5, n=3)
    assert len(mid) == 3

    far = mixer.stir("dog", distance=1.0, n=3)
    far_words = {r.word for r in far}
    assert len(far_words & {"car", "vehicle", "truck", "house", "building", "tree", "plant"}) >= 1


def test_batch_stir(tiny_model_path):
    mixer = WordMixer(tiny_model_path)
    results = mixer.stir_batch(["dog", "car", "house"], distance=0.0, n=2)
    assert len(results) == 3
    dog_cluster = {r.word for r in results[0]}
    assert len(dog_cluster & {"cat", "animal"}) >= 1
