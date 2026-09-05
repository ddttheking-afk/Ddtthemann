import numpy as np

from teachai.core.model import TeachableModel


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 0.3, size=(60, 2))
    b = rng.normal([3, 3], 0.3, size=(60, 2))
    c = rng.normal([0, 3], 0.3, size=(60, 2))
    X = np.vstack([a, b, c])
    y = ["a"] * 60 + ["b"] * 60 + ["c"] * 60
    return X, y


def test_learns_separable_classes():
    X, y = _blobs()
    model = TeachableModel(lr=0.2).fit(X, y, epochs=60)
    preds = model.predict(X)
    acc = np.mean([p == t for p, t in zip(preds, y)])
    assert acc > 0.95


def test_discovers_new_label_online():
    model = TeachableModel(lr=0.3)
    model.partial_fit([[0.0, 0.0]], ["a"])
    model.partial_fit([[5.0, 5.0]], ["b"])
    assert set(model.classes_) == {"a", "b"}
    # A brand new class introduced later should be accommodated.
    model.partial_fit([[-5.0, -5.0]], ["c"])
    assert "c" in model.classes_


def test_confidence_is_probability():
    X, y = _blobs()
    model = TeachableModel(lr=0.2).fit(X, y, epochs=60)
    label, conf = model.predict_one([0, 0])
    assert label == "a"
    assert 0.0 <= conf <= 1.0
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_save_and_load(tmp_path):
    X, y = _blobs()
    model = TeachableModel(lr=0.2).fit(X, y, epochs=40)
    path = tmp_path / "brain.npz"
    model.save(path)
    loaded = TeachableModel.load(path)
    assert loaded.classes_ == model.classes_
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
