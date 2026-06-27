import numpy as np
from PIL import Image

from teachai.vision import ImageSelector


def _make_image(path, color, noise_seed):
    rng = np.random.default_rng(noise_seed)
    base = np.array(color, dtype=np.float64)
    arr = np.clip(base + rng.normal(0, 12, size=(40, 40, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _build_dataset(root):
    colors = {"red": (200, 30, 30), "blue": (30, 30, 200), "green": (30, 200, 30)}
    for name, color in colors.items():
        d = root / name
        d.mkdir()
        for i in range(6):
            _make_image(d / f"{i}.png", color, noise_seed=hash((name, i)) % 1000)
    return colors


def test_teach_and_classify(tmp_path):
    _build_dataset(tmp_path)
    sel = ImageSelector().teach_from_folder(tmp_path, epochs=40)
    assert set(sel.model.classes_) == {"red", "blue", "green"}

    probe = tmp_path / "probe.png"
    _make_image(probe, (210, 20, 20), noise_seed=999)
    label, conf = sel.classify(probe)
    assert label == "red"
    assert conf > 0.5


def test_select_best_for_target(tmp_path):
    _build_dataset(tmp_path)
    sel = ImageSelector().teach_from_folder(tmp_path, epochs=40)

    pile = tmp_path / "pile"
    pile.mkdir()
    _make_image(pile / "a_red.png", (205, 25, 25), 1)
    _make_image(pile / "b_blue.png", (25, 25, 205), 2)
    _make_image(pile / "c_green.png", (25, 205, 25), 3)

    best = sel.select_best(pile.glob("*.png"), "blue")
    assert best.name == "b_blue.png"


def test_persistence_roundtrip(tmp_path):
    _build_dataset(tmp_path)
    sel = ImageSelector().teach_from_folder(tmp_path, epochs=30)
    model_path = tmp_path / "vision.npz"
    sel.save(model_path)

    probe = tmp_path / "probe.png"
    _make_image(probe, (25, 25, 205), 7)
    reloaded = ImageSelector.load(model_path)
    assert reloaded.classify(probe)[0] == "blue"
