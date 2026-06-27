"""Teach the AI to tell pictures apart, then have it *select* one.

This example invents three solid-colour "categories" so it runs with no
downloads. Swap in your own folders of real photos to do the same thing for
cats vs dogs, screenshots, memes -- anything.

Run:  python examples/teach_pictures.py
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from teachai.vision import ImageSelector


def paint(path, color, seed):
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color) + rng.normal(0, 12, (40, 40, 3)), 0, 255)
    Image.fromarray(arr.astype("uint8")).save(path)


def main() -> None:
    work = Path(tempfile.mkdtemp())
    palette = {"sunny": (240, 220, 40), "ocean": (30, 90, 200), "forest": (30, 160, 60)}
    for name, color in palette.items():
        (work / name).mkdir()
        for i in range(6):
            paint(work / name / f"{i}.png", color, seed=i)

    print("Teaching from example folders:", ", ".join(palette))
    sel = ImageSelector().teach_from_folder(work, epochs=40)

    pile = work / "pile"
    pile.mkdir()
    paint(pile / "photo1.png", (235, 215, 50), 1)   # sunny-ish
    paint(pile / "photo2.png", (40, 95, 195), 2)    # ocean-ish
    paint(pile / "photo3.png", (35, 165, 70), 3)    # forest-ish

    for target in palette:
        best = sel.select_best(pile.glob("*.png"), target)
        print(f"  best '{target}' picture -> {best.name}")


if __name__ == "__main__":
    main()
