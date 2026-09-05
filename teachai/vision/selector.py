"""Teach the AI to recognise and *select* pictures.

Workflow:
  1. Show it example images grouped by what they are. Either point it at a
     folder whose subfolders are the categories, or teach images one by one.
  2. Ask it to classify a new picture (with a confidence score).
  3. Ask it to *select* the best picture for a target category out of a pile.

Example
-------
>>> sel = ImageSelector()
>>> sel.teach_from_folder("photos")      # photos/cat/*.jpg, photos/dog/*.jpg
>>> sel.classify("mystery.jpg")          # ("cat", 0.93)
>>> sel.select_best(glob("pile/*.jpg"), "cat")   # path of the most cat-like
"""

from __future__ import annotations

from pathlib import Path
from typing import Hashable, Iterable, List, Sequence, Tuple

import numpy as np

from ..core.features import DEFAULT_SIZE, image_to_vector
from ..core.model import TeachableModel

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


class ImageSelector:
    """A teachable image classifier and selector."""

    def __init__(self, size: Tuple[int, int] = DEFAULT_SIZE,
                 model: TeachableModel | None = None) -> None:
        self.size = size
        self.model = model or TeachableModel(lr=0.5, l2=1e-4)

    # ------------------------------------------------------------------ #
    # Teaching
    # ------------------------------------------------------------------ #
    def teach(self, image_path: str | Path, label: Hashable) -> "ImageSelector":
        """Teach a single labelled example."""
        vec = image_to_vector(image_path, self.size)
        self.model.partial_fit([vec], [label])
        return self

    def teach_many(self, image_paths: Sequence, labels: Sequence[Hashable],
                   epochs: int = 25) -> "ImageSelector":
        """Teach from a list of images and matching labels."""
        vecs = [image_to_vector(p, self.size) for p in image_paths]
        self.model.fit(vecs, list(labels), epochs=epochs)
        return self

    def teach_from_folder(self, root: str | Path, epochs: int = 25) -> "ImageSelector":
        """Teach from ``root`` where each subfolder name is a category.

        e.g. ``root/cat/1.jpg``, ``root/dog/2.png`` -> labels ``cat``/``dog``.
        """
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a folder.")
        paths: List[Path] = []
        labels: List[str] = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(sub.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(img)
                    labels.append(sub.name)
        if not paths:
            raise ValueError(
                f"No images found under {root}. Expected subfolders of images."
            )
        return self.teach_many(paths, labels, epochs=epochs)

    # ------------------------------------------------------------------ #
    # Using
    # ------------------------------------------------------------------ #
    def classify(self, image_path: str | Path) -> Tuple[Hashable, float]:
        """Return ``(label, confidence)`` for one image."""
        vec = image_to_vector(image_path, self.size)
        return self.model.predict_one(vec)

    def score(self, image_path: str | Path, label: Hashable) -> float:
        """Probability that ``image_path`` belongs to ``label``."""
        if label not in self.model.classes_:
            raise ValueError(f"Unknown category {label!r}. Teach it first.")
        vec = image_to_vector(image_path, self.size)
        proba = self.model.predict_proba([vec])[0]
        return float(proba[self.model.classes_.index(label)])

    def select_best(self, image_paths: Iterable, label: Hashable) -> Path:
        """Pick the single image that best matches ``label``."""
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("No images to choose from.")
        scores = [self.score(p, label) for p in paths]
        return paths[int(np.argmax(scores))]

    def rank(self, image_paths: Iterable, label: Hashable) -> List[Tuple[Path, float]]:
        """Rank images from best to worst match for ``label``."""
        paths = [Path(p) for p in image_paths]
        scored = [(p, self.score(p, label)) for p in paths]
        return sorted(scored, key=lambda t: t[1], reverse=True)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, size: Tuple[int, int] = DEFAULT_SIZE) -> "ImageSelector":
        return cls(size=size, model=TeachableModel.load(path))
