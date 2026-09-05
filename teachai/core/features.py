"""Turn raw things (images, screens) into feature vectors the model can learn.

The model in :mod:`teachai.core.model` only understands flat numeric vectors,
so everything that wants to be learned must first be turned into one. These
helpers do that for images and for raw RGB screen frames.

Pillow is only imported when you actually load an image file, so the rest of
TeachAI works without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Default working resolution. Small enough to learn fast, big enough to keep
# the gist of a picture.
DEFAULT_SIZE: Tuple[int, int] = (32, 32)


def frame_to_vector(frame: np.ndarray, size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Convert an RGB (or grayscale) numpy frame into a normalised vector.

    ``frame`` is ``(H, W, 3)`` or ``(H, W)`` with values in ``0..255``. The
    result is resized to ``size`` and scaled to ``0..1``.
    """
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    h, w = size
    # Nearest-neighbour resize using pure numpy (no SciPy/Pillow needed here).
    ys = np.linspace(0, frame.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, frame.shape[1] - 1, w).astype(int)
    small = frame[np.ix_(ys, xs)][:, :, :3]
    return (small.astype(np.float64) / 255.0).reshape(-1)


def image_to_vector(path: str | Path, size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Load an image file from disk and turn it into a feature vector."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - only when Pillow missing
        raise ImportError(
            "Pillow is required to read image files. Install it with "
            "`pip install Pillow`."
        ) from exc

    with Image.open(path) as im:
        im = im.convert("RGB").resize(size[::-1])
        arr = np.asarray(im, dtype=np.float64) / 255.0
    return arr.reshape(-1)


def feature_size(size: Tuple[int, int] = DEFAULT_SIZE) -> int:
    """Length of the vector produced for ``size`` (handy for sanity checks)."""
    return size[0] * size[1] * 3
