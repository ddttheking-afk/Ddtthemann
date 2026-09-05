"""Ready-made reward/done detectors for the emulator adapter.

Reinforcement learning needs *some* signal that says "good thing happened".
For on-screen games the easiest reliable signal is: does the screen right now
look like a known moment -- a "Victory" banner, a "Battle Won" screen, a level-up
flash? :class:`TemplateReward` gives you that without writing pixel math: save a
screenshot crop of the moment you want to reward, point this at it, and you get a
``reward_fn`` / ``done_fn`` you can hand straight to :class:`EmulatorEnv`.

It compares images with normalised correlation on a small grayscale version, so
it's tolerant of brightness/scale changes and needs only numpy (+ Pillow to read
the template file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..core.features import frame_to_vector

# Resolution the template and each frame are reduced to before comparing.
_MATCH_SIZE: Tuple[int, int] = (24, 24)


def _grayscale_vector(frame: np.ndarray, region: Tuple[int, int, int, int] | None) -> np.ndarray:
    """Reduce a frame (optionally cropped) to a normalised grayscale vector."""
    frame = np.asarray(frame)
    if region is not None:
        left, top, width, height = region
        frame = frame[top:top + height, left:left + width]
    rgb = frame_to_vector(frame, _MATCH_SIZE).reshape(_MATCH_SIZE[0], _MATCH_SIZE[1], 3)
    gray = rgb.mean(axis=2).reshape(-1)
    # Zero-mean, unit-norm so the comparison is contrast/brightness tolerant.
    gray = gray - gray.mean()
    norm = np.linalg.norm(gray)
    return gray / norm if norm > 1e-8 else gray


def similarity(frame: np.ndarray, template: np.ndarray,
               region: Tuple[int, int, int, int] | None = None) -> float:
    """Correlation in ``[-1, 1]`` between ``frame`` (region) and ``template``.

    1.0 means "looks exactly like the template".
    """
    a = _grayscale_vector(frame, region)
    b = _grayscale_vector(template, None)
    return float(np.dot(a, b))


class TemplateReward:
    """Turn a saved screenshot into a reward/done detector.

    Parameters
    ----------
    template:
        Path to an image file, or an RGB numpy array, of the moment to detect
        (e.g. a crop of the "Victory" banner).
    region:
        Optional ``(left, top, width, height)`` sub-rectangle of the live frame
        to look in. Narrowing it to where the banner appears makes matching far
        more reliable. Coordinates are relative to the captured frame.
    threshold:
        Similarity above which the moment counts as detected (0..1).
    reward:
        Value returned when detected (0 otherwise).

    Examples
    --------
    >>> win = TemplateReward("victory.png", threshold=0.7)
    >>> env = EmulatorEnv(region=..., keymap=..., reward_fn=win, done_fn=win.detected)
    """

    def __init__(self, template, region: Tuple[int, int, int, int] | None = None,
                 threshold: float = 0.7, reward: float = 1.0) -> None:
        self.template = _load_template(template)
        self.region = region
        self.threshold = float(threshold)
        self.reward = float(reward)

    def score(self, frame: np.ndarray) -> float:
        """Raw similarity of the current frame to the template."""
        return similarity(frame, self.template, self.region)

    def detected(self, frame: np.ndarray) -> bool:
        """Whether the template is currently on screen."""
        return self.score(frame) >= self.threshold

    def __call__(self, frame: np.ndarray) -> float:
        """Reward callback: ``reward`` when detected, else 0."""
        return self.reward if self.detected(frame) else 0.0


def _load_template(template) -> np.ndarray:
    if isinstance(template, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Pillow is required to read a template image file.") from exc
        with Image.open(template) as im:
            return np.asarray(im.convert("RGB"))
    return np.asarray(template)
