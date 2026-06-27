"""Drive a *real* emulator with the same agents that play the demo game.

This wraps any on-screen game window so it looks like a normal
:class:`Environment`. It captures a region of your screen as the state and
sends key presses as actions. That means an agent trained on the demo game (or
freshly taught by imitation) can be pointed at, for example:

  * **Pokemon** running in mGBA / VisualBoyAdvance / DeSmuME.
  * **Epic Seven** running in an Android emulator (BlueStacks, LDPlayer) or
    mirrored from a phone via scrcpy.

Because TeachAI can't know when *you* won a battle, you provide a ``reward_fn``
that looks at the captured frame and decides the reward (e.g. detect the
"Victory" banner). Until then, imitation learning -- where you simply play and
the agent copies you -- needs no reward function at all.

Heavy, platform-specific dependencies (screen capture + key sending) are
imported lazily, so importing TeachAI never requires them.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..core.features import DEFAULT_SIZE, frame_to_vector
from .env import Environment


def _require(module: str, package: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"The emulator adapter needs '{package}'. Install it with "
            f"`pip install {package}`. (Screen capture / key sending are "
            f"platform-specific, which is why they're optional.)"
        ) from exc


class EmulatorEnv:
    """Turn a region of the screen + a keymap into a learnable environment.

    Parameters
    ----------
    region:
        ``(left, top, width, height)`` screen rectangle of the game window.
    keymap:
        Ordered list mapping each action index to a keyboard key, e.g.
        ``["left", "right", "x", "z"]``. ``None`` entries mean "do nothing".
    reward_fn:
        Optional ``frame -> float`` callback for reinforcement learning. If
        omitted, every step yields reward 0 (fine for imitation learning).
    done_fn:
        Optional ``frame -> bool`` callback to detect episode end.
    step_delay:
        Seconds to wait after each action so the game can advance a frame.
    """

    def __init__(self, region: Tuple[int, int, int, int], keymap: Sequence,
                 reward_fn: Callable[[np.ndarray], float] | None = None,
                 done_fn: Callable[[np.ndarray], bool] | None = None,
                 size: Tuple[int, int] = DEFAULT_SIZE,
                 step_delay: float = 0.1) -> None:
        self.region = region
        self.keymap = list(keymap)
        self.n_actions = len(self.keymap)
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.size = size
        self.step_delay = step_delay
        self._sct = None       # screen-capture handle (lazy)
        self._keyboard = None  # keyboard controller (lazy)
        self._last_frame: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Lazy backends
    # ------------------------------------------------------------------ #
    def _capture(self):
        if self._sct is None:
            mss = _require("mss", "mss")
            self._sct = mss.mss()
        left, top, width, height = self.region
        shot = self._sct.grab({"left": left, "top": top,
                               "width": width, "height": height})
        # mss returns BGRA; keep RGB.
        return np.asarray(shot)[:, :, :3][:, :, ::-1]

    def _press(self, key) -> None:
        if key is None:
            return
        if self._keyboard is None:
            pynput = _require("pynput", "pynput")
            self._keyboard = pynput.keyboard.Controller()
        from pynput.keyboard import Key
        k = getattr(Key, key, key) if isinstance(key, str) else key
        self._keyboard.press(k)
        time.sleep(0.02)
        self._keyboard.release(k)

    # ------------------------------------------------------------------ #
    # Environment interface
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        frame = self._capture()
        self._last_frame = frame
        return frame_to_vector(frame, self.size)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._press(self.keymap[action])
        time.sleep(self.step_delay)
        frame = self._capture()
        self._last_frame = frame
        reward = float(self.reward_fn(frame)) if self.reward_fn else 0.0
        done = bool(self.done_fn(frame)) if self.done_fn else False
        return frame_to_vector(frame, self.size), reward, done, {"frame": frame}

    def render(self) -> str:
        if self._last_frame is None:
            return "<no frame captured yet>"
        h, w = self._last_frame.shape[:2]
        return f"<emulator frame {w}x{h} from region {self.region}>"


def record_human_play(env: EmulatorEnv, action_for_key: Dict[str, int],
                      max_steps: int = 10_000) -> List[Tuple[np.ndarray, int]]:
    """Record your own play through an emulator for imitation learning.

    Watches the keyboard; whenever you press one of the keys in
    ``action_for_key`` it stores the current screen state paired with that
    action. Press Esc to stop. Returns a list of ``(state, action)`` you can
    feed to :meth:`ImitationAgent.record`.
    """
    pynput = _require("pynput", "pynput")
    from pynput.keyboard import Key, Listener

    samples: List[Tuple[np.ndarray, int]] = []
    stop = {"flag": False}

    def on_press(key):
        if key == Key.esc:
            stop["flag"] = True
            return False
        name = getattr(key, "char", None) or getattr(key, "name", None)
        if name in action_for_key:
            state = frame_to_vector(env._capture(), env.size)
            samples.append((state, action_for_key[name]))

    with Listener(on_press=on_press) as listener:
        steps = 0
        while not stop["flag"] and steps < max_steps:
            time.sleep(0.01)
            steps += 1
        listener.join()
    return samples
