"""The contract every game (real or built-in) follows.

Both the built-in demo game and the real-emulator adapter implement this same
small interface, so the learning agents don't care which one they're driving.
It mirrors the classic Gym-style loop:

    state = env.reset()
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import numpy as np


class Environment(Protocol):
    """Minimal environment protocol."""

    #: Number of discrete actions available (e.g. 3 = left/stay/right).
    n_actions: int

    def reset(self) -> np.ndarray:
        """Start a new episode and return the first state vector."""
        ...

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply ``action``; return ``(state, reward, done, info)``."""
        ...

    def render(self) -> str:
        """Return a human-readable view of the current state (optional)."""
        ...
