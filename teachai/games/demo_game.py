"""A tiny built-in game so you can watch the AI learn *today*.

CatcherGame: items fall from the top of a grid; you move a paddle along the
bottom to catch them. Catch = +1, miss = -1. It has a clear reward signal and
is simple enough that the agent visibly gets better within seconds -- which
makes it the perfect stand-in while you set up a real emulator.

The state the agent sees is a small normalised vector, exactly like the one a
real screen frame would be reduced to, so an agent trained here uses the same
code path as one driving Pokemon or Epic Seven through the emulator adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Actions
LEFT, STAY, RIGHT = 0, 1, 2


class CatcherGame:
    """Catch falling items with a paddle. Implements :class:`Environment`."""

    n_actions = 3

    def __init__(self, width: int = 7, height: int = 7, seed: int = 0) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.paddle = width // 2
        self.item_x = 0
        self.item_y = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.paddle = self.width // 2
        self._spawn_item()
        return self._state()

    def _spawn_item(self) -> None:
        self.item_x = int(self.rng.integers(0, self.width))
        self.item_y = 0

    def _state(self) -> np.ndarray:
        """Normalised features: paddle, item position, and their gap."""
        return np.array([
            self.paddle / (self.width - 1),
            self.item_x / (self.width - 1),
            self.item_y / (self.height - 1),
            (self.item_x - self.paddle) / (self.width - 1),
        ], dtype=np.float64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action == LEFT:
            self.paddle = max(0, self.paddle - 1)
        elif action == RIGHT:
            self.paddle = min(self.width - 1, self.paddle + 1)
        # STAY does nothing.

        self.item_y += 1
        reward = 0.0
        done = False
        if self.item_y >= self.height - 1:
            caught = self.item_x == self.paddle
            reward = 1.0 if caught else -1.0
            done = True
            info = {"caught": caught}
            return self._state(), reward, done, info
        return self._state(), reward, False, {}

    def render(self) -> str:
        rows = []
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                if y == self.item_y and x == self.item_x:
                    cells.append("*")
                elif y == self.height - 1 and x == self.paddle:
                    cells.append("=")
                else:
                    cells.append(".")
            rows.append("".join(cells))
        return "\n".join(rows)

    # -- helper for imitation: what a perfect player would do right now -- #
    def expert_action(self) -> int:
        """A simple built-in 'expert' used to generate demonstrations."""
        if self.item_x < self.paddle:
            return LEFT
        if self.item_x > self.paddle:
            return RIGHT
        return STAY
