"""Two ways to teach the AI to play a game.

ImitationAgent -- "watch me, then copy me".
    You (or an expert) play; the agent records (state, action) pairs and trains
    the shared TeachableModel to reproduce your choices. This is the fastest way
    to get a competent agent and works for any game where you can play yourself,
    including emulated Pokemon or Epic Seven.

QLearningAgent -- "learn by trial, reward, and error".
    The agent plays on its own, keeps a table of how good each action is in each
    situation, and nudges those values toward whatever earns reward. No
    demonstrations needed -- just a reward signal from the game.

Both expose the same ``act(state)`` method, so the run loop is identical.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np

from ..core.model import TeachableModel
from .env import Environment


# --------------------------------------------------------------------------- #
# Imitation learning
# --------------------------------------------------------------------------- #
class ImitationAgent:
    """Learns to play by copying demonstrated (state -> action) examples."""

    def __init__(self, n_actions: int, model: TeachableModel | None = None) -> None:
        self.n_actions = n_actions
        self.model = model or TeachableModel(lr=0.3, l2=1e-4)
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []

    def record(self, state: np.ndarray, action: int) -> None:
        """Remember one demonstrated decision."""
        self._states.append(np.asarray(state, dtype=np.float64))
        self._actions.append(int(action))

    def learn(self, epochs: int = 40) -> "ImitationAgent":
        """Train on everything recorded so far."""
        if not self._states:
            raise RuntimeError("No demonstrations recorded yet.")
        self.model.fit(self._states, self._actions, epochs=epochs)
        return self

    def act(self, state: np.ndarray) -> int:
        """Choose an action for ``state`` (falls back to STAY before training)."""
        if self.model.W is None:
            return min(1, self.n_actions - 1)
        label, _ = self.model.predict_one(np.asarray(state, dtype=np.float64))
        return int(label)

    def demonstrate(self, env: Environment, expert, episodes: int = 200) -> "ImitationAgent":
        """Auto-record demonstrations from an ``expert(env) -> action`` fn."""
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = expert(env)
                self.record(state, action)
                state, _, done, _ = env.step(action)
        return self

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, n_actions: int) -> "ImitationAgent":
        return cls(n_actions=n_actions, model=TeachableModel.load(path))


# --------------------------------------------------------------------------- #
# Reinforcement learning
# --------------------------------------------------------------------------- #
class QLearningAgent:
    """Tabular Q-learning: learns purely from rewards, no demonstrations.

    Continuous state vectors are discretised into buckets so they can index a
    Q-table. ``epsilon`` controls exploration (random moves) and decays as the
    agent grows more confident.
    """

    def __init__(self, n_actions: int, n_bins: int = 6, lr: float = 0.2,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.02, epsilon_decay: float = 0.995,
                 seed: int = 0) -> None:
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

    def _key(self, state: np.ndarray) -> Tuple[int, ...]:
        state = np.clip(np.asarray(state, dtype=np.float64), -1.0, 1.0)
        # Map roughly [-1, 1] -> [0, n_bins-1].
        bins = ((state + 1.0) / 2.0 * (self.n_bins - 1)).round().astype(int)
        return tuple(int(b) for b in bins)

    def act(self, state: np.ndarray, explore: bool = False) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q[self._key(state)]))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        key, nkey = self._key(state), self._key(next_state)
        best_next = 0.0 if done else float(np.max(self.q[nkey]))
        target = reward + self.gamma * best_next
        self.q[key][action] += self.lr * (target - self.q[key][action])

    def train(self, env: Environment, episodes: int = 2000) -> List[float]:
        """Play ``episodes`` games, learning from each. Returns reward history."""
        history: List[float] = []
        for _ in range(episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                action = self.act(state, explore=True)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total += reward
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            history.append(total)
        return history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "n_actions": self.n_actions, "n_bins": self.n_bins,
                "lr": self.lr, "gamma": self.gamma,
            },
            "q": {",".join(map(str, k)): v.tolist() for k, v in self.q.items()},
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text())
        agent = cls(**payload["config"])
        for k, v in payload["q"].items():
            agent.q[tuple(int(x) for x in k.split(","))] = np.array(v)
        return agent
