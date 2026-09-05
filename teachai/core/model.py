"""The shared "brain": a teachable online classifier.

This is the single learning model that powers everything in TeachAI. You
*teach* it incrementally with ``partial_fit`` (a few examples at a time) or
all at once with ``fit``. It is a multinomial logistic-regression classifier
trained with stochastic gradient descent.

Why this design:
  * It learns online, so you can keep teaching it new things forever.
  * It discovers new labels on the fly -- show it a brand-new category and it
    grows to accommodate it.
  * It reports calibrated confidences (via softmax), which the vision module
    uses to *select* the best picture and the game agents use to decide how
    sure they are about an action.

It only depends on numpy, so it runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Hashable, List, Sequence

import numpy as np


class TeachableModel:
    """An incrementally teachable multiclass classifier.

    Parameters
    ----------
    lr:
        Learning rate for SGD.
    l2:
        L2 regularisation strength (keeps weights small, fights overfitting).
    seed:
        RNG seed for reproducible weight initialisation.
    """

    def __init__(self, lr: float = 0.1, l2: float = 1e-4, seed: int = 0) -> None:
        self.lr = float(lr)
        self.l2 = float(l2)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.classes_: List[Hashable] = []
        self.n_features: int | None = None
        self.W: np.ndarray | None = None  # shape (n_classes, n_features)
        self.b: np.ndarray | None = None  # shape (n_classes,)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _class_index(self, label: Hashable) -> int:
        """Return the row index for ``label``, creating it if unseen."""
        if label not in self.classes_:
            self.classes_.append(label)
            new_row = self._rng.normal(0.0, 0.01, size=(1, self.n_features))
            if self.W is None:
                self.W = new_row
                self.b = np.zeros(1)
            else:
                self.W = np.vstack([self.W, new_row])
                self.b = np.append(self.b, 0.0)
        return self.classes_.index(label)

    def _init_features(self, n_features: int) -> None:
        if self.n_features is None:
            self.n_features = int(n_features)
        elif self.n_features != n_features:
            raise ValueError(
                f"Feature size changed: model expects {self.n_features}, "
                f"got {n_features}. All inputs must have the same shape."
            )

    def _logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    # Teaching
    # ------------------------------------------------------------------ #
    def partial_fit(self, X: Sequence, y: Sequence[Hashable]) -> "TeachableModel":
        """Teach the model with a small batch, updating it in place.

        ``X`` is a 2D array-like of feature vectors; ``y`` is the matching
        labels. New labels are added automatically.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = list(y)
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows.")
        self._init_features(X.shape[1])
        for label in y:  # make sure every class row exists first
            self._class_index(label)

        target_idx = np.array([self.classes_.index(label) for label in y])
        probs = self._softmax(self._logits(X))
        # One-hot targets.
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(y)), target_idx] = 1.0
        # Gradient of cross-entropy w.r.t. logits.
        grad = (probs - onehot) / len(y)
        self.W -= self.lr * (grad.T @ X + self.l2 * self.W)
        self.b -= self.lr * grad.sum(axis=0)
        return self

    def fit(self, X: Sequence, y: Sequence[Hashable], epochs: int = 30,
            batch_size: int = 32, shuffle: bool = True) -> "TeachableModel":
        """Teach the model from scratch over multiple passes of the data."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(list(y), dtype=object)
        n = X.shape[0]
        self._init_features(X.shape[1])
        for label in y:
            self._class_index(label)
        for _ in range(epochs):
            order = self._rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self.partial_fit(X[idx], y[idx])
        return self

    # ------------------------------------------------------------------ #
    # Predicting
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: Sequence) -> np.ndarray:
        """Return class probabilities, shape ``(n_samples, n_classes)``."""
        if self.W is None:
            raise RuntimeError("Model has not been taught anything yet.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return self._softmax(self._logits(X))

    def predict(self, X: Sequence) -> List[Hashable]:
        """Return the most likely label for each input row."""
        proba = self.predict_proba(X)
        return [self.classes_[i] for i in proba.argmax(axis=1)]

    def predict_one(self, x: Sequence) -> tuple[Hashable, float]:
        """Convenience: label + confidence for a single feature vector."""
        proba = self.predict_proba([x])[0]
        i = int(proba.argmax())
        return self.classes_[i], float(proba[i])

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Save weights + metadata so a trained brain can be reused."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W=self.W if self.W is not None else np.zeros((0, 0)),
            b=self.b if self.b is not None else np.zeros(0),
            meta=json.dumps({
                "lr": self.lr,
                "l2": self.l2,
                "seed": self.seed,
                "classes": self.classes_,
                "n_features": self.n_features,
            }),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeachableModel":
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        model = cls(lr=meta["lr"], l2=meta["l2"], seed=meta["seed"])
        model.classes_ = list(meta["classes"])
        model.n_features = meta["n_features"]
        model.W = data["W"] if data["W"].size else None
        model.b = data["b"] if data["b"].size else None
        return model
