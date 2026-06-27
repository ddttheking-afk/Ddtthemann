"""TeachAI -- a single AI brain you teach by example.

Teach it to:
  * select and classify pictures  (:mod:`teachai.vision`)
  * play games, demo or emulated   (:mod:`teachai.games`)

All of it runs on one small, online learning model in
:mod:`teachai.core` that you keep teaching over time.
"""

from .core import TeachableModel
from .vision import ImageSelector
from .games import CatcherGame, ImitationAgent, QLearningAgent

__version__ = "0.1.0"

__all__ = [
    "TeachableModel",
    "ImageSelector",
    "CatcherGame",
    "ImitationAgent",
    "QLearningAgent",
]
