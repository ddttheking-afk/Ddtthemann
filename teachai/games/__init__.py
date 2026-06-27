"""Teach the AI to play games -- the demo, or real emulators."""

from .agent import ImitationAgent, QLearningAgent
from .demo_game import CatcherGame
from .env import Environment

__all__ = ["ImitationAgent", "QLearningAgent", "CatcherGame", "Environment"]
