"""
Poker McPokerface — bots package.

LLM-powered (and otherwise) decision-makers that satisfy the engine's
`PlayerAgent` protocol. The engine doesn't import from here, but
everything in here imports from `engine` for `GameView`, `Action`,
etc.

Public surface:
    BaseBot, Personality, BotResponse, MockBot   (from .base)
    OllamaBot                                    (from .ollama_bot)
"""

from .base import BaseBot, Personality, BotResponse, MockBot
# OllamaBot is imported lazily — if the `ollama` package isn't installed
# (e.g. fresh Colab session before pip install runs), the import still
# succeeds; only constructing one would raise.
from .ollama_bot import OllamaBot

__all__ = ["BaseBot", "Personality", "BotResponse", "MockBot", "OllamaBot"]
