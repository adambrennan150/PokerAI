"""
Poker McPokerface — bots package.

LLM-powered (and otherwise) decision-makers that satisfy the engine's
`PlayerAgent` protocol. The engine doesn't import from here, but
everything in here imports from `engine` for `GameView`, `Action`,
etc.

Public surface:
    BaseBot, Personality, BotResponse       (from .base)
"""

from .base import BaseBot, Personality, BotResponse, MockBot

__all__ = ["BaseBot", "Personality", "BotResponse", "MockBot"]
