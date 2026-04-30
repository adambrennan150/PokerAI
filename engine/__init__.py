"""
Poker McPokerface — engine package.

Pure-Python game logic for 5-card draw poker. This package has zero
dependencies on the UI layer or on any LLM bot code. Everything in here
should be deterministic (given a seed), unit-testable in isolation, and
free of side effects beyond what the game state demands.

Public surface:
    Card, Suit, Rank, Deck                  (from .deck)
    HandRank, HandResult, evaluate          (from .hand)
    Player, PlayerStatus                    (from .player)
    Game, Seat, PlayerAgent                 (from .game)
    Action, ActionType, Phase               (from .game)
    GameView, PublicPlayerInfo              (from .game)
    ActionRecord, SeatResult, HandSummary   (from .game)
"""

from .deck import Card, Suit, Rank, Deck
from .hand import HandRank, HandResult, evaluate
from .player import Player, PlayerStatus
from .game import (
    Game, Seat, PlayerAgent,
    Action, ActionType, Phase,
    GameView, PublicPlayerInfo,
    ActionRecord, SeatResult, HandSummary,
)

__all__ = [
    "Card", "Suit", "Rank", "Deck",
    "HandRank", "HandResult", "evaluate",
    "Player", "PlayerStatus",
    "Game", "Seat", "PlayerAgent",
    "Action", "ActionType", "Phase",
    "GameView", "PublicPlayerInfo",
    "ActionRecord", "SeatResult", "HandSummary",
]
