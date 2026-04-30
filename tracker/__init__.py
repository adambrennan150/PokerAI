"""
Poker McPokerface — tracker package.

Persistent, append-only logging of poker hands so that bot-vs-bot
sessions can be replayed and analysed afterwards. The brief's three
research questions all reduce to pandas groupbys over the .jsonl
files written by this package.

Public surface:
    SeatConfig, HandTracker, TrackingAgent      (from .tracker)
    load_config, load_hands, load_actions,
    load_reasoning                              (from .tracker)
"""

from .tracker import (
    SeatConfig,
    HandTracker,
    TrackingAgent,
    load_config,
    load_hands,
    load_actions,
    load_reasoning,
)

__all__ = [
    "SeatConfig",
    "HandTracker",
    "TrackingAgent",
    "load_config",
    "load_hands",
    "load_actions",
    "load_reasoning",
]
