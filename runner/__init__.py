"""
Poker McPokerface — runner package.

The orchestration layer that turns a list of bots + a config into a
full session: it creates a Game, wires up a HandTracker, wraps each
bot in a TrackingAgent, plays N hands, and persists everything.

Public surface:
    RunnerConfig, TournamentResult, TournamentRunner   (from .runner)
"""

from .runner import RunnerConfig, TournamentResult, TournamentRunner

__all__ = ["RunnerConfig", "TournamentResult", "TournamentRunner"]
