"""
Poker McPokerface — UI package.

Notebook + terminal UI for human-vs-bot play. The engine is unaware
of this package; nothing here imports from `bots/` either.

Public surface:
    HumanAgent, parse_action_input, parse_discard_input  (from .human)
    render_table_html, render_table_text                 (from .rendering)
"""

from .human import HumanAgent, parse_action_input, parse_discard_input
from .rendering import render_table_html, render_table_text

__all__ = [
    "HumanAgent",
    "parse_action_input",
    "parse_discard_input",
    "render_table_html",
    "render_table_text",
]
