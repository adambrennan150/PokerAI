"""
Poker McPokerface — config package.

Project-wide configuration data: model rosters, environment defaults.
Plain-Python data (lists, dataclasses) — no behaviour, no I/O.

Public surface:
    LOCAL_ROSTER, COLAB_ROSTER, ModelSpec     (from .models)
"""

from .models import LOCAL_ROSTER, COLAB_ROSTER, ModelSpec

__all__ = ["LOCAL_ROSTER", "COLAB_ROSTER", "ModelSpec"]
