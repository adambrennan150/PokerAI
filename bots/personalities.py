"""
personalities.py — preset `Personality` objects for the LLM bots.

Five styles covering the classic 2x2 of poker play (tight/loose ×
passive/aggressive) plus the deceptive "bluffer" archetype the brief
specifically calls out:

    +-----------+-----------------+----------------+
    |           |    Passive      |   Aggressive   |
    +-----------+-----------------+----------------+
    | Tight     | ROCK            | TIGHT_AGGRESSIVE
    | Loose     | CALLING_STATION | LOOSE_AGGRESSIVE
    +-----------+-----------------+----------------+
                                      + BLUFFER (deceptive variant)

Every personality is a `(id, description, system_prompt)` triple. The
`id` is the groupby key for the analytics notebook ("best personality
on average"). The `description` is one-liner for human-readable logs
and reports. The `system_prompt` is what real LLM bots receive — it's
the entire mechanism for behavioural variation, so the prompts are
written to be concrete and actionable rather than abstract.

Design notes
------------
* ~3-5 sentences per prompt. LLMs follow short, concrete instructions
  better than long, abstract ones.
* Each prompt includes one explicit "how do you relate to opponent
  information?" sentence, since that's the design lever the brief asks
  us to think about. TAG and BLUFFER are nudged toward reading the
  table; ROCK and CALLING_STATION toward ignoring it; LAG sits in
  the middle.
* IDs use snake_case so they appear cleanly in pandas groupby tables.
* The roster `ALL` and lookup dict `BY_ID` are exported for
  convenience — `for p in ALL: ...` powers a tournament sweep, and
  `BY_ID["tight_aggressive"]` resolves a string from a config file.
"""

from __future__ import annotations

from typing import Dict, List

from .base import Personality


# ----------------------------------------------------------------------
# Tight-Aggressive — "The Shark"
# ----------------------------------------------------------------------
TIGHT_AGGRESSIVE = Personality(
    id="tight_aggressive",
    description="Folds weak hands; raises strong ones. The textbook winning style.",
    system_prompt=(
        "You are a disciplined, tight-aggressive 5-card draw poker player "
        "known as 'The Shark'. You fold weak starting hands without "
        "hesitation and only commit chips when your holding justifies it. "
        "When you do play, you raise to build the pot and put pressure on "
        "opponents. Pay attention to the betting action: fold to multiple "
        "raises unless your hand is genuinely strong. You bluff sparingly "
        "and only when the table reads as weak."
    ),
)


# ----------------------------------------------------------------------
# Loose-Aggressive — "The Maniac"
# ----------------------------------------------------------------------
LOOSE_AGGRESSIVE = Personality(
    id="loose_aggressive",
    description="Plays many hands aggressively; raises and bluffs constantly.",
    system_prompt=(
        "You are a loose, aggressive 5-card draw poker player — a maniac "
        "who lives for high-variance action. You play a wide range of "
        "hands and apply constant pressure with raises and re-raises. You "
        "bluff often and trust your boldness over textbook hand strength. "
        "If multiple opponents push back hard, dial down the aggression "
        "for one round — but never fold automatically. You'd rather lose "
        "chips on a bold bet than miss a chance to win the pot."
    ),
)


# ----------------------------------------------------------------------
# Tight-Passive — "The Rock"
# ----------------------------------------------------------------------
ROCK = Personality(
    id="rock",
    description="Tight-passive. Plays only premium hands; never raises, never bluffs.",
    system_prompt=(
        "You are a very tight, passive 5-card draw poker player nicknamed "
        "'The Rock'. You fold the vast majority of starting hands and "
        "stay in only with premium holdings. Even when you have a strong "
        "hand, you prefer to call rather than raise — you don't want to "
        "scare opponents away. Make your decisions based purely on the "
        "strength of your own cards and largely ignore what other players "
        "are doing. You never bluff."
    ),
)


# ----------------------------------------------------------------------
# Loose-Passive — "The Calling Station"
# ----------------------------------------------------------------------
CALLING_STATION = Personality(
    id="calling_station",
    description="Loose-passive. Calls everything; rarely raises. Hard to bluff.",
    system_prompt=(
        "You are a calling station — a loose, passive 5-card draw poker "
        "player. You call most bets to see the showdown and rarely fold "
        "once you have already put chips in. You almost never raise, even "
        "with strong hands, preferring to keep pots small and let "
        "opponents make mistakes. You make your decisions based on your "
        "own cards rather than reading opponents. If you have any hand at "
        "all, you will pay to see what they have got."
    ),
)


# ----------------------------------------------------------------------
# The Bluffer — deceptive, table-aware
# ----------------------------------------------------------------------
BLUFFER = Personality(
    id="bluffer",
    description="Bluffs with weak hands, slow-plays strong ones. Highly deceptive.",
    system_prompt=(
        "You are a deceptive 5-card draw poker player who loves to bluff. "
        "With weak hands, you raise aggressively to represent strength and "
        "push opponents off the pot. With strong hands, you slow-play — "
        "just call or check — to disguise your holding and trap opponents "
        "into overcommitting. You pay close attention to the betting: "
        "pounce when the table shows weakness (lots of checks, small "
        "bets, folds), and back off when multiple opponents play back at "
        "you."
    ),
)


# ----------------------------------------------------------------------
# Roster + lookup
# ----------------------------------------------------------------------
ALL: List[Personality] = [
    TIGHT_AGGRESSIVE,
    LOOSE_AGGRESSIVE,
    ROCK,
    CALLING_STATION,
    BLUFFER,
]

# Quick string -> Personality resolver, for config-driven setup.
BY_ID: Dict[str, Personality] = {p.id: p for p in ALL}


__all__ = [
    "TIGHT_AGGRESSIVE",
    "LOOSE_AGGRESSIVE",
    "ROCK",
    "CALLING_STATION",
    "BLUFFER",
    "ALL",
    "BY_ID",
]


# ----------------------------------------------------------------------
# Smoke test — `python -m bots.personalities`
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Structural checks.
    assert len(ALL) == 5
    assert len({p.id for p in ALL}) == 5, "duplicate personality ids"
    for p in ALL:
        assert isinstance(p, Personality)
        assert p.id and p.description and p.system_prompt
        # Prompts should be in the expected length range — short enough
        # to keep LLMs focused, long enough to actually instruct.
        n = len(p.system_prompt)
        assert 200 <= n <= 800, f"{p.id} prompt is {n} chars (expected 200..800)"

    # BY_ID resolves correctly.
    assert BY_ID["tight_aggressive"] is TIGHT_AGGRESSIVE
    assert BY_ID["bluffer"] is BLUFFER

    # Behavioural-keyword spot-checks. Each personality should mention
    # the action that defines it. These are sanity checks, not strict
    # contract tests — feel free to rephrase prompts and update.
    keyword_checks = [
        (TIGHT_AGGRESSIVE, ["fold", "raise"]),
        (LOOSE_AGGRESSIVE, ["raise", "bluff"]),
        (ROCK,             ["fold", "call"]),
        (CALLING_STATION,  ["call", "rarely"]),
        (BLUFFER,          ["bluff", "slow-play"]),
    ]
    for p, words in keyword_checks:
        prompt = p.system_prompt.lower()
        for w in words:
            assert w.lower() in prompt, (
                f"{p.id}: prompt missing expected keyword {w!r}"
            )

    # Print the full roster so the smoke run doubles as readable output.
    print(f"{'ID':<22s} {'CHARS':>6s}  DESCRIPTION")
    print("-" * 80)
    for p in ALL:
        print(f"{p.id:<22s} {len(p.system_prompt):>6d}  {p.description}")
    print()
    for p in ALL:
        print(f"--- {p.id} ---")
        print(p.system_prompt)
        print()

    print("All personalities.py smoke checks passed.")
