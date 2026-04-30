"""
hand.py — 5-card poker hand evaluation.

Given any 5 `Card`s, this module classifies them into one of the nine
poker categories and produces a comparison key so two hands can be
compared with `<`, `>`, `==`. The same key resolves tiebreakers within
a category (e.g. pair of Kings with Ace kicker beats pair of Kings with
Queen kicker).

Design notes
------------
* The comparison strategy is: encode every hand as a tuple
  `(category_value, *tiebreaker_ranks_descending)`. Python's built-in
  tuple comparison then does *all* the tie-breaking for free —
  lexicographic comparison naturally walks down kickers in the right
  order. No special-case branching per category.

* Each category builds its tiebreaker tuple in a deliberate order:
    High card     -> all 5 ranks, desc
    One pair      -> pair rank, then 3 kickers desc
    Two pair      -> higher pair, lower pair, kicker
    Three of kind -> trip rank, then 2 kickers desc
    Straight      -> top-of-straight rank (wheel A-2-3-4-5 = 5)
    Flush         -> all 5 ranks, desc
    Full house    -> trip rank, pair rank
    Four of kind  -> quad rank, kicker
    Straight flush-> top-of-straight rank
  This is exactly what real poker tiebreakers do.

* The "wheel" (A-2-3-4-5) is the one special case in the rules. It's
  the lowest straight, with the Ace counted as 1, and we encode that
  by setting top-of-straight to 5 instead of Ace's usual 14.

* Pure functions only — no state, no I/O. Easy to unit-test by passing
  in literal `Card`s and checking the returned `HandResult`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Sequence, Tuple

from .deck import Card, Rank, Suit


# ----------------------------------------------------------------------
# Category enum
# ----------------------------------------------------------------------
class HandRank(IntEnum):
    """Poker hand categories, ordered from worst to best.

    `IntEnum` so the value participates directly in tuple comparison —
    a `STRAIGHT` (4) automatically beats `THREE_OF_A_KIND` (3) without
    any custom logic.
    """

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    # Royal flush is just an Ace-high straight flush; we don't give it
    # its own category because the comparison key already makes it the
    # highest possible STRAIGHT_FLUSH.

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'Two Pair'."""
        return self.name.replace("_", " ").title()


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class HandResult:
    """The classified result of evaluating a 5-card hand.

    `key` is the field used for comparison (declared first so dataclass
    `order=True` uses it). The `cards` and `category` fields tag along
    for display and analysis but don't influence ordering.
    """

    # Lexicographic comparison key: (category_value, *tiebreakers).
    # Two HandResults with equal keys represent a true split-pot tie.
    key: Tuple[int, ...]

    # Non-comparing metadata. `compare=False` keeps these out of the
    # ordering — they're for display, logging, and the tracker.
    category: HandRank = field(compare=False)
    cards: Tuple[Card, ...] = field(compare=False)

    def describe(self) -> str:
        """Compact human-readable description, suitable for logging or
        showing in the UI / showdown text."""
        cards_str = " ".join(str(c) for c in self.cards)
        return f"{self.category.label} [{cards_str}]"

    def __str__(self) -> str:
        return self.describe()


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def evaluate(cards: Sequence[Card]) -> HandResult:
    """Classify a 5-card poker hand.

    Raises ValueError if the hand isn't exactly 5 distinct cards — this
    is a hard contract: 5-card draw never evaluates anything else, and
    catching the off-by-one early is much friendlier than producing a
    wrong-but-plausible result.
    """
    if len(cards) != 5:
        raise ValueError(f"A poker hand must contain exactly 5 cards (got {len(cards)}).")
    if len(set(cards)) != 5:
        raise ValueError(f"Duplicate cards in hand: {cards}")

    # Sort cards descending by rank value. Most categories want the
    # highest-rank info first when building the tiebreaker tuple.
    sorted_cards = tuple(sorted(cards, key=lambda c: c.rank.value, reverse=True))

    # ------------------------------------------------------------------
    # Pre-compute everything once: cheaper, and the checks below stay
    # readable.
    # ------------------------------------------------------------------
    rank_values = [c.rank.value for c in sorted_cards]      # e.g. [14, 13, 9, 9, 4]
    rank_counts = Counter(rank_values)                      # e.g. {9: 2, 14: 1, 13: 1, 4: 1}
    suits = [c.suit for c in sorted_cards]

    is_flush = len(set(suits)) == 1
    straight_top = _straight_top(rank_values)               # None if not a straight

    # `count_groups` is a list of (count, rank) sorted by count desc,
    # then rank desc. This is the canonical shape for matching pairs/
    # trips/quads tiebreakers: the most-numerous rank dominates, ties
    # broken by higher rank.
    count_groups: List[Tuple[int, int]] = sorted(
        ((cnt, rank) for rank, cnt in rank_counts.items()),
        key=lambda pair: (pair[0], pair[1]),
        reverse=True,
    )
    counts_only = tuple(cnt for cnt, _ in count_groups)     # e.g. (2, 1, 1, 1)
    ranks_in_group_order = tuple(rank for _, rank in count_groups)

    # ------------------------------------------------------------------
    # Classify, from strongest to weakest. First match wins.
    # ------------------------------------------------------------------
    if is_flush and straight_top is not None:
        return _make_result(HandRank.STRAIGHT_FLUSH, (straight_top,), sorted_cards)

    if counts_only == (4, 1):
        # Four of a kind: (quad rank, kicker rank)
        return _make_result(HandRank.FOUR_OF_A_KIND, ranks_in_group_order, sorted_cards)

    if counts_only == (3, 2):
        # Full house: (trip rank, pair rank)
        return _make_result(HandRank.FULL_HOUSE, ranks_in_group_order, sorted_cards)

    if is_flush:
        # Plain flush: all 5 ranks descending serve as the tiebreaker.
        return _make_result(HandRank.FLUSH, tuple(rank_values), sorted_cards)

    if straight_top is not None:
        return _make_result(HandRank.STRAIGHT, (straight_top,), sorted_cards)

    if counts_only == (3, 1, 1):
        # Three of a kind: (trip rank, kicker1, kicker2)
        return _make_result(HandRank.THREE_OF_A_KIND, ranks_in_group_order, sorted_cards)

    if counts_only == (2, 2, 1):
        # Two pair: count_groups already sorts higher pair first because
        # we sorted by (count desc, rank desc), so this is correct.
        return _make_result(HandRank.TWO_PAIR, ranks_in_group_order, sorted_cards)

    if counts_only == (2, 1, 1, 1):
        # One pair: (pair rank, then 3 kickers descending)
        return _make_result(HandRank.ONE_PAIR, ranks_in_group_order, sorted_cards)

    # No pairs, no straight, no flush — high card. All 5 ranks descending.
    return _make_result(HandRank.HIGH_CARD, tuple(rank_values), sorted_cards)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_result(
    category: HandRank,
    tiebreakers: Tuple[int, ...],
    sorted_cards: Tuple[Card, ...],
) -> HandResult:
    """Bundle a category + tiebreakers into a `HandResult` with the
    correct comparison key. Centralised so the key shape is always
    `(category_value, *tiebreakers)`."""
    key = (int(category),) + tuple(tiebreakers)
    return HandResult(key=key, category=category, cards=sorted_cards)


def _straight_top(rank_values: Sequence[int]) -> int | None:
    """If `rank_values` (length 5) forms a straight, return the rank of
    the top card of the straight; otherwise return None.

    The wheel (A-2-3-4-5) is the one special case: the Ace plays low,
    so we report 5 as the top — that ranks the wheel below 6-high
    straights, which is correct.
    """
    distinct = sorted(set(rank_values))
    if len(distinct) != 5:
        # A straight needs 5 distinct ranks; any pair/trip/quad rules
        # this out immediately.
        return None

    # Standard straight: 5 consecutive ranks.
    if distinct[-1] - distinct[0] == 4:
        return distinct[-1]

    # Wheel: A-2-3-4-5. With Ace = 14 in our enum, this set is
    # {2, 3, 4, 5, 14}. Treat top-of-straight as 5.
    if distinct == [2, 3, 4, 5, 14]:
        return 5

    return None


# ----------------------------------------------------------------------
# Smoke test — run `python -m engine.hand` from the project root.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    def card(short: str) -> Card:
        """Tiny parser for test convenience: 'AS' = A♠, 'TH' = T♥, '2C' = 2♣."""
        rank_char, suit_char = short[0].upper(), short[1].upper()
        rank_lookup = {r.short: r for r in Rank}
        suit_lookup = {"C": Suit.CLUBS, "D": Suit.DIAMONDS,
                       "H": Suit.HEARTS, "S": Suit.SPADES}
        return Card(rank_lookup[rank_char], suit_lookup[suit_char])

    examples = [
        ("Royal flush",     ["AS", "KS", "QS", "JS", "TS"]),
        ("Straight flush",  ["9H", "8H", "7H", "6H", "5H"]),
        ("Four of a kind",  ["QC", "QD", "QH", "QS", "3D"]),
        ("Full house",      ["JC", "JD", "JS", "4H", "4D"]),
        ("Flush",           ["AC", "9C", "7C", "5C", "2C"]),
        ("Straight",        ["9C", "8D", "7H", "6S", "5C"]),
        ("Wheel",           ["AC", "2D", "3H", "4S", "5C"]),
        ("Three of a kind", ["7C", "7D", "7H", "KS", "2C"]),
        ("Two pair",        ["KC", "KD", "5H", "5S", "9C"]),
        ("One pair",        ["AC", "AD", "9H", "5S", "2C"]),
        ("High card",       ["AC", "JD", "8H", "5S", "2C"]),
    ]

    results = []
    for label, shorts in examples:
        result = evaluate([card(s) for s in shorts])
        results.append((label, result))
        print(f"{label:18s} -> {result.describe()}  key={result.key}")

    # Sanity: stronger categories should compare greater than weaker.
    print()
    print("Royal > Straight Flush?", results[0][1] > results[1][1])
    print("Wheel < 9-high straight?", results[6][1] < results[5][1])
    print("Aces > Kings (one pair)?", results[9][1] > evaluate([
        card("KC"), card("KD"), card("9H"), card("5S"), card("2C")
    ]))
