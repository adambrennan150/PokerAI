"""
deck.py — cards and the deck for 5-card draw poker.

Design notes
------------
* `Card` is a frozen dataclass: cards are immutable value objects, so two
  cards with the same rank and suit are equal and hashable. This makes
  set membership, deduping, and hand comparison straightforward.
* `Suit` and `Rank` are `Enum`s. Rank values are the comparable integer
  values used by the hand evaluator (2..14, with Ace = 14). Suits carry
  no ordering — poker doesn't rank suits in 5-card draw.
* `Deck` owns a single internal list of cards and a `random.Random`
  instance. Passing a `seed` makes shuffles reproducible, which matters
  for testing the hand evaluator and for replaying bot-vs-bot runs.
* The deck only deals from the "top" (end of the list — `pop()` is O(1)).
  Burned cards on draw can simply be `pop()`ed and discarded.
* No printing, no UI, no I/O. The engine layer stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional
import random


class Suit(Enum):
    """The four suits. Symbols are for human-readable display only;
    suits are not ordered in 5-card draw."""

    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"

    def __str__(self) -> str:
        return self.value


class Rank(Enum):
    """Card ranks. The integer value IS the rank used for comparison —
    Ace is high (14). The hand evaluator can treat Ace as low (1) for
    A-2-3-4-5 wheel straights as a special case."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def short(self) -> str:
        """Single-character label used in compact card strings ('A', 'K',
        'Q', 'J', 'T', '9'..'2')."""
        mapping = {
            Rank.TEN: "T",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        return mapping.get(self, str(self.value))

    def __str__(self) -> str:
        return self.short


@dataclass(frozen=True, order=True)
class Card:
    """A single playing card.

    Frozen + ordered: cards are immutable, hashable, and sort by rank then
    suit. Sorting by rank first is what the hand evaluator wants; the
    suit tiebreak is only there to give a deterministic order in
    sorted hands (it has no poker meaning).
    """

    # Order matters here: dataclass uses field declaration order for
    # comparison, so rank dominates suit.
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        # e.g. "A♠", "T♥", "2♣"
        return f"{self.rank.short}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card({self.rank.name}, {self.suit.name})"


class Deck:
    """A standard 52-card deck.

    Typical use:
        deck = Deck(seed=42)
        deck.shuffle()
        hand = deck.deal(5)
        ...
        deck.reset()       # rebuild full 52, NOT shuffled
        deck.shuffle()     # shuffle again
    """

    SIZE = 52

    def __init__(self, seed: Optional[int] = None) -> None:
        # A dedicated RNG instance — never touch the global `random`
        # module. This keeps reproducibility local: two Decks with the
        # same seed produce the same shuffle order regardless of what
        # else the rest of the program is doing with randomness.
        self._rng = random.Random(seed)
        self._cards: List[Card] = self._build_full_deck()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_full_deck() -> List[Card]:
        """Return a fresh, ordered list of all 52 cards."""
        return [Card(rank, suit) for suit in Suit for rank in Rank]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def shuffle(self) -> None:
        """Shuffle the remaining cards in place using the deck's RNG."""
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> List[Card]:
        """Deal `n` cards from the top of the deck.

        Raises ValueError if `n` is negative or if there aren't enough
        cards left — failing loudly here is much easier to debug than
        silently dealing a short hand.
        """
        if n < 0:
            raise ValueError(f"Cannot deal a negative number of cards (got {n}).")
        if n > len(self._cards):
            raise ValueError(
                f"Cannot deal {n} cards: only {len(self._cards)} left in the deck."
            )
        # `pop()` from the end is O(1). We treat the end of the list as
        # the "top" of the deck.
        dealt = [self._cards.pop() for _ in range(n)]
        return dealt

    def deal_one(self) -> Card:
        """Convenience: deal a single card. Equivalent to `deal(1)[0]`."""
        return self.deal(1)[0]

    def reset(self, shuffle: bool = False) -> None:
        """Rebuild the deck back to a full 52 cards.

        By default returns the deck to its ordered state — call
        `shuffle()` afterwards (or pass `shuffle=True`) to re-randomise.
        Useful between hands.
        """
        self._cards = self._build_full_deck()
        if shuffle:
            self.shuffle()

    # ------------------------------------------------------------------
    # Introspection — read-only views of deck state
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        # Iterate over a copy so callers can't mutate the deck via the
        # iterator (e.g. by calling .remove() on the underlying list).
        return iter(list(self._cards))

    def __contains__(self, card: object) -> bool:
        return card in self._cards

    def remaining(self) -> int:
        """Number of cards still in the deck."""
        return len(self._cards)

    def peek(self, n: int = 1) -> List[Card]:
        """Look at the top `n` cards without removing them. Intended for
        tests and debugging — game logic should always go through
        `deal()`."""
        if n < 0 or n > len(self._cards):
            raise ValueError(f"Cannot peek {n} cards from a deck of {len(self._cards)}.")
        # Top of deck = end of list, so peek the last `n` in dealing
        # order (last element is dealt first).
        return list(reversed(self._cards[-n:]))

    def __repr__(self) -> str:
        return f"Deck(remaining={len(self._cards)})"


if __name__ == "__main__":
    # Tiny smoke test — run `python -m engine.deck` from the project
    # root to sanity-check shuffling and dealing.
    d = Deck(seed=0)
    print(d)                          # Deck(remaining=52)
    d.shuffle()
    hand = d.deal(5)
    print("Dealt:", " ".join(str(c) for c in hand))
    print("Remaining:", len(d))
    d.reset(shuffle=True)
    print("After reset:", d)
