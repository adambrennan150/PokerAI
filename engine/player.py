"""
player.py — per-player state for 5-card draw poker.

A `Player` owns three things during a hand:
  1. A bankroll of chips.
  2. A hand of up to 5 cards.
  3. A status (active / folded / all-in) plus per-round betting bookkeeping.

Design notes
------------
* The engine `Player` is **agnostic** about who or what controls it.
  Whether the player is a human typing in the UI or an LLM bot deciding
  via `decide_action(game_state)`, the engine sees the same object.
  Decision-making lives in the `bots/` and `ui/` layers and never
  reaches into Player internals — it only calls methods like `fold()`,
  `post()`, `discard()`.

* Chips are integers. Real poker uses chip denominations and we don't
  care about cent-level precision; integers also avoid every floating
  point rounding bug you can imagine.

* All-in handling is centralised in `post()`. If a player is asked to
  post more than they have, they post everything they have, transition
  to ALL_IN, and the method returns the amount actually posted. The
  game loop reads that return value to update the pot. This is the
  *only* place chips leave a player, which makes the rule trivial to
  enforce.

* Per-hand vs per-round state. Some bookkeeping resets between betting
  rounds within a hand (`current_bet`), some resets between hands
  (`hand`, `total_contributed`, `status`). They have separate reset
  methods so the game loop can call the right one at the right
  boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence

from .deck import Card


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------
class PlayerStatus(Enum):
    """Where a player stands in the current hand.

    * ACTIVE: in the hand, has chips, can still bet.
    * FOLDED: out of the hand. Cannot win the pot. Stays FOLDED until
      the next hand starts.
    * ALL_IN: still in the hand and eligible to win up to their
      contribution, but has no chips left to bet, so they take no more
      betting actions.
    """

    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"

    def __str__(self) -> str:
        return self.value


# ----------------------------------------------------------------------
# Player
# ----------------------------------------------------------------------
@dataclass
class Player:
    """A seated player.

    `name` is just a label (used for logs and the tracker). `chips` is
    the bankroll. Everything else is hand- and round-scoped state that
    gets reset by the game loop at the appropriate boundary.
    """

    name: str
    chips: int

    # Current 5-card hand. Empty between hands.
    hand: List[Card] = field(default_factory=list)

    # Lifecycle status within the current hand.
    status: PlayerStatus = PlayerStatus.ACTIVE

    # Chips put in *during the current betting round*. Resets between
    # rounds. Used by the game loop to figure out what each player
    # still owes to call.
    current_bet: int = 0

    # Total chips this player has put in the pot during the entire
    # current hand (across both betting rounds). Used by the game loop
    # for side-pot construction when someone goes all-in.
    total_contributed: int = 0

    # ------------------------------------------------------------------
    # Convenience predicates — small wrappers, but they make the game
    # loop read like English ("if player.is_in_hand(): ...").
    # ------------------------------------------------------------------
    def is_active(self) -> bool:
        """True iff the player can still take betting actions."""
        return self.status is PlayerStatus.ACTIVE

    def is_in_hand(self) -> bool:
        """True iff the player is eligible to win (or split) the pot —
        i.e. has not folded. All-in players are still in the hand."""
        return self.status is not PlayerStatus.FOLDED

    def has_chips(self) -> bool:
        return self.chips > 0

    # ------------------------------------------------------------------
    # Card management
    # ------------------------------------------------------------------
    def receive_cards(self, cards: Sequence[Card]) -> None:
        """Receive a fresh 5-card deal. Replaces any existing hand —
        intended to be called once per hand by the dealer."""
        self.hand = list(cards)

    def discard(self, indices: Sequence[int]) -> List[Card]:
        """Discard cards by their position in `self.hand` (0..len-1).

        Returns the discarded cards so the game loop can return them to
        the deck's discard pile if it wants to track them. After this
        call, the player's hand is shorter — the game loop is expected
        to follow up with `receive_replacement(...)` to refill to 5.

        Raises ValueError on out-of-range or duplicate indices, because
        a silent off-by-one here would be very hard to debug at the
        showdown.
        """
        if not all(0 <= i < len(self.hand) for i in indices):
            raise ValueError(
                f"Discard indices {list(indices)} out of range for hand of size {len(self.hand)}."
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate discard indices: {list(indices)}")

        # Sort descending so each pop() doesn't shift the meaning of
        # later indices.
        discarded: List[Card] = []
        for i in sorted(indices, reverse=True):
            discarded.append(self.hand.pop(i))
        # Reverse so the returned list matches the original positional
        # order (more intuitive for logs).
        discarded.reverse()
        return discarded

    def receive_replacement(self, cards: Sequence[Card]) -> None:
        """Append replacement cards after a discard. Asserts the hand
        ends at 5 cards — drawing the wrong number is a game-loop bug
        we want to catch immediately."""
        self.hand.extend(cards)
        if len(self.hand) != 5:
            raise ValueError(
                f"After replacement, {self.name} has {len(self.hand)} cards (expected 5)."
            )

    # ------------------------------------------------------------------
    # Betting actions
    # ------------------------------------------------------------------
    def post(self, amount: int) -> int:
        """Move `amount` chips from this player's stack into the pot.

        Returns the amount actually posted, which may be less than
        `amount` if the player doesn't have enough chips — in which
        case the player goes all-in. This is the single chokepoint
        for chips leaving a player, so the all-in rule is enforced
        in exactly one place.

        Raises ValueError on negative amounts or if the player is
        already folded — both indicate a logic bug upstream.
        """
        if amount < 0:
            raise ValueError(f"Cannot post a negative amount (got {amount}).")
        if self.status is PlayerStatus.FOLDED:
            raise ValueError(f"{self.name} has folded and cannot post chips.")

        # Cap by available chips. If the player can't cover the full
        # request, they post everything and go all-in.
        posted = min(amount, self.chips)
        self.chips -= posted
        self.current_bet += posted
        self.total_contributed += posted

        if self.chips == 0 and self.status is PlayerStatus.ACTIVE:
            self.status = PlayerStatus.ALL_IN

        return posted

    def fold(self) -> None:
        """Forfeit the current hand. Idempotent — folding twice is a
        no-op rather than an error, because it's harmless and the game
        loop occasionally double-checks state."""
        if self.status is not PlayerStatus.FOLDED:
            self.status = PlayerStatus.FOLDED

    # ------------------------------------------------------------------
    # Resets — called by the game loop at round/hand boundaries
    # ------------------------------------------------------------------
    def reset_for_new_betting_round(self) -> None:
        """Clear per-round bet bookkeeping. Status (folded / all-in) is
        preserved — it persists for the whole hand."""
        self.current_bet = 0

    def reset_for_new_hand(self) -> None:
        """Clear all per-hand state. Chips persist (that's the point of
        the bankroll). If the player has no chips left, they stay
        FOLDED — they can't be dealt back in until the game logic
        decides what to do with broke players (rebuy / eliminate)."""
        self.hand = []
        self.current_bet = 0
        self.total_contributed = 0
        # A player with 0 chips cannot meaningfully be ACTIVE; the
        # bot-vs-bot tournament loop will probably eliminate them.
        # Until then, mark them folded so the round logic skips them.
        self.status = PlayerStatus.ACTIVE if self.has_chips() else PlayerStatus.FOLDED

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        hand_str = " ".join(str(c) for c in self.hand) if self.hand else "—"
        return (
            f"Player(name={self.name!r}, chips={self.chips}, "
            f"status={self.status.value}, bet={self.current_bet}, "
            f"hand=[{hand_str}])"
        )


# ----------------------------------------------------------------------
# Smoke test — run `python -m engine.player` from the project root.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from .deck import Deck

    deck = Deck(seed=1)
    deck.shuffle()
    p = Player(name="Adam", chips=100)
    print("Start:", p)

    # Deal 5 cards
    p.receive_cards(deck.deal(5))
    print("Dealt:", p)

    # Post a 10-chip bet
    posted = p.post(10)
    print(f"Posted {posted}; player now: {p}")

    # Discard positions 0 and 2, draw 2 replacements
    discarded = p.discard([0, 2])
    print("Discarded:", discarded)
    p.receive_replacement(deck.deal(2))
    print("After draw:", p)

    # Try to post more than they have — should go all-in
    posted = p.post(500)
    print(f"Tried to post 500, actually posted {posted}; player now: {p}")
    assert p.status is PlayerStatus.ALL_IN
    assert p.chips == 0

    # New hand — chips=0 means they should auto-fold
    p.reset_for_new_hand()
    print("After reset (broke):", p)
    assert p.status is PlayerStatus.FOLDED

    # Give them chips back, reset again — should be ACTIVE
    p.chips = 50
    p.reset_for_new_hand()
    print("After reset (with chips):", p)
    assert p.status is PlayerStatus.ACTIVE

    # Negative-post and folded-post both raise
    try:
        p.post(-5)
    except ValueError as e:
        print("Caught expected error:", e)
    p.fold()
    try:
        p.post(5)
    except ValueError as e:
        print("Caught expected error:", e)

    print("\nAll player.py smoke checks passed.")
