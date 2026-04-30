"""
game.py — 5-card draw poker round state machine.

The orchestration layer of the engine. It owns the deck, the seated
players, the betting state, and turns dealing → bet → draw → bet →
showdown into a single `play_hand()` call.

Design notes
------------
* The Game is **agnostic about who controls each seat**. A `Seat`
  pairs a `Player` (state) with a `PlayerAgent` (decisions). The
  agent is anything implementing the protocol — a human-input
  adapter, an LLM bot, a random test bot, all live outside this
  layer. The engine never imports from `bots/` or `ui/`.

* The agent sees the game through a `GameView` snapshot, which is
  immutable and tailored to one player (their own hand is visible;
  other players' hands are hidden). This is the single point where
  the brief's "what can LLMs see?" decision lives — adjust the view,
  don't touch the agents.

* All chips flow through `Player.post()`. The Game never directly
  mutates chip counts, which keeps the all-in rule enforced in one
  place. The "pot" is just the sum of `total_contributed` across
  players — no separate ledger to keep in sync.

* Side pots (when one or more players go all-in for less than the
  full bet) are computed at showdown by walking distinct contribution
  levels. Each layer goes to the best hand among players who paid
  into that layer.

* Betting round termination uses a `last_aggressor` pointer: the
  round ends when action wraps back around without anyone
  re-raising. This handles re-raises naturally without any
  special-casing.

* `Action.raise_to(N)` means "make my total bet for this round
  equal to N", not "raise by N". This is much clearer for LLM
  prompts ("raise to 50") than the alternative.

* Every hand emits a `HandSummary` with structured per-action
  records. The tracker layer (separate module) will serialise these
  to .jsonl for analytics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable,
)

from .deck import Deck, Card
from .hand import evaluate, HandResult
from .player import Player, PlayerStatus


# ----------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------
class Phase(Enum):
    PRE_DEAL = "pre_deal"
    FIRST_BETTING = "first_betting"
    DRAW = "draw"
    SECOND_BETTING = "second_betting"
    SHOWDOWN = "showdown"

    def __str__(self) -> str:
        return self.value


# ----------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------
class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Action:
    """An action taken by a player during a betting round.

    For RAISE, `amount` is the new *total bet level* for this round
    (the "to" in "raise to 50"). The Game works out how many chips
    that means the player must post given how much they've already
    bet this round.
    """

    type: ActionType
    amount: int = 0

    @classmethod
    def fold(cls) -> "Action":
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> "Action":
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls) -> "Action":
        return cls(ActionType.CALL)

    @classmethod
    def raise_to(cls, amount: int) -> "Action":
        return cls(ActionType.RAISE, amount)

    def __str__(self) -> str:
        if self.type is ActionType.RAISE:
            return f"raise to {self.amount}"
        return self.type.value


# ----------------------------------------------------------------------
# Agent protocol — what a seat needs to implement
# ----------------------------------------------------------------------
@runtime_checkable
class PlayerAgent(Protocol):
    """Decision-making interface. Implementations live outside the
    engine — humans (input prompt) and bots (LLM wrapper) both fit
    this shape."""

    def decide_action(self, view: "GameView") -> Action: ...
    def decide_discards(self, view: "GameView") -> Sequence[int]: ...


@dataclass
class Seat:
    """A player + their decision-maker. Order in the seat list defines
    turn order at the table."""

    player: Player
    agent: PlayerAgent


# ----------------------------------------------------------------------
# Public state — what agents see
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PublicPlayerInfo:
    """Everything observable about another seat. Notably no `hand` —
    that's private."""

    name: str
    chips: int
    current_bet: int
    total_contributed: int
    status: PlayerStatus
    cards_held: int     # 0..5, count only — the cards themselves are hidden


@dataclass(frozen=True)
class ActionRecord:
    """A logged action — used both for the agent's view of betting
    history during the hand, and persisted by the tracker afterwards."""

    player: str
    phase: Phase
    action: Action
    chips_posted: int   # actual chips moved on this action
    pot_after: int


@dataclass(frozen=True)
class GameView:
    """A snapshot of the game from one player's POV. Passed into
    `agent.decide_action()` and `agent.decide_discards()`. This is
    the entire interface the agent has to the world — change what's
    on this struct to change what the agent (LLM or human) can see."""

    # Self-info — only the active player sees their own hand.
    your_name: str
    your_hand: Tuple[Card, ...]
    your_chips: int
    your_current_bet: int
    your_total_contributed: int

    # Pot / round info
    pot: int
    highest_bet: int        # highest current_bet anyone has this round
    amount_to_call: int     # chips this player must post to call
    min_raise_to: int       # smallest legal "raise to" total

    # Other seats — public info only
    other_players: Tuple[PublicPlayerInfo, ...]

    # Phase + action history for this hand
    phase: Phase
    history: Tuple[ActionRecord, ...]


# ----------------------------------------------------------------------
# Hand summary — returned by play_hand() for the tracker
# ----------------------------------------------------------------------
@dataclass
class SeatResult:
    name: str
    starting_chips: int
    ending_chips: int
    net_change: int
    final_hand: Optional[Tuple[Card, ...]]
    final_evaluation: Optional[HandResult]
    folded: bool


@dataclass
class HandSummary:
    hand_id: int
    actions: List[ActionRecord]
    seat_results: List[SeatResult]
    winners: List[Tuple[str, int]]      # (name, chips_won) — one row per pot win


# ----------------------------------------------------------------------
# Game class
# ----------------------------------------------------------------------
class Game:
    """One table of 5-card draw."""

    def __init__(
        self,
        seats: Sequence[Seat],
        ante: int = 1,
        min_bet: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if len(seats) < 2:
            raise ValueError("Need at least 2 seats to play poker.")
        self.seats: List[Seat] = list(seats)
        self.ante = ante
        self.min_bet = min_bet
        self.deck = Deck(seed=seed)
        self.hand_count = 0
        self.dealer_index = 0       # rotates each hand

    # ------------------------------------------------------------------
    # Convenience views over seats
    # ------------------------------------------------------------------
    @property
    def players(self) -> List[Player]:
        return [s.player for s in self.seats]

    def _agent_for(self, name: str) -> PlayerAgent:
        for s in self.seats:
            if s.player.name == name:
                return s.agent
        raise KeyError(name)

    def _pot(self) -> int:
        return sum(p.total_contributed for p in self.players)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def play_hand(self) -> HandSummary:
        """Play one full hand of 5-card draw, mutating player chip
        counts. Returns a structured summary suitable for logging /
        tracking."""

        self.hand_count += 1
        history: List[ActionRecord] = []

        # Snapshot starting chips for the summary.
        starting = {p.name: p.chips for p in self.players}

        # 1. Reset everyone for the new hand. Players with 0 chips
        #    auto-fold inside reset_for_new_hand().
        for p in self.players:
            p.reset_for_new_hand()

        # 2. Reshuffle a fresh deck.
        self.deck.reset(shuffle=True)

        # 3. Antes — anyone who can't ante folds out of this hand.
        self._ante_up(history)

        # 4. Deal 5 cards to each in-hand player.
        self._deal()

        # 5. First betting round.
        if self._count_in_hand() > 1:
            self._betting_round(Phase.FIRST_BETTING, history)

        # 6. Draw phase: discard + replace, for those still in.
        if self._count_in_hand() > 1:
            self._draw_phase(history)

        # 7. Second betting round.
        if self._count_in_hand() > 1:
            self._betting_round(Phase.SECOND_BETTING, history)

        # 8. Award the pot.
        winners = self._settle(history)

        # 9. Build the per-seat summary.
        seat_results: List[SeatResult] = []
        for p in self.players:
            evaluation = (
                evaluate(p.hand) if (p.is_in_hand() and len(p.hand) == 5) else None
            )
            seat_results.append(SeatResult(
                name=p.name,
                starting_chips=starting[p.name],
                ending_chips=p.chips,
                net_change=p.chips - starting[p.name],
                final_hand=tuple(p.hand) if p.hand else None,
                final_evaluation=evaluation,
                folded=p.status is PlayerStatus.FOLDED,
            ))

        # Rotate the dealer for next hand.
        self.dealer_index = (self.dealer_index + 1) % len(self.seats)

        return HandSummary(
            hand_id=self.hand_count,
            actions=history,
            seat_results=seat_results,
            winners=winners,
        )

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def _ante_up(self, history: List[ActionRecord]) -> None:
        """Each player posts the ante. Players who can't afford it sit
        out this hand (marked FOLDED)."""
        if self.ante <= 0:
            return
        for p in self.players:
            if p.status is PlayerStatus.FOLDED:
                continue
            if p.chips < self.ante:
                # Can't ante: sit out.
                p.fold()
                continue
            posted = p.post(self.ante)
            history.append(ActionRecord(
                player=p.name,
                phase=Phase.PRE_DEAL,
                action=Action(ActionType.CALL, amount=self.ante),
                chips_posted=posted,
                pot_after=self._pot(),
            ))
        # Antes don't count as "current bet" for the betting round —
        # they're a forced contribution. Reset the per-round bet
        # tracking before the first betting round opens.
        for p in self.players:
            p.reset_for_new_betting_round()

    def _deal(self) -> None:
        """Deal 5 cards to every player still in the hand."""
        for p in self.players:
            if p.is_in_hand():
                p.receive_cards(self.deck.deal(5))

    def _draw_phase(self, history: List[ActionRecord]) -> None:
        """Each in-hand, non-all-in player chooses cards to discard;
        the deck deals replacements."""
        for p in self.players:
            if p.status is not PlayerStatus.ACTIVE:
                continue        # folded or all-in players don't draw
            view = self._build_view(p, Phase.DRAW, history)
            agent = self._agent_for(p.name)
            indices = list(agent.decide_discards(view))
            # Defensive validation — agents should never violate this
            # but a stray bot prompt could.
            indices = [i for i in indices if 0 <= i < 5]
            indices = sorted(set(indices))
            if not indices:
                continue        # standing pat
            p.discard(indices)
            p.receive_replacement(self.deck.deal(len(indices)))
            history.append(ActionRecord(
                player=p.name,
                phase=Phase.DRAW,
                # Re-use Action with amount = number of cards drawn.
                # This is a slight overload but keeps the log uniform.
                action=Action(ActionType.RAISE, amount=len(indices)),
                chips_posted=0,
                pot_after=self._pot(),
            ))

    # ------------------------------------------------------------------
    # Betting round — the heart of the state machine
    # ------------------------------------------------------------------
    def _betting_round(self, phase: Phase, history: List[ActionRecord]) -> None:
        """Run one betting round (first or second).

        Termination: maintain a `last_aggressor` index. Each call action
        moves the turn forward. A raise resets `last_aggressor` to the
        raiser. The round ends when the turn advances back to the
        last_aggressor without anyone re-raising in between (i.e. action
        has gone full circle and everyone has matched the highest bet).
        """
        # Per-round bookkeeping: clear `current_bet` on every player.
        for p in self.players:
            p.reset_for_new_betting_round()

        n = len(self.seats)
        # First to act is the seat after the dealer who's still active.
        first = self._next_active(self.dealer_index)
        if first is None:
            return      # nobody can act
        current = first
        last_aggressor = first
        # `started` flag because we only check "wrapped back to
        # last_aggressor" *after* at least one action has been taken.
        started = False

        while True:
            p = self.seats[current].player

            # If this player is still active (not folded, not all-in),
            # ask them to act.
            if p.status is PlayerStatus.ACTIVE:
                # If there's only one in-hand player left, betting stops.
                if self._count_in_hand() <= 1:
                    return

                view = self._build_view(p, phase, history)
                agent = self._agent_for(p.name)
                action = agent.decide_action(view)
                action = self._normalise_action(p, view, action)
                posted, was_aggressive = self._apply_action(p, view, action)

                history.append(ActionRecord(
                    player=p.name,
                    phase=phase,
                    action=action,
                    chips_posted=posted,
                    pot_after=self._pot(),
                ))

                if was_aggressive:
                    last_aggressor = current

                # If everyone else has folded, end immediately.
                if self._count_in_hand() <= 1:
                    return
                # If no remaining ACTIVE players (all all-in/folded), stop.
                if not any(s.player.status is PlayerStatus.ACTIVE for s in self.seats):
                    return

            # Advance to the next seat.
            current = (current + 1) % n
            started = True
            # End-of-round condition: we've come full circle back to
            # the last aggressor without anyone re-raising.
            if started and current == last_aggressor:
                # But only end if the last_aggressor has already acted
                # this round. They have, by definition, since they set
                # the position via raising.
                # Edge case: if last_aggressor folded all-in
                # immediately (impossible normally), we'd loop forever
                # — guard against that by checking they actually had a
                # turn. They did, because they raised.
                return

    def _normalise_action(self, p: Player, view: GameView, action: Action) -> Action:
        """Coerce dubious / illegal actions to their nearest legal
        equivalent. Two reasons:
          * Bots and humans both produce sloppy outputs; better to
            interpret charitably than crash mid-hand.
          * The semantics of "check" vs "call 0" and "call" with
            insufficient chips need sorting out somewhere."""
        if action.type is ActionType.FOLD:
            return action
        if action.type is ActionType.CHECK:
            # If there's something to call, a CHECK is illegal —
            # interpret as CALL.
            return Action.call() if view.amount_to_call > 0 else action
        if action.type is ActionType.CALL:
            return action
        if action.type is ActionType.RAISE:
            # Raise must be at least min_raise_to. If too small,
            # silently bump it. If the player can't afford min_raise_to,
            # they go all-in via the post() cap.
            target = max(action.amount, view.min_raise_to)
            return Action.raise_to(target)
        return action

    def _apply_action(
        self, p: Player, view: GameView, action: Action,
    ) -> Tuple[int, bool]:
        """Execute the action and return `(chips_posted, was_aggressive)`.
        `was_aggressive` is True iff the action raises the highest bet
        in this round — used by the betting-round loop to update the
        last_aggressor pointer."""
        if action.type is ActionType.FOLD:
            p.fold()
            return 0, False

        if action.type is ActionType.CHECK:
            return 0, False

        if action.type is ActionType.CALL:
            owed = view.amount_to_call
            posted = p.post(owed) if owed > 0 else 0
            return posted, False

        if action.type is ActionType.RAISE:
            # `amount` is the new total bet level for this round.
            owed = action.amount - p.current_bet
            if owed <= 0:
                # Degenerate: "raise" to a level we're already at.
                # Treat as a check.
                return 0, False
            posted = p.post(owed)
            # An aggressive action only counts as one if it actually
            # raised the high water mark. A "raise" by an all-in player
            # who can't make the minimum still counts because they put
            # extra chips in (other players need a chance to respond).
            new_bet_level = view.your_current_bet + posted
            was_aggressive = new_bet_level > view.highest_bet
            return posted, was_aggressive

        raise ValueError(f"Unknown action type: {action.type}")

    # ------------------------------------------------------------------
    # Pot construction and showdown
    # ------------------------------------------------------------------
    def _settle(self, history: List[ActionRecord]) -> List[Tuple[str, int]]:
        """Award all pots. Returns a list of (winner_name, chips_won)
        records — one per pot per winner (so a 3-way split gives three
        records for that pot)."""
        pots = self._build_pots()
        winners: List[Tuple[str, int]] = []

        # Evaluate each in-hand player's final hand once.
        evaluations: Dict[str, HandResult] = {}
        for p in self.players:
            if p.is_in_hand() and len(p.hand) == 5:
                evaluations[p.name] = evaluate(p.hand)

        for amount, eligible_names in pots:
            # Among eligibles, find the best hand(s).
            scored: List[Tuple[str, HandResult]] = [
                (name, evaluations[name]) for name in eligible_names
                if name in evaluations
            ]
            if not scored:
                # Edge case: pot with no eligible evaluable hands. Can
                # arise if everyone in the pot folded. Award to the
                # last in-hand player, falling back to the highest
                # contributor (shouldn't happen in normal play).
                survivors = [p for p in self.players if p.is_in_hand()]
                if survivors:
                    target = survivors[0]
                    target.chips += amount
                    winners.append((target.name, amount))
                continue

            best_key = max(r.key for _, r in scored)
            top = [name for name, r in scored if r.key == best_key]

            share = amount // len(top)
            remainder = amount - share * len(top)
            # Whole-chip remainder goes to the first winner in turn
            # order. (Real cardrooms award it to the first player left
            # of the dealer; close enough.)
            for i, name in enumerate(top):
                give = share + (remainder if i == 0 else 0)
                self._player_by_name(name).chips += give
                winners.append((name, give))

        return winners

    def _build_pots(self) -> List[Tuple[int, List[str]]]:
        """Compute main + side pots. Returns a list of `(amount,
        eligible_names)` from main-pot first to outermost side pot."""
        contributors = [p for p in self.players if p.total_contributed > 0]
        if not contributors:
            return []

        # Distinct contribution levels, ascending. Each adjacent pair
        # of levels defines one "layer" of the pot.
        levels = sorted({p.total_contributed for p in contributors})

        pots: List[Tuple[int, List[str]]] = []
        prev_level = 0
        for level in levels:
            layer_height = level - prev_level
            funders = [p for p in contributors if p.total_contributed >= level]
            amount = layer_height * len(funders)
            if amount <= 0:
                prev_level = level
                continue
            eligibles = [p.name for p in funders if p.is_in_hand()]
            if eligibles:
                pots.append((amount, eligibles))
            else:
                # Pathological: all funders of this layer folded.
                # Return chips to the largest contributor still around.
                survivor_pool = [p for p in self.players if p.is_in_hand()]
                if survivor_pool:
                    pots.append((amount, [survivor_pool[0].name]))
            prev_level = level
        return pots

    def _player_by_name(self, name: str) -> Player:
        for p in self.players:
            if p.name == name:
                return p
        raise KeyError(name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _count_in_hand(self) -> int:
        return sum(1 for p in self.players if p.is_in_hand())

    def _next_active(self, from_index: int) -> Optional[int]:
        """Return the index of the next ACTIVE seat after `from_index`,
        or None if there are no active seats."""
        n = len(self.seats)
        for offset in range(1, n + 1):
            idx = (from_index + offset) % n
            if self.seats[idx].player.status is PlayerStatus.ACTIVE:
                return idx
        return None

    def _build_view(
        self, p: Player, phase: Phase, history: Sequence[ActionRecord],
    ) -> GameView:
        """Construct the player-specific snapshot passed to the agent."""
        highest_bet = max((q.current_bet for q in self.players), default=0)
        amount_to_call = max(0, highest_bet - p.current_bet)
        # Minimum legal raise: the highest bet plus the min_bet
        # increment. (No-limit poker has more nuanced rules involving
        # the previous raise size; we keep it simple.)
        min_raise_to = highest_bet + self.min_bet

        others = tuple(
            PublicPlayerInfo(
                name=q.name,
                chips=q.chips,
                current_bet=q.current_bet,
                total_contributed=q.total_contributed,
                status=q.status,
                cards_held=len(q.hand),
            )
            for q in self.players if q is not p
        )

        return GameView(
            your_name=p.name,
            your_hand=tuple(p.hand),
            your_chips=p.chips,
            your_current_bet=p.current_bet,
            your_total_contributed=p.total_contributed,
            pot=self._pot(),
            highest_bet=highest_bet,
            amount_to_call=amount_to_call,
            min_raise_to=min_raise_to,
            other_players=others,
            phase=phase,
            history=tuple(history),
        )


# ----------------------------------------------------------------------
# A toy random agent for the smoke test. Real bots live in `bots/`.
# ----------------------------------------------------------------------
class _RandomAgent:
    """Picks legal actions uniformly at random. Used only by the
    `python -m engine.game` smoke test below."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def decide_action(self, view: GameView) -> Action:
        choices: List[str] = ["call_or_check"]
        if view.amount_to_call > 0:
            choices.append("fold")
        # Only consider raising if we can afford the minimum.
        if view.your_chips >= max(0, view.min_raise_to - view.your_current_bet):
            choices.append("raise_min")
        choice = self.rng.choice(choices)
        if choice == "fold":
            return Action.fold()
        if choice == "raise_min":
            return Action.raise_to(view.min_raise_to)
        return Action.call() if view.amount_to_call > 0 else Action.check()

    def decide_discards(self, view: GameView) -> Sequence[int]:
        # Discard a random subset of size 0..3 — typical 5-card draw
        # behaviour.
        n = self.rng.randint(0, 3)
        return self.rng.sample(range(5), n) if n > 0 else []


# ----------------------------------------------------------------------
# Smoke test: run `python -m engine.game` from the project root.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    seats = [
        Seat(Player("Alice",   chips=200), _RandomAgent(seed=1)),
        Seat(Player("Bob",     chips=200), _RandomAgent(seed=2)),
        Seat(Player("Carol",   chips=200), _RandomAgent(seed=3)),
        Seat(Player("Dave",    chips=200), _RandomAgent(seed=4)),
    ]
    game = Game(seats, ante=2, min_bet=5, seed=42)

    for h in range(3):
        summary = game.play_hand()
        print(f"\n=== Hand {summary.hand_id} ===")
        for rec in summary.actions:
            posted = f" (+{rec.chips_posted})" if rec.chips_posted else ""
            print(f"  [{rec.phase}] {rec.player}: {rec.action}{posted}  pot={rec.pot_after}")
        print("  Winners:", summary.winners)
        for sr in summary.seat_results:
            tag = "folded" if sr.folded else (
                str(sr.final_evaluation.category.label) if sr.final_evaluation else "—"
            )
            print(f"    {sr.name}: {sr.starting_chips} -> {sr.ending_chips} "
                  f"(net {sr.net_change:+}) [{tag}]")

    print("\nFinal chip counts:")
    for p in game.players:
        print(f"  {p.name}: {p.chips}")
    total = sum(p.chips for p in game.players)
    print(f"  total = {total}  (started at {200*4} = {200*4})")
    assert total == 200 * 4, "Chip conservation failed!"
    print("Chip conservation OK.")
