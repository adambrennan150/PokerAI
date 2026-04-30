"""
human.py — a HumanAgent that drives the engine via text input.

Implements the engine's `PlayerAgent` protocol. On each `decide_*`
call it:
  1. Renders the table (HTML in Jupyter, text otherwise).
  2. Prompts via `input()`.
  3. Parses leniently and re-prompts on failure.

Design notes
------------
* Parsing lives in pure functions (`parse_action_input`,
  `parse_discard_input`) that the agent class only orchestrates. This
  keeps the parsing logic unit-testable without mocking `input()` —
  the smoke test below covers ~30 input variants in milliseconds.

* Display mode auto-detects: HTML if running in a Jupyter / IPython
  kernel, plain text otherwise. Forced via `display_mode="html"` or
  `"text"` if the auto-detect mis-fires (e.g. running this from a
  Jupyter terminal where you actually want plain text).

* Lenient parsing. Humans type "f", "fold", "raise 25", "r 25",
  "raise to 25", "all-in", "allin", "", and we treat them all
  reasonably. Empty string defaults to check when legal — the most
  common keystroke-to-action mapping.

* Re-prompts on parse failure rather than crashing — the agent
  protocol is sync, and one fat-fingered input shouldn't end the
  hand.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from engine import Action, ActionType, GameView

from .rendering import render_table_html, render_table_text


# ----------------------------------------------------------------------
# Pure parsers — testable without input() mocking
# ----------------------------------------------------------------------
# Compiled once at import time. The action regex matches things like:
#     "raise 25", "r 25", "raise to 25", "r to 25"
_RAISE_RE = re.compile(r"^\s*(?:raise|r)\s*(?:to\s+)?(\d+)\s*$", re.IGNORECASE)
# Discard input: anything containing digits 0..4. We pull all digit
# tokens and validate after.
_DIGIT_RE = re.compile(r"\d+")


def parse_action_input(text: str, view: GameView) -> Optional[Action]:
    """Best-effort parse of human input → engine `Action`. Returns
    `None` if the input is unparseable (caller should re-prompt).

    Accepted forms:
        ""                       -> check if legal, else None
        "fold" / "f"
        "check" / "ch"
        "call" / "c"             (also "check" if amount_to_call==0)
        "raise N" / "r N"
        "raise to N" / "r to N"
        "all-in" / "allin" / "all in"
    """
    s = text.strip().lower()

    # Empty input: most common case is the user hitting Enter to
    # check when free.
    if s == "":
        return Action.check() if view.amount_to_call == 0 else None

    if s in ("f", "fold"):
        return Action.fold()

    if s in ("ch", "check"):
        # Engine normalises a check-when-must-call to a call, but the
        # human's clear intent here is "I don't want to put more chips
        # in", so on a non-zero call we reject and re-prompt.
        return Action.check() if view.amount_to_call == 0 else None

    if s in ("c", "call"):
        # If there's nothing to call, treat 'call' as 'check'. Saves
        # an annoying re-prompt when the user types 'c' on a free
        # check round.
        return Action.call() if view.amount_to_call > 0 else Action.check()

    if s in ("all-in", "allin", "all in", "shove", "jam"):
        # Bet everything: total bet level = current bet + remaining stack.
        return Action.raise_to(view.your_current_bet + view.your_chips)

    m = _RAISE_RE.match(s)
    if m:
        return Action.raise_to(int(m.group(1)))

    return None


def parse_discard_input(text: str) -> Optional[List[int]]:
    """Best-effort parse of discard input → list of indices in 0..4.

    Returns `None` only if the input mentions out-of-range indices —
    ambiguous or empty input becomes "stand pat" (`[]`), which is
    always legal and a sensible default.

    Accepted forms:
        ""                  -> []
        "stand" / "pat" / "none"
        "all"               -> [0,1,2,3,4]
        "0,2,4" / "[0, 2, 4]" / "0 2 4" / "024"
    """
    s = text.strip().lower()

    if s == "" or s in ("stand", "pat", "none", "stand pat", "no"):
        return []

    if s == "all":
        return [0, 1, 2, 3, 4]

    digits = _DIGIT_RE.findall(s)
    if not digits:
        return None

    indices: List[int] = []
    for tok in digits:
        # If the user types "024" (no separator), treat each character
        # as one index. Otherwise the token is a single index.
        if len(tok) > 1:
            for ch in tok:
                indices.append(int(ch))
        else:
            indices.append(int(tok))

    # Reject explicit out-of-range — better to re-prompt than to
    # silently drop "5" and discard fewer cards than the user wanted.
    if any(i < 0 or i > 4 for i in indices):
        return None

    # De-dup, preserve order.
    seen = set()
    clean: List[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            clean.append(i)
    return clean


# ----------------------------------------------------------------------
# HumanAgent
# ----------------------------------------------------------------------
class HumanAgent:
    """A text-input PlayerAgent for human play in a notebook or terminal.

    Usage:
        agent = HumanAgent(name="Adam")
        # Pass it to engine via Seat(player, agent) just like a bot.
    """

    def __init__(
        self,
        name: str = "Human",
        display_mode: str = "auto",     # "auto" | "html" | "text"
    ) -> None:
        self.name = name
        if display_mode not in ("auto", "html", "text"):
            raise ValueError("display_mode must be 'auto', 'html', or 'text'")
        self.display_mode = display_mode

    # ------------------------------------------------------------------
    # PlayerAgent protocol
    # ------------------------------------------------------------------
    def decide_action(self, view: GameView) -> Action:
        self._show(view)
        prompt = self._action_prompt(view)
        while True:
            try:
                text = input(prompt)
            except EOFError:
                # Stdin closed (often in tests) — fall back to safest
                # action so the hand can still complete.
                return Action.fold() if view.amount_to_call > 0 else Action.check()
            action = parse_action_input(text, view)
            if action is not None:
                return action
            print("  Couldn't parse that. Try: fold | check | call | raise N | all-in")

    def decide_discards(self, view: GameView) -> Sequence[int]:
        self._show(view)
        prompt = self._discard_prompt(view)
        while True:
            try:
                text = input(prompt)
            except EOFError:
                return []
            indices = parse_discard_input(text)
            if indices is not None:
                return indices
            print("  Couldn't parse that. Use indices 0..4, e.g. '0,2,4', "
                  "'all', or empty to stand pat.")

    # ------------------------------------------------------------------
    # Display + prompt formatting
    # ------------------------------------------------------------------
    def _show(self, view: GameView) -> None:
        if self._use_html():
            try:
                from IPython.display import HTML, display
                display(HTML(render_table_html(view)))
                return
            except ImportError:
                pass    # fall through to text
        print(render_table_text(view))

    def _use_html(self) -> bool:
        if self.display_mode == "html":
            return True
        if self.display_mode == "text":
            return False
        # auto: use HTML iff we appear to be in an IPython kernel.
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is None:
                return False
            # Kernel has IPKernelApp in config; bare IPython terminal
            # does not, and falls back to text.
            return "IPKernelApp" in ipy.config
        except ImportError:
            return False

    @staticmethod
    def _action_prompt(view: GameView) -> str:
        opts = []
        if view.amount_to_call > 0:
            opts.append("fold")
            opts.append(f"call ({view.amount_to_call})")
        else:
            opts.append("check")
        if view.your_chips > 0:
            opts.append(f"raise to N (>= {view.min_raise_to})")
        choices = " | ".join(opts)
        return f"  > Your action [{choices}]: "

    @staticmethod
    def _discard_prompt(view: GameView) -> str:
        return "  > Discard which indices? (0..4, comma-separated; empty = stand pat): "


# ----------------------------------------------------------------------
# Smoke test — `python -m ui.human`
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from engine import (
        Card, GameView, Phase, PlayerStatus, PublicPlayerInfo, Rank, Suit,
    )

    # Synthetic GameView — same one the bots/base smoke test uses.
    hand = (
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.JACK, Suit.DIAMONDS),
        Card(Rank.TEN, Suit.CLUBS),
    )
    view_to_call = GameView(
        your_name="Adam",
        your_hand=hand,
        your_chips=190, your_current_bet=5, your_total_contributed=7,
        pot=23, highest_bet=10, amount_to_call=5, min_raise_to=15,
        other_players=(
            PublicPlayerInfo(name="Bob", chips=200, current_bet=10,
                             total_contributed=12,
                             status=PlayerStatus.ACTIVE, cards_held=5),
            PublicPlayerInfo(name="Carol", chips=180, current_bet=10,
                             total_contributed=12,
                             status=PlayerStatus.ACTIVE, cards_held=5),
            PublicPlayerInfo(name="Dave", chips=0, current_bet=0,
                             total_contributed=2,
                             status=PlayerStatus.FOLDED, cards_held=5),
        ),
        phase=Phase.FIRST_BETTING, history=(),
    )
    view_free_check = GameView(
        your_name="Adam", your_hand=hand,
        your_chips=190, your_current_bet=0, your_total_contributed=2,
        pot=8, highest_bet=0, amount_to_call=0, min_raise_to=5,
        other_players=view_to_call.other_players,
        phase=Phase.SECOND_BETTING, history=(),
    )

    # ----------------- Action parser tests -----------------
    print("=== parse_action_input tests ===")
    cases_to_call = [
        ("fold",        ActionType.FOLD,  None),
        ("f",           ActionType.FOLD,  None),
        ("call",        ActionType.CALL,  None),
        ("c",           ActionType.CALL,  None),
        ("raise 25",    ActionType.RAISE, 25),
        ("r 30",        ActionType.RAISE, 30),
        ("raise to 50", ActionType.RAISE, 50),
        ("R TO 100",    ActionType.RAISE, 100),
        ("all-in",      ActionType.RAISE, 195),  # current_bet 5 + chips 190
        ("allin",       ActionType.RAISE, 195),
        ("shove",       ActionType.RAISE, 195),
        # Empty + check + nonsense should be None when there's a call.
        ("",            None, None),
        ("check",       None, None),
        ("hello",       None, None),
        ("raise abc",   None, None),
    ]
    for text, want_kind, want_amt in cases_to_call:
        got = parse_action_input(text, view_to_call)
        if want_kind is None:
            assert got is None, f"{text!r} -> {got} (want None)"
        else:
            assert got is not None and got.type is want_kind
            if want_amt is not None:
                assert got.amount == want_amt, f"{text!r} amount={got.amount} (want {want_amt})"
        print(f"  {text!r:<20s} -> {got}")

    print()
    cases_free_check = [
        ("",       ActionType.CHECK, None),
        ("check",  ActionType.CHECK, None),
        ("ch",     ActionType.CHECK, None),
        ("call",   ActionType.CHECK, None),    # treated as check when free
        ("c",      ActionType.CHECK, None),
        ("fold",   ActionType.FOLD,  None),
        ("raise 5",ActionType.RAISE, 5),
    ]
    for text, want_kind, want_amt in cases_free_check:
        got = parse_action_input(text, view_free_check)
        assert got is not None and got.type is want_kind, f"{text!r} -> {got}"
        if want_amt is not None:
            assert got.amount == want_amt
        print(f"  {text!r:<20s} -> {got}   (free-check view)")

    # ----------------- Discard parser tests -----------------
    print("\n=== parse_discard_input tests ===")
    discard_cases = [
        ("",            []),
        ("stand",       []),
        ("pat",         []),
        ("none",        []),
        ("all",         [0, 1, 2, 3, 4]),
        ("0",           [0]),
        ("4",           [4]),
        ("0,2,4",       [0, 2, 4]),
        ("0, 2, 4",     [0, 2, 4]),
        ("0 2 4",       [0, 2, 4]),
        ("[0, 2, 4]",   [0, 2, 4]),
        ("024",         [0, 2, 4]),     # consecutive digits
        ("0,0,2",       [0, 2]),         # de-dup
        ("5",           None),           # out of range -> reject
        ("0,5",         None),
        ("abc",         None),           # no digits -> ambiguous
    ]
    for text, want in discard_cases:
        got = parse_discard_input(text)
        assert got == want, f"{text!r} -> {got} (want {want})"
        print(f"  {text!r:<14s} -> {got}")

    # ----------------- Render output sanity checks -----------------
    print("\n=== render_table_text output ===\n")
    out = render_table_text(view_to_call)
    print(out)
    assert "POT: 23" in out
    assert "YOUR HAND: A♠ K♠ Q♥ J♦ T♣" in out
    assert "Bob" in out and "Carol" in out and "Dave" in out
    assert "to_call=5" in out

    print("\n=== render_table_html sanity ===")
    html = render_table_html(view_to_call)
    assert "<div" in html
    assert "POT:" in html
    assert "K" in html  # rank labels
    assert "♠" in html
    assert "Bob" in html and "Carol" in html
    print(f"  HTML length: {len(html)} chars (looks structurally OK)")

    # ----------------- HumanAgent end-to-end with stubbed input ----
    print("\n=== HumanAgent with stubbed input ===")
    import builtins
    inputs = iter(["raise 25", "0,2"])     # action then discards
    builtins.input = lambda prompt="": next(inputs)
    try:
        agent = HumanAgent(name="Adam", display_mode="text")
        action = agent.decide_action(view_to_call)
        assert action.type is ActionType.RAISE and action.amount == 25
        discards = agent.decide_discards(view_to_call)
        assert list(discards) == [0, 2]
        print(f"  Action parsed:   {action}")
        print(f"  Discards parsed: {list(discards)}")
    finally:
        # Restore builtins.input — no global pollution after the test.
        builtins.input = __builtins__.input if isinstance(__builtins__, dict) \
            else __builtins__.input

    print("\nAll ui smoke checks passed.")
