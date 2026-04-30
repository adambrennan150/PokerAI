"""
base.py — abstract base class for all LLM-powered poker bots.

Design notes
------------
* `BaseBot` is the bridge between the engine and any concrete LLM. It
  satisfies the engine's `PlayerAgent` protocol (`decide_action`,
  `decide_discards`), and exposes a single hook for subclasses:
  `_generate(system, user) -> str`. An Ollama bot, a Claude bot, a
  GPT bot — they all only differ in how that one method is wired up.

* A bot's `Personality` is a data object: an `id` (used for tracker
  groupby), a human description, and a `system_prompt` that gets
  prepended to every LLM call. This is how the same code drives an
  "aggressive" bot and a "conservative" bot — by swapping data, not
  code.

* Output format is strict JSON. LLMs are generally good at producing
  it when prompted, but they will sometimes wrap the JSON in prose,
  use code fences, or produce malformed structures. The parser walks
  three layers of defence before giving up:
      1. Regex-extract the first {...} block and `json.loads` it.
      2. Keyword-scan the response ("fold", "call", "raise N").
      3. Fall back to a safe default (call if free, otherwise fold).
  In practice layer 1 catches >95% of responses; the rest exist so
  one bad token from the model can't crash a 100-hand tournament.

* Every call records a `BotResponse` on `self.last_response`,
  capturing the raw text, the LLM's stated reasoning, the parsed
  action, and any parse error. The tracker layer reads this after
  each engine call to log "what the LLMs are doing and saying", as
  the brief requires.

* Concrete bot classes (Ollama, Claude, etc.) live in sibling files.
  This module ships a tiny `MockBot` that responds with hard-coded
  JSON — useful for unit-testing the prompt/parsing pipeline before
  any LLM is downloaded.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from engine import (
    Action, ActionType, Card, GameView, Phase, PlayerStatus,
)


# ----------------------------------------------------------------------
# Personality: data describing how a bot should "play"
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Personality:
    """A play-style identity. The `id` is the groupby key for the
    tracker's "best personality on average" analytics."""

    id: str                 # short slug, e.g. "aggressive"
    description: str        # one-liner for logs / reports
    system_prompt: str      # the instruction style that goes to the LLM

    def __str__(self) -> str:
        return self.id


# ----------------------------------------------------------------------
# BotResponse: what the engine got back, plus everything the tracker
# wants to log.
# ----------------------------------------------------------------------
@dataclass
class BotResponse:
    """One LLM round-trip's worth of bookkeeping. The bot stores its
    most recent on `self.last_response` so the tracker (or game loop)
    can read it after each `decide_action` / `decide_discards` call."""

    action: Optional[Action] = None
    discards: Optional[List[int]] = None
    raw_response: str = ""
    reasoning: str = ""
    parse_error: Optional[str] = None
    prompt: str = ""        # the full user prompt — handy for debugging


# ----------------------------------------------------------------------
# BaseBot: the abstract LLM bot. Subclasses only need to provide
# `_generate(system, user) -> str`.
# ----------------------------------------------------------------------
class BaseBot(ABC):
    """Abstract LLM-powered poker bot. Implements the engine's
    `PlayerAgent` protocol via prompt formatting + JSON parsing.

    Subclasses provide the actual LLM call by overriding `_generate`.
    """

    def __init__(self, name: str, personality: Personality, model_id: str) -> None:
        # `name` is what shows on the table (and in the engine's
        # Player.name). `model_id` is the LLM identifier used by the
        # tracker to group results by model — e.g. "llama3:8b". The
        # `(model_id, personality.id)` pair is the bot's full identity.
        self.name = name
        self.personality = personality
        self.model_id = model_id
        self.last_response: Optional[BotResponse] = None

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    @abstractmethod
    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send (system, user) to the LLM and return the raw text
        response. Implementations should NOT do any parsing — that's
        the base class's job. Should raise on transport errors so the
        base class can fall back gracefully."""

    # ------------------------------------------------------------------
    # PlayerAgent protocol — what the engine calls
    # ------------------------------------------------------------------
    def decide_action(self, view: GameView) -> Action:
        """Format the action prompt, ask the LLM, parse the answer."""
        prompt = self._format_action_prompt(view)
        raw = self._safe_generate(prompt)
        action, reasoning, error = self._parse_action_response(raw, view)
        self.last_response = BotResponse(
            action=action,
            raw_response=raw,
            reasoning=reasoning,
            parse_error=error,
            prompt=prompt,
        )
        return action

    def decide_discards(self, view: GameView) -> Sequence[int]:
        """Format the draw prompt, ask the LLM, parse the answer."""
        prompt = self._format_discard_prompt(view)
        raw = self._safe_generate(prompt)
        discards, reasoning, error = self._parse_discard_response(raw)
        self.last_response = BotResponse(
            discards=list(discards),
            raw_response=raw,
            reasoning=reasoning,
            parse_error=error,
            prompt=prompt,
        )
        return discards

    # ------------------------------------------------------------------
    # Safety wrapper around _generate so a transport error doesn't
    # crash the hand.
    # ------------------------------------------------------------------
    def _safe_generate(self, prompt: str) -> str:
        try:
            return self._generate(self.personality.system_prompt, prompt)
        except Exception as e:      # noqa: BLE001 — broad on purpose
            # We swallow *any* exception from the LLM transport. If the
            # local Ollama server has crashed mid-tournament we don't
            # want a 200-hand run to fail; we want the bot to take a
            # safe default action and let the tournament continue. The
            # error gets recorded in `parse_error` for forensics.
            return f"<<<LLM_ERROR: {type(e).__name__}: {e}>>>"

    # ------------------------------------------------------------------
    # Prompt formatting — the same shape regardless of model. Subclasses
    # *can* override these if a particular LLM benefits from a different
    # phrasing, but the default works for everything we plan to use.
    # ------------------------------------------------------------------
    def _format_action_prompt(self, view: GameView) -> str:
        """Render the GameView as a betting decision prompt."""

        hand_str = " ".join(str(c) for c in view.your_hand) if view.your_hand else "(none)"
        others_str = "\n".join(self._format_other_player(o) for o in view.other_players)
        history_str = self._format_history(view.history) or "  (no actions yet)"
        legal_str = self._format_legal_actions(view)

        return (
            f"You are playing 5-card draw poker. It is your turn to act.\n\n"
            f"YOUR SEAT\n"
            f"  Name: {view.your_name}\n"
            f"  Hand: {hand_str}\n"
            f"  Chips: {view.your_chips}\n"
            f"  Bet this round: {view.your_current_bet}\n"
            f"  Total in pot this hand: {view.your_total_contributed}\n\n"
            f"TABLE STATE\n"
            f"  Phase: {view.phase}\n"
            f"  Pot: {view.pot}\n"
            f"  Highest bet this round: {view.highest_bet}\n"
            f"  Amount you must call: {view.amount_to_call}\n"
            f"  Minimum legal raise (raise to): {view.min_raise_to}\n\n"
            f"OTHER PLAYERS\n{others_str}\n\n"
            f"ACTION HISTORY THIS HAND\n{history_str}\n\n"
            f"LEGAL ACTIONS\n{legal_str}\n\n"
            f"Decide what to do, in character with your personality.\n"
            f"Respond with VALID JSON ONLY, in this exact shape:\n"
            f'{{\n'
            f'  "reasoning": "<one or two sentences explaining your decision>",\n'
            f'  "action": "<fold|check|call|raise>",\n'
            f'  "amount": <integer or null; required and used only when action is "raise" — the total bet level you are raising TO>\n'
            f'}}\n'
            f"Do not include any text before or after the JSON object."
        )

    def _format_discard_prompt(self, view: GameView) -> str:
        """Render the GameView as a discard/draw prompt."""

        # Indexed hand — the agent picks indices, not cards, to avoid
        # ambiguity (no card-name lookup on our side).
        indexed_hand = "\n".join(
            f"  [{i}] {card}" for i, card in enumerate(view.your_hand)
        ) or "  (none)"
        others_str = "\n".join(self._format_other_player(o) for o in view.other_players)
        history_str = self._format_history(view.history) or "  (no actions yet)"

        return (
            f"You are playing 5-card draw poker. The first betting round is over and "
            f"it is now the draw phase. You may discard 0 to 5 cards and receive "
            f"replacements from the deck. Standing pat (zero discards) is allowed.\n\n"
            f"YOUR HAND (indexed 0..4)\n{indexed_hand}\n\n"
            f"YOUR CHIPS: {view.your_chips}\n"
            f"POT: {view.pot}\n\n"
            f"OTHER PLAYERS\n{others_str}\n\n"
            f"ACTION HISTORY THIS HAND\n{history_str}\n\n"
            f"Choose which cards to discard, in character with your personality.\n"
            f"Respond with VALID JSON ONLY, in this exact shape:\n"
            f'{{\n'
            f'  "reasoning": "<one or two sentences explaining your decision>",\n'
            f'  "discards": [<list of integer indices in the range 0..4; empty list to stand pat>]\n'
            f'}}\n'
            f"Do not include any text before or after the JSON object."
        )

    @staticmethod
    def _format_other_player(o) -> str:
        return (
            f"  - {o.name}: chips={o.chips}, bet={o.current_bet}, "
            f"contributed={o.total_contributed}, status={o.status.value}, "
            f"cards_held={o.cards_held}"
        )

    @staticmethod
    def _format_history(history) -> str:
        if not history:
            return ""
        lines = []
        for rec in history:
            posted = f" (+{rec.chips_posted})" if rec.chips_posted else ""
            lines.append(
                f"  [{rec.phase}] {rec.player}: {rec.action}{posted}  "
                f"pot_after={rec.pot_after}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_legal_actions(view: GameView) -> str:
        """List the legal actions for the current state. Spelling these
        out helps the LLM more than relying on it to infer them from
        the rest of the prompt."""
        lines = []
        if view.amount_to_call > 0:
            lines.append("  - fold")
            lines.append(f"  - call (post {view.amount_to_call} chips to match)")
        else:
            lines.append("  - check")
        # Raise legality: must be able to afford the minimum raise.
        owed_for_min = view.min_raise_to - view.your_current_bet
        if view.your_chips >= owed_for_min:
            lines.append(
                f"  - raise (any total from {view.min_raise_to} up to "
                f"{view.your_current_bet + view.your_chips})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing — three layers of defence
    # ------------------------------------------------------------------
    def _parse_action_response(
        self, raw: str, view: GameView,
    ) -> Tuple[Action, str, Optional[str]]:
        """Return (action, reasoning_text, parse_error_or_none).

        The base class never refuses to return an action — even on a
        completely garbled response, it falls back to a safe default
        so the engine can keep running."""

        # Layer 1: extract the first JSON object and parse it.
        obj = _extract_json_object(raw)
        if obj is not None:
            try:
                action = self._action_from_dict(obj, view)
                reasoning = str(obj.get("reasoning", "")).strip()
                return action, reasoning, None
            except (KeyError, ValueError, TypeError) as e:
                # Fell off the well-formed-JSON path — try keyword scan.
                err1 = f"json shape error: {e}"
        else:
            err1 = "no JSON object found"

        # Layer 2: scan the raw text for action keywords.
        action = self._parse_action_keywords(raw, view)
        if action is not None:
            return action, _truncate(raw), f"fallback: keyword scan ({err1})"

        # Layer 3: safe default. Fold if there's something to call,
        # otherwise check.
        default = Action.fold() if view.amount_to_call > 0 else Action.check()
        return default, _truncate(raw), f"fallback: default ({err1})"

    def _parse_discard_response(
        self, raw: str,
    ) -> Tuple[List[int], str, Optional[str]]:
        """Return (discard_indices, reasoning_text, parse_error_or_none).
        On any failure, returns an empty list (stand pat) — a safe and
        legal choice in 5-card draw."""

        obj = _extract_json_object(raw)
        if obj is not None:
            try:
                discards = obj.get("discards", [])
                if not isinstance(discards, list):
                    raise TypeError("'discards' is not a list")
                indices = [int(i) for i in discards if 0 <= int(i) <= 4]
                # De-duplicate, preserving order.
                seen: set = set()
                clean: List[int] = []
                for i in indices:
                    if i not in seen:
                        seen.add(i)
                        clean.append(i)
                reasoning = str(obj.get("reasoning", "")).strip()
                return clean, reasoning, None
            except (KeyError, ValueError, TypeError) as e:
                err = f"json shape error: {e}"
        else:
            err = "no JSON object found"

        # Fallback: try to find a [list, of, ints] anywhere in the text.
        m = re.search(r"\[\s*([0-9,\s]*?)\s*\]", raw)
        if m:
            try:
                parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
                indices = [int(p) for p in parts if 0 <= int(p) <= 4]
                # de-dup
                clean = list(dict.fromkeys(indices))
                return clean, _truncate(raw), f"fallback: regex list ({err})"
            except ValueError:
                pass

        # Total failure: stand pat.
        return [], _truncate(raw), f"fallback: stand pat ({err})"

    # ------------------------------------------------------------------
    # Helpers for translating dict -> Action and keyword -> Action
    # ------------------------------------------------------------------
    @staticmethod
    def _action_from_dict(obj: dict, view: GameView) -> Action:
        """Validate a parsed JSON action object and turn it into an
        engine `Action`. Raises on anything we can't make sense of."""
        kind = str(obj.get("action", "")).strip().lower()

        if kind == "fold":
            return Action.fold()
        if kind == "check":
            # Engine will normalise to call if there's something to
            # match.
            return Action.check()
        if kind == "call":
            return Action.call()
        if kind == "raise":
            amount = obj.get("amount", None)
            if amount is None:
                raise ValueError("raise requires 'amount'")
            amount = int(amount)
            if amount < view.min_raise_to:
                # Don't reject — the engine will bump small raises up
                # to the legal minimum. We just pass through.
                pass
            return Action.raise_to(amount)
        raise ValueError(f"unknown action kind: {kind!r}")

    @staticmethod
    def _parse_action_keywords(raw: str, view: GameView) -> Optional[Action]:
        """Last-ditch keyword scan. Order matters — check for the more
        specific tokens first."""
        text = raw.lower()
        # "raise to N" or "raise N"
        m = re.search(r"raise\s*(?:to\s*)?(\d+)", text)
        if m:
            try:
                return Action.raise_to(int(m.group(1)))
            except ValueError:
                pass
        if "fold" in text:
            return Action.fold()
        if "call" in text:
            return Action.call()
        if "check" in text:
            return Action.check()
        if "raise" in text:
            # Mentioned a raise without an amount — bump to minimum.
            return Action.raise_to(view.min_raise_to)
        return None


# ----------------------------------------------------------------------
# JSON extraction helper — finds the first balanced { ... } block.
# ----------------------------------------------------------------------
def _extract_json_object(raw: str) -> Optional[dict]:
    """Pull the first balanced top-level JSON object out of `raw` and
    return it parsed. Returns None if no parseable object is found.

    We don't trust `json.loads(raw)` outright because LLMs love to
    wrap their answer in prose ('Sure! Here is my decision: { ... }')
    or in ```json fences. A balanced-brace scan handles both.
    """
    if not raw:
        return None
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break       # try next opening brace
                    if isinstance(parsed, dict):
                        return parsed
                    return None
        # Couldn't close this brace; advance to the next one.
        start = raw.find("{", start + 1)
    return None


def _truncate(text: str, n: int = 600) -> str:
    """Cap the stored raw_response for logs — LLMs occasionally produce
    huge outputs that we don't need to keep verbatim."""
    return text if len(text) <= n else text[: n - 3] + "..."


# ----------------------------------------------------------------------
# MockBot — a no-LLM concrete subclass for testing the parser pipeline.
# ----------------------------------------------------------------------
class MockBot(BaseBot):
    """A bot that responds with a hard-coded JSON string. Used by the
    smoke test below and useful for unit-testing the prompt + parser
    pipeline before any real LLM is configured."""

    def __init__(
        self,
        name: str,
        personality: Personality,
        canned_action_response: str,
        canned_discard_response: str,
        model_id: str = "mock",
    ) -> None:
        super().__init__(name=name, personality=personality, model_id=model_id)
        self.canned_action_response = canned_action_response
        self.canned_discard_response = canned_discard_response
        # Ping-pong between the two so a real game loop sees the right
        # response per phase. In a real bot this is just `_generate`
        # talking to the LLM directly.
        self._next = "action"

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        # Crude but works for testing: detect which prompt we got from
        # the user_prompt's content.
        if "discard" in user_prompt.lower() and "draw phase" in user_prompt.lower():
            return self.canned_discard_response
        return self.canned_action_response


# ----------------------------------------------------------------------
# Smoke test — run `python -m bots.base` from the project root.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from engine import (
        ActionRecord, Card, Phase, Player, PublicPlayerInfo, Rank, Suit,
    )

    # Build a synthetic GameView for the bot to chew on.
    hand = (
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.JACK, Suit.DIAMONDS),
        Card(Rank.TEN, Suit.CLUBS),
    )
    view = GameView(
        your_name="Adam",
        your_hand=hand,
        your_chips=190,
        your_current_bet=5,
        your_total_contributed=7,
        pot=23,
        highest_bet=10,
        amount_to_call=5,
        min_raise_to=15,
        other_players=(
            PublicPlayerInfo(
                name="Bob", chips=200, current_bet=10, total_contributed=12,
                status=PlayerStatus.ACTIVE, cards_held=5,
            ),
        ),
        phase=Phase.FIRST_BETTING,
        history=(),
    )

    aggressive = Personality(
        id="aggressive",
        description="Aggressive player who raises and bluffs often.",
        system_prompt="You are an aggressive 5-card draw poker player. You raise often and bluff confidently when you think you can push opponents off their hands.",
    )

    # Test 1: well-formed JSON response is parsed cleanly.
    bot1 = MockBot(
        name="Adam",
        personality=aggressive,
        canned_action_response='Sure! Here is my decision:\n```json\n{"reasoning": "Open-ended straight draw, time to apply pressure.", "action": "raise", "amount": 25}\n```\nDone.',
        canned_discard_response='{"reasoning": "Standing pat — almost a straight.", "discards": []}',
    )
    action = bot1.decide_action(view)
    assert action.type is ActionType.RAISE and action.amount == 25, action
    assert "pressure" in (bot1.last_response.reasoning or "")
    print("Test 1 (clean JSON, prose wrapper)         OK ->", action,
          f"reasoning={bot1.last_response.reasoning!r}")

    # Test 2: malformed JSON triggers keyword fallback.
    bot2 = MockBot(
        name="Adam",
        personality=aggressive,
        canned_action_response="Yeah I think I'll just call this one.",
        canned_discard_response="discard cards [1, 3] please",
    )
    action = bot2.decide_action(view)
    assert action.type is ActionType.CALL, action
    assert bot2.last_response.parse_error and "fallback" in bot2.last_response.parse_error
    print("Test 2 (no JSON, keyword fallback)         OK ->", action,
          f"err={bot2.last_response.parse_error!r}")

    # Test 3: total garbage falls through to safe default (fold,
    # because amount_to_call > 0).
    bot3 = MockBot(
        name="Adam",
        personality=aggressive,
        canned_action_response="completely_unrelated_text_xyzzy",
        canned_discard_response="abc xyz no list here",
    )
    action = bot3.decide_action(view)
    assert action.type is ActionType.FOLD, action
    print("Test 3 (garbage, safe default fold)        OK ->", action,
          f"err={bot3.last_response.parse_error!r}")

    # Test 4: discard parsing — clean JSON
    discards = bot1.decide_discards(view)
    assert list(discards) == []
    print("Test 4 (discards clean JSON, stand pat)    OK ->", list(discards))

    # Test 5: discard parsing — regex fallback
    discards = bot2.decide_discards(view)
    assert sorted(discards) == [1, 3]
    print("Test 5 (discards regex fallback [1,3])     OK ->", list(discards),
          f"err={bot2.last_response.parse_error!r}")

    # Test 6: simulated transport error never crashes — just records it.
    class FailingBot(MockBot):
        def _generate(self, system_prompt, user_prompt):
            raise ConnectionError("Ollama is not running")
    bot4 = FailingBot(
        name="Adam",
        personality=aggressive,
        canned_action_response="",
        canned_discard_response="",
    )
    action = bot4.decide_action(view)
    assert action.type is ActionType.FOLD
    assert "LLM_ERROR" in bot4.last_response.raw_response
    print("Test 6 (LLM transport error, safe default) OK ->", action,
          f"raw={bot4.last_response.raw_response!r}")

    # Test 7: prompt actually contains the structural cues an LLM needs.
    prompt = bot1._format_action_prompt(view)
    assert "5-card draw" in prompt
    assert "Hand: A♠ K♠ Q♥ J♦ T♣" in prompt
    assert "Amount you must call: 5" in prompt
    assert "Minimum legal raise (raise to): 15" in prompt
    assert '"action": "<fold|check|call|raise>"' in prompt
    print("Test 7 (action prompt contains key fields) OK")

    print("\nAll bots/base.py smoke checks passed.")
