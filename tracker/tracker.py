"""
tracker.py — append-only JSONL logging for poker sessions.

What gets written
-----------------
Per session, in `runs/<session_id>/`:
    config.json        Once-only metadata: who was at the table, what
                       model + personality each used, the seed,
                       game config, start/end timestamps.
    hands.jsonl        One row per hand: seat outcomes (chip deltas,
                       final hand, folded?) and pot winners. Primary
                       analytics input.
    actions.jsonl      One row per action: phase, player, action type,
                       chips posted, pot after. Finer-grained — for
                       playing-style analysis.
    reasoning.jsonl    One row per LLM call: prompt + raw response +
                       parsed reasoning + parse_error. Heavy — loaded
                       only when you want to read what the LLMs are
                       *thinking*.

Why split actions and reasoning into separate files
---------------------------------------------------
Reasoning logs explode in size — full LLM responses, per action, per
hand, per bot. Keeping them out of the main analytics path means the
default "load all sessions and compute win rates" query stays fast,
and you only pay the cost when you opt in to inspecting reasoning.

Why JSONL not SQLite or CSV
---------------------------
* Reasoning is variable-length text that's awkward in CSV.
* Append-only writes mean a crash mid-session can't corrupt prior
  hands — every completed line is safe on disk.
* JSONL loads trivially into a pandas DataFrame at analysis time
  (`pd.read_json(path, lines=True)`), which gives the SQL-like power
  the analytics notebook needs.
* Swapping the backend to SQLite later is a one-file change because
  no other code in the project touches these files directly.

Three required aggregations from the brief
------------------------------------------
* Best bot      → groupby(['model_id', 'personality_id'])  on net_change
* Best LLM      → groupby('model_id')                      on net_change
* Best persona  → groupby('personality_id')                on net_change

This module only handles the writing and raw reading. The analytics
notebook will run the groupbys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

# We import from engine and bots — both upstream layers. The tracker
# never imports from the UI layer.
from engine import (
    Action, ActionRecord, GameView, HandSummary, Phase, SeatResult,
)
from bots import BaseBot, BotResponse


# ----------------------------------------------------------------------
# Session config — what's in config.json
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SeatConfig:
    """Static metadata about a seat for a session. Frozen because once
    a session has started, this can't change."""

    name: str
    model_id: str
    personality_id: str
    personality_description: str = ""
    starting_chips: int = 0


# ----------------------------------------------------------------------
# HandTracker — the writer
# ----------------------------------------------------------------------
class HandTracker:
    """Writes append-only logs for one session.

    Use as a context manager to guarantee files are flushed and the
    session metadata is finalised even on exception:

        with HandTracker(sessions_root="runs") as t:
            t.start_session(seats, ante=2, min_bet=5, seed=42)
            for h in range(100):
                t.start_hand(game.hand_count + 1)
                summary = game.play_hand()
                t.log_hand(summary, seat_meta)
    """

    HANDS_FILE = "hands.jsonl"
    ACTIONS_FILE = "actions.jsonl"
    REASONING_FILE = "reasoning.jsonl"
    CONFIG_FILE = "config.json"

    def __init__(
        self,
        sessions_root: str | os.PathLike = "runs",
        session_id: Optional[str] = None,
    ) -> None:
        # If no session_id given, mint one from the current timestamp.
        # Including microseconds so back-to-back sessions never collide.
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_id: str = session_id
        self.session_dir: Path = Path(sessions_root) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # File handles are opened lazily on first write so creating a
        # tracker doesn't already touch disk.
        self._hands_f: Optional[TextIO] = None
        self._actions_f: Optional[TextIO] = None
        self._reasoning_f: Optional[TextIO] = None

        # Set by start_hand(), read by log_reasoning().
        self._current_hand_id: Optional[int] = None
        self._config_path: Path = self.session_dir / self.CONFIG_FILE
        self._seat_meta: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def start_session(
        self,
        seats: Sequence[SeatConfig],
        **game_config: Any,
    ) -> None:
        """Write the session config. Call once per session."""
        config: Dict[str, Any] = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ended_at": None,
            "seats": [asdict(s) for s in seats],
            "game_config": dict(game_config),
        }
        self._config_path.write_text(json.dumps(config, indent=2))

        # Cache seat metadata so per-hand and per-reasoning logs can
        # tag rows with model_id / personality_id without the caller
        # having to repeat themselves.
        self._seat_meta = {
            s.name: {"model_id": s.model_id, "personality_id": s.personality_id}
            for s in seats
        }

    def end_session(self) -> None:
        """Stamp the config with an `ended_at` timestamp."""
        if not self._config_path.exists():
            return
        try:
            cfg = json.loads(self._config_path.read_text())
        except json.JSONDecodeError:
            return
        cfg["ended_at"] = datetime.now().isoformat(timespec="seconds")
        self._config_path.write_text(json.dumps(cfg, indent=2))

    # ------------------------------------------------------------------
    # Per-hand lifecycle
    # ------------------------------------------------------------------
    def start_hand(self, hand_id: int) -> None:
        """Mark which hand subsequent reasoning rows belong to.
        TrackingAgent reads `_current_hand_id` to tag its writes."""
        self._current_hand_id = hand_id

    def log_hand(
        self,
        summary: HandSummary,
        seat_meta: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """Append one hand-level row to hands.jsonl, plus one row per
        action to actions.jsonl. `seat_meta` overrides the session-
        scoped seat metadata for one hand if needed (e.g. mid-session
        swap)."""
        meta = seat_meta or self._seat_meta

        hand_row = self._hand_to_row(summary, meta)
        self._write_jsonl(self._hands_handle(), hand_row)

        for action_record in summary.actions:
            self._write_jsonl(
                self._actions_handle(),
                self._action_to_row(summary.hand_id, action_record, meta),
            )

    def log_reasoning(
        self,
        *,
        phase: Phase,
        player: str,
        decision_type: str,         # "action" or "discard"
        response: BotResponse,
        model_id: str,
        personality_id: str,
        hand_id: Optional[int] = None,
    ) -> None:
        """Append one row to reasoning.jsonl. Called automatically by
        TrackingAgent — game/runner code shouldn't normally need to
        call this directly."""
        if hand_id is None:
            hand_id = self._current_hand_id
        row = {
            "session_id": self.session_id,
            "hand_id": hand_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "phase": str(phase),
            "player": player,
            "model_id": model_id,
            "personality_id": personality_id,
            "decision_type": decision_type,
            "prompt": response.prompt,
            "raw_response": response.raw_response,
            "reasoning": response.reasoning,
            "parse_error": response.parse_error,
            # Record what the bot actually decided so reasoning rows
            # are self-contained.
            "decided_action": _action_to_str(response.action) if response.action else None,
            "decided_discards": list(response.discards) if response.discards else None,
        }
        self._write_jsonl(self._reasoning_handle(), row)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close any open file handles. Idempotent."""
        for attr in ("_hands_f", "_actions_f", "_reasoning_f"):
            f = getattr(self, attr, None)
            if f is not None and not f.closed:
                f.close()
            setattr(self, attr, None)

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_session()
        self.close()

    # ------------------------------------------------------------------
    # Internal: file handle accessors (lazy-open + line-buffered).
    # We open with `buffering=1` so each newline flushes — crash-safe
    # without an explicit flush() after every write.
    # ------------------------------------------------------------------
    def _hands_handle(self) -> TextIO:
        if self._hands_f is None or self._hands_f.closed:
            self._hands_f = open(self.session_dir / self.HANDS_FILE, "a", buffering=1)
        return self._hands_f

    def _actions_handle(self) -> TextIO:
        if self._actions_f is None or self._actions_f.closed:
            self._actions_f = open(self.session_dir / self.ACTIONS_FILE, "a", buffering=1)
        return self._actions_f

    def _reasoning_handle(self) -> TextIO:
        if self._reasoning_f is None or self._reasoning_f.closed:
            self._reasoning_f = open(self.session_dir / self.REASONING_FILE, "a", buffering=1)
        return self._reasoning_f

    @staticmethod
    def _write_jsonl(handle: TextIO, row: Dict[str, Any]) -> None:
        handle.write(json.dumps(row, default=_json_default) + "\n")

    # ------------------------------------------------------------------
    # Internal: turn engine objects into JSON-friendly dicts.
    # ------------------------------------------------------------------
    def _hand_to_row(
        self, summary: HandSummary, meta: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "hand_id": summary.hand_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seat_results": [
                self._seat_result_to_dict(sr, meta) for sr in summary.seat_results
            ],
            "winners": [
                {"name": name, "chips_won": chips} for name, chips in summary.winners
            ],
        }

    @staticmethod
    def _seat_result_to_dict(
        sr: SeatResult, meta: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        m = meta.get(sr.name, {})
        return {
            "name": sr.name,
            "model_id": m.get("model_id"),
            "personality_id": m.get("personality_id"),
            "starting_chips": sr.starting_chips,
            "ending_chips": sr.ending_chips,
            "net_change": sr.net_change,
            "final_hand": [str(c) for c in sr.final_hand] if sr.final_hand else None,
            "final_hand_category": (
                sr.final_evaluation.category.label if sr.final_evaluation else None
            ),
            "folded": sr.folded,
        }

    def _action_to_row(
        self, hand_id: int, rec: ActionRecord,
        meta: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        m = meta.get(rec.player, {})
        return {
            "session_id": self.session_id,
            "hand_id": hand_id,
            "phase": str(rec.phase),
            "player": rec.player,
            "model_id": m.get("model_id"),
            "personality_id": m.get("personality_id"),
            "action_type": rec.action.type.value,
            "amount": rec.action.amount,
            "chips_posted": rec.chips_posted,
            "pot_after": rec.pot_after,
        }


# ----------------------------------------------------------------------
# TrackingAgent — wraps a BaseBot to auto-log reasoning per call.
# ----------------------------------------------------------------------
class TrackingAgent:
    """Adapter that satisfies the engine's `PlayerAgent` protocol while
    auto-logging reasoning to a HandTracker.

    Use this when wiring up a bot-vs-bot session — the runner wraps
    each `BaseBot` in a `TrackingAgent` and seats those, so the engine
    sees normal agents but the tracker captures every LLM round-trip.
    """

    def __init__(self, bot: BaseBot, tracker: HandTracker) -> None:
        self.bot = bot
        self.tracker = tracker

    # Convenience pass-throughs so seating code can read these without
    # reaching into self.bot. Not required by the protocol but nice.
    @property
    def name(self) -> str:
        return self.bot.name

    @property
    def model_id(self) -> str:
        return self.bot.model_id

    @property
    def personality_id(self) -> str:
        return self.bot.personality.id

    # ------------------------------------------------------------------
    # PlayerAgent protocol
    # ------------------------------------------------------------------
    def decide_action(self, view: GameView) -> Action:
        action = self.bot.decide_action(view)
        if self.bot.last_response is not None:
            self.tracker.log_reasoning(
                phase=view.phase,
                player=view.your_name,
                decision_type="action",
                response=self.bot.last_response,
                model_id=self.bot.model_id,
                personality_id=self.bot.personality.id,
            )
        return action

    def decide_discards(self, view: GameView) -> Sequence[int]:
        discards = self.bot.decide_discards(view)
        if self.bot.last_response is not None:
            self.tracker.log_reasoning(
                phase=view.phase,
                player=view.your_name,
                decision_type="discard",
                response=self.bot.last_response,
                model_id=self.bot.model_id,
                personality_id=self.bot.personality.id,
            )
        return discards


# ----------------------------------------------------------------------
# Loader functions — used by the analytics notebook.
# ----------------------------------------------------------------------
def load_config(session_dir: str | os.PathLike) -> Dict[str, Any]:
    """Read a session's config.json."""
    path = Path(session_dir) / HandTracker.CONFIG_FILE
    return json.loads(path.read_text())


def load_hands(session_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    """Read hands.jsonl as a list of dicts. Empty list if not present."""
    return _read_jsonl(Path(session_dir) / HandTracker.HANDS_FILE)


def load_actions(session_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    return _read_jsonl(Path(session_dir) / HandTracker.ACTIONS_FILE)


def load_reasoning(session_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    return _read_jsonl(Path(session_dir) / HandTracker.REASONING_FILE)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------------------
# JSON serialisation helpers
# ----------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """Fallback for `json.dumps(default=...)` — handles engine enums
    and any object that exposes __str__."""
    # Lazy imports to avoid circular dependencies.
    from enum import Enum
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def _action_to_str(action: Action) -> str:
    return f"{action.type.value}:{action.amount}" if action.amount else action.type.value


# ----------------------------------------------------------------------
# Smoke test — run `python -m tracker.tracker` from the project root.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import tempfile

    from engine import Game, Player, Seat
    from bots import MockBot, Personality

    # Clean temp dir so reruns don't accumulate.
    runs_root = Path(tempfile.mkdtemp(prefix="poker_tracker_test_"))

    aggressive = Personality(
        id="aggressive",
        description="Plays loose and aggressive.",
        system_prompt="You are an aggressive poker player.",
    )
    cautious = Personality(
        id="cautious",
        description="Folds easily, calls rather than raises.",
        system_prompt="You are a cautious poker player.",
    )

    # Two MockBots with hard-coded responses. They'll always check/call
    # so a hand will run cleanly through every phase.
    alice_bot = MockBot(
        name="Alice", personality=aggressive, model_id="mock-llama",
        canned_action_response='{"reasoning": "I have a strong feeling.", "action": "call"}',
        canned_discard_response='{"reasoning": "Standing pat.", "discards": []}',
    )
    bob_bot = MockBot(
        name="Bob", personality=cautious, model_id="mock-mistral",
        canned_action_response='{"reasoning": "Better to be safe.", "action": "check"}',
        canned_discard_response='{"reasoning": "Discarding the low cards.", "discards": [3, 4]}',
    )

    # Wire up the tracker and wrap bots in TrackingAgent.
    with HandTracker(sessions_root=runs_root, session_id="test_session") as tracker:
        seat_configs = [
            SeatConfig(name="Alice", model_id="mock-llama",
                       personality_id="aggressive",
                       personality_description=aggressive.description,
                       starting_chips=200),
            SeatConfig(name="Bob", model_id="mock-mistral",
                       personality_id="cautious",
                       personality_description=cautious.description,
                       starting_chips=200),
        ]
        tracker.start_session(seat_configs, ante=2, min_bet=5, seed=7)

        seats = [
            Seat(Player("Alice", chips=200), TrackingAgent(alice_bot, tracker)),
            Seat(Player("Bob", chips=200), TrackingAgent(bob_bot, tracker)),
        ]
        game = Game(seats, ante=2, min_bet=5, seed=7)

        for _ in range(2):
            tracker.start_hand(game.hand_count + 1)
            summary = game.play_hand()
            tracker.log_hand(summary)
            print(f"Hand {summary.hand_id}: winners={summary.winners}")

    # Verify files exist and contain the expected counts.
    session_dir = runs_root / "test_session"
    hands = load_hands(session_dir)
    actions = load_actions(session_dir)
    reasoning = load_reasoning(session_dir)
    config = load_config(session_dir)

    print(f"\nSession dir: {session_dir}")
    print(f"  config.json   ended_at={config['ended_at']}")
    print(f"                seats={len(config['seats'])} ({[s['name'] for s in config['seats']]})")
    print(f"  hands.jsonl   {len(hands)} rows")
    print(f"  actions.jsonl {len(actions)} rows")
    print(f"  reasoning.jsonl {len(reasoning)} rows")

    # Spot-checks
    assert len(hands) == 2
    assert all(h["seat_results"][0]["model_id"] == "mock-llama" for h in hands)
    assert all(h["seat_results"][1]["personality_id"] == "cautious" for h in hands)
    assert len(actions) > 0
    assert len(reasoning) > 0
    # Reasoning rows should be tagged with the right hand_id
    assert {r["hand_id"] for r in reasoning} == {1, 2}
    # Reasoning rows for actions and discards
    assert {r["decision_type"] for r in reasoning} == {"action", "discard"}

    # Show a sample reasoning row
    print("\nSample reasoning row:")
    sample = reasoning[0]
    for k in ("hand_id", "phase", "player", "model_id", "personality_id",
              "decision_type", "reasoning", "decided_action"):
        print(f"  {k}: {sample[k]}")

    # Show a sample hand row
    print("\nSample hand row (seat results):")
    for sr in hands[0]["seat_results"]:
        print(f"  {sr}")

    # Cleanup
    shutil.rmtree(runs_root)
    print("\nAll tracker.py smoke checks passed.")
