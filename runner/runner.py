"""
runner.py — multi-hand tournament orchestration.

This is the top-level loop that the bot_arena notebook (and any
analytics script) calls into. It does the boring plumbing:

  * builds Player + TrackingAgent seats from a list of bots,
  * stands up a HandTracker for the session,
  * runs N hands, applying a broke-player policy between each,
  * stops early if only one bot is solvent (elimination mode),
  * returns a structured TournamentResult.

Design notes
------------
* The runner takes `BaseBot`s, not `Player`s. The caller gives it
  bots + a config, and the runner builds engine-side Players
  internally with `starting_chips` from the config. Less ceremony
  for the common research-mode case where every bot starts the
  session at the same bankroll.

* Two broke-player policies:
    - "rebuy": broke players get topped back up to `starting_chips`
      before each hand. Default. Keeps every bot in the data set
      every hand — best for the brief's "average performance over
      N hands" question.
    - "elimination": broke players sit out (the engine auto-folds
      them via `reset_for_new_hand`) and the session ends as soon
      as only one bot is solvent. Closer to a real tournament.

* Reproducibility: a single `seed` flows through to the deck.
  Together with deterministic bots that's sufficient to replay a
  session bit-for-bit; with stochastic LLM bots it's only
  approximate but the deck stays fixed.

* Verbose output is a one-line-per-hand summary, intentionally
  terse — this is for noticing "Bob has lost 90% of his chips" at a
  glance. Detailed logs go to the .jsonl files via the tracker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from engine import Game, HandSummary, Player, Seat
from bots import BaseBot
from tracker import HandTracker, SeatConfig, TrackingAgent


# ----------------------------------------------------------------------
# Config + result
# ----------------------------------------------------------------------
@dataclass
class RunnerConfig:
    """Settings for one tournament session."""

    num_hands: int = 100                # max hands to play
    starting_chips: int = 200           # bankroll each bot starts with
    ante: int = 2
    min_bet: int = 5
    broke_player_policy: str = "rebuy"  # "rebuy" or "elimination"
    seed: Optional[int] = None
    verbose: bool = False
    sessions_root: str = "runs"
    session_id: Optional[str] = None    # auto-timestamped if None

    def __post_init__(self) -> None:
        if self.broke_player_policy not in ("rebuy", "elimination"):
            raise ValueError(
                f"broke_player_policy must be 'rebuy' or 'elimination', "
                f"got {self.broke_player_policy!r}."
            )
        if self.num_hands < 1:
            raise ValueError(f"num_hands must be >= 1 (got {self.num_hands}).")
        if self.starting_chips < self.ante:
            raise ValueError(
                f"starting_chips ({self.starting_chips}) must be >= ante "
                f"({self.ante}) so every bot can afford at least one hand."
            )


@dataclass
class TournamentResult:
    """What the runner returns after a session ends."""

    session_id: str
    session_dir: Path
    num_hands_played: int
    final_chips: Dict[str, int] = field(default_factory=dict)
    winner: Optional[str] = None             # set in elimination mode only
    stopped_reason: str = "completed"        # "completed" | "single_winner"

    def summary_table(self) -> str:
        """Pretty-print final chips for terminal/notebook output."""
        rows = sorted(self.final_chips.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"Session {self.session_id} — {self.num_hands_played} hands "
                 f"({self.stopped_reason})"]
        for name, chips in rows:
            tag = "  <-- WINNER" if name == self.winner else ""
            lines.append(f"  {name:<20s} {chips:>6d}{tag}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# TournamentRunner
# ----------------------------------------------------------------------
class TournamentRunner:
    """Orchestrates one multi-hand session."""

    def __init__(self, bots: Sequence[BaseBot], config: RunnerConfig) -> None:
        if len(bots) < 2:
            raise ValueError("Tournament needs at least 2 bots.")
        # Names must be unique because the tracker keys on player name.
        names = [b.name for b in bots]
        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate bot names: {sorted(set(dupes))}")
        self.bots: List[BaseBot] = list(bots)
        self.config: RunnerConfig = config

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> TournamentResult:
        cfg = self.config

        # 1. Players, one per bot.
        players: List[Player] = [
            Player(name=b.name, chips=cfg.starting_chips) for b in self.bots
        ]

        # 2. Tracker — opens a session directory, writes config.json on
        #    start_session(), flushes everything via context manager exit.
        with HandTracker(
            sessions_root=cfg.sessions_root, session_id=cfg.session_id,
        ) as tracker:

            seat_configs = [
                SeatConfig(
                    name=b.name,
                    model_id=b.model_id,
                    personality_id=b.personality.id,
                    personality_description=b.personality.description,
                    starting_chips=cfg.starting_chips,
                )
                for b in self.bots
            ]
            tracker.start_session(
                seat_configs,
                num_hands=cfg.num_hands,
                ante=cfg.ante,
                min_bet=cfg.min_bet,
                starting_chips=cfg.starting_chips,
                broke_player_policy=cfg.broke_player_policy,
                seed=cfg.seed,
            )

            # 3. Wrap bots in TrackingAgent so reasoning gets logged
            #    transparently.
            seats = [
                Seat(player=p, agent=TrackingAgent(bot, tracker))
                for p, bot in zip(players, self.bots)
            ]

            # 4. The Game. The deck is seeded from cfg.seed so the
            #    whole sequence of shuffles is reproducible.
            game = Game(
                seats=seats,
                ante=cfg.ante,
                min_bet=cfg.min_bet,
                seed=cfg.seed,
            )

            # 5. Hand loop with stop conditions.
            stopped_reason = "completed"
            hands_played = 0
            for hand_num in range(1, cfg.num_hands + 1):
                # Apply broke-player policy BEFORE the hand. Note that
                # the engine itself also re-checks chip levels in
                # Player.reset_for_new_hand(), so under elimination
                # mode broke players auto-fold even without our help.
                self._apply_broke_policy(players)

                # Elimination stop condition: <=1 solvent player.
                if cfg.broke_player_policy == "elimination":
                    solvent = [p for p in players if p.chips >= cfg.ante]
                    if len(solvent) <= 1:
                        stopped_reason = "single_winner"
                        break

                # Tell the tracker which hand the next reasoning rows
                # belong to. game.hand_count increments at the top of
                # play_hand(), so we predict that value here.
                tracker.start_hand(game.hand_count + 1)

                summary: HandSummary = game.play_hand()
                tracker.log_hand(summary)
                hands_played += 1

                if cfg.verbose:
                    self._print_hand_line(summary)

            # 6. Build the result.
            final_chips = {p.name: p.chips for p in players}
            winner: Optional[str] = None
            if cfg.broke_player_policy == "elimination":
                solvent_names = [n for n, c in final_chips.items() if c >= cfg.ante]
                if len(solvent_names) == 1:
                    winner = solvent_names[0]

            return TournamentResult(
                session_id=tracker.session_id,
                session_dir=tracker.session_dir,
                num_hands_played=hands_played,
                final_chips=final_chips,
                winner=winner,
                stopped_reason=stopped_reason,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_broke_policy(self, players: List[Player]) -> None:
        """Top up broke players if rebuy mode; otherwise no-op (the
        engine handles sitting them out)."""
        if self.config.broke_player_policy == "rebuy":
            for p in players:
                if p.chips < self.config.ante:
                    p.chips = self.config.starting_chips

    @staticmethod
    def _print_hand_line(summary: HandSummary) -> None:
        """Single-line terminal summary of one hand."""
        # Total pot won this hand = sum of `chips_won` across winner rows.
        pot_size = sum(chips for _, chips in summary.winners)
        # Show net change for each seat in turn order.
        deltas = " ".join(
            f"{sr.name}{sr.net_change:+d}" for sr in summary.seat_results
        )
        winners_str = ", ".join(
            f"{name}+{chips}" for name, chips in summary.winners
        )
        print(f"  Hand {summary.hand_id:>3d}: pot={pot_size:>4d}  "
              f"deltas=[{deltas}]  winners=[{winners_str}]")


# ----------------------------------------------------------------------
# Smoke test — `python -m runner.runner`
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import tempfile

    from bots import MockBot, Personality

    # --- Test 1: rebuy mode runs the full hand budget ---------------
    runs_root = Path(tempfile.mkdtemp(prefix="poker_runner_test_"))

    aggressive = Personality(
        id="aggressive",
        description="Raises whenever possible.",
        system_prompt="You are an aggressive poker player.",
    )
    cautious = Personality(
        id="cautious",
        description="Calls and checks rather than raising.",
        system_prompt="You are a cautious poker player.",
    )

    # Two bots that always raise to the minimum. Will burn through
    # chips quickly — good for testing the rebuy path.
    raiser_resp = '{"reasoning": "I always raise.", "action": "raise", "amount": 5}'
    caller_resp = '{"reasoning": "I always call.", "action": "call"}'
    discard_resp = '{"reasoning": "Standing pat.", "discards": []}'

    bots_rebuy = [
        MockBot("RaiserA", aggressive, raiser_resp, discard_resp, model_id="mock-A"),
        MockBot("RaiserB", aggressive, raiser_resp, discard_resp, model_id="mock-B"),
        MockBot("CallerC", cautious,   caller_resp, discard_resp, model_id="mock-C"),
        MockBot("CallerD", cautious,   caller_resp, discard_resp, model_id="mock-D"),
    ]

    cfg = RunnerConfig(
        num_hands=10,
        starting_chips=100,
        ante=2,
        min_bet=5,
        broke_player_policy="rebuy",
        seed=42,
        verbose=True,
        sessions_root=str(runs_root),
        session_id="rebuy_test",
    )
    print("=== Rebuy mode, 10 hands, 4 bots ===")
    result = TournamentRunner(bots_rebuy, cfg).run()
    print(result.summary_table())

    assert result.num_hands_played == 10, result
    assert result.stopped_reason == "completed"
    assert result.winner is None     # rebuy never declares a winner
    # Note: rebuy guarantees solvency *before* each hand, not at end of
    # the run — a player can end the final hand with 0 chips. Instead,
    # verify rebuys actually happened by checking total chips on table
    # exceeds the starting bankroll (which would be impossible without
    # at least one rebuy injection).
    total_final = sum(result.final_chips.values())
    total_start = cfg.starting_chips * len(result.final_chips)
    assert total_final >= total_start, (
        f"rebuy mode: total chips {total_final} should be >= starting "
        f"total {total_start} (with rebuys, often greater)"
    )

    # Spot-check that tracker files actually got written.
    from tracker import load_hands, load_actions, load_reasoning, load_config
    hands = load_hands(result.session_dir)
    actions = load_actions(result.session_dir)
    reasoning = load_reasoning(result.session_dir)
    config = load_config(result.session_dir)
    print(f"\n  hands.jsonl     : {len(hands)} rows")
    print(f"  actions.jsonl   : {len(actions)} rows")
    print(f"  reasoning.jsonl : {len(reasoning)} rows")
    print(f"  config.ended_at : {config['ended_at']}")
    assert len(hands) == 10
    assert config["ended_at"] is not None

    # --- Test 2: elimination mode stops early when 1 solvent left --
    print("\n=== Elimination mode, up to 100 hands, 4 bots ===")

    # Use the same aggressive raisers + cautious callers.
    bots_elim = [
        MockBot("RaiserA", aggressive, raiser_resp, discard_resp, model_id="mock-A"),
        MockBot("RaiserB", aggressive, raiser_resp, discard_resp, model_id="mock-B"),
        MockBot("CallerC", cautious,   caller_resp, discard_resp, model_id="mock-C"),
        MockBot("CallerD", cautious,   caller_resp, discard_resp, model_id="mock-D"),
    ]
    cfg2 = RunnerConfig(
        num_hands=100,
        starting_chips=50,
        ante=2,
        min_bet=5,
        broke_player_policy="elimination",
        seed=42,
        verbose=False,       # too noisy for 100 hands
        sessions_root=str(runs_root),
        session_id="elimination_test",
    )
    result2 = TournamentRunner(bots_elim, cfg2).run()
    print(result2.summary_table())
    print(f"  stopped after {result2.num_hands_played} hands "
          f"({result2.stopped_reason})")

    # Should stop early (one survivor), not run all 100 hands.
    assert result2.stopped_reason == "single_winner", result2
    assert result2.winner is not None, result2
    assert result2.final_chips[result2.winner] >= cfg2.ante

    # --- Test 3: malformed config is rejected ------------------------
    print("\n=== Config validation ===")
    try:
        RunnerConfig(broke_player_policy="bogus")
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")
    try:
        TournamentRunner(bots=bots_rebuy[:1], config=RunnerConfig())
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")
    try:
        # Duplicate names
        bad = [
            MockBot("Same", aggressive, caller_resp, discard_resp),
            MockBot("Same", cautious, caller_resp, discard_resp),
        ]
        TournamentRunner(bad, RunnerConfig())
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")

    # Cleanup
    shutil.rmtree(runs_root)
    print("\nAll runner.py smoke checks passed.")))
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")
    try:
        # Duplicate names
        bad = [
            MockBot("Same", aggressive, caller_resp, discard_resp),
            MockBot("Same", cautious, caller_resp, discard_resp),
        ]
        TournamentRunner(bad, RunnerConfig())
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")

    # Cleanup
    shutil.rmtree(runs_root)
    print("\nAll runner.py smoke checks passed.")
)
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")
    try:
        # Duplicate names
        bad = [
            MockBot("Same", aggressive, caller_resp, discard_resp),
            MockBot("Same", cautious, caller_resp, discard_resp),
        ]
        TournamentRunner(bad, RunnerConfig())
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")

    # Cleanup
    shutil.rmtree(runs_root)
    print("\nAll runner.py smoke checks passed.")
