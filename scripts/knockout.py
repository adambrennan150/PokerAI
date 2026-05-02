"""
knockout.py — single-elimination bracket on round-robin top performers.

Reads the round-robin's `hands.jsonl`, picks the top N performers by
mean net chips per hand, and runs them through a knockout bracket.
Each match is a heads-up (2-player) game over a fixed number of
hands. Winner advances; loser is out. Final survivor is the champion.

Why we run this on top of the round-robin
-----------------------------------------
The round-robin gives the brief's three required answers (best
combo / best LLM / best personality) with statistical power. The
knockout bracket adds a *narrative* layer for the report — "and the
McPokerface champion is X, having beaten Y in the final" reads
better than "X had the highest mean delta" alone.

The bracket data is NOT used to answer the brief's three questions.
That's the round-robin's job. This is the showcase.

Output
------
runs/<SESSION_ID>/
    config.json        (plus extra: bracket structure, source session)
    hands.jsonl        all hands from all matches, hand_id global
    actions.jsonl
    reasoning.jsonl
    bracket.json       round-by-round results, written incrementally

Each hand is tagged with a `match_id` derivable from `hand_id` since
matches play `HANDS_PER_MATCH` hands each in sequence:
    match_id = (hand_id - 1) // HANDS_PER_MATCH
The round_id likewise derives from match_id in seed-pair brackets.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Project root on sys.path.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.base import BaseBot, Personality
from bots.personalities import BY_ID as PERSONALITIES_BY_ID
from config.models import LOCAL_ROSTER
from engine import Game, Player, Seat
from tracker import HandTracker, SeatConfig, TrackingAgent, load_hands, load_config


# ======================================================================
# Configuration
# ======================================================================
ROUND_ROBIN_SESSION = "main_round_robin_v1"   # source of the top-N picks
SESSION_ID = "knockout_v1"

TOP_N = 8                  # bracket size — must be a power of 2
HANDS_PER_MATCH = 200      # hands per heads-up match
SEED = 42                  # reproducibility

# Heads-up matches use bigger stacks than 4-player tables — fewer
# all-ins per hand, longer signal-to-noise window.
ANTE = 2
MIN_BET = 5
STARTING_CHIPS = 1000


# ======================================================================
# BotPlan + materialise — duplicated from round_robin for self-
# containment. If we end up with three+ scripts using these, factor
# them out into bots/factory.py.
# ======================================================================
@dataclass(frozen=True)
class BotPlan:
    name: str
    model_id: str
    personality: Personality

    @property
    def personality_id(self) -> str:
        return self.personality.id


def materialise_bot(plan: BotPlan) -> BaseBot:
    from bots import OllamaBot
    return OllamaBot(
        name=plan.name,
        personality=plan.personality,
        model_id=plan.model_id,
    )


# ======================================================================
# Identifying top performers from a round-robin session
# ======================================================================
def identify_top_performers(
    rr_session_dir: Path, top_n: int,
) -> List[Tuple[BotPlan, float, int]]:
    """Read the round-robin session's hands.jsonl, compute each bot's
    mean net_change per hand, and return the top `top_n` as
    (plan, mean_delta, hands_played) triples.

    The returned plans carry the same name / model_id / personality
    metadata the round-robin used, so the knockout's logs join cleanly
    with the round-robin's for any downstream analysis.
    """
    if not rr_session_dir.exists():
        raise SystemExit(
            f"Round-robin session directory not found: {rr_session_dir}\n"
            f"Run scripts/round_robin.py first (or update ROUND_ROBIN_SESSION)."
        )

    hands = load_hands(rr_session_dir)
    if not hands:
        raise SystemExit(f"No hands found in {rr_session_dir}/hands.jsonl")

    # Aggregate per-bot stats from the seat_results lists.
    totals = defaultdict(lambda: {
        "delta": 0, "hands": 0, "model_id": None, "personality_id": None,
    })
    for h in hands:
        for sr in h["seat_results"]:
            t = totals[sr["name"]]
            t["delta"] += sr["net_change"]
            t["hands"] += 1
            t["model_id"] = sr["model_id"]
            t["personality_id"] = sr["personality_id"]

    # Sort by mean net_change per hand, descending.
    ranked = sorted(
        [
            (name, t["delta"] / t["hands"], t["hands"], t["model_id"],
             t["personality_id"])
            for name, t in totals.items()
            if t["hands"] > 0
        ],
        key=lambda row: row[1],
        reverse=True,
    )

    if len(ranked) < top_n:
        raise SystemExit(
            f"Round-robin only had {len(ranked)} bots; need at least {top_n}."
        )

    # Resolve each top-N row back into a BotPlan via the canonical
    # personality registry so we can construct OllamaBots later.
    out: List[Tuple[BotPlan, float, int]] = []
    for name, mean_delta, hands_played, model_id, personality_id in ranked[:top_n]:
        personality = PERSONALITIES_BY_ID.get(personality_id)
        if personality is None:
            raise SystemExit(
                f"Unknown personality_id {personality_id!r} in round-robin "
                f"session — was the personalities config updated?"
            )
        plan = BotPlan(name=name, model_id=model_id, personality=personality)
        out.append((plan, mean_delta, hands_played))
    return out


# ======================================================================
# Bracket pairings — standard "1 vs N" seeding
# ======================================================================
def make_round1_pairings(
    seeded: Sequence[BotPlan],
) -> List[Tuple[BotPlan, BotPlan]]:
    """Pair top seed with bottom seed, second with second-bottom, etc.
    For 8 seeds [s1..s8]: [(s1, s8), (s2, s7), (s3, s6), (s4, s5)]."""
    n = len(seeded)
    return [(seeded[i], seeded[n - 1 - i]) for i in range(n // 2)]


def pair_winners(winners: Sequence[BotPlan]) -> List[Tuple[BotPlan, BotPlan]]:
    """Subsequent rounds simply pair adjacent winners in order:
    [w1, w2, w3, w4] -> [(w1, w2), (w3, w4)]."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


# ======================================================================
# Schedule + bracket pretty-print
# ======================================================================
def print_top_performers(
    top: Sequence[Tuple[BotPlan, float, int]],
) -> None:
    print(f"\n=== Top {len(top)} performers from round-robin ===")
    print(f"  {'Seed':<5s} {'Name':<28s} {'Mean Δ':>10s} {'Hands':>7s}")
    for i, (plan, mean_d, hands_p) in enumerate(top, 1):
        print(f"  {i:<5d} {plan.name:<28s} {mean_d:>+9.2f}  {hands_p:>6d}")


def print_round1_bracket(
    pairings: Sequence[Tuple[BotPlan, BotPlan]],
) -> None:
    print(f"\n=== Round 1 pairings ({len(pairings)} matches) ===")
    for i, (a, b) in enumerate(pairings, 1):
        print(f"  Match {i}: {a.name}  vs  {b.name}")


# ======================================================================
# Heads-up match — runs N hands, returns winner BotPlan
# ======================================================================
def run_match(
    tracker: HandTracker,
    a: BotPlan,
    b: BotPlan,
    num_hands: int,
    starting_hand_id: int,
    rng_seed: int,
) -> Tuple[BotPlan, dict]:
    """Run a heads-up match. Returns (winner_plan, summary_dict).

    Match dynamics:
      * Both bots start with STARTING_CHIPS.
      * Elimination policy: if one bot runs out, the other wins early.
      * If both still have chips after `num_hands`, whoever has more
        wins. Ties broken by `a` (stable, deterministic).
    """
    # Build the table for this match.
    bots = [materialise_bot(a), materialise_bot(b)]
    players = [Player(name=bot.name, chips=STARTING_CHIPS) for bot in bots]
    seats = [
        Seat(player=p, agent=TrackingAgent(bot, tracker))
        for p, bot in zip(players, bots)
    ]
    game = Game(seats=seats, ante=ANTE, min_bet=MIN_BET, seed=rng_seed)

    # Hand loop.
    hands_played = 0
    early_winner: Optional[BotPlan] = None
    next_hand_id = starting_hand_id
    for _ in range(num_hands):
        # Elimination check before the hand: if either is below the
        # ante, the match is over.
        solvent = [p for p in players if p.chips >= ANTE]
        if len(solvent) <= 1:
            if len(solvent) == 1:
                early_winner = a if solvent[0].name == a.name else b
            break

        tracker.start_hand(next_hand_id)
        summary = game.play_hand()
        summary.hand_id = next_hand_id
        tracker.log_hand(summary)
        next_hand_id += 1
        hands_played += 1

    # Determine winner.
    final_chips = {p.name: p.chips for p in players}
    if early_winner is not None:
        winner = early_winner
        decided_by = "elimination"
    else:
        if final_chips[a.name] >= final_chips[b.name]:
            winner = a
        else:
            winner = b
        decided_by = "chip count"

    summary = {
        "hands_played": hands_played,
        "next_hand_id": next_hand_id,
        "final_chips": final_chips,
        "winner": winner.name,
        "decided_by": decided_by,
    }
    return winner, summary


# ======================================================================
# Top-level orchestration
# ======================================================================
def main() -> int:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    # Sanity: TOP_N must be a power of 2 for a clean bracket without byes.
    if TOP_N & (TOP_N - 1) != 0 or TOP_N < 2:
        raise SystemExit(f"TOP_N ({TOP_N}) must be a power of 2 (>=2).")

    runs_root = PROJECT_ROOT / "runs"
    rr_dir = runs_root / ROUND_ROBIN_SESSION

    # 1. Identify top performers from the round-robin.
    print(f"Loading round-robin session: {rr_dir}")
    top = identify_top_performers(rr_dir, TOP_N)
    print_top_performers(top)
    seeded = [plan for plan, _, _ in top]

    # 2. Round-1 pairings.
    pairings = make_round1_pairings(seeded)
    print_round1_bracket(pairings)

    total_matches = TOP_N - 1                # 8→4→2→1 = 7 for TOP_N=8
    eta_seconds = total_matches * HANDS_PER_MATCH * 5   # rough estimate
    print(f"\n  Total matches : {total_matches}")
    print(f"  Hands/match   : {HANDS_PER_MATCH}")
    print(f"  Total hands   : {total_matches * HANDS_PER_MATCH}")
    print(f"  Estimated time: {eta_seconds / 3600:.1f}h "
          f"(at ~5s/heads-up hand)")

    if dry_run:
        print("\n--dry-run: bracket displayed, not executing.")
        return 0

    # 3. Refuse to overwrite an existing knockout session.
    target_dir = runs_root / SESSION_ID
    if target_dir.exists():
        raise SystemExit(
            f"Session directory already exists: {target_dir}\n"
            f"Either delete it, or change SESSION_ID to a new version."
        )

    # 4. Build SeatConfigs for every bot in the bracket.
    seat_configs = [
        SeatConfig(
            name=p.name, model_id=p.model_id,
            personality_id=p.personality.id,
            personality_description=p.personality.description,
            starting_chips=STARTING_CHIPS,
        )
        for p in seeded
    ]

    # 5. Run the bracket — single shared tracker session.
    print(f"\n=== Starting knockout — output to {target_dir} ===\n")
    t_start = time.time()
    bracket_results: List[dict] = []        # list of round-by-round records

    with HandTracker(sessions_root=str(runs_root), session_id=SESSION_ID) as tracker:
        tracker.start_session(
            seat_configs,
            source_session=ROUND_ROBIN_SESSION,
            top_n=TOP_N,
            hands_per_match=HANDS_PER_MATCH,
            ante=ANTE,
            min_bet=MIN_BET,
            starting_chips=STARTING_CHIPS,
            seed=SEED,
            initial_seeds=[p.name for p in seeded],
        )

        next_hand_id = 1
        round_num = 1
        current_pairings = pairings

        while current_pairings:
            print(f"\n--- Round {round_num} ({len(current_pairings)} matches) ---")
            round_record = {"round": round_num, "matches": []}
            winners: List[BotPlan] = []

            for match_idx, (a, b) in enumerate(current_pairings, 1):
                print(f"\n  Match {round_num}.{match_idx}: "
                      f"{a.name}  vs  {b.name}")
                t_match_start = time.time()
                winner, m_summary = run_match(
                    tracker=tracker,
                    a=a, b=b,
                    num_hands=HANDS_PER_MATCH,
                    starting_hand_id=next_hand_id,
                    rng_seed=SEED + round_num * 1000 + match_idx,
                )
                next_hand_id = m_summary["next_hand_id"]
                t_match = time.time() - t_match_start
                print(f"    Winner: {winner.name}  "
                      f"({m_summary['decided_by']}, "
                      f"{m_summary['hands_played']} hands, {t_match:.0f}s)")
                print(f"    Final stacks: {m_summary['final_chips']}")
                round_record["matches"].append({
                    "match_id": f"{round_num}.{match_idx}",
                    "a": a.name, "b": b.name,
                    "winner": winner.name,
                    "hands_played": m_summary["hands_played"],
                    "decided_by": m_summary["decided_by"],
                    "final_chips": m_summary["final_chips"],
                })
                winners.append(winner)

            bracket_results.append(round_record)
            # Persist incrementally so a crash mid-bracket leaves a
            # readable record of what happened so far.
            (target_dir / "bracket.json").write_text(
                json.dumps({"rounds": bracket_results}, indent=2)
            )

            current_pairings = (
                pair_winners(winners) if len(winners) > 1 else []
            )
            round_num += 1

    # 6. Champion + final summary.
    champion = bracket_results[-1]["matches"][0]["winner"]
    t_total = time.time() - t_start
    print(f"\n=== KNOCKOUT COMPLETE ===")
    print(f"  Wall-clock: {t_total / 3600:.2f}h ({t_total:.0f}s)")
    print(f"  Rounds    : {len(bracket_results)}")
    print(f"  CHAMPION  : {champion}")
    print(f"  Output    : {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
