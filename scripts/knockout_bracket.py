"""
knockout_bracket.py — Phase 5 single-elimination tournament.

Takes the top N (default 8) (model × personality) combos from a
finished round-robin and runs them through a single-elimination
bracket of heads-up matches. The eventual winner is the project's
"champion" — the narrative-friendly counterpart to the round-
robin's statistical rankings.

Design choices
--------------
* Seeded bracket. Round 1 pairs 1v8, 2v7, 3v6, 4v5 — the standard
  format. This means the top round-robin performer plays the
  weakest qualifier first, and top seeds only meet in later rounds.
* Heads-up matches with rebuy. Each match is a fixed N hands (default
  200). Both bots keep playing even if one would otherwise bust, so
  every match contributes the same volume of data. Winner = higher
  cumulative net chip change over the match (NOT ending chips, since
  rebuy artificially inflates those).
* All matches under one HandTracker session, with monotonically-
  increasing global hand IDs. A separate `bracket.json` file written
  to the session dir maps each hand-id range to its (round, match,
  p1, p2, winner) so analytics can reconstruct the bracket.
* Per-match seed is `SEED + round_num*1000 + match_num` so individual
  matches are reproducible without any cross-match leakage.

Selecting qualifiers
--------------------
* Reads the source round-robin session's hands.jsonl.
* Excludes "broken" models (anyone with >=50% parse-error rate in
  reasoning.jsonl) — their decisions are fallbacks, not real play.
* Ranks remaining (model, personality) combos by mean chip change
  per hand. Top N go to the bracket.

Usage
-----
    python scripts/knockout_bracket.py                    # auto-pick latest main_*
    python scripts/knockout_bracket.py main_round_robin_v2
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from itertools import count
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.base import Personality
from bots.personalities import BY_ID as PERSONALITY_BY_ID
from config.models import BY_ID as MODEL_BY_ID
from engine import Game, Player, Seat
from tracker import HandTracker, SeatConfig, TrackingAgent


# ======================================================================
# Configuration
# ======================================================================
NUM_QUALIFIERS = 8
HANDS_PER_MATCH = 200
SESSION_ID = "knockout_bracket_v1"
SEED = 42

ANTE = 2
MIN_BET = 5
STARTING_CHIPS = 200

# Models with this much parse-error rate in the source round-robin
# are excluded from qualifier selection (their wins were fallbacks).
BROKEN_THRESHOLD_PCT = 50.0


# ======================================================================
# Source-session loading + qualifier selection
# ======================================================================
def _safe_load_jsonl(path: Path):
    """Tolerant JSONL reader — skips malformed lines."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def resolve_source_session(arg: Optional[str]) -> Path:
    """Find the round-robin session to use as the source for qualifiers."""
    runs = PROJECT_ROOT / "runs"
    if arg:
        candidate = runs / arg
        if not candidate.exists():
            raise SystemExit(f"Source session not found: {candidate}")
        return candidate
    candidates = sorted(p for p in runs.iterdir()
                        if p.is_dir() and p.name.startswith("main_round_robin"))
    if not candidates:
        raise SystemExit(
            "No main_round_robin_* sessions in runs/. Pass a session id "
            "explicitly: python scripts/knockout_bracket.py <id>"
        )
    return candidates[-1]


def identify_broken_models(reasoning) -> set:
    """Models with >= BROKEN_THRESHOLD_PCT parse-error rate."""
    by_model: dict = defaultdict(list)
    for r in reasoning:
        mid = r.get("model_id", "")
        if mid.startswith("mock"):
            continue
        by_model[mid].append(r)
    broken = set()
    for model, rs in by_model.items():
        errs = sum(1 for r in rs if r.get("parse_error"))
        if rs and (100 * errs / len(rs)) >= BROKEN_THRESHOLD_PCT:
            broken.add(model)
    return broken


def select_qualifiers(source_dir: Path, n: int) -> List[dict]:
    """Pick the top N (model, personality) combos by mean chip change.

    Returns a list of dicts with keys: name, model_id, personality_id,
    rr_mean_delta, rr_hands. Seed is implicit in the list order
    (rank 1 = seed 1).
    """
    hands = _safe_load_jsonl(source_dir / "hands.jsonl")
    reasoning = _safe_load_jsonl(source_dir / "reasoning.jsonl")
    broken = identify_broken_models(reasoning)
    if broken:
        print(f"  Excluding broken models from qualifier pool: "
              f"{sorted(broken)}")

    # Aggregate per (model, personality, name) so we have the full
    # identity to reconstruct the bot.
    agg: dict = {}
    for h in hands:
        for sr in h.get("seat_results", []):
            if sr["model_id"] in broken:
                continue
            key = (sr["model_id"], sr["personality_id"], sr["name"])
            entry = agg.setdefault(key, {"sum": 0, "n": 0})
            entry["sum"] += sr["net_change"]
            entry["n"] += 1

    rows = []
    for (model_id, pers_id, name), v in agg.items():
        if v["n"] == 0:
            continue
        rows.append({
            "name": name,
            "model_id": model_id,
            "personality_id": pers_id,
            "rr_mean_delta": v["sum"] / v["n"],
            "rr_hands": v["n"],
        })
    rows.sort(key=lambda r: r["rr_mean_delta"], reverse=True)

    if len(rows) < n:
        raise SystemExit(
            f"Only {len(rows)} eligible combos after filtering — "
            f"can't fill {n} qualifier slots."
        )
    return rows[:n]


# ======================================================================
# Bracket construction
# ======================================================================
def seed_pairings(qualifiers: List[dict]) -> List[Tuple[dict, dict]]:
    """Standard tournament seeding: 1v8, 2v7, 3v6, 4v5 for an 8-bracket."""
    n = len(qualifiers)
    if n & (n - 1) != 0:
        raise ValueError(f"Bracket size must be a power of 2 (got {n}).")
    return [(qualifiers[i], qualifiers[n - 1 - i]) for i in range(n // 2)]


# ======================================================================
# Bot materialisation (mirrors round_robin.materialise_bot)
# ======================================================================
def materialise_bot(plan: dict):
    """Construct an OllamaBot from a plan dict."""
    from bots import OllamaBot   # lazy import — only needed at run time
    spec = MODEL_BY_ID[plan["model_id"]]
    pers = PERSONALITY_BY_ID[plan["personality_id"]]
    return OllamaBot(
        name=plan["name"],
        personality=pers,
        model_id=plan["model_id"],
        num_predict=spec.num_predict,
        system_prefix=spec.system_prefix,
        think=spec.think,
    )


# ======================================================================
# Match runner
# ======================================================================
def run_match(
    p1: dict, p2: dict,
    round_num: int, match_num: int,
    hands_per_match: int,
    tracker: HandTracker,
    gid_counter,
) -> dict:
    """Run a heads-up match between p1 and p2. Returns a result dict
    with cumulative net chip changes and the winner.

    Both bots play `hands_per_match` hands with rebuy enabled, so a
    consistent volume of data comes out of every match. Winner is the
    bot with higher cumulative net chip change (NOT ending chips —
    rebuy inflates those)."""

    bots = [materialise_bot(p1), materialise_bot(p2)]
    players = [Player(name=b.name, chips=STARTING_CHIPS) for b in bots]
    seats = [Seat(player=p, agent=TrackingAgent(b, tracker))
             for p, b in zip(players, bots)]
    # Per-match seed — reproducible without cross-match leakage.
    game = Game(seats=seats, ante=ANTE, min_bet=MIN_BET,
                seed=SEED + round_num * 1000 + match_num)

    print(f"\n--- Round {round_num} match {match_num}: "
          f"{p1['name']} vs {p2['name']} ---")
    t_match = time.time()

    hand_id_start = None
    cumulative = {p1["name"]: 0, p2["name"]: 0}
    for _ in range(hands_per_match):
        # Rebuy if needed
        for p in players:
            if p.chips < ANTE:
                p.chips = STARTING_CHIPS
        gid = next(gid_counter)
        if hand_id_start is None:
            hand_id_start = gid
        tracker.start_hand(gid)
        summary = game.play_hand()
        summary.hand_id = gid
        tracker.log_hand(summary)
        # Track cumulative net change per player from the summary.
        for sr in summary.seat_results:
            cumulative[sr.name] += sr.net_change
    hand_id_end = gid

    dt = time.time() - t_match
    print(f"    Done in {dt:.0f}s ({dt / hands_per_match:.1f}s/hand)")
    print(f"    Cumulative net: {p1['name']}={cumulative[p1['name']]:+d}, "
          f"{p2['name']}={cumulative[p2['name']]:+d}")

    # Winner = higher cumulative net.
    if cumulative[p1["name"]] > cumulative[p2["name"]]:
        winner = p1
    elif cumulative[p2["name"]] > cumulative[p1["name"]]:
        winner = p2
    else:
        # Tie — extremely unlikely over 200 hands. Fall back to p1
        # (top-seed wins ties; standard tournament practice).
        winner = p1
        print(f"    TIE on cumulative net — top seed advances.")
    print(f"    Winner: {winner['name']}")

    return {
        "round": round_num,
        "match": match_num,
        "p1": p1,
        "p2": p2,
        "p1_net": cumulative[p1["name"]],
        "p2_net": cumulative[p2["name"]],
        "hand_id_start": hand_id_start,
        "hand_id_end": hand_id_end,
        "winner": winner,
    }


# ======================================================================
# Bracket diagram printer
# ======================================================================
def print_bracket(matches: List[dict], champion: dict) -> None:
    """Pretty-print the bracket result."""
    by_round: dict = defaultdict(list)
    for m in matches:
        by_round[m["round"]].append(m)
    print(f"\n{'='*60}")
    print(f"=== Bracket result ===")
    print(f"{'='*60}")
    for round_num in sorted(by_round.keys()):
        round_matches = by_round[round_num]
        label = (
            "Quarterfinals" if len(round_matches) == 4 else
            "Semifinals"    if len(round_matches) == 2 else
            "Final"         if len(round_matches) == 1 else
            f"Round {round_num}"
        )
        print(f"\n  {label}:")
        for m in round_matches:
            p1, p2 = m["p1"]["name"], m["p2"]["name"]
            n1, n2 = m["p1_net"], m["p2_net"]
            w = m["winner"]["name"]
            print(f"    {p1:<32s} ({n1:+5d})  vs  {p2:<32s} ({n2:+5d})  →  {w}")
    print(f"\n  CHAMPION: {champion['name']}")
    print(f"{'='*60}")


# ======================================================================
# Entry point
# ======================================================================
def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    arg = argv[0] if argv and not argv[0].startswith("--") else None
    source_dir = resolve_source_session(arg)
    print(f"Source round-robin: {source_dir.name}")
    print(f"Bracket session id: {SESSION_ID}")
    print(f"Qualifiers        : top {NUM_QUALIFIERS}")
    print(f"Hands per match   : {HANDS_PER_MATCH}")
    print()

    # 1. Pick qualifiers
    qualifiers = select_qualifiers(source_dir, NUM_QUALIFIERS)
    print(f"=== Qualifiers (seeded by round-robin mean chip change) ===")
    for i, q in enumerate(qualifiers, 1):
        print(f"  Seed {i}: {q['name']:<32s}  rr_mean={q['rr_mean_delta']:+7.2f}  "
              f"rr_hands={q['rr_hands']}")
    print()

    # 2. Refuse to overwrite existing session
    runs_root = PROJECT_ROOT / "runs"
    target_dir = runs_root / SESSION_ID
    if target_dir.exists():
        raise SystemExit(
            f"Session directory already exists: {target_dir}\n"
            f"Either delete it or change SESSION_ID to a new version."
        )

    # 3. Build round-1 pairings
    current = qualifiers
    all_matches: List[dict] = []
    expected_total = sum(NUM_QUALIFIERS // (2 ** r)
                        for r in range(1, int(math.log2(NUM_QUALIFIERS)) + 1))
    expected_hands = expected_total * HANDS_PER_MATCH
    print(f"=== Bracket plan ===")
    print(f"  {expected_total} matches across "
          f"{int(math.log2(NUM_QUALIFIERS))} rounds")
    print(f"  ~{expected_hands} total hands ≈ "
          f"{expected_hands * 8 / 3600:.1f}h at 8s/hand")
    print()

    # 4. Open tracker once for the whole bracket
    seat_configs = [
        SeatConfig(
            name=q["name"],
            model_id=q["model_id"],
            personality_id=q["personality_id"],
            personality_description=PERSONALITY_BY_ID[q["personality_id"]].description,
            starting_chips=STARTING_CHIPS,
        )
        for q in qualifiers
    ]
    t_start = time.time()
    with HandTracker(sessions_root=str(runs_root), session_id=SESSION_ID) as tracker:
        tracker.start_session(
            seat_configs,
            ante=ANTE,
            min_bet=MIN_BET,
            starting_chips=STARTING_CHIPS,
            broke_player_policy="rebuy",
            seed=SEED,
            source_session=source_dir.name,
            num_qualifiers=NUM_QUALIFIERS,
            hands_per_match=HANDS_PER_MATCH,
            bracket_kind="single_elimination",
        )

        gid_counter = count(1)
        round_num = 1
        while len(current) > 1:
            print(f"\n{'#'*60}")
            print(f"# Round {round_num}: {len(current)} → {len(current)//2}")
            print(f"{'#'*60}")
            pairings = seed_pairings(current)
            winners: List[dict] = []
            for match_num, (p1, p2) in enumerate(pairings, 1):
                result = run_match(
                    p1, p2,
                    round_num=round_num,
                    match_num=match_num,
                    hands_per_match=HANDS_PER_MATCH,
                    tracker=tracker,
                    gid_counter=gid_counter,
                )
                all_matches.append(result)
                winners.append(result["winner"])
            current = winners
            round_num += 1

        champion = current[0]

    # 5. Save bracket metadata for the analytics
    bracket_path = target_dir / "bracket.json"
    bracket_data = {
        "session_id": SESSION_ID,
        "source_session": source_dir.name,
        "num_qualifiers": NUM_QUALIFIERS,
        "hands_per_match": HANDS_PER_MATCH,
        "qualifiers": [
            {**q, "seed": i + 1} for i, q in enumerate(qualifiers)
        ],
        "matches": [
            {
                "round": m["round"],
                "match": m["match"],
                "p1": m["p1"]["name"],
                "p2": m["p2"]["name"],
                "p1_net": m["p1_net"],
                "p2_net": m["p2_net"],
                "hand_id_start": m["hand_id_start"],
                "hand_id_end": m["hand_id_end"],
                "winner": m["winner"]["name"],
            }
            for m in all_matches
        ],
        "champion": champion["name"],
    }
    bracket_path.write_text(json.dumps(bracket_data, indent=2))

    dt = time.time() - t_start
    print(f"\nTotal wall-clock: {dt / 60:.1f} min ({dt:.0f}s)")
    print_bracket(all_matches, champion)
    print(f"\nBracket metadata: {bracket_path}")
    print(f"Commit: git add -f runs/{SESSION_ID}/ && "
          f"git commit -m 'Knockout bracket results'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
