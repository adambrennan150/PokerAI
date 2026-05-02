"""
validation_full_roster.py — small-scale dress rehearsal of v2 round-robin.

Runs a tiny multi-table tournament with the full 35-bot roster (all 7
models × all 5 personalities) before committing to the 17-hour
overnight run. The point is to verify three things at scale that the
single-table validation can't:

1. Every model still works when sharing tables with every other model
   (no cross-model interaction bugs).
2. Per-model parse-error rates remain near zero in the round-robin
   harness (not just the heads-up validation harness).
3. Per-table timing is reasonable — confirms the v2 round-robin will
   complete in a manageable wall-clock.

Setup
-----
* 4 tables × 5 hands = 20 hands total.
* Tables are sampled randomly from the full 35-combo roster, then
  sorted by model membership for swap efficiency.
* Output goes to runs/validation_full_roster_v1/.

Expected runtime: ~30-45 minutes on a 32GB GPU. If it takes longer,
something's wrong (hung model, swap thrash) and we'd see it before
committing to an overnight run.

Pass criteria
-------------
For every one of the 7 models that appears at any table:
* Parse-error rate < 20%
* Mean response length > 50 chars (rules out empty responses)

If any fails, abort before launching v2.

Usage
-----
    python -u scripts/validation_full_roster.py 2>&1 \\
        | tee runs/validation_full_roster_v1.log
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from itertools import count
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.personalities import ALL as ALL_PERSONALITIES
from config.models import LOCAL_ROSTER, BY_ID
from engine import Game, Player, Seat
from tracker import HandTracker, SeatConfig, TrackingAgent, load_reasoning


# ----------------------------------------------------------------------
# Configuration — small for a quick dress rehearsal
# ----------------------------------------------------------------------
NUM_TABLES = 4
HANDS_PER_TABLE = 5
SESSION_ID = "validation_full_roster_v1"
SEED = 42
ANTE = 2
MIN_BET = 5
STARTING_CHIPS = 200


def main() -> int:
    # ------------------------------------------------------------------
    # Build the 35-combo roster as plain dicts (avoid importing OllamaBot
    # until run time, in case `ollama` isn't installed on the planning
    # machine).
    # ------------------------------------------------------------------
    plans = []
    for model in LOCAL_ROSTER:
        slug = model.id.replace(":", "-").replace(".", "")
        for p in ALL_PERSONALITIES:
            plans.append({
                "name": f"{slug}-{p.id}",
                "model_id": model.id,
                "personality": p,
            })
    print(f"Roster: {len(plans)} bot plans "
          f"({len(LOCAL_ROSTER)} models x {len(ALL_PERSONALITIES)} personalities)")

    # ------------------------------------------------------------------
    # Generate random tables, sorted by model membership for swap
    # efficiency (groups tables with overlapping model sets adjacent).
    # ------------------------------------------------------------------
    rng = random.Random(SEED)
    tables = []
    for _ in range(NUM_TABLES):
        seats = rng.sample(plans, 4)
        rng.shuffle(seats)
        tables.append(seats)
    tables.sort(key=lambda t: tuple(sorted({b["model_id"] for b in t})))

    print(f"\n=== Schedule ({NUM_TABLES} tables x {HANDS_PER_TABLE} hands) ===")
    for i, t in enumerate(tables, 1):
        models = sorted({b["model_id"] for b in t})
        print(f"  T{i}: models={models}")
        for b in t:
            print(f"      {b['name']}")

    # ------------------------------------------------------------------
    # Refuse to overwrite an existing run with this session_id.
    # ------------------------------------------------------------------
    runs_root = PROJECT_ROOT / "runs"
    runs_root.mkdir(exist_ok=True)
    if (runs_root / SESSION_ID).exists():
        print(f"\nERROR: runs/{SESSION_ID}/ already exists. "
              "Delete it or change SESSION_ID and retry.")
        return 1

    # Construct the full session metadata up front. The tracker writes
    # config.json once on start_session(), and our 35-combo roster is
    # the canonical seat list even though only some appear per table.
    seat_configs = [
        SeatConfig(
            name=p["name"],
            model_id=p["model_id"],
            personality_id=p["personality"].id,
            personality_description=p["personality"].description,
            starting_chips=STARTING_CHIPS,
        )
        for p in plans
    ]

    # ------------------------------------------------------------------
    # Run the tournament. OllamaBot imported here so import-time errors
    # surface clearly.
    # ------------------------------------------------------------------
    from bots import OllamaBot

    print(f"\n=== Starting tournament — writing to runs/{SESSION_ID} ===")
    t_start = time.time()
    with HandTracker(sessions_root=str(runs_root), session_id=SESSION_ID) as tracker:
        tracker.start_session(
            seat_configs,
            num_tables=NUM_TABLES,
            hands_per_table=HANDS_PER_TABLE,
            ante=ANTE,
            min_bet=MIN_BET,
            starting_chips=STARTING_CHIPS,
            broke_player_policy="rebuy",
            seed=SEED,
            roster_size=len(plans),
        )

        gid_counter = count(1)
        for t_idx, table_plans in enumerate(tables, 1):
            print(f"\n--- Table {t_idx}/{NUM_TABLES} ---")
            for p in table_plans:
                print(f"    {p['name']}")

            # Materialise OllamaBots with per-model overrides
            bots = []
            for p in table_plans:
                spec = BY_ID[p["model_id"]]
                bots.append(OllamaBot(
                    name=p["name"],
                    personality=p["personality"],
                    model_id=p["model_id"],
                    num_predict=spec.num_predict,
                    system_prefix=spec.system_prefix,
                    think=spec.think,
                ))

            players = [Player(name=b.name, chips=STARTING_CHIPS) for b in bots]
            seats = [
                Seat(player=p, agent=TrackingAgent(b, tracker))
                for p, b in zip(players, bots)
            ]
            game = Game(seats=seats, ante=ANTE, min_bet=MIN_BET,
                        seed=SEED + t_idx * 100)

            t_table = time.time()
            for _ in range(HANDS_PER_TABLE):
                for p in players:
                    if p.chips < ANTE:
                        p.chips = STARTING_CHIPS
                gid = next(gid_counter)
                tracker.start_hand(gid)
                summary = game.play_hand()
                summary.hand_id = gid
                tracker.log_hand(summary)
            dt = time.time() - t_table
            print(f"    Done in {dt:.0f}s ({dt / HANDS_PER_TABLE:.1f}s/hand)")

    t_total = time.time() - t_start
    print(f"\n=== Tournament complete in {t_total / 60:.1f} min ===")

    # ------------------------------------------------------------------
    # Diagnostics — the actual point of this script
    # ------------------------------------------------------------------
    reasoning = load_reasoning(runs_root / SESSION_ID)
    by_model = defaultdict(list)
    for r in reasoning:
        by_model[r["model_id"]].append(r)

    print(f"\n=== Per-model diagnostics ===")
    print(f"  {'model':<20s} {'calls':>6s} {'errors':>7s} {'err%':>6s} "
          f"{'mean_len':>9s}  result")

    all_pass = True
    seen_models = set()
    for model in sorted(by_model.keys()):
        if model.startswith("mock"):
            continue
        seen_models.add(model)
        rows = by_model[model]
        errors = sum(1 for r in rows if r.get("parse_error"))
        err_rate = 100 * errors / len(rows) if rows else 0
        lens = [len(r.get("raw_response") or "") for r in rows]
        mean_len = sum(lens) / len(lens) if lens else 0

        length_ok = mean_len > 50
        parse_ok = err_rate < 20
        pass_flag = length_ok and parse_ok
        if not pass_flag:
            all_pass = False

        flag = "PASS" if pass_flag else "FAIL"
        print(f"  {model:<20s} {len(rows):>6d} {errors:>7d} "
              f"{err_rate:>5.1f}% {mean_len:>8.0f}  [{flag}]")

    # Flag any models in the roster that didn't get any calls (didn't
    # appear in any table — possible at small NUM_TABLES).
    expected = {m.id for m in LOCAL_ROSTER}
    missed = expected - seen_models
    if missed:
        print(f"\n  NOTE: these models didn't appear in any table this run "
              f"(small sample): {sorted(missed)}")

    print(f"\n=== Verdict ===")
    if all_pass and not missed:
        print("  ALL MODELS PASSED. Cleared to launch main_round_robin_v2.")
        return 0
    elif all_pass and missed:
        print("  All seated models passed, but some didn't get sampled.")
        print("  Re-run with NUM_TABLES bumped to 6+ for full coverage.")
        return 0
    else:
        print("  SOME MODELS FAILED. Inspect before scaling up:")
        print(f"    runs/{SESSION_ID}/reasoning.jsonl")
        return 2


if __name__ == "__main__":
    sys.exit(main())
