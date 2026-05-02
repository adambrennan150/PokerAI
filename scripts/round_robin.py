"""
round_robin.py — main experimental tournament.

Builds the full 35-bot roster (7 models × 5 personalities) as
OllamaBots, generates ~N balanced 4-bot tables, and runs them all
under a single HandTracker session. Output lands in
runs/<SESSION_ID>/ where the analytics notebook can pick it up.

Why this isn't single-elimination
---------------------------------
The brief asks for *averages* across many hands, which is what gives
the (model × personality) groupbys statistical power. A bracket
collects only one match's worth of data per round-1 loser. Round-
robin with rotated tables gives every combo many data points.

Balance + scheduling
--------------------
Tables are sampled with a soft inverse-appearance bias, so each
combo plays roughly the same number of hands. After generation,
tables are sorted by their tuple of sorted model IDs — this groups
tables sharing a heavy model (qwen3:14b in particular) adjacent so
Ollama can keep that model resident across them rather than
unloading and reloading.

Configuration
-------------
The dials at the top of the file (NUM_TABLES, HANDS_PER_TABLE, etc.)
are the only knobs you need. Defaults target ~8 hours overnight on a
32GB GPU; halve `NUM_TABLES` for a faster pilot run.

Usage
-----
    python scripts/round_robin.py            # run the full thing
    python scripts/round_robin.py --dry-run  # show schedule, don't execute

If a session with this id already exists, the script aborts. Use a
versioned id (v1, v2, ...) to reproduce or vary the run.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Project root on sys.path so we can import the project modules.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.base import BaseBot, Personality
from bots.personalities import ALL as ALL_PERSONALITIES
from config.models import LOCAL_ROSTER
from engine import Game, Player, Seat
from tracker import HandTracker, SeatConfig, TrackingAgent

# OllamaBot is imported lazily inside `materialise_bot` so that
# `--dry-run` works in environments where the `ollama` package is
# not installed (e.g. the planning machine vs the GPU box).


# ======================================================================
# Configuration — the only dials that matter
# ======================================================================
SESSION_ID = "main_round_robin_v1"

# Each combo will appear in roughly NUM_TABLES * 4 / 35 tables.
# At 30 tables × 50 hands × ~25s/hand → ~10h. Halve NUM_TABLES for a
# pilot, double HANDS_PER_TABLE for more per-table data.
NUM_TABLES = 30
HANDS_PER_TABLE = 50

ANTE = 2
MIN_BET = 5
STARTING_CHIPS = 200
BROKE_PLAYER_POLICY = "rebuy"      # keeps every combo in the data
SEED = 42                          # reproducible deck shuffles + table picks

TABLE_SIZE = 4                     # 5-card draw seats; not the dial to tweak
SECONDS_PER_HAND_GUESS = 25        # rough first-pass ETA estimate


# ======================================================================
# Roster construction — two-stage: plans first, real bots at run time.
# ======================================================================
@dataclass(frozen=True)
class BotPlan:
    """Lightweight description of a (model × personality) combo.

    Holds everything we need for planning (table generation, scheduling,
    pretty-printing, SeatConfig construction) without importing or
    constructing the LLM client. Real `OllamaBot` instances get built
    only at run time via `materialise_bot()` — that way `--dry-run`
    works even on machines without `ollama` installed.
    """
    name: str
    model_id: str
    personality: Personality

    @property
    def personality_id(self) -> str:
        return self.personality.id


def build_plans() -> List[BotPlan]:
    """Cartesian product of LOCAL_ROSTER × ALL_PERSONALITIES. Names are
    constructed `<model>-<personality>` so they're identifiable in the
    tracker outputs and in the analytics groupbys."""
    plans: List[BotPlan] = []
    for model in LOCAL_ROSTER:
        # Make the model id safe to embed in a player name (no colons).
        model_slug = model.id.replace(":", "-").replace(".", "")
        for personality in ALL_PERSONALITIES:
            plans.append(BotPlan(
                name=f"{model_slug}-{personality.id}",
                model_id=model.id,
                personality=personality,
            ))
    return plans


def materialise_bot(plan: BotPlan) -> BaseBot:
    """Construct a real OllamaBot from a plan. Imported lazily so that
    the dry-run path doesn't require `ollama` to be installed."""
    from bots import OllamaBot   # lazy import — only run-time path
    return OllamaBot(
        name=plan.name,
        personality=plan.personality,
        model_id=plan.model_id,
    )


# ======================================================================
# Table generation — balanced sampling
# ======================================================================
def generate_balanced_tables(
    bots: Sequence[BotPlan],
    num_tables: int,
    table_size: int,
    rng: random.Random,
) -> List[List[BotPlan]]:
    """Generate `num_tables` tables of `table_size` distinct bots each.

    Soft inverse-appearance weighting: the fewer times a bot has been
    seated so far, the higher its sampling weight. This produces
    roughly-balanced exposure without needing strict Latin-square
    bookkeeping. Within a table, the four picks are without
    replacement.

    The function is deterministic given `rng` — same seed produces
    the same schedule, which matters for reproducibility.
    """
    if table_size > len(bots):
        raise ValueError(
            f"table_size ({table_size}) cannot exceed roster size ({len(bots)})"
        )

    appearances = {bot.name: 0 for bot in bots}
    tables: List[List[BotPlan]] = []
    for _ in range(num_tables):
        # Weight = 1 / (1 + appearances) — under-seen combos boost.
        candidates = list(bots)
        weights = [1.0 / (1 + appearances[c.name]) for c in candidates]

        seats: List[BotPlan] = []
        for _ in range(table_size):
            # Without-replacement weighted sample: pick one, remove,
            # repeat. `random.choices` with k=1 plus a manual remove
            # is the most readable way given the small table size.
            idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
            seats.append(candidates.pop(idx))
            weights.pop(idx)

        # Shuffle seat order so no bot consistently acts first.
        rng.shuffle(seats)
        tables.append(seats)

        for s in seats:
            appearances[s.name] += 1

    return tables


def sort_tables_by_model_membership(
    tables: Sequence[Sequence[BotPlan]],
) -> List[List[BotPlan]]:
    """Reorder tables so adjacent tables share model membership where
    possible. Reduces Ollama swap thrash — particularly important for
    the heavy qwen3:14b model.

    Strategy: sort by the sorted tuple of distinct model_ids. Tables
    with identical model sets become contiguous; tables differing by
    one model also tend to cluster.
    """
    def key(table):
        return tuple(sorted({b.model_id for b in table}))

    return sorted([list(t) for t in tables], key=key)


# ======================================================================
# Schedule introspection — printed both for --dry-run and at start of run
# ======================================================================
def appearance_summary(tables: Iterable[Sequence[BotPlan]]) -> List[Tuple[str, int]]:
    counts: dict = {}
    for t in tables:
        for b in t:
            counts[b.name] = counts.get(b.name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def print_schedule(tables: Sequence[Sequence[BotPlan]]) -> None:
    """Pretty-print the planned schedule plus appearance balance."""
    total_hands = len(tables) * HANDS_PER_TABLE
    eta_seconds = total_hands * SECONDS_PER_HAND_GUESS

    print(f"=== Round-robin schedule ===")
    print(f"  Session id      : {SESSION_ID}")
    print(f"  Tables          : {len(tables)}")
    print(f"  Hands per table : {HANDS_PER_TABLE}")
    print(f"  Total hands     : {total_hands}")
    print(f"  Seed            : {SEED}")
    print(f"  Estimated time  : {eta_seconds / 3600:.1f} h "
          f"(at ~{SECONDS_PER_HAND_GUESS}s/hand)")

    print(f"\n=== Per-bot appearance counts ===")
    counts = appearance_summary(tables)
    if counts:
        max_count = max(c for _, c in counts)
        min_count = min(c for _, c in counts)
        print(f"  range: {min_count}..{max_count} appearances per bot")
    for name, n in counts:
        print(f"  {name:<28s} {n:>2d}")

    print(f"\n=== Tables (sorted by model membership) ===")
    for i, table in enumerate(tables, 1):
        models = sorted({b.model_id for b in table})
        members = ", ".join(b.name for b in table)
        print(f"  T{i:>2d}: models={models}")
        print(f"       seats=[{members}]")


# ======================================================================
# The actual tournament loop
# ======================================================================
def run_tournament(
    tables: Sequence[Sequence[BotPlan]],
    full_plans: Sequence[BotPlan],
    runs_root: Path,
) -> None:
    """Execute every table in sequence under a single HandTracker
    session. Hand IDs are globally monotonic across tables so the
    analytics can treat the whole run as one dataset.

    Real `OllamaBot` instances are materialised here, not at planning
    time — so a missing `ollama` install only fails at this point,
    not during `--dry-run`.
    """

    # The tracker's session config records every bot in the roster,
    # not just the ones at any single table — so analytics can join
    # cleanly even on bots that didn't appear in every table.
    seat_configs = [
        SeatConfig(
            name=p.name,
            model_id=p.model_id,
            personality_id=p.personality.id,
            personality_description=p.personality.description,
            starting_chips=STARTING_CHIPS,
        )
        for p in full_plans
    ]

    # Refuse to overwrite an existing session — force a versioned name.
    target_dir = runs_root / SESSION_ID
    if target_dir.exists():
        raise SystemExit(
            f"Session directory already exists: {target_dir}\n"
            f"Either delete it, or change SESSION_ID to a new version "
            f"(e.g. {SESSION_ID}_v2)."
        )

    print(f"\n=== Starting tournament — writing to {target_dir} ===\n")
    t_start = time.time()

    with HandTracker(sessions_root=str(runs_root), session_id=SESSION_ID) as tracker:
        tracker.start_session(
            seat_configs,
            num_tables=len(tables),
            hands_per_table=HANDS_PER_TABLE,
            ante=ANTE,
            min_bet=MIN_BET,
            starting_chips=STARTING_CHIPS,
            broke_player_policy=BROKE_PLAYER_POLICY,
            seed=SEED,
            roster_size=len(full_plans),
        )

        global_hand_id = count(1)
        for t_idx, table_plans in enumerate(tables, 1):
            print(f"\n--- Table {t_idx}/{len(tables)} ---")
            for p in table_plans:
                print(f"    {p.name}")

            # Materialise real OllamaBots for this table.
            table_bots = [materialise_bot(p) for p in table_plans]

            # Fresh Players for each table; chips reset to STARTING_CHIPS
            # at the table boundary regardless of broke_player_policy.
            players = [Player(name=b.name, chips=STARTING_CHIPS) for b in table_bots]
            seats = [
                Seat(player=p, agent=TrackingAgent(b, tracker))
                for p, b in zip(players, table_bots)
            ]

            # Per-table seed so individual table behaviour is
            # reproducible while the overall pattern stays seeded.
            game = Game(
                seats=seats, ante=ANTE, min_bet=MIN_BET,
                seed=SEED + t_idx * 100,
            )

            t_table_start = time.time()
            for _ in range(HANDS_PER_TABLE):
                # Apply broke-player policy before each hand.
                if BROKE_PLAYER_POLICY == "rebuy":
                    for p in players:
                        if p.chips < ANTE:
                            p.chips = STARTING_CHIPS

                gid = next(global_hand_id)
                tracker.start_hand(gid)            # tags reasoning rows
                summary = game.play_hand()
                summary.hand_id = gid              # tag hand + action rows
                tracker.log_hand(summary)

            dt_table = time.time() - t_table_start
            elapsed = time.time() - t_start
            avg_per_table = elapsed / t_idx
            remaining = (len(tables) - t_idx) * avg_per_table
            print(f"    Done in {dt_table:.0f}s "
                  f"({dt_table / HANDS_PER_TABLE:.1f}s/hand). "
                  f"ETA remaining: {remaining / 3600:.1f}h")

    t_total = time.time() - t_start
    print(f"\n=== Tournament complete ===")
    print(f"  Wall-clock: {t_total / 3600:.2f}h ({t_total:.0f}s)")
    print(f"  Tables    : {len(tables)}")
    print(f"  Total hands: {len(tables) * HANDS_PER_TABLE}")
    print(f"  Output     : {target_dir}")


# ======================================================================
# Entry point
# ======================================================================
def main() -> int:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    rng = random.Random(SEED)

    # 1. Build the planning roster of 35 (model × personality) combos.
    plans = build_plans()
    print(f"Roster: {len(plans)} bot plans "
          f"({len(LOCAL_ROSTER)} models × {len(ALL_PERSONALITIES)} personalities)")

    # 2. Generate balanced tables.
    tables = generate_balanced_tables(
        plans, NUM_TABLES, TABLE_SIZE, rng=rng,
    )

    # 3. Sort for VRAM-swap efficiency.
    tables = sort_tables_by_model_membership(tables)

    # 4. Print the plan.
    print_schedule(tables)

    if dry_run:
        print("\n--dry-run: schedule printed, not executing.")
        return 0

    # 5. Run — real OllamaBots get constructed inside run_tournament.
    runs_root = PROJECT_ROOT / "runs"
    runs_root.mkdir(exist_ok=True)
    run_tournament(tables, plans, runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
