"""
validation_heads_up.py — Phase 3 of LLM bot validation.

Two real LLM bots (Llama 3.1:8b and Mistral 7B) with deliberately
contrasting personalities, plus two MockBots for table fill. Runs a
15-hand tournament, then prints diagnostics:

  * per-bot LLM call counts and parse-error rates,
  * a sample of reasoning from each LLM bot.

Goal of this validation step: confirm that
  1. Two different LLMs run reliably side-by-side at the same table.
  2. The personality system prompts produce visibly different
     reasoning between bots (the brief's "best personality" question
     depends on this being true).
  3. The full pipeline (engine → tracker → analytics) handles a
     realistic mixed-LLM session without manual intervention.

Output goes to <project_root>/runs/validation_heads_up_v1/ rather
than a temp directory, so the resulting session can be committed and
referred to from the report. Re-running overwrites that session
(append-only files are reset by the runner each run).

Usage:
    python scripts/validation_heads_up.py

Roughly 5–10 minutes wall-clock on a 32GB GPU; ~300 LLM calls at ~1s
each once both models are warm.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the project root importable when invoked directly via
#   `python scripts/validation_heads_up.py`
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots import MockBot, OllamaBot
from bots.personalities import TIGHT_AGGRESSIVE, CALLING_STATION
from runner import RunnerConfig, TournamentRunner
from tracker import load_reasoning


# ----------------------------------------------------------------------
# Configuration — tweak here, not in the analytics layer.
# ----------------------------------------------------------------------
NUM_HANDS = 15
SESSION_ID = "validation_heads_up_v1"
SEED = 42

LLAMA_MODEL = "llama3.1:8b"
MISTRAL_MODEL = "mistral"


# ----------------------------------------------------------------------
# Build the table.
# ----------------------------------------------------------------------
# Two MockBot styles to provide table fill — one passive, one
# fold-prone — so the LLMs experience varied opponents without
# dominating each other.
caller_resp = '{"reasoning": "Just calling.", "action": "call"}'
fold_resp   = '{"reasoning": "Not worth it.", "action": "fold"}'
stand_pat   = '{"reasoning": "Standing pat.", "discards": []}'

bots = [
    OllamaBot(
        name="Llama-TAG",                  # Llama 3.1:8b + tight-aggressive
        personality=TIGHT_AGGRESSIVE,
        model_id=LLAMA_MODEL,
    ),
    OllamaBot(
        name="Mistral-Station",            # Mistral 7B + calling-station
        personality=CALLING_STATION,
        model_id=MISTRAL_MODEL,
    ),
    MockBot(
        "MockTight", TIGHT_AGGRESSIVE,
        fold_resp, stand_pat, model_id="mock-tight",
    ),
    MockBot(
        "MockCaller", CALLING_STATION,
        caller_resp, stand_pat, model_id="mock-caller",
    ),
]


# ----------------------------------------------------------------------
# Run the tournament. Output lands in <project_root>/runs/<session>/
# so it persists for the analytics notebook + git commits.
# ----------------------------------------------------------------------
runs_root = PROJECT_ROOT / "runs"
runs_root.mkdir(exist_ok=True)

cfg = RunnerConfig(
    num_hands=NUM_HANDS,
    starting_chips=200,
    ante=2,
    min_bet=5,
    broke_player_policy="rebuy",
    seed=SEED,
    verbose=True,
    sessions_root=str(runs_root),
    session_id=SESSION_ID,
)

print(f"=== Heads-up validation: {LLAMA_MODEL} (TAG) vs {MISTRAL_MODEL} (Station) ===")
print(f"  {NUM_HANDS} hands, seed {SEED}, both LLMs + 2 MockBots\n")

t0 = time.time()
result = TournamentRunner(bots, cfg).run()
dt = time.time() - t0

print(f"\nFinished in {dt:.1f}s ({dt / NUM_HANDS:.1f}s/hand)")
print(result.summary_table())


# ----------------------------------------------------------------------
# Diagnostics — what we actually want to verify from this run.
# ----------------------------------------------------------------------
reasoning = load_reasoning(result.session_dir)

print(f"\n=== Per-bot LLM diagnostics ===")
for bot_name in ("Llama-TAG", "Mistral-Station"):
    rows = [r for r in reasoning if r["player"] == bot_name]
    errors = [r for r in rows if r["parse_error"]]
    rate = (100 * len(errors) / len(rows)) if rows else 0.0
    print(f"  {bot_name}:")
    print(f"    LLM calls         : {len(rows)}")
    print(f"    Parse-error rate  : {len(errors)}/{len(rows)} ({rate:.0f}%)")

print(f"\n=== Sample reasoning (one per LLM bot) ===")
for bot_name in ("Llama-TAG", "Mistral-Station"):
    actions = [
        r for r in reasoning
        if r["player"] == bot_name and r["decision_type"] == "action"
    ]
    if not actions:
        print(f"\n  {bot_name}: no action-decision reasoning recorded.")
        continue
    sample = actions[0]
    print(f"\n  {bot_name} (hand {sample['hand_id']}, {sample['phase']}):")
    print(f"    decision : {sample['decided_action']}")
    print(f"    reasoning: {sample['reasoning']!r}")
    if sample["parse_error"]:
        print(f"    parse_error: {sample['parse_error']!r}")

print(f"\n=== Session persisted ===")
print(f"  Path        : {result.session_dir}")
print(f"  In analytics: open notebooks/analytics.ipynb, set SESSION_ID = "
      f"{SESSION_ID!r}")
print(f"  Commit      : git add -f runs/{SESSION_ID}/ && "
      f"git commit -m 'Heads-up validation run'")
