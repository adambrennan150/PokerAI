"""
validation_reasoning_models.py — Phase 4 validation.

Targeted re-test of the three models that returned ~100% empty
responses in the v1 round-robin:
    deepseek-r1:7b   (chain-of-thought, no thinking-disable toggle)
    qwen3:8b         (thinking mode by default)
    qwen3:14b        (thinking mode by default)

Goal: confirm that the per-model fixes in `config/models.py` —
bumped `num_predict` and `/no_think` for Qwen3 — produce non-empty
responses with low parse-error rates. If they do, we're cleared to
re-run the full round-robin (`main_round_robin_v2`).

Setup
-----
* 4-bot table with all three reasoning models + 1 passive MockBot.
* 10 hands, ~2-5 minutes wall-clock per LLM-heavy hand once warmed
  (DeepSeek-R1 with 4k token budget can be slow when thinking runs
  long, so total time may approach 30-60 min).
* Output lands in runs/validation_reasoning_models_v1/ so it can be
  inspected and committed.

Pass criteria
-------------
For each of the three target models:
* Mean response length > 100 chars (real answer present, not empty).
* Parse-error rate < 20% (strict JSON path mostly succeeding).
* Sample reasoning is coherent — references cards, pot, opponents.

Usage
-----
    python scripts/validation_reasoning_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots import MockBot, OllamaBot
from bots.personalities import (
    TIGHT_AGGRESSIVE, LOOSE_AGGRESSIVE, BLUFFER, CALLING_STATION,
)
from config.models import BY_ID
from runner import RunnerConfig, TournamentRunner
from tracker import load_reasoning


# Configuration
NUM_HANDS = 10
SESSION_ID = "validation_reasoning_models_v1"
SEED = 42

TARGET_MODELS = ["deepseek-r1:7b", "qwen3:8b", "qwen3:14b"]


def make_ollama_bot(name, personality, model_id):
    """Build an OllamaBot using the per-model overrides from config."""
    spec = BY_ID[model_id]
    return OllamaBot(
        name=name,
        personality=personality,
        model_id=model_id,
        num_predict=spec.num_predict,
        system_prefix=spec.system_prefix,
    )


# Build the table — three reasoning bots with diverse personalities
# (so the parse-error analysis can rule out personality-specific
# failures), plus one MockBot for table fill.
caller_resp = '{"reasoning": "Just calling.", "action": "call"}'
stand_pat   = '{"reasoning": "Standing pat.", "discards": []}'

bots = [
    make_ollama_bot("DeepSeek-TAG",   TIGHT_AGGRESSIVE,  "deepseek-r1:7b"),
    make_ollama_bot("Qwen8b-Bluffer", BLUFFER,           "qwen3:8b"),
    make_ollama_bot("Qwen14b-LAG",    LOOSE_AGGRESSIVE,  "qwen3:14b"),
    MockBot("MockFiller", CALLING_STATION, caller_resp, stand_pat,
            model_id="mock-filler"),
]

print(f"=== Validation: 3 reasoning models + 1 MockBot, {NUM_HANDS} hands ===")
for b in bots:
    if isinstance(b, OllamaBot):
        spec = BY_ID[b.model_id]
        print(f"  {b.name:<20s} {b.model_id:<18s} "
              f"num_predict={spec.num_predict}, "
              f"system_prefix={spec.system_prefix!r}")
    else:
        print(f"  {b.name:<20s} (MockBot)")
print()


# Run the tournament — saves to project runs/ for inspection / commit
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

t0 = time.time()
result = TournamentRunner(bots, cfg).run()
dt = time.time() - t0

print(f"\nFinished in {dt:.1f}s ({dt / NUM_HANDS:.1f}s/hand)")
print(result.summary_table())


# Diagnostics — the actual point of this script
reasoning = load_reasoning(result.session_dir)

print(f"\n=== Per-bot LLM diagnostics ===")
all_pass = True
for bot_name in ("DeepSeek-TAG", "Qwen8b-Bluffer", "Qwen14b-LAG"):
    rows = [r for r in reasoning if r["player"] == bot_name]
    if not rows:
        print(f"  {bot_name:<20s} NO LLM CALLS RECORDED")
        all_pass = False
        continue

    errors = [r for r in rows if r.get("parse_error")]
    error_rate = 100 * len(errors) / len(rows)
    lengths = [len(r.get("raw_response") or "") for r in rows]
    mean_len = sum(lengths) / len(lengths)

    # Pass criteria
    length_ok = mean_len > 100
    parse_ok = error_rate < 20

    flag = "PASS" if (length_ok and parse_ok) else "FAIL"
    if not (length_ok and parse_ok):
        all_pass = False

    print(f"  {bot_name:<20s} [{flag}]")
    print(f"    LLM calls         : {len(rows)}")
    print(f"    Parse-error rate  : {len(errors)}/{len(rows)} ({error_rate:.1f}%)  "
          f"{'[ok]' if parse_ok else '[FAIL: target <20%]'}")
    print(f"    Mean response len : {mean_len:.0f} chars  "
          f"{'[ok]' if length_ok else '[FAIL: target >100]'}")

print(f"\n=== Sample reasoning (one per LLM bot) ===")
for bot_name in ("DeepSeek-TAG", "Qwen8b-Bluffer", "Qwen14b-LAG"):
    actions = [
        r for r in reasoning
        if r["player"] == bot_name
        and r.get("decision_type") == "action"
        and r.get("reasoning")    # skip empty ones
    ]
    if not actions:
        print(f"\n  {bot_name}: no parseable action reasoning recorded.")
        continue
    sample = actions[0]
    print(f"\n  {bot_name} (hand {sample['hand_id']}, {sample['phase']}):")
    print(f"    decision : {sample.get('decided_action')}")
    print(f"    reasoning: {sample['reasoning']!r}")

print(f"\n=== Verdict ===")
if all_pass:
    print("  ALL THREE MODELS PASSED.")
    print("  Cleared to re-run the full round-robin "
          "(SESSION_ID = 'main_round_robin_v2').")
else:
    print("  ONE OR MORE MODELS FAILED.")
    print("  Inspect runs/{}/reasoning.jsonl for failing samples".format(SESSION_ID))
    print("  before scaling up.")

print(f"\nSession persisted: {result.session_dir}")
print(f"Commit: git add -f runs/{SESSION_ID}/ && git commit -m "
      f"'Validation: reasoning models with bumped num_predict + /no_think'")
