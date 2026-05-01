"""
inspect_parse_errors.py — drill into parse-error rows for a session.

Run this whenever a tournament finishes with non-zero parse-error rate
to see exactly what the LLM produced that the parser couldn't handle.
Useful both for prompt tuning (if the same model keeps failing in the
same way) and for the report (so we can characterise model robustness).

Usage:
    python scripts/inspect_parse_errors.py [SESSION_ID]

Defaults to `validation_heads_up_v1` if no session id is provided. Run
with the session folder name as it appears under runs/, e.g.:

    python scripts/inspect_parse_errors.py main_round_robin_v1
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Project root on sys.path so we can import the tracker.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracker import load_reasoning


SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else "validation_heads_up_v1"
session_dir = PROJECT_ROOT / "runs" / SESSION_ID

if not session_dir.exists():
    raise SystemExit(f"Session not found: {session_dir}")


reasoning = load_reasoning(session_dir)
errors = [r for r in reasoning if r.get("parse_error")]

# ----------------------------------------------------------------------
# Summary header
# ----------------------------------------------------------------------
total = len(reasoning)
print(f"Session: {SESSION_ID}")
print(f"  Total reasoning rows : {total}")
print(f"  Parse-error rows     : {len(errors)} "
      f"({100 * len(errors) / max(1, total):.1f}%)")

if not errors:
    print("\nNo parse errors found.")
    raise SystemExit(0)

# ----------------------------------------------------------------------
# Aggregations — quick sense of where errors cluster.
# ----------------------------------------------------------------------
print(f"\n=== Errors by player ===")
by_player = Counter(e["player"] for e in errors)
for name, n in by_player.most_common():
    print(f"  {name:<24s} {n}")

print(f"\n=== Errors by decision_type ===")
by_type = Counter(e["decision_type"] for e in errors)
for kind, n in by_type.most_common():
    print(f"  {kind:<10s} {n}")

print(f"\n=== Errors by error message (first 60 chars) ===")
by_msg = Counter((e["parse_error"] or "")[:60] for e in errors)
for msg, n in by_msg.most_common():
    print(f"  [{n}] {msg!r}")

# ----------------------------------------------------------------------
# Per-error detail. Truncate long raw responses so the output stays
# scannable; the full text is on disk if you need it.
# ----------------------------------------------------------------------
print(f"\n=== Per-error details ===")
for i, e in enumerate(errors, 1):
    raw = e.get("raw_response") or ""
    if len(raw) > 400:
        raw = raw[:400] + f"... ({len(e['raw_response']) - 400} more chars)"
    print(f"\n--- Error #{i} of {len(errors)} ---")
    print(f"  hand_id      : {e['hand_id']}")
    print(f"  player       : {e['player']}  "
          f"(model={e['model_id']}, personality={e['personality_id']})")
    print(f"  phase        : {e['phase']}")
    print(f"  decision_type: {e['decision_type']}")
    print(f"  parse_error  : {e['parse_error']!r}")
    print(f"  decided      : "
          f"{e.get('decided_action') or e.get('decided_discards')}")
    print(f"  raw_response : {raw!r}")
