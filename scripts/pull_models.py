"""
pull_models.py — pull every model in the configured roster via Ollama.

Reads the canonical roster from `config/models.py` and runs
`ollama pull <id>` for each. Idempotent — Ollama itself skips models
that are already pulled, so re-running is cheap.

Usage:
    python scripts/pull_models.py            # pulls LOCAL_ROSTER (7 models)
    python scripts/pull_models.py --colab    # pulls COLAB_ROSTER (4 models)
    python scripts/pull_models.py --dry-run  # show what would be pulled, no action

Prereqs: Ollama installed and the daemon running. Verify with
`ollama list` before kicking this off.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Project root on sys.path so we can import config.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.models import LOCAL_ROSTER, COLAB_ROSTER, ModelSpec


def main() -> int:
    args = set(sys.argv[1:])
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    use_colab = "--colab" in args
    dry_run   = "--dry-run" in args
    roster = COLAB_ROSTER if use_colab else LOCAL_ROSTER

    label = "COLAB" if use_colab else "LOCAL"
    total_ram = sum(m.ram_gb for m in roster)
    print(f"=== {label} roster: {len(roster)} models "
          f"(~{total_ram:.1f} GB total on disk) ===")
    for m in roster:
        print(f"  {m.id:<22s} {m.family:<10s} {m.size_b:>5.1f}B  ~{m.ram_gb:.1f}G")

    if dry_run:
        print("\n--dry-run: not pulling anything.")
        return 0

    if not _ollama_available():
        print("\nERROR: `ollama` CLI not found. Install Ollama first:")
        print("  Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  Other: see https://ollama.com/download")
        return 1

    print()  # spacer before the pulls
    failed: list[ModelSpec] = []
    succeeded: list[ModelSpec] = []
    t_start = time.time()
    for i, m in enumerate(roster, 1):
        print(f"--- [{i}/{len(roster)}] Pulling {m.id} ---")
        t0 = time.time()
        rc = subprocess.run(["ollama", "pull", m.id]).returncode
        dt = time.time() - t0
        if rc == 0:
            print(f"    OK ({dt:.1f}s)\n")
            succeeded.append(m)
        else:
            print(f"    FAILED (rc={rc}, {dt:.1f}s)\n")
            failed.append(m)
    t_total = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"=== Pull summary ({t_total:.1f}s) ===")
    print(f"  Succeeded: {len(succeeded)}/{len(roster)}")
    print(f"  Failed   : {len(failed)}/{len(roster)}")
    if failed:
        print("\nFailed models:")
        for m in failed:
            print(f"  - {m.id}")
        print(
            "\nThe tag may not exist in the Ollama registry. Try:\n"
            "  ollama search <model_name>\n"
            "Then update the tag in config/models.py."
        )

    print(f"\n=== Currently pulled (`ollama list`) ===")
    subprocess.run(["ollama", "list"])

    return 0 if not failed else 2


def _ollama_available() -> bool:
    """True iff the `ollama` CLI is on PATH."""
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True, check=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


if __name__ == "__main__":
    sys.exit(main())
