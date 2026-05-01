"""
models.py — canonical roster of LLM models for the project.

Two rosters, sharing the same `ModelSpec` schema:

    LOCAL_ROSTER  — 7 models for the main experiments on the 32GB GPU.
                    Spans 1B → 14B and six families (Gemma, Phi, Llama,
                    Mistral, Qwen, DeepSeek). This is the dataset the
                    report's analysis is based on.

    COLAB_ROSTER  — 4-model subset that fits on a free Colab T4
                    (~16GB VRAM). The submission notebook uses this so
                    the assessor can reproduce a representative slice
                    of the experiment without paid Colab.

Each entry carries enough metadata that the round-robin tournament
driver (built later) can pack tables that fit in available VRAM
without manual juggling.

Notes on model identifiers
--------------------------
Tags below are what we believe the Ollama registry has at time of
writing. If `ollama pull` reports "manifest not found" for any of
them, run `ollama search <name>` to find the closest available tag
and update this file. Common drifts:

    gemma3 → gemma2 (older Gemma family if 3 isn't yet on Ollama)
    qwen3  → qwen2.5 (Qwen 2.5 is widely available; Qwen 3 newer)
    phi4-mini → phi3:mini (Phi 3 is the older mini-variant)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    """One LLM in the roster.

    `id`      — Ollama tag, must match what `ollama list` reports.
    `family`  — Vendor / training family. Used as the tracker's
                "best LLM family" groupby key for the analytics.
    `size_b`  — Parameter count in billions. Useful for size-vs-skill
                plots in the report.
    `ram_gb`  — Approximate VRAM footprint (4-bit quant). Used by the
                tournament driver to pack tables that fit in budget.
    `notes`   — Short human description for logs and reports.
    """

    id: str
    family: str
    size_b: float
    ram_gb: float
    notes: str = ""

    def __str__(self) -> str:
        return self.id


# ----------------------------------------------------------------------
# Local roster (32GB GPU) — the experimental dataset for the report.
# ----------------------------------------------------------------------
LOCAL_ROSTER: List[ModelSpec] = [
    ModelSpec(
        id="gemma3:1b",
        family="Gemma",
        size_b=1.0, ram_gb=1.5,
        notes="Google. Smallest viable baseline — does scale matter?",
    ),
    ModelSpec(
        id="phi4-mini",
        family="Phi",
        size_b=3.8, ram_gb=3.0,
        notes="Microsoft. Small-tier alternative family.",
    ),
    ModelSpec(
        id="llama3.1:8b",
        family="Llama",
        size_b=8.0, ram_gb=6.0,
        notes="Meta. Medium reference; large open ecosystem.",
    ),
    ModelSpec(
        id="mistral",
        family="Mistral",
        size_b=7.0, ram_gb=5.0,
        notes="Mistral AI. Strong instruction-following at 7B.",
    ),
    ModelSpec(
        id="qwen3:7b",
        family="Qwen",
        size_b=7.0, ram_gb=5.0,
        notes="Alibaba. Third 7B family for cross-family breadth.",
    ),
    ModelSpec(
        id="deepseek-r1:7b",
        family="DeepSeek",
        size_b=7.0, ram_gb=5.0,
        notes="Visible chain-of-thought reasoning — wildcard for poker.",
    ),
    ModelSpec(
        id="qwen3:14b",
        family="Qwen",
        size_b=14.0, ram_gb=12.0,
        notes="Larger class — does 14B meaningfully outperform 7B?",
    ),
]


# ----------------------------------------------------------------------
# Colab roster — lean subset for the submission notebook.
# Hand-picked for: smallest baseline, medium reference (most-supported),
# alternative medium (different family), and the reasoning wildcard.
# ----------------------------------------------------------------------
_COLAB_IDS = {"gemma3:1b", "llama3.1:8b", "mistral", "deepseek-r1:7b"}
COLAB_ROSTER: List[ModelSpec] = [m for m in LOCAL_ROSTER if m.id in _COLAB_IDS]


# ----------------------------------------------------------------------
# Sanity check — run `python -m config.models`.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    def _show(label: str, roster: List[ModelSpec]) -> None:
        total_ram = sum(m.ram_gb for m in roster)
        print(f"\n=== {label} ({len(roster)} models, ~{total_ram:.1f} GB total) ===")
        print(f"  {'ID':<22s} {'FAMILY':<10s} {'SIZE':>6s}  {'RAM':>5s}  NOTES")
        for m in roster:
            print(f"  {m.id:<22s} {m.family:<10s} "
                  f"{m.size_b:>5.1f}B  {m.ram_gb:>4.1f}G  {m.notes}")

    _show("LOCAL roster", LOCAL_ROSTER)
    _show("COLAB roster", COLAB_ROSTER)
