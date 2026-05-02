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

    `id`             — Ollama tag, must match what `ollama list` reports.
    `family`         — Vendor / training family. Tracker's "best LLM
                       family" groupby key.
    `size_b`         — Parameter count in billions.
    `ram_gb`         — Approximate VRAM footprint (4-bit quant). Used
                       by the tournament driver to pack tables.
    `notes`          — Short human description.
    `num_predict`    — Per-model token budget. Reasoning-style models
                       (DeepSeek-R1, Qwen3 in thinking mode) need much
                       more headroom than vanilla instruction-tuned
                       models, otherwise they exhaust the budget inside
                       their <think> block and produce empty output.
    `system_prefix`  — Optional text prepended to every system prompt.
                       Used to disable thinking on Qwen3 via `/no_think`.
    """

    id: str
    family: str
    size_b: float
    ram_gb: float
    notes: str = ""
    num_predict: int = 512
    system_prefix: str = ""

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
        id="qwen3:8b",
        family="Qwen",
        size_b=8.0, ram_gb=5.5,
        notes="Alibaba. Medium-tier — Qwen3 has no 7B, jumps 4B→8B.",
        # /no_think disables Qwen3's reasoning preamble so the response
        # is direct JSON. Without this, the <think> block consumed the
        # whole token budget in the v1 round-robin and the bot never
        # actually answered.
        system_prefix="/no_think",
        num_predict=1024,
    ),
    ModelSpec(
        id="deepseek-r1:7b",
        family="DeepSeek",
        size_b=7.0, ram_gb=5.0,
        notes="Visible chain-of-thought reasoning — wildcard for poker.",
        # DeepSeek-R1 has no thinking-disable toggle; its reasoning is
        # baked in. We give it a 4k token budget so thinking can run to
        # completion AND leave room for the final JSON answer.
        num_predict=4096,
    ),
    ModelSpec(
        id="qwen3:14b",
        family="Qwen",
        size_b=14.0, ram_gb=12.0,
        notes="Larger class — does 14B meaningfully outperform 7B?",
        system_prefix="/no_think",
        num_predict=1024,
    ),
]


# ----------------------------------------------------------------------
# Colab roster — lean subset for the submission notebook.
# Hand-picked for: smallest baseline, medium reference (most-supported),
# alternative medium (different family), and the reasoning wildcard.
# ----------------------------------------------------------------------
_COLAB_IDS = {"gemma3:1b", "llama3.1:8b", "mistral", "deepseek-r1:7b"}
COLAB_ROSTER: List[ModelSpec] = [m for m in LOCAL_ROSTER if m.id in _COLAB_IDS]

# String id → ModelSpec lookup. Lets the tournament driver materialise
# bots without needing the full ModelSpec passed around — it can resolve
# `plan.model_id` to the spec at run time.
BY_ID: dict = {m.id: m for m in LOCAL_ROSTER}


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
