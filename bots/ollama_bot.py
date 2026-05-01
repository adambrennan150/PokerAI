"""
ollama_bot.py — concrete LLM bot that talks to a local Ollama daemon.

A subclass of `BaseBot` whose only job is to implement `_generate`
against Ollama's chat endpoint. Every model in the project's roster
(llama3.1:8b, mistral, deepseek-r1:7b, etc.) is the same class with a
different `model_id`. Personality, prompt formatting, JSON parsing,
fallback handling, reasoning capture — all inherited from BaseBot.

Design notes
------------
* The bot owns its own `ollama.Client` instance. Sharing one across all
  bots would be marginally more efficient (single connection pool) but
  harder to debug — separate clients mean separate error traces. With
  ~10 LLM calls per hand and HTTP keep-alive on the underlying socket,
  the per-call overhead is negligible anyway.

* `temperature` and `num_predict` are tuned for poker: low-ish
  randomness so the personality system prompt has room to dominate,
  and a modest token budget (~250) so the LLM doesn't waste compute
  writing paragraphs we'd have to throw away. Both are overridable
  per-bot if a particular model wants different settings.

* Connection errors, missing models, and any other Ollama transport
  failures propagate from `_generate` as exceptions. The inherited
  `_safe_generate` wrapper in BaseBot catches them and falls back to
  a safe default action so a hung daemon doesn't crash a tournament.
  The error gets recorded in `bot.last_response.parse_error` for
  forensics.

* `OLLAMA_KEEP_ALIVE` is a daemon-side env var, not a per-call option.
  Set it before starting the Ollama service if you need finer control
  over model residency (e.g. `OLLAMA_KEEP_ALIVE=24h` for "always
  loaded" or `0` for "swap on demand"). The bot doesn't need to care.
"""

from __future__ import annotations

from typing import Optional

# Lazy import so the bots package can be imported on systems without
# `ollama` installed (e.g. a Colab session before pip install runs).
try:
    import ollama
except ImportError:
    ollama = None    # noqa: PLW0603 — sentinel for the constructor

from .base import BaseBot, Personality


class OllamaBot(BaseBot):
    """A poker bot backed by a local Ollama-served LLM.

    Args:
        name:        Display name on the poker table.
        personality: Personality preset (drives the system prompt).
        model_id:    Ollama model identifier. Must already be `ollama
                     pull`-ed; matches what `ollama list` reports.
        host:        Ollama daemon URL. Default localhost:11434.
        temperature: Sampling randomness. Default 0.7.
        num_predict: Max tokens per response. Default 250 — enough
                     for a JSON action + a few sentences of reasoning.
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_NUM_PREDICT = 250

    def __init__(
        self,
        name: str,
        personality: Personality,
        model_id: str,
        *,
        host: str = DEFAULT_HOST,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ) -> None:
        if ollama is None:
            raise ImportError(
                "The 'ollama' Python package is not installed. "
                "Run `pip install ollama` (or `pip install -r requirements.txt`)."
            )
        super().__init__(name=name, personality=personality, model_id=model_id)
        # Per-bot client — fine for the ~tens-of-bots scale we operate at.
        self._client = ollama.Client(host=host)
        self._temperature = temperature
        self._num_predict = num_predict

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send (system, user) to the Ollama daemon and return the raw
        text. Errors propagate; the inherited `_safe_generate` wrapper
        handles them."""
        response = self._client.chat(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        )
        # The ollama Python client returns a dict-compatible object
        # whose ['message']['content'] holds the assistant's text.
        # Index-style access works across recent versions.
        return response["message"]["content"]


# ----------------------------------------------------------------------
# Smoke test — `python -m bots.ollama_bot`
#
# Two phases:
#   1. Connectivity check: one direct call to verify the daemon is up
#      and the model responds.
#   2. Mini-tournament: 1 OllamaBot vs 3 MockBots, 5 hands, validating
#      that the bot integrates cleanly with the engine + tracker.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import sys
    import tempfile
    import time
    from pathlib import Path

    from .base import MockBot
    # A couple of pre-defined personalities for the test bots.
    from .personalities import TIGHT_AGGRESSIVE, CALLING_STATION

    MODEL = "llama3.1:8b"

    # ------------------------------------------------------------------
    # Phase 1 — connectivity
    # ------------------------------------------------------------------
    print(f"=== Phase 1: connectivity check against '{MODEL}' ===")
    try:
        bot = OllamaBot(
            name="Probe",
            personality=TIGHT_AGGRESSIVE,
            model_id=MODEL,
            num_predict=40,        # short, this is just a ping
        )
        t0 = time.time()
        text = bot._generate(
            system_prompt="You are a terse assistant.",
            user_prompt="Say hello in 5 words or fewer.",
        )
        dt = time.time() - t0
        print(f"  Response ({dt:.2f}s): {text.strip()!r}")
        if not text.strip():
            print("  WARNING: empty response. Check that the model is actually pulled.")
            sys.exit(1)
    except Exception as e:        # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")
        print("  Hints:")
        print("    - Is the Ollama daemon running? Try `ollama list` from another shell.")
        print(f"    - Is the model pulled? Try `ollama pull {MODEL}`.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2 — mini tournament with 1 OllamaBot vs 3 MockBots
    # ------------------------------------------------------------------
    print(f"\n=== Phase 2: 5-hand tournament with 1 {MODEL} bot vs 3 MockBots ===")

    # Imports here so a Phase 1 failure doesn't pull in the rest.
    from runner import RunnerConfig, TournamentRunner

    # Canned MockBot responses — passive opponents so the OllamaBot
    # gets to drive the action.
    caller_resp = '{"reasoning": "Just calling.", "action": "call"}'
    fold_resp   = '{"reasoning": "Not worth it.", "action": "fold"}'
    stand_pat   = '{"reasoning": "Standing pat.", "discards": []}'

    bots = [
        OllamaBot(
            name="Llama-Aggro",
            personality=TIGHT_AGGRESSIVE,
            model_id=MODEL,
        ),
        MockBot("MockA", CALLING_STATION,   caller_resp, stand_pat, model_id="mock-A"),
        MockBot("MockB", CALLING_STATION,   caller_resp, stand_pat, model_id="mock-B"),
        MockBot("MockC", TIGHT_AGGRESSIVE,  fold_resp,   stand_pat, model_id="mock-C"),
    ]

    runs_root = Path(tempfile.mkdtemp(prefix="poker_ollama_smoke_"))
    cfg = RunnerConfig(
        num_hands=5,
        starting_chips=200,
        ante=2,
        min_bet=5,
        broke_player_policy="rebuy",
        seed=42,
        verbose=True,
        sessions_root=str(runs_root),
        session_id="ollama_smoke",
    )
    t0 = time.time()
    result = TournamentRunner(bots, cfg).run()
    dt = time.time() - t0
    print(f"\n  Tournament finished in {dt:.1f}s ({dt / cfg.num_hands:.1f}s/hand)")
    print(result.summary_table())

    # ------------------------------------------------------------------
    # Inspect what the LLM actually said.
    # ------------------------------------------------------------------
    from tracker import load_reasoning
    reasoning_rows = load_reasoning(result.session_dir)
    llama_rows = [r for r in reasoning_rows if r["player"] == "Llama-Aggro"]
    print(f"\n  Llama-Aggro made {len(llama_rows)} LLM calls.")
    parse_errors = [r for r in llama_rows if r["parse_error"]]
    print(f"  Parse-error rate: {len(parse_errors)}/{len(llama_rows)} "
          f"({100 * len(parse_errors) / max(1, len(llama_rows)):.0f}%)")
    if llama_rows:
        sample = llama_rows[0]
        print(f"\n  Sample reasoning (hand {sample['hand_id']}, "
              f"{sample['phase']}, {sample['decision_type']}):")
        print(f"    decision : {sample['decided_action'] or sample['decided_discards']}")
        print(f"    reasoning: {sample['reasoning']!r}")
        if sample["parse_error"]:
            print(f"    NOTE: parse_error = {sample['parse_error']!r}")
            print(f"    raw      : {sample['raw_response'][:200]!r}")

    shutil.rmtree(runs_root)
    print("\nSmoke test finished — review the output above to confirm "
          "responses look like real LLM-flavoured reasoning.")
