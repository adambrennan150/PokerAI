# Poker McPokerface

Evaluating LLM agents in 5-card draw poker. Seven open-weight LLMs (1B to 14B parameters, six families) are paired with five distinct play-style personalities to produce 35 (model × personality) bot configurations, then run through a round-robin tournament and a single-elimination knockout bracket. The data answers three research questions: which (LLM × personality) combination performs best, which LLM averages best across personalities, and which personality averages best across LLMs.

This is a course submission for COMP41830: Advanced Language Models.

## Quick start

The canonical entry point is the submission notebook. Open it in Google Colab:

[`notebooks/submission.ipynb`](notebooks/submission.ipynb)

Run all cells top to bottom. The first run takes ~15 minutes because Ollama and several model weights have to download; subsequent cells are fast. Total assessor runtime on a free Colab T4 is roughly **45 minutes** end to end.

For a local run instead of Colab, open it in Jupyter from the project root:

```bash
jupyter notebook notebooks/submission.ipynb
```

## What's in this repo

```
PokerAI/
├── engine/             Pure-Python game logic (deck, hand evaluator, player, game state machine).
│                       Zero dependencies on UI or LLM code; everything else imports from here.
│
├── bots/               Decision-making layer.
│   ├── base.py         Abstract BaseBot — prompt formatting, JSON parsing, fallback handling.
│   ├── ollama_bot.py   Concrete LLM bot that talks to a local Ollama daemon.
│   ├── personalities.py  Five personality presets (TIGHT_AGGRESSIVE, LOOSE_AGGRESSIVE,
│                         ROCK, CALLING_STATION, BLUFFER) — pure data.
│
├── tracker/            Append-only JSONL logging. Crash-safe, pandas-friendly.
│
├── runner/             Multi-hand tournament orchestration.
│
├── ui/                 Notebook + terminal rendering. HumanAgent for human-vs-bot play.
│
├── config/
│   └── models.py       Canonical roster of LLMs with per-model overrides
│                       (num_predict, system_prefix, think parameter).
│
├── scripts/
│   ├── round_robin.py              Main multi-table round-robin tournament driver.
│   ├── knockout_bracket.py         Single-elimination bracket on the top 8 round-robin combos.
│   ├── analyse_round_robin.py      Reusable analytics — five tables + five figures.
│   ├── validation_*.py             Various pre-flight validation scripts.
│   ├── pull_models.py              Idempotent `ollama pull` of every model in the roster.
│   └── inspect_parse_errors.py     Drill-down on LLM parse failures.
│
├── notebooks/
│   ├── submission.ipynb            ★ The submission entry point.
│   ├── game.ipynb                  Human-vs-bots interactive play.
│   ├── bot_arena.ipynb             Automated bot tournament setup.
│   └── analytics.ipynb             Exploratory data analysis on session output.
│
├── runs/
│   ├── main_round_robin_v2/        Canonical round-robin results (1500 hands × 35 bots).
│   ├── knockout_bracket_v1/        Knockout bracket from the top 8 round-robin combos.
│   └── (validation_*)              Pre-flight test runs documenting the v1 → v2 fix arc.
│
├── Instructions/
│   ├── Poker Brief.pdf             The course brief.
│   ├── Project Report template 2026.pdf  The required report structure.
│   ├── report_draft.md             The report itself, in markdown.
│   └── (supporting materials)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Key results headline

<span style="color:gray">[Numbers below are from v1 round-robin; v2 with the fix in place is in progress / canonical.]</span>

- **Best LLM:** Llama 3.1 8B at +8.87 mean chips per hand, with Phi-4 Mini second at +6.87. Phi-4 Mini punches well above its 3.8B weight.
- **Best personality:** Loose-aggressive at +8.42 mean chips per hand. Calling station is worst at -8.74, exactly as poker theory predicts (passive callers bleed chips against aggressive opponents).
- **Best (model × personality) combo:** Phi-4 Mini + loose-aggressive at +121 chips per hand on the v1 sample (small N — see report for caveats).

The full report is [`Instructions/report_draft.md`](Instructions/report_draft.md) (PDF version submitted alongside this repo).

## Running the experiments yourself

If you want to reproduce the local round-robin (requires a GPU with at least ~28 GB VRAM):

```bash
# 1. Set up Python environment
python -m venv .venv
source .venv/bin/activate            # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Install Ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download/windows
# Then pull the model roster:
python scripts/pull_models.py

# 3. Run the round-robin (takes ~17 hours)
export OLLAMA_KEEP_ALIVE=1h
python -u scripts/round_robin.py 2>&1 | tee runs/main_round_robin_v2.log

# 4. Run the knockout bracket on the top 8 (~6 hours)
python -u scripts/knockout_bracket.py main_round_robin_v2 \
    2>&1 | tee runs/knockout_bracket_v1.log

# 5. Analyse the results
python scripts/analyse_round_robin.py main_round_robin_v2
```

## Methodology note: the v1 → v2 fix arc

The first round-robin run (v1) revealed that three of seven models — DeepSeek-R1 7B and both Qwen3 variants — returned essentially-empty responses on >99% of LLM calls. Their reasoning preambles consumed the configured token budget before any JSON was produced, leaving the parser to fall back to a safe default action on every turn.

The fix has two parts:
- **Per-model `num_predict` overrides** in `config/models.py`: 4096 tokens for DeepSeek-R1 (whose `<think>` block cannot be disabled by user prompt), 1024 for Qwen3.
- **`think=False` Ollama API parameter** for the Qwen3 family (the canonical way to disable Qwen3's reasoning preamble; the prompt-level `/no_think` directive proved unreliable in isolation).

A targeted validation run (`scripts/validation_reasoning_models.py`, results in `runs/validation_reasoning_models_v1/`) confirmed the fix works on all three models. The full v2 round-robin uses these overrides.

This methodology arc is documented in §9 of the report. It's a useful artifact in its own right — when evaluating reasoning-style LLMs in agentic frameworks, token budgets and thinking-mode toggles must be set per-model rather than globally.

## AI use disclosure

This project was developed in collaboration with Anthropic's Claude (via the Cowork mode of the Claude Desktop app). See §9 of the report for the full disclosure. Approximately 70% of the codebase by line count originated from AI-generated suggestions, reviewed and edited before commit.

## Licence

This is academic coursework. No formal licence is attached.
