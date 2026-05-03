# Poker McPokerface

Evaluating LLM agents in 5-card draw poker. Seven open-weight LLMs (1B to 14B parameters, six families) are paired with five distinct play-style personalities to produce 35 (model × personality) bot configurations, then run through a multi-table round-robin tournament. The data answers three research questions: which (LLM × personality) combination performs best, which LLM averages best across personalities, and which personality averages best across LLMs.

This is a course submission for COMP41830: Advanced Language Models.

## Quick start

The canonical entry point is the submission notebook. Open it in Google Colab:

[`notebooks/submission.ipynb`](notebooks/submission.ipynb)

Run all cells top to bottom. The first run takes ~15 minutes because Ollama and several model weights have to download; subsequent cells are fast. Total assessor runtime on a free Colab T4 is roughly **45 minutes** end to end.

For a local run instead of Colab, open it in Jupyter from the project root:

```bash
jupyter notebook notebooks/submission.ipynb
```

The full report (PDF) is in [`Report/COMP41830 Assignment Report.pdf`](Report/).

## What's in this repo

```
Poker/
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
│   ├── knockout_bracket.py         Single-elimination bracket script (built and validated;
│                                   scoped out of reported findings due to compute time —
│                                   see §7.5 of the report).
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
│   ├── main_round_robin_v1/        Earlier run; kept for the v1 → v2 fix-arc evidence.
│   └── (validation_*)              Pre-flight test runs documenting the v1 → v2 fix arc.
│
├── figs/                Project-level diagrams (architecture, data model, etc.).
│
├── Report/              Final submission PDF.
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Key results headline

From the canonical v2 round-robin (1500 hands across 30 tables, all 7 models contributing valid responses):

- **Best LLM:** Qwen3 8B (with `think=False`) at **+46.36 mean chips per hand**, with Llama 3.1 8B a distant second at +10.00. Notably, Qwen3 14B underperforms Qwen3 8B at -5.67 — same family, almost double the parameters, worse poker. "Bigger is better" is falsified within this comparison.
- **Best personality:** Tight-aggressive at **+33.25 mean chips per hand** — the textbook winning style poker theory predicts. The bluffer is worst at -29.06 (raises with weak hands; gets called and punished). The calling station, second-best at +18.95, is the only result that bucks classical theory — it works only because the opponent pool contains many bluffers it can collect from.
- **Best (model × personality) combo:** Qwen3 8B + tight-aggressive at **+148.21 chips per hand** on a 200-hand sample, with Qwen3 8B + calling-station second at +120.26.

The v1 → v2 fix arc (three of seven models silenced in v1 by misconfigured token budgets and reasoning-mode toggles) is itself a methodology contribution and is documented in §9 of the report.

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

# 3. Run the round-robin (takes ~17 hours on a 32 GB GPU)
export OLLAMA_KEEP_ALIVE=1h
python -u scripts/round_robin.py 2>&1 | tee runs/main_round_robin_v2.log

# 4. Analyse the results
python scripts/analyse_round_robin.py main_round_robin_v2

# 5. (Optional) Run the knockout bracket on the top 8 — scoped out of the
#     submitted report for compute reasons, but the script is here if you
#     want to crown a champion (~6 hours additional):
# python -u scripts/knockout_bracket.py main_round_robin_v2
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
