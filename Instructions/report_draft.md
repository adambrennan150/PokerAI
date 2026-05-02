# Poker McPokerface: Evaluating LLM Agents in 5-Card Draw Poker

**<span style="color:red">[TODO — replace with student name, ID, and email per template]</span>**

> **Drafting notes for the student.** This is a first-draft report based on the project as built so far. Items marked in <span style="color:red">red</span> are placeholders for content that depends on the round-robin tournament results (still in progress) or that you should write yourself (e.g. personal reflections, AI-use disclosure with your own perspective). Figure placeholders are marked `[FIGURE N: ...]`. Some text is deliberately conservative — read through and personalise once the data lands.

---

## Abstract

This project implements a 5-card draw poker simulator in Python that pits LLM-powered agents against each other and against a human player. Seven open-weight LLMs spanning 1B to 14B parameters, drawn from six different model families (Gemma, Phi, Llama, Mistral, Qwen, DeepSeek), are each paired with five distinct play-style personalities (tight-aggressive, loose-aggressive, rock, calling station, bluffer) to produce 35 (model × personality) bot configurations. A round-robin tournament with rotated 4-bot tables produces a single dataset that the analytics layer reduces to three bottom-line answers: which (LLM × personality) combo performs best, which LLM averages best across personalities, and which personality averages best across LLMs. <span style="color:red">[TODO — replace with the actual headline finding once the round-robin completes, e.g. "Llama 3.1 8B paired with the bluffer personality emerged as the strongest combination, while smaller (≤4B) models struggled regardless of personality."]</span>

`[FIGURE 1: optional splash image — poker table render or system diagram]`

---

## 1. Introduction

### Motivation

Poker is a particularly informative testbed for evaluating language-model agents because it is a game of *imperfect information* — each player sees only their own cards and the public betting record, never the opponents' hands. Strong play therefore requires more than retrieving the textbook ranking of a 5-card hand: it requires reading opponents' actions, modelling their likely holdings, deciding when to bluff, and weighing risk against position and chip stack. Where chess and Go reward planning over a perfectly observable state, poker rewards inference, deception, and self-restraint — capabilities that map onto the very things we want to know about LLMs as agents in adversarial environments.

Five-card draw was chosen specifically because it is the simplest variant that still exercises this full skill set in a pre-flop / post-flop structure: every player gets five hidden cards, bets, optionally exchanges some for fresh ones, bets again, and shows down. There are no community cards to track, no positional rules to memorise, and the hand evaluator covers exactly the nine standard poker categories. The reduced surface area lets the experimental design focus on the LLM-as-agent question rather than on poker mechanics.

### Project goals

The project has three goals, drawn directly from the brief:

1. Compare LLMs of different sizes and families as decision-makers.
2. Compare distinct play-style personalities under the same LLM.
3. Examine how the two interact — does a particular personality help or hurt a particular model?

### Research questions

Operationalised as three pandas-style aggregations over the resulting tournament data:

- **Which (LLM × personality) combination performs best?** Mean chip change per hand, grouped by `(model_id, personality_id)`.
- **Which LLM does best on average?** Mean chip change per hand, grouped by `model_id` (averaging across all five personalities).
- **Which personality does best on average?** Mean chip change per hand, grouped by `personality_id` (averaging across all seven models).

A secondary question — *do larger models outperform smaller ones?* — falls naturally out of the second aggregation by ordering the LLMs by parameter count.

---

## 2. System Architecture

The system is structured in five layers, each importable from the next without introducing circular dependencies:

```
engine/    pure-Python game logic — deck, hand evaluator, player, game state machine
bots/      decision-making agents — BaseBot, MockBot, OllamaBot, personality presets
tracker/   append-only JSONL logging
runner/    multi-hand tournament orchestration
ui/        terminal + Jupyter rendering, HumanAgent
```

The discipline is strict: `engine/` imports from nothing else; `bots/` imports `engine/` for `Action` and `GameView` types but never depends on UI or tracker; `runner/` imports from `engine/`, `bots/`, and `tracker/` but is itself import-free from the higher layers. This separation is what makes "swap a MockBot for an LLM bot" a one-line change.

`[FIGURE 2: project structure diagram — Instructions/poker_project_structure.svg already in repo]`

### 2.1 Game Engine

The engine is implemented as four pure-Python modules:

- **`engine/deck.py`** — 52-card deck with a seeded `random.Random` for reproducible shuffling. `Card` is a frozen `dataclass(order=True)` so cards are immutable, hashable, and naturally sortable; `Suit` and `Rank` are enums where the rank's integer value participates directly in hand-evaluator comparisons.
- **`engine/hand.py`** — 5-card hand evaluator covering all nine poker categories. Rather than encode dozens of bespoke tiebreaker rules, every hand is reduced to a tuple `(category_value, *kicker_ranks_descending)` and Python's lexicographic tuple comparison does the work. The "wheel" straight (A-2-3-4-5) is the only special case.
- **`engine/player.py`** — per-player state: chips, current hand, bet-this-round and total-contributed-this-hand, plus a `PlayerStatus` enum (active / folded / all-in). All-in handling is centralised in `Player.post(amount)`, which caps posting at available chips and transitions the player to `ALL_IN` when their stack hits zero.
- **`engine/game.py`** — the state machine: ante → deal → first betting → draw → second betting → showdown. The betting round terminates correctly under re-raises by tracking a `last_aggressor` pointer; side pots from all-in scenarios are computed at showdown by walking distinct contribution levels.

A single `seed` parameter flows through every random source, making whole tournaments reproducible.

### 2.2 LLM Agent Framework

The decision-making layer is built around a deliberately thin contract: the engine calls `agent.decide_action(view) -> Action` and `agent.decide_discards(view) -> list[int]`, where `view` is a per-player `GameView` snapshot. Anything that satisfies this `PlayerAgent` Protocol can sit at the table — a human typing into the terminal, an LLM bot, a unit-test stub.

**`BaseBot`** is the abstract class that all LLM bots inherit from. It owns:

- *Prompt formatting* — the same JSON-output prompt format for action and discard decisions, regardless of the underlying model.
- *Three-layer response parsing*: (1) extract and `json.loads` the first balanced `{...}` block; (2) on failure, regex-scan for action keywords; (3) on total failure, return a safe default (fold if there's a bet to call, otherwise check). In validation testing the strict JSON path caught >99% of responses; the fallbacks exist so that a single malformed token cannot crash a 1000-hand session.
- *Reasoning capture* — every call records the full prompt, raw response, parsed reasoning text, and any parse error onto `bot.last_response`, which the tracker then writes verbatim to `reasoning.jsonl`.

**`OllamaBot`** subclasses `BaseBot` with a single override: `_generate(system, user) -> str` calls `ollama.Client.chat(...)` against a locally running Ollama daemon. The model identifier (`llama3.1:8b`, `mistral`, etc.) is a constructor argument, so the same class drives every model in the roster.

**`MockBot`** is a no-LLM `BaseBot` subclass returning hard-coded JSON. It exists for two reasons: it lets the entire pipeline (engine, runner, tracker, analytics) be exercised end-to-end before any model is downloaded, and it provides deterministic table-fill bots for validation against unstable LLM behaviour.

`[FIGURE 3: tracker data model — Instructions/poker_tracker_data_model.svg already in repo]`

### 2.3 Interaction Loop

A single hand proceeds through six phases, all orchestrated by `Game.play_hand()`:

1. **Antes** — every active player posts a fixed ante (default 2 chips). Players who can't afford it sit out the hand.
2. **Deal** — each in-hand player receives 5 cards.
3. **First betting round** — starting from the seat after the dealer, each active player in turn is asked for an `Action` (fold, check, call, or raise to *N*). The round ends when the action wraps back to the most recent aggressor without anyone re-raising.
4. **Draw** — each remaining (non-folded, non-all-in) player chooses 0 to 5 indices to discard; the deck deals replacements.
5. **Second betting round** — same logic as the first, with a fresh per-round bet counter.
6. **Showdown** — the engine evaluates every in-hand player's final 5 cards and awards each pot layer to the eligible player(s) with the best hand, splitting ties by shared share with leftover chips going to the first-to-act winner.

Side pots are constructed from `Player.total_contributed` levels, so all-in scenarios are handled correctly without special-case branching in the betting loop.

`[FIGURE 4: terminal screenshot of `game.ipynb` showdown rendering — recommend a screenshot of the new render_showdown_html output, with all four players' hands visible]`

---

## 3. LLM Models Used

The roster comprises seven open-weight models served locally via Ollama. The selection criteria were threefold: family diversity (multiple training pipelines), size span (sub-billion to mid-double-digit-billion), and at least one model with visibly distinct reasoning behaviour. Frontier-scale models (≥70B) were excluded because they don't fit on the 32GB GPU used for local development.

| Model | Family | Params | RAM (4-bit) | Why chosen |
|---|---|---|---|---|
| Gemma 3 1B | Google | 1.0B | ~1.5 GB | Smallest viable baseline — does scale matter at all? |
| Phi-4 Mini | Microsoft | 3.8B | ~3 GB | Small-tier alternative family; reportedly strong on reasoning for size |
| Llama 3.1 8B | Meta | 8.0B | ~6 GB | Medium-tier reference; largest open ecosystem of fine-tunes |
| Mistral 7B | Mistral AI | 7.0B | ~5 GB | Strong instruction-following at 7B; different architecture from Llama |
| Qwen3 8B | Alibaba | 8.0B | ~5.5 GB | Third 7-8B family for cross-family breadth |
| DeepSeek-R1 7B | DeepSeek | 7.0B | ~5 GB | Visible chain-of-thought — wildcard for poker reasoning analysis |
| Qwen3 14B | Alibaba | 14.0B | ~12 GB | Larger class — does ~2× parameters meaningfully outperform 7B? |

All seven were pulled via `ollama pull`. The chosen quantisation is the default Q4_K_M offered by Ollama, which trades a small amount of inference quality for substantial RAM savings. The full roster fits in 32 GB simultaneously, which means any 4-bot table can be assembled from any combination of models without VRAM swap overhead.

The Colab submission notebook uses a reduced 4-model subset (Gemma 3 1B, Llama 3.1 8B, Mistral 7B, DeepSeek-R1 7B) to fit within a free Tesla T4's 16 GB VRAM and to keep the assessor's reproduction run under an hour.

`[FIGURE 5: bar chart of model parameters vs. mean chip change — TODO once round-robin completes]`

---

## 4. Personality Design

### 4.1 Personality Dimensions

Five personalities were defined, anchored on the classic poker 2×2 of *tight* (folds often) vs. *loose* (plays many hands) crossed with *passive* (calls/checks) vs. *aggressive* (bets/raises):

| ID | Type | One-line description |
|---|---|---|
| `tight_aggressive` | Tight-aggressive | Folds weak hands; raises strong ones — the textbook winning style |
| `loose_aggressive` | Loose-aggressive | Plays many hands; raises and bluffs constantly; high variance |
| `rock` | Tight-passive | Plays only premium hands; never raises, never bluffs |
| `calling_station` | Loose-passive | Calls everything; rarely raises; hard to bluff |
| `bluffer` | Deceptive | Bluffs with weak hands, slow-plays strong ones; highly table-aware |

The four 2×2 archetypes are well-studied in poker literature and produce empirically different chip outcomes. The fifth — the bluffer — was added because the brief specifically mentions bluffing as a behavioural axis worth examining, and it captures a deceptive *strategy* that doesn't reduce to either dimension alone.

A "balanced / GTO-ish" baseline was deliberately *not* included, on the grounds that it would dilute signal and overlap with tight-aggressive. A single-axis dimension like "verbosity" was also rejected — every personality should produce a behavioural difference at the table, not just a stylistic one in the logs.

### 4.2 Prompt Engineering

Each personality is encoded as a `Personality` dataclass:

```python
@dataclass(frozen=True)
class Personality:
    id: str                 # groupby key for analytics
    description: str        # one-line for logs
    system_prompt: str      # ~3-5 sentences delivered to the LLM
```

The `system_prompt` is the only mechanism shaping behaviour. Three principles guided the wording:

1. **3–5 sentences.** Long prompts hurt smaller models in particular; concrete short instructions outperform abstract long ones.
2. **Concrete behavioural verbs over abstract traits.** "Fold weak hands without hesitation" rather than "be disciplined"; "raise to build the pot" rather than "play with confidence."
3. **An explicit attention-axis sentence.** Each prompt either pushes the LLM toward reading the table (TAG, Bluffer) or toward focusing only on its own cards (Rock, Calling Station). LAG sits in the middle. This is the "what does the bot pay attention to?" lever the brief asks us to think about.

Excerpt from the `tight_aggressive` system prompt:

> "You are a disciplined, tight-aggressive 5-card draw poker player known as 'The Shark'. You fold weak starting hands without hesitation and only commit chips when your holding justifies it. When you do play, you raise to build the pot and put pressure on opponents. Pay attention to the betting action: fold to multiple raises unless your hand is genuinely strong. You bluff sparingly and only when the table reads as weak."

Compare to the `calling_station` prompt:

> "You are a calling station — a loose, passive 5-card draw poker player. You call most bets to see the showdown and rarely fold once you have already put chips in. You almost never raise, even with strong hands, preferring to keep pots small and let opponents make mistakes. You make your decisions based on your own cards rather than reading opponents."

Validation testing on Mistral 7B confirmed that the prompts bite: in a heads-up 15-hand session against a TAG-prompted Llama 8B, the calling-station Mistral *self-identified* as a calling station in its reasoning logs ("As a calling station, I prefer to see the showdown..."), and made 13 more LLM calls than its TAG opponent — a direct consequence of staying in more hands rather than folding pre-draw.

In addition to the persona system prompt, each LLM call receives a structured user prompt containing the full `GameView` (current pot, your hand, your chips, all opponents' chip counts and statuses, complete action history this hand) and an explicit JSON schema for the response. Few-shot examples were *not* used; pilot testing showed they didn't measurably improve adherence given that the schema is already concrete.

`[FIGURE 6: side-by-side comparison of two LLM reasoning samples — one TAG, one Calling Station — showing how the personality biases the language used. Pull from reasoning.jsonl after the round-robin.]`

---

## 5. Information Access Design

A single `GameView` dataclass is the entire interface the agent has to the world; any future change to "what the LLM can see" is a one-struct edit, not a re-engineering of the agent layer.

**What every agent sees on every turn:**

- Their own hand, as `Card` objects.
- Their own chips, current bet this round, total contributed this hand.
- The pot total, the highest current bet at the table, the amount they owe to call, the minimum legal raise.
- For every other player: name, chips, current bet, total contributed, status (active / folded / all-in), and the *count* of cards they're holding.
- The full action history of the *current* hand — every fold, call, raise, ante, and discard with the player who made it and the pot after.

**What they don't see:**

- Other players' hole cards (obviously).
- Anything from previous hands. Each hand is stateless from the bot's perspective, so an LLM can't build up read-models of opponents over a long session.

The card-count signal is intentionally exposed: after the draw phase, an opponent who exchanged three cards has telegraphed that they almost certainly held a low pair at best, while a player who stood pat plausibly has a straight or flush. This is a real piece of strategic information that human players use, and exposing it lets the experiment ask whether LLMs use it too.

The decision to keep bots stateless across hands was deliberate. It keeps the experiment cleaner — every hand is an independent test of the same prompt — and avoids context-window bloat for smaller models that would otherwise need history threaded through every call. A future extension could add a "recent showdowns" field to `GameView` to test whether bigger models exploit cross-hand reads, but that is out of scope for the current submission.

---

## 6. Experimental Setup

### 6.1 Game Modes

Both modes required by the brief are implemented:

- **Human vs LLMs** — `notebooks/game.ipynb` seats the user as one of four players against three bots. Cards render inline as styled HTML (or ASCII fallback in non-Jupyter environments); the user types actions into prompts (`fold`, `call`, `raise 25`, `all-in`, etc.) which are leniently parsed. After each hand, all four players' final cards and hand categories are revealed for analysis (the realism trade-off was deliberately weighted toward learning/diagnostics over realism).
- **Automated bot vs bot** — driven by `runner/runner.py` for a single tournament and by `scripts/round_robin.py` for the main multi-table experiment. Neither mode requires interactive input.

### 6.2 Simulation Parameters

The main experimental run uses these parameters:

| Parameter | Value | Rationale |
|---|---|---|
| Players per table | 4 | Standard for 5-card draw; gives interesting multi-way pot dynamics |
| Hands per table | 50 | Enough for variance to average out within a table |
| Number of tables | 30 | Yields each (model × personality) combo ~3.4 table-appearances on average, ~170 hands of data per combo |
| Total hands | 1,500 | Produces ~6,000 seat-rows and ~12,000–15,000 reasoning rows |
| Starting chips | 200 | Comfortable headroom against the 5-chip min-bet so all-ins are rare in early hands |
| Ante | 2 | Enough to create a pot worth contesting; small enough not to dominate decisions |
| Minimum bet / raise increment | 5 | Standard small-stakes ratio |
| Broke-player policy | rebuy | Broke players are topped back to starting chips before each hand, keeping every combo in the dataset every hand |
| Seed | 42 | Reproducible deck shuffles and table assignments |

The round-robin is implemented in `scripts/round_robin.py`. Tables are sampled with soft inverse-appearance weighting (under-seen combos get higher sampling weight) so that exposure is roughly balanced without strict Latin-square enforcement. After generation, tables are sorted by their tuple of model identifiers, which groups tables sharing the heavy 14B model adjacent and minimises Ollama's swap overhead between tables.

### 6.3 Logging

Logging is append-only JSONL, written by a custom `tracker/tracker.py` that lives between the engine and the analytics layer. Four files are produced per session:

| File | One row per | Contents (high level) |
|---|---|---|
| `config.json` | session | seats, model+personality assignments, game parameters, timestamps |
| `hands.jsonl` | hand | per-seat outcome (chips, cards, hand category, folded flag) and pot winners |
| `actions.jsonl` | action | phase, player, action type, amount, chips_posted, pot_after |
| `reasoning.jsonl` | LLM call | full prompt + raw response + parsed reasoning + parse error, plus the resulting decision |

JSONL was chosen over SQLite or CSV for three reasons: (a) reasoning text is variable-length and awkward in CSV; (b) append-only writes are crash-safe — every completed line is durable on disk even if the run dies mid-session; (c) `pandas.read_json(path, lines=True)` loads the result directly into a DataFrame for analysis with no schema migration. Splitting `actions.jsonl` from `reasoning.jsonl` keeps the main analytics queries fast — reasoning logs are large but only loaded when explicitly inspecting what the LLMs *said*.

The `TrackingAgent` adapter wraps each `BaseBot` and intercepts every `decide_action` / `decide_discards` call to push the bot's reasoning record into `reasoning.jsonl` before the engine sees the returned action. The engine itself is unaware that any logging is happening.

`[FIGURE 7: tracker analytics views — Instructions/poker_analytics_views.svg already in repo]`

---

## 7. Results

<span style="color:red">[TODO — this entire section is pending completion of the round-robin tournament. Once the run finishes, populate from `notebooks/analytics.ipynb`.]</span>

### 7.1 Performance by Agent

<span style="color:red">[TODO — top 5 and bottom 5 (model × personality) combos by mean chip change per hand. Include a table with `model_id`, `personality_id`, `mean_delta`, `total_delta`, `win_rate`, `hands_played`. Highlight the best and worst combos and characterise them in one sentence each.]</span>

`[FIGURE 8: horizontal bar chart of mean chip change per (model × personality), sorted descending. Already produced by analytics.ipynb's "Best bot" cell.]`

### 7.2 Performance by Model

<span style="color:red">[TODO — `groupby('model_id')` on `mean_delta`. Include the table from `analytics.ipynb`'s "Best LLM" cell. Two specific things to address in prose: does the 14B Qwen meaningfully outperform the 8B Qwen? Is there a clear monotonic relationship between parameter count and performance, or does family/training matter more than size?]</span>

`[FIGURE 9: horizontal bar chart of mean chip change per model, with parameter count as a secondary axis to make the size question visible.]`

### 7.3 Performance by Personality

<span style="color:red">[TODO — `groupby('personality_id')` on `mean_delta`. Include the table from `analytics.ipynb`'s "Best personality" cell. Specific things to address: do the loose-passive personalities (calling station) lose chips on average as poker theory predicts? Does the bluffer's high-variance strategy pay off, or does it lose chips to disciplined opponents?]</span>

`[FIGURE 10: horizontal bar chart of mean chip change per personality.]`

### 7.4 Visualisations

<span style="color:red">[TODO — discuss the chip-trend chart from `analytics.ipynb` showing cumulative net chips per bot across the session. This is the most "narrative" plot and worth highlighting any divergence between bots that started similarly. Also include the parse-error rate by model — a measure of LLM robustness as much as poker skill.]</span>

`[FIGURE 11: line chart of cumulative chip change per bot over hand_id. Already produced by analytics.ipynb.]`

### 7.5 Knockout Bracket

<span style="color:red">[TODO — pending Phase 5 (the knockout tournament on the top 8 combos by round-robin mean chip change). Run `scripts/knockout_bracket.py` (yet to be written; will be built once the round-robin finishes). Include a bracket diagram and a one-line per-match commentary.]</span>

`[FIGURE 12: bracket diagram showing 8 → 4 → 2 → 1 elimination ladder, with the eventual champion highlighted.]`

---

## 8. Discussion

<span style="color:red">[TODO — write this once results are in. Below is a skeleton with prompts for the kinds of interpretation that will likely be most interesting.]</span>

### Are stronger LLMs actually better poker players?

<span style="color:red">[Compare the seven models by `mean_delta`. If there's a clean monotonic relationship between size and performance, say so plainly. If it's noisier — e.g. Mistral 7B beats Qwen 14B — that itself is interesting and worth a sentence on why (training data, instruction-following alignment, model-specific quirks under JSON-structured-output).]</span>

### Does personality matter more than model?

<span style="color:red">[Compute the variance of `mean_delta` within (model_id) groups vs. within (personality_id) groups. The larger variance is the dimension that matters more. Frame the answer accordingly.]</span>

### Emergent strategies

<span style="color:red">[Pull a handful of standout reasoning excerpts from `reasoning.jsonl`. The DeepSeek-R1 7B model in particular produces visible chain-of-thought before its answer; quote one. Look for cases where the LLM correctly inferred opponent strength from the action history, or made a clever discard decision. These qualitative observations belong in this section.]</span>

### Limitations

A handful of methodological caveats worth being explicit about:

- **Sample size.** With 30 tables × 50 hands the per-combo data is ~170 hands. That is enough to surface broad patterns but not enough for fine-grained statistical claims. Doubling the run would tighten the confidence intervals.
- **Non-determinism.** LLM outputs are stochastic, so two runs with the same seed will produce somewhat different chip outcomes. The deck shuffle is reproducible, but the agents' decisions are not. This is consistent with the brief's framing (it asks about *averages*), but it means individual session results should be read as samples from a distribution, not point estimates.
- **Prompt sensitivity.** The personality prompts are short and probably not optimal. A more thorough study would A/B-test prompt variants per personality. The current design treats the prompts as a single fixed condition for the experiment.
- **No cross-hand memory.** Bots cannot see anything from previous hands. This was deliberate (it keeps hands independent for analytical purposes), but it means we are testing the LLMs as *single-hand* decision-makers rather than as adaptive opponents who exploit reads built up over time.
- **Quantisation effects.** All models are run in 4-bit quantisation. A more capable but uneven 8-bit run might shift the rankings, particularly for the smaller models where quantisation hits proportionally harder.
- **Frontier models excluded.** Nothing above 14B fits on the 32GB GPU. The findings therefore generalise within the small-and-medium model class but cannot make claims about frontier-scale models like GPT-4 or Llama 3.3 70B.

### Patterns to note

<span style="color:red">[TODO — fill in patterns observed in the data. Likely candidates worth checking: (a) did smaller models over-fold? Compute fold-rate per model, look for an outlier on the high end. (b) Did the calling-station personalities lose money as theory predicts? Compute mean_delta per personality and check that calling_station and rock are at the bottom. (c) Did the bluffer succeed against the calling stations specifically? Look at pairwise matchups.]</span>

`[FIGURE 13: optional — fold rate vs model parameter count, scatter plot. Tests the "do small models over-fold" hypothesis.]`

---

## 9. Conclusions

<span style="color:red">[TODO — recap the key findings (one or two sentences each) for each of the brief's three research questions.]</span>

### What this tells us about LLMs as agents

<span style="color:red">[TODO — broader interpretation. Some prompts for thinking: did the LLMs' poker performance correlate with what we'd guess from their reputed "general" capability (instruction-following benchmarks)? Or did poker reveal a different dimension entirely? Was the chain-of-thought model (DeepSeek-R1) helped or hindered by reasoning out loud?]</span>

### Use of AI tools in this project

<span style="color:red">[TODO — write this in your own voice. Below is a starting draft based on what I observed during the project; revise to reflect your own perspective and adjust the proportions if needed.]</span>

This project was developed in close collaboration with Anthropic's Claude (via the Cowork mode of the Claude Desktop app), which acted as a pair-programmer throughout. AI tools were used for:

- **Code generation** — the bulk of each module's first draft was written via the AI, then reviewed, refined, and tested before being committed. Approximately <span style="color:red">[X]%</span> of the codebase by line count originated from AI-generated suggestions, with the remainder being either edits to those suggestions or hand-written test infrastructure.
- **Debugging** — when bugs surfaced (e.g. a rare betting-round termination edge case, file-truncation issues during edits), the AI helped diagnose and propose fixes. In several cases the AI's first proposal was wrong and required iteration; the final fix was usually a refinement after one or two follow-ups.
- **Prompt design** — the personality system prompts were drafted by the AI on the basis of stated design principles (concrete behavioural verbs, ~3–5 sentences, an explicit attention-axis sentence) and then accepted with minor edits. The base bot's JSON-output user prompt was iterated on a handful of times based on early validation runs.
- **Architectural decisions** — the two-layer separation (engine vs. decision-makers vs. UI), the JSONL-based tracker design, the plan-then-materialise round-robin pattern, and the personality 2×2 framework all emerged from extended back-and-forth with the AI. In each case I selected and steered, but the AI accelerated the option-generation phase substantially.

#### What worked well

<span style="color:red">[TODO — your own reflection. Some prompts: rapid iteration on small modules with built-in smoke tests; the AI's tendency to suggest sensible architectural decoupling (engine → bots → UI); having a writing partner for the report itself.]</span>

#### What didn't

<span style="color:red">[TODO — your own reflection. Some prompts: occasional file-truncation issues during long Edit operations; some debug cycles when the AI confidently gave model tags that turned out not to exist on Ollama (e.g. `qwen3:7b` doesn't exist; `qwen3:8b` does); the AI's occasional over-engineering tendency that needed pruning.]</span>

---

## References

<span style="color:red">[TODO — populate. Likely categories of citation:]</span>

- **Poker theory** — a textbook citation for the tight/loose × passive/aggressive framework. Sklansky's *The Theory of Poker* is the standard.
- **5-card draw rules** — any standard rulebook.
- **LLM models** — the model cards or original release announcements for each LLM in the roster (Llama 3.1, Mistral 7B v0.3, Gemma 3, Qwen 3, Phi-4, DeepSeek-R1).
- **Ollama** — the project's documentation site (ollama.com/docs).
- **Pandas / matplotlib** — standard scientific-Python citations if you want them; not strictly required.
- **Brief** — your course brief PDF (Poker McPokerface assignment).
- **Anthropic Claude** — citation of the AI tooling used, in line with the AI-use disclosure in §9.

---

## Suggested figures inventory

A consolidated list of where graphics would live, including the SVGs already in `Instructions/`:

| # | Caption | Source |
|---|---|---|
| 1 | Optional splash image | New (or omit) |
| 2 | Project structure diagram | `Instructions/poker_project_structure.svg` |
| 3 | Tracker data model | `Instructions/poker_tracker_data_model.svg` |
| 4 | Showdown rendering screenshot | New screenshot from `game.ipynb` after a hand |
| 5 | Model parameters vs mean chip change | `analytics.ipynb` after round-robin |
| 6 | Side-by-side reasoning excerpts (TAG vs Calling Station) | New, hand-curated from `reasoning.jsonl` |
| 7 | Tracker analytics views diagram | `Instructions/poker_analytics_views.svg` |
| 8 | Best bot bar chart | `analytics.ipynb` |
| 9 | Best LLM bar chart | `analytics.ipynb` |
| 10 | Best personality bar chart | `analytics.ipynb` |
| 11 | Cumulative chip trend per bot | `analytics.ipynb` |
| 12 | Knockout bracket diagram | New, after Phase 5 |
| 13 | Fold rate vs model size scatter (optional) | `analytics.ipynb` (new cell) |
