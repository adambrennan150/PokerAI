"""
analyse_round_robin.py — full analytics suite for a round-robin session.

Reads a session's JSONL logs and produces every table and figure used
in the project's report:

  Tables (printed to stdout — tee to a log if you want to keep them):
  * Per-model parse-error rates and mean response lengths
  * Action-type frequency mix by model
  * Action-type frequency mix by personality
  * Best (model × personality) combos by mean chip change per hand
  * Best LLM averaged across personalities
  * Best personality averaged across LLMs

  Figures saved to runs/<SESSION_ID>/figs/:
  * 01_heatmap_model_x_personality.png
  * 02_chip_trajectory.png
  * 03_action_mix_by_model.png
  * 04_action_mix_by_personality.png
  * 05_parse_error_rate_by_model.png

Models with >=50% parse-error rate are flagged as "broken" and
excluded from the chip-change rankings (their bots' decisions
aren't real, just safe-default fallbacks). Their behavioural
fingerprint is still useful — it shows up in the action-mix charts
as a near-100% check/fold pattern.

Usage:
    python scripts/analyse_round_robin.py                  # auto-pick latest main_*
    python scripts/analyse_round_robin.py main_round_robin_v2
    python scripts/analyse_round_robin.py 2>&1 | tee analysis.log
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent

# Models with this much parse-error rate get filtered out of chip-change
# rankings (their decisions aren't real — just safe defaults).
BROKEN_THRESHOLD_PCT = 50.0


# ----------------------------------------------------------------------
# Data loading — tolerant to malformed JSONL rows (we've seen null-byte
# corruption from partial writes).
# ----------------------------------------------------------------------
def _safe_load_jsonl(path: Path):
    """Parse a JSONL file, skipping malformed lines with a warning."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed {path.name} line {i}: {e}",
                      file=sys.stderr)
    return rows


def _resolve_session_dir(arg: str | None) -> Path:
    """Resolve which session to analyse. Explicit arg wins; otherwise
    pick the most recent main_round_robin_* session in runs/."""
    runs = PROJECT_ROOT / "runs"
    if arg:
        candidate = runs / arg
        if not candidate.exists():
            raise SystemExit(f"Session not found: {candidate}")
        return candidate
    candidates = sorted(p for p in runs.iterdir()
                        if p.is_dir() and p.name.startswith("main_round_robin"))
    if not candidates:
        raise SystemExit(
            "No main_round_robin_* sessions in runs/. Pass a session id "
            "explicitly: python scripts/analyse_round_robin.py <id>"
        )
    return candidates[-1]


def load_data(session_dir: Path):
    """Returns (seat_df, actions_df, reasoning_rows)."""
    hands = _safe_load_jsonl(session_dir / "hands.jsonl")
    actions = _safe_load_jsonl(session_dir / "actions.jsonl")
    reasoning = _safe_load_jsonl(session_dir / "reasoning.jsonl")

    # Flatten hands → one row per (hand, seat).
    seat_rows = []
    for h in hands:
        for sr in h.get("seat_results", []):
            seat_rows.append({
                "hand_id": h["hand_id"],
                "name": sr["name"],
                "model_id": sr["model_id"],
                "personality_id": sr["personality_id"],
                "starting_chips": sr["starting_chips"],
                "ending_chips": sr["ending_chips"],
                "net_change": sr["net_change"],
                "folded": sr["folded"],
                "won": any(w["name"] == sr["name"] for w in h.get("winners", [])),
                "final_hand_category": sr.get("final_hand_category"),
            })
    seat_df = pd.DataFrame(seat_rows)
    actions_df = pd.DataFrame(actions)
    return seat_df, actions_df, reasoning


# ----------------------------------------------------------------------
# Per-model parse-error rates + identifying broken models
# ----------------------------------------------------------------------
def parse_error_summary(reasoning):
    """Build a DataFrame of parse-error stats per model. Print and return."""
    by_model = defaultdict(list)
    for r in reasoning:
        if r.get("model_id", "").startswith("mock"):
            continue
        by_model[r["model_id"]].append(r)

    rows = []
    for model, rs in sorted(by_model.items()):
        errors = sum(1 for r in rs if r.get("parse_error"))
        err_rate = 100 * errors / len(rs) if rs else 0.0
        lens = [len(r.get("raw_response") or "") for r in rs]
        mean_len = sum(lens) / len(lens) if lens else 0.0
        rows.append({
            "model_id": model,
            "calls": len(rs),
            "errors": errors,
            "err_pct": round(err_rate, 1),
            "mean_resp_len": round(mean_len, 0),
            "broken": err_rate >= BROKEN_THRESHOLD_PCT,
        })
    df = pd.DataFrame(rows).set_index("model_id")
    print("=== Per-model parse-error rates ===")
    print(df.to_string())
    print()
    broken = df[df["broken"]].index.tolist()
    if broken:
        print(f"  WARNING: models with >={BROKEN_THRESHOLD_PCT:.0f}% parse-error "
              f"rate (excluded from chip rankings):")
        for m in broken:
            print(f"    {m}")
        print()
    return df, broken


# ----------------------------------------------------------------------
# Action-type frequency tables
# ----------------------------------------------------------------------
def action_mix(actions_df):
    """Action-type % by model and by personality (betting rounds only)."""
    # Filter to actual betting decisions; antes are forced and not informative.
    real = actions_df[actions_df["phase"].isin(["first_betting", "second_betting"])]

    by_model = (real.groupby(["model_id", "action_type"]).size()
                .unstack(fill_value=0))
    by_model_pct = by_model.div(by_model.sum(axis=1), axis=0) * 100

    by_pers = (real.groupby(["personality_id", "action_type"]).size()
               .unstack(fill_value=0))
    by_pers_pct = by_pers.div(by_pers.sum(axis=1), axis=0) * 100

    print("=== Action-type frequency by model (% of betting actions) ===")
    print(by_model_pct.round(1).to_string())
    print()
    print("=== Action-type frequency by personality (% of betting actions) ===")
    print(by_pers_pct.round(1).to_string())
    print()
    return by_model_pct, by_pers_pct


# ----------------------------------------------------------------------
# Three required brief aggregations
# ----------------------------------------------------------------------
def best_bot(df):
    out = df.groupby(["model_id", "personality_id"]).agg(
        hands=("hand_id", "count"),
        mean_delta=("net_change", "mean"),
        total_delta=("net_change", "sum"),
        win_rate=("won", "mean"),
        fold_rate=("folded", "mean"),
    ).sort_values("mean_delta", ascending=False)
    print("=== Best (model x personality) by mean chip change per hand ===")
    print(out.round(2).to_string())
    print()
    return out


def best_llm(df):
    out = df.groupby("model_id").agg(
        hands=("hand_id", "count"),
        seats=("name", "nunique"),
        mean_delta=("net_change", "mean"),
        total_delta=("net_change", "sum"),
        win_rate=("won", "mean"),
        fold_rate=("folded", "mean"),
    ).sort_values("mean_delta", ascending=False)
    print("=== Best LLM (averaged across personalities) ===")
    print(out.round(2).to_string())
    print()
    return out


def best_personality(df):
    out = df.groupby("personality_id").agg(
        hands=("hand_id", "count"),
        seats=("name", "nunique"),
        mean_delta=("net_change", "mean"),
        total_delta=("net_change", "sum"),
        win_rate=("won", "mean"),
        fold_rate=("folded", "mean"),
    ).sort_values("mean_delta", ascending=False)
    print("=== Best personality (averaged across LLMs) ===")
    print(out.round(2).to_string())
    print()
    return out


# ----------------------------------------------------------------------
# Sample reasoning excerpts — one per (model, personality) combo
# ----------------------------------------------------------------------
def sample_reasoning(reasoning, max_rows=20):
    """Print a sample reasoning string from each (model, personality) pair
    where one is available. Skips parse-error rows so samples are real."""
    seen = set()
    print("=== Sample reasoning excerpts (one per combo) ===\n")
    shown = 0
    for r in reasoning:
        if r.get("model_id", "").startswith("mock"):
            continue
        if r.get("decision_type") != "action":
            continue
        if r.get("parse_error") or not r.get("reasoning"):
            continue
        key = (r["model_id"], r["personality_id"])
        if key in seen:
            continue
        text = r["reasoning"].strip()
        if len(text) < 30:
            continue
        seen.add(key)
        print(f"--- {r['model_id']} + {r['personality_id']} "
              f"(decision: {r.get('decided_action')}) ---")
        print(f"  {text[:300]!r}\n")
        shown += 1
        if shown >= max_rows:
            break


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
ACTION_COLOURS = {
    "fold":  "#e76f51",
    "check": "#264653",
    "call":  "#f4a261",
    "raise": "#2a9d8f",
}


def fig_heatmap(df_clean, figdir):
    """Mean chip change per (model x personality)."""
    pivot = df_clean.groupby(["model_id", "personality_id"])["net_change"].mean().unstack()
    # Order rows by overall model strength (descending mean_delta).
    order = (df_clean.groupby("model_id")["net_change"].mean()
             .sort_values(ascending=False).index)
    pivot = pivot.loc[order, :]

    fig, ax = plt.subplots(figsize=(8, 4))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10)
    ax.set_title("Mean chip change per hand (working models)")
    plt.colorbar(im, ax=ax, label="chips/hand")
    plt.tight_layout()
    p = figdir / "01_heatmap_model_x_personality.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  saved: {p.name}")


def fig_chip_trajectory(df, figdir, top_n=5, bot_n=5):
    """Cumulative net chips per bot over hand_id — top + bottom performers."""
    trend = df.sort_values("hand_id").groupby("name")["net_change"].cumsum()
    df_t = df.assign(cum=trend)
    totals = df.groupby("name")["net_change"].sum().sort_values()
    bot = list(totals.head(bot_n).index)
    top = list(totals.tail(top_n).index)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in top + bot:
        sub = df_t[df_t["name"] == name].sort_values("hand_id")
        style = "-" if name in top else "--"
        ax.plot(sub["hand_id"], sub["cum"], style, label=name,
                alpha=0.85, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hand number")
    ax.set_ylabel("Cumulative net chips")
    ax.set_title(f"Chip trajectory — top {top_n} + bottom {bot_n}")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    p = figdir / "02_chip_trajectory.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  saved: {p.name}")


def fig_action_mix_by_model(by_model_pct, figdir):
    cols = [c for c in ["fold", "check", "call", "raise"] if c in by_model_pct.columns]
    colours = [ACTION_COLOURS[c] for c in cols]
    fig, ax = plt.subplots(figsize=(8, 4))
    by_model_pct[cols].plot(kind="bar", stacked=True, ax=ax, color=colours)
    ax.set_ylabel("% of betting actions")
    ax.set_xlabel("")
    ax.set_title("Action mix by model (betting rounds only)")
    ax.legend(title="action", loc="upper right", bbox_to_anchor=(1.18, 1))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = figdir / "03_action_mix_by_model.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  saved: {p.name}")


def fig_action_mix_by_personality(by_pers_pct, figdir):
    cols = [c for c in ["fold", "check", "call", "raise"] if c in by_pers_pct.columns]
    colours = [ACTION_COLOURS[c] for c in cols]
    fig, ax = plt.subplots(figsize=(8, 4))
    by_pers_pct[cols].plot(kind="bar", stacked=True, ax=ax, color=colours)
    ax.set_ylabel("% of betting actions")
    ax.set_xlabel("")
    ax.set_title("Action mix by personality")
    ax.legend(title="action", loc="upper right", bbox_to_anchor=(1.18, 1))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = figdir / "04_action_mix_by_personality.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  saved: {p.name}")


def fig_parse_error_rate(parse_df, figdir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    parse_df.sort_values("err_pct")["err_pct"].plot(
        kind="barh", ax=ax,
        color=["#2a9d8f" if v < BROKEN_THRESHOLD_PCT else "#e76f51"
               for v in parse_df.sort_values("err_pct")["err_pct"]],
    )
    ax.set_xlabel("Parse-error rate (%)")
    ax.set_title("Per-model parse-error rate")
    ax.axvline(BROKEN_THRESHOLD_PCT, color="black", linestyle="--", linewidth=0.8,
               label=f"{BROKEN_THRESHOLD_PCT:.0f}% threshold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    p = figdir / "05_parse_error_rate_by_model.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  saved: {p.name}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(argv=None):
    argv = argv or sys.argv[1:]
    arg = argv[0] if argv and not argv[0].startswith("--") else None
    session_dir = _resolve_session_dir(arg)
    print(f"Analysing session: {session_dir.name}\n")

    # Load + flatten
    seat_df, actions_df, reasoning = load_data(session_dir)
    print(f"Loaded: {len(seat_df)} seat-rows, {len(actions_df)} action-rows, "
          f"{len(reasoning)} reasoning-rows")
    print(f"        {seat_df['hand_id'].nunique()} distinct hands, "
          f"{seat_df['name'].nunique()} bots, "
          f"{seat_df['model_id'].nunique()} models, "
          f"{seat_df['personality_id'].nunique()} personalities")
    print()

    # Parse-error rates → identifies broken models
    parse_df, broken = parse_error_summary(reasoning)

    # Action-mix tables (across ALL models — broken models' fingerprints
    # are themselves informative)
    by_model_pct, by_pers_pct = action_mix(actions_df)

    # Working-models-only filter for chip rankings
    seat_df_clean = seat_df[~seat_df["model_id"].isin(broken)] if broken else seat_df
    if broken:
        print(f"Chip-change rankings filter out broken models — "
              f"{len(seat_df_clean)} rows kept of {len(seat_df)}.\n")

    best_bot(seat_df_clean)
    best_llm(seat_df_clean)
    best_personality(seat_df_clean)

    # Sample reasoning (across ALL models)
    sample_reasoning(reasoning)

    # Figures
    figdir = session_dir / "figs"
    figdir.mkdir(exist_ok=True)
    print(f"=== Generating figures into {figdir} ===")
    fig_heatmap(seat_df_clean, figdir)
    fig_chip_trajectory(seat_df_clean, figdir)
    fig_action_mix_by_model(by_model_pct, figdir)
    fig_action_mix_by_personality(by_pers_pct, figdir)
    fig_parse_error_rate(parse_df, figdir)
    print()
    print(f"Done. Figures in {figdir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
