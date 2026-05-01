"""
rendering.py — pure GameView → string rendering.

Two output flavours:
  * `render_table_html(view)` — styled HTML suitable for IPython.display
    inside a Jupyter notebook (green felt, card faces, player boxes).
  * `render_table_text(view)` — ASCII fallback for terminals and tests.

Design notes
------------
* These are PURE functions. They take a GameView and return a string.
  No printing, no display calls, no side effects. That makes them
  trivially testable, and lets the same rendering be reused by a
  notebook display, a terminal print, or a future widget layer.

* The HTML uses inline styles only — no external stylesheet, no
  JavaScript. This makes the output portable: Jupyter, Voilà, an
  exported HTML file, all work the same.

* Hidden cards (other players' hole cards) are shown as card backs
  using a Unicode block character. The number of card backs reflects
  `cards_held` so a 4-card draw is visually distinct from a stand-pat.

* The action history is truncated to the most recent N rows so a long
  hand doesn't push the table off the screen.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from engine import (
    Card, GameView, HandSummary, Phase, PlayerStatus, PublicPlayerInfo, Suit,
)


# Most recent N action rows to show. Past that the prompt gets unwieldy.
HISTORY_TAIL = 8


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------
# Inline-style colour palette. Kept small and centralised so a tweak
# touches one place. Hearts and diamonds are red; clubs and spades
# black, exactly like real cards.
_RED_SUITS = {Suit.HEARTS, Suit.DIAMONDS}
_FELT_BG = "#0a5d3a"
_FELT_BORDER = "#063620"
_CARD_BG = "#fafafa"
_CARD_BORDER = "#222"
_CARD_BACK = "#244"
_TEXT_LIGHT = "#f0f0f0"
_TEXT_MUTED = "#cfcfcf"
_RED = "#c0392b"
_BLACK = "#1c1c1c"


def render_card_html(card: Card) -> str:
    """A single visible card."""
    colour = _RED if card.suit in _RED_SUITS else _BLACK
    return (
        f'<span style="display:inline-block; width:42px; height:60px; '
        f'background:{_CARD_BG}; border:1px solid {_CARD_BORDER}; '
        f'border-radius:5px; margin:2px; padding:4px 0 0 6px; '
        f'font-family:Menlo,Consolas,monospace; font-size:18px; '
        f'font-weight:bold; color:{colour}; vertical-align:middle;">'
        f'{card.rank.short}<br>{card.suit.value}'
        f'</span>'
    )


def _render_card_back_html() -> str:
    return (
        f'<span style="display:inline-block; width:42px; height:60px; '
        f'background:{_CARD_BACK}; border:1px solid {_CARD_BORDER}; '
        f'border-radius:5px; margin:2px; vertical-align:middle;"></span>'
    )


def _render_other_player_html(p: PublicPlayerInfo) -> str:
    status_colour = {
        PlayerStatus.ACTIVE: "#9bd6a8",
        PlayerStatus.FOLDED: "#888",
        PlayerStatus.ALL_IN: "#f5b942",
    }[p.status]
    backs = "".join(_render_card_back_html() for _ in range(p.cards_held))
    return (
        f'<div style="display:inline-block; vertical-align:top; '
        f'margin:8px 12px; padding:8px 12px; background:rgba(0,0,0,0.18); '
        f'border-radius:8px; min-width:160px; color:{_TEXT_LIGHT}; '
        f'font-family:system-ui,-apple-system,sans-serif;">'
        f'<div style="font-weight:600; font-size:14px;">{p.name}</div>'
        f'<div style="color:{_TEXT_MUTED}; font-size:12px;">'
        f'chips {p.chips} &nbsp;|&nbsp; bet {p.current_bet} '
        f'&nbsp;|&nbsp; <span style="color:{status_colour};">'
        f'{p.status.value}</span></div>'
        f'<div style="margin-top:4px;">{backs or "&nbsp;"}</div>'
        f'</div>'
    )


def _render_history_html(history) -> str:
    if not history:
        return f'<div style="color:{_TEXT_MUTED};">(no actions yet)</div>'
    rows = list(history)[-HISTORY_TAIL:]
    lines = []
    for rec in rows:
        posted = f' (+{rec.chips_posted})' if rec.chips_posted else ''
        lines.append(
            f'<div style="font-family:monospace; font-size:12px; '
            f'color:{_TEXT_MUTED};">'
            f'[{rec.phase}] {rec.player}: {rec.action}{posted} '
            f'(pot={rec.pot_after})</div>'
        )
    return "".join(lines)


def render_table_html(view: GameView) -> str:
    """Full-table HTML for notebook display."""

    your_cards = "".join(render_card_html(c) for c in view.your_hand) or \
        '<span style="color:#888;">(no cards)</span>'

    others = "".join(_render_other_player_html(o) for o in view.other_players) or \
        '<div style="color:#888;">(no other players)</div>'

    return (
        f'<div style="background:{_FELT_BG}; border:6px solid {_FELT_BORDER}; '
        f'border-radius:14px; padding:18px; max-width:780px; '
        f'font-family:system-ui,-apple-system,sans-serif; color:{_TEXT_LIGHT};">'
        # Header row: pot + phase + dealer
        f'<div style="display:flex; justify-content:space-between; '
        f'align-items:center; margin-bottom:10px;">'
        f'<div style="font-size:18px; font-weight:600;">'
        f'POT: <span style="color:#f5e063;">{view.pot}</span></div>'
        f'<div style="font-size:13px; color:{_TEXT_MUTED}; '
        f'text-transform:uppercase; letter-spacing:1.5px;">'
        f'{view.phase.value.replace("_", " ")}</div>'
        f'</div>'
        # Other players row
        f'<div style="margin-bottom:10px;">{others}</div>'
        # Divider
        f'<hr style="border:none; border-top:1px dashed rgba(255,255,255,0.2); '
        f'margin:12px 0;">'
        # Your seat
        f'<div style="background:rgba(0,0,0,0.25); padding:12px 14px; '
        f'border-radius:10px;">'
        f'<div style="font-weight:700; font-size:15px; margin-bottom:6px;">'
        f'{view.your_name} (you)</div>'
        f'<div style="margin-bottom:8px;">{your_cards}</div>'
        f'<div style="font-size:13px; color:{_TEXT_MUTED};">'
        f'chips {view.your_chips} &nbsp;|&nbsp; bet this round '
        f'{view.your_current_bet} &nbsp;|&nbsp; total in pot '
        f'{view.your_total_contributed} &nbsp;|&nbsp; '
        f'<b style="color:#f5e063;">to call: {view.amount_to_call}</b>'
        f' &nbsp;|&nbsp; min raise to: {view.min_raise_to}'
        f'</div>'
        f'</div>'
        # History
        f'<div style="margin-top:14px; max-height:200px; overflow-y:auto;">'
        f'<div style="font-size:12px; color:{_TEXT_MUTED}; '
        f'text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;">'
        f'Recent actions</div>'
        f'{_render_history_html(view.history)}'
        f'</div>'
        f'</div>'
    )


# ----------------------------------------------------------------------
# Plain-text rendering — for terminals and tests
# ----------------------------------------------------------------------
def render_table_text(view: GameView, width: int = 64) -> str:
    """Plain-text fallback. Suitable for stdout, log files, doctests.
    Uses ASCII-safe characters in the structure; suit glyphs are still
    Unicode (♥♦♣♠) but they degrade to ?'s gracefully on legacy
    terminals."""

    bar = "+" + "-" * (width - 2) + "+"

    def fit(s: str) -> str:
        # Pad/truncate `s` to fit between the bars.
        s = s if len(s) <= width - 4 else s[: width - 7] + "..."
        return f"| {s:<{width - 4}} |"

    lines = [bar]
    lines.append(fit(f"POT: {view.pot}    PHASE: {view.phase.value.upper()}"))
    lines.append(bar)

    if view.other_players:
        for p in view.other_players:
            backs = "##" * p.cards_held if p.cards_held > 0 else "  "
            lines.append(fit(
                f"{p.name:<14s} chips={p.chips:<5d} bet={p.current_bet:<4d} "
                f"{p.status.value:<7s} [{backs}]"
            ))
    else:
        lines.append(fit("(no other players)"))
    lines.append(bar)

    your_cards = " ".join(str(c) for c in view.your_hand) if view.your_hand else "(no cards)"
    lines.append(fit(f"YOUR HAND: {your_cards}"))
    lines.append(fit(
        f"chips={view.your_chips}  bet={view.your_current_bet}  "
        f"to_call={view.amount_to_call}  min_raise_to={view.min_raise_to}"
    ))
    lines.append(bar)

    lines.append(fit("RECENT ACTIONS:"))
    if view.history:
        for rec in list(view.history)[-HISTORY_TAIL:]:
            posted = f"+{rec.chips_posted}" if rec.chips_posted else ""
            lines.append(fit(
                f"  [{str(rec.phase):<14s}] {rec.player}: {rec.action} "
                f"{posted} (pot={rec.pot_after})"
            ))
    else:
        lines.append(fit("  (none)"))
    lines.append(bar)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Showdown rendering — pretty per-hand result with everyone's cards
# ----------------------------------------------------------------------
# Used at the end of a hand to show winners + per-seat outcomes. The
# `reveal_folded` flag controls whether folded players' cards are shown
# (full information for analysis) or hidden (real-poker realism). For
# this project we default to True — the brief is research-focused and
# seeing what opponents actually held is far more useful than
# preserving table realism.

def render_showdown_text(
    summary: HandSummary, reveal_folded: bool = True,
) -> str:
    """Plain-text showdown summary. Returns the full block as one string.

    Shows: hand id, winners with chip awards, then a per-seat row with
    each player's final hand category, the cards themselves, ending
    chip count, and net change. Folded players' cards are revealed iff
    `reveal_folded=True`.
    """
    lines = [f"=== Hand {summary.hand_id} complete ==="]

    # Winners line. Multiple rows possible per pot in side-pot hands.
    if summary.winners:
        winners_str = ", ".join(
            f"{name} +{chips}" for name, chips in summary.winners
        )
        lines.append(f"Winners: {winners_str}")
    else:
        lines.append("Winners: (none)")

    lines.append("")
    lines.append(f"  {'Player':<10s} {'Chips':>6s} {'Δ':>6s}  Hand")
    for sr in summary.seat_results:
        cards_str = (
            " ".join(str(c) for c in sr.final_hand) if sr.final_hand else "—"
        )

        if sr.folded:
            if reveal_folded and sr.final_hand:
                # Show what they walked away from. Note: folded players'
                # hands aren't run through the evaluator (they're not
                # eligible to win), so we just label them "folded" and
                # let the cards speak for themselves.
                hand_display = f"{'(folded)':<14s}  [{cards_str}]"
            else:
                hand_display = "(folded)"
        elif sr.final_evaluation:
            category = sr.final_evaluation.category.label
            hand_display = f"{category:<14s}  [{cards_str}]"
        else:
            hand_display = "—"

        delta = f"{sr.net_change:+d}"
        # A small visual cue for winners — easy to scan a long table.
        winner_tag = " <-- won" if any(
            w[0] == sr.name for w in summary.winners
        ) else ""
        lines.append(
            f"  {sr.name:<10s} {sr.ending_chips:>6d} {delta:>6s}  "
            f"{hand_display}{winner_tag}"
        )
    return "\n".join(lines)


def render_showdown_html(
    summary: HandSummary, reveal_folded: bool = True,
) -> str:
    """HTML showdown summary, intended for IPython.display in a notebook.
    Same information as the text version, but with proper card faces
    and a green-felt frame to match the table renderer."""

    # Header with the hand id and winners.
    if summary.winners:
        winners_str = ", ".join(
            f"<b>{name}</b> +{chips}" for name, chips in summary.winners
        )
    else:
        winners_str = "<i>(none)</i>"

    rows_html = []
    for sr in summary.seat_results:
        is_winner = any(w[0] == sr.name for w in summary.winners)
        cards_html = (
            "".join(render_card_html(c) for c in sr.final_hand)
            if sr.final_hand else
            f'<span style="color:{_TEXT_MUTED};">—</span>'
        )

        if sr.folded:
            category_text = "(folded)"
            row_label_colour = _TEXT_MUTED
            cards_disp = cards_html if (reveal_folded and sr.final_hand) else (
                f'<span style="color:{_TEXT_MUTED};">cards hidden</span>'
            )
        elif sr.final_evaluation:
            category_text = sr.final_evaluation.category.label
            row_label_colour = "#f5e063" if is_winner else _TEXT_LIGHT
            cards_disp = cards_html
        else:
            category_text = "—"
            row_label_colour = _TEXT_MUTED
            cards_disp = cards_html

        delta_colour = "#9bd6a8" if sr.net_change > 0 else (
            "#e76f51" if sr.net_change < 0 else _TEXT_MUTED
        )
        winner_marker = (
            ' <span style="color:#f5e063; font-weight:600;">★ WON</span>'
            if is_winner else ""
        )

        rows_html.append(
            f'<tr>'
            f'<td style="padding:6px 12px; color:{row_label_colour}; '
            f'font-weight:{"700" if is_winner else "500"};">'
            f'{sr.name}{winner_marker}</td>'
            f'<td style="padding:6px 12px; text-align:right; '
            f'color:{_TEXT_LIGHT};">{sr.ending_chips}</td>'
            f'<td style="padding:6px 12px; text-align:right; '
            f'color:{delta_colour}; font-family:monospace;">'
            f'{sr.net_change:+d}</td>'
            f'<td style="padding:6px 12px; color:{row_label_colour};">'
            f'{category_text}</td>'
            f'<td style="padding:6px 12px;">{cards_disp}</td>'
            f'</tr>'
        )

    return (
        f'<div style="background:{_FELT_BG}; border:6px solid {_FELT_BORDER}; '
        f'border-radius:14px; padding:18px; max-width:780px; '
        f'font-family:system-ui,-apple-system,sans-serif; color:{_TEXT_LIGHT};">'
        f'<div style="font-size:18px; font-weight:600; margin-bottom:6px;">'
        f'Hand {summary.hand_id} — Showdown</div>'
        f'<div style="color:{_TEXT_MUTED}; font-size:13px; margin-bottom:14px;">'
        f'Winners: {winners_str}</div>'
        f'<table style="width:100%; border-collapse:collapse; font-size:14px;">'
        f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.2); '
        f'color:{_TEXT_MUTED}; font-size:11px; text-transform:uppercase; '
        f'letter-spacing:1.5px;">'
        f'<th style="padding:6px 12px; text-align:left;">Player</th>'
        f'<th style="padding:6px 12px; text-align:right;">Chips</th>'
        f'<th style="padding:6px 12px; text-align:right;">Δ</th>'
        f'<th style="padding:6px 12px; text-align:left;">Category</th>'
        f'<th style="padding:6px 12px; text-align:left;">Hand</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
        f'</div>'
    )


# ----------------------------------------------------------------------
# Showdown rendering — pretty per-hand result with everyone's cards
# ----------------------------------------------------------------------
# Used at the end of a hand to show winners + per-seat outcomes. The
# `reveal_folded` flag controls whether folded players' cards are shown
# (full information for analysis) or hidden (real-poker realism). For
# this project we default to True — the brief is research-focused and
# seeing what opponents actually held is far more useful than
# preserving table realism.

def render_showdown_text(
    summary: HandSummary, reveal_folded: bool = True,
) -> str:
    """Plain-text showdown summary. Returns the full block as one string.

    Shows: hand id, winners with chip awards, then a per-seat row with
    each player's final hand category, the cards themselves, ending
    chip count, and net change. Folded players' cards are revealed iff
    `reveal_folded=True`.
    """
    lines = [f"=== Hand {summary.hand_id} complete ==="]

    if summary.winners:
        winners_str = ", ".join(
            f"{name} +{chips}" for name, chips in summary.winners
        )
        lines.append(f"Winners: {winners_str}")
    else:
        lines.append("Winners: (none)")

    lines.append("")
    lines.append(f"  {'Player':<10s} {'Chips':>6s} {'Delta':>6s}  Hand")
    for sr in summary.seat_results:
        cards_str = (
            " ".join(str(c) for c in sr.final_hand) if sr.final_hand else "-"
        )

        if sr.folded:
            if reveal_folded and sr.final_hand:
                hand_display = f"{'(folded)':<14s}  [{cards_str}]"
            else:
                hand_display = "(folded)"
        elif sr.final_evaluation:
            category = sr.final_evaluation.category.label
            hand_display = f"{category:<14s}  [{cards_str}]"
        else:
            hand_display = "-"

        delta = f"{sr.net_change:+d}"
        winner_tag = " <-- won" if any(
            w[0] == sr.name for w in summary.winners
        ) else ""
        lines.append(
            f"  {sr.name:<10s} {sr.ending_chips:>6d} {delta:>6s}  "
            f"{hand_display}{winner_tag}"
        )
    return "\n".join(lines)


def render_showdown_html(
    summary: HandSummary, reveal_folded: bool = True,
) -> str:
    """HTML showdown summary, intended for IPython.display in a notebook.
    Same information as the text version, but with proper card faces
    and a green-felt frame to match the table renderer."""

    if summary.winners:
        winners_str = ", ".join(
            f"<b>{name}</b> +{chips}" for name, chips in summary.winners
        )
    else:
        winners_str = "<i>(none)</i>"

    rows_html = []
    for sr in summary.seat_results:
        is_winner = any(w[0] == sr.name for w in summary.winners)
        cards_html = (
            "".join(render_card_html(c) for c in sr.final_hand)
            if sr.final_hand else
            f'<span style="color:{_TEXT_MUTED};">-</span>'
        )

        if sr.folded:
            category_text = "(folded)"
            row_label_colour = _TEXT_MUTED
            cards_disp = cards_html if (reveal_folded and sr.final_hand) else (
                f'<span style="color:{_TEXT_MUTED};">cards hidden</span>'
            )
        elif sr.final_evaluation:
            category_text = sr.final_evaluation.category.label
            row_label_colour = "#f5e063" if is_winner else _TEXT_LIGHT
            cards_disp = cards_html
        else:
            category_text = "-"
            row_label_colour = _TEXT_MUTED
            cards_disp = cards_html

        delta_colour = "#9bd6a8" if sr.net_change > 0 else (
            "#e76f51" if sr.net_change < 0 else _TEXT_MUTED
        )
        winner_marker = (
            ' <span style="color:#f5e063; font-weight:600;">* WON</span>'
            if is_winner else ""
        )

        rows_html.append(
            f'<tr>'
            f'<td style="padding:6px 12px; color:{row_label_colour}; '
            f'font-weight:{"700" if is_winner else "500"};">'
            f'{sr.name}{winner_marker}</td>'
            f'<td style="padding:6px 12px; text-align:right; '
            f'color:{_TEXT_LIGHT};">{sr.ending_chips}</td>'
            f'<td style="padding:6px 12px; text-align:right; '
            f'color:{delta_colour}; font-family:monospace;">'
            f'{sr.net_change:+d}</td>'
            f'<td style="padding:6px 12px; color:{row_label_colour};">'
            f'{category_text}</td>'
            f'<td style="padding:6px 12px;">{cards_disp}</td>'
            f'</tr>'
        )

    return (
        f'<div style="background:{_FELT_BG}; border:6px solid {_FELT_BORDER}; '
        f'border-radius:14px; padding:18px; max-width:780px; '
        f'font-family:system-ui,-apple-system,sans-serif; color:{_TEXT_LIGHT};">'
        f'<div style="font-size:18px; font-weight:600; margin-bottom:6px;">'
        f'Hand {summary.hand_id} - Showdown</div>'
        f'<div style="color:{_TEXT_MUTED}; font-size:13px; margin-bottom:14px;">'
        f'Winners: {winners_str}</div>'
        f'<table style="width:100%; border-collapse:collapse; font-size:14px;">'
        f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.2); '
        f'color:{_TEXT_MUTED}; font-size:11px; text-transform:uppercase; '
        f'letter-spacing:1.5px;">'
        f'<th style="padding:6px 12px; text-align:left;">Player</th>'
        f'<th style="padding:6px 12px; text-align:right;">Chips</th>'
        f'<th style="padding:6px 12px; text-align:right;">Delta</th>'
        f'<th style="padding:6px 12px; text-align:left;">Category</th>'
        f'<th style="padding:6px 12px; text-align:left;">Hand</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
        f'</div>'
    )
        f'<th style="padding:6px 12px; text-align:left;">Player</th>'
        f'<th style="padding:6px 12px; text-align:right;">Chips</th>'
        f'<th style="padding:6px 12px; text-align:right;">Delta</th>'
        f'<th style="padding:6px 12px; text-align:left;">Category</th>'
        f'<th style="padding:6px 12px; text-align:left;">Hand</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
        f'</div>'
    )
