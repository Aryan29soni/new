"""Draws the opponent-elixir panel with OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from .cards import pretty
from .engine import MAX_ELIXIR, REGULATION_SECONDS, ElixirTracker

W, H = 440, 640
BG = (28, 22, 30)
ELIXIR = (214, 60, 205)  # BGR purple-pink
ELIXIR_DIM = (70, 40, 70)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
WARN = (60, 170, 255)
GOOD = (120, 220, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clock(tracker: ElixirTracker, now: float) -> str:
    if not tracker.running:
        return "--:--"
    e = tracker.elapsed(now)
    if e < REGULATION_SECONDS:
        r = REGULATION_SECONDS - e
        return f"{int(r) // 60}:{int(r) % 60:02d}"
    o = e - REGULATION_SECONDS
    return f"OT +{int(o) // 60}:{int(o) % 60:02d}"


def render(tracker: ElixirTracker, now: float, status: str = "", vision_on: bool = True) -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)
    e = tracker.elixir(now)

    cv2.putText(img, "OPPONENT ELIXIR", (20, 34), FONT, 0.7, MUTED, 2)
    cv2.putText(img, _clock(tracker, now), (W - 150, 34), FONT, 0.7, TEXT, 2)

    # big number
    big = f"{e:.1f}"
    color = WARN if e >= 9.5 else ELIXIR if e >= 4 else GOOD
    cv2.putText(img, big, (20, 120), FONT, 2.8, color, 6)
    cv2.putText(img, f"~{int(e)} usable", (250, 82), FONT, 0.6, TEXT, 1)
    full = tracker.seconds_to_full(now)
    cv2.putText(img, "FULL" if e >= MAX_ELIXIR else f"full in {full:.0f}s", (250, 112), FONT, 0.6, MUTED, 1)

    # 10-segment bar, last segment partially filled
    x0, y0, seg, gap, bh = 20, 140, 38, 2, 26
    for i in range(int(MAX_ELIXIR)):
        x = x0 + i * (seg + gap)
        cv2.rectangle(img, (x, y0), (x + seg, y0 + bh), ELIXIR_DIM, -1)
        fill = min(1.0, max(0.0, e - i))
        if fill > 0:
            cv2.rectangle(img, (x, y0), (x + int(seg * fill), y0 + bh), ELIXIR, -1)
        cv2.putText(img, str(i + 1), (x + 12, y0 + bh + 16), FONT, 0.4, MUTED, 1)

    label = tracker.phase_label(now) if tracker.running else "Press SPACE when the battle starts"
    cv2.putText(img, label, (20, 206), FONT, 0.6, WARN if tracker.running else TEXT, 1)
    if tracker.leaked >= 0.1:
        cv2.putText(img, f"leaked {tracker.leaked:.1f}", (W - 130, 206), FONT, 0.55, MUTED, 1)

    # deck / cycle
    y = 240
    cv2.putText(img, "DECK  (key = slot, green = in hand)", (20, y), FONT, 0.5, MUTED, 1)
    in_hand, queued = set(tracker.in_hand()), tracker.not_in_hand()
    for i in range(8):
        cx, cy = 20 + (i % 4) * 102, y + 14 + (i // 4) * 34
        if i < len(tracker.deck):
            c = tracker.deck[i]
            col = GOOD if c in in_hand else MUTED
            cv2.rectangle(img, (cx, cy), (cx + 96, cy + 28), col, 1)
            cost = tracker.cards.cost(c)
            txt = f"{i + 1} {pretty(c)[:9]}"
            cv2.putText(img, txt, (cx + 4, cy + 19), FONT, 0.42, col, 1)
            cv2.putText(img, str(cost) if cost != "mirror" else "M", (cx + 84, cy + 19), FONT, 0.42, ELIXIR, 1)
        else:
            cv2.rectangle(img, (cx, cy), (cx + 96, cy + 28), (60, 60, 60), 1)
            cv2.putText(img, f"{i + 1} ?", (cx + 4, cy + 19), FONT, 0.42, (90, 90, 90), 1)
    y += 96
    nxt = tracker.next_to_cycle_in()
    if queued:
        cv2.putText(img, "Queue: " + " > ".join(pretty(c)[:8] for c in queued), (20, y), FONT, 0.45, MUTED, 1)
    if nxt:
        cv2.putText(img, f"Next back in hand: {pretty(nxt)}", (20, y + 22), FONT, 0.5, TEXT, 1)

    # recent plays
    y += 56
    cv2.putText(img, "RECENT PLAYS", (20, y), FONT, 0.5, MUTED, 1)
    for i, p in enumerate(reversed(tracker.plays[-7:])):
        yy = y + 24 + i * 24
        t = p.elapsed
        name = pretty(p.card) + (f" ({pretty(p.mirrored)})" if p.mirrored else "")
        cv2.putText(img, f"{int(t) // 60}:{int(t) % 60:02d}", (20, yy), FONT, 0.45, MUTED, 1)
        cv2.putText(img, name[:24], (80, yy), FONT, 0.5, TEXT, 1)
        cv2.putText(img, f"-{p.cost}", (300, yy), FONT, 0.5, ELIXIR, 1)
        if p.correction > 0.05:
            cv2.putText(img, f"fix +{p.correction:.1f}", (345, yy), FONT, 0.45, WARN, 1)

    cv2.putText(img, f"vision {'ON' if vision_on else 'OFF'}   {status}"[:60], (20, H - 14), FONT, 0.45, MUTED, 1)
    return img
