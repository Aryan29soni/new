"""Opponent elixir model.

Pure logic with no camera or OpenCV dependency, so it can be unit tested with a fake clock.

Model:
  * Both players start with START_ELIXIR and regenerate 1 elixir every BASE_SECONDS_PER_ELIXIR
    seconds, scaled by the current phase multiplier (x2 in the last minute, and so on).
  * Elixir is capped at MAX_ELIXIR; anything gained above the cap is counted as "leaked".
  * Playing a card subtracts its cost. If the estimate was lower than the cost, the estimate
    was wrong (the opponent obviously had enough), so it is corrected: they had exactly the
    cost and are now at 0. This makes the counter self-correcting over a match.
  * Mirror costs the opponent's previous card + 1.
  * Cycle tracking: the 4 most recently played cards are in the queue, not in the hand.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

from .cards import MIRROR, CardDB, normalize

START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
BASE_SECONDS_PER_ELIXIR = 2.8
REGULATION_SECONDS = 180
HAND_SIZE = 4
DECK_SIZE = 8


@dataclass(frozen=True)
class Phase:
    start: float  # seconds since match start
    multiplier: float
    label: str


# Standard 1v1 ladder: single elixir for 2 minutes, double for the last minute and overtime.
# Game modes and Supercell updates change this, so it can be overridden with --schedule.
DEFAULT_SCHEDULE: tuple[Phase, ...] = (
    Phase(0, 1, "Single elixir"),
    Phase(120, 2, "Double elixir"),
    Phase(180, 2, "Overtime"),
)


def parse_schedule(text: str) -> tuple[Phase, ...]:
    """'0:1,120:2,240:3' -> phases. Each item is <start seconds>:<multiplier>."""
    phases = []
    for item in text.split(","):
        start, mult = item.split(":")
        m = float(mult)
        phases.append(Phase(float(start), m, f"x{m:g} elixir"))
    phases.sort(key=lambda p: p.start)
    if not phases or phases[0].start != 0:
        raise ValueError("schedule must start at 0, e.g. 0:1,120:2")
    return tuple(phases)


@dataclass
class Play:
    card: str
    cost: int
    elapsed: float  # match seconds when played
    estimate_before: float  # what we thought they had
    correction: float  # how much we had underestimated (0 if the estimate was fine)
    mirrored: str | None = None


@dataclass
class _State:
    elixir: float = START_ELIXIR
    last_t: float | None = None
    leaked: float = 0.0
    plays: list[Play] = field(default_factory=list)
    deck: list[str] = field(default_factory=list)
    phase_override: tuple[float, float] | None = None  # (from_elapsed, multiplier)


class ElixirTracker:
    def __init__(
        self,
        cards: CardDB,
        schedule: tuple[Phase, ...] = DEFAULT_SCHEDULE,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cards = cards
        self.schedule = schedule
        self.clock = clock
        self.match_start: float | None = None
        self.s = _State()
        self._history: list[tuple[_State, float | None]] = []

    # ---------------------------------------------------------------- match control
    @property
    def running(self) -> bool:
        return self.match_start is not None

    def start(self, now: float | None = None) -> None:
        """Call when the battle starts (the moment the countdown ends)."""
        now = self.clock() if now is None else now
        self.match_start = now
        self.s = _State(last_t=now)
        self._history.clear()

    def reset(self) -> None:
        self.match_start = None
        self.s = _State()
        self._history.clear()

    def sync_clock(self, remaining_seconds: float, now: float | None = None) -> None:
        """Align the match clock with the in-game timer (regulation time remaining)."""
        now = self.clock() if now is None else now
        if not self.running:
            self.start(now)
        self._advance(now)
        self.match_start = now - (REGULATION_SECONDS - remaining_seconds)

    def elapsed(self, now: float | None = None) -> float:
        if not self.running:
            return 0.0
        now = self.clock() if now is None else now
        return max(0.0, now - self.match_start)

    # ---------------------------------------------------------------- phases
    def multiplier_at(self, elapsed: float) -> float:
        o = self.s.phase_override
        if o is not None and elapsed >= o[0]:
            return o[1]
        mult = self.schedule[0].multiplier
        for p in self.schedule:
            if elapsed >= p.start:
                mult = p.multiplier
        return mult

    def phase_label(self, now: float | None = None) -> str:
        e = self.elapsed(now)
        if self.s.phase_override is not None and e >= self.s.phase_override[0]:
            return f"x{self.s.phase_override[1]:g} elixir (manual)"
        label = self.schedule[0].label
        for p in self.schedule:
            if e >= p.start:
                label = p.label
        return label

    def set_multiplier(self, multiplier: float, now: float | None = None) -> None:
        """Manual override, e.g. when the game shows 'Double Elixir!' and we are out of sync."""
        now = self.clock() if now is None else now
        self._advance(now)
        self.s.phase_override = (self.elapsed(now), multiplier)

    def _boundaries(self) -> list[float]:
        b = [p.start for p in self.schedule]
        if self.s.phase_override is not None:
            b.append(self.s.phase_override[0])
        return sorted(set(b))

    # ---------------------------------------------------------------- regeneration
    def _advance(self, now: float) -> None:
        """Integrate elixir regeneration from the last update up to `now`."""
        if not self.running or self.s.last_t is None or now <= self.s.last_t:
            if self.running and self.s.last_t is None:
                self.s.last_t = now
            return
        e0 = self.s.last_t - self.match_start
        e1 = now - self.match_start
        gain = 0.0
        cuts = [e0] + [b for b in self._boundaries() if e0 < b < e1] + [e1]
        for a, b in zip(cuts, cuts[1:]):
            gain += (b - a) * self.multiplier_at(a) / BASE_SECONDS_PER_ELIXIR
        total = self.s.elixir + gain
        if total > MAX_ELIXIR:
            self.s.leaked += total - max(self.s.elixir, MAX_ELIXIR)
            total = MAX_ELIXIR
        self.s.elixir = total
        self.s.last_t = now

    def elixir(self, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        self._advance(now)
        return self.s.elixir

    @property
    def leaked(self) -> float:
        return self.s.leaked

    def seconds_to_full(self, now: float | None = None) -> float:
        """Rough time until the opponent reaches 10 at the current rate."""
        now = self.clock() if now is None else now
        missing = MAX_ELIXIR - self.elixir(now)
        rate = self.multiplier_at(self.elapsed(now)) / BASE_SECONDS_PER_ELIXIR
        return missing / rate if rate > 0 else float("inf")

    # ---------------------------------------------------------------- plays
    def play(self, card: str, now: float | None = None) -> Play:
        now = self.clock() if now is None else now
        key = normalize(card)
        raw = self.cards.cost(key)  # KeyError for unknown cards
        if not self.running:
            self.start(now)  # first card seen: assume the match just started
        self._advance(now)
        self._history.append((copy.deepcopy(self.s), self.match_start))

        mirrored = None
        if raw == MIRROR:
            if not self.s.plays:
                self._history.pop()
                raise ValueError("Mirror needs a previous card to copy")
            prev = self.s.plays[-1]
            mirrored = prev.mirrored or prev.card
            cost = int(self.cards.cost(mirrored)) + 1
        else:
            cost = int(raw)

        before = self.s.elixir
        correction = max(0.0, cost - before)
        self.s.elixir = max(0.0, before - cost)
        p = Play(key, cost, self.elapsed(now), before, correction, mirrored)
        self.s.plays.append(p)
        if key not in self.s.deck and len(self.s.deck) < DECK_SIZE:
            self.s.deck.append(key)
        return p

    def adjust(self, delta: float, now: float | None = None) -> None:
        """Nudge the estimate by hand (e.g. opponent has an Elixir Collector)."""
        now = self.clock() if now is None else now
        self._advance(now)
        self._history.append((copy.deepcopy(self.s), self.match_start))
        self.s.elixir = min(MAX_ELIXIR, max(0.0, self.s.elixir + delta))

    def undo(self, now: float | None = None) -> bool:
        """Revert the last play/adjustment (e.g. a false detection)."""
        if not self._history:
            return False
        now = self.clock() if now is None else now
        self.s, self.match_start = self._history.pop()
        self._advance(now)
        return True

    # ---------------------------------------------------------------- cycle tracking
    @property
    def plays(self) -> list[Play]:
        return self.s.plays

    @property
    def deck(self) -> list[str]:
        """Distinct cards seen so far, in order of first appearance (max 8)."""
        return self.s.deck

    def not_in_hand(self) -> list[str]:
        """The last 4 cards played are in the opponent's queue, oldest first."""
        return [p.card for p in self.s.plays[-HAND_SIZE:]]

    def in_hand(self) -> list[str]:
        """Known cards that must be in the opponent's hand (complete once all 8 are seen)."""
        queued = set(self.not_in_hand())
        return [c for c in self.s.deck if c not in queued]

    def next_to_cycle_in(self) -> str | None:
        """Card that returns to their hand after their next play."""
        return self.s.plays[-HAND_SIZE].card if len(self.s.plays) >= HAND_SIZE else None
