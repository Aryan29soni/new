"""Command line entry points: calibrate, teach, run, cards."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .cards import CardDB, pretty
from .engine import DEFAULT_SCHEDULE, ElixirTracker, parse_schedule
from .hud import render
from .vision import Calibration, FrameSource, TemplateDetector, pick_corners

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "calibration.json"
DEFAULT_TEMPLATES = ROOT / "templates"

HELP = """
Keys (click the HUD or camera window first):
  SPACE   battle started (starts the clock at 5 elixir)
  1-8     opponent played deck slot N (slots fill as cards are seen)
  n       type a card name in the terminal (e.g. 'hog', 'log', 'mirror')
  u       undo last play / adjustment (fix a false detection)
  + / -   adjust estimate by 1 elixir
  e       cycle elixir rate x1 -> x2 -> x3 (if the game says Double/Triple Elixir)
  c       sync clock: type the in-game time left, e.g. 1:42
  v       toggle vision detection
  r       reset for a new match
  h       print this help
  q/Esc   quit
"""


def _load_calibration(args, frame) -> Calibration:
    if args.no_calibration:
        return Calibration.identity(frame.shape)
    if not Path(args.config).exists():
        sys.exit(f"No calibration at {args.config}. Run `python -m elixir_counter calibrate` "
                 "first, or pass --no-calibration for a direct screen recording.")
    return Calibration.load(args.config)


# ============================================================================ calibrate
def cmd_calibrate(args) -> None:
    src = FrameSource(args.source)
    frame, _ = src.read()
    src.release()
    if frame is None:
        sys.exit("No frame from the camera.")
    print("Click the 4 corners of the phone screen: top-left, top-right, bottom-right, bottom-left.")
    corners = pick_corners(frame)
    if corners is None:
        sys.exit("Cancelled.")
    cal = Calibration(corners, (args.width, args.height))
    flat = cal.warp(frame)
    print("Now drag a box over the OPPONENT's side of the arena (where their cards get placed),"
          " then press Enter. Press c to keep the default zone.")
    r = cv2.selectROI("Detection zone: opponent side of the arena", flat, showCrosshair=False)
    cv2.destroyAllWindows()
    if r[2] > 0 and r[3] > 0:
        cal.roi = tuple(int(v) for v in r)
    cal.save(args.config)
    print(f"Saved {args.config}  (zone {cal.roi})")


# ============================================================================ teach
def cmd_teach(args) -> None:
    """Build templates: freeze a frame, box the thing that appears when a card is played, name it."""
    cards = CardDB.load(args.cards)
    src = FrameSource(args.source)
    frame, _ = src.read()
    if frame is None:
        sys.exit("No frame from the source.")
    cal = _load_calibration(args, frame)
    print("Live view. Press SPACE to freeze a frame and crop a template, q to quit.\n"
          "Tip: crop tightly around the opponent's unit right after it is placed (include its"
          " red team colour, leave out the arena floor). 2-4 crops per card work best.")
    paused = not src.live  # video files: step with any key
    while True:
        frame, _ = src.read()
        if frame is None:
            break
        flat = cal.warp(frame)
        view = flat.copy()
        x, y, w, h = cal.roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (255, 200, 0), 1)
        cv2.imshow("teach (SPACE = capture, q = quit)", view)
        k = cv2.waitKey(0 if paused else 30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k != ord(" "):
            continue
        r = cv2.selectROI("crop the card's sprite, Enter to confirm, c to cancel", flat, showCrosshair=False)
        cv2.destroyWindow("crop the card's sprite, Enter to confirm, c to cancel")
        if r[2] < 8 or r[3] < 8:
            print("  (too small / cancelled)")
            continue
        crop = flat[r[1] : r[1] + r[3], r[0] : r[0] + r[2]]
        name = input("  Card name (blank to discard): ").strip()
        if not name:
            continue
        key = cards.resolve(name)
        if key is None:
            print(f"  Unknown card. Did you mean: {', '.join(cards.suggestions(name)) or '?'}")
            continue
        out = Path(args.templates) / key
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{int(time.time() * 1000)}.png"
        cv2.imwrite(str(path), crop)
        print(f"  saved {path}  ({pretty(key)}, {cards.cost(key)} elixir)")
    src.release()
    cv2.destroyAllWindows()


# ============================================================================ run
def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def cmd_run(args) -> None:
    cards = CardDB.load(args.cards)
    schedule = parse_schedule(args.schedule) if args.schedule else DEFAULT_SCHEDULE
    src = FrameSource(args.source)
    frame, t = src.read()
    if frame is None:
        sys.exit("No frame from the source.")
    cal = _load_calibration(args, frame)
    clock_now = [t]
    tracker = ElixirTracker(cards, schedule, clock=lambda: clock_now[0])
    detector = TemplateDetector(Path(args.templates), threshold=args.threshold, confirm_frames=args.confirm)
    vision_on = bool(detector.templates) and not args.manual
    if not detector.templates:
        print(f"No templates in {args.templates}: running in manual mode (use `teach` to add some).")
    else:
        print(f"Loaded templates for {len(detector.templates)} cards: {', '.join(sorted(detector.templates))}")
    print(HELP)
    status = ""

    def log(msg):
        nonlocal status
        status = msg
        print(f"[{tracker.elapsed():6.1f}s] {msg}")

    def play(card):
        try:
            p = tracker.play(card)
        except (KeyError, ValueError) as ex:
            log(f"cannot play {card}: {ex}")
            return
        fix = f" (estimate was {p.correction:.1f} low, corrected)" if p.correction > 0.05 else ""
        log(f"{pretty(p.card)} -{p.cost} -> {tracker.elixir():.1f}{fix}")

    while True:
        if frame is None:
            if src.live:
                frame, t = src.read()
                continue
            print("End of video.")
            cv2.waitKey(0)
            break
        clock_now[0] = t
        flat = cal.warp(frame)

        if vision_on:
            # once the whole deck is known, only look for those 8 cards (faster, fewer false hits)
            only = set(tracker.deck) if len(tracker.deck) >= 8 else None
            for d in detector.process(flat, t, cal.roi, only):
                play(d.card)
                print(f"           detected at ({d.x},{d.y}) score {d.score:.2f}")

        if args.show_camera:
            view = flat.copy()
            detector.draw(view, cal.roi)
            cv2.imshow("camera (flattened)", cv2.resize(view, None, fx=args.view_scale, fy=args.view_scale))
        cv2.imshow("Opponent Elixir", render(tracker, t, status, vision_on))

        k = cv2.waitKey(1 if src.live else max(1, int(1000 / (src.cap.get(cv2.CAP_PROP_FPS) or 30)))) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            tracker.start()
            detector.reset()
            log("battle started: 5.0 elixir")
        elif ord("1") <= k <= ord("8"):
            slot = k - ord("1")
            if slot < len(tracker.deck):
                play(tracker.deck[slot])
            else:
                log(f"slot {slot + 1} empty: press n to enter a new card")
        elif k == ord("n"):
            name = _ask("Opponent played: ")
            key = cards.resolve(name) if name else None
            if key:
                play(key)
            elif name:
                log(f"unknown card '{name}'. Did you mean: {', '.join(cards.suggestions(name))}")
        elif k == ord("u"):
            log("undone" if tracker.undo() else "nothing to undo")
        elif k in (ord("+"), ord("=")):
            tracker.adjust(+1)
            log(f"+1 -> {tracker.elixir():.1f}")
        elif k in (ord("-"), ord("_")):
            tracker.adjust(-1)
            log(f"-1 -> {tracker.elixir():.1f}")
        elif k == ord("e"):
            cur = tracker.multiplier_at(tracker.elapsed())
            nxt = {1: 2, 2: 3}.get(int(cur), 1)
            tracker.set_multiplier(nxt)
            log(f"elixir rate set to x{nxt}")
        elif k == ord("c"):
            txt = _ask("Time left on the in-game clock (m:ss): ")
            try:
                m, s = (txt.split(":") + ["0"])[:2] if ":" in txt else ("0", txt)
                tracker.sync_clock(int(m) * 60 + float(s))
                log(f"clock synced to {txt}")
            except ValueError:
                log(f"could not read time '{txt}'")
        elif k == ord("v"):
            vision_on = not vision_on and bool(detector.templates)
            log(f"vision {'on' if vision_on else 'off'}")
        elif k == ord("r"):
            tracker.reset()
            detector.reset()
            log("reset")
        elif k == ord("h"):
            print(HELP)

        frame, t = src.read()

    src.release()
    cv2.destroyAllWindows()


# ============================================================================ cards
def cmd_cards(args) -> None:
    cards = CardDB.load(args.cards)
    for key, cost in sorted(cards.costs.items(), key=lambda kv: (str(kv[1]), kv[0])):
        print(f"{cost!s:>7}  {pretty(key)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="elixir_counter", description="Clash Royale opponent elixir counter")
    ap.add_argument("--cards", default=None, help="card cost JSON (default: built-in)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--source", default="0", help="camera index, IP-camera URL, video or image file")
        p.add_argument("--config", default=str(DEFAULT_CONFIG), help="calibration file")
        p.add_argument("--templates", default=str(DEFAULT_TEMPLATES))
        p.add_argument("--no-calibration", action="store_true", help="use the frame as-is (screen recordings)")

    p = sub.add_parser("calibrate", help="mark the phone screen in the camera image")
    common(p)
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=960)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("teach", help="capture card templates for automatic detection")
    common(p)
    p.set_defaults(fn=cmd_teach)

    p = sub.add_parser("run", help="run the elixir counter")
    common(p)
    p.add_argument("--threshold", type=float, default=0.80, help="template match score needed (0-1)")
    p.add_argument("--confirm", type=int, default=2, help="frames a match must persist before it counts")
    p.add_argument("--schedule", help="elixir phases as start_s:multiplier, e.g. 0:1,120:2,240:3")
    p.add_argument("--manual", action="store_true", help="start with vision off (keyboard only)")
    p.add_argument("--no-camera-window", dest="show_camera", action="store_false")
    p.add_argument("--view-scale", type=float, default=0.75)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("cards", help="list known cards and costs")
    p.set_defaults(fn=cmd_cards)

    args = ap.parse_args(argv)
    args.fn(args)
