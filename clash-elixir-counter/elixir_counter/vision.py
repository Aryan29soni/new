"""Camera input, perspective correction and card detection."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cards import normalize

DEFAULT_SIZE = (540, 960)  # width, height of the flattened phone screen (portrait)


# ======================================================================= config
@dataclass
class Calibration:
    corners: list[list[float]]  # TL, TR, BR, BL of the phone screen in camera pixels
    size: tuple[int, int] = DEFAULT_SIZE
    roi: tuple[int, int, int, int] | None = None  # detection zone (x, y, w, h) in flat coords

    def __post_init__(self):
        self.size = tuple(self.size)
        w, h = self.size
        if self.roi is None:
            # Opponent's half of the arena, roughly: below their towers' top edge to the river.
            self.roi = (0, int(h * 0.06), w, int(h * 0.38))
        self.roi = tuple(int(v) for v in self.roi)
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        self.matrix = cv2.getPerspectiveTransform(np.float32(self.corners), dst)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix, self.size)

    def save(self, path: str | Path) -> None:
        data = {"corners": self.corners, "size": list(self.size), "roi": list(self.roi)}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(d["corners"], tuple(d.get("size", DEFAULT_SIZE)), d.get("roi"))

    @classmethod
    def identity(cls, frame_shape, size=DEFAULT_SIZE) -> "Calibration":
        """No camera skew: use the whole frame (e.g. a screen recording)."""
        h, w = frame_shape[:2]
        return cls([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], size)


# ======================================================================= capture
class FrameSource:
    """Webcam index, phone IP-camera URL, or a video/image file.

    `read()` returns (frame, t). For live sources t is wall-clock time and a background
    thread always keeps only the newest frame (no lag from OpenCV's buffer). For video files
    t is the video timestamp, so a recording is analysed exactly as if it were live.
    """

    def __init__(self, source: str | int):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.live = isinstance(source, int) or "://" in str(source)
        self.image = None
        if isinstance(source, str) and Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            self.image = cv2.imread(source)
            if self.image is None:
                raise RuntimeError(f"cannot read image {source}")
            self.live = True
            return
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        if self.live:
            threading.Thread(target=self._grab_loop, daemon=True).start()

    def _grab_loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> tuple[np.ndarray | None, float]:
        if self.image is not None:
            return self.image.copy(), time.monotonic()
        if self.live:
            for _ in range(500):  # wait up to ~5 s for the first frame
                with self._lock:
                    if self._frame is not None:
                        return self._frame.copy(), time.monotonic()
                time.sleep(0.01)
            return None, time.monotonic()
        ok, frame = self.cap.read()
        return (frame if ok else None), self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def release(self):
        self._stop = True
        if self.image is None:
            self.cap.release()


# ======================================================================= calibration UI
def pick_corners(frame: np.ndarray, window="Calibrate: click TL, TR, BR, BL of the phone screen") -> list[list[float]] | None:
    """Let the user click the 4 screen corners. Enter = accept, r = redo, Esc = cancel."""
    scale = min(1.0, 1280 / frame.shape[1], 800 / frame.shape[0])
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    while True:
        img = view.copy()
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 255, 255), -1)
            cv2.putText(img, names[i], (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(pts) > 1:
            cv2.polylines(img, [np.int32(pts)], len(pts) == 4, (0, 255, 0), 2)
        msg = f"Click {names[len(pts)]} corner" if len(pts) < 4 else "Enter = accept, r = redo"
        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(window)
            return None
        if k == ord("r"):
            pts.clear()
        if k in (13, 10) and len(pts) == 4:
            cv2.destroyWindow(window)
            return [[x / scale, y / scale] for x, y in pts]


# ======================================================================= detection
@dataclass
class Detection:
    card: str
    x: int
    y: int
    score: float


@dataclass
class _Track:
    card: str
    x: float
    y: float
    hits: int
    last_seen: float
    emitted: bool = False


@dataclass
class TemplateDetector:
    """Detects opponent cards by matching reference crops inside the detection zone.

    Templates live in templates/<card_name>/*.png (made with the `teach` command) and are
    crops of the flattened screen, so their scale matches the live image.

    A matched sprite becomes a *track* that follows the unit as it walks. A play is reported
    only when a new track has been seen for `confirm_frames` frames, so a unit standing on
    the field is counted once, not every frame, and a second copy of the same card placed
    elsewhere still counts as a new play.
    """

    templates_dir: Path
    threshold: float = 0.80
    confirm_frames: int = 2
    forget_seconds: float = 3.0  # how long a lost track is remembered (occlusion, spell effects)
    radius: float = 45.0  # px in the flattened image: how far a tracked unit may move per update
    speed: float = 60.0  # px/s extra radius allowance while a track is not seen
    max_peaks: int = 3  # instances of one template per frame
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    tracks: list[_Track] = field(default_factory=list)
    last_candidates: list[Detection] = field(default_factory=list)

    def __post_init__(self):
        self.templates_dir = Path(self.templates_dir)
        self.reload()

    def reload(self) -> None:
        self.templates = {}
        if not self.templates_dir.exists():
            return
        for d in sorted(p for p in self.templates_dir.iterdir() if p.is_dir()):
            imgs = [cv2.imread(str(f)) for f in sorted(d.glob("*.png"))]
            imgs = [i for i in imgs if i is not None and i.shape[0] >= 8 and i.shape[1] >= 8]
            if imgs:
                self.templates[normalize(d.name)] = imgs

    def reset(self) -> None:
        self.tracks.clear()

    # ---------------------------------------------------------------- matching
    def _candidates(self, zone: np.ndarray, only: set[str] | None) -> list[Detection]:
        found: list[Detection] = []
        for card, tmpls in self.templates.items():
            if only is not None and card not in only:
                continue
            for t in tmpls:
                th, tw = t.shape[:2]
                if th > zone.shape[0] or tw > zone.shape[1]:
                    continue
                res = cv2.matchTemplate(zone, t, cv2.TM_CCOEFF_NORMED)
                for _ in range(self.max_peaks):
                    _, score, _, (x, y) = cv2.minMaxLoc(res)
                    if score < self.threshold:
                        break
                    found.append(Detection(card, x + tw // 2, y + th // 2, float(score)))
                    # suppress this peak so the next one is a different instance
                    cv2.rectangle(res, (x - tw // 2, y - th // 2), (x + tw // 2, y + th // 2), -1.0, -1)
        # Non-max suppression across all cards: overlapping hits keep only the best one,
        # so two similar-looking cards cannot both fire for one unit.
        found.sort(key=lambda d: -d.score)
        kept: list[Detection] = []
        for d in found:
            if all(np.hypot(d.x - k.x, d.y - k.y) > self.radius for k in kept):
                kept.append(d)
        return kept

    def process(self, flat: np.ndarray, t: float, roi, only: set[str] | None = None) -> list[Detection]:
        """Feed one flattened frame; returns cards that were *newly* played this frame."""
        x0, y0, w, h = roi
        zone = flat[y0 : y0 + h, x0 : x0 + w]
        cands = self._candidates(zone, only) if self.templates else []
        for c in cands:
            c.x += x0
            c.y += y0
        self.last_candidates = cands

        # associate candidates with existing tracks of the same card (greedy, nearest first)
        pairs = []
        for ci, c in enumerate(cands):
            for ti, tr in enumerate(self.tracks):
                if tr.card != c.card:
                    continue
                dist = np.hypot(c.x - tr.x, c.y - tr.y)
                if dist <= self.radius + self.speed * max(0.0, t - tr.last_seen):
                    pairs.append((dist, ci, ti))
        used_c, used_t = set(), set()
        for _, ci, ti in sorted(pairs):
            if ci in used_c or ti in used_t:
                continue
            used_c.add(ci)
            used_t.add(ti)
            tr, c = self.tracks[ti], cands[ci]
            tr.x, tr.y, tr.last_seen = c.x, c.y, t
            tr.hits += 1
        for ci, c in enumerate(cands):
            if ci not in used_c:
                self.tracks.append(_Track(c.card, c.x, c.y, 1, t))

        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.forget_seconds]

        new: list[Detection] = []
        for tr in self.tracks:
            if not tr.emitted and tr.hits >= self.confirm_frames and tr.last_seen == t:
                tr.emitted = True
                score = max((c.score for c in cands if c.card == tr.card and c.x == tr.x and c.y == tr.y), default=0.0)
                new.append(Detection(tr.card, int(tr.x), int(tr.y), score))
        return new

    def draw(self, img: np.ndarray, roi) -> None:
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 1)
        for tr in self.tracks:
            color = (0, 0, 255) if tr.emitted else (0, 200, 255)
            cv2.circle(img, (int(tr.x), int(tr.y)), 18, color, 2)
            cv2.putText(img, tr.card, (int(tr.x) - 30, int(tr.y) - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
