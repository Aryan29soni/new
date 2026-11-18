"""Detector tests on synthetic frames: a textured 'sprite' pasted onto a noisy arena."""

import cv2
import numpy as np
import pytest

from elixir_counter.vision import Calibration, TemplateDetector

ROI = (0, 0, 540, 400)


def sprite(seed):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 255, (40, 34, 3), dtype=np.uint8)
    return cv2.GaussianBlur(s, (3, 3), 0)


@pytest.fixture
def setup(tmp_path):
    knight, hog = sprite(1), sprite(2)
    for name, img in [("knight", knight), ("hog_rider", hog)]:
        (tmp_path / name).mkdir()
        cv2.imwrite(str(tmp_path / name / "a.png"), img)
    det = TemplateDetector(tmp_path, threshold=0.8, confirm_frames=2)
    return det, knight, hog


def frame(*placed):
    rng = np.random.default_rng(99)
    f = np.full((960, 540, 3), (60, 130, 70), np.uint8)
    f = cv2.add(f, rng.integers(0, 25, f.shape, dtype=np.uint8))
    for img, x, y in placed:
        f[y : y + img.shape[0], x : x + img.shape[1]] = img
    return f


def test_unit_counted_once_while_it_walks(setup):
    det, knight, _ = setup
    plays = []
    for i in range(30):
        plays += det.process(frame((knight, 200, 100 + i * 3)), i / 30, ROI)
    assert [p.card for p in plays] == ["knight"]


def test_single_frame_glitch_ignored(setup):
    det, knight, _ = setup
    assert det.process(frame((knight, 200, 100)), 0.0, ROI) == []
    assert det.process(frame(), 0.033, ROI) == []
    assert det.process(frame(), 0.066, ROI) == []


def test_second_copy_elsewhere_counts(setup):
    det, knight, _ = setup
    plays = []
    for i in range(5):
        plays += det.process(frame((knight, 100, 100)), i / 30, ROI)
    for i in range(5, 10):
        plays += det.process(frame((knight, 100, 100), (knight, 400, 250)), i / 30, ROI)
    assert [p.card for p in plays] == ["knight", "knight"]


def test_two_different_cards(setup):
    det, knight, hog = setup
    plays = []
    for i in range(4):
        plays += det.process(frame((knight, 100, 100), (hog, 350, 200)), i / 30, ROI)
    assert sorted(p.card for p in plays) == ["hog_rider", "knight"]


def test_only_filter_and_outside_roi(setup):
    det, knight, hog = setup
    plays = []
    for i in range(4):
        plays += det.process(frame((knight, 100, 100), (hog, 300, 600)), i / 30, ROI, only={"hog_rider"})
    assert plays == []  # knight filtered out, hog outside the zone


def test_reappears_after_forget_counts_again(setup):
    det, knight, _ = setup
    plays = []
    for i in range(3):
        plays += det.process(frame((knight, 100, 100)), i * 0.03, ROI)
    plays += det.process(frame(), 5.0, ROI)  # gone for longer than forget_seconds
    for i in range(3):
        plays += det.process(frame((knight, 100, 100)), 6 + i * 0.03, ROI)
    assert len(plays) == 2


def test_calibration_warp_roundtrip(tmp_path):
    img = np.zeros((1080, 1920, 3), np.uint8)
    corners = [[600, 100], [1300, 120], [1320, 1000], [580, 980]]
    cv2.fillConvexPoly(img, np.int32(corners), (255, 255, 255))
    cal = Calibration(corners)
    flat = cal.warp(img)
    assert flat.shape == (960, 540, 3)
    assert flat[10:-10, 10:-10].min() > 200  # whole output is inside the screen quad
    cal.save(tmp_path / "c.json")
    again = Calibration.load(tmp_path / "c.json")
    assert again.roi == cal.roi
