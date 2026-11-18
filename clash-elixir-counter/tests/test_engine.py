import pytest

from elixir_counter.cards import CardDB, normalize
from elixir_counter.engine import ElixirTracker, Phase, parse_schedule


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def tr(clock):
    t = ElixirTracker(CardDB.load(), clock=clock)
    t.start()
    return t


def test_normalize_and_resolve():
    db = CardDB.load()
    assert normalize("Mini P.E.K.K.A") == "mini_pekka"
    assert normalize("X-Bow") == "x_bow"
    assert db.resolve("hog") == "hog_rider"
    assert db.resolve("the log") == "the_log"
    assert db.resolve("muskateer") == "musketeer"
    assert db.cost("Three Musketeers") == 9


def test_starts_at_five_and_regenerates(tr, clock):
    assert tr.elixir() == pytest.approx(5)
    clock.t += 2.8
    assert tr.elixir() == pytest.approx(6)
    clock.t += 2.8 * 2
    assert tr.elixir() == pytest.approx(8)


def test_caps_at_ten_and_counts_leak(tr, clock):
    clock.t += 2.8 * 7  # would be 12
    assert tr.elixir() == pytest.approx(10)
    assert tr.leaked == pytest.approx(2)
    clock.t += 2.8
    tr.elixir()
    assert tr.leaked == pytest.approx(3)


def test_play_subtracts_cost(tr, clock):
    tr.play("Hog Rider")
    assert tr.elixir() == pytest.approx(1)
    clock.t += 2.8
    assert tr.elixir() == pytest.approx(2)


def test_underestimate_self_corrects(tr, clock):
    tr.play("golem")  # 8 cost at 5 estimated: they must have had 8
    p = tr.plays[-1]
    assert p.correction == pytest.approx(3)
    assert tr.elixir() == pytest.approx(0)


def test_double_elixir_after_two_minutes(tr, clock):
    tr.play("pekka")  # 5 -> 0 (corrected)
    clock.t += 119  # elixir capped at 10 by now
    tr.play("golem")  # -> 2
    tr.play("knight")  # -> 0 (corrected)
    clock.t += 1  # crosses 120 s: 1 s at x1
    clock.t += 1.4 * 3  # 3 elixir at x2
    assert tr.elixir() == pytest.approx(1 / 2.8 + 3)


def test_phase_boundary_split_inside_one_update(clock):
    tr = ElixirTracker(CardDB.load(), schedule=(Phase(0, 1, "a"), Phase(10, 2, "b")), clock=clock)
    tr.start()
    tr.adjust(-5)  # 0 elixir
    clock.t += 15.6  # 5.6 s @x1 + 5.6 s @x2... starts at 0 elapsed, so 10 s @x1 then 5.6 s @x2
    assert tr.elixir() == pytest.approx(10 / 2.8 + 5.6 * 2 / 2.8)


def test_manual_multiplier(tr, clock):
    tr.adjust(-5)
    tr.set_multiplier(3)
    clock.t += 2.8
    assert tr.elixir() == pytest.approx(3)


def test_mirror_costs_previous_plus_one(tr, clock):
    clock.t += 2.8 * 5  # 10
    tr.play("knight")  # 7
    p = tr.play("mirror")
    assert p.cost == 4 and p.mirrored == "knight"
    assert tr.elixir() == pytest.approx(3)


def test_mirror_first_card_raises(tr):
    with pytest.raises(ValueError):
        tr.play("mirror")
    assert tr.plays == []


def test_undo_restores_and_keeps_regen(tr, clock):
    tr.play("fireball")  # 1
    clock.t += 2.8
    assert tr.undo()
    assert tr.elixir() == pytest.approx(6)
    assert tr.plays == []


def test_cycle_tracking(tr, clock):
    deck = ["knight", "archers", "fireball", "the_log", "hog_rider", "ice_spirit", "cannon", "musketeer"]
    for c in deck:
        clock.t += 10
        tr.play(c)
    assert tr.deck == deck
    assert tr.not_in_hand() == deck[4:]
    assert tr.in_hand() == deck[:4]
    assert tr.next_to_cycle_in() == "hog_rider"
    clock.t += 10
    tr.play("knight")
    assert tr.in_hand() == ["archers", "fireball", "the_log", "hog_rider"]


def test_sync_clock(tr, clock):
    tr.sync_clock(60)  # 1:00 left -> 120 s elapsed
    assert tr.elapsed() == pytest.approx(120)
    assert tr.multiplier_at(tr.elapsed()) == 2


def test_parse_schedule():
    s = parse_schedule("0:1,120:2,240:3")
    assert [p.multiplier for p in s] == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_schedule("10:1")


def test_play_before_start_autostarts(clock):
    tr = ElixirTracker(CardDB.load(), clock=clock)
    tr.play("knight")
    assert tr.running and tr.elixir() == pytest.approx(2)
