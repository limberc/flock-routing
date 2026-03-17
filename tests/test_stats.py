import threading
import pytest
from privacy_serving.stats import StatsTracker


def test_initial_state():
    t = StatsTracker()
    s = t.snapshot()
    assert s["local"] == 0
    assert s["remote"] == 0
    assert s["total"] == 0
    assert s["local_rate"] == 0.0


def test_record_local():
    t = StatsTracker()
    t.record("local")
    s = t.snapshot()
    assert s["local"] == 1
    assert s["remote"] == 0
    assert s["total"] == 1
    assert s["local_rate"] == 1.0


def test_record_remote():
    t = StatsTracker()
    t.record("remote")
    s = t.snapshot()
    assert s["local"] == 0
    assert s["remote"] == 1
    assert s["local_rate"] == 0.0


def test_mixed_rate():
    t = StatsTracker()
    for _ in range(3):
        t.record("local")
    for _ in range(1):
        t.record("remote")
    s = t.snapshot()
    assert s["total"] == 4
    assert s["local_rate"] == pytest.approx(0.75)


def test_thread_safety():
    t = StatsTracker()
    threads = [threading.Thread(target=t.record, args=("local",)) for _ in range(100)]
    threads += [threading.Thread(target=t.record, args=("remote",)) for _ in range(100)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    s = t.snapshot()
    assert s["total"] == 200
    assert s["local"] == 100
    assert s["remote"] == 100


def test_invalid_destination_raises():
    t = StatsTracker()
    with pytest.raises(ValueError, match="destination"):
        t.record("unknown")
