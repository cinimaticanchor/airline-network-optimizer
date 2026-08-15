"""The shared-reference-clock bucket arithmetic the time-space fleet-assignment
core's fleet-count cap depends on being correct."""

from __future__ import annotations

import pytest

from anos.network.timespace_clock import (
    avail_bucket,
    bucket_midpoint_hour,
    bucket_of,
    n_buckets,
    straddles_anchor,
)

CFG = {"buckets": 8, "anchor_hour": 3.0}  # 3h buckets: [0,3),[3,6),...,[21,24)


def test_n_buckets_reads_config():
    assert n_buckets(CFG) == 8


def test_bucket_of_boundaries():
    assert bucket_of(0.0, CFG) == 0
    assert bucket_of(2.99, CFG) == 0
    assert bucket_of(3.0, CFG) == 1
    assert bucket_of(23.99, CFG) == 7


def test_bucket_of_wraps_past_midnight():
    assert bucket_of(24.5, CFG) == bucket_of(0.5, CFG) == 0


def test_bucket_midpoint_hour():
    assert bucket_midpoint_hour(0, CFG) == pytest.approx(1.5)
    assert bucket_midpoint_hour(7, CFG) == pytest.approx(22.5)


def test_avail_bucket_short_duration_stays_in_next_bucket():
    idx, duration_buckets = avail_bucket(0, 2.0, CFG)
    assert idx == 1  # 1.5h + 2h = 3.5h, falls in [3,6)
    assert duration_buckets == pytest.approx(2.0 / 3.0)


def test_avail_bucket_wraps_past_midnight():
    idx, _ = avail_bucket(7, 3.0, CFG)  # departs 22:30, +3h = 01:30
    assert idx == 0


def test_avail_bucket_flags_a_full_cyclic_day():
    _, duration_buckets = avail_bucket(0, 25.0, CFG)
    assert duration_buckets >= n_buckets(CFG)  # caller's job to reject this


# -- straddles_anchor -----------------------------------------------------------


def test_straddles_anchor_includes_departure_bucket():
    assert straddles_anchor(0, 1, anchor=0, buckets=8) is True


def test_straddles_anchor_excludes_arrival_bucket():
    assert straddles_anchor(0, 1, anchor=1, buckets=8) is False


def test_straddles_anchor_false_outside_the_interval():
    assert straddles_anchor(0, 1, anchor=4, buckets=8) is False


def test_straddles_anchor_degenerate_zero_length_is_false():
    assert straddles_anchor(3, 3, anchor=3, buckets=8) is False


def test_straddles_anchor_wraps_past_midnight():
    # Interval [7, 1) wrapping: contains 7 and 0, not 1 or 2.
    assert straddles_anchor(7, 1, anchor=7, buckets=8) is True
    assert straddles_anchor(7, 1, anchor=0, buckets=8) is True
    assert straddles_anchor(7, 1, anchor=1, buckets=8) is False
    assert straddles_anchor(7, 1, anchor=2, buckets=8) is False


def test_straddles_anchor_multi_bucket_interval():
    # Interval [2, 5): contains 2,3,4; not 5,6,1.
    for anchor in (2, 3, 4):
        assert straddles_anchor(2, 5, anchor=anchor, buckets=8) is True
    for anchor in (5, 6, 1):
        assert straddles_anchor(2, 5, anchor=anchor, buckets=8) is False
