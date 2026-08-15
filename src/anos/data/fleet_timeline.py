"""Fleet availability as a curve over time, not a constant.

Air India's fleet is mid-transformation: ~190 aircraft in service against an order
book of ~560, with a cabin-retrofit programme simultaneously taking airframes out of
service. A model that treats "fleet size" as one number will over-promise capacity in
exactly the years the airline is growing fastest.

This module answers one question: **on date D, how many airframes of type T can the
network plan actually count on?** That means starting from the count on the books and
subtracting the things that make an aircraft unflyable on any given day:

  nominal   = in-service at anchor date + deliveries to date - retirements to date
  available = nominal - retrofit downtime - heavy maintenance - unscheduled downtime

Delivery dates are forecasts. Both OEMs have already slipped dates on this order book,
so `probabilistic=True` blends the nominal delivery curve with a slipped one, weighted
by the planner's confidence in each programme. Use it whenever a decision depends on
capacity more than a year out.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from anos.config import Params, load_params
from anos.data.loaders import (
    load_aircraft_types,
    load_fleet_config,
    load_maintenance_programmes,
)
from anos.models import FleetSnapshot

# A programme the planner has no confidence in is assumed to slip by up to this much.
MAX_SLIP_MONTHS = 24
DAYS_PER_MONTH = 30.44


def _ramp_fraction(target: date, start: date, end: date) -> float:
    """Fraction of a delivery/retirement programme complete by `target`.

    Linear between start and end, clamped outside. Real delivery streams are lumpier
    than this, but absent a slot-level schedule from the OEM a linear ramp is the
    honest assumption -- and the scenario engine is where lumpiness gets tested.
    """
    if target <= start:
        return 0.0
    if target >= end:
        return 1.0
    span = (end - start).days
    if span <= 0:
        return 1.0
    return (target - start).days / span


def _delivered_count(entry: dict[str, Any], target: date, probabilistic: bool) -> float:
    """Aircraft delivered from one order by `target`.

    In probabilistic mode the result is an expected value: with probability
    `confidence` the programme holds its published window, otherwise it slips by an
    amount inversely proportional to that confidence.
    """
    on_time = _ramp_fraction(target, entry["start"], entry["end"]) * entry["count"]
    if not probabilistic:
        return on_time

    confidence = float(entry.get("confidence", 1.0))
    if confidence >= 1.0:
        return on_time

    slip_days = int((1.0 - confidence) * MAX_SLIP_MONTHS * DAYS_PER_MONTH)
    slipped = (
        _ramp_fraction(
            target,
            entry["start"] + timedelta(days=slip_days),
            entry["end"] + timedelta(days=slip_days),
        )
        * entry["count"]
    )
    return confidence * on_time + (1.0 - confidence) * slipped


def _concurrent_in_shop(entry: dict[str, Any], target: date) -> float:
    """Aircraft simultaneously out of service for a retrofit programme.

    A programme putting `count` airframes through a shop over a window, each losing
    `downtime_days`, has this many in the shop at any moment while it runs.
    """
    if not (entry["start"] <= target <= entry["end"]):
        return 0.0
    window_days = max((entry["end"] - entry["start"]).days, 1)
    return entry["count"] * entry["downtime_days"] / window_days


def _heavy_maintenance_fraction(code: str, programmes: dict[str, Any]) -> float:
    """Steady-state fraction of a fleet sitting in a C- or D-check.

    C-checks every ~20-30 months costing ~2-4 weeks, plus D-checks every ~8-12 years
    costing ~4-6 weeks, remove a few percent of any mature fleet permanently. A-checks
    are excluded: they are worked overnight and do not cost a flying day. Intervals
    are per aircraft type -- see `data/aircraft_types.yaml`'s `maintenance` block.
    """
    prog = programmes[code]
    c_cycle_days = prog.c_check_interval_months * DAYS_PER_MONTH
    d_cycle_days = prog.d_check_interval_years * 365.25
    return (
        prog.c_check_downtime_days / c_cycle_days
        + prog.d_check_downtime_days / d_cycle_days
    )


def fleet_on(
    target: date,
    *,
    probabilistic: bool = False,
    config: dict[str, Any] | None = None,
    params: Params | None = None,
) -> FleetSnapshot:
    """Build the fleet snapshot for a date.

    Args:
        target: the date to evaluate.
        probabilistic: risk-adjust delivery dates using per-programme confidence.
        config: fleet configuration override (defaults to data/fleet.yaml).
        params: economic parameters override.
    """
    cfg = config if config is not None else load_fleet_config()
    par = params or load_params()
    types = load_aircraft_types()
    programmes = load_maintenance_programmes()

    anchor: date = cfg["as_of"]
    nominal: dict[str, float] = {code: 0.0 for code in types}
    for code, count in cfg.get("in_service", {}).items():
        nominal[code] = float(count)

    # Deliveries only count from the anchor date forward; anything before it is
    # already reflected in the in-service count.
    if target >= anchor:
        for entry in cfg.get("orders", []) or []:
            nominal[entry["type"]] += _delivered_count(entry, target, probabilistic)

        for entry in cfg.get("retirements", []) or []:
            retired = _ramp_fraction(target, entry["start"], entry["end"]) * entry["count"]
            nominal[entry["type"]] = max(0.0, nominal[entry["type"]] - retired)

    in_retrofit: dict[str, float] = {}
    for entry in cfg.get("retrofits", []) or []:
        shop = _concurrent_in_shop(entry, target)
        if shop > 0:
            in_retrofit[entry["type"]] = in_retrofit.get(entry["type"], 0.0) + shop

    in_maintenance: dict[str, float] = {}
    available: dict[str, float] = {}
    for code, count in nominal.items():
        if count <= 0:
            available[code] = 0.0
            continue
        heavy = count * _heavy_maintenance_fraction(code, programmes)
        unscheduled = count * par.unscheduled_downtime_rate(types[code].generation)
        in_maintenance[code] = heavy + unscheduled
        available[code] = max(
            0.0, count - heavy - unscheduled - in_retrofit.get(code, 0.0)
        )

    return FleetSnapshot(
        as_of=target,
        nominal=nominal,
        available=available,
        in_maintenance=in_maintenance,
        in_retrofit=in_retrofit,
    )


def daily_aircraft_hours(snapshot: FleetSnapshot) -> dict[str, float]:
    """Flyable aircraft-hours per day, by type.

    This is the capacity the optimiser spends. It is the product of available
    airframes and the type's planned daily utilisation -- a new A350 sustains ~15 h/day,
    a legacy A319 closer to 10.
    """
    types = load_aircraft_types()
    return {
        code: count * types[code].max_daily_util_h
        for code, count in snapshot.available.items()
        if count > 0
    }


def timeline(
    start: date,
    end: date,
    *,
    step_days: int = 90,
    probabilistic: bool = False,
) -> list[FleetSnapshot]:
    """Fleet snapshots at regular intervals -- the capacity curve itself."""
    out: list[FleetSnapshot] = []
    current = start
    while current <= end:
        out.append(fleet_on(current, probabilistic=probabilistic))
        current += timedelta(days=step_days)
    return out
