"""Fuel burn is not a constant over an aircraft's life -- a turbofan's specific fuel
consumption creeps upward as its hot section wears, and every type's `fuel_burn_kgh`
in `aircraft_types.yaml` is a fresh-engine figure that ignores that.

This module estimates each type's fleet-average age and applies a degradation curve
grounded in real published research (see `data/engine_degradation.yaml`'s header
comment for full citations) rather than any per-tail engine-health data, which does
not exist anywhere in this project. Age is estimated from data this project already
has: `fleet_timeline`'s own delivery-ramp math for aircraft still arriving, plus a
stated average age for the current in-service baseline.

Entirely opt-in: nothing here changes a solve unless a caller explicitly asks for
`degraded_fuel_burn_by_type()` and threads it into `costs.economics.build_economics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from anos.data.fleet_timeline import _ramp_fraction
from anos.data.loaders import load_aircraft_types, load_engine_degradation_config, load_fleet_config
from anos.models import AircraftType

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class DegradationAssumptions:
    """Typed view over engine_degradation.yaml, same thin-wrapper pattern as
    `Params`/`GrowthAssumptions`/`ExpansionAssumptions`."""

    raw: dict[str, Any]

    @property
    def d_max(self) -> float:
        return float(self.raw["curve"]["d_max"])

    @property
    def age_ref_years(self) -> float:
        return float(self.raw["curve"]["age_ref_years"])

    def baseline_age(self, type_code: str) -> float | None:
        """Stated average age of the current in-service count of this type, or
        None if it has none (purely on order)."""
        value = self.raw.get("baseline_avg_age_years", {}).get(type_code)
        return None if value is None else float(value)

    def sensitivity_multiplier(self, band: str) -> float:
        return float(self.raw["sensitivity"][band])

    def scaled(self, multiplier: float) -> DegradationAssumptions:
        """A copy with `d_max` scaled -- how `anos.planning` sensitivity scenarios
        get a low/central/high band without three separate data files."""
        return DegradationAssumptions(
            raw={**self.raw, "curve": {**self.raw["curve"], "d_max": self.d_max * multiplier}}
        )


@lru_cache(maxsize=1)
def load_degradation_assumptions(path: Path | None = None) -> DegradationAssumptions:
    return DegradationAssumptions(raw=load_engine_degradation_config(path))


def sensitivity_bands(
    assumptions: DegradationAssumptions | None = None,
) -> dict[str, DegradationAssumptions]:
    """The named low/central/high degradation assumption sets, spanning the real
    cited 2-6% fuel-burn-impact range (see engine_degradation.yaml's header) --
    for genuine sensitivity analysis, not a single fabricated point estimate."""
    a = assumptions or load_degradation_assumptions()
    bands = a.raw["sensitivity"]
    return {band: a.scaled(a.sensitivity_multiplier(band)) for band in bands}


def degradation_pct(age_years: float, assumptions: DegradationAssumptions | None = None) -> float:
    """Fractional fuel-burn increase over a fresh engine, at this age.

    `degradation_pct(0) == 0`; grows logarithmically toward `d_max` at
    `age_ref_years`, per the arXiv A320-family empirical study this is calibrated
    to (see engine_degradation.yaml's header comment).
    """
    if age_years <= 0:
        return 0.0
    a = assumptions or load_degradation_assumptions()
    return a.d_max * math.log(1 + age_years) / math.log(1 + a.age_ref_years)


def _baseline_remaining_count_and_age(
    type_code: str, fleet_config: dict[str, Any], target: date, assumptions: DegradationAssumptions
) -> tuple[float, float]:
    """(count, average age in years) of the current in-service baseline still
    flying at `target`, after subtracting whatever retirement ramp has completed --
    mirrors `fleet_timeline.fleet_on`'s own retirement math."""
    base_count = float(fleet_config.get("in_service", {}).get(type_code, 0.0))
    if base_count <= 0:
        return 0.0, 0.0

    retired = 0.0
    for entry in fleet_config.get("retirements", []) or []:
        if entry["type"] == type_code:
            retired += _ramp_fraction(target, entry["start"], entry["end"]) * entry["count"]
    remaining = max(0.0, base_count - retired)
    if remaining <= 0:
        return 0.0, 0.0

    anchor_age = assumptions.baseline_age(type_code)
    if anchor_age is None:
        return 0.0, 0.0

    anchor: date = fleet_config["as_of"]
    elapsed_years = (target - anchor).days / DAYS_PER_YEAR
    return remaining, anchor_age + elapsed_years


def _tranche_delivered_count_and_age(entry: dict[str, Any], target: date) -> tuple[float, float]:
    """(count, average age in years) of one order tranche's delivered-so-far
    aircraft at `target`, assuming the same linear delivery ramp `fleet_timeline`
    uses. The average delivery date of a linear ramp's delivered-so-far portion is
    the midpoint of [start, min(target, end)] -- not `target - start`, which would
    overstate age since most of that span's aircraft haven't delivered yet early on.
    """
    start, end = entry["start"], entry["end"]
    delivered = _ramp_fraction(target, start, end) * entry["count"]
    if delivered <= 0:
        return 0.0, 0.0

    effective_end = min(target, end)
    midpoint = start + (effective_end - start) / 2
    age_years = (target - midpoint).days / DAYS_PER_YEAR
    return delivered, age_years


def estimate_fleet_average_age(
    type_code: str,
    target_date: date,
    *,
    fleet_config: dict[str, Any] | None = None,
    assumptions: DegradationAssumptions | None = None,
) -> float:
    """Delivery-count-weighted average age (years) of `type_code`'s fleet at
    `target_date`, combining the in-service baseline and every order tranche.

    Returns 0.0 for a type with no aircraft at all at `target_date` -- callers
    computing a fuel-burn adjustment should treat that as "no degradation data
    needed" rather than "brand new," since it means the type isn't flying.
    """
    cfg = fleet_config if fleet_config is not None else load_fleet_config()
    a = assumptions or load_degradation_assumptions()

    base_count, base_age = _baseline_remaining_count_and_age(type_code, cfg, target_date, a)

    order_count = 0.0
    order_weighted_age = 0.0
    for entry in cfg.get("orders", []) or []:
        if entry["type"] != type_code:
            continue
        count, age = _tranche_delivered_count_and_age(entry, target_date)
        order_count += count
        order_weighted_age += count * age

    total_count = base_count + order_count
    if total_count <= 0:
        return 0.0
    return (base_count * base_age + order_weighted_age) / total_count


def degraded_fuel_burn_kgh(
    ac: AircraftType, age_years: float, assumptions: DegradationAssumptions | None = None
) -> float:
    return ac.fuel_burn_kgh * (1 + degradation_pct(age_years, assumptions))


def degraded_fuel_burn_by_type(
    target_date: date,
    *,
    fleet_config: dict[str, Any] | None = None,
    assumptions: DegradationAssumptions | None = None,
) -> dict[str, float]:
    """Every type's degraded `fuel_burn_kgh` at `target_date`, ready to hand to
    `anos.costs.economics.build_economics`'s `degraded_fuel_burn` parameter."""
    cfg = fleet_config if fleet_config is not None else load_fleet_config()
    a = assumptions or load_degradation_assumptions()
    types = load_aircraft_types()
    return {
        code: degraded_fuel_burn_kgh(
            ac, estimate_fleet_average_age(code, target_date, fleet_config=cfg, assumptions=a), a
        )
        for code, ac in types.items()
    }
