"""Hand-built-fixture tests for anos.planning.constraint_detection.

Fixtures build TrajectoryPoint/NetworkPlan/MarketPlan objects directly rather than
solving, so each test controls exactly the utilisation/spill/slot numbers it needs --
the point is to pin the detection logic itself, not re-derive it from a real solve.
"""

from __future__ import annotations

from datetime import date

import pytest

from anos.models import (
    FleetSnapshot,
    LadderResult,
    Market,
    MarketPlan,
    NetworkPlan,
    TrajectoryPoint,
)
from anos.planning.constraint_detection import (
    detect_signals,
    first_persistent_binding_year,
    is_slot_saturated,
)

# -- first_persistent_binding_year -------------------------------------------------


def test_single_spike_does_not_trigger():
    assert first_persistent_binding_year([0.5, 0.95, 0.5], [2027, 2028, 2029], 0.9) is None


def test_two_consecutive_checkpoints_trigger_on_the_first_of_the_pair():
    assert first_persistent_binding_year([0.5, 0.95, 0.96], [2027, 2028, 2029], 0.9) == 2028


def test_returns_the_first_qualifying_run_not_a_later_one():
    values = [0.95, 0.95, 0.5, 0.95, 0.95]
    years = [2027, 2028, 2029, 2030, 2031]
    assert first_persistent_binding_year(values, years, 0.9) == 2027


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        first_persistent_binding_year([0.5], [2027, 2028], 0.9)


# -- fixtures -----------------------------------------------------------------------


def _market(market_id: str, origin: str, destination: str) -> Market:
    return Market(
        market_id=market_id,
        origin=origin,
        destination=destination,
        region="domestic_trunk",
        base_daily_demand=100.0,
        avg_fare_usd=5000.0,
        business_mix=0.2,
        competition_index=0.5,
        min_daily_freq=0,
        max_daily_freq=10,
        seasonality_profile="flat",
        strategic=False,
    )


def _market_plan(market_id: str, *, total_frequency: int, demand_effective: float, pax_carried: float) -> MarketPlan:
    return MarketPlan(
        market_id=market_id,
        frequencies={},
        total_frequency=total_frequency,
        seats_offered=max(int(pax_carried), 1),
        pax_carried=pax_carried,
        demand_effective=demand_effective,
        revenue_usd=0.0,
        cost_usd=0.0,
    )


def _checkpoint(
    year: int,
    *,
    market_plans: list[MarketPlan],
    markets: list[Market],
    fleet_used: dict[str, float] | None = None,
    fleet_available: dict[str, float] | None = None,
) -> TrajectoryPoint:
    as_of = date(year, 3, 1)
    plan = NetworkPlan(
        as_of=as_of,
        month=3,
        market_plans=market_plans,
        fleet_used=fleet_used or {},
        fleet_available=fleet_available or {},
        solver_status="OPTIMAL",
        objective_usd=0.0,
        solve_seconds=0.0,
    )
    return TrajectoryPoint(
        year=year,
        as_of=as_of,
        plan=plan,
        fleet=FleetSnapshot(as_of=as_of),
        grown_markets=tuple(markets),
    )


# IXM has a real daily_slot_cap of 12 -- small enough that a 12/day market pins it.
SAT_MARKET = _market("SAT-MKT", "IXM", "PAT")
# GAU/BBI have caps of 24/18 -- a 2/day market leaves them nowhere near saturated.
FREE_MARKET = _market("FREE-MKT", "GAU", "BBI")


def test_is_slot_saturated_true_when_an_endpoint_is_pinned():
    checkpoint = _checkpoint(
        2027,
        market_plans=[_market_plan("SAT-MKT", total_frequency=12, demand_effective=100, pax_carried=90)],
        markets=[SAT_MARKET],
    )
    assert is_slot_saturated("SAT-MKT", checkpoint) is True


def test_is_slot_saturated_false_when_well_under_cap():
    checkpoint = _checkpoint(
        2027,
        market_plans=[_market_plan("FREE-MKT", total_frequency=2, demand_effective=100, pax_carried=90)],
        markets=[FREE_MARKET],
    )
    assert is_slot_saturated("FREE-MKT", checkpoint) is False


# -- detect_signals -------------------------------------------------------------


YEARS = (2027, 2028, 2029)


def _spill_ladder(spill_shares: list[float]) -> LadderResult:
    """A ladder (up to 3 checkpoints) where both markets share the same spill
    trajectory, one slot-saturated and one not, plus a fleet type whose utilisation
    follows the same trajectory."""
    checkpoints = []
    for year, share in zip(YEARS[: len(spill_shares)], spill_shares, strict=True):
        demand = 100.0
        pax = demand * (1 - share)
        checkpoints.append(
            _checkpoint(
                year,
                market_plans=[
                    _market_plan("SAT-MKT", total_frequency=12, demand_effective=demand, pax_carried=pax),
                    _market_plan("FREE-MKT", total_frequency=2, demand_effective=demand, pax_carried=pax),
                ],
                markets=[SAT_MARKET, FREE_MARKET],
                fleet_used={"TESTTYPE": 5.0 + share * 20.0},
                fleet_available={"TESTTYPE": 10.0},
            )
        )
    return LadderResult(checkpoints=tuple(checkpoints))


def test_persistent_spill_triggers_but_slot_saturated_market_is_not_actionable():
    ladder = _spill_ladder([0.05, 0.20, 0.25])
    signals = detect_signals(ladder)

    by_subject = {s.subject: s for s in signals if s.signal_type == "market_spill"}
    assert by_subject["FREE-MKT"].actionable is True
    assert by_subject["FREE-MKT"].first_binding_year == 2028
    assert by_subject["SAT-MKT"].actionable is False
    assert by_subject["SAT-MKT"].first_binding_year == 2028


def test_single_year_spike_produces_no_signal():
    ladder = _spill_ladder([0.05, 0.20, 0.05])
    signals = detect_signals(ladder)
    assert not any(s.subject in ("FREE-MKT", "SAT-MKT") for s in signals)


def test_fleet_utilisation_signal_fires_when_persistently_binding():
    ladder = _spill_ladder([0.05, 0.20, 0.25])  # drives TESTTYPE utilisation to 0.6, 0.9, 1.0
    signals = detect_signals(ladder)
    fleet_signals = {s.subject: s for s in signals if s.signal_type == "fleet_utilisation"}
    assert "TESTTYPE" in fleet_signals
    assert fleet_signals["TESTTYPE"].actionable is True


def test_short_ladder_returns_no_signals():
    ladder = LadderResult(checkpoints=(_spill_ladder([0.5]).checkpoints[0],))
    assert detect_signals(ladder) == []
