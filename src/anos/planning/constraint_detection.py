"""Turn a multi-year `LadderResult` into a short list of persistent constraints worth
acting on.

Two traps this module exists to avoid, both learned the hard way earlier in this
project's life:

1. **Solver-gap noise looks like a trend.** CP-SAT's own relative gap moves a
   checkpoint's utilisation by roughly 0.5-1% run to run, unrelated to demand growth.
   `first_persistent_binding_year` requires **two consecutive** checkpoints at or
   above threshold before it calls something a trend, not one.
2. **A slot-saturated market needs slots, not aircraft.** This project's own
   headline finding was that adding aircraft to a network pinned at its airports'
   slot caps makes the plan worse, not better. `is_slot_saturated` is the guard: a
   market-spill signal traced to a saturated airport is marked `actionable=False` so
   `anos.planning.fleet_expansion` never proposes an order to fix it.
"""

from __future__ import annotations

from collections.abc import Sequence

from anos.data.loaders import load_airports
from anos.models import ConstraintSignal, LadderResult, TrajectoryPoint
from anos.optimize.tail_feasibility import slot_usage_by_airport

# Deliberately below tail_feasibility's own 0.95 utilisation-warning line, so a
# fleet-expansion trigger fires with runway to act on before ops sees a warning.
FLEET_UTILISATION_THRESHOLD = 0.90

# Matches tail_feasibility.SPILL_WARNING -- the same "leaving money on the table"
# bar used to flag a market in the single-day feasibility check.
MARKET_SPILL_THRESHOLD = 0.15

# The slot-usage share, at a market's origin or destination airport, above which
# that airport is considered saturated. Matches tail_feasibility's own slot-warning
# threshold (`used >= cap * 0.95`).
SLOT_SATURATION_SHARE = 0.95


def first_persistent_binding_year(
    values: Sequence[float], years: Sequence[int], threshold: float
) -> int | None:
    """The year of the first of two *consecutive* values at/above `threshold`.

    A single spike returns None -- only a run of two or more counts as a trend, the
    direct fix for conflating one-off solver-gap noise with genuine demand growth.
    """
    if len(values) != len(years):
        raise ValueError("values and years must be the same length")
    for i in range(len(values) - 1):
        if values[i] >= threshold and values[i + 1] >= threshold:
            return years[i]
    return None


def is_slot_saturated(
    market_id: str,
    checkpoint: TrajectoryPoint,
    *,
    threshold: float = SLOT_SATURATION_SHARE,
) -> bool:
    """True if either endpoint airport of `market_id` is at/above `threshold` of its
    daily slot cap under this checkpoint's plan."""
    markets_by_id = {m.market_id: m for m in checkpoint.grown_markets}
    market = markets_by_id[market_id]
    ports = load_airports()
    usage = slot_usage_by_airport(checkpoint.plan, markets_by_id)
    return any(
        usage.get(iata, 0) >= threshold * ports[iata].daily_slot_cap
        for iata in (market.origin, market.destination)
    )


def detect_signals(
    ladder: LadderResult,
    *,
    fleet_utilisation_threshold: float = FLEET_UTILISATION_THRESHOLD,
    market_spill_threshold: float = MARKET_SPILL_THRESHOLD,
) -> list[ConstraintSignal]:
    """Scan a ladder for persistent fleet-utilisation and market-spill constraints.

    Two checkpoints minimum are required to detect anything at all -- a single-year
    ladder has no consecutive pair to confirm a trend against.
    """
    if len(ladder.checkpoints) < 2:
        return []

    years = [c.year for c in ladder.checkpoints]
    signals: list[ConstraintSignal] = []

    type_codes = sorted({t for c in ladder.checkpoints for t in c.plan.fleet_available})
    for type_code in type_codes:
        series = ladder.utilisation_series(type_code)
        first_year = first_persistent_binding_year(series, years, fleet_utilisation_threshold)
        if first_year is None:
            continue
        signals.append(
            ConstraintSignal(
                signal_type="fleet_utilisation",
                subject=type_code,
                first_binding_year=first_year,
                last_checkpoint_year=years[-1],
                detail=(
                    f"{type_code} utilisation reaches {fleet_utilisation_threshold:.0%}+ for "
                    f"two consecutive checkpoints starting {first_year}"
                ),
            )
        )

    market_ids = sorted({m.market_id for c in ladder.checkpoints for m in c.grown_markets})
    for market_id in market_ids:
        spill_series = []
        for c in ladder.checkpoints:
            mp = next((p for p in c.plan.market_plans if p.market_id == market_id), None)
            share = mp.spilled_demand / mp.demand_effective if mp and mp.demand_effective > 0 else 0.0
            spill_series.append(share)

        first_year = first_persistent_binding_year(spill_series, years, market_spill_threshold)
        if first_year is None:
            continue

        latest = ladder.checkpoints[-1]
        saturated = is_slot_saturated(market_id, latest)
        detail = (
            f"{market_id} spills {market_spill_threshold:.0%}+ of demand for two consecutive "
            f"checkpoints starting {first_year}"
        )
        if saturated:
            detail += " -- but the market is slot-saturated, not fleet-constrained"

        signals.append(
            ConstraintSignal(
                signal_type="market_spill",
                subject=market_id,
                first_binding_year=first_year,
                last_checkpoint_year=years[-1],
                detail=detail,
                actionable=not saturated,
            )
        )

    return signals
