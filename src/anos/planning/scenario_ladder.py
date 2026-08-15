"""Walk the network forward year by year, growing demand as we go.

A single `solve_network` call answers "what should we fly on one representative
day?" A multi-year plan needs to know how that answer *changes* as demand grows and
the order book delivers -- not just what the fleet can carry today. `run_ladder`
re-solves at each annual checkpoint against that year's grown markets and that year's
`fleet_on()` snapshot (which already correctly simulates delivery/retirement/retrofit
ramps for any future date), producing a `LadderResult` the rest of `anos.planning`
builds on.

Checkpoints hold the month fixed (default March) throughout the horizon. This is
deliberate: it isolates demand-growth effects from monthly seasonality by controlling
for the one thing seasonality would otherwise confound.

Mirrors `anos.scenarios.engine.run_scenario`'s infeasibility fallback: if a year's
grown demand cannot honour every minimum service floor, it is re-solved with floors
released rather than aborting the whole ladder, and the checkpoint records which
floors went unmet.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from anos.config import Params, load_params
from anos.data.fleet_timeline import fleet_on
from anos.data.loaders import load_markets
from anos.forecast.demand_growth import GrowthAssumptions, grow_markets, load_growth_assumptions
from anos.models import LadderResult, Market, NetworkPlan, TrajectoryPoint
from anos.optimize.fleet_assignment import InfeasibleNetworkError, solve_network


def build_checkpoints(
    start_year: int, end_year: int, *, step_years: int = 1, month: int = 3
) -> list[date]:
    """The checkpoint dates a ladder run will solve at -- the first of `month` in
    each year from `start_year` to `end_year` inclusive, stepped by `step_years`."""
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")
    if step_years < 1:
        raise ValueError(f"step_years must be >= 1, got {step_years}")
    return [date(year, month, 1) for year in range(start_year, end_year + 1, step_years)]


def run_ladder(
    start_year: int,
    end_year: int,
    *,
    step_years: int = 1,
    checkpoint_month: int = 3,
    markets: list[Market] | None = None,
    fleet_config: dict[str, Any] | None = None,
    params: Params | None = None,
    growth: GrowthAssumptions | None = None,
    probabilistic_fleet: bool = False,
    solve_fn: Callable[..., NetworkPlan] = solve_network,
    solve_kwargs: dict[str, Any] | None = None,
) -> LadderResult:
    """Solve one checkpoint per year and return the resulting trajectory.

    Args:
        start_year, end_year: inclusive calendar-year horizon.
        step_years: gap between checkpoints (1 = every year).
        checkpoint_month: month (1-12) used for every checkpoint's date, growth
            evaluation, and seasonality -- held fixed across the whole horizon.
        markets: base markets to grow forward; defaults to `load_markets()`. Always
            treated as the pristine, ungrown baseline, regardless of `start_year`.
        fleet_config: fleet configuration override, passed through to `fleet_on`.
        params: economic parameters override.
        growth: demand-growth assumptions override; defaults to
            `load_growth_assumptions()`.
        probabilistic_fleet: risk-adjust delivery dates (see `fleet_on`).
        solve_fn: the solver to call at each checkpoint -- defaults to
            `solve_network`, overridable to reuse this ladder with
            `solve_extended_network`/`solve_network_timespace`.
        solve_kwargs: extra keyword arguments forwarded to `solve_fn` at every
            checkpoint (e.g. `enable_connections=True`).
    """
    base_markets = markets if markets is not None else load_markets()
    par = params or load_params()
    growth_assumptions = growth or load_growth_assumptions()
    extra_kwargs = dict(solve_kwargs or {})

    checkpoints: list[TrajectoryPoint] = []
    for target in build_checkpoints(start_year, end_year, step_years=step_years, month=checkpoint_month):
        grown = grow_markets(base_markets, target.year, assumptions=growth_assumptions)
        snapshot = fleet_on(
            target, probabilistic=probabilistic_fleet, config=fleet_config, params=par
        )

        try:
            plan = solve_fn(
                target,
                month=checkpoint_month,
                markets=grown,
                fleet=snapshot,
                params=par,
                **extra_kwargs,
            )
            floors_unmeetable = False
            unmet: tuple[str, ...] = ()
        except InfeasibleNetworkError:
            fallback_kwargs = {**extra_kwargs, "enforce_min_service": False}
            plan = solve_fn(
                target,
                month=checkpoint_month,
                markets=grown,
                fleet=snapshot,
                params=par,
                **fallback_kwargs,
            )
            served = {p.market_id: p.total_frequency for p in plan.market_plans}
            unmet = tuple(
                sorted(
                    m.market_id
                    for m in grown
                    if m.min_daily_freq > 0 and served.get(m.market_id, 0) < m.min_daily_freq
                )
            )
            floors_unmeetable = True

        checkpoints.append(
            TrajectoryPoint(
                year=target.year,
                as_of=target,
                plan=plan,
                fleet=snapshot,
                grown_markets=tuple(grown),
                service_floors_unmeetable=floors_unmeetable,
                unmet_floors=unmet,
            )
        )

    return LadderResult(checkpoints=tuple(checkpoints))
