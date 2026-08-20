"""Turn a persistent fleet-utilisation constraint into a sized, NPV-evaluated order.

The algorithm is a **greedy, sequential** loop, not a joint optimisation over every
possible future order simultaneously -- a deliberate, stated simplification (see
`recommend_expansion`'s docstring):

    detect -> pick the most urgent signal -> size -> evaluate -> commit or stop

Each accepted candidate is baked into a **rolling baseline** fleet config before the
next signal is detected and evaluated. This matters because contribution is not
additively separable across aircraft types when slots and hub capacity are shared:
evaluating two candidates independently against the same unmodified baseline would
double-count the relief either one provides. Evaluating candidate B against a config
that already contains committed candidate A is the direct fix.

`size_candidate` searches a small integer count the same way this codebase already
enumerates frequency and fare levels -- exactly, one integer at a time, not via a
continuous relaxation. `evaluate_candidate` re-solves WITH and WITHOUT the candidate
at every checkpoint from its delivery year through the horizon end, because a
delivered aircraft never "un-delivers": `fleet_on`'s own delivery ramp means later
years can differ from the trigger year too.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from anos.config import Params, load_params
from anos.data.fleet_timeline import _ramp_fraction, fleet_on
from anos.data.loaders import (
    load_aircraft_types,
    load_expansion_assumptions_config,
    load_fleet_config,
    load_markets,
)
from anos.forecast.demand_growth import GrowthAssumptions, grow_markets, load_growth_assumptions
from anos.models import (
    CandidateOrder,
    ConstraintSignal,
    ExpansionPlan,
    FleetSnapshot,
    LadderResult,
    Market,
    NetworkPlan,
    OrderEvaluation,
)
from anos.optimize.fleet_assignment import InfeasibleNetworkError, solve_network
from anos.planning.constraint_detection import detect_signals
from anos.planning.scenario_ladder import run_ladder

# Both `total_contribution_usd` (a single representative day) and ownership cost
# (published per day) are annualised by this same simplified factor, so cash flows
# stay on one consistent time basis end to end.
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class ExpansionAssumptions:
    """Typed view over expansion_assumptions.yaml, same thin-wrapper pattern as
    `Params` and `GrowthAssumptions`."""

    raw: dict[str, Any]

    @property
    def discount_rate_annual(self) -> float:
        return float(self.raw["discount_rate_annual"])

    def lead_time_years(self, type_code: str, body: str) -> float:
        overrides = self.raw["lead_time_years"]["overrides"]
        if type_code in overrides:
            return float(overrides[type_code])
        return float(self.raw["lead_time_years"]["by_body"][body])

    @property
    def delivery_window_years(self) -> float:
        return float(self.raw["delivery_window_years"])

    @property
    def fleet_utilisation_threshold(self) -> float:
        return float(self.raw["detection"]["fleet_utilisation_threshold"])

    @property
    def target_utilisation_ceiling(self) -> float:
        return float(self.raw["detection"]["target_utilisation_ceiling"])

    @property
    def financial_relative_gap(self) -> float:
        return float(self.raw["financial_relative_gap"])

    @property
    def min_candidate_count(self) -> int:
        return int(self.raw["sizing"]["min_candidate_count"])

    @property
    def max_candidate_count(self) -> int:
        return int(self.raw["sizing"]["max_candidate_count"])

    @property
    def min_years_between_same_type_orders(self) -> int:
        return int(self.raw["review"]["min_years_between_same_type_orders"])


@lru_cache(maxsize=1)
def load_expansion_assumptions(path: Path | None = None) -> ExpansionAssumptions:
    return ExpansionAssumptions(raw=load_expansion_assumptions_config(path))


def _with_added_order(
    fleet_config: dict[str, Any],
    type_code: str,
    count: int,
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    """A deep copy of `fleet_config` with one additional order tranche appended.

    Never mutates the caller's config -- `recommend_expansion` builds its rolling
    baseline by committing one candidate at a time, and each committed step must
    leave earlier evaluations' inputs untouched.
    """
    cfg = copy.deepcopy(fleet_config)
    cfg.setdefault("orders", []).append(
        {"type": type_code, "count": count, "start": start, "end": end, "confidence": 1.0}
    )
    return cfg


def _tighten_for_financial_precision(params: Params, assumptions: ExpansionAssumptions) -> Params:
    """`params` with the solver's relative gap overridden to
    `assumptions.financial_relative_gap`, every other value untouched.

    See expansion_assumptions.yaml's `financial_relative_gap` comment: the default
    gap is fine for a single day's operational plan, but is not tight enough for a
    WITH-minus-WITHOUT contribution subtraction, which amplifies solver noise.
    """
    raw = copy.deepcopy(params.raw)
    raw["solver"]["relative_gap"] = assumptions.financial_relative_gap
    return Params(raw=raw)


def _solve_or_relax(
    target: date,
    *,
    month: int,
    markets: list[Market],
    fleet: FleetSnapshot,
    params: Params,
    solve_kwargs: dict[str, Any] | None = None,
) -> NetworkPlan:
    """Solve one checkpoint, relaxing minimum-service floors if they cannot be met.

    Same fallback `anos.planning.scenario_ladder.run_ladder` uses -- a WITH/WITHOUT
    financial comparison must not fail just because one side alone cannot clear
    every floor. `solve_kwargs` forwards arbitrary opt-in `solve_network` flags
    (e.g. `enable_engine_degradation`) through both the primary and fallback solve.
    """
    extra = solve_kwargs or {}
    try:
        return solve_network(target, month=month, markets=markets, fleet=fleet, params=params, **extra)
    except InfeasibleNetworkError:
        return solve_network(
            target,
            month=month,
            markets=markets,
            fleet=fleet,
            params=params,
            **{**extra, "enforce_min_service": False},
        )


def size_candidate(
    signal: ConstraintSignal,
    *,
    end_year: int,
    fleet_config: dict[str, Any],
    markets: list[Market] | None = None,
    params: Params | None = None,
    growth: GrowthAssumptions | None = None,
    assumptions: ExpansionAssumptions | None = None,
    solve_kwargs: dict[str, Any] | None = None,
) -> CandidateOrder:
    """The smallest integer count of additional `signal.subject`-type aircraft that
    brings utilisation at the horizon's *final* checkpoint down to a comfortable
    ceiling.

    Sized against the last year of the horizon, not the year the signal first
    binds: demand keeps growing after that year under every curve this codebase
    uses, so sizing for the earliest binding year would systematically undersize
    the fleet for every later one.

    The search itself treats delivery as instantaneous once `delivery_start_year`
    begins (no ramp) -- a deliberate simplification to keep the integer search
    fast. `evaluate_candidate` applies the real delivery ramp when it computes the
    financial case for whichever count this settles on.
    """
    a = assumptions or load_expansion_assumptions()
    par = params or load_params()
    growth_assumptions = growth or load_growth_assumptions()
    base_markets = markets if markets is not None else load_markets()
    type_code = signal.subject

    types = load_aircraft_types()
    lead_time = a.lead_time_years(type_code, types[type_code].body)
    delivery_start_year = signal.first_binding_year
    delivery_end_year = delivery_start_year + math.ceil(a.delivery_window_years)
    order_year = delivery_start_year - math.ceil(lead_time)

    grown = grow_markets(base_markets, end_year, assumptions=growth_assumptions)
    target = date(end_year, 3, 1)

    count = a.min_candidate_count
    while True:
        trial_config = _with_added_order(
            fleet_config,
            type_code,
            count,
            start=date(delivery_start_year, 1, 1),
            end=date(delivery_start_year, 12, 31),
        )
        snapshot = fleet_on(target, config=trial_config, params=par)
        plan = _solve_or_relax(
            target, month=3, markets=grown, fleet=snapshot, params=par, solve_kwargs=solve_kwargs
        )
        utilisation = plan.utilisation().get(type_code, 0.0)

        if utilisation <= a.target_utilisation_ceiling or count >= a.max_candidate_count:
            break
        count += 1

    return CandidateOrder(
        type_code=type_code,
        count=count,
        trigger=signal,
        order_year=order_year,
        delivery_start_year=delivery_start_year,
        delivery_end_year=delivery_end_year,
    )


def net_present_value(
    cash_flows: dict[int, float], *, discount_rate: float, base_year: int
) -> float:
    """Discount a year -> cash-flow mapping back to `base_year`.

    A pure function on plain numbers so the discounting arithmetic itself is
    testable without a solve -- `base_year` is normally the candidate's
    `order_year`, i.e. when the capital commitment is actually made.
    """
    return sum(
        flow / ((1 + discount_rate) ** (year - base_year)) for year, flow in cash_flows.items()
    )


def payback_year_of(cash_flows: dict[int, float]) -> int | None:
    """The first year (in year order) by which cumulative, undiscounted cash flow
    turns non-negative, or None if it never does within the given years."""
    cumulative = 0.0
    for year in sorted(cash_flows):
        cumulative += cash_flows[year]
        if cumulative >= 0:
            return year
    return None


def evaluate_candidate(
    candidate: CandidateOrder,
    *,
    baseline_ladder: LadderResult,
    base_fleet_config: dict[str, Any],
    params: Params | None = None,
    assumptions: ExpansionAssumptions | None = None,
    solve_kwargs: dict[str, Any] | None = None,
) -> OrderEvaluation:
    """WITH-vs-WITHOUT re-solve at every checkpoint from the candidate's delivery
    year through the horizon end.

    `baseline_ladder` supplies both the WITHOUT side and each checkpoint's already-
    grown market list -- it is whatever ladder the caller already solved against
    `base_fleet_config` this iteration, so it is reused, not re-solved. Only the
    WITH side (`base_fleet_config` plus this candidate's order) is solved here.
    `solve_kwargs` must match whatever produced `baseline_ladder`, or the two sides
    of the comparison rest on different assumptions.
    """
    a = assumptions or load_expansion_assumptions()
    par = params or load_params()

    with_config = _with_added_order(
        base_fleet_config,
        candidate.type_code,
        candidate.count,
        start=date(candidate.delivery_start_year, 1, 1),
        end=date(candidate.delivery_end_year, 12, 31),
    )

    types = load_aircraft_types()
    ownership_per_day = types[candidate.type_code].ownership_cost_per_day_usd

    contribution_delta: dict[int, float] = {}
    ownership_delta: dict[int, float] = {}
    net_cash_flow: dict[int, float] = {}

    for checkpoint in baseline_ladder.checkpoints:
        if checkpoint.year < candidate.delivery_start_year:
            continue

        with_snapshot = fleet_on(checkpoint.as_of, config=with_config, params=par)
        with_plan = _solve_or_relax(
            checkpoint.as_of,
            month=checkpoint.as_of.month,
            markets=list(checkpoint.grown_markets),
            fleet=with_snapshot,
            params=par,
            solve_kwargs=solve_kwargs,
        )

        # total_contribution_usd is a *single representative day*'s figure (the
        # whole fleet-assignment core solves one day at a time) -- annualise it by
        # the same "one March day x 365" simplification used for ownership cost
        # below, so both sides of the cash flow are on the same time basis.
        c_delta = (with_plan.total_contribution_usd - checkpoint.plan.total_contribution_usd) * DAYS_PER_YEAR
        delivered_fraction = _ramp_fraction(
            checkpoint.as_of,
            date(candidate.delivery_start_year, 1, 1),
            date(candidate.delivery_end_year, 12, 31),
        )
        o_delta = delivered_fraction * candidate.count * ownership_per_day * DAYS_PER_YEAR

        contribution_delta[checkpoint.year] = c_delta
        ownership_delta[checkpoint.year] = o_delta
        net_cash_flow[checkpoint.year] = c_delta - o_delta

    npv = net_present_value(net_cash_flow, discount_rate=a.discount_rate_annual, base_year=candidate.order_year)
    payback_year = payback_year_of(net_cash_flow)

    return OrderEvaluation(
        candidate=candidate,
        contribution_delta_usd=contribution_delta,
        ownership_cost_delta_usd=ownership_delta,
        net_cash_flow_usd=net_cash_flow,
        npv_usd=npv,
        payback_year=payback_year,
        recommended=npv > 0,
    )


def recommend_expansion(
    start_year: int,
    end_year: int,
    *,
    markets: list[Market] | None = None,
    fleet_config: dict[str, Any] | None = None,
    params: Params | None = None,
    growth: GrowthAssumptions | None = None,
    assumptions: ExpansionAssumptions | None = None,
    max_orders: int = 8,
    solve_kwargs: dict[str, Any] | None = None,
) -> ExpansionPlan:
    """The staged expansion loop: detect, size, evaluate, commit-or-stop, repeat.

    **Explicit limitations, by design, not oversight:**
      - Only `fleet_utilisation` signals drive a candidate order. A `market_spill`
        signal (surfaced in `ExpansionPlan.signals`) means demand is being turned
        away, but *which* type would relieve it is itself a re-optimisation
        question this heuristic does not attempt to answer -- it is left for a
        human planner, not auto-converted into an order.
      - This is greedy and sequential, evaluated in time-urgency order, not a
        joint optimisation across every candidate at once. Order can matter in
        principle when two types are partial substitutes.
      - A same-type order placed again within `min_years_between_same_type_orders`
        of a prior one is still evaluated and can still be accepted, but is
        flagged via `OrderEvaluation.review_flag` for a human to double check
        rather than silently approved back-to-back.

    `solve_kwargs` forwards arbitrary opt-in `solve_network` flags (e.g.
    `enable_engine_degradation`, `degradation_assumptions`) through every solve in
    this pipeline -- detection, sizing, and WITH/WITHOUT evaluation alike, so a
    degradation-aware run genuinely compares degradation-aware economics end to end,
    not just at the reporting layer.
    """
    a = assumptions or load_expansion_assumptions()
    par = _tighten_for_financial_precision(params or load_params(), a)
    base_markets = markets if markets is not None else load_markets()
    growth_assumptions = growth or load_growth_assumptions()

    rolling_config = copy.deepcopy(fleet_config if fleet_config is not None else load_fleet_config())
    ladder = run_ladder(
        start_year, end_year, markets=base_markets, fleet_config=rolling_config,
        params=par, growth=growth_assumptions, solve_kwargs=solve_kwargs,
    )

    evaluations: list[OrderEvaluation] = []
    committed_order_years: dict[str, list[int]] = {}
    # (subject, first_binding_year) pairs already evaluated this run. A signal that
    # reappears with the *same* pair after a candidate was sized against it means
    # sizing/evaluation did not resolve it -- skip it rather than loop forever, but
    # keep evaluating whatever *other* signals remain (an unrelated type's genuine
    # shortage should not go unaddressed just because one candidate got stuck).
    attempted_signal_keys: set[tuple[str, int]] = set()

    for _ in range(max_orders):
        fleet_signals: list[ConstraintSignal] = [
            s
            for s in detect_signals(ladder)
            if s.actionable
            and s.signal_type == "fleet_utilisation"
            and (s.subject, s.first_binding_year) not in attempted_signal_keys
        ]
        if not fleet_signals:
            break

        most_urgent = min(fleet_signals, key=lambda s: (s.first_binding_year, s.subject))
        attempted_signal_keys.add((most_urgent.subject, most_urgent.first_binding_year))

        candidate = size_candidate(
            most_urgent, end_year=end_year, fleet_config=rolling_config,
            markets=base_markets, params=par, growth=growth_assumptions, assumptions=a,
            solve_kwargs=solve_kwargs,
        )
        evaluation = evaluate_candidate(
            candidate, baseline_ladder=ladder, base_fleet_config=rolling_config,
            params=par, assumptions=a, solve_kwargs=solve_kwargs,
        )

        prior_years = committed_order_years.get(candidate.type_code, [])
        if any(
            abs(candidate.order_year - y) < a.min_years_between_same_type_orders for y in prior_years
        ):
            evaluation = replace(
                evaluation,
                review_flag=(
                    f"{candidate.type_code} was already ordered within "
                    f"{a.min_years_between_same_type_orders} years of this candidate's order "
                    "year -- flagged for human review, not auto-approved back-to-back"
                ),
            )

        evaluations.append(evaluation)
        if not evaluation.recommended:
            # This candidate does not pencil out -- move on to whatever other
            # signal is next most urgent instead of abandoning the whole run.
            continue

        rolling_config = _with_added_order(
            rolling_config, candidate.type_code, candidate.count,
            start=date(candidate.delivery_start_year, 1, 1),
            end=date(candidate.delivery_end_year, 12, 31),
        )
        committed_order_years.setdefault(candidate.type_code, []).append(candidate.order_year)
        ladder = run_ladder(
            start_year, end_year, markets=base_markets, fleet_config=rolling_config,
            params=par, growth=growth_assumptions, solve_kwargs=solve_kwargs,
        )

    final_signals = tuple(detect_signals(ladder))
    return ExpansionPlan(ladder=ladder, signals=final_signals, evaluations=tuple(evaluations))
