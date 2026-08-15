"""What-if analysis.

A single optimal plan is a weak deliverable: it is optimal against assumptions that
are guaranteed to be wrong. Delivery dates slip, fuel moves, a slot is lost, a market
softens. What a network planner needs is not one answer but a sense of which answers
are robust.

Each scenario is a set of mutations applied to a *copy* of the inputs -- the fleet
book, the economic parameters, the market table -- after which the network is re-solved
and compared against the baseline. Nothing mutates shared state, so scenarios can be
composed and run in any order.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from anos.config import Params, load_params
from anos.data.loaders import load_fleet_config, load_markets
from anos.models import Market, NetworkPlan
from anos.optimize.fleet_assignment import InfeasibleNetworkError, solve_network

FleetMutator = Callable[[dict[str, Any]], None]
ParamsMutator = Callable[[dict[str, Any]], None]
MarketMutator = Callable[[list[Market]], list[Market]]


@dataclass
class Scenario:
    """A named perturbation of the baseline inputs."""

    name: str
    description: str
    fleet: FleetMutator | None = None
    params: ParamsMutator | None = None
    markets: MarketMutator | None = None
    probabilistic_fleet: bool = False


@dataclass
class ScenarioResult:
    """A solved scenario alongside its delta from the baseline.

    A scenario that cannot meet the network's minimum service floors is not an
    error -- it is the most important result the suite can produce. Those scenarios
    are re-solved with the floors released, and `service_floors_unmeetable` records
    that the obligations themselves are what broke, so the reported plan is the best
    achievable network rather than no answer at all.
    """

    scenario: Scenario
    plan: NetworkPlan
    baseline: NetworkPlan
    service_floors_unmeetable: bool = False
    unmet_floors: list[str] = field(default_factory=list)

    @property
    def contribution_delta_usd(self) -> float:
        return self.plan.total_contribution_usd - self.baseline.total_contribution_usd

    @property
    def contribution_delta_pct(self) -> float:
        base = self.baseline.total_contribution_usd
        return (self.contribution_delta_usd / base * 100) if base else 0.0

    @property
    def pax_delta(self) -> float:
        return self.plan.total_pax - self.baseline.total_pax

    @property
    def markets_dropped(self) -> list[str]:
        """Markets the baseline served that this scenario cannot."""
        served_now = {p.market_id for p in self.plan.market_plans if p.total_frequency > 0}
        served_base = {p.market_id for p in self.baseline.market_plans if p.total_frequency > 0}
        return sorted(served_base - served_now)


# -- scenario constructors ---------------------------------------------------


def delivery_slip(months: int, types: list[str] | None = None) -> Scenario:
    """Push delivery windows to the right.

    Both OEMs have already moved dates on this order book, so this is the single most
    important scenario to run before committing to a growth plan.
    """
    from datetime import timedelta

    days = int(months * 30.44)
    label = ", ".join(types) if types else "all programmes"

    def mutate(cfg: dict[str, Any]) -> None:
        for entry in cfg.get("orders", []) or []:
            if types is None or entry["type"] in types:
                entry["start"] += timedelta(days=days)
                entry["end"] += timedelta(days=days)

    return Scenario(
        name=f"delivery_slip_{months}m",
        description=f"{label} slip {months} months",
        fleet=mutate,
    )


def fuel_shock(pct: float) -> Scenario:
    """Move the jet fuel price by a percentage."""

    def mutate(raw: dict[str, Any]) -> None:
        raw["fuel"]["price_usd_per_kg"] *= 1 + pct / 100.0

    return Scenario(
        name=f"fuel_{'up' if pct >= 0 else 'down'}_{abs(int(pct))}pct",
        description=f"jet fuel price moves {pct:+.0f}%",
        params=mutate,
    )


def demand_shock(pct: float, region: str | None = None) -> Scenario:
    """Move demand across the network, or in one region."""
    scope = region or "network-wide"

    def mutate(markets: list[Market]) -> list[Market]:
        out = []
        for m in markets:
            if region is None or m.region == region:
                out.append(
                    Market(**{**m.__dict__, "base_daily_demand": m.base_daily_demand * (1 + pct / 100.0)})
                )
            else:
                out.append(m)
        return out

    tag = region.replace("_", "") if region else "network"
    return Scenario(
        name=f"demand_{tag}_{'up' if pct >= 0 else 'down'}_{abs(int(pct))}pct",
        description=f"{scope} demand moves {pct:+.0f}%",
        markets=mutate,
    )


def retrofit_overrun(extra_days: int) -> Scenario:
    """Cabin retrofit programmes run longer than planned.

    Retrofit downtime is real capacity loss during exactly the window in which the
    airline is trying to grow.
    """
    from datetime import timedelta

    def mutate(cfg: dict[str, Any]) -> None:
        for entry in cfg.get("retrofits", []) or []:
            entry["downtime_days"] += extra_days
            entry["end"] += timedelta(days=extra_days)

    return Scenario(
        name=f"retrofit_overrun_{extra_days}d",
        description=f"cabin retrofits take {extra_days} days longer per airframe",
        fleet=mutate,
    )


def slot_loss(iata: str, pct: float) -> Scenario:
    """Lose a share of the departure slot allocation at one airport."""

    def mutate(markets: list[Market]) -> list[Market]:
        # Slot caps live on the airport record, which the optimiser reads directly.
        # Rather than mutate shared airport state, cap the affected markets instead.
        out = []
        for m in markets:
            if iata in (m.origin, m.destination):
                new_max = max(m.min_daily_freq, int(m.max_daily_freq * (1 - pct / 100.0)))
                out.append(Market(**{**m.__dict__, "max_daily_freq": new_max}))
            else:
                out.append(m)
        return out

    return Scenario(
        name=f"slot_loss_{iata}_{int(pct)}pct",
        description=f"lose {pct:.0f}% of departure slots at {iata}",
        markets=mutate,
    )


def risk_adjusted_fleet() -> Scenario:
    """Use expected rather than nominal delivery dates.

    Weights each programme's delivery curve by the planner's confidence in it, so
    low-confidence programmes such as the 777X contribute less capacity.
    """
    return Scenario(
        name="risk_adjusted_fleet",
        description="deliveries risk-adjusted by programme confidence",
        probabilistic_fleet=True,
    )


def standard_suite() -> list[Scenario]:
    """The scenarios worth running against any plan before committing to it."""
    return [
        risk_adjusted_fleet(),
        delivery_slip(12),
        delivery_slip(24, types=["B777-9", "A350-1000"]),
        retrofit_overrun(20),
        fuel_shock(25),
        fuel_shock(-15),
        demand_shock(-12),
        demand_shock(-25, region="longhaul_americas"),
        slot_loss("BOM", 20),
    ]


# -- execution ---------------------------------------------------------------


def run_scenario(
    scenario: Scenario,
    target: date,
    baseline: NetworkPlan,
    *,
    month: int | None = None,
) -> ScenarioResult:
    """Solve one scenario against its own copy of the inputs."""
    fleet_cfg = copy.deepcopy(load_fleet_config())
    raw_params = copy.deepcopy(load_params().raw)
    markets = list(load_markets())

    if scenario.fleet:
        scenario.fleet(fleet_cfg)
    if scenario.params:
        scenario.params(raw_params)
    if scenario.markets:
        markets = scenario.markets(markets)

    params = Params(raw=raw_params)

    from anos.data.fleet_timeline import fleet_on

    snapshot = fleet_on(
        target,
        probabilistic=scenario.probabilistic_fleet,
        config=fleet_cfg,
        params=params,
    )

    try:
        plan = solve_network(
            target,
            month=month,
            markets=markets,
            fleet=snapshot,
            params=params,
            scenario=scenario.name,
        )
        return ScenarioResult(scenario=scenario, plan=plan, baseline=baseline)
    except InfeasibleNetworkError:
        pass

    # The scenario could not honour every minimum service floor. Re-solve with the
    # floors released so the planner sees what the network *can* do, and which
    # commitments would have to be renegotiated to get there.
    plan = solve_network(
        target,
        month=month,
        markets=markets,
        fleet=snapshot,
        params=params,
        enforce_min_service=False,
        scenario=scenario.name,
    )
    served = {p.market_id: p.total_frequency for p in plan.market_plans}
    unmet = sorted(
        m.market_id
        for m in markets
        if m.min_daily_freq > 0 and served.get(m.market_id, 0) < m.min_daily_freq
    )
    return ScenarioResult(
        scenario=scenario,
        plan=plan,
        baseline=baseline,
        service_floors_unmeetable=True,
        unmet_floors=unmet,
    )


def run_suite(
    target: date,
    baseline: NetworkPlan,
    scenarios: list[Scenario] | None = None,
    *,
    month: int | None = None,
) -> list[ScenarioResult]:
    """Run a set of scenarios and return results ordered by impact."""
    suite = scenarios if scenarios is not None else standard_suite()
    results = [run_scenario(s, target, baseline, month=month) for s in suite]
    return sorted(results, key=lambda r: r.contribution_delta_usd)
