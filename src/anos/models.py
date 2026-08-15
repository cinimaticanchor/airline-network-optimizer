"""Core domain objects for the Air India network optimisation model.

These are deliberately plain dataclasses: every stage of the pipeline reads and
writes them, so they carry no behaviour that depends on a particular solver,
data source or reporting format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

BodyType = Literal["narrow", "wide"]

EARTH_RADIUS_KM = 6371.0088


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class AircraftType:
    """Published specification for one aircraft type."""

    code: str
    name: str
    body: BodyType
    seats: int
    range_km: float
    cruise_kmh: float
    fuel_burn_kgh: float
    min_turn_min: int
    max_daily_util_h: float
    ownership_cost_per_day_usd: float
    crew_cost_per_block_hour_usd: float
    generation: str

    @property
    def is_widebody(self) -> bool:
        return self.body == "wide"

    def block_hours(self, distance_km: float) -> float:
        """Gate-to-gate block time for one sector.

        Cruise time plus a fixed allowance for taxi, climb and descent, which
        dominates the error on short sectors if left out.
        """
        cruise_h = distance_km / self.cruise_kmh
        fixed_overhead_h = 0.40 if self.body == "narrow" else 0.55
        return cruise_h + fixed_overhead_h


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    widebody_capable: bool
    slot_constrained: bool
    daily_slot_cap: int
    curfew: bool
    timezone_offset_h: float

    @property
    def is_domestic(self) -> bool:
        return self.country == "IN"


@dataclass(frozen=True)
class Market:
    """A directional-agnostic O&D pair the airline may serve.

    Frequencies are expressed as daily round trips, which is how network planners
    actually reason about a market.
    """

    market_id: str
    origin: str
    destination: str
    region: str
    base_daily_demand: float
    avg_fare_usd: float
    business_mix: float
    competition_index: float
    min_daily_freq: int
    max_daily_freq: int
    seasonality_profile: str
    strategic: bool
    interline_available: bool = False

    @property
    def is_domestic(self) -> bool:
        return self.region.startswith("domestic")


@dataclass(frozen=True)
class MaintenanceProgramme:
    """Check intervals and downtime for one body type."""

    a_check_interval_fh: float
    a_check_downtime_h: float
    c_check_interval_months: int
    c_check_downtime_days: int
    d_check_interval_years: int
    d_check_downtime_days: int


@dataclass
class FleetSnapshot:
    """How many airframes of each type are actually flyable on a given date.

    `nominal` is the raw count on the books. `available` is what the optimiser may
    plan against: nominal less retrofit downtime, scheduled heavy maintenance and
    an unscheduled-downtime allowance.
    """

    as_of: date
    nominal: dict[str, float] = field(default_factory=dict)
    available: dict[str, float] = field(default_factory=dict)
    in_maintenance: dict[str, float] = field(default_factory=dict)
    in_retrofit: dict[str, float] = field(default_factory=dict)

    def total_nominal(self) -> float:
        return sum(self.nominal.values())

    def total_available(self) -> float:
        return sum(self.available.values())


@dataclass(frozen=True)
class LegEconomics:
    """Unit economics of flying one round trip in one market with one type.

    Everything the optimiser needs about a (market, type) pairing is precomputed
    here so the solve itself stays linear.
    """

    market_id: str
    type_code: str
    distance_km: float
    block_hours_round_trip: float
    aircraft_hours_consumed: float  # block time plus planned ground time
    seats_round_trip: int
    cost_per_round_trip_usd: float
    eligible: bool
    ineligible_reason: str = ""


@dataclass
class MarketPlan:
    """The optimiser's recommendation for a single market."""

    market_id: str
    frequencies: dict[str, int]  # type code -> daily round trips
    total_frequency: int
    seats_offered: int
    pax_carried: float
    demand_effective: float
    revenue_usd: float
    cost_usd: float
    connecting_pax_carried: float = 0.0  # pax reaching/leaving via a banked hub connection
    fare_usd: float | None = None  # realised fare, when fare elasticity chose a non-reference level

    @property
    def contribution_usd(self) -> float:
        return self.revenue_usd - self.cost_usd

    @property
    def load_factor(self) -> float:
        return self.pax_carried / self.seats_offered if self.seats_offered else 0.0

    @property
    def spilled_demand(self) -> float:
        return max(0.0, self.demand_effective - self.pax_carried)


@dataclass(frozen=True)
class ConnectionCandidate:
    """A one-stop itinerary that could serve `market_id` via a hub, rather than nonstop.

    `market_id` is the (currently possibly unserved) O&D market this connection would
    carry -- e.g. "MAA-LHR". `feeder_market_id`/`trunk_market_id` are existing markets.csv
    rows (hub-first, e.g. "DEL-MAA" and "DEL-LHR") whose spare seats the connecting
    passenger occupies. No new aircraft-hours are consumed: a connection only competes
    for capacity that feeder and trunk flights already have.
    """

    connection_id: str
    market_id: str
    hub: str
    feeder_market_id: str
    trunk_market_id: str


@dataclass(frozen=True)
class RotationCandidate:
    """A multi-stop loop (hub -> stop1 -> stop2 -> hub) flown by one aircraft.

    Scoped to a small, explicit squad of markets (typically ones flying at low load
    factor as separate out-and-back legs) rather than searched for generally. Flying
    the loop once counts as one daily round trip for *each* of the two leg markets it
    replaces, while consuming the loop's shared aircraft-hours instead of two separate
    round trips' worth.
    """

    rotation_id: str
    hub: str
    stops: tuple[str, str]
    leg_market_ids: tuple[str, str]


@dataclass(frozen=True)
class RotationEconomics:
    """Unit economics of flying one rotation loop once with one aircraft type."""

    rotation_id: str
    type_code: str
    aircraft_hours_consumed: float
    cost_usd: float
    eligible: bool
    ineligible_reason: str = ""


@dataclass(frozen=True)
class ScheduledLegGroup:
    """One or more identical directional departures the time-space model chose.

    `count` interchangeable departures sharing a (market, type, direction, bucket) --
    not `count` individually distinguishable flight instances. `FleetSnapshot` and
    `data/fleet.yaml` carry only aggregate per-type counts, never individual tail
    identities, so there is nothing here for a "flight #1 vs flight #2" distinction to
    refer to. A future tail-routing phase would need to invent `count` synthetic tail
    identities per type to turn this into an operable rotation.
    """

    market_id: str
    type_code: str
    direction: Literal["outbound", "return"]  # outbound = market.origin -> destination
    departure_bucket: int
    avail_bucket: int  # already includes planned turn -- "usable again", not just landed
    count: int


@dataclass(frozen=True)
class TailLegAssignment:
    """One hop in a synthetic tail's daily rotation -- either a flight or a wait on
    the ground. Produced by decomposing the time-space core's solved circulation
    (`anos.optimize.tail_routing`), not by a fresh solve -- see that module's
    docstring for what this can and cannot claim about real maintenance feasibility.
    """

    kind: Literal["fly", "ground"]
    from_station: str
    to_station: str  # == from_station for a ground hop
    from_bucket: int
    to_bucket: int
    market_id: str | None = None  # populated when kind == "fly"
    direction: Literal["outbound", "return"] | None = None  # populated when kind == "fly"
    flight_hours: float = 0.0  # one-way block + turn; 0.0 for ground hops


@dataclass(frozen=True)
class TailRotation:
    """A repeating daily flying pattern, shared by `tail_count` interchangeable
    synthetic tails of `type_code`.

    No real tail-registration data exists anywhere in this codebase (see
    `ScheduledLegGroup`'s docstring) -- naming `tail_count` individually distinct
    fake IDs when they are functionally identical would be false precision, so this
    reports the distinct pattern and how many tails follow it instead. One of
    possibly several valid decompositions of the same aggregate flow, not a claim
    about which physical airframe does what.
    """

    type_code: str
    tail_count: int
    legs: tuple[TailLegAssignment, ...]  # ordered, cyclic: legs[-1] connects back to legs[0]
    daily_flight_hours: float


@dataclass
class NetworkPlan:
    """A complete solved network: what to fly, with what, and what it earns."""

    as_of: date
    month: int
    market_plans: list[MarketPlan]
    fleet_used: dict[str, float]  # type code -> aircraft-hours consumed per day
    fleet_available: dict[str, float]  # type code -> aircraft-hours available per day
    solver_status: str
    objective_usd: float
    solve_seconds: float
    scenario: str = "baseline"
    connections_used: list[str] = field(default_factory=list)  # market_ids served via a banked connection
    rotations_flown: dict[str, int] = field(default_factory=dict)  # rotation_id -> daily loops flown
    interline_used: list[str] = field(default_factory=list)  # market_ids sold via a partner itinerary
    scheduled_legs: list[ScheduledLegGroup] = field(default_factory=list)  # only from solve_network_timespace
    timespace_types: list[str] = field(default_factory=list)  # types capacity-checked via the time-space core
    ground_occupancy: dict[tuple[str, str, int], int] = field(default_factory=dict)  # (type,station,bucket) -> count
    tail_rotations: list[TailRotation] = field(default_factory=list)  # only when build_tail_rotations=True

    @property
    def total_revenue_usd(self) -> float:
        return sum(p.revenue_usd for p in self.market_plans)

    @property
    def total_cost_usd(self) -> float:
        return sum(p.cost_usd for p in self.market_plans)

    @property
    def total_contribution_usd(self) -> float:
        return self.total_revenue_usd - self.total_cost_usd

    @property
    def total_pax(self) -> float:
        return sum(p.pax_carried for p in self.market_plans)

    @property
    def total_seats(self) -> int:
        return sum(p.seats_offered for p in self.market_plans)

    @property
    def network_load_factor(self) -> float:
        return self.total_pax / self.total_seats if self.total_seats else 0.0

    @property
    def total_departures(self) -> int:
        return sum(p.total_frequency for p in self.market_plans) * 2

    def utilisation(self) -> dict[str, float]:
        """Fraction of each type's available aircraft-hours the plan consumes."""
        return {
            t: (self.fleet_used.get(t, 0.0) / avail if avail > 0 else 0.0)
            for t, avail in self.fleet_available.items()
        }

    def markets_served(self) -> int:
        return sum(1 for p in self.market_plans if p.total_frequency > 0)


@dataclass
class FeasibilityFinding:
    """One problem the tail-feasibility check found with a proposed plan."""

    severity: Literal["blocker", "warning", "info"]
    category: str
    subject: str
    detail: str


@dataclass
class FeasibilityReport:
    """Whether a NetworkPlan can actually be flown by real airframes."""

    findings: list[FeasibilityFinding] = field(default_factory=list)
    maintenance_hours_required: dict[str, float] = field(default_factory=dict)
    slot_usage: dict[str, int] = field(default_factory=dict)

    @property
    def blockers(self) -> list[FeasibilityFinding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def warnings(self) -> list[FeasibilityFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def is_feasible(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class TrajectoryPoint:
    """One year's checkpoint in a multi-year `anos.planning.scenario_ladder` run:
    the grown market set, the fleet on the books that year, and the resulting solve.
    """

    year: int
    as_of: date
    plan: NetworkPlan
    fleet: FleetSnapshot
    grown_markets: tuple[Market, ...]
    service_floors_unmeetable: bool = False
    unmet_floors: tuple[str, ...] = ()

    def utilisation(self) -> dict[str, float]:
        return self.plan.utilisation()

    def spill_by_market(self) -> dict[str, float]:
        return {p.market_id: p.spilled_demand for p in self.plan.market_plans}

    def unserved_strategic_markets(self) -> tuple[str, ...]:
        served = {p.market_id for p in self.plan.market_plans if p.total_frequency > 0}
        return tuple(
            m.market_id for m in self.grown_markets if m.strategic and m.market_id not in served
        )


@dataclass(frozen=True)
class LadderResult:
    """A full multi-year scenario ladder: one `TrajectoryPoint` per checkpoint year,
    in chronological order."""

    checkpoints: tuple[TrajectoryPoint, ...]

    def contribution_series(self) -> tuple[float, ...]:
        return tuple(c.plan.total_contribution_usd for c in self.checkpoints)

    def utilisation_series(self, type_code: str) -> tuple[float, ...]:
        return tuple(c.utilisation().get(type_code, 0.0) for c in self.checkpoints)

    def checkpoint_for_year(self, year: int) -> TrajectoryPoint | None:
        return next((c for c in self.checkpoints if c.year == year), None)


@dataclass(frozen=True)
class ConstraintSignal:
    """One persistent constraint `anos.planning.constraint_detection` found in a
    `LadderResult` -- a candidate reason to consider a fleet order.

    `actionable` is False for a market-spill signal traced to a slot-saturated
    airport: buying an aircraft would not relieve it, so `fleet_expansion` must
    never treat it as an order trigger, only surface it as an informational note.
    """

    signal_type: Literal["fleet_utilisation", "market_spill"]
    subject: str  # type code for fleet_utilisation, market_id for market_spill
    first_binding_year: int
    last_checkpoint_year: int
    detail: str
    actionable: bool = True


@dataclass(frozen=True)
class CandidateOrder:
    """A specific fleet order `anos.planning.fleet_expansion` is considering, sized
    to relieve one `ConstraintSignal`."""

    type_code: str
    count: int
    trigger: ConstraintSignal
    order_year: int  # year the order would need to be placed, given the type's lead time
    delivery_start_year: int  # year first deliveries land
    delivery_end_year: int  # year the tranche is fully delivered


@dataclass(frozen=True)
class OrderEvaluation:
    """The WITH-vs-WITHOUT financial case for one `CandidateOrder`.

    Cash flows are keyed by calendar year, covering every checkpoint from
    `candidate.delivery_start_year` through the planning horizon's end.
    """

    candidate: CandidateOrder
    contribution_delta_usd: dict[int, float]  # WITH minus WITHOUT total_contribution_usd, by year
    ownership_cost_delta_usd: dict[int, float]  # incremental ramp-weighted ownership cost, by year
    net_cash_flow_usd: dict[int, float]  # contribution_delta minus ownership_cost_delta, by year
    npv_usd: float
    payback_year: int | None
    recommended: bool
    review_flag: str | None = None


@dataclass(frozen=True)
class ExpansionPlan:
    """The output of `anos.planning.fleet_expansion.recommend_expansion`: what was
    detected against the final committed fleet, and every candidate evaluated along
    the way, in the order they were considered."""

    ladder: LadderResult
    signals: tuple[ConstraintSignal, ...]
    evaluations: tuple[OrderEvaluation, ...]

    @property
    def accepted_orders(self) -> tuple[CandidateOrder, ...]:
        return tuple(e.candidate for e in self.evaluations if e.recommended)
