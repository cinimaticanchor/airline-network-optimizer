"""Filesystem layout and parameter access."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

AIRCRAFT_TYPES_FILE = DATA_DIR / "aircraft_types.yaml"
FLEET_FILE = DATA_DIR / "fleet.yaml"
AIRPORTS_FILE = DATA_DIR / "airports.csv"
MARKETS_FILE = DATA_DIR / "markets.csv"
COST_PARAMS_FILE = DATA_DIR / "cost_params.yaml"
HUB_TOPOLOGY_FILE = DATA_DIR / "hub_topology.yaml"
TIMESPACE_FILE = DATA_DIR / "timespace.yaml"
DEMAND_GROWTH_FILE = DATA_DIR / "demand_growth.yaml"
EXPANSION_ASSUMPTIONS_FILE = DATA_DIR / "expansion_assumptions.yaml"


@dataclass(frozen=True)
class Params:
    """Typed view over cost_params.yaml.

    Kept as a thin wrapper rather than a deep dataclass tree so that adding a
    parameter to the YAML does not require a code change to read it.
    """

    raw: dict[str, Any]

    # -- fuel ---------------------------------------------------------------
    @property
    def fuel_price_usd_per_kg(self) -> float:
        return float(self.raw["fuel"]["price_usd_per_kg"])

    @property
    def taxi_burn_kg_per_cycle(self) -> float:
        return float(self.raw["fuel"]["taxi_burn_kg_per_cycle"])

    @property
    def fuel_contingency_factor(self) -> float:
        return float(self.raw["fuel"]["contingency_factor"])

    # -- demand behaviour ---------------------------------------------------
    @property
    def frequency_ref(self) -> float:
        return float(self.raw["demand"]["frequency_ref"])

    @property
    def max_frequency_uplift(self) -> float:
        return float(self.raw["demand"]["max_frequency_uplift"])

    @property
    def saturation_k(self) -> float:
        return float(self.raw["demand"]["saturation_k"])

    @property
    def business_mix_weight(self) -> float:
        return float(self.raw["demand"]["business_mix_weight"])

    # -- operations ---------------------------------------------------------
    @property
    def range_safety_factor(self) -> float:
        return float(self.raw["operations"]["range_safety_factor"])

    @property
    def payload_penalty_threshold(self) -> float:
        return float(self.raw["operations"].get("payload_penalty_threshold", 0.95))

    @property
    def turn_buffer_factor(self) -> float:
        return float(self.raw["operations"]["turn_buffer_factor"])

    @property
    def fast_turn_buffer_factor(self) -> float:
        return float(self.raw["operations"]["fast_turn_buffer_factor"])

    def unscheduled_downtime_rate(self, generation: str) -> float:
        """Fraction of the fleet unavailable for unscheduled maintenance/AOG/spares,
        for aircraft of this `AircraftType.generation` -- see cost_params.yaml's
        `operations.unscheduled_downtime_rate` block for why this is per-generation."""
        return float(self.raw["operations"]["unscheduled_downtime_rate"][generation])

    @property
    def target_load_factor(self) -> float:
        return float(self.raw["operations"]["target_load_factor"])

    # -- solver -------------------------------------------------------------
    @property
    def solver_max_seconds(self) -> float:
        return float(self.raw["solver"]["max_seconds"])

    @property
    def solver_num_workers(self) -> int:
        return int(self.raw["solver"]["num_workers"])

    @property
    def solver_relative_gap(self) -> float:
        return float(self.raw["solver"]["relative_gap"])

    def seasonality(self, profile: str, month: int) -> float:
        """Demand multiplier for a seasonality profile in a given month (1-12)."""
        table = self.raw["seasonality"]
        if profile not in table:
            return 1.0
        return float(table[profile][month - 1])

    def charges_per_departure(self, *, domestic: bool, widebody: bool) -> float:
        block = self.raw["charges"]["domestic" if domestic else "international"]
        key = "wide_per_departure_usd" if widebody else "narrow_per_departure_usd"
        return float(block[key])

    @property
    def slot_surcharge_usd(self) -> float:
        return float(self.raw["charges"]["slot_constrained_surcharge_usd"])

    def cost_per_pax(self, *, domestic: bool) -> float:
        key = "domestic_cost_per_pax_usd" if domestic else "international_cost_per_pax_usd"
        return float(self.raw["passenger"][key])

    # -- fare elasticity (opt-in) --------------------------------------------
    @property
    def fare_multipliers(self) -> list[float]:
        return [float(m) for m in self.raw["fare_elasticity"]["fare_multipliers"]]

    @property
    def leisure_elasticity(self) -> float:
        return float(self.raw["fare_elasticity"]["leisure_elasticity"])

    @property
    def business_elasticity(self) -> float:
        return float(self.raw["fare_elasticity"]["business_elasticity"])

    @property
    def fare_elasticity_multiplier_band(self) -> tuple[float, float]:
        lo, hi = self.raw["fare_elasticity"]["multiplier_band"]
        return float(lo), float(hi)

    # -- connections / banking (opt-in) --------------------------------------
    @property
    def connection_capture_rate(self) -> float:
        return float(self.raw["connections"]["capture_rate"])

    @property
    def connection_handling_cost_usd_per_pax(self) -> float:
        return float(self.raw["connections"]["handling_cost_usd_per_pax"])

    @property
    def connection_recapture_rate(self) -> float:
        return float(self.raw["connections"]["recapture_rate"])

    # -- carbon / SAF (opt-in) ------------------------------------------------
    @property
    def co2_per_kg_fuel(self) -> float:
        return float(self.raw["carbon"]["co2_per_kg_fuel"])

    @property
    def carbon_price_usd_per_tonne(self) -> float:
        return float(self.raw["carbon"]["price_usd_per_tonne_co2"])

    @property
    def carbon_priced_countries(self) -> set[str]:
        return set(self.raw["carbon"]["priced_countries"])

    # -- interline (opt-in) ---------------------------------------------------
    @property
    def interline_prorate_usd_per_pax(self) -> float:
        return float(self.raw["interline"]["prorate_usd_per_pax"])

    @property
    def interline_capture_rate(self) -> float:
        return float(self.raw["interline"]["capture_rate"])


@lru_cache(maxsize=1)
def load_params(path: Path | None = None) -> Params:
    """Load and cache the economic parameter set."""
    target = path or COST_PARAMS_FILE
    with open(target, encoding="utf-8") as fh:
        return Params(raw=yaml.safe_load(fh))
