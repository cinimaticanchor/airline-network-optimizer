"""Load reference data into domain objects.

Every loader is cached: the reference set is small, read-only, and touched by every
stage of the pipeline.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from anos.config import (
    AIRCRAFT_TYPES_FILE,
    AIRPORTS_FILE,
    DEMAND_GROWTH_FILE,
    EXPANSION_ASSUMPTIONS_FILE,
    FLEET_FILE,
    HUB_TOPOLOGY_FILE,
    MARKETS_FILE,
    TIMESPACE_FILE,
)
from anos.models import AircraftType, Airport, MaintenanceProgramme, Market


def _yes(value: str) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


@lru_cache(maxsize=1)
def load_aircraft_types(path: Path | None = None) -> dict[str, AircraftType]:
    """Aircraft specifications, keyed by type code."""
    with open(path or AIRCRAFT_TYPES_FILE, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    out: dict[str, AircraftType] = {}
    for code, spec in raw["types"].items():
        out[code] = AircraftType(
            code=code,
            name=spec["name"],
            body=spec["body"],
            seats=int(spec["seats"]),
            range_km=float(spec["range_km"]),
            cruise_kmh=float(spec["cruise_kmh"]),
            fuel_burn_kgh=float(spec["fuel_burn_kgh"]),
            min_turn_min=int(spec["min_turn_min"]),
            max_daily_util_h=float(spec["max_daily_util_h"]),
            ownership_cost_per_day_usd=float(spec["ownership_cost_per_day_usd"]),
            crew_cost_per_block_hour_usd=float(spec["crew_cost_per_block_hour_usd"]),
            generation=spec["generation"],
        )
    return out


@lru_cache(maxsize=1)
def load_maintenance_programmes(
    path: Path | None = None,
) -> dict[str, MaintenanceProgramme]:
    """Maintenance check intervals and downtime, keyed by aircraft type code."""
    with open(path or AIRCRAFT_TYPES_FILE, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return {
        code: MaintenanceProgramme(
            a_check_interval_fh=float(spec["a_check_interval_fh"]),
            a_check_downtime_h=float(spec["a_check_downtime_h"]),
            c_check_interval_months=int(spec["c_check_interval_months"]),
            c_check_downtime_days=int(spec["c_check_downtime_days"]),
            d_check_interval_years=int(spec["d_check_interval_years"]),
            d_check_downtime_days=int(spec["d_check_downtime_days"]),
        )
        for code, spec in raw["maintenance"].items()
    }


@lru_cache(maxsize=1)
def load_airports(path: Path | None = None) -> dict[str, Airport]:
    """Airport reference data, keyed by IATA code."""
    out: dict[str, Airport] = {}
    with open(path or AIRPORTS_FILE, encoding="utf-8", newline="") as fh:
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        for row in csv.DictReader(rows):
            out[row["iata"]] = Airport(
                iata=row["iata"],
                name=row["name"],
                city=row["city"],
                country=row["country"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                widebody_capable=_yes(row["widebody_capable"]),
                slot_constrained=_yes(row["slot_constrained"]),
                daily_slot_cap=int(row["daily_slot_cap"]),
                curfew=_yes(row["curfew"]),
                timezone_offset_h=float(row["timezone_offset_h"]),
            )
    return out


@lru_cache(maxsize=1)
def load_markets(path: Path | None = None) -> list[Market]:
    """Addressable O&D markets."""
    out: list[Market] = []
    with open(path or MARKETS_FILE, encoding="utf-8", newline="") as fh:
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        for row in csv.DictReader(rows):
            out.append(
                Market(
                    market_id=row["market_id"],
                    origin=row["origin"],
                    destination=row["destination"],
                    region=row["region"],
                    base_daily_demand=float(row["base_daily_demand"]),
                    avg_fare_usd=float(row["avg_fare_usd"]),
                    business_mix=float(row["business_mix"]),
                    competition_index=float(row["competition_index"]),
                    min_daily_freq=int(row["min_daily_freq"]),
                    max_daily_freq=int(row["max_daily_freq"]),
                    seasonality_profile=row["seasonality_profile"],
                    strategic=_yes(row["strategic"]),
                    interline_available=_yes(row.get("interline_available", "no")),
                )
            )
    return out


@lru_cache(maxsize=1)
def load_fleet_config(path: Path | None = None) -> dict[str, Any]:
    """Raw fleet and order-book configuration.

    Returned as a dict rather than a domain object because the fleet timeline is
    what actually consumes it, and that module needs the delivery/retirement
    windows in their raw form.
    """
    with open(path or FLEET_FILE, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    raw["as_of"] = _as_date(raw["as_of"])
    for block in ("orders", "retirements", "retrofits"):
        for entry in raw.get(block, []) or []:
            entry["start"] = _as_date(entry["start"])
            entry["end"] = _as_date(entry["end"])
    return raw


@lru_cache(maxsize=1)
def load_hub_topology(path: Path | None = None) -> dict[str, Any]:
    """Hub structure for connection/banking and rotation strategies (opt-in features)."""
    with open(path or HUB_TOPOLOGY_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_timespace_config(path: Path | None = None) -> dict[str, Any]:
    """Bucket count and debugging-anchor config for the time-space fleet-assignment
    core (opt-in, `anos.optimize.timespace_assignment`)."""
    with open(path or TIMESPACE_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_demand_growth_config(path: Path | None = None) -> dict[str, Any]:
    """Long-range demand growth curves and region->bucket map, for
    `anos.forecast.demand_growth` (used by `anos.planning.scenario_ladder`)."""
    with open(path or DEMAND_GROWTH_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_expansion_assumptions_config(path: Path | None = None) -> dict[str, Any]:
    """Discount rate, order lead times and detection/sizing thresholds for
    `anos.planning.fleet_expansion` (opt-in, multi-year expansion planning only)."""
    with open(path or EXPANSION_ASSUMPTIONS_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_reference_data() -> list[str]:
    """Cross-check the reference set for internal consistency.

    Returns a list of problems; empty means the data hangs together. Called by the
    CLI before any solve, because a silently missing airport or an order for an
    unknown type produces a plan that looks fine and is wrong.
    """
    problems: list[str] = []
    types = load_aircraft_types()
    airports = load_airports()
    markets = load_markets()
    fleet = load_fleet_config()

    for code in fleet.get("in_service", {}):
        if code not in types:
            problems.append(f"fleet.yaml in_service references unknown type '{code}'")

    for block in ("orders", "retirements", "retrofits"):
        for entry in fleet.get(block, []) or []:
            if entry["type"] not in types:
                problems.append(f"fleet.yaml {block} references unknown type '{entry['type']}'")
            if entry["end"] < entry["start"]:
                problems.append(
                    f"fleet.yaml {block} entry for '{entry['type']}' ends before it starts"
                )

    in_service = fleet.get("in_service", {})
    for entry in fleet.get("retirements", []) or []:
        held = in_service.get(entry["type"], 0)
        if entry["count"] > held:
            problems.append(
                f"fleet.yaml retires {entry['count']} {entry['type']} but only {held} in service"
            )

    seen_ids: set[str] = set()
    for m in markets:
        if m.market_id in seen_ids:
            problems.append(f"markets.csv has duplicate market_id '{m.market_id}'")
        seen_ids.add(m.market_id)
        for code in (m.origin, m.destination):
            if code not in airports:
                problems.append(f"market '{m.market_id}' references unknown airport '{code}'")
        if m.min_daily_freq > m.max_daily_freq:
            problems.append(f"market '{m.market_id}' has min_daily_freq above max_daily_freq")
        if not 0.0 <= m.business_mix <= 1.0:
            problems.append(f"market '{m.market_id}' has business_mix outside 0..1")
        if not 0.0 <= m.competition_index <= 1.0:
            problems.append(f"market '{m.market_id}' has competition_index outside 0..1")

    topology = load_hub_topology()
    for hub in topology["hubs"]["primary"] + topology["hubs"]["secondary_candidates"]:
        if hub not in airports:
            problems.append(f"hub_topology.yaml references unknown airport '{hub}'")

    growth = load_demand_growth_config()
    region_bucket_map = growth["region_bucket_map"]
    buckets = growth["buckets"]
    seen_regions: set[str] = set()
    for m in markets:
        if m.region in seen_regions:
            continue
        seen_regions.add(m.region)
        if m.region not in region_bucket_map:
            problems.append(
                f"market region '{m.region}' has no entry in demand_growth.yaml's "
                "region_bucket_map"
            )
        elif region_bucket_map[m.region] not in buckets:
            problems.append(
                f"demand_growth.yaml region_bucket_map maps '{m.region}' to unknown "
                f"bucket '{region_bucket_map[m.region]}'"
            )

    return problems
