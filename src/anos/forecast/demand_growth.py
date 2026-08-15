"""Multi-year demand growth: how big the addressable market itself gets, not how well
a schedule captures it (that is `anos.forecast.demand`).

Every market in `markets.csv` carries one `base_daily_demand` figure, calibrated to
roughly the present day. A single-year solve treats that as a constant. A multi-year
plan cannot: India domestic traffic is growing at a real, well-documented double-digit
clip, and a fleet sized for today's demand is undersized in three years by
construction, not by bad luck.

The growth curve is deliberately **not** a flat CAGR held forever. A market cannot
compound at 11%/year for two decades -- India's own near-term rate is assumed to decay
toward the mature, GDP-linked global rate (Airbus GMF: ~3.9%/year) as the market
matures:

    g(t) = g_inf + (g0 - g_inf) * 0.5 ** (t / half_life)

where `t` is years elapsed since the anchor year, `g0` is the near-term rate and
`g_inf` is the long-run rate. At t=0 this is exactly `g0`; as t grows it decays
geometrically toward `g_inf`, halving the gap every `half_life` years.

Cumulative growth is compounded **year by year**, not via a closed-form integral of
the decay curve -- the same "enumerate the exact small step" philosophy this codebase
already applies to frequency levels and fare levels, traded here for a few extra
arithmetic operations instead of a closed-form shortcut that would be harder to audit.

See `data/demand_growth.yaml` for the actual assumptions and their sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from anos.data.loaders import load_demand_growth_config
from anos.models import Market


@dataclass(frozen=True)
class GrowthCurve:
    """One region-bucket's near-term/long-run growth rates and decay speed."""

    near_term_cagr: float
    long_run_cagr: float
    decay_half_life_years: float


@dataclass(frozen=True)
class GrowthAssumptions:
    """Typed view over demand_growth.yaml, same thin-wrapper pattern as `Params`."""

    raw: dict[str, Any]

    @property
    def anchor_year(self) -> int:
        return int(self.raw["anchor_year"])

    def bucket_for_region(self, region: str) -> str:
        """Raises KeyError for an unmapped region -- validate_reference_data() is
        what should catch this before a solve ever gets here."""
        return self.raw["region_bucket_map"][region]

    def curve_for_region(self, region: str) -> GrowthCurve:
        bucket = self.bucket_for_region(region)
        spec = self.raw["buckets"][bucket]
        return GrowthCurve(
            near_term_cagr=float(spec["near_term_cagr"]),
            long_run_cagr=float(spec["long_run_cagr"]),
            decay_half_life_years=float(spec["decay_half_life_years"]),
        )


@lru_cache(maxsize=1)
def load_growth_assumptions(path: Path | None = None) -> GrowthAssumptions:
    return GrowthAssumptions(raw=load_demand_growth_config(path))


def annual_growth_rate(curve: GrowthCurve, years_since_anchor: int) -> float:
    """The instantaneous growth rate `years_since_anchor` years after the anchor."""
    decay = 0.5 ** (years_since_anchor / curve.decay_half_life_years)
    return curve.long_run_cagr + (curve.near_term_cagr - curve.long_run_cagr) * decay


def cumulative_growth_factor(curve: GrowthCurve, from_year: int, to_year: int) -> float:
    """Compound `annual_growth_rate` year by year from `from_year` to `to_year`.

    `from_year` is treated as the curve's own t=0 -- i.e. this always answers "how
    much bigger is the market at `to_year`, given it was at this curve's near-term
    rate at `from_year`." Callers wanting growth since the configured anchor year
    pass `assumptions.anchor_year` as `from_year` (see `grow_market`).
    """
    if to_year < from_year:
        raise ValueError(f"to_year ({to_year}) must be >= from_year ({from_year})")
    factor = 1.0
    for year in range(from_year, to_year):
        factor *= 1.0 + annual_growth_rate(curve, year - from_year)
    return factor


def grow_market(
    market: Market, to_year: int, *, assumptions: GrowthAssumptions | None = None
) -> Market:
    """The same market with `base_daily_demand` grown to `to_year`.

    Always call this on `load_markets()`'s pristine output, never on an
    already-grown `Market` -- the anchor year is a single global fact, and chaining
    calls would compound the same years twice.
    """
    a = assumptions or load_growth_assumptions()
    curve = a.curve_for_region(market.region)
    factor = cumulative_growth_factor(curve, a.anchor_year, to_year)
    return Market(**{**market.__dict__, "base_daily_demand": market.base_daily_demand * factor})


def grow_markets(
    markets: list[Market], to_year: int, *, assumptions: GrowthAssumptions | None = None
) -> list[Market]:
    a = assumptions or load_growth_assumptions()
    return [grow_market(m, to_year, assumptions=a) for m in markets]
