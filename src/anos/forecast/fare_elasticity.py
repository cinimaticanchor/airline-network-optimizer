"""Fare elasticity: an opt-in demand response to price, layered on top of the
frequency-driven ceiling in `anos.forecast.demand`.

`demand.py`'s `effective_demand(market, freq, month)` interface is explicitly promised
not to change (see its module docstring), and its demand ceiling is real: it is how
much a market will fly *at the reference fare*. What it cannot express is a passenger
who would fly if the fare were lower. This module adds that as a second, independent
multiplier -- kept in a separate file so `demand.py`'s contract stays untouched.

STATED ASSUMPTION, not derived from data: the elasticity coefficients below are a
judgement call (see cost_params.yaml's `fare_elasticity` block), not fitted to booking
data. Treat `enable_fare_elasticity=True` results as a what-if, not a forecast, until
Tier 2 replaces this with real fare-class curves.
"""

from __future__ import annotations

from anos.config import Params, load_params
from anos.models import Market


def market_elasticity(market: Market, params: Params | None = None) -> float:
    """Own-price elasticity for this market, blended by business mix.

    Mirrors the mix_factor idiom in `demand.py`'s `_max_uplift`: leisure demand is
    materially more price-sensitive than business demand, which books on schedule and
    itinerary rather than fare.
    """
    par = params or load_params()
    b = market.business_mix
    return b * par.business_elasticity + (1.0 - b) * par.leisure_elasticity


def fare_levels(market: Market, params: Params | None = None) -> list[float]:
    """The small enumerated set of fares this market may be sold at."""
    par = params or load_params()
    return [round(market.avg_fare_usd * m, 2) for m in par.fare_multipliers]


def demand_multiplier_for_fare(
    market: Market, fare_usd: float, params: Params | None = None
) -> float:
    """Demand multiplier for selling this market at `fare_usd` instead of its
    reference fare. Exactly 1.0 at the reference fare by construction."""
    par = params or load_params()
    if market.avg_fare_usd <= 0:
        return 1.0
    ratio = fare_usd / market.avg_fare_usd
    eps = market_elasticity(market, par)
    multiplier = ratio**eps
    lo, hi = par.fare_elasticity_multiplier_band
    return max(lo, min(hi, multiplier))
