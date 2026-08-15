"""Demand forecasting: how many people actually want to fly, and when.

Two effects drive the numbers the optimiser sees.

**Seasonality.** A Goa route peaks in December; a Hajj route peaks in June. Each
market carries a profile and the monthly multipliers live in cost_params.yaml.

**Schedule quality.** Frequency is not just a cost lever -- it is a demand lever.
A market served twice daily captures more than twice what a daily service captures,
because business travellers pick the airline that can get them back the same evening.
The effect saturates: the tenth daily frequency adds far less than the second. This is
modelled as a concave curve normalised so that the reference frequency reproduces the
base demand figure in markets.csv.

Two things damp the uplift:
  * leisure markets reward frequency far less than business markets, so it scales
    with the market's business mix;
  * in contested markets competitors match added frequency, so the capture is
    scaled down by the competition index.

In Tier 2 this whole module is replaced by a model trained on real PSS/GDS booking
data. The interface -- `effective_demand(market, freq, month)` -- is what the rest of
the pipeline depends on, and would not change.
"""

from __future__ import annotations

import math
from functools import lru_cache

from anos.config import Params, load_params
from anos.models import Market

# How much of the frequency uplift competitors are assumed to neutralise in a
# fully contested market.
COMPETITIVE_DAMPING = 0.40


def seasonal_demand(market: Market, month: int, params: Params | None = None) -> float:
    """Base daily demand for a market in a given month, before schedule effects."""
    par = params or load_params()
    return market.base_daily_demand * par.seasonality(market.seasonality_profile, month)


@lru_cache(maxsize=4096)
def _max_uplift(business_mix: float, competition_index: float, ceiling: float, weight: float) -> float:
    """Peak demand uplift this market can earn from frequency alone."""
    mix_factor = (1.0 - weight) + weight * business_mix
    competitive_factor = 1.0 - COMPETITIVE_DAMPING * competition_index
    return ceiling * mix_factor * competitive_factor


def schedule_quality(market: Market, freq: int, params: Params | None = None) -> float:
    """Demand multiplier from operating `freq` daily round trips.

    Normalised so that `frequency_ref` returns 1.0 -- i.e. the demand figures in
    markets.csv are quoted at the reference schedule. Zero frequency returns zero:
    an unserved market carries nobody.
    """
    if freq <= 0:
        return 0.0

    par = params or load_params()
    k = par.saturation_k
    f_ref = par.frequency_ref

    saturating = 1.0 - math.exp(-k * freq)
    reference = 1.0 - math.exp(-k * f_ref)
    quality = saturating / reference

    ceiling = 1.0 + _max_uplift(
        market.business_mix,
        market.competition_index,
        par.max_frequency_uplift,
        par.business_mix_weight,
    )
    return min(quality, ceiling)


def effective_demand(
    market: Market, freq: int, month: int, params: Params | None = None
) -> float:
    """Daily demand each way, given the month and the frequency offered."""
    par = params or load_params()
    return seasonal_demand(market, month, par) * schedule_quality(market, freq, par)


def demand_curve(
    market: Market, month: int, params: Params | None = None
) -> dict[int, float]:
    """Demand at every feasible frequency for this market.

    The optimiser consumes this directly. Because daily frequencies are small
    integers, the curve is enumerated exactly rather than approximated by piecewise
    segments -- which removes linearisation error from the model at negligible cost.
    """
    par = params or load_params()
    return {
        f: effective_demand(market, f, month, par)
        for f in range(0, market.max_daily_freq + 1)
    }
