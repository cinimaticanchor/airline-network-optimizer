"""When a flying month's routes must be finalized, and whether that deadline has
already arrived.

The rule comes directly from the user's own worked example: routes for August 2027
must be finalized by June 2027 -- two calendar months of lead time for crew bidding,
GDS/schedule loading, and slot confirmation. This module is the pure date
arithmetic behind that rule, plus a small on-disk ledger (`data/planning_state.yaml`)
recording which target months have already been finalized, so a re-run does not
churn on a month that is already locked.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from anos.config import DATA_DIR

PLANNING_STATE_FILE = DATA_DIR / "planning_state.yaml"

# Calendar months of lead time between finalizing a month's routes and flying them.
LEAD_MONTHS = 2


def _add_months(d: date, months: int) -> date:
    """`d` (normalized to the 1st) shifted by a signed number of calendar months."""
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def finalization_cutoff(target_month: date) -> date:
    """The date by which `target_month`'s routes must be finalized.

    `target_month` is normalized to the first of its month; the result is exactly
    `LEAD_MONTHS` calendar months earlier -- August 2027 -> June 2027 is the
    literal worked example this rule comes from.
    """
    return _add_months(target_month.replace(day=1), -LEAD_MONTHS)


def is_due_for_finalization(target_month: date, as_of: date) -> bool:
    """True once `as_of` has reached or passed `target_month`'s finalization cutoff."""
    return as_of >= finalization_cutoff(target_month)


def load_planning_state(path: Path | None = None) -> dict[str, Any]:
    """The finalization ledger -- an empty one if it does not exist yet."""
    target = path or PLANNING_STATE_FILE
    if not target.exists():
        return {"finalized": []}
    with open(target, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {"finalized": []}


def is_already_finalized(target_month: date, state: dict[str, Any] | None = None) -> bool:
    st = state if state is not None else load_planning_state()
    normalized = target_month.replace(day=1).isoformat()
    return any(entry["target_month"] == normalized for entry in st.get("finalized", []))


def record_finalization(
    target_month: date,
    *,
    report_path: str,
    finalized_on: date,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one finalization record and persist the ledger. Returns the updated
    state (so a caller can chain further checks without re-reading the file)."""
    target = path or PLANNING_STATE_FILE
    state = load_planning_state(target)
    state.setdefault("finalized", []).append(
        {
            "target_month": target_month.replace(day=1).isoformat(),
            "finalized_on": finalized_on.isoformat(),
            "report_path": report_path,
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(state, fh, sort_keys=False)
    return state


def due_target_months(
    as_of: date, *, state: dict[str, Any] | None = None, horizon_months: int = 12
) -> list[date]:
    """Target months whose finalization cutoff has passed as of `as_of` but which
    are not yet recorded as finalized -- what an automated refresh run should act
    on next. Because the cutoff is `LEAD_MONTHS` months before the target, this can
    only ever surface a target month from the current one up to `LEAD_MONTHS`
    months ahead; `horizon_months` is just a generous scan bound, not a real limit.
    """
    st = state if state is not None else load_planning_state()
    start = as_of.replace(day=1)
    due = []
    for i in range(horizon_months):
        candidate = _add_months(start, i)
        if is_due_for_finalization(candidate, as_of) and not is_already_finalized(candidate, st):
            due.append(candidate)
    return due
