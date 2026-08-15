"""Feasibility checking, scenario isolation and report rendering."""

from __future__ import annotations

import copy
from datetime import date

import pytest

from anos.data.loaders import load_fleet_config, load_markets, validate_reference_data
from anos.optimize.fleet_assignment import solve_network
from anos.optimize.tail_feasibility import check_plan
from anos.scenarios.engine import (
    delivery_slip,
    demand_shock,
    fuel_shock,
    run_scenario,
    standard_suite,
)

TARGET = date(2027, 3, 1)


@pytest.fixture(scope="module")
def plan():
    return solve_network(TARGET)


# -- reference data ----------------------------------------------------------


def test_shipped_reference_data_is_consistent():
    assert validate_reference_data() == []


# -- feasibility -------------------------------------------------------------


def test_baseline_plan_is_operable(plan):
    """The shipped configuration should not produce a plan nobody can fly."""
    report = check_plan(plan)
    assert report.is_feasible, [f.detail for f in report.blockers]


def test_findings_are_well_formed(plan):
    report = check_plan(plan)
    for f in report.findings:
        assert f.severity in ("blocker", "warning", "info")
        assert f.category and f.subject and f.detail


def test_slot_usage_is_reported_for_served_airports(plan):
    report = check_plan(plan)
    assert report.slot_usage
    assert all(v >= 0 for v in report.slot_usage.values())


def test_saturated_fleet_raises_a_capacity_finding(plan):
    """Any type above the warning threshold must be surfaced, not silently accepted."""
    report = check_plan(plan)
    saturated = {c for c, r in plan.utilisation().items() if r > 0.95}
    flagged = {f.subject for f in report.findings if f.category == "fleet capacity"}
    assert saturated <= flagged


def test_zero_capacity_for_a_used_type_is_a_blocker(plan):
    """Contrived: strip a type's availability and the checker must call it a blocker."""
    broken = copy.deepcopy(plan)
    used_type = next(iter(broken.fleet_used))
    broken.fleet_available[used_type] = 0.0

    report = check_plan(broken)
    assert not report.is_feasible
    assert any(f.subject == used_type for f in report.blockers)


# -- scenarios ---------------------------------------------------------------


def test_scenarios_do_not_mutate_shared_state(plan):
    """Every scenario works on a copy. If this fails, results depend on run order."""
    before_fleet = copy.deepcopy(load_fleet_config())
    before_markets = list(load_markets())

    for scenario in (delivery_slip(18), fuel_shock(30), demand_shock(-20)):
        run_scenario(scenario, TARGET, plan)

    assert load_fleet_config() == before_fleet
    assert load_markets() == before_markets


def test_fuel_increase_reduces_contribution(plan):
    result = run_scenario(fuel_shock(30), TARGET, plan)
    assert result.contribution_delta_usd < 0


def test_fuel_decrease_improves_contribution(plan):
    result = run_scenario(fuel_shock(-20), TARGET, plan)
    assert result.contribution_delta_usd > 0


def test_demand_collapse_reduces_passengers(plan):
    result = run_scenario(demand_shock(-30), TARGET, plan)
    assert result.pax_delta < 0
    assert result.contribution_delta_usd < 0


def test_delivery_slip_never_improves_the_network(plan):
    """Fewer aircraft cannot make the plan better."""
    result = run_scenario(delivery_slip(18), TARGET, plan)
    assert result.contribution_delta_usd <= 1.0


def test_unmeetable_service_floors_are_reported_not_raised(plan):
    """A scenario that breaks the service floors is a finding, not a crash."""
    result = run_scenario(delivery_slip(12), TARGET, plan)
    assert result.plan is not None
    if result.service_floors_unmeetable:
        assert result.unmet_floors


def test_whole_standard_suite_runs(plan):
    for scenario in standard_suite():
        result = run_scenario(scenario, TARGET, plan)
        assert result.plan.markets_served() > 0


# -- reporting ---------------------------------------------------------------


def test_report_renders_valid_self_contained_html(plan, tmp_path):
    from anos.report.html_report import render_report, write_report

    report = check_plan(plan)
    results = [run_scenario(fuel_shock(25), TARGET, plan)]

    doc = render_report(plan, report, results)
    assert doc.startswith("<!doctype html>")
    assert doc.rstrip().endswith("</html>")
    # Self-contained: no external fetches of any kind.
    for marker in ("<script src=", 'href="http', "@import"):
        assert marker not in doc

    out = write_report(plan, report, results, tmp_path / "r.html")
    assert out.exists() and out.stat().st_size > 5000


def test_report_escapes_untrusted_text(plan, tmp_path):
    """Market ids reach the page as text and must never render as markup."""
    from anos.models import FeasibilityFinding
    from anos.report.html_report import render_report

    report = check_plan(plan)
    report.findings.append(
        FeasibilityFinding(
            severity="info", category="test", subject="<img src=x onerror=1>", detail="a & b"
        )
    )
    doc = render_report(plan, report, [])
    assert "<img src=x" not in doc
    assert "&lt;img src=x" in doc
