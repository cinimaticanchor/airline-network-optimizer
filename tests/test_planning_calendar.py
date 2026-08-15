"""Tests for anos.planning.planning_calendar: the 2-month finalization-lead-time
rule and its on-disk ledger.

`due_target_months`/`record_finalization` tests write to a `tmp_path` ledger file,
never the real `data/planning_state.yaml`.
"""

from __future__ import annotations

from datetime import date

from anos.planning.planning_calendar import (
    due_target_months,
    finalization_cutoff,
    is_already_finalized,
    is_due_for_finalization,
    load_planning_state,
    record_finalization,
)

# -- finalization_cutoff ----------------------------------------------------------


def test_finalization_cutoff_matches_the_worked_example():
    """The user's own literal example: August 2027 routes finalize by June 2027."""
    assert finalization_cutoff(date(2027, 8, 1)) == date(2027, 6, 1)


def test_finalization_cutoff_normalizes_to_the_first_of_the_month():
    assert finalization_cutoff(date(2027, 8, 15)) == date(2027, 6, 1)


def test_finalization_cutoff_wraps_across_a_year_boundary():
    assert finalization_cutoff(date(2027, 1, 1)) == date(2026, 11, 1)
    assert finalization_cutoff(date(2027, 2, 1)) == date(2026, 12, 1)


# -- is_due_for_finalization -------------------------------------------------------


def test_is_due_for_finalization_true_exactly_at_the_cutoff():
    assert is_due_for_finalization(date(2027, 8, 1), as_of=date(2027, 6, 1)) is True


def test_is_due_for_finalization_false_the_day_before_the_cutoff():
    assert is_due_for_finalization(date(2027, 8, 1), as_of=date(2027, 5, 31)) is False


def test_is_due_for_finalization_true_well_after_the_cutoff():
    assert is_due_for_finalization(date(2027, 8, 1), as_of=date(2027, 7, 15)) is True


# -- ledger: load / record / is_already_finalized ----------------------------------


def test_missing_ledger_file_loads_as_empty(tmp_path):
    state = load_planning_state(tmp_path / "does-not-exist.yaml")
    assert state == {"finalized": []}


def test_record_finalization_persists_and_round_trips(tmp_path):
    ledger = tmp_path / "planning_state.yaml"
    record_finalization(
        date(2027, 8, 1),
        report_path="output/expansion-plan-2027-08.html",
        finalized_on=date(2027, 6, 1),
        path=ledger,
    )

    state = load_planning_state(ledger)
    assert len(state["finalized"]) == 1
    entry = state["finalized"][0]
    assert entry["target_month"] == "2027-08-01"
    assert entry["finalized_on"] == "2027-06-01"
    assert entry["report_path"] == "output/expansion-plan-2027-08.html"


def test_is_already_finalized_true_after_recording(tmp_path):
    ledger = tmp_path / "planning_state.yaml"
    record_finalization(
        date(2027, 8, 1), report_path="x", finalized_on=date(2027, 6, 1), path=ledger
    )
    state = load_planning_state(ledger)

    assert is_already_finalized(date(2027, 8, 1), state) is True
    assert is_already_finalized(date(2027, 8, 15), state) is True  # same month, any day
    assert is_already_finalized(date(2027, 9, 1), state) is False


def test_record_finalization_appends_not_overwrites(tmp_path):
    ledger = tmp_path / "planning_state.yaml"
    record_finalization(date(2027, 8, 1), report_path="a", finalized_on=date(2027, 6, 1), path=ledger)
    record_finalization(date(2027, 9, 1), report_path="b", finalized_on=date(2027, 7, 1), path=ledger)

    state = load_planning_state(ledger)
    assert len(state["finalized"]) == 2
    assert {e["target_month"] for e in state["finalized"]} == {"2027-08-01", "2027-09-01"}


# -- due_target_months --------------------------------------------------------------


def test_due_target_months_finds_the_current_month_when_ledger_is_empty(tmp_path):
    ledger = tmp_path / "planning_state.yaml"
    state = load_planning_state(ledger)
    due = due_target_months(date(2027, 6, 15), state=state)
    assert date(2027, 6, 1) in due


def test_due_target_months_excludes_a_month_already_finalized(tmp_path):
    ledger = tmp_path / "planning_state.yaml"
    record_finalization(date(2027, 6, 1), report_path="x", finalized_on=date(2027, 6, 1), path=ledger)
    state = load_planning_state(ledger)

    due = due_target_months(date(2027, 6, 15), state=state)
    assert date(2027, 6, 1) not in due


def test_due_target_months_never_includes_more_than_two_months_ahead(tmp_path):
    state = load_planning_state(tmp_path / "does-not-exist.yaml")
    due = due_target_months(date(2027, 6, 1), state=state)
    assert all(d <= date(2027, 8, 1) for d in due)
    assert date(2027, 9, 1) not in due
