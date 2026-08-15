"""Command line interface.

    anos validate                 check the reference data hangs together
    anos fleet                    fleet availability on a date
    anos solve                    optimise the network
    anos feasibility              optimise, then check the plan can be flown
    anos scenarios                stress the plan against the standard suite
    anos report                   write an HTML decision-support report
    anos compare                  baseline vs. banking/topology/elasticity extensions
    anos expansion-plan           multi-year fleet-expansion NPV plan
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from anos.config import OUTPUT_DIR, load_params
from anos.data.fleet_timeline import daily_aircraft_hours, fleet_on
from anos.data.loaders import load_aircraft_types, load_markets, validate_reference_data
from anos.models import NetworkPlan
from anos.optimize.fleet_assignment import InfeasibleNetworkError, solve_network
from anos.optimize.tail_feasibility import check_plan

BAR = "-" * 78


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _usd(value: float) -> str:
    """Format money at whatever magnitude reads best."""
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}k"
    return f"${value:,.0f}"


# -- commands ----------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    problems = validate_reference_data()
    if not problems:
        markets = load_markets()
        types = load_aircraft_types()
        print(f"Reference data is consistent: {len(markets)} markets, {len(types)} aircraft types.")
        return 0

    print(f"Found {len(problems)} problem(s) in the reference data:\n")
    for p in problems:
        print(f"  - {p}")
    return 1


def cmd_fleet(args: argparse.Namespace) -> int:
    snapshot = fleet_on(args.date, probabilistic=args.probabilistic)
    hours = daily_aircraft_hours(snapshot)
    types = load_aircraft_types()

    mode = "risk-adjusted" if args.probabilistic else "nominal"
    print(f"\nFleet availability on {args.date} ({mode} delivery dates)")
    print(BAR)
    print(f"{'Type':<12}{'Nominal':>10}{'Retrofit':>10}{'Maint':>9}{'Available':>11}{'Hours/day':>11}")
    print(BAR)

    for code in sorted(snapshot.nominal, key=lambda c: -snapshot.nominal[c]):
        nominal = snapshot.nominal[code]
        if nominal < 0.05:
            continue
        print(
            f"{code:<12}{nominal:>10.1f}{snapshot.in_retrofit.get(code, 0.0):>10.1f}"
            f"{snapshot.in_maintenance.get(code, 0.0):>9.1f}"
            f"{snapshot.available.get(code, 0.0):>11.1f}{hours.get(code, 0.0):>11.0f}"
        )

    print(BAR)
    narrow = sum(n for c, n in snapshot.nominal.items() if types[c].body == "narrow")
    wide = sum(n for c, n in snapshot.nominal.items() if types[c].body == "wide")
    print(
        f"{'TOTAL':<12}{snapshot.total_nominal():>10.1f}"
        f"{sum(snapshot.in_retrofit.values()):>10.1f}"
        f"{sum(snapshot.in_maintenance.values()):>9.1f}"
        f"{snapshot.total_available():>11.1f}{sum(hours.values()):>11.0f}"
    )
    print(f"\n  narrowbody {narrow:,.0f}   widebody {wide:,.0f}")
    print(
        f"  {snapshot.total_nominal() - snapshot.total_available():,.1f} airframes "
        f"({1 - snapshot.total_available() / max(snapshot.total_nominal(), 1):.1%}) "
        "unavailable to the schedule\n"
    )
    return 0


def _print_plan_summary(plan: NetworkPlan) -> None:
    params = load_params()
    print(f"\nNetwork plan for {plan.as_of} (month {plan.month:02d}, scenario '{plan.scenario}')")
    print(BAR)
    print(f"  solver              {plan.solver_status} in {plan.solve_seconds:.2f}s")
    print(f"  markets served      {plan.markets_served()} of {len(plan.market_plans)}")
    print(f"  daily departures    {plan.total_departures:,}")
    print(f"  daily passengers    {plan.total_pax * 2:,.0f}")
    print(f"  seats offered       {plan.total_seats * 2:,.0f}")
    print(f"  load factor         {plan.network_load_factor:.1%}  (target {params.target_load_factor:.0%})")
    print(BAR)
    print(f"  daily revenue       {_usd(plan.total_revenue_usd):>12}")
    print(f"  daily cost          {_usd(plan.total_cost_usd):>12}")
    print(f"  daily contribution  {_usd(plan.total_contribution_usd):>12}")
    print(f"  annualised          {_usd(plan.total_contribution_usd * 365):>12}")
    print(BAR)


def _print_fleet_utilisation(plan: NetworkPlan) -> None:
    print("\nFleet utilisation")
    print(BAR)
    print(f"{'Type':<12}{'Hours used':>12}{'Available':>12}{'Utilisation':>13}")
    print(BAR)
    for code, ratio in sorted(plan.utilisation().items(), key=lambda kv: -kv[1]):
        used = plan.fleet_used.get(code, 0.0)
        if used < 0.5:
            continue
        flag = "  <- saturated" if ratio > 0.95 else ""
        print(
            f"{code:<12}{used:>12,.0f}{plan.fleet_available.get(code, 0.0):>12,.0f}"
            f"{ratio:>12.1%}{flag}"
        )
    print(BAR)


def _print_route_table(plan: NetworkPlan, n: int) -> None:
    ranked = sorted(plan.market_plans, key=lambda p: -p.contribution_usd)
    served = [p for p in ranked if p.total_frequency > 0]

    def rows(entries, title):
        print(f"\n{title}")
        print(BAR)
        print(f"{'Market':<12}{'Freq':>5}  {'Fleet mix':<26}{'LF':>7}{'Contribution/day':>18}")
        print(BAR)
        for p in entries:
            mix = ", ".join(f"{t}x{n}" for t, n in sorted(p.frequencies.items()))
            if len(mix) > 25:
                mix = mix[:22] + "..."
            print(
                f"{p.market_id:<12}{p.total_frequency:>5}  {mix:<26}"
                f"{p.load_factor:>6.0%}{_usd(p.contribution_usd):>18}"
            )
        print(BAR)

    rows(served[:n], f"Top {min(n, len(served))} routes by daily contribution")
    losers = [p for p in served if p.contribution_usd < 0]
    if losers:
        rows(sorted(losers, key=lambda p: p.contribution_usd)[:n], "Loss-making routes")

    unserved = [p for p in ranked if p.total_frequency == 0]
    if unserved:
        print(f"\n{len(unserved)} market(s) left unserved: " + ", ".join(p.market_id for p in unserved[:18]))
        if len(unserved) > 18:
            print(f"  ... and {len(unserved) - 18} more")


def cmd_solve(args: argparse.Namespace) -> int:
    hubs = args.hubs.split(",") if args.hubs else None
    try:
        plan = solve_network(
            args.date,
            month=args.month,
            probabilistic_fleet=args.probabilistic,
            enforce_min_service=not args.no_min_service,
            verbose=args.verbose,
            fast_turn=args.fast_turn,
            enable_fare_elasticity=args.enable_fare_elasticity,
            enable_connections=args.enable_connections,
            hubs=hubs,
            enable_rotations=args.enable_rotations,
            rotation_squad=args.rotation_squad.split(",") if args.rotation_squad else None,
        )
    except InfeasibleNetworkError as exc:
        print(f"\nNo feasible plan: {exc}\n", file=sys.stderr)
        return 1

    _print_plan_summary(plan)
    _print_fleet_utilisation(plan)
    _print_route_table(plan, args.top)
    if plan.connections_used:
        print(f"\nServed via a banked connection: {', '.join(plan.connections_used)}")
    if plan.rotations_flown:
        print("\nRotations flown:")
        for rid, n in sorted(plan.rotations_flown.items()):
            print(f"  {rid:<20}{n} loop(s)/day")
    print()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from anos.network.itinerary import solve_extended_network

    hubs = args.hubs.split(",") if args.hubs else ["DEL", "BOM"]

    print("Solving baseline...")
    try:
        baseline, extended = solve_extended_network(
            args.date,
            month=args.month,
            hubs=hubs,
            fast_turn=not args.no_fast_turn,
            enable_fare_elasticity=not args.no_fare_elasticity,
            enable_connections=not args.no_connections,
            enable_rotations=not args.no_rotations,
            rotation_lf_threshold=args.rotation_lf_threshold,
        )
    except InfeasibleNetworkError as exc:
        print(f"\nNo feasible plan: {exc}\n", file=sys.stderr)
        return 1

    dep_delta = extended.total_departures - baseline.total_departures
    contrib_delta = extended.total_contribution_usd - baseline.total_contribution_usd
    contrib_delta_pct = (
        contrib_delta / baseline.total_contribution_usd * 100
        if baseline.total_contribution_usd
        else 0.0
    )

    print(f"\nBaseline vs. extended network -- {args.date} (hubs: {', '.join(hubs)})")
    print(BAR)
    print(f"{'':<28}{'Baseline':>16}{'Extended':>16}{'Delta':>16}")
    print(BAR)
    print(
        f"{'Contribution/day':<28}{_usd(baseline.total_contribution_usd):>16}"
        f"{_usd(extended.total_contribution_usd):>16}{_usd(contrib_delta):>16}"
        f"  ({contrib_delta_pct:+.1f}%)"
    )
    print(
        f"{'Annualised':<28}{_usd(baseline.total_contribution_usd * 365):>16}"
        f"{_usd(extended.total_contribution_usd * 365):>16}"
        f"{_usd(contrib_delta * 365):>16}"
    )
    print(
        f"{'Daily departures':<28}{baseline.total_departures:>16}"
        f"{extended.total_departures:>16}{dep_delta:>+16}"
    )
    print(BAR)

    base_plans = {p.market_id: p for p in baseline.market_plans}
    if extended.connections_used:
        print("\nNewly reachable or reinforced via a banked connection:")
        for mid in extended.connections_used:
            was_unserved = base_plans.get(mid) is None or base_plans[mid].total_frequency == 0
            flag = "  <- was unserved nonstop" if was_unserved else ""
            print(f"  {mid}{flag}")

    if extended.rotations_flown:
        print("\nRotations flown (squad markets pooled into a shared loop):")
        for rid, n in sorted(extended.rotations_flown.items()):
            print(f"  {rid:<20}{n} loop(s)/day")

    from anos.network.itinerary import low_load_factor_squad

    squad = low_load_factor_squad(baseline, threshold=args.rotation_lf_threshold)
    if squad:
        print(f"\nSquad (<= {args.rotation_lf_threshold:.0%} load factor in the baseline):")
        for mid in squad:
            print(f"  {mid}  (baseline LF {base_plans[mid].load_factor:.0%})")

    unserved_baseline = sorted(
        p.market_id for p in baseline.market_plans if p.total_frequency == 0
    )
    still_unserved = [
        mid
        for mid in unserved_baseline
        if mid not in extended.connections_used
        and (
            mid not in {p.market_id: p for p in extended.market_plans}
            or {p.market_id: p for p in extended.market_plans}[mid].total_frequency == 0
        )
    ]
    if still_unserved:
        print(
            "\nStill unserved (no domestic feeder exists at the hubs tried, so "
            "banking cannot help these without a different hub or a new market row):"
        )
        print(f"  {', '.join(still_unserved)}")
    print()
    return 0


def cmd_feasibility(args: argparse.Namespace) -> int:
    try:
        plan = solve_network(
            args.date, month=args.month, probabilistic_fleet=args.probabilistic
        )
    except InfeasibleNetworkError as exc:
        print(f"\nNo feasible plan: {exc}\n", file=sys.stderr)
        return 1

    _print_plan_summary(plan)
    report = check_plan(plan)

    verdict = "OPERABLE" if report.is_feasible else "NOT OPERABLE"
    print(f"\nTail-assignment feasibility: {verdict}")
    print(f"  {len(report.blockers)} blocker(s), {len(report.warnings)} warning(s)")
    print(BAR)

    order = {"blocker": 0, "warning": 1, "info": 2}
    for finding in sorted(report.findings, key=lambda f: (order[f.severity], f.category)):
        print(f"\n  [{finding.severity.upper()}] {finding.category} -- {finding.subject}")
        print(f"      {finding.detail}")

    if not report.findings:
        print("\n  No findings. The plan is operable as drawn.")
    print(f"\n{BAR}\n")
    return 0 if report.is_feasible else 1


def cmd_scenarios(args: argparse.Namespace) -> int:
    from anos.scenarios.engine import run_suite

    print("Solving baseline...")
    try:
        baseline = solve_network(args.date, month=args.month)
    except InfeasibleNetworkError as exc:
        print(f"\nBaseline is infeasible: {exc}\n", file=sys.stderr)
        return 1

    print(f"Baseline daily contribution: {_usd(baseline.total_contribution_usd)}")
    print("Running scenario suite...\n")

    results = run_suite(args.date, baseline, month=args.month)

    print(f"Scenario impact on {args.date}")
    print(BAR)
    width = max(34, max(len(r.scenario.name) for r in results) + 2)
    print(f"{'Scenario':<{width}}{'Contribution/day':>18}{'Delta':>14}{'Delta %':>10}")
    print(BAR)
    print(f"{'baseline':<{width}}{_usd(baseline.total_contribution_usd):>18}{'-':>14}{'-':>10}")
    for r in results:
        flag = "  !" if r.service_floors_unmeetable else ""
        print(
            f"{r.scenario.name:<{width}}{_usd(r.plan.total_contribution_usd):>18}"
            f"{_usd(r.contribution_delta_usd):>14}{r.contribution_delta_pct:>9.1f}%{flag}"
        )
    print(BAR)

    breached = [r for r in results if r.service_floors_unmeetable]
    if breached:
        print("\n  ! These scenarios cannot honour the network's minimum service floors.")
        print("    The figures above are the best network achievable with those floors released.")
        for r in breached:
            shown = ", ".join(r.unmet_floors[:10])
            more = f" (+{len(r.unmet_floors) - 10} more)" if len(r.unmet_floors) > 10 else ""
            print(f"      {r.scenario.name}: {len(r.unmet_floors)} commitment(s) unmet -- {shown}{more}")

    print("\nWhat each scenario assumes")
    for r in results:
        print(f"  {r.scenario.name:<28} {r.scenario.description}")
        dropped = r.markets_dropped
        if dropped:
            shown = ", ".join(dropped[:8])
            more = f" (+{len(dropped) - 8} more)" if len(dropped) > 8 else ""
            print(f"  {'':<28} drops: {shown}{more}")
    print()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from anos.report.html_report import write_report
    from anos.scenarios.engine import run_suite

    print("Solving baseline...")
    try:
        plan = solve_network(args.date, month=args.month)
    except InfeasibleNetworkError as exc:
        print(f"\nNo feasible plan: {exc}\n", file=sys.stderr)
        return 1

    print("Checking operational feasibility...")
    feasibility = check_plan(plan)

    results = []
    if not args.no_scenarios:
        print("Running scenario suite...")
        results = run_suite(args.date, plan, month=args.month)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or OUTPUT_DIR / f"network-plan-{args.date}.html"
    write_report(plan, feasibility, results, out)
    print(f"\nWrote {out}\n")
    return 0


def cmd_expansion_plan(args: argparse.Namespace) -> int:
    from anos.data.loaders import load_fleet_config
    from anos.forecast.demand_growth import load_growth_assumptions
    from anos.planning.fleet_expansion import load_expansion_assumptions, recommend_expansion

    if args.end_year < args.start_year:
        print("--end-year must be >= --start-year", file=sys.stderr)
        return 1

    growth = load_growth_assumptions()
    assumptions = load_expansion_assumptions()
    today = load_fleet_config()["as_of"]

    print(f"Growing demand and re-solving {args.start_year}-{args.end_year} (this takes a while)...")
    result = recommend_expansion(args.start_year, args.end_year)

    print(f"\nMulti-year expansion plan: {args.start_year}-{args.end_year}")
    print(BAR)

    print("\nDemand-growth assumptions (see data/demand_growth.yaml for sources)")
    print(f"  anchor year: {growth.anchor_year}")
    for bucket, spec in growth.raw["buckets"].items():
        print(
            f"  {bucket:<28}{spec['near_term_cagr']:>6.1%} -> {spec['long_run_cagr']:>5.1%} CAGR"
            f"  (decaying with a {spec['decay_half_life_years']:.0f}-year half-life)"
        )

    print("\nYear-by-year trajectory (March checkpoints)")
    print(BAR)
    print(f"{'Year':<6}{'Contribution/day':>18}{'Annualised':>16}{'Markets served':>16}")
    print(BAR)
    for checkpoint in result.ladder.checkpoints:
        floor_flag = "  !" if checkpoint.service_floors_unmeetable else ""
        print(
            f"{checkpoint.year:<6}{_usd(checkpoint.plan.total_contribution_usd):>18}"
            f"{_usd(checkpoint.plan.total_contribution_usd * 365):>16}"
            f"{checkpoint.plan.markets_served():>16}{floor_flag}"
        )
    print(BAR)
    if any(c.service_floors_unmeetable for c in result.ladder.checkpoints):
        print("  ! that year could not honour every minimum service floor at the modelled fleet;")
        print("    the figures shown are the best network achievable with floors released.")

    fleet_signals = [s for s in result.signals if s.signal_type == "fleet_utilisation"]
    spill_signals = [s for s in result.signals if s.signal_type == "market_spill"]
    actionable_spill = [s for s in spill_signals if s.actionable]
    slot_bound_spill = [s for s in spill_signals if not s.actionable]

    print(f"\nConstraints outstanding against the final committed fleet (threshold {assumptions.fleet_utilisation_threshold:.0%} utilisation, {assumptions.min_candidate_count}+ years persistent)")
    if fleet_signals:
        print("  Fleet types still trending toward saturation:")
        for s in sorted(fleet_signals, key=lambda s: s.first_binding_year):
            print(f"    {s.subject:<14} binding from {s.first_binding_year}")
    else:
        print("  No fleet type is persistently saturated against the committed order book.")
    if actionable_spill:
        print("  Markets spilling demand that more fleet could plausibly relieve:")
        for s in sorted(actionable_spill, key=lambda s: s.first_binding_year):
            print(f"    {s.subject:<14} spilling from {s.first_binding_year}")
    if slot_bound_spill:
        print(f"  Markets spilling demand but slot-saturated, not fleet-constrained ({len(slot_bound_spill)} -- see anos.planning.constraint_detection):")
        shown = ", ".join(s.subject for s in sorted(slot_bound_spill, key=lambda s: s.first_binding_year)[:10])
        more = f" (+{len(slot_bound_spill) - 10} more)" if len(slot_bound_spill) > 10 else ""
        print(f"    {shown}{more}")

    print("\nCandidate orders evaluated")
    print(BAR)
    if not result.evaluations:
        print("  No persistent fleet-utilisation signal crossed the detection threshold in this horizon.")
    for e in result.evaluations:
        c = e.candidate
        verdict = "ACCEPTED" if e.recommended else "rejected"
        print(f"  [{verdict}] {c.count}x {c.type_code}")
        print(f"      trigger:    {c.trigger.detail}")
        overdue = "  ! already due given today's date" if c.order_year <= today.year else ""
        print(f"      order by:   {c.order_year}  (lead time to first delivery){overdue}")
        print(f"      delivers:   {c.delivery_start_year}-{c.delivery_end_year}")
        print(f"      NPV:        {_usd(e.npv_usd)}  (discounted at {assumptions.discount_rate_annual:.0%}/yr to the order year)")
        print(f"      payback:    {e.payback_year if e.payback_year is not None else 'not within this horizon'}")
        for year in sorted(e.net_cash_flow_usd):
            print(
                f"        {year}  contribution {_usd(e.contribution_delta_usd[year]):>10}"
                f"   ownership {_usd(e.ownership_cost_delta_usd[year]):>10}"
                f"   net {_usd(e.net_cash_flow_usd[year]):>10}"
            )
        if e.review_flag:
            print(f"      ! {e.review_flag}")
        print()
    print(BAR)

    print("\nFleet-availability risk")
    print(
        "  Legacy-generation types (A319/A320ceo/A321ceo, B777-200LR/300ER, B787-8) carry a\n"
        "  materially higher unscheduled_downtime_rate than neo/new-widebody types in every\n"
        "  fleet_on() snapshot above -- grounded in Air India's own documented spares-funding\n"
        "  history (13 B787s grounded for years, ~23% of fleet idle at peak pre-Tata era), not\n"
        "  an engine-family defect: the neo narrowbody fleet is CFM LEAP-powered, not the\n"
        "  Pratt & Whitney GTF variant behind the industry-wide grounding crisis. See\n"
        "  cost_params.yaml's operations.unscheduled_downtime_rate comment for the full citation."
    )

    print("\nWhat this plan cannot claim")
    print("  - Every checkpoint fixes the month at March; it does not capture within-year")
    print("    seasonality shifting which month is actually most fleet-constrained.")
    print("  - Order evaluation is greedy and sequential (most urgent signal first), not a")
    print("    joint optimisation across every candidate at once.")
    print("  - Only persistent fleet_utilisation signals are converted into candidate orders;")
    print("    a market_spill signal (even an actionable one) is surfaced above but left for a")
    print("    human planner to judge which type, if any, should relieve it.")
    print(f"  - Bounded to the {args.start_year}-{args.end_year} horizon; demand-growth curves are stated")
    print("    assumptions (Boeing/Airbus/DGCA-sourced), not fitted to Air India's own bookings,")
    print("    and are trusted less the further beyond this horizon they are extrapolated.")
    print("  - NPV is a solved-model estimate, not an exact figure: CP-SAT's own solver gap")
    print("    (tightened for this evaluation, but not eliminated) means a candidate whose NPV")
    print("    sits close to zero can flip sign between runs -- treat those as genuine toss-ups")
    print("    for a human to weigh, not as a confident accept or reject.")
    print()
    return 0


# -- entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anos",
        description="Air India network and fleet optimisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, *, month: bool = True) -> None:
        p.add_argument(
            "--date",
            type=_parse_date,
            default=date(2027, 3, 1),
            metavar="YYYY-MM-DD",
            help="date whose fleet availability the plan is built against",
        )
        p.add_argument(
            "--probabilistic",
            action="store_true",
            help="risk-adjust delivery dates by programme confidence",
        )
        if month:
            p.add_argument(
                "--month",
                type=int,
                choices=range(1, 13),
                metavar="1-12",
                help="month whose seasonality applies (defaults to the date's month)",
            )

    p_validate = sub.add_parser("validate", help="check the reference data")
    p_validate.set_defaults(func=cmd_validate)

    p_fleet = sub.add_parser("fleet", help="fleet availability on a date")
    add_common(p_fleet, month=False)
    p_fleet.set_defaults(func=cmd_fleet)

    p_solve = sub.add_parser("solve", help="optimise the network")
    add_common(p_solve)
    p_solve.add_argument("--top", type=int, default=15, help="routes to list (default 15)")
    p_solve.add_argument(
        "--no-min-service",
        action="store_true",
        help="drop minimum service floors to see unconstrained economics",
    )
    p_solve.add_argument("--verbose", action="store_true", help="stream solver progress")
    p_solve.add_argument(
        "--fast-turn",
        action="store_true",
        help="minimum-connect-time turnaround, freeing hours for more departures",
    )
    p_solve.add_argument(
        "--enable-fare-elasticity",
        action="store_true",
        help="let the solver choose a fare level per market (stated elasticity assumption)",
    )
    p_solve.add_argument(
        "--enable-connections",
        action="store_true",
        help="allow one-stop itineraries through --hubs to compete for spare seats",
    )
    p_solve.add_argument(
        "--enable-rotations",
        action="store_true",
        help="allow --rotation-squad markets sharing a hub to share one loop's hours",
    )
    p_solve.add_argument(
        "--hubs",
        type=str,
        default=None,
        metavar="IATA,IATA,...",
        help="hubs connections/rotations may route through, e.g. DEL,BOM",
    )
    p_solve.add_argument(
        "--rotation-squad",
        type=str,
        default=None,
        metavar="MID,MID,...",
        help="market_ids eligible to be pooled into a rotation loop",
    )
    p_solve.set_defaults(func=cmd_solve)

    p_feas = sub.add_parser("feasibility", help="optimise, then check the plan can be flown")
    add_common(p_feas)
    p_feas.set_defaults(func=cmd_feasibility)

    p_scen = sub.add_parser("scenarios", help="stress the plan against the standard suite")
    add_common(p_scen)
    p_scen.set_defaults(func=cmd_scenarios)

    p_report = sub.add_parser("report", help="write an HTML decision-support report")
    add_common(p_report)
    p_report.add_argument("--out", type=str, default=None, help="output path")
    p_report.add_argument(
        "--no-scenarios", action="store_true", help="skip the scenario suite"
    )
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser(
        "compare", help="baseline vs. banking/topology/elasticity extensions"
    )
    add_common(p_compare)
    p_compare.add_argument(
        "--hubs",
        type=str,
        default="DEL,BOM",
        metavar="IATA,IATA,...",
        help="hubs to try connections/rotations through (default DEL,BOM)",
    )
    p_compare.add_argument(
        "--rotation-lf-threshold",
        type=float,
        default=0.50,
        help="baseline load factor at or below which a market joins the rotation squad",
    )
    p_compare.add_argument("--no-fast-turn", action="store_true", help="disable the turnaround-time lever")
    p_compare.add_argument("--no-fare-elasticity", action="store_true", help="disable fare elasticity")
    p_compare.add_argument("--no-connections", action="store_true", help="disable hub banking")
    p_compare.add_argument("--no-rotations", action="store_true", help="disable rotation loops")
    p_compare.set_defaults(func=cmd_compare)

    p_expansion = sub.add_parser(
        "expansion-plan", help="multi-year fleet-expansion NPV plan"
    )
    p_expansion.add_argument(
        "--start-year", type=int, default=2027, help="first checkpoint year (default 2027)"
    )
    p_expansion.add_argument(
        "--end-year", type=int, default=2032, help="last checkpoint year (default 2032)"
    )
    p_expansion.set_defaults(func=cmd_expansion_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
