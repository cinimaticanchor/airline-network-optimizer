# Air India network optimiser (`anos`)

Decision support for Air India network planning: given the fleet the airline will
actually have on a date, decide **which markets to fly, how often, and with what**, to
maximise network contribution — then check the answer could survive contact with an
operations department.

This is the Phase 1–3 implementation of the planning brief: schedule design → fleet
assignment → tail-assignment feasibility → scenario engine. Crew rostering and
revenue management are deliberately out of scope; Air India already licenses systems
for both, and this model is designed to hand recommendations to them rather than
replace them.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

```bash
.venv/bin/anos solve --date 2027-03-01
```

In VS Code: open the folder, pick `.venv/bin/python` as the interpreter, then use
**Run and Debug** (7 preconfigured launch targets) or **Terminal → Run Task**
(solve, feasibility, scenarios, report, tests, lint).

## Commands

| Command | What it does |
|---|---|
| `anos validate` | Cross-check the reference data for internal consistency |
| `anos fleet --date D` | Fleet availability on a date, after retrofit and maintenance downtime |
| `anos solve --date D` | Optimise the network; print route economics and fleet utilisation |
| `anos feasibility --date D` | Optimise, then check the plan can actually be flown |
| `anos scenarios --date D` | Stress the plan against the standard scenario suite |
| `anos report --date D` | Write a self-contained HTML report to `output/` |

Useful flags: `--probabilistic` risk-adjusts delivery dates by programme confidence;
`--month N` applies a different month's seasonality to the same fleet;
`--no-min-service` releases minimum service floors to expose their cost.

---

## How the model works

### 1. Fleet availability is a curve, not a number

Air India has roughly 190 aircraft in service against an order book of about 560, while
simultaneously running a cabin-retrofit programme that takes airframes out of service.
Treating "fleet size" as one number over-promises capacity in exactly the years the
airline is growing fastest.

`anos.data.fleet_timeline` answers: *on date D, how many type-T airframes can the plan
count on?*

```
nominal   = in-service at anchor + deliveries to date - retirements to date
available = nominal - retrofit downtime - heavy maintenance - unscheduled removals
```

Delivery dates are OEM **forecasts**, and both OEMs have already slipped dates on this
order book. `--probabilistic` blends each programme's nominal delivery curve with a
slipped one, weighted by a per-programme confidence in `data/fleet.yaml`. The 777X
sits at 0.45; the A320neo at 0.90.

### 2. Demand responds to frequency

Frequency is a demand lever, not just a cost lever: a twice-daily market captures more
than twice what a daily one does, because time-sensitive travellers pick the airline
that can get them home. The effect saturates, is stronger in business-heavy markets,
and is damped where competitors will match the capacity.

### 3. Fleet assignment is a MIP

```
maximise  Σ ( fare × passengers − passenger cost − cost of trips flown )

subject to  fleet capacity in aircraft-hours, by type
            range and airport compatibility
            passengers ≤ demand, and passengers ≤ seats offered
            minimum service floors
            per-airport departure slot allocations
```

Solved with OR-Tools CP-SAT. **Why it stays linear:** demand depends on frequency
non-linearly, which would normally make the objective non-convex. Because daily
frequency is a small integer, the model enumerates every feasible frequency level per
market as a one-hot set of booleans and reads demand straight off the curve — exact,
no linearisation error, and still a MIP. The 101-market network solves in under a
second.

### 4. Feasibility is a separate gate

The optimiser works in aggregate aircraft-hours, which is right for choosing types and
frequencies but will happily produce a plan no sequence of real tails can fly.
`anos.optimize.tail_feasibility` checks utilisation headroom, A-check load against
overnight ground time, slot allocations, fleet fragility, payload-range limits and
slot opportunity cost. Findings are graded **blocker / warning / info**.

### 5. Scenarios, because the assumptions are wrong

Each scenario mutates a *copy* of the inputs and re-solves. A scenario that cannot
meet the network's minimum service floors is not an error — it is the most important
result the suite produces, so those are re-solved with floors released and reported as
commitments at risk.

---

## What the model currently says

Running `anos feasibility --date 2027-03-01` on the shipped Tier-1 data produces three
findings worth the build on their own:

**The binding constraint is slots, not aircraft.** Delhi and Mumbai both run at 100% of
Air India's departure allocation while the A320neo fleet sits at 67% utilisation. On
this data, buying more narrowbodies does not relieve the network — acquiring slots
does. Domestic trunk routes spill 30–60% of their demand as a direct result.

**Minimum service floors are expensive, and the cost is invisible in the P&L.** At
Delhi the weakest slot goes to a route losing $17.6k/day under a service floor, while
one more Delhi–Bengaluru frequency would earn $3.6k/day — a swap worth about $7.7M a
year. Releasing all floors is worth roughly $73M a year, at the cost of 26 markets.

**Delivery risk shows up as broken commitments, not lost margin.** A 12-month
across-the-board delivery slip costs only ~3.5% of contribution, because the network
re-optimises around it. What it actually breaks is 29 minimum-service commitments.
Fuel (−17% at +25%) and demand (−21% at −12%) dominate the margin risk.

---

## Data

Everything in `data/` is a **Tier-1 public-source proxy**, not a system of record:

| File | Contents | Source |
|---|---|---|
| `fleet.yaml` | In-service fleet, order book, retirements, retrofits | Air India monthly fleet disclosure, OEM order books |
| `aircraft_types.yaml` | Seats, range, burn, turn times, maintenance intervals | Manufacturer specs, published maintenance benchmarks |
| `airports.csv` | 60 airports with coordinates, slot caps, curfews | Published airport and slot-coordination data |
| `markets.csv` | 101 O&D markets with demand, fare and competition estimates | DGCA traffic statistics, published schedules, fare sampling |
| `cost_params.yaml` | Fuel, charges, demand behaviour, solver settings | ATF notifications, published tariff schedules |

**Treat the ranking and the direction of results as the deliverable. Treat the absolute
dollar figures as indicative** until the model is rebuilt on Air India's own PSS/GDS
bookings, revenue accounting and engineering data.

Before using any absolute number, re-pull Air India's own monthly fleet PDF — the
figures here move every month.

---

## Known limitations

These are real and worth understanding before anyone quotes a number from this model.

**Point-to-point demand only.** The model has no connecting traffic. Air India's Delhi
hub exists largely to feed long-haul from regional cities, so thin routes like
Delhi–Madurai look loss-making here when their real value is the long-haul itineraries
they feed. This is the single largest gap, and closing it requires itinerary-level
O&D data. Until then, expect the model to under-value regional feeders.

**No payload-range penalty.** Sectors near an aircraft's range limit are flown with
blocked seats or forgone cargo. The model prices them at full seat count, overstating
ultra-long-haul economics. The feasibility check flags affected sectors under
`payload-range` rather than leaving it implicit — on the shipped data this catches the
optimiser putting an A321neo on Delhi–Heathrow at 99% of usable range.

**No crew feasibility.** DGCA flight-duty-time limits can make a schedule unflyable
even when the airframe is free. Crew is an interface, not a constraint, in this version.

**Aggregate hours, not rotations.** Fleet capacity is modelled as aircraft-hours by
type. Real tail rotations must also close geographically — an aircraft has to end the
day somewhere it can start tomorrow. The feasibility gate approximates this; it does
not build rotations.

**One representative day per month.** Day-of-week variation is not modelled, and
weekend/weekday demand differs materially in business markets.

**Slot caps are Air India's allocation, not airport capacity.** The figures in
`airports.csv` are estimates of Air India's own slot holdings and drive the model's
headline finding, so they deserve verification first.

---

## Layout

```
data/                         Tier-1 reference data (YAML + CSV)
src/anos/
  models.py                   Domain objects; great-circle and block-time maths
  config.py                   Paths and typed parameter access
  data/loaders.py             Loading and cross-validation
  data/fleet_timeline.py      Fleet availability over time
  forecast/demand.py          Seasonality and schedule-quality curves
  costs/economics.py          Per-round-trip unit economics and eligibility
  optimize/fleet_assignment.py  The CP-SAT model
  optimize/tail_feasibility.py  Operational feasibility gate
  scenarios/engine.py         What-if engine
  report/html_report.py       Self-contained HTML report
  cli.py                      Command line interface
tests/                        63 tests
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite checks the things that would otherwise fail silently: that the optimiser
never exceeds fleet capacity or slot allocations, never assigns an ineligible
aircraft, never carries more passengers than seats or demand; that relaxing a
constraint cannot worsen the objective and more aircraft cannot make the network
worse (mis-specification canaries); that scenarios never mutate shared state; and
that the report escapes untrusted text and fetches nothing externally.

## Next

The roadmap's Phase 4 is a shadow-mode pilot: run this against real data alongside
live planning, change nothing, and log the gap between recommendation and reality
weekly. Nothing here should touch a published schedule until that gap is understood.
