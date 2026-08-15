"""Render a solved plan as a self-contained HTML decision-support report.

The output is one file with no external dependencies -- it can be mailed, committed
or opened from a share. It is written for a network planner who wants to know three
things fast: what the plan earns, what would break it, and where the binding
constraint actually is.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from anos.config import load_params
from anos.data.loaders import load_airports, load_markets
from anos.models import FeasibilityReport, NetworkPlan

SEVERITY_ORDER = {"blocker": 0, "warning": 1, "info": 2}


def _usd(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}k"
    return f"${value:,.0f}"


def _esc(text: object) -> str:
    return html.escape(str(text))


STYLE = """
:root{
  --bg:#F6F3EC; --surface:#FFFFFF; --ink:#1B1D22; --muted:#5B5C63;
  --accent:#B9631E; --accent-soft:#F0DCC4; --teal:#22615F; --teal-soft:#DCEAE8;
  --line:#DED7C7; --good:#2F6B45; --warn:#9A6512; --bad:#9B2C2C;
  --good-soft:#DFEDE4; --warn-soft:#F6E6C8; --bad-soft:#F5DEDE;
  --track:#E8E2D6;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#14171C; --surface:#1B1F26; --ink:#E9E6DE; --muted:#9CA0A8;
    --accent:#E2954C; --accent-soft:#3A2C1C; --teal:#59B3AC; --teal-soft:#1B2D2C;
    --line:#2A2F38; --good:#6FBF8E; --warn:#D9A441; --bad:#E07A7A;
    --good-soft:#1C2B22; --warn-soft:#2E2517; --bad-soft:#2E1C1C;
    --track:#262B33;
  }
}
:root[data-theme="dark"]{
  --bg:#14171C; --surface:#1B1F26; --ink:#E9E6DE; --muted:#9CA0A8;
  --accent:#E2954C; --accent-soft:#3A2C1C; --teal:#59B3AC; --teal-soft:#1B2D2C;
  --line:#2A2F38; --good:#6FBF8E; --warn:#D9A441; --bad:#E07A7A;
  --good-soft:#1C2B22; --warn-soft:#2E2517; --bad-soft:#2E1C1C; --track:#262B33;
}
:root[data-theme="light"]{
  --bg:#F6F3EC; --surface:#FFFFFF; --ink:#1B1D22; --muted:#5B5C63;
  --accent:#B9631E; --accent-soft:#F0DCC4; --teal:#22615F; --teal-soft:#DCEAE8;
  --line:#DED7C7; --good:#2F6B45; --warn:#9A6512; --bad:#9B2C2C;
  --good-soft:#DFEDE4; --warn-soft:#F6E6C8; --bad-soft:#F5DEDE; --track:#E8E2D6;
}
*{box-sizing:border-box;}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:Charter,Georgia,Cambria,serif; font-size:15px; line-height:1.55;
}
h1,h2,h3,.label,th,.tile-label,.chip{
  font-family:Futura,"Century Gothic","Avenir Next","Segoe UI",sans-serif;
}
.mono,.num,td.num,th.num,.tile-value{
  font-family:"SF Mono","Cascadia Code",Consolas,monospace;
  font-variant-numeric:tabular-nums;
}
.wrap{max-width:1120px;margin:0 auto;padding:0 1.5rem 4rem;}
header{border-bottom:1px solid var(--line);padding:2.4rem 0 1.4rem;margin-bottom:2rem;}
.eyebrow{
  font-family:Futura,"Century Gothic",sans-serif;font-size:0.7rem;letter-spacing:0.16em;
  text-transform:uppercase;color:var(--accent);font-weight:600;
}
h1{font-size:2rem;margin:0.45rem 0 0.4rem;letter-spacing:-0.01em;text-wrap:balance;}
.sub{color:var(--muted);margin:0;}
section{margin-bottom:2.8rem;scroll-margin-top:1rem;}
h2{
  font-size:1.15rem;margin:0 0 0.9rem;padding-bottom:0.45rem;
  border-bottom:2px solid var(--accent);display:inline-block;
}
p.note{color:var(--muted);max-width:44rem;margin:0 0 1rem;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:0.8rem;}
.tile{
  background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:0.9rem 1rem;
}
.tile-label{
  font-size:0.66rem;letter-spacing:0.09em;text-transform:uppercase;color:var(--muted);
}
.tile-value{font-size:1.5rem;font-weight:600;margin-top:0.25rem;letter-spacing:-0.02em;}
.tile-foot{font-size:0.78rem;color:var(--muted);margin-top:0.1rem;}
.table-wrap{
  overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface);
}
table{border-collapse:collapse;width:100%;min-width:600px;font-size:0.88rem;}
thead th{
  text-align:left;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
  color:var(--muted);padding:0.6rem 0.85rem;border-bottom:2px solid var(--accent);
  white-space:nowrap;background:var(--surface);
}
tbody td{padding:0.5rem 0.85rem;border-bottom:1px solid var(--line);vertical-align:middle;}
tbody tr:last-child td{border-bottom:none;}
td.num,th.num{text-align:right;}
.bar-cell{min-width:150px;}
.bar{height:7px;background:var(--track);border-radius:4px;overflow:hidden;}
.bar span{display:block;height:100%;border-radius:4px;background:var(--teal);}
.bar span.hot{background:var(--warn);}
.bar span.crit{background:var(--bad);}
.chip{
  display:inline-block;font-size:0.64rem;letter-spacing:0.05em;text-transform:uppercase;
  padding:0.12rem 0.45rem;border-radius:3px;font-weight:700;white-space:nowrap;
}
.chip.blocker{background:var(--bad-soft);color:var(--bad);}
.chip.warning{background:var(--warn-soft);color:var(--warn);}
.chip.info{background:var(--teal-soft);color:var(--teal);}
.chip.good{background:var(--good-soft);color:var(--good);}
.pos{color:var(--good);}
.neg{color:var(--bad);}
.finding{
  display:grid;grid-template-columns:5.5rem 1fr;gap:0.9rem;padding:0.7rem 0;
  border-bottom:1px solid var(--line);align-items:start;
}
.finding:last-child{border-bottom:none;}
.finding .subject{font-weight:600;font-size:0.9rem;}
.finding .detail{color:var(--muted);font-size:0.88rem;}
.verdict{
  display:flex;align-items:baseline;gap:0.7rem;padding:0.9rem 1.1rem;border-radius:6px;
  border:1px solid var(--line);margin-bottom:1rem;
}
.verdict.ok{background:var(--good-soft);border-color:var(--good);}
.verdict.bad{background:var(--bad-soft);border-color:var(--bad);}
.verdict b{font-family:Futura,"Century Gothic",sans-serif;font-size:1rem;}
.verdict span{color:var(--muted);font-size:0.88rem;}
.tornado td{padding:0.35rem 0.85rem;}
.tornado .track{position:relative;height:20px;background:var(--track);border-radius:3px;}
.tornado .fill{position:absolute;top:0;height:100%;border-radius:3px;}
.tornado .fill.down{background:var(--bad);}
.tornado .fill.up{background:var(--good);}
.tornado .mid{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--muted);opacity:0.5;}
.callout{
  border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:0.85rem 1.05rem;border-radius:0 4px 4px 0;margin:1rem 0;max-width:46rem;
}
.callout .lbl{
  font-family:Futura,"Century Gothic",sans-serif;font-size:0.66rem;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--accent);font-weight:700;margin-bottom:0.25rem;
}
footer{border-top:1px solid var(--line);padding-top:1.2rem;color:var(--muted);font-size:0.8rem;}
@media (max-width:720px){
  .wrap{padding:0 1rem 3rem;} h1{font-size:1.5rem;}
  .finding{grid-template-columns:1fr;gap:0.2rem;}
}
"""


def _tile(label: str, value: str, foot: str = "") -> str:
    foot_html = f'<div class="tile-foot">{_esc(foot)}</div>' if foot else ""
    return (
        f'<div class="tile"><div class="tile-label">{_esc(label)}</div>'
        f'<div class="tile-value">{_esc(value)}</div>{foot_html}</div>'
    )


def _bar(ratio: float) -> str:
    pct = max(0.0, min(ratio, 1.0)) * 100
    cls = "crit" if ratio > 0.99 else "hot" if ratio > 0.9 else ""
    return f'<div class="bar"><span class="{cls}" style="width:{pct:.1f}%"></span></div>'


def _summary_section(plan: NetworkPlan) -> str:
    params = load_params()
    lf_foot = f"target {params.target_load_factor:.0%}"
    return f"""
<section id="summary">
  <h2>Plan at a glance</h2>
  <div class="tiles">
    {_tile("Contribution / day", _usd(plan.total_contribution_usd))}
    {_tile("Annualised", _usd(plan.total_contribution_usd * 365))}
    {_tile("Revenue / day", _usd(plan.total_revenue_usd))}
    {_tile("Load factor", f"{plan.network_load_factor:.1%}", lf_foot)}
    {_tile("Markets served", f"{plan.markets_served()}", f"of {len(plan.market_plans)} modelled")}
    {_tile("Daily departures", f"{plan.total_departures:,}")}
    {_tile("Daily passengers", f"{plan.total_pax * 2:,.0f}")}
    {_tile("Solve time", f"{plan.solve_seconds:.2f}s", plan.solver_status)}
  </div>
</section>"""


def _fleet_section(plan: NetworkPlan) -> str:
    rows = []
    for code, ratio in sorted(plan.utilisation().items(), key=lambda kv: -kv[1]):
        used = plan.fleet_used.get(code, 0.0)
        if used < 0.5:
            continue
        available = plan.fleet_available.get(code, 0.0)
        rows.append(
            f"<tr><td><b>{_esc(code)}</b></td>"
            f'<td class="num">{used:,.0f}</td>'
            f'<td class="num">{available:,.0f}</td>'
            f'<td class="num">{ratio:.1%}</td>'
            f'<td class="bar-cell">{_bar(ratio)}</td></tr>'
        )
    return f"""
<section id="fleet">
  <h2>Fleet utilisation</h2>
  <p class="note">Aircraft-hours per day consumed by the plan against the hours the fleet
  timeline says are actually flyable on this date &mdash; after retrofit downtime, heavy
  maintenance and an unscheduled-removal allowance.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Type</th><th class="num">Hours used</th><th class="num">Available</th>
    <th class="num">Utilisation</th><th></th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>"""


def _slot_section(feasibility: FeasibilityReport) -> str:
    if not feasibility.slot_usage:
        return ""
    ports = load_airports()
    ranked = sorted(
        feasibility.slot_usage.items(),
        key=lambda kv: -(kv[1] / max(ports[kv[0]].daily_slot_cap, 1)),
    )[:12]
    rows = []
    for iata, used in ranked:
        port = ports[iata]
        ratio = used / max(port.daily_slot_cap, 1)
        tag = '<span class="chip warning">coordinated</span>' if port.slot_constrained else ""
        rows.append(
            f"<tr><td><b>{_esc(iata)}</b> {tag}</td>"
            f"<td>{_esc(port.city)}</td>"
            f'<td class="num">{used}</td>'
            f'<td class="num">{port.daily_slot_cap}</td>'
            f'<td class="num">{ratio:.0%}</td>'
            f'<td class="bar-cell">{_bar(ratio)}</td></tr>'
        )
    return f"""
<section id="slots">
  <h2>Slot usage</h2>
  <p class="note">Daily departure slots consumed at each airport against Air India's
  allocation. Where this saturates it, and not the fleet, is the network's binding
  constraint &mdash; and adding aircraft will not relieve it.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Airport</th><th>City</th><th class="num">Used</th>
    <th class="num">Allocation</th><th class="num">Share</th><th></th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>"""


def _routes_section(plan: NetworkPlan) -> str:
    markets = {m.market_id: m for m in load_markets()}
    ranked = sorted(plan.market_plans, key=lambda p: -p.contribution_usd)
    served = [p for p in ranked if p.total_frequency > 0]

    def table(entries, caption: str) -> str:
        rows = []
        for p in entries:
            mix = ", ".join(f"{_esc(t)}&times;{n}" for t, n in sorted(p.frequencies.items()))
            cls = "pos" if p.contribution_usd >= 0 else "neg"
            market = markets[p.market_id]
            flags = []
            if market.strategic:
                flags.append('<span class="chip info">strategic</span>')
            if market.min_daily_freq > 0:
                flags.append(f'<span class="chip warning">floor {market.min_daily_freq}</span>')
            rows.append(
                f"<tr><td><b>{_esc(p.market_id)}</b> {' '.join(flags)}</td>"
                f'<td class="num">{p.total_frequency}</td>'
                f"<td>{mix}</td>"
                f'<td class="num">{p.load_factor:.0%}</td>'
                f'<td class="num">{p.spilled_demand:,.0f}</td>'
                f'<td class="num {cls}">{_usd(p.contribution_usd)}</td></tr>'
            )
        return f"""
  <h3 style="font-size:0.95rem;margin:1.4rem 0 0.6rem;">{_esc(caption)}</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Market</th><th class="num">Freq</th><th>Fleet mix</th>
    <th class="num">LF</th><th class="num">Spill/day</th>
    <th class="num">Contribution/day</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""

    losers = sorted([p for p in served if p.contribution_usd < 0], key=lambda p: p.contribution_usd)
    unserved = [p for p in ranked if p.total_frequency == 0]

    body = table(served[:15], "Strongest routes by daily contribution")
    if losers:
        body += table(losers[:12], "Routes losing money")
    if unserved:
        names = ", ".join(_esc(p.market_id) for p in unserved)
        body += (
            f'<p class="note" style="margin-top:1.2rem;"><b>{len(unserved)} market(s) '
            f"left unserved:</b> {names}</p>"
        )

    return f'<section id="routes"><h2>Route economics</h2>{body}</section>'


def _feasibility_section(report: FeasibilityReport) -> str:
    verdict_cls = "ok" if report.is_feasible else "bad"
    verdict_txt = "Operable" if report.is_feasible else "Not operable"
    notes = len(report.findings) - len(report.blockers) - len(report.warnings)
    detail = f"{len(report.blockers)} blocker(s), {len(report.warnings)} warning(s), {notes} note(s)"

    findings = sorted(
        report.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.category, f.subject)
    )
    items = "".join(
        f'<div class="finding"><div><span class="chip {f.severity}">{_esc(f.severity)}</span></div>'
        f'<div><div class="subject">{_esc(f.subject)} '
        f'<span style="color:var(--muted);font-weight:400;">&middot; {_esc(f.category)}</span></div>'
        f'<div class="detail">{_esc(f.detail)}</div></div></div>'
        for f in findings
    )
    if not items:
        items = '<p class="note">No findings. The plan is operable as drawn.</p>'

    return f"""
<section id="feasibility">
  <h2>Operational feasibility</h2>
  <p class="note">Whether real airframes could fly this plan: utilisation headroom,
  maintenance load, slot allocations and fleet robustness. A blocker means the plan
  must go back to the optimiser.</p>
  <div class="verdict {verdict_cls}"><b>{verdict_txt}</b><span>{_esc(detail)}</span></div>
  {items}
</section>"""


def _scenario_section(results: list) -> str:
    if not results:
        return ""

    peak = max((abs(r.contribution_delta_usd) for r in results), default=1.0) or 1.0
    rows = []
    for r in sorted(results, key=lambda r: r.contribution_delta_usd):
        delta = r.contribution_delta_usd
        width = abs(delta) / peak * 50.0
        left = 50.0 - width if delta < 0 else 50.0
        cls = "down" if delta < 0 else "up"
        sign_cls = "neg" if delta < 0 else "pos"
        flag = (
            '<span class="chip blocker">floors unmet</span>'
            if r.service_floors_unmeetable
            else ""
        )
        rows.append(
            f"<tr><td><b>{_esc(r.scenario.name)}</b> {flag}<br>"
            f'<span style="color:var(--muted);font-size:0.82rem;">'
            f"{_esc(r.scenario.description)}</span></td>"
            f'<td class="bar-cell" style="min-width:220px;">'
            f'<div class="track"><div class="mid" style="left:50%"></div>'
            f'<div class="fill {cls}" style="left:{left:.1f}%;width:{width:.1f}%"></div></div></td>'
            f'<td class="num {sign_cls}">{_usd(delta)}</td>'
            f'<td class="num {sign_cls}">{r.contribution_delta_pct:+.1f}%</td></tr>'
        )

    breached = [r for r in results if r.service_floors_unmeetable]
    warning = ""
    if breached:
        lines = "".join(
            f"<li><b>{_esc(r.scenario.name)}</b> &mdash; {len(r.unmet_floors)} commitment(s) "
            f"unmet: {_esc(', '.join(r.unmet_floors[:10]))}"
            f"{f' (+{len(r.unmet_floors) - 10} more)' if len(r.unmet_floors) > 10 else ''}</li>"
            for r in breached
        )
        warning = f"""
  <div class="callout">
    <div class="lbl">Service commitments at risk</div>
    These scenarios cannot honour the network's minimum service floors. The figures
    shown are the best network achievable with those floors released &mdash; the
    commercial impact is modest, but the commitments below would have to be renegotiated.
    <ul style="margin:0.5rem 0 0;padding-left:1.1rem;">{lines}</ul>
  </div>"""

    return f"""
<section id="scenarios">
  <h2>Scenario stress test</h2>
  <p class="note">Change in daily contribution when an assumption moves. A plan that is
  optimal only under its own assumptions is not a plan &mdash; this is where its
  robustness gets measured.</p>
  <div class="table-wrap"><table class="tornado">
    <thead><tr><th>Scenario</th><th>Impact</th><th class="num">Delta/day</th>
    <th class="num">Change</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  {warning}
</section>"""


def render_report(
    plan: NetworkPlan,
    feasibility: FeasibilityReport,
    scenarios: list | None = None,
) -> str:
    """Build the full HTML document as a string."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    month_name = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ][plan.month - 1]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Air India network plan &mdash; {plan.as_of}</title>
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">Network optimisation &middot; decision support</div>
  <h1>Air India network plan &mdash; {month_name} {plan.as_of.year}</h1>
  <p class="sub">Fleet availability as at {plan.as_of} &middot;
  {month_name} seasonality &middot; scenario <b>{_esc(plan.scenario)}</b>
  &middot; generated {generated}</p>
</header>
{_summary_section(plan)}
{_scenario_section(scenarios or [])}
{_fleet_section(plan)}
{_slot_section(feasibility)}
{_feasibility_section(feasibility)}
{_routes_section(plan)}
<footer>
  <p><b>Tier 1 model output.</b> Every input is a public-source proxy: published
  schedules, DGCA traffic statistics, OEM order books and Air India's monthly fleet
  disclosure. Demand, fares and route costs are estimates, not booked data. Delivery
  dates are OEM forecasts that have already slipped on this order book. Treat the
  ranking and the direction of these results as the deliverable; treat the absolute
  dollar figures as indicative until the model is rebuilt on Air India's own PSS,
  revenue-accounting and engineering data.</p>
</footer>
</div>
</body>
</html>"""


def write_report(
    plan: NetworkPlan,
    feasibility: FeasibilityReport,
    scenarios: list | None,
    path: str | Path,
) -> Path:
    """Render and write the report, returning the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(plan, feasibility, scenarios), encoding="utf-8")
    return out
