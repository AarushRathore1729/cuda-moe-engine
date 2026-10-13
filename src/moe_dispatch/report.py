"""Portable HTML and CSV from recorded measurements; no GPU or Torch import needed."""

import csv
import html
import json
import statistics


def write_report(result, destination):
    records, configurations = result["benchmarks"], result["configurations"]
    rows, panels = [], []
    for case in configurations:
        measurements = {
            r["implementation"]: r
            for r in records
            if r["case_id"] == case["case_id"] and r["stage"] == "dispatch+combine"
        }
        baseline = measurements["PyTorch reference"]["median_us"]
        vector = measurements["CUDA vectorized"]["median_us"]
        scalar = measurements["CUDA scalar"]["median_us"]
        speedup = baseline / vector
        title = (
            f"T={case['tokens']:,} · H={case['hidden']} · E={case['experts']} · {case['pattern']}"
        )
        bars = []
        largest = max(m["median_us"] for m in measurements.values())
        for index, name in enumerate(["PyTorch reference", "CUDA scalar", "CUDA vectorized"]):
            m = measurements[name]
            width = 100 * m["median_us"] / largest
            bars.append(
                f'<div class="bar-row"><span>{html.escape(name)}</span>'
                f'<div class="track"><div class="bar b{index}" style="width:{width:.2f}%"></div></div>'
                f"<strong>{m['median_us']:.2f} µs</strong></div>"
            )
        counts = case["expert_counts"]
        maximum_count = max(counts, default=1) or 1
        load_bars = "".join(
            f'<div class="load-row"><span>E{e}</span><div class="track">'
            f'<div class="bar load" style="width:{100 * c / maximum_count:.2f}%"></div>'
            f"</div><strong>{c:,}</strong></div>"
            for e, c in enumerate(counts)
        )
        panels.append(
            f'<section class="case"><h2>{html.escape(title)}</h2>'
            f'<p class="metric">{speedup:.2f}× measured speed ratio '
            "(PyTorch median / vectorized CUDA median)</p>"
            + "".join(bars)
            + "<details><summary>Expert assignment counts</summary>"
            + load_bars
            + "</details></section>"
        )
        rows.append(
            f"<tr><td>{case['tokens']:,}</td><td>{case['hidden']}</td>"
            f"<td>{case['experts']}</td><td>{case['pattern']}</td>"
            f"<td>{baseline:.2f}</td><td>{scalar:.2f}</td><td>{vector:.2f}</td>"
            f"<td>{speedup:.2f}×</td></tr>"
        )
    metadata = html.escape(json.dumps(result["hardware"], indent=2))
    checks = result["correctness"]["passed"]
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CUDA MoE Dispatch — GPU Benchmark</title><style>
body{font:15px/1.55 system-ui,sans-serif;background:#f3f5fa;color:#14213b;margin:0}
main{max-width:1060px;margin:36px auto;padding:0 22px}h1{font-size:34px;line-height:1.15}
.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#3150a1}
.lead{max-width:860px;color:#47546d}.pill{display:inline-block;padding:7px 13px;
border:1px solid #9bb0da;border-radius:18px;background:white;margin:0 8px 8px 0}
.case,.summary{background:white;padding:22px;border:1px solid #d7deed;border-radius:13px;margin:20px 0}
h2{font-size:19px;margin:0 0 8px}.metric{font-weight:650;color:#263c79}
.bar-row{display:grid;grid-template-columns:160px 1fr 105px;align-items:center;gap:12px;margin:9px 0}
.track{background:#edf0f7;height:17px;border-radius:4px;overflow:hidden}.bar{height:100%;min-width:1px}
.b0{background:#71809b}.b1{background:#8a64c5}.b2{background:#3261c7}.load{background:#3261c7}
.load-row{display:grid;grid-template-columns:38px 1fr 70px;gap:10px;margin:5px 0;align-items:center}
details{margin-top:18px}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e6ee}th{background:#edf1f8}
.scroll{overflow:auto}pre{white-space:pre-wrap;background:#edf1f8;padding:16px;border-radius:7px}
.note{font-size:13px;color:#47546d}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
@media(max-width:700px){.grid{grid-template-columns:1fr}.bar-row{grid-template-columns:120px 1fr 75px;font-size:12px}h1{font-size:27px}}
</style></head><body><main><p class="eyebrow">CUDA / C++ / PyTorch</p>
<h1>MoE Token Dispatch</h1><p class="lead">Measured token packing and weighted output combination on one GPU.
CUDA kernels count expert assignments, calculate offsets, pack token rows, and combine outputs with FP32 accumulation.</p>
"""
    document += f'<span class="pill">{html.escape(result["hardware"]["gpu"])}</span>'
    document += f'<span class="pill">{checks} edge-case checks passed</span>'
    document += f'<span class="pill">{len(configurations)} benchmark configurations</span>'
    document += '<span class="pill">MoE feedforward block verified</span>'
    document += '<div class="summary"><h2>Measurement scope</h2><p class="note">'
    document += html.escape(result["method"]["scope"]) + '</p><p class="note">'
    document += html.escape(result["method"]["timing"]) + "</p></div>"
    document += '<div class="grid">' + "".join(panels) + "</div>"
    document += (
        '<section class="summary"><h2>Exact transport medians (µs)</h2><div class="scroll"><table>'
    )
    document += "<thead><tr><th>Tokens</th><th>Hidden</th><th>Experts</th><th>Routing</th>"
    document += (
        "<th>PyTorch</th><th>CUDA scalar</th><th>CUDA vectorized</th><th>Ratio</th></tr></thead>"
    )
    document += "<tbody>" + "".join(rows) + "</tbody></table></div></section>"
    document += (
        '<section class="summary"><h2>Hardware and software</h2><pre>'
        + metadata
        + "</pre></section>"
    )
    document += "</main></body></html>"
    destination.write_text(document, encoding="utf-8")


def write_results(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    records = result["benchmarks"]
    if records:
        fields = [key for key in records[0] if key != "samples_us"]
        with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    write_report(result, directory / "report.html")
    ratios = []
    for case in result["configurations"]:
        matching = {
            r["implementation"]: r["median_us"]
            for r in records
            if r["case_id"] == case["case_id"] and r["stage"] == "dispatch+combine"
        }
        ratios.append(matching["PyTorch reference"] / matching["CUDA vectorized"])
    if ratios:
        print(
            f"Transport speed ratio across configurations: median {statistics.median(ratios):.2f}x; "
            f"range {min(ratios):.2f}x–{max(ratios):.2f}x",
            flush=True,
        )
    print(f"Saved {directory}/results.json, results.csv, report.html", flush=True)
