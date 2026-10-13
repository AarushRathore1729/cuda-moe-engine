"""Display full-block measurements separately from the completed transport benchmark."""

import csv
import html
import json
from pathlib import Path
import statistics

LABELS = {
    "torch": "PyTorch transport + grouped GEMM",
    "cuda": "CUDA transport + grouped GEMM",
    "cuda_fused": "CUDA + fused routing/SwiGLU",
}


def write_block_report(result, destination):
    records = result["benchmarks"]
    panels, ratios = [], []
    for case in result["configurations"]:
        rows = [r for r in records if r["case_id"] == case["case_id"]]
        forward = {r["implementation"]: r for r in rows if r["stage"] == "forward"}
        ratio = forward["torch"]["median_us"] / forward["cuda_fused"]["median_us"]
        ratios.append(ratio)
        largest = max(row["median_us"] for row in forward.values())
        bars = ""
        for index, name in enumerate(LABELS):
            value = forward[name]["median_us"]
            bars += (
                f'<div class="row"><span>{LABELS[name]}</span><div class="track">'
                f'<div class="bar c{index}" style="width:{100 * value / largest:.2f}%"></div></div>'
                f"<b>{value:.2f} µs</b></div>"
            )
        stages = ""
        for stage in ("router", "dispatch", "experts", "combine", "forward", "forward_graph"):
            matching = {r["implementation"]: r for r in rows if r["stage"] == stage}
            if matching:
                stages += (
                    "<tr><td>"
                    + stage
                    + "</td>"
                    + "".join(
                        f"<td>{matching[name]['median_us']:.2f}</td>"
                        if name in matching
                        else "<td>—</td>"
                        for name in LABELS
                    )
                    + "</tr>"
                )
        panels.append(
            f"<section><h2>T={case['tokens']:,} · H={case['hidden']} · "
            f"I={case['intermediate']} · E={case['experts']} · {case['pattern']}</h2>"
            f"<p><b>{ratio:.2f}×</b> measured full-block speed ratio</p>{bars}"
            '<details><summary>Isolated stages and full forward (µs)</summary><div class="scroll">'
            "<table><tr><th>Stage</th><th>PyTorch</th><th>CUDA</th><th>CUDA + fused routing/SwiGLU</th></tr>"
            + stages
            + '</table></div><p class="note">Stages are measured separately; their medians '
            "need not sum to forward latency.</p></details></section>"
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CUDA MoE — Full Forward</title>
<style>body{font:15px/1.55 system-ui;background:#f3f5fa;color:#172743;margin:0}
main{max-width:1100px;margin:35px auto;padding:0 22px}h1{font-size:34px}h2{font-size:19px}
section,.intro{background:white;border:1px solid #d4deee;padding:24px;border-radius:14px;margin:20px 0}
.row{display:grid;grid-template-columns:275px 1fr 95px;gap:12px;align-items:center;margin:12px 0}
.track{height:18px;background:#edf1f8;border-radius:4px;overflow:hidden}.bar{height:100%}
.c0{background:#7686a1}.c1{background:#9571c9}.c2{background:#3566c7}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce3ef}
.note{color:#526179;font-size:13px}pre{white-space:pre-wrap}.scroll{overflow:auto}
@media(max-width:700px){.row{grid-template-columns:145px 1fr 80px;font-size:12px}h1{font-size:27px}section{padding:16px}}
</style><body><main><p class="note">C++ / CUDA / PyTorch / grouped expert matmuls</p>
<h1>MoE Forward Pipeline</h1>"""
    document += f'<div class="intro"><b>{html.escape(result["hardware"]["gpu"])}</b> · BF16 · '
    document += f"{len(result['configurations'])} configurations · backend: {html.escape(result['method']['expert_backend'])}"
    document += f"<p>Median of per-configuration full-forward speed ratios: <b>{statistics.median(ratios):.2f}×</b>.</p>"
    document += '<p class="note">' + html.escape(result["method"]["scope"]) + "</p>"
    document += '<p class="note">' + html.escape(result["method"]["timing"]) + "</p></div>"
    document += "".join(panels)
    document += "<section><h2>Validation and reproducibility</h2><pre>"
    document += html.escape(
        json.dumps(
            {
                "hardware": result["hardware"],
                "validation": result["validation"],
                "source_hashes": result["source_hashes"],
            },
            indent=2,
        )
    )
    document += "</pre></section></main></body></html>"
    Path(destination).write_text(document)
    csv_path = Path(destination).with_suffix(".csv")
    fields = [key for key in records[0] if key != "samples_us"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
