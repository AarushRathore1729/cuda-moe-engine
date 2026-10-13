"""Regenerate README figures and a portable explorer from recorded A100 samples."""

import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "benchmarks/a100-block/block_results.json").read_text())
summary = json.loads((ROOT / "benchmarks/a100-block/summary.json").read_text())
rows = {(r["case_id"], r["implementation"], r["stage"]): r for r in data["benchmarks"]}
labels = [f"T={c['tokens']}  H={c['hidden']}  E={c['experts']}\n{c['pattern']}"
          for c in data["configurations"]]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "cuda-moe-v020"})
fig, (ax, ratio_ax) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.6, 1]})
colors = ["#858b97", "#0072b2", "#d55e00"]
names = ["PyTorch", "CUDA transport", "CUDA + fused routing/SwiGLU"]
for offset, (implementation, name, color) in enumerate(zip(("torch", "cuda", "cuda_fused"), names, colors)):
    values = [rows[c['case_id'], implementation, "forward"]["median_us"] for c in data["configurations"]]
    ax.barh([n + (offset-1)*0.23 for n in range(8)], values, height=.22,
            color=color, label=name, hatch=("", "//", "..")[offset], linewidth=.4, edgecolor="white")
ax.set_yticks(range(8), labels)
ax.invert_yaxis()
ax.set_xlabel("Median eager full-forward latency (µs) · lower is faster")
fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(.5, .89), ncol=3, fontsize=9, frameon=False)
ax.grid(axis="x", alpha=.15)
ax.set_axisbelow(True)
ratios = [rows[c['case_id'], "torch", "forward"]["median_us"] /
          rows[c['case_id'], "cuda_fused", "forward"]["median_us"] for c in data["configurations"]]
ratio_ax.barh(range(8), ratios, color=colors[2], height=.56)
ratio_ax.invert_yaxis()
ratio_ax.set_yticks([])
ratio_ax.axvline(1, color="#444", linewidth=1, linestyle="--")
ratio_ax.set_xlim(0, 4)
ratio_ax.set_xlabel("PyTorch / fused CUDA ratio")
for n, r in enumerate(ratios):
    ratio_ax.text(r+.06, n, f"{r:.2f}×", va="center", fontsize=10)
for a in (ax, ratio_ax):
    a.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle(f"A100 synthetic MoE block · {statistics.median(ratios):.2f}× median forward ratio",
             fontsize=17, fontweight="bold", x=.5, y=.98)
fig.text(.5, .915, "BF16 · top-2 routing · shared weights and Triton expert GEMMs · 10 warmups / 40 samples",
         ha="center", color="#444")
fig.text(.5, .015, "CUDA-event eager stream timings can include CPU launch gaps. No full-model or vendor-library comparison.",
         ha="center", fontsize=9, color="#444")
fig.tight_layout(rect=(0,.04,1,.90))
(ROOT / "assets").mkdir(exist_ok=True)
fig.savefig(ROOT / "assets/a100_forward.png", dpi=180)
fig.savefig(ROOT / "assets/a100_forward.svg", metadata={"Date": None})
plt.close(fig)

template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUDA MoE Engine — Recorded A100 Explorer</title>
<style>
:root{color-scheme:light;--ink:#172743;--muted:#526179;--line:#d4deee;--blue:#0072b2;--orange:#d55e00}
*{box-sizing:border-box}body{margin:0;background:#f3f5fa;color:var(--ink);font:15px/1.55 system-ui}
main{max-width:1200px;margin:auto;padding:32px 24px 60px}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:12px 0}
h2{font-size:21px;margin:0 0 12px}.eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:12px;font-weight:700}
.muted{color:var(--muted)}.hero,.card{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:18px 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#eef3f8;border-radius:10px;padding:16px}.stat b{display:block;font-size:29px}
.pipeline{padding:15px;background:#172743;color:white;border-radius:10px;margin:20px 0;font-family:ui-monospace,monospace;overflow-wrap:anywhere}
.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:end}label{font-size:13px;font-weight:700}select,button{font:inherit;border:1px solid #a4b3c8;border-radius:8px;background:white;padding:9px 12px;display:block;color:var(--ink)}
button{cursor:pointer;background:#edf4fa}button:hover{background:#dcecf7}select:focus-visible,button:focus-visible{outline:3px solid var(--orange)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid .card{margin:0}.bars{margin:18px 0}
.barrow{display:grid;grid-template-columns:130px minmax(60px,1fr) 90px;gap:10px;align-items:center;margin:10px 0;font-size:13px}.track{height:20px;background:#eef1f6;border-radius:3px;overflow:hidden}.bar{height:100%}
.bar0{background:#858b97}.bar1{background:repeating-linear-gradient(135deg,#0072b2 0px,#0072b2 5px,#4294c2 5px,#4294c2 7px)}.bar2{background:#d55e00}
.badge{display:inline-block;padding:4px 10px;border-radius:7px;background:#fbece4;font-weight:700}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}
summary{cursor:pointer;font-weight:600;padding:8px 0}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#eef3f8;padding:16px;border-radius:8px;max-height:480px;overflow:auto}
.note{font-size:13px;color:var(--muted)}.footer{font-size:13px;margin-top:24px}a{color:#0068a0}
@media(max-width:800px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}main{padding:20px 14px}.card,.hero{padding:18px}.barrow{grid-template-columns:110px 1fr 75px;font-size:12px}}
</style></head><body><main>
<div class="eyebrow">C++ · CUDA · PyTorch · Triton</div><h1>CUDA MoE Dispatch Engine</h1>
<p class="muted">Explore recorded A100 measurements for one synthetic, forward-only mixture-of-experts block.</p>
<section class="hero"><div class="stats"><div class="stat"><b>3.17×</b>median eager forward ratio</div><div class="stat"><b>8</b>balanced / skewed cases</div><div class="stat"><b>24 / 24</b>oracle comparisons passed</div><div class="stat"><b>39 / 39</b>recorded GPU tests passed</div></div>
<div class="pipeline">FP32 router → fused bias / top-2 / softmax → aligned dispatch → grouped BF16 gate/up GEMM → fused SwiGLU → grouped BF16 down GEMM → FP32 weighted combine</div>
<p><strong>NVIDIA A100-SXM4-40GB</strong> · PyTorch 2.11.0+cu130 · CUDA 13.0 · top-2 routing</p>
<p class="note">All three variants share weights and the Triton expert backend. Ratios compare the included PyTorch variant with CUDA transport plus fused routing/SwiGLU. CUDA-event eager timings can include CPU launch gaps. This is a synthetic single-block experiment; no full-model or production MoE-library speed comparison is included.</p>
</section>
<section class="card"><h2>Explore the measurements</h2><div class="controls">
<label>Token count<select id="tokens"><option value="all">All</option><option>128</option><option>2048</option></select></label>
<label>Routing load<select id="pattern"><option value="all">All</option><option value="balanced">Balanced</option><option value="skewed">Skewed</option></select></label>
<label>Measured stage<select id="stage"><option value="forward">Complete forward</option><option value="router">Router</option><option value="dispatch">Dispatch</option><option value="experts">Experts + activation</option><option value="combine">Combine</option></select></label>
<label>Statistic<select id="statistic"><option value="median_us">Median</option><option value="p95_us">p95</option></select></label>
<button id="download" type="button">Download raw JSON</button></div>
<p class="note">Bars show microseconds; shorter is faster. Displayed ratios use the selected stage and statistic. Isolated stage medians do not sum to forward latency. The 3.17× headline uses the median of eight full-forward median ratios.</p></section>
<div class="grid" id="cases" aria-live="polite"></div>
<section class="card"><h2>Earlier transport checkpoint</h2><p><strong>6.36× median dispatch-and-combine ratio</strong> across 12 A100 configurations; <strong>72.7% lower explicit persistent transport workspace</strong> versus the included preallocated PyTorch reference. Vector packing reduced dispatch latency by up to 34.9% versus scalar CUDA.</p><p class="note">This separate FP16 transport checkpoint excludes router and expert compute. Its workspace comparison excludes weights, backend temporaries, and peak allocator usage. These figures do not describe the BF16 full-block benchmark above.</p></section>
<section class="card"><h2>Validation and reproduction</h2><p>Recorded A100 run: 81 CPU tests, 39 GPU tests, and 36 standalone transport checks passed. All 22 measured source hashes match the release. Maximum quick-suite absolute error: 0.0078125; relative L2 error: 0.0002143. Profiler trace contains 36 GPU kernel events.</p>
<pre>python -m pip install -e '.[dev,gpu]'
moe-dispatch check-env
python -m pytest -q
moe-dispatch validate
moe-dispatch block-benchmark --suite quick --expert-backend triton --output runs/quick
python scripts/verify_release.py</pre>
<p class="note">CUDA Graph replay and the H=4096 / I=14336 larger preset are validated in the additional experiments above. Their ratios remain separate from the primary eager checkpoint. This explorer is an offline view of recorded results; it does not execute GPU inference in the browser.</p>
<details><summary>Hardware, methodology, and source hashes</summary><pre id="evidence"></pre></details></section>
<div class="footer">MIT licensed · Raw evidence is included in benchmarks/a100-block · Generated from recorded samples</div>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const names=['torch','cuda','cuda_fused'], labels=['PyTorch','CUDA transport','CUDA + fused'];
const controls=['tokens','pattern','stage','statistic'].map(id=>document.getElementById(id));
const row=(cid,impl,stage)=>data.benchmarks.find(r=>r.case_id===cid&&r.implementation===impl&&r.stage===stage);
function render(){const [tokens,pattern,stage,stat]=controls.map(e=>e.value);
document.getElementById('cases').innerHTML=data.configurations.filter(c=>(tokens==='all'||String(c.tokens)===tokens)&&(pattern==='all'||c.pattern===pattern)).map(c=>{
const values=names.map(i=>row(c.case_id,i,stage)[stat]);const largest=Math.max(...values);
const ratio=values[0]/values[2];
const bars=values.map((v,i)=>`<div class="barrow"><span>${labels[i]}</span><div class="track"><div class="bar bar${i}" style="width:${100*v/largest}%"></div></div><strong>${v.toFixed(2)} µs</strong></div>`).join('');
const table=['router','dispatch','experts','combine','forward'].map(s=>`<tr><td>${s}</td>${names.map(i=>`<td>${row(c.case_id,i,s).median_us.toFixed(2)} / ${row(c.case_id,i,s).p95_us.toFixed(2)}</td>`).join('')}</tr>`).join('');
return `<section class="card"><h2>T=${c.tokens.toLocaleString()} · H=${c.hidden} · E=${c.experts}</h2><p class="muted">I=${c.intermediate} · ${c.pattern} · BF16 · top-${c.topk}</p><span class="badge">${ratio.toFixed(2)}× ${stage} ${stat==='median_us'?'median':'p95'} ratio</span><div class="bars">${bars}</div><details><summary>All stages: median / p95 (µs)</summary><div class="scroll"><table><tr><th>Stage</th><th>PyTorch</th><th>CUDA transport</th><th>CUDA + fused</th></tr>${table}</table></div></details><details><summary>Expert token counts</summary><p class="note">${c.expert_counts.map((n,i)=>`E${i}: ${n}`).join(' · ')}</p></details></section>`;}).join('');}
controls.forEach(e=>e.addEventListener('change',render));render();
document.getElementById('evidence').textContent=JSON.stringify({hardware:data.hardware,method:data.method,source_hashes:data.source_hashes},null,2);
document.getElementById('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='a100_block_results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
</script></body></html>'''
extras = "<section class='card'><h2>Additional validated A100 experiments</h2><p>Both optional runs completed using the same 22 measured source files and shared Triton expert backend.</p>"
for key, title, stage in (("graphs", "CUDA Graph replay", "forward_graph"),
                           ("model", "Larger synthetic preset", "forward")):
    path = ROOT / "benchmarks/a100-block" / key / "block_results.json"
    if not path.exists():
        continue
    extra = json.loads(path.read_text())
    stats = summary["optional"][key]
    ratio = stats["median_graph_ratio"] if key == "graphs" else stats["median_forward_ratio"]
    extra_rows = {(r["case_id"], r["implementation"], r["stage"]): r for r in extra["benchmarks"]}
    extras += f"<h3>{title} · {ratio:.2f}× median ratio</h3>"
    extras += "<div class='scroll'><table><tr><th>Shape / routing</th><th>PyTorch (µs)</th><th>CUDA transport (µs)</th><th>CUDA + fused (µs)</th><th>Ratio</th></tr>"
    for case in extra["configurations"]:
        values = [extra_rows[case["case_id"], name, stage]["median_us"]
                  for name in ("torch", "cuda", "cuda_fused")]
        extras += f"<tr><td>T={case['tokens']} H={case['hidden']} I={case['intermediate']} E={case['experts']} {case['pattern']}</td>"
        extras += "".join(f"<td>{value:.2f}</td>" for value in values)
        extras += f"<td>{values[0] / values[2]:.2f}×</td></tr>"
    extras += "</table></div>"
    if key == "graphs":
        extras += "<p class='note'>24/24 graph replay checks passed against eager output. Ratios compare graph replay for both variants; range 1.15–2.53×. The eager rerun in this same experiment measured 3.08× median. This is separate from the original 3.17× eager checkpoint.</p>"
    else:
        extras += "<p class='note'>6/6 independent-oracle comparisons passed. H=4096, I=14336, E=8, top-2. Two cases: T=128 balanced and T=256 skewed. Eager ratios: 1.0595× and 1.0699×. Maximum absolute error 0.015625; relative L2 error 0.0031672. Shape materially affects the measured benefit.</p>"
extras += "</section>"
template = template.replace("<section class=\"card\"><h2>Earlier transport checkpoint</h2>",
                            extras + "<section class=\"card\"><h2>Earlier transport checkpoint</h2>")
(ROOT / "demo").mkdir(exist_ok=True)
(ROOT / "demo/index.html").write_text(template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c")))
print("Generated assets/a100_forward.{png,svg} and demo/index.html")
