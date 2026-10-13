"""Full forward and stage measurements for matched MoE blocks."""

from dataclasses import asdict
import json
from pathlib import Path
import torch
from .benchmark import hardware_metadata, measure
from .layer import MoEBlock, MoEConfig, make_weights, token_major_oracle
from ._extension import source_hashes


IMPLEMENTATIONS = ("torch", "cuda", "cuda_fused")


def workload_cases(suite):
    if suite == "smoke":
        return [(128, MoEConfig(), pattern) for pattern in ("balanced", "skewed")]
    if suite == "model":
        config = MoEConfig(hidden=4096, intermediate=14336, experts=8, topk=2)
        return [(128, config, "balanced"), (256, config, "skewed")]
    return [
        (tokens, config, pattern)
        for tokens in (128, 2048)
        for config in (MoEConfig(), MoEConfig(1024, 2048, 16, 2))
        for pattern in ("balanced", "skewed")
    ]


@torch.inference_mode()
def check_forward(block, x):
    actual = block(x).clone()
    expected = token_major_oracle(block, x)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    difference = actual.float() - expected.float()
    relative = float(difference.norm() / expected.float().norm().clamp_min(1e-8))
    if relative > 0.02:
        raise AssertionError(f"relative L2 error {relative} exceeds BF16 tolerance .02")
    return {
        "passed": True,
        "max_absolute_error": float(difference.abs().max()),
        "relative_l2_error": relative,
        "rtol": 0.03,
        "atol": 0.03,
        "relative_l2_limit": 0.02,
    }


@torch.inference_mode()
def capture_forward(block, x):
    """Optional replay check. It must match eager output before timings are accepted."""
    expected = block(x).clone()
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            block(x)
    torch.cuda.current_stream(x.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = block(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
    return graph


@torch.inference_mode()
def run_block_benchmark(
    suite="quick",
    expert_backend="auto",
    warmup=10,
    repeats=40,
    output=Path("runs/block"),
    graphs=False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("full-block CUDA comparison requires an NVIDIA GPU")
    if warmup < 1 or repeats < 5:
        raise ValueError("use warmup>=1 and repeats>=5")
    configurations, records, validation = [], [], []
    backend_used = None
    for case_id, (tokens, config, pattern) in enumerate(workload_cases(suite)):
        weights = make_weights(config, "cuda", torch.bfloat16, 42 + case_id)
        generator = torch.Generator(device="cuda").manual_seed(100 + case_id)
        x = torch.randn(tokens, config.hidden, device="cuda", generator=generator).bfloat16()
        blocks = {
            name: MoEBlock(
                config,
                device="cuda",
                implementation=name,
                expert_backend=expert_backend,
                weights=weights,
            )
            for name in IMPLEMENTATIONS
        }
        for block in blocks.values():
            block.set_routing_pattern(pattern)
        base = blocks["torch"]
        backend_used = base.expert_backend
        ids32, ids64, probabilities = base.route(x)
        counts = torch.bincount(ids64.flatten(), minlength=config.experts).cpu().tolist()
        configurations.append(
            {
                "case_id": case_id,
                "tokens": tokens,
                **asdict(config),
                "pattern": pattern,
                "dtype": "bfloat16",
                "expert_backend": backend_used,
                "expert_counts": counts,
            }
        )
        names = list(IMPLEMENTATIONS)
        rotation = case_id % len(names)
        names = names[rotation:] + names[:rotation]
        for name in names:
            block = blocks[name]
            checked = check_forward(block, x)
            validation.append({"case_id": case_id, "implementation": name, **checked})
            state = block.workspace(x)
            state.transport.dispatch(x, ids32, ids64)
            block.experts_forward(state)

            def route():
                block.route(x)

            def dispatch():
                state.transport.dispatch(x, ids32, ids64)

            def experts():
                block.experts_forward(state)

            def combine():
                state.transport.combine(state.expert_output, probabilities)

            def forward():
                block(x)

            workspace_bytes = state.transport.workspace_bytes + sum(
                tensor.numel() * tensor.element_size()
                for tensor in (state.projection, state.activated, state.expert_output)
            )
            workspace_bytes += sum(
                tensor.numel() * tensor.element_size()
                for buffers in block._route_buffers.values()
                for tensor in buffers
            )
            for stage, function in (
                ("router", route),
                ("dispatch", dispatch),
                ("experts", experts),
                ("combine", combine),
                ("forward", forward),
            ):
                records.append(
                    {
                        "case_id": case_id,
                        "implementation": name,
                        "stage": stage,
                        "persistent_workspace_bytes": workspace_bytes,
                        **measure(function, warmup, repeats),
                    }
                )
            if graphs:
                try:
                    graph = capture_forward(block, x)
                    records.append(
                        {
                            "case_id": case_id,
                            "implementation": name,
                            "stage": "forward_graph",
                            "persistent_workspace_bytes": workspace_bytes,
                            **measure(graph.replay, warmup, repeats),
                        }
                    )
                    checked["graph_status"] = "passed"
                except RuntimeError as error:
                    checked["graph_status"] = "unsupported"
                    checked["graph_error"] = str(error)
                validation[-1].update(checked)
        print(
            f"MoE block {case_id + 1}/{len(workload_cases(suite))}: T={tokens}, H={config.hidden}, "
            f"I={config.intermediate}, E={config.experts}, {pattern}, backend={backend_used}",
            flush=True,
        )
    result = {
        "kind": "moe_block_benchmark",
        "package_version": "0.2.0",
        "hardware": hardware_metadata(),
        "source_hashes": source_hashes(),
        "configurations": configurations,
        "validation": validation,
        "benchmarks": records,
        "method": {
            "suite": suite,
            "warmup": warmup,
            "repeats": repeats,
            "expert_backend": backend_used,
            "graphs_requested": graphs,
            "scope": "One synthetic MoE feedforward block: FP32 router/top-k/softmax, "
            "BF16 grouped gate/up GEMM, FP32 SwiGLU math with one BF16 cast, "
            "BF16 grouped down GEMM, and FP32 weighted combine. Shared weights and "
            "GEMM backend across all variants. This is not complete language-model inference.",
            "timing": "CUDA events after warmup; router/expert computation included in forward. "
            "Expert stages include both GEMMs and activation. Per-stage medians are measured "
            "in isolation and need not sum to the full-forward median. Compilation and validation "
            "are excluded; eager samples can include CPU launch gaps. Persistent buffers exclude "
            "shared weights and backend temporary allocations. Optional graph timing is separate.",
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "block_results.json").write_text(json.dumps(result, indent=2))
    from .block_report import write_block_report

    write_block_report(result, output / "block_report.html")
    print(f"Saved {output}/block_results.json and block_report.html", flush=True)
    return result


@torch.inference_mode()
def profile_block(destination, expert_backend="auto"):
    config = MoEConfig()
    block = MoEBlock(config, expert_backend=expert_backend)
    x = torch.randn(2048, config.hidden, device="cuda").bfloat16()
    for _ in range(10):
        block(x)
    torch.cuda.synchronize()
    state = block.workspace(x)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        for _ in range(3):
            with torch.autograd.profiler.record_function("MoE/router"):
                ids32, ids64, weights = block.route(x)
            with torch.autograd.profiler.record_function("MoE/dispatch"):
                state.transport.dispatch(x, ids32, ids64)
            with torch.autograd.profiler.record_function("MoE/grouped_experts"):
                block.experts_forward(state)
            with torch.autograd.profiler.record_function("MoE/combine"):
                state.transport.combine(state.expert_output, weights)
        torch.cuda.synchronize()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(destination))
    return destination
