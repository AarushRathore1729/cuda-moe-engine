"""Command-line entrypoint. Recorded reports work without a GPU or importing PyTorch."""

import argparse
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog="moe-dispatch", description="CUDA MoE transport experiment")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check-env", help="check GPU and CUDA build prerequisites")
    commands.add_parser("validate", help="run CUDA/reference edge-case checks")
    commands.add_parser("demo", help="verify a real MoE block")
    benchmark = commands.add_parser("benchmark", help="validate and record token-transport timings")
    benchmark.add_argument("--suite", choices=["smoke", "quick", "full"], default="quick")
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--repeats", type=int, default=40)
    benchmark.add_argument("--output", type=Path, default=Path("runs/latest"))
    for command in ["report", "summarize"]:
        sub = commands.add_parser(command, help="inspect saved measurements without a GPU")
        sub.add_argument("--input", type=Path, required=True)
        if command == "report":
            sub.add_argument("--output", type=Path, default=Path("runs/report.html"))
    profile = commands.add_parser("profile", help="export a warmed CPU/CUDA Chrome trace")
    profile.add_argument("--output", type=Path, default=Path("runs/transport_trace.json"))
    block = commands.add_parser("block-benchmark", help="measure the complete MoE forward pipeline")
    block.add_argument("--suite", choices=["smoke", "quick", "model"], default="quick")
    block.add_argument("--expert-backend", choices=["auto", "torch", "triton"], default="auto")
    block.add_argument("--warmup", type=int, default=10)
    block.add_argument("--repeats", type=int, default=40)
    block.add_argument("--output", type=Path, default=Path("runs/block"))
    block.add_argument(
        "--cuda-graphs", action="store_true", help="also validate and time graph replay"
    )
    block_profile = commands.add_parser(
        "profile-block", help="trace router, dispatch, experts, combine"
    )
    block_profile.add_argument(
        "--expert-backend", choices=["auto", "torch", "triton"], default="auto"
    )
    block_profile.add_argument("--output", type=Path, default=Path("runs/moe_block_trace.json"))
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command in ("report", "summarize"):
            from .evidence import read_results, summarize

            result = read_results(arguments.input)
            if arguments.command == "summarize":
                print(json.dumps(summarize(result), indent=2))
            else:
                from .report import write_report

                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                write_report(result, arguments.output)
                print(arguments.output)
            return 0
        if arguments.command == "check-env":
            from .environment import check_environment

            state = check_environment()
            print(json.dumps(state, indent=2))
            return int(bool(state["problems"]))
        from .environment import check_environment

        state = check_environment()
        if state["problems"]:
            raise RuntimeError("; ".join(state["problems"]))
        from ._extension import get_extension

        print("Loading CUDA extension (cached after the first build)...", flush=True)
        extension = get_extension()
        if arguments.command == "block-benchmark":
            from .block_benchmark import run_block_benchmark

            run_block_benchmark(
                arguments.suite,
                arguments.expert_backend,
                arguments.warmup,
                arguments.repeats,
                arguments.output,
                arguments.cuda_graphs,
            )
        elif arguments.command == "profile-block":
            from .block_benchmark import profile_block

            print(profile_block(arguments.output, arguments.expert_backend))
        elif arguments.command == "validate":
            from .validation import run_tests

            print(json.dumps(run_tests(extension), indent=2))
        elif arguments.command == "demo":
            from .demo import block_demo

            print(json.dumps(block_demo(extension), indent=2))
        elif arguments.command == "profile":
            from .profiling import export_trace

            print(export_trace(arguments.output))
        elif arguments.command == "benchmark":
            if arguments.warmup < 1 or arguments.repeats < 5:
                raise ValueError("use warmup>=1 and repeats>=5")
            from .benchmark import benchmark, hardware_metadata
            from .demo import block_demo
            from .validation import run_tests
            from .report import write_results
            from ._extension import source_hashes
            from .evidence import read_results

            correctness = run_tests(extension)
            demo = block_demo(extension)
            configurations, records = benchmark(
                extension, arguments.suite, arguments.warmup, arguments.repeats
            )
            result = {
                "hardware": hardware_metadata(),
                "correctness": correctness,
                "block_demo": demo,
                "configurations": configurations,
                "benchmarks": records,
                "source_hashes": source_hashes(),
                "package_version": "0.2.0",
                "method": {
                    "suite": arguments.suite,
                    "warmup": arguments.warmup,
                    "repeats": arguments.repeats,
                    "seed": 42,
                    "timing": "CUDA-event GPU stream elapsed time; per-call samples after warmup. "
                    "Both paths reuse workspaces; count reset is timed. Compilation, input generation, "
                    "router, validation, and expert computation are excluded. Small cases include launch gaps.",
                    "scope": "Dispatch/combine transport only; FP16 input, FP32 weighted accumulation. "
                    "A GPU-sorting PyTorch reference is compared with scalar/vectorized CUDA packing "
                    "and fused weighted gather. The separate MoE block validates functionality; "
                    "no model/block speedup is claimed. Persistent workspace bytes are not peak memory. "
                    "No production MoE-library comparison has been run.",
                },
            }
            write_results(result, arguments.output)
            read_results(arguments.output / "results.json")
        return 0
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        print(f"moe-dispatch: {error}", file=sys.stderr)
        return 1
