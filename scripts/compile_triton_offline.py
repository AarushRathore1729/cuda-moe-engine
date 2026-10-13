"""Check BF16 grouped-GEMM code generation for SM80 without executing on a GPU."""

import hashlib
import json
from pathlib import Path
import sys
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_dispatch._triton_gemm import _grouped_gemm  # noqa: E402

checks = []
for inner, columns in ((64, 128), (64, 32)):
    constants = dict(
        K=inner,
        N=columns,
        E=8,
        A0=inner,
        A1=1,
        B0=inner * columns,
        B1=1,
        B2=inner,
        C0=columns,
        C1=1,
        WORKERS=108,
        BM=64,
        BN=64,
        BK=32,
    )
    source = ASTSource(
        _grouped_gemm,
        signature={"A": "*bf16", "B": "*bf16", "C": "*bf16", "ENDS": "*i32"},
        constexprs=constants,
    )
    kernel = compile(
        source, target=GPUTarget("cuda", 80, 32), options={"num_warps": 4, "num_stages": 3}
    )
    checks.append(
        {
            "inner": inner,
            "columns": columns,
            "dtype": "bfloat16",
            "ptx_bytes": len(kernel.asm["ptx"].encode()),
            "cubin_bytes": len(kernel.asm["cubin"]),
            "status": "compiled",
        }
    )
print(
    json.dumps(
        {
            "triton": triton.__version__,
            "target": "NVIDIA SM80",
            "source_sha256": hashlib.sha256(
                (ROOT / "src/moe_dispatch/_triton_gemm.py").read_bytes()
            ).hexdigest(),
            "checks": checks,
            "limit": "Offline compiler check only; no GPU execution or numerical result.",
        },
        indent=2,
    )
)
