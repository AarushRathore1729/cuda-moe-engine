"""Load packaged CUDA sources with PyTorch's cache; importing the package never compiles."""

from functools import lru_cache
import hashlib
from importlib.resources import files
import os
from pathlib import Path


def source_texts():
    directory = files("moe_dispatch").joinpath("csrc")
    return (
        directory.joinpath("bindings.cpp").read_text(),
        directory.joinpath("dispatch.cu").read_text(),
    )


def source_hashes():
    cpp, cuda = source_texts()
    result = {
        "bindings.cpp": hashlib.sha256(cpp.encode()).hexdigest(),
        "dispatch.cu": hashlib.sha256(cuda.encode()).hexdigest(),
    }
    for source in files("moe_dispatch").iterdir():
        if source.name.endswith(".py") and source.is_file():
            result[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


@lru_cache(maxsize=1)
def get_extension():
    import torch
    import torch.utils.cpp_extension as extension

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA operators require an NVIDIA GPU. Use report/summarize on CPU.")
    if extension.CUDA_HOME is None:
        candidates = [os.environ.get("CUDA_HOME"), "/usr/local/cuda"]
        for directory in candidates:
            if directory and (Path(directory) / "bin/nvcc").is_file():
                extension.CUDA_HOME = directory
                break
    if extension.CUDA_HOME is None:
        raise RuntimeError("nvcc/CUDA toolkit is missing. Run moe-dispatch check-env first.")
    os.environ.setdefault("MAX_JOBS", "2")
    cpp, cuda = source_texts()
    digest = hashlib.sha256((cpp + cuda).encode()).hexdigest()[:10]
    return extension.load_inline(
        name=f"moe_dispatch_{digest}",
        cpp_sources=cpp,
        cuda_sources=cuda,
        functions=[
            "dispatch_out",
            "dispatch_aligned_out",
            "combine_out",
            "swiglu_out",
            "topk_route_out",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=os.environ.get("MOE_DISPATCH_VERBOSE_BUILD") == "1",
    )
