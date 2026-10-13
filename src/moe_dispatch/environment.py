"""Small preflight that reports practical CUDA build prerequisites."""

import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def check_environment():
    checks = {"python": platform.python_version(), "platform": platform.system()}
    problems = []
    if importlib.util.find_spec("torch") is None:
        problems.append("PyTorch is missing; use a prepared PyTorch GPU environment")
        checks["problems"] = problems
        return checks
    import torch

    checks.update(
        torch_version=str(torch.__version__),
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
    )
    if not checks["cuda_available"]:
        problems.append("NVIDIA GPU is not attached or PyTorch cannot access CUDA")
    else:
        checks["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    checks["cxx"] = shutil.which(os.environ.get("CXX", "c++"))
    if not checks["cxx"]:
        problems.append("C++ compiler is missing")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidates = [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"]
        candidates.extend(sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True))
        nvcc = next((str(p) for p in candidates if p.is_file()), None)
    checks["nvcc"] = nvcc
    if not nvcc:
        problems.append("CUDA toolkit/nvcc is missing; a PyTorch wheel alone is insufficient")
    else:
        output = subprocess.check_output([nvcc, "--version"], text=True, timeout=15)
        match = re.search(r"release (\d+)\.(\d+)", output)
        checks["toolkit"] = match.group(0) if match else output.strip()
        if match and torch.version.cuda and match.group(1) != torch.version.cuda.split(".")[0]:
            problems.append("nvcc and PyTorch have incompatible CUDA major versions")
        if "CUDA_HOME" not in os.environ:
            os.environ["CUDA_HOME"] = str(Path(nvcc).parent.parent)
    checks["ninja"] = shutil.which("ninja")
    if not checks["ninja"]:
        problems.append("Ninja is missing; install ninja")
    checks["problems"] = problems
    return checks
