"""Forward-only MoE transport operators; CUDA is compiled lazily on first GPU use."""

from importlib import import_module

__version__ = "0.2.0"
__all__ = ["dispatch", "combine", "PackedTokens", "CudaWorkspace", "TorchWorkspace", "MoEBlock"]
_EXPORTS = {
    "dispatch": (".ops", "dispatch"),
    "combine": (".ops", "combine"),
    "PackedTokens": (".ops", "PackedTokens"),
    "CudaWorkspace": (".workspace", "CudaWorkspace"),
    "TorchWorkspace": (".reference", "TorchWorkspace"),
    "MoEBlock": (".layer", "MoEBlock"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
