"""A complete single-GPU MoE feedforward block with matched implementation variants."""

from dataclasses import dataclass
import math
import torch
from .experts import grouped_linear, resolve_backend, swiglu_reference
from .reference import TorchWorkspace
from .workspace import CudaWorkspace
from ._extension import get_extension


@dataclass(frozen=True)
class MoEConfig:
    hidden: int = 512
    intermediate: int = 1024
    experts: int = 8
    topk: int = 2
    alignment: int = 8

    def __post_init__(self):
        if self.hidden < 8 or self.intermediate < 8:
            raise ValueError("hidden/intermediate must be at least 8")
        if self.hidden % 8 or self.intermediate % 8:
            raise ValueError("grouped GEMM dimensions must be multiples of 8")
        if not 1 <= self.experts <= 64 or self.topk not in (1, 2) or self.topk > self.experts:
            raise ValueError("invalid expert/topk configuration")
        if not 8 <= self.alignment <= 128 or self.alignment % 8:
            raise ValueError("full-block grouped GEMM alignment must be a multiple of 8 up to 128")


@dataclass
class MoEWeights:
    router: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor


def make_weights(config, device, dtype, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    router = torch.randn(
        config.experts, config.hidden, device=device, generator=generator
    ) / math.sqrt(config.hidden)
    gate_up = (
        torch.randn(
            config.experts,
            2 * config.intermediate,
            config.hidden,
            device=device,
            generator=generator,
        )
        / math.sqrt(config.hidden)
    ).to(dtype)
    down = (
        torch.randn(
            config.experts, config.hidden, config.intermediate, device=device, generator=generator
        )
        / math.sqrt(config.intermediate)
    ).to(dtype)
    return MoEWeights(router, gate_up, down)


@dataclass
class LayerWorkspace:
    transport: object
    projection: torch.Tensor
    activated: torch.Tensor
    expert_output: torch.Tensor


class MoEBlock(torch.nn.Module):
    """Inference-only top-k MoE with router, dispatch, two GEMMs, SwiGLU, and combine.

    `torch` and `cuda` variants use FP32 SwiGLU math via PyTorch operations;
    `cuda_fused` fuses bias/top-k/softmax and uses one CUDA activation kernel.
    GEMM backend and weights are shared
    across comparisons. Output is a reusable view: clone it before the next call.
    """

    def __init__(
        self,
        config=None,
        *,
        device="cuda",
        dtype=torch.bfloat16,
        implementation="cuda_fused",
        expert_backend="auto",
        seed=42,
        weights=None,
    ):
        super().__init__()
        self.config = config or MoEConfig()
        if implementation not in ("torch", "cuda", "cuda_fused"):
            raise ValueError("implementation must be torch, cuda, or cuda_fused")
        device = torch.device(device)
        if implementation != "torch" and device.type != "cuda":
            raise RuntimeError("CUDA implementations require CUDA")
        if device.type == "cuda" and dtype != torch.bfloat16:
            raise ValueError("full CUDA block uses BF16 grouped GEMMs")
        self.implementation = implementation
        self.expert_backend = resolve_backend(device, expert_backend)
        weights = weights or make_weights(self.config, device, dtype, seed)
        expected_gate = (self.config.experts, 2 * self.config.intermediate, self.config.hidden)
        expected_down = (self.config.experts, self.config.hidden, self.config.intermediate)
        if weights.gate_up.shape != expected_gate or weights.down.shape != expected_down:
            raise ValueError("expert weights do not match the block configuration")
        if weights.router.shape != (self.config.experts, self.config.hidden):
            raise ValueError("router weights do not match the block configuration")
        self.register_buffer("router_weight", weights.router)
        self.register_buffer("gate_up_weight", weights.gate_up)
        self.register_buffer("down_weight", weights.down)
        self.register_buffer("routing_bias", torch.zeros(self.config.experts, device=device))
        self._workspaces = {}
        self._route_buffers = {}

    def set_routing_pattern(self, pattern):
        if pattern not in ("balanced", "skewed"):
            raise ValueError("routing pattern must be balanced or skewed")
        self.routing_bias.zero_()
        if pattern == "skewed":
            self.routing_bias[: self.config.topk] = 5.0

    def route(self, x):
        logits = torch.nn.functional.linear(x.float(), self.router_weight)
        if self.implementation == "cuda_fused":
            key = (x.shape[0], x.device)
            if key not in self._route_buffers:
                shape = (x.shape[0], self.config.topk)
                self._route_buffers[key] = (
                    torch.empty(shape, device=x.device, dtype=torch.int32),
                    torch.empty(shape, device=x.device, dtype=torch.int64),
                    torch.empty(shape, device=x.device, dtype=torch.float32),
                )
            ids32, ids64, weights = self._route_buffers[key]
            get_extension().topk_route_out(logits, self.routing_bias, ids32, ids64, weights)
            return ids32, ids64, weights
        logits = logits + self.routing_bias
        selected, ids64 = torch.topk(logits, self.config.topk, dim=-1)
        weights = torch.softmax(selected, dim=-1).contiguous()
        return ids64.to(torch.int32).contiguous(), ids64.contiguous(), weights

    def workspace(self, x):
        key = (x.shape[0], x.dtype, x.device)
        if key not in self._workspaces:
            cfg = self.config
            common = (x.shape[0], cfg.hidden, cfg.experts, cfg.topk, x.dtype)
            if self.implementation == "torch":
                transport = TorchWorkspace(*common, device=x.device, alignment=cfg.alignment)
            else:
                transport = CudaWorkspace(*common, device=x.device, alignment=cfg.alignment)
            rows = transport.capacity_rows
            self._workspaces[key] = LayerWorkspace(
                transport,
                torch.empty((rows, 2 * cfg.intermediate), device=x.device, dtype=x.dtype),
                torch.empty((rows, cfg.intermediate), device=x.device, dtype=x.dtype),
                torch.empty((rows, cfg.hidden), device=x.device, dtype=x.dtype),
            )
        return self._workspaces[key]

    def experts_forward(self, state):
        transport = state.transport
        ends = transport.offsets[1:]
        grouped_linear(
            transport.packed,
            self.gate_up_weight,
            ends,
            backend=self.expert_backend,
            output=state.projection,
        )
        if self.implementation == "cuda_fused":
            transport.extension.swiglu_out(
                state.projection, transport.offsets[-1:], state.activated
            )
        else:
            state.activated.copy_(swiglu_reference(state.projection))
        grouped_linear(
            state.activated,
            self.down_weight,
            ends,
            backend=self.expert_backend,
            output=state.expert_output,
        )
        return state.expert_output

    @torch.inference_mode()
    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.config.hidden or not x.is_contiguous():
            raise ValueError("expected contiguous x[T,hidden]")
        if x.dtype != self.gate_up_weight.dtype or x.device != self.gate_up_weight.device:
            raise ValueError("input dtype/device must match expert weights")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        state = self.workspace(x)
        ids32, ids64, weights = self.route(x)
        state.transport.dispatch(x, ids32, ids64)
        output = self.experts_forward(state)
        state.transport.combine(output, weights)
        return state.transport.output


@torch.inference_mode()
def token_major_oracle(block, x):
    """Independent dense expert evaluation; correctness oracle, excluded from timings."""
    # Keep router selection independent of the custom CUDA routing kernel too.
    logits = torch.nn.functional.linear(x.float(), block.router_weight) + block.routing_bias
    selected, ids64 = torch.topk(logits, block.config.topk, dim=-1)
    weights = torch.softmax(selected, dim=-1)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(block.config.experts):
        projection = torch.nn.functional.linear(x, block.gate_up_weight[expert])
        activation = swiglu_reference(projection)
        output = torch.nn.functional.linear(activation, block.down_weight[expert])
        probability = (weights * (ids64 == expert)).sum(dim=1)
        expected += output.float() * probability[:, None]
    return expected.to(x.dtype)
