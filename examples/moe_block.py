import torch
from moe_dispatch import MoEBlock
from moe_dispatch.layer import MoEConfig


def main():
    block = MoEBlock(MoEConfig(hidden=512, intermediate=1024, experts=8, topk=2))
    x = torch.randn(128, 512, device="cuda").bfloat16()
    with torch.inference_mode():
        y = block(x).clone()
    print("Output:", tuple(y.shape), "dtype:", y.dtype)


if __name__ == "__main__":
    main()
