#!/usr/bin/env python3
"""Safe PyTorch CUDA diagnostic for Windows or Linux training servers."""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is not installed: install the CUDA-enabled torch and torchvision wheels first.")
        raise SystemExit(1)
    available = torch.cuda.is_available()
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={available}")
    if not available:
        print("CUDA is unavailable: install a CUDA-enabled PyTorch wheel matching the NVIDIA driver.")
        raise SystemExit(1)
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"gpu_count={torch.cuda.device_count()}")


if __name__ == "__main__":
    main()
