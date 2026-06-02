"""Print basic environment information."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    try:
        import torch

        print(f"torch={getattr(torch, '__version__', 'unknown')}")
        cuda_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False
        print(f"cuda_available={cuda_available}")
    except ImportError:
        print("torch=not-installed")
    except Exception as exc:
        print(f"torch=unavailable ({exc})")


if __name__ == "__main__":
    main()
