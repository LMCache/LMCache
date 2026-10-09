# SPDX-License-Identifier: Apache-2.0
"""Check that the RBLN runtime works and LMCache selected the rbln backend.

Run after LMCache is installed. Exits non-zero when torch.rbln sees no NPU,
a tensor op on rbln:0 gives a wrong result, or LMCache auto-detected a
different device backend.

Usage: python preflight.py
"""

# Standard
import importlib.metadata

# Third Party
import torch

# First Party
import lmcache


def main() -> None:
    """Print the RBLN stack versions and fail on any broken check.

    Raises:
        SystemExit: If no NPU is visible, the probe op is wrong, or LMCache
            did not select the ``rbln`` backend.
    """
    print("torch", torch.__version__)
    print("torch-rbln", importlib.metadata.version("torch-rbln"))
    print("rebel-compiler", importlib.metadata.version("rebel-compiler"))

    if torch.rbln.device_count() == 0:
        raise SystemExit("torch.rbln sees no NPU")
    probe = torch.arange(6, dtype=torch.float32).to("rbln:0")
    result = (probe + probe).cpu().tolist()
    if result != [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        raise SystemExit(f"wrong result on rbln:0: {result}")

    if lmcache.torch_device_type != "rbln":
        raise SystemExit(
            f"LMCache selected {lmcache.torch_device_type!r}, expected 'rbln'"
        )
    print("rbln preflight ok")


if __name__ == "__main__":
    main()
