# SPDX-License-Identifier: Apache-2.0


def get_correct_nixl_device(nixl_device: str, worker_id: int) -> str:
    """
    Get the correct Nixl device based on the given device string.

    Args:
        nixl_device (str): The device string, could be cpu or cuda

    Returns:
        str: The correct device string for Nixl -- with correct
          device id.
    """
    if nixl_device == "cpu":
        return "cpu"
    elif nixl_device.startswith("cuda"):
        return f"cuda:{worker_id}"
    else:
        raise ValueError(f"Invalid Nixl device: {nixl_device}")
