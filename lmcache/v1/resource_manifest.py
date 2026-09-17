# SPDX-License-Identifier: Apache-2.0
"""Resource Manifest and GPU Device Binding Policies.

Declares available hardware resources (PCIe BAR devices, DRAM) and provides
pluggable policies that assign subsets of those resources to each GPU worker.

Manifest file format::

    # /etc/lmcache/resources.yaml
    resources:
      - name: "bar_0"
        type: "pcie_bar"
        path: "/sys/bus/pci/devices/0000:41:00.0/resource4"
        capacity_gb: 64.0
        bandwidth_gbps: 32.0
      - name: "dram_tier"
        type: "dram"
        capacity_gb: 32.0
    device_binding_policy: "equal_capacity"   # non-MP only
    allocation_policy: "interleaving"

MP mode — a single server process owns the full manifest, so
``device_binding_policy`` is ignored. All resources are visible to the
server and shared across GPU workers::

    lmcache_server --resource-manifest /etc/lmcache/resources.yaml ...

Non-MP mode — each GPU worker applies ``device_binding_policy`` to
select its partition. Inline or as a file reference::

    extra_config:
      resource_manifest:
        resources:
          - name: "bar_0"
            type: "pcie_bar"
            path: "/sys/bus/pci/devices/0000:41:00.0/resource4"
            capacity_gb: 64.0
        device_binding_policy: "equal_capacity"
        allocation_policy: "interleaving"

    extra_config:
      resource_manifest: "/etc/lmcache/resources.yaml"
"""


# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class ResourceDescriptor:
    """One hardware resource declared in the manifest."""

    name: str
    type: str  # "pcie_bar" or "dram"
    capacity_gb: float
    path: str | None = None
    offset: int = 0
    bandwidth_gbps: float | None = None


@dataclass
class ResourceManifest:
    """Full resource manifest: list of resources + policy selections."""

    resources: list[ResourceDescriptor] = field(default_factory=list)
    device_binding_policy: str = "auto"
    allocation_policy: str | None = None


class DeviceBindingPolicy(ABC):
    """Assigns resources to GPU workers.

    Given the full list of available resources and the GPU topology,
    returns the subset of resources assigned to a specific GPU.

    When a device must be shared by multiple GPUs, the returned descriptors
    have adjusted offset and capacity_gb so each GPU gets a non-overlapping
    partition of the physical address space.
    """

    @abstractmethod
    def bind(
        self,
        resources: list[ResourceDescriptor],
        num_gpus: int,
        gpu_id: int,
    ) -> list[ResourceDescriptor]:
        """Return the resources assigned to gpu_id."""
        raise NotImplementedError


class EqualCapacityBinding(DeviceBindingPolicy):
    """Partition resources so each GPU gets exactly equal total capacity.

    Treats the resource list as a linear address space and cuts it into
    num_gpus equal-sized contiguous slices. A single resource may be split
    across a GPU boundary — each GPU gets the intersection of its slice
    with each resource.

    Example: bar_0 (30 GB) + bar_1 (10 GB), 2 GPUs (total 40 GB, 20 each):
      GPU 0: 20 GB of bar_0 (offset 0)
      GPU 1: 10 GB of bar_0 (offset 20 GB) + 10 GB of bar_1 (offset 0)
    """

    def bind(
        self,
        resources: list[ResourceDescriptor],
        num_gpus: int,
        gpu_id: int,
    ) -> list[ResourceDescriptor]:
        if not resources:
            return []

        total_capacity = sum(r.capacity_gb for r in resources)
        per_gpu = total_capacity / num_gpus

        gpu_start = gpu_id * per_gpu
        gpu_end = gpu_start + per_gpu

        result = []
        pos = 0.0

        for r in resources:
            r_end = pos + r.capacity_gb

            slice_start = max(gpu_start, pos)
            slice_end = min(gpu_end, r_end)

            if slice_start < slice_end:
                slice_capacity = slice_end - slice_start
                slice_offset_in_resource = slice_start - pos
                offset_bytes = int(
                    r.offset + slice_offset_in_resource * 1024**3
                )

                result.append(
                    ResourceDescriptor(
                        name=r.name,
                        type=r.type,
                        capacity_gb=slice_capacity,
                        path=r.path,
                        offset=offset_bytes,
                        bandwidth_gbps=r.bandwidth_gbps,
                    )
                )

            pos = r_end

        return result


class InterleavedBinding(DeviceBindingPolicy):
    """Each GPU gets one equal slice from every resource, interleaved.

    Every resource is divided into num_gpus equal slices. GPU i gets
    slice i of every resource. This maximizes bandwidth by spreading
    each GPU's accesses across all physical devices.

    Example: bar_0 (64 GB) + bar_1 (32 GB), 4 GPUs:
      GPU 0: bar_0[0:16 GB] + bar_1[0:8 GB]
      GPU 1: bar_0[16:32 GB] + bar_1[8:16 GB]
      GPU 2: bar_0[32:48 GB] + bar_1[16:24 GB]
      GPU 3: bar_0[48:64 GB] + bar_1[24:32 GB]
    """

    def bind(
        self,
        resources: list[ResourceDescriptor],
        num_gpus: int,
        gpu_id: int,
    ) -> list[ResourceDescriptor]:
        if not resources:
            return []

        result = []
        for r in resources:
            slice_capacity = r.capacity_gb / num_gpus
            slice_offset_bytes = int(
                r.offset + gpu_id * slice_capacity * 1024**3
            )
            result.append(
                ResourceDescriptor(
                    name=r.name,
                    type=r.type,
                    capacity_gb=slice_capacity,
                    path=r.path,
                    offset=slice_offset_bytes,
                    bandwidth_gbps=r.bandwidth_gbps,
                )
            )

        return result

class BandwidthWeightedBinding(DeviceBindingPolicy):
    """Assign resources to GPUs prioritizing total bandwidth balance.

    Like EqualCapacityBinding but optimizes for bandwidth instead of capacity.
    Falls back to capacity-based if bandwidth is not declared.

    If there are fewer resources than GPUs, resources are partitioned by
    offset (same as EqualCapacityBinding).
    """

    @staticmethod
    def _partition_resource(
        resource: ResourceDescriptor,
        num_shares: int,
        share_index: int,
    ) -> ResourceDescriptor:
        """Partition a resource into non-overlapping slices by offset.

        Each share gets ``capacity_gb / num_shares`` at a distinct offset
        within the device.

        Args:
            resource: The resource to partition.
            num_shares: Total number of slices.
            share_index: Which slice to return (0-based).

        Returns:
            A new ResourceDescriptor for the requested slice.
        """
        share_capacity = resource.capacity_gb / num_shares
        share_offset_bytes = int(
            resource.offset + share_index * share_capacity * 1024**3
        )
        return ResourceDescriptor(
            name=f"{resource.name}_s{share_index}",
            type=resource.type,
            capacity_gb=share_capacity,
            path=resource.path,
            offset=share_offset_bytes,
            bandwidth_gbps=resource.bandwidth_gbps,
        )

    def bind(
        self,
        resources: list[ResourceDescriptor],
        num_gpus: int,
        gpu_id: int,
    ) -> list[ResourceDescriptor]:
        if not resources:
            return []

        def bw_key(r: ResourceDescriptor) -> float:
            return r.bandwidth_gbps if r.bandwidth_gbps is not None else r.capacity_gb

        if len(resources) >= num_gpus:
            sorted_res = sorted(resources, key=bw_key, reverse=True)
            bins: list[list[ResourceDescriptor]] = [[] for _ in range(num_gpus)]
            bin_totals = [0.0] * num_gpus
            for r in sorted_res:
                target = min(range(num_gpus), key=lambda i: bin_totals[i])
                bins[target].append(r)
                bin_totals[target] += bw_key(r)
            return bins[gpu_id]

        # Fewer resources than GPUs: partition by offset (same as EqualCapacity)
        gpus_per_resource = num_gpus // len(resources)
        remainder = num_gpus % len(resources)

        sorted_res = sorted(resources, key=bw_key, reverse=True)
        gpu_offset = 0
        for i, r in enumerate(sorted_res):
            shares = gpus_per_resource + (1 if i < remainder else 0)
            if gpu_offset <= gpu_id < gpu_offset + shares:
                share_index = gpu_id - gpu_offset
                return [self._partition_resource(r, shares, share_index)]
            gpu_offset += shares
        return []


# ─── Registry ──────────────────────────────────────────────────────────────────

_BINDING_POLICIES: dict[str, type[DeviceBindingPolicy]] = {
    "equal_capacity": EqualCapacityBinding,
    "interleaved": InterleavedBinding,
    "bandwidth": BandwidthWeightedBinding,
}


def resolve_device_binding(name: str) -> DeviceBindingPolicy:
    """Instantiate a DeviceBindingPolicy by name."""
    cls = _BINDING_POLICIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown device_binding_policy: {name!r}. "
            f"Available: {list(_BINDING_POLICIES.keys())}"
        )
    return cls()


def register_binding_policy(
    name: str, cls: type[DeviceBindingPolicy]
) -> None:
    """Register a custom DeviceBindingPolicy for use in manifests."""
    _BINDING_POLICIES[name] = cls


# ─── Parsing ───────────────────────────────────────────────────────────────────


def _parse_resource(raw: dict) -> ResourceDescriptor:
    """Parse a single resource dict into a ResourceDescriptor."""
    if "offset_gb" in raw:
        offset = int(float(raw["offset_gb"]) * 1024**3)
    else:
        offset = int(raw.get("offset", 0))
    return ResourceDescriptor(
        name=raw["name"],
        type=raw["type"],
        capacity_gb=float(raw["capacity_gb"]),
        path=raw.get("path"),
        offset=offset,
        bandwidth_gbps=(
            float(raw["bandwidth_gbps"]) if "bandwidth_gbps" in raw else None
        ),
    )


def _parse_manifest(data: dict) -> ResourceManifest:
    """Parse a manifest dict into a ResourceManifest."""
    resources = [_parse_resource(r) for r in data.get("resources", [])]
    return ResourceManifest(
        resources=resources,
        device_binding_policy=data.get("device_binding_policy", "auto"),
        allocation_policy=data.get("allocation_policy"),
    )


def _load_manifest_file(path: Path) -> dict:
    """Load a YAML manifest file as a dict."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def resolve_manifest_from_path(manifest_path: str) -> ResourceManifest:
    """Load a resource manifest from a YAML file.

    Args:
        manifest_path: Path to the manifest file (.yaml or .yml).

    Returns:
        Parsed ResourceManifest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Resource manifest file not found: {manifest_path}")
    data = _load_manifest_file(path)
    manifest = _parse_manifest(data)
    logger.info(
        "Resource manifest loaded from %s: %d resources, "
        "binding_policy=%s, allocation_policy=%s",
        manifest_path,
        len(manifest.resources),
        manifest.device_binding_policy,
        manifest.allocation_policy,
    )
    return manifest


def resolve_manifest(config: "LMCacheEngineConfig") -> ResourceManifest | None:
    """Load resource manifest from config.

    Looks in extra_config["resource_manifest"] for either:
      - An inline dict (the manifest itself)
      - A string path to a YAML file
    Returns None if no manifest is configured.
    """
    raw = config.get_extra_config_value("resource_manifest", None)
    logger.info("Resolving resource manifest from config: %s", raw)
    if raw is None:
        return None

    if isinstance(raw, str):
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"Resource manifest file not found: {raw}"
            )
        data = _load_manifest_file(path)
    elif isinstance(raw, dict):
        data = raw
    else:
        raise TypeError(
            f"resource_manifest must be a dict or a file path string, "
            f"got {type(raw)}"
        )

    manifest = _parse_manifest(data)
    logger.info(
        "Resource manifest loaded: %d resources, "
        "binding_policy=%s, allocation_policy=%s",
        len(manifest.resources),
        manifest.device_binding_policy,
        manifest.allocation_policy,
    )
    return manifest


def parse_pcie_bar_env_vars(
    default_size_gb: float,
) -> ResourceManifest | None:
    """Parse PCIE_BAR_DEVICES / PCIE_BAR_DRAM_GB env vars into a ResourceManifest.

    Returns None when PCIE_BAR_DEVICES is not set.

    Env vars:
        PCIE_BAR_DEVICES: comma-separated entries ``path[:offset_gb[:size_gb]]``
        PCIE_BAR_DRAM_GB: optional DRAM region size (prepended to the list)
        PCIE_BAR_POLICY: allocation policy name (stored in the manifest)
    """
    devices_str = os.environ.get("PCIE_BAR_DEVICES")
    if not devices_str:
        return None

    resources: list[ResourceDescriptor] = []

    dram_gb_str = os.environ.get("PCIE_BAR_DRAM_GB")
    if dram_gb_str:
        resources.append(
            ResourceDescriptor(
                name="dram",
                type="dram",
                capacity_gb=float(dram_gb_str),
            )
        )

    for i, entry in enumerate(devices_str.split(",")):
        parts = entry.strip().rsplit(":", 2)
        if not parts[0]:
            raise ValueError(
                f"PCIE_BAR_DEVICES entry {i} is missing a path. "
                "Format: path[:offset_gb[:size_gb]]"
            )
        path = parts[0]
        offset_gb = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        offset = int(offset_gb * 1024**3)
        size_gb = (
            float(parts[2]) if len(parts) > 2 and parts[2] else default_size_gb
        )
        resources.append(
            ResourceDescriptor(
                name=f"bar_{i}",
                type="pcie_bar",
                capacity_gb=size_gb,
                path=path,
                offset=offset,
            )
        )

    policy = os.environ.get("PCIE_BAR_POLICY", "auto").lower()
    return ResourceManifest(
        resources=resources,
        allocation_policy=policy,
    )
