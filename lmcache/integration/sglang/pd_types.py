# SPDX-License-Identifier: Apache-2.0
"""Dependency-free data types for SGLang PD (prefill/decode) disaggregation.

This module intentionally imports nothing from ``sglang`` or ``torch`` so that
the routing types can be constructed and validated (and unit-tested) without a
GPU or a full SGLang install. The heavier adapter in
:mod:`lmcache.integration.sglang.sglang_adapter` re-exports :class:`DisaggSpec`.
"""

# Standard
from dataclasses import dataclass, field
from typing import List


@dataclass
class DisaggSpec:
    """Per-request routing information for LMCache PD disaggregation.

    Describes the decode (receiver) instance that a prefill (sender) instance
    must push this request's KV cache to over NIXL. It is duck-typed by
    ``lmcache.v1.storage_backend.pd_backend.PDBackend.batched_submit_put_task``,
    which reads ``receiver_host`` and indexes the port lists by the sender's
    tensor-parallel rank (``receiver_init_port[tp_rank]`` etc.). The port fields
    are therefore per-tp-rank lists, one listen port per receiver rank.

    Args:
        req_id: Correlation id shared with the proxy; sent back in the
            ``ProxyNotif`` once the last-prefill transfer completes.
        receiver_host: Hostname / IP of the decode instance.
        receiver_init_port: NIXL side-channel init ports, indexed by tp rank.
        receiver_alloc_port: Remote-allocation ports, indexed by tp rank.
        receiver_query_port: Cache-query ports, indexed by tp rank. Only used
            when bidirectional cache query is enabled; empty otherwise.
        is_last_prefill: True on the final store for a request, which triggers
            the proxy notification. Single-shot stores set this True.
    """

    req_id: str
    receiver_host: str
    receiver_init_port: List[int]
    receiver_alloc_port: List[int]
    receiver_query_port: List[int] = field(default_factory=list)
    is_last_prefill: bool = True

    @classmethod
    def from_dict(cls, spec: dict) -> "DisaggSpec":
        """Build a ``DisaggSpec`` from a proxy-injected ``kv_transfer_params``
        ``disagg_spec`` mapping.

        Expected schema (all required unless noted):
            ``req_id`` (str), ``receiver_host`` (str),
            ``receiver_init_port`` (list[int]), ``receiver_alloc_port`` (list[int]),
            ``receiver_query_port`` (list[int], optional),
            ``is_last_prefill`` (bool, optional, default True).

        Args:
            spec: The ``disagg_spec`` mapping described above.

        Returns:
            The parsed ``DisaggSpec``.

        Raises:
            ValueError: If a required key is missing or a port field is not a
                list of ints.
        """
        required = (
            "req_id",
            "receiver_host",
            "receiver_init_port",
            "receiver_alloc_port",
        )
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(
                f"disagg_spec is missing required keys: {missing}. "
                f"Got keys: {sorted(spec.keys())}"
            )
        for port_key in ("receiver_init_port", "receiver_alloc_port"):
            value = spec[port_key]
            if not isinstance(value, list) or not all(
                isinstance(p, int) for p in value
            ):
                raise ValueError(
                    f"disagg_spec['{port_key}'] must be a list[int] "
                    f"(one port per receiver tp rank), got {value!r}"
                )
        return cls(
            req_id=str(spec["req_id"]),
            receiver_host=str(spec["receiver_host"]),
            receiver_init_port=list(spec["receiver_init_port"]),
            receiver_alloc_port=list(spec["receiver_alloc_port"]),
            receiver_query_port=list(spec.get("receiver_query_port", [])),
            is_last_prefill=bool(spec.get("is_last_prefill", True)),
        )
