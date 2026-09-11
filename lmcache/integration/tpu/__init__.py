# Copyright 2024-2025 LMCache Authors
# SPDX-License-Identifier: Apache-2.0
"""LMCache TPU integration.

Enables LMCache as a host-side KV storage tier behind the vLLM tpu-inference
``TPUOffloadConnector``. Unlike the CUDA/XPU/HPU GPU connectors, this path does
NOT move torch tensors out of device memory: the TPUOffloadConnector already
owns the JAX-array HBM<->host movement, and hands LMCache a flat, host-resident
byte buffer per KV block. LMCache provides the tiers below host RAM (disk today;
remote/P2P/cross-instance next).

This module has no torch_xla / JAX dependency and runs on any host (incl. CPU).
"""
