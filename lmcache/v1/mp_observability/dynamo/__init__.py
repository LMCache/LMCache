# SPDX-License-Identifier: Apache-2.0

"""Dynamo KV-event publishing for the MP cache server.

Emits ``BlockStored`` / ``BlockRemoved`` events in vLLM-compatible wire format
over ZMQ so a Dynamo-side relay can forward them into Dynamo's KV-aware router.
"""
