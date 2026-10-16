# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# fmt: off
# isort: skip_file
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class IpcCacheServerKey(_message.Message):
    __slots__ = ("model_name", "world_size", "worker_id", "token_ids", "start", "end", "request_id", "cache_salt", "encoded_request_configs", "num_kv_readers", "event_profile", "event_bindings")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CACHE_SALT_FIELD_NUMBER: _ClassVar[int]
    ENCODED_REQUEST_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    NUM_KV_READERS_FIELD_NUMBER: _ClassVar[int]
    EVENT_PROFILE_FIELD_NUMBER: _ClassVar[int]
    EVENT_BINDINGS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    world_size: int
    worker_id: int
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    start: int
    end: int
    request_id: str
    cache_salt: str
    encoded_request_configs: bytes
    num_kv_readers: int
    event_profile: KVEventProfile
    event_bindings: _containers.RepeatedCompositeFieldContainer[EventBinding]
    def __init__(self, model_name: _Optional[str] = ..., world_size: _Optional[int] = ..., worker_id: _Optional[int] = ..., token_ids: _Optional[_Iterable[int]] = ..., start: _Optional[int] = ..., end: _Optional[int] = ..., request_id: _Optional[str] = ..., cache_salt: _Optional[str] = ..., encoded_request_configs: _Optional[bytes] = ..., num_kv_readers: _Optional[int] = ..., event_profile: _Optional[_Union[KVEventProfile, _Mapping]] = ..., event_bindings: _Optional[_Iterable[_Union[EventBinding, _Mapping]]] = ...) -> None: ...

class EventIpcHandleResult(_message.Message):
    __slots__ = ("event_ipc_handle", "success")
    EVENT_IPC_HANDLE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    event_ipc_handle: bytes
    success: bool
    def __init__(self, event_ipc_handle: _Optional[bytes] = ..., success: bool = ...) -> None: ...

class BlockIdGroup(_message.Message):
    __slots__ = ("block_ids",)
    BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    block_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, block_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class DeviceIpcWrapper(_message.Message):
    __slots__ = ("pickled_payload",)
    PICKLED_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    pickled_payload: bytes
    def __init__(self, pickled_payload: _Optional[bytes] = ...) -> None: ...

class EngineGroupInfo(_message.Message):
    __slots__ = ("engine_group_id", "layer_indices", "tokens_per_block", "sw_size_tokens", "extra_object_group_tag", "recurrent_state")
    ENGINE_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_INDICES_FIELD_NUMBER: _ClassVar[int]
    TOKENS_PER_BLOCK_FIELD_NUMBER: _ClassVar[int]
    SW_SIZE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EXTRA_OBJECT_GROUP_TAG_FIELD_NUMBER: _ClassVar[int]
    RECURRENT_STATE_FIELD_NUMBER: _ClassVar[int]
    engine_group_id: int
    layer_indices: _containers.RepeatedScalarFieldContainer[int]
    tokens_per_block: int
    sw_size_tokens: int
    extra_object_group_tag: int
    recurrent_state: bool
    def __init__(self, engine_group_id: _Optional[int] = ..., layer_indices: _Optional[_Iterable[int]] = ..., tokens_per_block: _Optional[int] = ..., sw_size_tokens: _Optional[int] = ..., extra_object_group_tag: _Optional[int] = ..., recurrent_state: bool = ...) -> None: ...

class KVEventProfile(_message.Message):
    __slots__ = ("profile_id", "block_size")
    PROFILE_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    profile_id: str
    block_size: int
    def __init__(self, profile_id: _Optional[str] = ..., block_size: _Optional[int] = ...) -> None: ...

class EventBinding(_message.Message):
    __slots__ = ("profile_id", "encoded_block_hashes", "encoded_parent_block_hash", "token_ids", "block_size")
    PROFILE_ID_FIELD_NUMBER: _ClassVar[int]
    ENCODED_BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    ENCODED_PARENT_BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    profile_id: str
    encoded_block_hashes: _containers.RepeatedScalarFieldContainer[bytes]
    encoded_parent_block_hash: bytes
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    block_size: int
    def __init__(self, profile_id: _Optional[str] = ..., encoded_block_hashes: _Optional[_Iterable[bytes]] = ..., encoded_parent_block_hash: _Optional[bytes] = ..., token_ids: _Optional[_Iterable[int]] = ..., block_size: _Optional[int] = ...) -> None: ...
