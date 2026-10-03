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

class ObjectKey(_message.Message):
    __slots__ = ("chunk_hash", "model_name", "kv_rank", "object_group_id", "cache_salt")
    CHUNK_HASH_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    KV_RANK_FIELD_NUMBER: _ClassVar[int]
    OBJECT_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    CACHE_SALT_FIELD_NUMBER: _ClassVar[int]
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    object_group_id: int
    cache_salt: str
    def __init__(self, chunk_hash: _Optional[bytes] = ..., model_name: _Optional[str] = ..., kv_rank: _Optional[int] = ..., object_group_id: _Optional[int] = ..., cache_salt: _Optional[str] = ...) -> None: ...

class TensorShape(_message.Message):
    __slots__ = ("dims",)
    DIMS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...

class MemoryLayoutDesc(_message.Message):
    __slots__ = ("shapes", "dtypes")
    SHAPES_FIELD_NUMBER: _ClassVar[int]
    DTYPES_FIELD_NUMBER: _ClassVar[int]
    shapes: _containers.RepeatedCompositeFieldContainer[TensorShape]
    dtypes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, shapes: _Optional[_Iterable[_Union[TensorShape, _Mapping]]] = ..., dtypes: _Optional[_Iterable[str]] = ...) -> None: ...

class TransferChannelAddress(_message.Message):
    __slots__ = ("offset", "size")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    offset: int
    size: int
    def __init__(self, offset: _Optional[int] = ..., size: _Optional[int] = ...) -> None: ...

class P2pLookupAndLockRequest(_message.Message):
    __slots__ = ("keys", "group_layout_descs")
    class GroupLayoutDescsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: MemoryLayoutDesc
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[MemoryLayoutDesc, _Mapping]] = ...) -> None: ...
    KEYS_FIELD_NUMBER: _ClassVar[int]
    GROUP_LAYOUT_DESCS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedCompositeFieldContainer[ObjectKey]
    group_layout_descs: _containers.MessageMap[int, MemoryLayoutDesc]
    def __init__(self, keys: _Optional[_Iterable[_Union[ObjectKey, _Mapping]]] = ..., group_layout_descs: _Optional[_Mapping[int, MemoryLayoutDesc]] = ...) -> None: ...

class P2pLookupAndLockResponse(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    def __init__(self, task_id: _Optional[int] = ...) -> None: ...

class P2pQueryLookupResultsRequest(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    def __init__(self, task_id: _Optional[int] = ...) -> None: ...

class TransferChannelAddressList(_message.Message):
    __slots__ = ("addresses",)
    ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    addresses: _containers.RepeatedCompositeFieldContainer[TransferChannelAddress]
    def __init__(self, addresses: _Optional[_Iterable[_Union[TransferChannelAddress, _Mapping]]] = ...) -> None: ...

class P2pQueryLookupResultsResponse(_message.Message):
    __slots__ = ("addresses",)
    ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    addresses: TransferChannelAddressList
    def __init__(self, addresses: _Optional[_Union[TransferChannelAddressList, _Mapping]] = ...) -> None: ...

class P2pUnlockObjectsRequest(_message.Message):
    __slots__ = ("keys",)
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedCompositeFieldContainer[ObjectKey]
    def __init__(self, keys: _Optional[_Iterable[_Union[ObjectKey, _Mapping]]] = ...) -> None: ...

class P2pUnlockObjectsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
