# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# fmt: off
# isort: skip_file
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Envelope(_message.Message):
    __slots__ = ("region_id", "expected_region_epoch", "client_id", "client_incarnation", "request_id")
    REGION_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_REGION_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    region_id: str
    expected_region_epoch: int
    client_id: str
    client_incarnation: int
    request_id: int
    def __init__(self, region_id: _Optional[str] = ..., expected_region_epoch: _Optional[int] = ..., client_id: _Optional[str] = ..., client_incarnation: _Optional[int] = ..., request_id: _Optional[int] = ...) -> None: ...

class RegionContract(_message.Message):
    __slots__ = ("region_id", "region_epoch", "capacity_bytes", "alignment_bytes", "layout_fingerprint", "visibility_mode", "max_batch_entries", "reset_required")
    REGION_ID_FIELD_NUMBER: _ClassVar[int]
    REGION_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    ALIGNMENT_BYTES_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_MODE_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    RESET_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    region_id: str
    region_epoch: int
    capacity_bytes: int
    alignment_bytes: int
    layout_fingerprint: bytes
    visibility_mode: str
    max_batch_entries: int
    reset_required: bool
    def __init__(self, region_id: _Optional[str] = ..., region_epoch: _Optional[int] = ..., capacity_bytes: _Optional[int] = ..., alignment_bytes: _Optional[int] = ..., layout_fingerprint: _Optional[bytes] = ..., visibility_mode: _Optional[str] = ..., max_batch_entries: _Optional[int] = ..., reset_required: bool = ...) -> None: ...

class RegisterClientRequest(_message.Message):
    __slots__ = ("env", "layout_fingerprint", "mapped_bytes", "visibility_mode")
    ENV_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    MAPPED_BYTES_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_MODE_FIELD_NUMBER: _ClassVar[int]
    env: Envelope
    layout_fingerprint: bytes
    mapped_bytes: int
    visibility_mode: str
    def __init__(self, env: _Optional[_Union[Envelope, _Mapping]] = ..., layout_fingerprint: _Optional[bytes] = ..., mapped_bytes: _Optional[int] = ..., visibility_mode: _Optional[str] = ...) -> None: ...

class RegisterClientReply(_message.Message):
    __slots__ = ("region_epoch", "retired_writes", "released_leases")
    REGION_EPOCH_FIELD_NUMBER: _ClassVar[int]
    RETIRED_WRITES_FIELD_NUMBER: _ClassVar[int]
    RELEASED_LEASES_FIELD_NUMBER: _ClassVar[int]
    region_epoch: int
    retired_writes: int
    released_leases: int
    def __init__(self, region_epoch: _Optional[int] = ..., retired_writes: _Optional[int] = ..., released_leases: _Optional[int] = ...) -> None: ...

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

class ObjectLayout(_message.Message):
    __slots__ = ("shapes", "dtypes")
    SHAPES_FIELD_NUMBER: _ClassVar[int]
    DTYPES_FIELD_NUMBER: _ClassVar[int]
    shapes: _containers.RepeatedCompositeFieldContainer[TensorShape]
    dtypes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, shapes: _Optional[_Iterable[_Union[TensorShape, _Mapping]]] = ..., dtypes: _Optional[_Iterable[str]] = ...) -> None: ...

class Handle(_message.Message):
    __slots__ = ("offset", "length", "generation")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    offset: int
    length: int
    generation: int
    def __init__(self, offset: _Optional[int] = ..., length: _Optional[int] = ..., generation: _Optional[int] = ...) -> None: ...

class WriteEntry(_message.Message):
    __slots__ = ("key", "payload_bytes", "layout")
    KEY_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_BYTES_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    key: ObjectKey
    payload_bytes: int
    layout: ObjectLayout
    def __init__(self, key: _Optional[_Union[ObjectKey, _Mapping]] = ..., payload_bytes: _Optional[int] = ..., layout: _Optional[_Union[ObjectLayout, _Mapping]] = ...) -> None: ...

class ReserveWriteRequest(_message.Message):
    __slots__ = ("env", "entries")
    ENV_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    env: Envelope
    entries: _containers.RepeatedCompositeFieldContainer[WriteEntry]
    def __init__(self, env: _Optional[_Union[Envelope, _Mapping]] = ..., entries: _Optional[_Iterable[_Union[WriteEntry, _Mapping]]] = ...) -> None: ...

class WriteGrant(_message.Message):
    __slots__ = ("status", "handle", "token")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[WriteGrant.Status]
        WRITE_GRANTED: _ClassVar[WriteGrant.Status]
        EXISTS_VALID: _ClassVar[WriteGrant.Status]
        BUSY_WRITING: _ClassVar[WriteGrant.Status]
        OUT_OF_SPACE: _ClassVar[WriteGrant.Status]
    STATUS_UNSPECIFIED: WriteGrant.Status
    WRITE_GRANTED: WriteGrant.Status
    EXISTS_VALID: WriteGrant.Status
    BUSY_WRITING: WriteGrant.Status
    OUT_OF_SPACE: WriteGrant.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    status: WriteGrant.Status
    handle: Handle
    token: bytes
    def __init__(self, status: _Optional[_Union[WriteGrant.Status, str]] = ..., handle: _Optional[_Union[Handle, _Mapping]] = ..., token: _Optional[bytes] = ...) -> None: ...

class ReserveWriteReply(_message.Message):
    __slots__ = ("grants",)
    GRANTS_FIELD_NUMBER: _ClassVar[int]
    grants: _containers.RepeatedCompositeFieldContainer[WriteGrant]
    def __init__(self, grants: _Optional[_Iterable[_Union[WriteGrant, _Mapping]]] = ...) -> None: ...

class ReadEntry(_message.Message):
    __slots__ = ("key", "reader_count")
    KEY_FIELD_NUMBER: _ClassVar[int]
    READER_COUNT_FIELD_NUMBER: _ClassVar[int]
    key: ObjectKey
    reader_count: int
    def __init__(self, key: _Optional[_Union[ObjectKey, _Mapping]] = ..., reader_count: _Optional[int] = ...) -> None: ...

class ReserveReadRequest(_message.Message):
    __slots__ = ("env", "entries")
    ENV_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    env: Envelope
    entries: _containers.RepeatedCompositeFieldContainer[ReadEntry]
    def __init__(self, env: _Optional[_Union[Envelope, _Mapping]] = ..., entries: _Optional[_Iterable[_Union[ReadEntry, _Mapping]]] = ...) -> None: ...

class ReadGrant(_message.Message):
    __slots__ = ("status", "handle", "leases", "layout", "payload_bytes")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[ReadGrant.Status]
        READ_GRANTED: _ClassVar[ReadGrant.Status]
        MISS: _ClassVar[ReadGrant.Status]
        BUSY_WRITING: _ClassVar[ReadGrant.Status]
    STATUS_UNSPECIFIED: ReadGrant.Status
    READ_GRANTED: ReadGrant.Status
    MISS: ReadGrant.Status
    BUSY_WRITING: ReadGrant.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    LEASES_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_BYTES_FIELD_NUMBER: _ClassVar[int]
    status: ReadGrant.Status
    handle: Handle
    leases: _containers.RepeatedScalarFieldContainer[bytes]
    layout: ObjectLayout
    payload_bytes: int
    def __init__(self, status: _Optional[_Union[ReadGrant.Status, str]] = ..., handle: _Optional[_Union[Handle, _Mapping]] = ..., leases: _Optional[_Iterable[bytes]] = ..., layout: _Optional[_Union[ObjectLayout, _Mapping]] = ..., payload_bytes: _Optional[int] = ...) -> None: ...

class ReserveReadReply(_message.Message):
    __slots__ = ("grants",)
    GRANTS_FIELD_NUMBER: _ClassVar[int]
    grants: _containers.RepeatedCompositeFieldContainer[ReadGrant]
    def __init__(self, grants: _Optional[_Iterable[_Union[ReadGrant, _Mapping]]] = ...) -> None: ...

class TokenBatch(_message.Message):
    __slots__ = ("env", "tokens")
    ENV_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    env: Envelope
    tokens: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, env: _Optional[_Union[Envelope, _Mapping]] = ..., tokens: _Optional[_Iterable[bytes]] = ...) -> None: ...

class ResultBatch(_message.Message):
    __slots__ = ("results",)
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[ResultBatch.Status]
        OK: _ClassVar[ResultBatch.Status]
        STALE_TOKEN: _ClassVar[ResultBatch.Status]
    STATUS_UNSPECIFIED: ResultBatch.Status
    OK: ResultBatch.Status
    STALE_TOKEN: ResultBatch.Status
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedScalarFieldContainer[ResultBatch.Status]
    def __init__(self, results: _Optional[_Iterable[_Union[ResultBatch.Status, str]]] = ...) -> None: ...

class RegionUsage(_message.Message):
    __slots__ = ("capacity_bytes", "allocated_bytes", "valid_bytes", "writing", "valid", "consumed", "read_leases", "clients")
    CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_BYTES_FIELD_NUMBER: _ClassVar[int]
    VALID_BYTES_FIELD_NUMBER: _ClassVar[int]
    WRITING_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    CONSUMED_FIELD_NUMBER: _ClassVar[int]
    READ_LEASES_FIELD_NUMBER: _ClassVar[int]
    CLIENTS_FIELD_NUMBER: _ClassVar[int]
    capacity_bytes: int
    allocated_bytes: int
    valid_bytes: int
    writing: int
    valid: int
    consumed: int
    read_leases: int
    clients: int
    def __init__(self, capacity_bytes: _Optional[int] = ..., allocated_bytes: _Optional[int] = ..., valid_bytes: _Optional[int] = ..., writing: _Optional[int] = ..., valid: _Optional[int] = ..., consumed: _Optional[int] = ..., read_leases: _Optional[int] = ..., clients: _Optional[int] = ...) -> None: ...

class CloseClientReply(_message.Message):
    __slots__ = ("aborted_writes", "released_leases")
    ABORTED_WRITES_FIELD_NUMBER: _ClassVar[int]
    RELEASED_LEASES_FIELD_NUMBER: _ClassVar[int]
    aborted_writes: int
    released_leases: int
    def __init__(self, aborted_writes: _Optional[int] = ..., released_leases: _Optional[int] = ...) -> None: ...
