# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any
import inspect
import warnings

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

_LEGACY_WARNED: set[tuple[type[Any], str]] = set()

_SIGNATURE_MISMATCH_PATTERNS = (
    "unexpected keyword argument",
    "missing required keyword-only argument",
    "missing 1 required positional argument",
    "missing 2 required positional arguments",
    "missing 3 required positional arguments",
    "required positional argument",
    "takes ",
    "positional argument",
    "positional arguments",
    "multiple values for argument",
    "too many positional arguments",
)


def _warn_legacy_once(backend: Any, method_name: str, details: str) -> None:
    backend_cls = backend.__class__
    warn_key = (backend_cls, method_name)
    if warn_key in _LEGACY_WARNED:
        return
    _LEGACY_WARNED.add(warn_key)
    warnings.warn(
        (
            f"Legacy backend signature detected for {backend_cls.__name__}."
            f"{method_name}: {details}"
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def reset_legacy_warning_state_for_tests() -> None:
    _LEGACY_WARNED.clear()


def _is_signature_mismatch(exc: TypeError) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _SIGNATURE_MISMATCH_PATTERNS)


def _get_signature(method: Any) -> inspect.Signature | None:
    try:
        return inspect.signature(method)
    except (TypeError, ValueError):
        return None


def _has_var_keyword(sig: inspect.Signature) -> bool:
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _supports_param(sig: inspect.Signature, name: str) -> bool:
    return name in sig.parameters or _has_var_keyword(sig)


def _positional_params(sig: inspect.Signature) -> list[inspect.Parameter]:
    return [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def _legacy_non_blocking_order(sig: inspect.Signature) -> bool:
    positional = _positional_params(sig)
    if not positional:
        return False
    return positional[0].name == "lookup_id"


def _legacy_transfer_positional(sig: inspect.Signature) -> bool:
    positional = _positional_params(sig)
    if len(positional) >= 3:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )


def _try_candidates(
    method_name: str,
    attempts: list[tuple[str, Any]],
) -> tuple[str, Any]:
    last_exc: TypeError | None = None
    for candidate_name, attempt in attempts:
        try:
            return candidate_name, attempt()
        except TypeError as exc:
            if not _is_signature_mismatch(exc):
                raise
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"No call attempts available for {method_name}")


def call_batched_get_non_blocking(
    backend: Any,
    *,
    keys: list[CacheEngineKey],
    lookup_id: str,
    transfer_spec: Any = None,
) -> Any:
    method = backend.batched_get_non_blocking
    sig = _get_signature(method)

    if sig is not None:
        if _legacy_non_blocking_order(sig):
            args: list[Any] = [lookup_id, keys]
            if _legacy_transfer_positional(sig):
                args.append(transfer_spec)
            _warn_legacy_once(
                backend,
                "batched_get_non_blocking",
                (
                    "please update to batched_get_non_blocking("
                    "keys, *, lookup_id, transfer_spec=None)."
                ),
            )
            return method(*args)

        kwargs: dict[str, Any] = {}
        if _supports_param(sig, "lookup_id"):
            kwargs["lookup_id"] = lookup_id
        if _supports_param(sig, "transfer_spec"):
            kwargs["transfer_spec"] = transfer_spec
        if "lookup_id" not in kwargs:
            _warn_legacy_once(
                backend,
                "batched_get_non_blocking",
                (
                    "backend does not accept lookup_id keyword; "
                    "please update to batched_get_non_blocking("
                    "keys, *, lookup_id, transfer_spec=None)."
                ),
            )
        return method(keys, **kwargs)

    attempts = [
        (
            "canonical",
            lambda: method(keys, lookup_id=lookup_id, transfer_spec=transfer_spec),
        ),
        (
            "legacy-positional",
            lambda: method(lookup_id, keys, transfer_spec),
        ),
        (
            "legacy-no-transfer-spec",
            lambda: method(lookup_id, keys),
        ),
    ]
    selected, result = _try_candidates("batched_get_non_blocking", attempts)
    if selected != "canonical":
        _warn_legacy_once(
            backend,
            "batched_get_non_blocking",
            (
                "fallback dispatch used; please update to "
                "batched_get_non_blocking(keys, *, lookup_id, transfer_spec=None)."
            ),
        )
    return result


def call_batched_get_blocking(
    backend: Any,
    *,
    keys: list[CacheEngineKey],
    lookup_id: str | None = None,
    transfer_spec: Any = None,
) -> Any:
    method = backend.batched_get_blocking
    sig = _get_signature(method)

    if sig is not None:
        kwargs: dict[str, Any] = {}
        if _supports_param(sig, "lookup_id"):
            kwargs["lookup_id"] = lookup_id
        if _supports_param(sig, "transfer_spec"):
            kwargs["transfer_spec"] = transfer_spec
        if "lookup_id" not in kwargs or "transfer_spec" not in kwargs:
            _warn_legacy_once(
                backend,
                "batched_get_blocking",
                (
                    "please update to batched_get_blocking("
                    "keys, *, lookup_id=None, transfer_spec=None)."
                ),
            )
        return method(keys, **kwargs)

    attempts = [
        (
            "canonical",
            lambda: method(keys, lookup_id=lookup_id, transfer_spec=transfer_spec),
        ),
        ("no-lookup-id", lambda: method(keys, transfer_spec=transfer_spec)),
        ("legacy-minimal", lambda: method(keys)),
    ]
    selected, result = _try_candidates("batched_get_blocking", attempts)
    if selected != "canonical":
        _warn_legacy_once(
            backend,
            "batched_get_blocking",
            (
                "fallback dispatch used; please update to "
                "batched_get_blocking(keys, *, lookup_id=None, transfer_spec=None)."
            ),
        )
    return result


def call_batched_submit_put_task(
    backend: Any,
    *,
    keys: Sequence[CacheEngineKey],
    objs: list[MemoryObj],
    lookup_id: str | None = None,
    transfer_spec: Any = None,
) -> Any:
    method = backend.batched_submit_put_task
    sig = _get_signature(method)

    if sig is not None:
        kwargs: dict[str, Any] = {}
        if _supports_param(sig, "lookup_id"):
            kwargs["lookup_id"] = lookup_id
        if _supports_param(sig, "transfer_spec"):
            kwargs["transfer_spec"] = transfer_spec
        if "lookup_id" not in kwargs or "transfer_spec" not in kwargs:
            _warn_legacy_once(
                backend,
                "batched_submit_put_task",
                (
                    "please update to batched_submit_put_task("
                    "keys, objs, *, lookup_id=None, transfer_spec=None)."
                ),
            )
        return method(keys, objs, **kwargs)

    attempts = [
        (
            "canonical",
            lambda: method(
                keys, objs, lookup_id=lookup_id, transfer_spec=transfer_spec
            ),
        ),
        ("no-lookup-id", lambda: method(keys, objs, transfer_spec=transfer_spec)),
        ("legacy-minimal", lambda: method(keys, objs)),
    ]
    selected, result = _try_candidates("batched_submit_put_task", attempts)
    if selected != "canonical":
        _warn_legacy_once(
            backend,
            "batched_submit_put_task",
            (
                "fallback dispatch used; please update to "
                "batched_submit_put_task(keys, objs, *, "
                "lookup_id=None, transfer_spec=None)."
            ),
        )
    return result
