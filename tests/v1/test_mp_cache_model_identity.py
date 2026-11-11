# SPDX-License-Identifier: Apache-2.0
"""Cache identity of the vLLM MP connector: the model revision tag.

The connector keeps the served model name and adds the resolved revision as an
``lmcache.tag.revision`` request-configuration entry, which the server turns
into an identity tag. These tests pin the three decisions: which revision
becomes a tag, that the tag reaches the request configuration, and that engines
without a pinned revision stay untagged. No GPU needed.

The modules that import vLLM are imported lazily so the k3 unit environment,
which has no vLLM, still collects this file.
"""

# Standard
from types import SimpleNamespace
import importlib.util

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import get_request_tags

requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="requires vLLM"
)

_COMMIT = "a" * 40
_TAG_KEY = "lmcache.tag.revision"


class ResolvedRevision(str):
    """Duck-typed stand-in for ``huggingface_hub.ResolvedRevision``.

    vLLM stores the commit hash of a Hub revision on the string it returns,
    keeping the requested ref as the string value.
    """

    def __new__(cls, resolved: str, initial: str | None = None) -> "ResolvedRevision":
        revision = super().__new__(cls, initial if initial is not None else "main")
        revision.initial = initial  # type: ignore[attr-defined]
        revision.resolved = resolved  # type: ignore[attr-defined]
        return revision


def _config(*, revision: object = None) -> SimpleNamespace:
    """Build the vLLM config subset the revision helper reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", revision=revision)
    )


def _revision_token(config: SimpleNamespace) -> str | None:
    """Resolve the revision token for ``config`` through the real helper."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        _revision_cache_token,
    )

    return _revision_cache_token(config)


def _apply(tag: str, configs: dict[str, object] | None) -> dict[str, object] | None:
    """Run the connector's tag injection on a minimal tracker stub."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    connector = SimpleNamespace(_revision_tag=tag)
    tracker = SimpleNamespace(request_id="req", request_configs=configs)
    LMCacheMPConnector._apply_revision_tag(connector, tracker)  # type: ignore[arg-type]
    return tracker.request_configs


@requires_vllm
def test_unpinned_revision_has_no_token():
    """Engines without ``--revision`` must not change cache identity."""
    assert _revision_token(_config()) is None
    assert _revision_token(_config(revision="")) is None


@requires_vllm
def test_resolved_default_branch_is_not_a_pin():
    """vLLM resolves the default branch for every Hub model; that is not a pin."""
    assert _revision_token(_config(revision=ResolvedRevision("c" * 40))) is None


@requires_vllm
def test_pinned_revision_uses_the_commit_hash():
    """A pinned branch tracks the commit, so moving it changes the identity."""
    assert (
        _revision_token(_config(revision=ResolvedRevision(_COMMIT, "main"))) == _COMMIT
    )


@requires_vllm
def test_unresolved_revision_is_used_verbatim():
    """A revision vLLM could not resolve still separates the identities."""
    assert _revision_token(_config(revision="refs/pr/4")) == "refs/pr/4"


@requires_vllm
def test_revision_tag_reaches_the_request_configuration():
    """The tag must be visible to the server through the request configs."""
    configs = _apply(_COMMIT, None)

    assert configs == {_TAG_KEY: _COMMIT}


@requires_vllm
def test_no_revision_leaves_request_configuration_alone():
    """Untagged requests are what engines without a pin keep sending."""
    assert _apply("", None) is None
    assert _apply("", {"lmcache.tag.salt": "x"}) == {"lmcache.tag.salt": "x"}


@requires_vllm
def test_engine_revision_wins_over_a_request_supplied_tag():
    """A stale request tag must not redirect objects to another revision."""
    configs = _apply(_COMMIT, {_TAG_KEY: "b" * 40, "lmcache.tag.salt": "x"})

    assert configs == {_TAG_KEY: _COMMIT, "lmcache.tag.salt": "x"}


def test_server_derives_the_revision_tag_from_the_configuration():
    """The server-side contract: ``lmcache.tag.*`` becomes identity tags."""
    tags = get_request_tags({_TAG_KEY: _COMMIT, "lmcache.mp.save_decode_cache": True})

    assert dict(tags) == {"revision": _COMMIT}


def test_unrelated_request_configuration_is_not_an_identity_tag():
    """Only the ``lmcache.tag.*`` namespace may partition the cache."""
    assert dict(get_request_tags({"lmcache.mp.save_decode_cache": True})) == {}
