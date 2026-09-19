# SPDX-License-Identifier: Apache-2.0
"""HTTP regression coverage for default quotas in coordinator status views."""

# Standard
from collections.abc import Iterator

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a fresh coordinator-backed client for one test case.

    Periodic health and eviction work is disabled so each test observes only
    its explicit HTTP requests. The fixture closes the client after the test.

    Yields:
        A ``TestClient`` bound to a newly created coordinator application.
    """
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    with TestClient(create_app(config)) as test_client:
        yield test_client


def _set_default(client: TestClient, limit_gb: float | None) -> None:
    """Configure an L2 fallback quota through the public endpoint.

    Args:
        client: Coordinator HTTP client supplied by the test fixture.
        limit_gb: Default L2 budget in GiB, or ``None`` to leave unregistered
            salts exempt from a default quota.

    Returns:
        ``None`` after the endpoint accepts and echoes the requested setting.

    Raises:
        AssertionError: If the endpoint does not return the expected success
            response.
    """
    response = client.put("/quota/config", json={"default_limit_gb": limit_gb})
    assert response.status_code == 200
    assert response.json() == {"default_limit_gb": limit_gb}


def _store(client: TestClient, cache_salt: str, *, tier: str = "l2") -> None:
    """Publish one admitted cache event through the public ingest endpoint.

    Args:
        client: Coordinator HTTP client supplied by the test fixture.
        cache_salt: Tenant identity carried by the event's encoded cache key.
        tier: Accounted cache tier for the event; test callers use ``"l1"``
            or ``"l2"``.

    Returns:
        ``None`` after the coordinator reports that it admitted the event.

    Raises:
        AssertionError: If the event endpoint does not report one admitted
            batch and no duplicate or stale batches.
    """
    backend = "fs" if tier == "l2" else "dram"
    response = client.post(
        "/events",
        json={
            "batches": [
                {
                    "instance_id": "quota-default-status-test",
                    "incarnation": 1,
                    "seq": 1,
                    "event_type": "store",
                    "tier": tier,
                    "backend": backend,
                    "entries": [
                        {
                            "key": {
                                "chunk_hash_hex": "abcdef",
                                "model_name": "test-model",
                                "kv_rank": 0,
                                "cache_salt": cache_salt,
                            },
                            "size_bytes": 1024,
                        }
                    ],
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {"applied": 1, "duplicates": 0, "stale": 0}


def _status(
    client: TestClient, cache_salt: str, *, tier: str = "l2"
) -> dict[str, str | float | bool]:
    """Read one tenant's status through the public quota endpoint.

    Args:
        client: Coordinator HTTP client supplied by the test fixture.
        cache_salt: Tenant identity to request, including ``"_default"`` for
            the API's empty-salt path sentinel.
        tier: Accounted cache tier whose quota and usage are requested.

    Returns:
        The successful JSON status row, containing the salt string, quota and
        usage GiB floats, and the explicit-quota boolean.

    Raises:
        AssertionError: If the status endpoint does not return HTTP 200.
    """
    response = client.get(f"/quota/{cache_salt}", params={"tier": tier})
    assert response.status_code == 200
    return response.json()


def _listed_status(
    client: TestClient, cache_salt: str, *, tier: str = "l2"
) -> dict[str, str | float | bool]:
    """Select one tenant's row from the public quota status listing.

    Args:
        client: Coordinator HTTP client supplied by the test fixture.
        cache_salt: Tenant identity whose row must exist in the listing.
        tier: Accounted cache tier whose listing is requested.

    Returns:
        The selected JSON status row, containing the salt string, quota and
        usage GiB floats, and the explicit-quota boolean.

    Raises:
        AssertionError: If the list endpoint does not return HTTP 200.
        KeyError: If the successful response has no row for ``cache_salt``.
    """
    response = client.get("/quota", params={"tier": tier})
    assert response.status_code == 200
    by_salt = {entry["cache_salt"]: entry for entry in response.json()["by_cache_salt"]}
    return by_salt[cache_salt]


@pytest.mark.parametrize(
    ("default_limit_gb", "expected_limit_gb"),
    [(None, 0.0), (0.0, 0.0), (2.5, 2.5)],
    ids=["unset", "zero", "positive"],
)
def test_l2_single_status_reports_the_effective_default_limit(
    client: TestClient, default_limit_gb: float | None, expected_limit_gb: float
) -> None:
    """Verify an L2 single-status read exposes the configured default budget.

    Args:
        client: Fresh coordinator HTTP client supplied by the fixture.
        default_limit_gb: Default L2 quota configured for unregistered salts,
            or ``None`` when no default applies.
        expected_limit_gb: Status budget expected after applying that default.

    Returns:
        ``None`` after assertions distinguish the effective budget from
        explicit quota registration.
    """
    _set_default(client, default_limit_gb)
    _store(client, "default-single")

    status = _status(client, "default-single")
    assert status["quota_exists"] is False
    assert status["quota_limit_gb"] == expected_limit_gb


@pytest.mark.parametrize(
    ("default_limit_gb", "expected_limit_gb"),
    [(None, 0.0), (0.0, 0.0), (2.5, 2.5)],
    ids=["unset", "zero", "positive"],
)
def test_l2_listing_reports_the_effective_default_limit(
    client: TestClient, default_limit_gb: float | None, expected_limit_gb: float
) -> None:
    """Verify the L2 listing has the same default-quota meaning as a read.

    Args:
        client: Fresh coordinator HTTP client supplied by the fixture.
        default_limit_gb: Default L2 quota configured for unregistered salts,
            or ``None`` when no default applies.
        expected_limit_gb: Listing budget expected after applying that default.

    Returns:
        ``None`` after assertions verify both the budget and explicit-entry
        indicator in the listed row.
    """
    _set_default(client, default_limit_gb)
    _store(client, "default-list")

    status = _listed_status(client, "default-list")
    assert status["quota_exists"] is False
    assert status["quota_limit_gb"] == expected_limit_gb


@pytest.mark.parametrize(
    ("explicit_limit_gb", "expected_limit_gb"),
    [(0.0, 0.0), (4.0, 4.0)],
    ids=["zero", "positive"],
)
def test_explicit_quota_overrides_the_default_in_both_status_views(
    client: TestClient, explicit_limit_gb: float, expected_limit_gb: float
) -> None:
    """Verify explicit zero and positive limits override a positive default.

    Args:
        client: Fresh coordinator HTTP client supplied by the fixture.
        explicit_limit_gb: Per-salt L2 quota set through the public endpoint.
        expected_limit_gb: Budget both status views must report after the
            explicit registration.

    Returns:
        ``None`` after assertions verify the override and explicit-entry
        indicator in single and list responses.
    """
    _set_default(client, 2.5)
    _store(client, "explicit-override")
    response = client.put(
        "/quota/explicit-override", json={"limit_gb": explicit_limit_gb}
    )
    assert response.status_code == 200

    for status in (
        _status(client, "explicit-override"),
        _listed_status(client, "explicit-override"),
    ):
        assert status["quota_exists"] is True
        assert status["quota_limit_gb"] == expected_limit_gb


@pytest.mark.parametrize(
    ("default_limit_gb", "expected_limit_gb"),
    [(0.0, 0.0), (2.5, 2.5)],
    ids=["zero", "positive"],
)
def test_deleting_explicit_quota_restores_the_default_status(
    client: TestClient, default_limit_gb: float, expected_limit_gb: float
) -> None:
    """Verify deleting an explicit quota restores the fallback in both views.

    Args:
        client: Fresh coordinator HTTP client supplied by the fixture.
        default_limit_gb: Default L2 quota configured before the explicit
            registration.
        expected_limit_gb: Budget expected after deleting that registration.

    Returns:
        ``None`` after assertions verify the fallback and loss of the
        explicit-entry indicator in single and list responses.
    """
    _set_default(client, default_limit_gb)
    _store(client, "deleted-override")
    assert (
        client.put("/quota/deleted-override", json={"limit_gb": 4.0}).status_code == 200
    )
    assert client.delete("/quota/deleted-override").status_code == 200

    for status in (
        _status(client, "deleted-override"),
        _listed_status(client, "deleted-override"),
    ):
        assert status["quota_exists"] is False
        assert status["quota_limit_gb"] == expected_limit_gb


def test_l1_status_does_not_borrow_an_l2_default_quota(client: TestClient) -> None:
    """Verify an L2 fallback budget never applies to L1 status rows.

    Args:
        client: Fresh coordinator HTTP client supplied by the fixture.

    Returns:
        ``None`` after assertions verify L1 single and list rows retain no
        applicable quota and no explicit L2 entry.
    """
    _set_default(client, 2.5)
    _store(client, "l1-isolation", tier="l1")

    for status in (
        _status(client, "l1-isolation", tier="l1"),
        _listed_status(client, "l1-isolation", tier="l1"),
    ):
        assert status["quota_exists"] is False
        assert status["quota_limit_gb"] == 0.0


def test_default_salt_sentinel_reports_the_effective_default_quota(
    client: TestClient,
) -> None:
    """Verify the empty-salt path sentinel uses the effective default quota.

    Args:
        client: Fresh coordinator HTTP client supplied by the fixture.

    Returns:
        ``None`` after assertions verify empty-salt single and list rows use
        the configured fallback without becoming explicit entries.
    """
    _set_default(client, 2.5)
    _store(client, "")

    status = _status(client, "_default")
    assert status["cache_salt"] == ""
    assert status["quota_exists"] is False
    assert status["quota_limit_gb"] == 2.5
    listed = _listed_status(client, "")
    assert listed["quota_exists"] is False
    assert listed["quota_limit_gb"] == 2.5
