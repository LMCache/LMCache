# SPDX-License-Identifier: Apache-2.0
"""Model-specific chunk-size and token-hasher profiles."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
import threading

# First Party
from lmcache.v1.common.token_hasher import TokenHasher


@dataclass
class _ModelChunkProfile:
    """Mutable state owned by :class:`ModelChunkProfileManager`."""

    configured_chunk_size: int
    """Configured lower bound used to negotiate the runtime chunk size."""
    chunk_size: int
    """Active chunk size after any required alignment adjustment."""
    hash_algorithm: str
    """Hash algorithm used by this profile's token hasher."""
    token_hasher: TokenHasher
    """Token hasher constructed with the active chunk size."""
    chunk_size_immutable: bool = False
    """Whether the runtime chunk size and token hasher are immutable."""
    finalized: bool = False
    """Whether all chunk-size bind listeners completed successfully."""


class ModelChunkProfileManager:
    """Own chunk-size and token-hasher state for the default and model profiles.

    The default profile preserves the current single-chunk-size server behavior.
    Model profiles are available for integrations to bind in a later stage; an
    unconfigured model starts from the default configured chunk size but receives
    independent mutable state and its own :class:`TokenHasher`.

    Args:
        default_chunk_size: Configured chunk size for the default profile and for
            models without an explicit override.
        hash_algorithm: Default hash algorithm inherited by model profiles
            without an explicit override.
        model_chunk_sizes: Optional configured chunk sizes keyed by model name.
        model_hash_algorithms: Optional hash algorithms keyed by model name.

    Raises:
        ValueError: If a chunk size is not positive or a model name is empty.
    """

    def __init__(
        self,
        default_chunk_size: int,
        hash_algorithm: str,
        model_chunk_sizes: Mapping[str, int] | None = None,
        model_hash_algorithms: Mapping[str, str] | None = None,
    ) -> None:
        self._validate_chunk_size(default_chunk_size)
        chunk_size_overrides = dict(model_chunk_sizes or {})
        hash_algorithm_overrides = dict(model_hash_algorithms or {})
        model_names = chunk_size_overrides.keys() | hash_algorithm_overrides.keys()
        for model_name in model_names:
            self._validate_model_name(model_name)
        for chunk_size in chunk_size_overrides.values():
            self._validate_chunk_size(chunk_size)

        self._default_chunk_size = default_chunk_size
        self._default_hash_algorithm = hash_algorithm
        self._profiles: dict[str | None, _ModelChunkProfile] = {
            None: self._create_profile(default_chunk_size, hash_algorithm)
        }
        self._profiles.update(
            {
                model_name: self._create_profile(
                    chunk_size_overrides.get(model_name, default_chunk_size),
                    hash_algorithm_overrides.get(model_name, hash_algorithm),
                )
                for model_name in sorted(model_names)
            }
        )
        self._lock = threading.Lock()

    def get_hash_algorithm_name(self, model_name: str | None = None) -> str:
        """Return the hash algorithm configured for a profile.

        Args:
            model_name: Model profile to read, or ``None`` for the default
                profile.

        Returns:
            The profile's hash algorithm.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            return self._get_locked(model_name).hash_algorithm

    def get_configured_chunk_size(self, model_name: str | None = None) -> int:
        """Return the configured chunk size for a profile.

        Args:
            model_name: Model profile to read, or ``None`` for the default
                profile.

        Returns:
            The profile's configured chunk size.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            return self._get_locked(model_name).configured_chunk_size

    def get_chunk_size(self, model_name: str | None = None) -> int:
        """Return the current chunk size for a profile.

        Args:
            model_name: Model profile to read, or ``None`` for the default
                profile.

        Returns:
            The profile's current chunk size.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            return self._get_locked(model_name).chunk_size

    def get_token_hasher(self, model_name: str | None = None) -> TokenHasher:
        """Return the token hasher bound to a profile's current chunk size.

        Args:
            model_name: Model profile to read, or ``None`` for the default
                profile.

        Returns:
            The profile's token hasher.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            return self._get_locked(model_name).token_hasher

    def is_chunk_size_immutable(self, model_name: str | None = None) -> bool:
        """Return whether a profile's chunk size and token hasher are immutable.

        Args:
            model_name: Model profile to read, or ``None`` for the default
                profile.

        Returns:
            ``True`` after the profile's chunk size becomes immutable.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            return self._get_locked(model_name).chunk_size_immutable

    def is_finalized(self, model_name: str | None = None) -> bool:
        """Return whether all listeners have accepted the immutable chunk size.

        Args:
            model_name: Model profile to read, or ``None`` for the default
                profile.

        Returns:
            ``True`` after profile finalization.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            return self._get_locked(model_name).finalized

    def ensure_model_profile(self, model_name: str) -> None:
        """Create a model profile when one is not already registered.

        An unconfigured model inherits the manager's default chunk size and
        hash algorithm. Repeated calls for the same model are a no-op.

        Args:
            model_name: Non-empty model-profile key.

        Raises:
            ValueError: If ``model_name`` is empty.
        """
        self._validate_model_name(model_name)
        with self._lock:
            if model_name not in self._profiles:
                self._profiles[model_name] = self._create_profile(
                    self._default_chunk_size,
                    self._default_hash_algorithm,
                )

    def bind_chunk_size(
        self, chunk_size: int, model_name: str | None = None
    ) -> TokenHasher:
        """Bind a profile to ``chunk_size`` and replace its token hasher.

        Args:
            chunk_size: Positive chunk size to bind.
            model_name: Model profile to update, or ``None`` for the default
                profile.

        Returns:
            The newly bound token hasher.

        Raises:
            ValueError: If ``chunk_size`` is not positive or the profile's
                runtime size is already immutable, or if the named model
                profile is not registered.
        """
        self._validate_chunk_size(chunk_size)
        with self._lock:
            profile = self._get_locked(model_name)
            if profile.chunk_size_immutable:
                raise ValueError("cannot bind an immutable chunk-size profile")
            profile.chunk_size = chunk_size
            profile.token_hasher = self._create_token_hasher(
                chunk_size, profile.hash_algorithm
            )
            return profile.token_hasher

    def mark_chunk_size_immutable(self, model_name: str | None = None) -> None:
        """Make a profile's current chunk size and token hasher immutable.

        Args:
            model_name: Model profile to update, or ``None`` for the default
                profile.

        Raises:
            ValueError: If the named model profile is not registered.
        """
        with self._lock:
            self._get_locked(model_name).chunk_size_immutable = True

    def mark_finalized(self, model_name: str | None = None) -> None:
        """Mark all listeners as having accepted the immutable chunk size.

        Args:
            model_name: Model profile to update, or ``None`` for the default
                profile.

        Raises:
            ValueError: If the named model profile is not registered or its
                runtime chunk size is still mutable.
        """
        with self._lock:
            profile = self._get_locked(model_name)
            if not profile.chunk_size_immutable:
                raise ValueError("cannot finalize a mutable chunk-size profile")
            profile.finalized = True

    def _get_locked(self, model_name: str | None) -> _ModelChunkProfile:
        """Return an existing profile while ``self._lock`` is held."""
        if model_name is not None:
            self._validate_model_name(model_name)
        profile = self._profiles.get(model_name)
        if profile is None:
            raise ValueError(f"no chunk-size profile registered for {model_name!r}")
        return profile

    def _create_profile(
        self, configured_chunk_size: int, hash_algorithm: str
    ) -> _ModelChunkProfile:
        """Create a mutable profile at its configured chunk size."""
        return _ModelChunkProfile(
            configured_chunk_size=configured_chunk_size,
            chunk_size=configured_chunk_size,
            hash_algorithm=hash_algorithm,
            token_hasher=self._create_token_hasher(
                configured_chunk_size, hash_algorithm
            ),
        )

    @staticmethod
    def _create_token_hasher(chunk_size: int, hash_algorithm: str) -> TokenHasher:
        """Create a token hasher for a profile."""
        return TokenHasher(
            chunk_size=chunk_size,
            hash_algorithm=hash_algorithm,
        )

    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> None:
        """Reject a non-positive configured or bound chunk size."""
        if chunk_size < 1:
            raise ValueError(f"chunk size must be positive, got {chunk_size}")

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        """Reject an empty model-profile key."""
        if not model_name:
            raise ValueError("model name must not be empty")
