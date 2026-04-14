# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM integration, including platform extension hooks."""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import (
    LMCacheConnectorV1Dynamic,
)
from lmcache.integration.vllm.lmcache_connector_v1 import (
    LMCacheConnectorV1Impl as LMCacheConnectorV1ImplV1,
)
from lmcache.integration.vllm.lmcache_connector_v1_085 import (
    LMCacheConnectorV1Dynamic as LMCacheConnectorV1Dynamic085,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.config import LMCacheEngineConfig


class TestServiceFactoryClsHook:
    """``LMCacheConnectorV1Impl.service_factory_cls`` extension point."""

    def test_default_is_vllm_service_factory(self) -> None:
        assert LMCacheConnectorV1Impl.service_factory_cls is VllmServiceFactory

    def test_subclass_can_override(self) -> None:
        class FakeServiceFactory(VllmServiceFactory):
            pass

        class PlatformImpl(LMCacheConnectorV1Impl):
            service_factory_cls = FakeServiceFactory

        assert PlatformImpl.service_factory_cls is FakeServiceFactory
        assert LMCacheConnectorV1Impl.service_factory_cls is VllmServiceFactory

    def test_overridden_factory_is_used_in_init(self) -> None:
        fake_factory_cls = MagicMock(spec=type)

        class PlatformImpl(LMCacheConnectorV1Impl):
            service_factory_cls = fake_factory_cls

        vllm_config = MagicMock()
        role = MagicMock()
        role.name = "WORKER"
        parent = MagicMock()

        with (
            patch(
                "lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config"
            ) as mock_get_config,
            patch(
                "lmcache.integration.vllm.vllm_v1_adapter.LMCacheManager"
            ) as mock_manager_cls,
            patch.object(LMCacheConnectorV1Impl, "_apply_extra_config"),
            patch.object(LMCacheConnectorV1Impl, "_init_connector_state"),
            patch.object(LMCacheConnectorV1Impl, "_setup_metrics"),
        ):
            mock_get_config.return_value = LMCacheEngineConfig.from_defaults()
            PlatformImpl(vllm_config, role, parent)

        fake_factory_cls.assert_called_once()
        factory_arg = mock_manager_cls.call_args[0][1]
        assert factory_arg is fake_factory_cls.return_value


class TestImplClsHook:
    """``LMCacheConnectorV1Dynamic.impl_cls`` extension point."""

    def test_default_is_lmcache_connector_v1_impl(self) -> None:
        assert LMCacheConnectorV1Dynamic.impl_cls is LMCacheConnectorV1ImplV1

    def test_default_is_lmcache_connector_v1_impl_on_085_shim(self) -> None:
        assert LMCacheConnectorV1Dynamic085.impl_cls is LMCacheConnectorV1ImplV1

    def test_subclass_can_override(self) -> None:
        class FakeImpl(LMCacheConnectorV1Impl):
            pass

        class PlatformDynamic(LMCacheConnectorV1Dynamic):
            impl_cls = FakeImpl

        assert PlatformDynamic.impl_cls is FakeImpl
        assert LMCacheConnectorV1Dynamic.impl_cls is LMCacheConnectorV1ImplV1

    def test_overridden_impl_is_instantiated(self) -> None:
        fake_impl_cls = MagicMock()

        class PlatformDynamic(LMCacheConnectorV1Dynamic):
            impl_cls = fake_impl_cls

        vllm_config = MagicMock()
        role = MagicMock()

        with patch(
            "lmcache.integration.vllm.lmcache_connector_v1.KVConnectorBase_V1.__init__",
            return_value=None,
        ):
            connector = PlatformDynamic(vllm_config, role)

        fake_impl_cls.assert_called_once_with(vllm_config, role, connector)
