# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

logger = init_logger(__name__)


class CuObjectS3ConnectorAdapter(ConnectorAdapter):
    """Adapter for RDMA-accelerated S3 connections via NVIDIA cuObject.

    Matches URLs of the form ``cuobj+s3://<bucket>.<region>...``
    or standard ``s3://`` URLs when ``extra_config["enable_cuobject"]``
    is set to ``true``.
    """

    def __init__(self) -> None:
        super().__init__("cuobj+s3://")

    def create_connector(
        self, context: ConnectorContext
    ) -> RemoteConnector:
        # Local
        from .cuobject_s3_connector import CuObjectS3Connector

        config = context.config
        assert config is not None

        extra_config = (
            config.extra_config if config.extra_config is not None else {}
        )

        save_chunk_meta = bool(
            extra_config.get("save_chunk_meta", False)
        )
        assert not save_chunk_meta, (
            "save_chunk_meta must be False for cuObject+S3"
        )

        s3_num_io_threads = int(
            extra_config.get("s3_num_io_threads", 64)
        )
        s3_prefer_http2 = bool(
            extra_config.get("s3_prefer_http2", True)
        )
        s3_region = extra_config.get("s3_region", None)
        assert s3_region is not None, "s3_region is required"
        s3_region = str(s3_region)
        s3_enable_s3express = bool(
            extra_config.get("s3_enable_s3express", False)
        )
        disable_tls = bool(extra_config.get("disable_tls", False))
        aws_access_key_id = extra_config.get("aws_access_key_id", None)
        aws_secret_access_key = extra_config.get(
            "aws_secret_access_key", None
        )

        # cuObject-specific config
        cuobj_nic_device = extra_config.get("cuobj_nic_device", None)
        cuobj_lib_path = extra_config.get("cuobj_lib_path", None)

        if context.metadata is None:
            raise ValueError(
                "metadata is required for CuObjectS3Connector"
            )

        # Strip the "cuobj+" prefix to get the real S3 endpoint
        s3_endpoint = context.url
        if s3_endpoint.startswith("cuobj+"):
            s3_endpoint = s3_endpoint[len("cuobj+"):]

        logger.info(
            f"Creating cuObject S3 connector for URL: {context.url} "
            f"(endpoint: {s3_endpoint})"
        )

        return CuObjectS3Connector(
            s3_endpoint=s3_endpoint,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            s3_num_io_threads=s3_num_io_threads,
            s3_prefer_http2=s3_prefer_http2,
            s3_region=s3_region,
            s3_enable_s3express=s3_enable_s3express,
            disable_tls=disable_tls,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            cuobj_nic_device=cuobj_nic_device,
            cuobj_lib_path=cuobj_lib_path,
        )
