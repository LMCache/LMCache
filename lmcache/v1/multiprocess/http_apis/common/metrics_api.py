# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter
from prometheus_client import REGISTRY, generate_latest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.get("/metrics")
async def get_metrics(request: Request):
    """
    Provide Prometheus metrics data
    """
    metrics_data = generate_latest(REGISTRY)
    return PlainTextResponse(content=metrics_data, media_type="text/plain")


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Compatibility no-op: returns ``ok`` without resetting any metric.

    MP metrics are OpenTelemetry cumulative instruments, which cannot be
    reset in place. Diff ``GET /metrics`` scrapes instead.
    """
    # TODO(baoloongmao): implement reset metrics later.
    return PlainTextResponse(content="ok", media_type="text/plain")
