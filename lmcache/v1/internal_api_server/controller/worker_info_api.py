# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, Optional

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter()


class WorkerInfoResponse(BaseModel):
    instance_id: str
    worker_id: int
    ip: str
    port: int
    distributed_url: Optional[str]
    registration_time: float
    last_heartbeat_time: float


class WorkerListResponse(BaseModel):
    workers: list[WorkerInfoResponse]
    total_count: int


@router.get("/controller/workers")
async def get_workers(
    request: Request,
    instance_id: Annotated[Optional[str], Query()] = None,
    worker_id: Annotated[Optional[int], Query()] = None,
):
    """
    Get worker information with flexible query parameters.

    - No parameters: List all registered workers across all instances
    - instance_id only: List all workers for a specific instance
    - instance_id and worker_id: Get detailed info about a specific worker

    Args:
        instance_id: Optional instance ID to filter workers
        worker_id: Optional worker ID to get specific worker details
    """
    try:
        controller_manager = getattr(
            request.app.state, "lmcache_controller_manager", None
        )

        if controller_manager is None:
            raise HTTPException(
                status_code=503, detail="Controller manager not available"
            )

        reg_controller = controller_manager.reg_controller

        # Case 1: Get specific worker by instance_id and worker_id
        if instance_id is not None and worker_id is not None:
            worker_key = (instance_id, worker_id)
            if worker_key not in reg_controller.worker_info_mapping:
                raise HTTPException(
                    status_code=404, detail=f"Worker {worker_key} not found"
                )

            worker_info = reg_controller.worker_info_mapping[worker_key]
            return WorkerInfoResponse(
                instance_id=worker_info.instance_id,
                worker_id=worker_info.worker_id,
                ip=worker_info.ip,
                port=worker_info.port,
                distributed_url=worker_info.distributed_url,
                registration_time=worker_info.registration_time,
                last_heartbeat_time=worker_info.last_heartbeat_time,
            )

        # Case 2: Get all workers for a specific instance
        elif instance_id is not None:
            worker_ids = reg_controller.get_workers(instance_id)
            if not worker_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"No workers found for instance {instance_id}",
                )

            workers = []
            for worker_id in worker_ids:
                worker_key = (instance_id, worker_id)
                if worker_key in reg_controller.worker_info_mapping:
                    worker_info = reg_controller.worker_info_mapping[worker_key]
                    workers.append(
                        WorkerInfoResponse(
                            instance_id=worker_info.instance_id,
                            worker_id=worker_info.worker_id,
                            ip=worker_info.ip,
                            port=worker_info.port,
                            distributed_url=worker_info.distributed_url,
                            registration_time=worker_info.registration_time,
                            last_heartbeat_time=worker_info.last_heartbeat_time,
                        )
                    )

            return WorkerListResponse(workers=workers, total_count=len(workers))

        # Case 3: Get all workers across all instances
        else:
            workers = []
            for worker_key, worker_info in reg_controller.worker_info_mapping.items():
                workers.append(
                    WorkerInfoResponse(
                        instance_id=worker_info.instance_id,
                        worker_id=worker_info.worker_id,
                        ip=worker_info.ip,
                        port=worker_info.port,
                        distributed_url=worker_info.distributed_url,
                        registration_time=worker_info.registration_time,
                        last_heartbeat_time=worker_info.last_heartbeat_time,
                    )
                )

            return WorkerListResponse(workers=workers, total_count=len(workers))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from None
