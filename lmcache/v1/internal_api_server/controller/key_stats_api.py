# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Third Party
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class InstanceKeyStats(BaseModel):
    instance_id: str
    key_count: int
    worker_count: int


class KeyStatsResponse(BaseModel):
    total_key_count: int
    total_instance_count: int
    total_worker_count: int
    instances: List[InstanceKeyStats]


@router.get("/controller/key-stats")
async def get_key_stats(request: Request):
    """
    Get key statistics across all instances and workers.

    Returns:
        - Total key count across all instances
        - Total instance count
        - Total worker count
        - Key count per instance
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
        registry = reg_controller.registry

        # Get total key count
        total_key_count = registry.get_total_kv_count()

        # Get instances and their key counts
        instances = []
        total_instance_count = 0
        total_worker_count = 0

        for ip_instances in registry.instances.values():
            for instance_id, instance_node in ip_instances.items():
                total_instance_count += 1
                instance_key_count = 0

                # Calculate key count for this instance
                for worker_node in instance_node.workers.values():
                    total_worker_count += 1
                    instance_key_count += worker_node.get_kv_count()

                instances.append(
                    InstanceKeyStats(
                        instance_id=instance_id,
                        key_count=instance_key_count,
                        worker_count=len(instance_node.workers),
                    )
                )

        return KeyStatsResponse(
            total_key_count=total_key_count,
            total_instance_count=total_instance_count,
            total_worker_count=total_worker_count,
            instances=instances,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from None
