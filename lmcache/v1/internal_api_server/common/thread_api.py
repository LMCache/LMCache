# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import threading

# Third Party
from fastapi import APIRouter, Query
from starlette.requests import Request

router = APIRouter()


@router.get("/threads")
async def get_threads(
    request: Request,
    name: Optional[str] = Query(
        None, description="Filter by thread name (fuzzy match)"
    ),
    thread_id: Optional[int] = Query(None, description="Filter by thread ID"),
):
    """Return information about active threads with optional filtering"""
    threads = threading.enumerate()

    filtered_threads = []
    for t in threads:
        # Apply filters
        if name and name.lower() not in t.name.lower():
            continue
        if thread_id and t.ident != thread_id:
            continue
        filtered_threads.append(t)

    thread_info = []

    for t in filtered_threads:
        # Get thread details - match the structure from api_server/__main__.py
        thread_data = {
            "thread_id": t.ident,
            "name": t.name,
            "state": "running" if t.is_alive() else "terminated",
            "function_name": str(t),  # This will show target function if available
            "cpu_time": 0,  # Placeholder - would need more complex tracking
            "memory_usage": 0,  # Placeholder - would need more complex tracking
        }
        thread_info.append(thread_data)

    # If no threads found, return some default info
    if not thread_info:
        thread_info = [
            {
                "thread_id": 1,
                "name": "MainThread",
                "state": "running",
                "function_name": "API Server",
                "cpu_time": 0,
                "memory_usage": 0,
            }
        ]

    return thread_info
