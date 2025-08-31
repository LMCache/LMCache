# SPDX-License-Identifier: Apache-2.0
# Standard

# Third Party
from fastapi import APIRouter

router = APIRouter()

try:
    # First Party
    from lmcache import _version  # type: ignore[attr-defined]

    VERSION = getattr(_version, "__version__", "")
    COMMIT_ID = getattr(_version, "__commit_id__", "")
except ImportError:
    VERSION = ""
    COMMIT_ID = ""


@router.get("/lmc_version")
async def get_lmc_version():
    return VERSION


@router.get("/commit_id")
async def get_commit_id():
    return COMMIT_ID


@router.get("/version")
async def get_version():
    version = await get_lmc_version()
    commit_id = await get_commit_id()
    version_display = version if version else "NA"
    commit_id_display = commit_id if commit_id else "NA"
    return f"{version_display}-{commit_id_display}"
