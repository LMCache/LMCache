# SPDX-License-Identifier: Apache-2.0
# Standard

# Third Party
from fastapi import APIRouter

router = APIRouter()


@router.get("/version")
def get_version():
    try:
        # First Party
        from lmcache import _version

        return _version.__version__
    except (ImportError, AttributeError):
        return ""


@router.get("/commit_id")
def get_commit_id():
    try:
        # First Party
        from lmcache import _version

        return _version.__commit_id__
    except (ImportError, AttributeError):
        return ""
