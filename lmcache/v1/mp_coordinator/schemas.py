# SPDX-License-Identifier: Apache-2.0
"""Shared request/response schemas for the mp coordinator REST API.

These Pydantic models are the wire contract between the coordinator and mp
servers. The coordinator uses them to validate requests and shape responses; an
mp server (when it registers) imports the same models to build its request
bodies and parse replies, so both sides agree on the schema in one place.
"""

# Standard
from typing import Annotated

# Third Party
from pydantic import BaseModel, Field, StringConstraints


class RegisterRequest(BaseModel):
    """Body of a ``POST /instances`` registration request.

    Attributes:
        instance_id: Identifier of the mp server. Optional -- if empty (or
            whitespace-only), the coordinator generates one and returns it.
        ip: IP/host of the mp server's HTTP server. Whitespace is stripped and a
            blank value is rejected, since the coordinator calls this address.
        http_port: Port of the mp server's HTTP server, which the coordinator
            calls to push work to this instance.
        metadata: Free-form registration hints.
    """

    instance_id: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    ip: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    http_port: int = Field(ge=1, le=65535)
    metadata: dict[str, str] = Field(default_factory=dict)


class RegisterResponse(BaseModel):
    """Reply to a successful ``POST /instances``.

    Attributes:
        instance_id: The registered instance's id.
        re_registered: ``True`` if this replaced an existing registration.
    """

    instance_id: str
    re_registered: bool


class HeartbeatResponse(BaseModel):
    """Reply to a successful ``PUT /instances/{id}/heartbeat``.

    Attributes:
        instance_id: The instance whose heartbeat was recorded.
    """

    instance_id: str
