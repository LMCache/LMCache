# Reconfiguration worker identity

`ReconfigureInstanceMiddleware` binds `/reconfigure/*` requests to one worker
lifetime. Its random, opaque instance ID changes on ASGI lifespan startup. It is
neither an authentication credential nor a capacity revision.

| Header | Contract |
| --- | --- |
| `X-LMCache-Instance` | Worker identity on authenticated reconfiguration responses, including errors. |
| `X-LMCache-Expected-Instance` | Optional request precondition. A mismatch returns 409 with the current identity before invoking the handler. |

Existing callers may omit the precondition. Pool controllers obtain the identity
from their initial status request and send it on later calls. A replacement
worker's missing range must not authorize releasing the old worker's memory.

Authentication remains outside this middleware and runs first. Invalid
credentials return the ordinary 401 challenge without exposing the worker
identity or executing a handler. The controller surfaces authentication errors
before checking identity; an absent header on a successful response still means
the endpoint does not support the required fencing contract.

An unhandled handler exception before response headers are sent becomes a generic
500 with the worker identity. The detailed exception is logged, not returned.
Once a response has started, the middleware re-raises instead of sending a second
response. A 500 remains an uncertain operation and never authorizes lease release.

An HTTP error status is not proof of unmapping. In particular, an overlap error
can name an existing live mapping, and backend mapping failures can also be
reported as 400 or 409. Automatic physical release requires a successful detach
or another explicit ownership guarantee, not the status code alone.

Tests in `test_reconfigure_instance.py` cover restart preconditions, authentication
precedence, and real ASGI 500 responses for failed cleanup.
