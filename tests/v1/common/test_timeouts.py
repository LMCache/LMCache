# SPDX-License-Identifier: Apache-2.0
"""Import cost and observability wiring of ``lmcache.v1.common.timeouts``."""

# Standard
import json
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import time

    from lmcache.v1.common.timeouts import LMCacheTimeoutError

    prefix = "lmcache.v1.mp_observability"
    loaded = sorted(m for m in sys.modules if m.startswith(prefix))

    from lmcache.v1.mp_observability.event import EventType
    from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus

    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=10))
    events = []
    bus.subscribe(EventType.TIMEOUT_RAISED, events.append)
    bus.start()
    try:
        raise LMCacheTimeoutError("client timeout", session_id="req-1")
    except TimeoutError as exc:
        caught = type(exc).__name__
    time.sleep(0.15)
    bus.stop()
    events = [{"session_id": e.session_id, **e.metadata} for e in events]
    print(json.dumps({"loaded": loaded, "caught": caught, "events": events}))
    """
)


def test_import_skips_observability_and_an_enabled_bus_reports_once() -> None:
    # A fresh interpreter: this session has already imported mp_observability.
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True
    )
    out = json.loads(result.stdout.strip().splitlines()[-1])

    assert out["loaded"] == []
    assert out["caught"] == "LMCacheTimeoutError"
    [event] = out["events"]
    assert event["session_id"] == "req-1"
    assert event["message"] == "client timeout"
    assert event["exception_type"] == "LMCacheTimeoutError"
    assert '"<string>"' in event["stacktrace"]
    assert "timeouts.py" not in event["stacktrace"]
    assert "errors.py" not in event["stacktrace"]
