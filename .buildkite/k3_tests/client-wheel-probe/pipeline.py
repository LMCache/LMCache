# SPDX-License-Identifier: Apache-2.0
"""Upload only the relevant #6972 validation step on each existing queue."""

# Standard
import json
import sys

backend = sys.argv[1]
command = f"bash .buildkite/k3_tests/client-wheel-probe/run.sh {backend}"
step = {
    "label": f"#6972 original client wheel / {backend}",
    "command": command,
    "timeout_in_minutes": 35,
    "artifact_paths": [f"client-wheel-evidence/{backend}/*"],
}
if backend == "cuda":
    step["command"] = command + " inside"
    step["agents"] = {"queue": "k8s"}
    step["plugins"] = [
        {
            "kubernetes": {
                "podSpec": {
                    "containers": [
                        {
                            "name": "container-0",
                            "image": "vllm/vllm-openai:v0.26.0",
                            "resources": {"limits": {"nvidia.com/gpu": "1"}},
                            "volumeMounts": [
                                {"name": "dshm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "dshm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"},
                        }
                    ],
                }
            }
        }
    ]
elif backend == "musa":
    step["agents"] = {"queue": "MooreThreads"}
    step["concurrency"] = 1
    step["concurrency_group"] = "lmcache/musa-hardware-smoke"
print(json.dumps({"steps": [step]}))
