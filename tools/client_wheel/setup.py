# SPDX-License-Identifier: Apache-2.0
"""Build glue for lmcache-client. Runs only in a staging root made by build.py."""

# Standard
from pathlib import Path
import importlib.util
import json
import os
import sys

ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / ".git").exists() or not (ROOT_DIR / "STAGING_MANIFEST.json").exists():
    raise SystemExit("build lmcache-client with tools/client_wheel/build.py")
if "sdist" in sys.argv:
    raise SystemExit("lmcache-client is published as wheels only")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Third Party
from setuptools import setup  # noqa: E402
from setuptools.command.build_py import build_py as _build_py  # noqa: E402

# First Party
from setup_extensions import BuildPolicy, BuildProfile  # noqa: E402
from setup_extensions.common_cpp import build_common_cpp  # noqa: E402

STAGING = json.loads((ROOT_DIR / "STAGING_MANIFEST.json").read_text())


class _BuildPyWithClientStubs(_build_py):
    """Generate only the client transport's gRPC bindings."""

    def run(self) -> None:
        path = ROOT_DIR / "lmcache/v1/transport/grpc/_proto_gen/_generate.py"
        spec = importlib.util.spec_from_file_location("_lmcache_proto_generate", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load gRPC proto generator: {path}")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        generator.generate(generator.GENERATED_PACKAGE)
        super().run()


def _extensions() -> tuple[list, dict]:
    """lmcache_native plus the CUDA ops; never the Redis/FS or L2 backends."""
    if BuildProfile.is_native_ext_disabled():
        return [], {}
    profile = BuildPolicy().resolve_profile()
    if profile is not None and profile.name != "cuda":
        raise SystemExit(f"lmcache-client does not support the {profile.name} build")
    if profile is not None and os.environ.get("LMCACHE_CUDA_MAJOR", "13") != "13":
        raise SystemExit("lmcache-client's [cuda] extra targets CUDA 13")
    allowed = set(STAGING["native_allow"])
    common, cmdclass = build_common_cpp(profile)
    ext_modules = [ext for ext in common if ext.name in allowed]
    if profile is not None:
        platform_ext, platform_cmd = profile.build()
        unexpected = [ext.name for ext in platform_ext if ext.name not in allowed]
        if unexpected:
            raise SystemExit(f"extensions outside native_allow: {unexpected}")
        ext_modules += platform_ext
        cmdclass.update(platform_cmd)
    return ext_modules, cmdclass


if __name__ == "__main__":
    ext_modules, cmdclass = _extensions()
    cmdclass["build_py"] = _BuildPyWithClientStubs
    setup(
        packages=STAGING["packages"],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=False,
    )
