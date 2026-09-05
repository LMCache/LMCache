# SPDX-License-Identifier: Apache-2.0
"""LMCache setup script — policy-driven extension build.

Uses the strategy pattern so that each platform lives in its own file
under ``setup_extensions/build_profiles/``.  Adding a new platform requires
zero changes to this file.
"""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import sys

ROOT_DIR = Path(__file__).parent
# Ensure the project root is importable when ``setup.py`` runs inside a
# PEP 517 build subprocess (where CWD is not added to ``sys.path``).
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Third Party
from setuptools import find_packages, setup  # noqa: E402
from setuptools.command.build_py import build_py as _build_py  # noqa: E402

# First Party
from setup_extensions import BuildPolicy, BuildProfile  # noqa: E402


def _read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    reqs: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


def _load_proto_generator() -> ModuleType:
    """Load the proto generator without importing the LMCache package."""
    generator_path = (
        ROOT_DIR
        / "lmcache"
        / "v1"
        / "multiprocess"
        / "transport"
        / "grpc_impl"
        / "protos"
        / "generate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lmcache_proto_generate", generator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load gRPC proto generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BuildPyWithGrpcStubs(_build_py):
    """Generate ignored gRPC bindings before packaging LMCache."""

    def run(self) -> None:
        distribution_name = self.distribution.get_name().replace("_", "-").lower()
        if distribution_name != "lmcache-cli":
            generator = _load_proto_generator()
            generator.generate()
        super().run()


if __name__ == "__main__":
    policy = BuildPolicy()
    profile = policy.resolve_profile()
    ext_modules, cmdclass, req_file = policy.collect_extensions(profile)
    cmdclass["build_py"] = _BuildPyWithGrpcStubs

    install_requires = _read_requirements(ROOT_DIR / "requirements" / "common.txt")
    extras_require: dict[str, list[str]] = {}
    if (
        not BuildProfile.is_gpu_ext_disabled()
        and req_file is not None
        and profile is not None
    ):
        install_requires += _read_requirements(ROOT_DIR / "requirements" / req_file)
        for extra_name, extra_req_file in profile.extras_requirements().items():
            extras_require[extra_name] = _read_requirements(
                ROOT_DIR / "requirements" / extra_req_file
            )

    setup(
        packages=find_packages(exclude=("csrc",)),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        install_requires=install_requires,
        extras_require=extras_require,
    )
