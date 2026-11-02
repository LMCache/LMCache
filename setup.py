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
import os
import shutil
import sys

ROOT_DIR = Path(__file__).parent
# Ensure the project root is importable when ``setup.py`` runs inside a
# PEP 517 build subprocess (where CWD is not added to ``sys.path``).
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Third Party
from setuptools import find_namespace_packages, setup  # noqa: E402
from setuptools.command.build import build as _build  # noqa: E402
from setuptools.command.build_py import build_py as _build_py  # noqa: E402
from setuptools.command.egg_info import egg_info as _egg_info  # noqa: E402

# First Party
from setup_extensions import BuildPolicy, BuildProfile, build_common_cpp  # noqa: E402
from setup_extensions.distributions import (  # noqa: E402
    CLIENT,
    CLIENT_EXTENSIONS,
    SERVER,
    distribution_name,
    owns,
    role,
)

DISTRIBUTION = distribution_name()
ROLE = role(DISTRIBUTION)
SERVER_SCRIPTS = [
    "lmcache=lmcache.cli.main:main",
    "lmcache_controller=lmcache.inprocess.v1.api_server.__main__:main",
]


def _read_requirements(path: Path) -> list[str]:
    # A wheel built from an sdist that lost this file would declare no
    # dependencies, so a missing file is an error.
    if not path.exists():
        raise SystemExit(f"missing requirements file {path}")
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
        / "_proto_gen"
        / "_generate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lmcache_proto_generate", generator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load gRPC proto generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owned(path: str) -> bool:
    relative = os.path.relpath(os.path.abspath(path), ROOT_DIR.resolve())
    return owns(ROLE, Path(relative).as_posix())


class _BuildPyWithGrpcStubs(_build_py):
    """Generate ignored gRPC bindings, then keep only this distribution's files."""

    def run(self) -> None:
        # build/lib is shared by every distribution built from this checkout.
        shutil.rmtree(self.build_lib, ignore_errors=True)
        if DISTRIBUTION != "lmcache-cli":
            generator = _load_proto_generator()
            # The orchestrator schema is the server's; it imports the client's
            # P2P messages, so a server build generates both packages.
            packages = generator.GENERATED_PACKAGES
            if ROLE == "client":
                packages = (generator.GENERATED_PACKAGE,)
            for package in packages:
                generator.generate(package)
        super().run()

    def find_package_modules(
        self, package: str, package_dir: str
    ) -> list[tuple[str, str, str]]:
        modules = super().find_package_modules(package, package_dir)
        return [module for module in modules if _owned(module[2])]

    def find_data_files(self, package: str, src_dir: str) -> list[str]:
        return [
            path for path in super().find_data_files(package, src_dir) if _owned(path)
        ]


class _BuildForRole(_build):
    """Drop the client's version file from a server build once every build_*
    step, including setuptools-scm's version file writer, has run."""

    def run(self) -> None:
        super().run()
        if ROLE == "server":
            (Path(self.build_lib) / "lmcache" / "_version.py").unlink(missing_ok=True)


class _EggInfoWithSplitRequirements(_egg_info):
    """Pin the other half of the split to the version of this build."""

    def run(self) -> None:
        version = self.distribution.get_version()
        if ROLE == "server":
            requirement = f"{CLIENT}[grpc]=={version}"
            if requirement not in self.distribution.install_requires:
                self.distribution.install_requires.append(requirement)
        if ROLE == "client":
            server_extra = self.distribution.extras_require["server"]
            if f"{SERVER}=={version}" not in server_extra:
                server_extra.append(f"{SERVER}=={version}")
        super().run()


if __name__ == "__main__":
    if ROLE == "client":
        # The client ships only the common CPU extension, never a GPU profile
        # or storage backends.
        profile, req_file = None, None
        if BuildProfile.is_building_sdist() or BuildProfile.is_native_ext_disabled():
            ext_modules: list = []
            cmdclass: dict = {}
        else:
            ext_modules, cmdclass = build_common_cpp(None, names=CLIENT_EXTENSIONS)
    else:
        policy = BuildPolicy()
        profile = policy.resolve_profile()
        ext_modules, cmdclass, req_file = policy.collect_extensions(profile)
        if ROLE == "server":
            ext_modules = [e for e in ext_modules if e.name not in CLIENT_EXTENSIONS]
    cmdclass["build"] = _BuildForRole
    cmdclass["build_py"] = _BuildPyWithGrpcStubs
    cmdclass["egg_info"] = _EggInfoWithSplitRequirements

    entry_points: dict = {}
    if ROLE == "client":
        install_requires = _read_requirements(ROOT_DIR / "requirements" / "client.txt")
        extras_require = {
            "grpc": _read_requirements(ROOT_DIR / "requirements" / "client_grpc.txt"),
            "blake3": _read_requirements(
                ROOT_DIR / "requirements" / "client_blake3.txt"
            ),
            "server": [],
        }
    else:
        install_requires = _read_requirements(ROOT_DIR / "requirements" / "common.txt")
        # Platform-neutral extras; the build profile adds its own below.
        extras_require = {
            "kafka": _read_requirements(ROOT_DIR / "requirements" / "kafka.txt"),
            "raw-block": ["lmcache-rust-raw-block-io"],
            "grpc-native": _read_requirements(
                ROOT_DIR / "requirements" / "grpc_native.txt"
            ),
        }
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
    if ROLE == "server":
        entry_points["console_scripts"] = SERVER_SCRIPTS

    setup(
        # Namespace discovery: a server-only tree has no lmcache/__init__.py.
        packages=find_namespace_packages(include=["lmcache", "lmcache.*"]),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points=entry_points,
    )
