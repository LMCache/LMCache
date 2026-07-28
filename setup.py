# SPDX-License-Identifier: Apache-2.0
"""LMCache setup script — policy-driven extension build.

Uses the strategy pattern so that each platform lives in its own file
under ``setup_extensions/build_profiles/``.  Adding a new platform requires
zero changes to this file.
"""

# Standard
from pathlib import Path
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


class _BuildPyWithProtoStubs(_build_py):
    """Generate gRPC stubs before packaging when grpcio-tools is present."""

    def run(self) -> None:
        # Avoid importing _proto_gen, whose initialization also loads stubs.
        gen_path = (
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
            "_lmcache_proto_generate", gen_path
        )
        assert spec is not None and spec.loader is not None, f"cannot load {gen_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            # Third Party
            import grpc_tools.protoc  # noqa: F401
        except ImportError:
            print(
                "warning: grpcio-tools not available at build time; "
                "skipping gRPC stub generation. Stubs will be produced "
                "lazily on first import at runtime.",
                file=sys.stderr,
            )
        else:
            rc = module.main()
            if rc != 0:
                raise SystemExit(
                    "failed to generate gRPC stubs during build; check the output above"
                )
        super().run()


if __name__ == "__main__":
    policy = BuildPolicy()
    profile = policy.resolve_profile()
    ext_modules, cmdclass, req_file = policy.collect_extensions(profile)
    cmdclass["build_py"] = _BuildPyWithProtoStubs

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
