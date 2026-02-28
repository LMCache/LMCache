# AGENTS.md

- Be aware that the default branch is `dev` and the upstream code is in `https://github.com/LMCache/LMCache`.
- Make sure license header (`# SPDX-License-Identifier: Apache-2.0`) is present on all Python files.
- LMCache mainly serves distributed LLM inference. Make sure to handle edge cases and make the code robust.
- Make sure thread safety is maintained for shared data structures.
- Make sure CUDA/GPU resources are properly managed (allocated, freed, synchronized).
- Module-level helpers are placed at the top; private methods at the end of the class.
- New features, bug fixes should include **docstrings, tests and documentations**. The documentation should be concrete and concise.
- Write tests against the **public interface and docstring contract**, not the implementation. Test as if you don't know the internals — verify that behavior matches what the docstring describes.
- Avoid accessing private members in tests unless absolutely needed.
- Ensure existing tests still pass before submitting changes. The standard test suite is: `pytest -xvs --ignore=tests/disagg --ignore=tests/v1/test_nixl_storage.py  --ignore=tests/v1/multiprocess/  --ignore=tests/v1/distributed/  --ignore=tests/skipped  --ignore=tests/v1/storage_backend/test_eic.py`
- Ensure pre-commit works before submitting changes.
