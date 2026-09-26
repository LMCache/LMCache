# AGENTS.md

Guidelines for AI coding agents (Copilot, Cursor, Claude Code, etc.) working in this repository.

## AGENTS.md Ownership

`AGENTS.md` is a human-owned policy file.
Agents must not modify this file unless the user explicitly requests changes to `AGENTS.md`.
If an agent discovers recurring guidance or environment notes, it should report them in chat instead of editing this file.

## Project Overview

LMCache is a KV cache management engine for LLM serving that reduces Time To First Token (TTFT) and increases throughput. It stores KV caches across multiple tiers (GPU, CPU, disk, S3) and integrates with vLLM and SGLang.

## Repository

The default branch is `dev`. Base all new branches and pull requests against `dev`.

All commits in pull requests must include a DCO `Signed-off-by` trailer. Use
`git commit -s` for new commits; if an existing commit is missing the trailer,
amend or rebase to add it before pushing.

## Python Environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies:

```bash
# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install torch               # pre-requisite for CUDA extensions
uv pip install -e . --no-build-isolation
```

## Build & Install

```bash
# Standard install with CUDA extensions (requires torch pre-installed)
pip install -e . --no-build-isolation

# Source-only (no native extensions)
NO_NATIVE_EXT=1 pip install -e .

# CPU-only (common C++ extensions, no GPU backend)
NO_GPU_EXT=1 pip install -e . --no-build-isolation

# HIP/ROCm build
BUILD_WITH_HIP=1 pip install -e .
```

## Testing

### Running Tests

```bash
# Run standard test suite (mirrors CI)
pytest -xvs --ignore=tests/disagg \
 --ignore=tests/v1/multiprocess/ \
 --ignore=tests/v1/distributed/ \
 --ignore=tests/skipped \
 --ignore=tests/v1/storage_backend/test_eic.py

# Run a single test file
pytest -xvs tests/v1/test_cache_engine.py

# Run a single test
pytest -xvs tests/v1/test_cache_engine.py::test_function_name
```

Test dependencies: `uv pip install -r requirements/test.txt`

Pytest marker: `@pytest.mark.no_shared_allocator` disables the shared-allocator monkeypatch for a test.

### Testing Practices

- Write tests against the **public interface and docstring contract**, not the implementation. Test as if you don't know the internals — verify that behavior matches what the docstring describes.
- Avoid accessing private members in tests unless strongly needed.
- All new features and bug fixes should include corresponding tests.
- Ensure existing tests still pass before submitting changes.

## Linting & Code Quality

```bash
# Run all checks (mirrors CI exactly)
pre-commit run --all-files

# Individual tools
ruff check .              # Lint (E, F, B, SLF rules)
ruff format .             # Format (line-length 88)
isort .                   # Import sorting (black profile, from_first=true)
mypy --config-file=pyproject.toml   # Type checking
codespell --toml pyproject.toml     # Spell checking
```

The pre-commit config includes Rust hooks, so the run fails when `cargo`,
`rustfmt`, or `clippy` are missing (`rust-clippy` also fails on macOS while
building `io-uring`). For non-Rust changes, run
`SKIP=rust-fmt,rust-clippy pre-commit run --all-files`.

C++/CUDA files use clang-format (Google style, 80-col). Rust code in `rust/` uses `cargo fmt` and `cargo clippy`.

All Python files require an `# SPDX-License-Identifier: Apache-2.0` header as the first line.

### Import Ordering

`isort` enforces the section-heading convention (`# Standard` / `# Third Party` /
`# First Party` / `# Local`); see `docs/coding_standards.md` Section 7.2.

### SLF (Private Member Access)

SLF lint rules are currently enforced by CI only in `lmcache/v1/multiprocess/` and `lmcache/v1/distributed/`. However, **all new code should follow SLF discipline regardless of location** — never access private members (prefixed with `_`) of other classes. Treat this as a project-wide coding standard for any new or modified code.

## Coding Conventions

The canonical standard is **`docs/coding_standards.md`** — read it before writing
or reviewing code. Highlights below; on any conflict, the standard wins.

### Type Hints

All functions and methods must have type hints for their arguments and return values.

### Docstrings

Every public function and method must have a clear docstring covering:
- What the function does
- Arguments (with types and descriptions)
- Return values
- Raised exceptions (if any)
- Additional notes when behavior is non-obvious

### Writing Documentation

LMCache has three documentation surfaces:

1. **User-facing docs** (`docs/source/`, reStructuredText, Sphinx-built). When adding
   or modifying user docs, place them in the appropriate subdirectory under
   `docs/source/` (e.g., `developer_guide/`, `getting_started/`, `kv_cache/`) and link
   new pages from a `toctree` so they appear in the built site.
   Do not edit `docs/source/locale/zh_CN/`; it is generated by the documentation
   localization workflow.
2. **Design docs** (`docs/design/`, Markdown). **`docs/design/` mirrors the `lmcache/`
   package tree** — a design doc for `lmcache/<path>/` lives at `docs/design/<path>/`.
   For example, `lmcache/v1/distributed/l2_adapters/` → `docs/design/v1/distributed/l2_adapters/`.
   When investigating a module, always check the mirrored `docs/design/<path>/` first
   for design rationale, contracts, and extension guides. When adding a design doc,
   place it at the path matching the module it describes; when touching existing
   docs, find them at the mirrored location. See `docs/design/README.md` for the
   full convention.
3. **Module READMEs** (`README.md` next to code). Stay in place as user-entry-points;
   they are symlinked from `docs/design/<path>/README.md`. Do not move them.

When writing or updating documentation, follow these principles:

- **Be concrete and concise.** State exactly what something does and why — avoid vague, hand-wavy descriptions. One precise sentence beats a paragraph of generalities.
- **Include examples.** Show concrete code snippets, command invocations, or data formats so the reader can immediately see how things work in practice.
- **Explain the _why_, not just the _what_.** Briefly state the design motivation or trade-off behind a decision so readers understand the reasoning.
- **Use diagrams or short flows for complex interactions.** When multiple components interact (e.g., the multiprocess pipeline), a short step-by-step flow or ASCII diagram is far clearer than prose alone.
- **Keep scope focused.** Each document should have a clear audience and purpose. Don't mix user-facing setup guides with internal architecture notes.

#### Building and verifying docs

After documentation changes, the Sphinx build must pass **without errors or
warnings**: `pip install -r requirements/docs.txt` (one-time), then
`cd docs && make clean && make html`. Preview with
`python -m http.server -d build/html/`.

### Encapsulation

Never access private members (prefixed with `_`) of other classes. Interact only through their public APIs.

### Code Organization

- **Module-level helper functions** go at the top of the file (after imports, before classes).
- **Private/helper methods** within a class go at the end of the class, after all public methods.

## AI Coding Discipline

Hard rules (details: `docs/coding_standards.md` Section 10):

- No handling for failures that cannot occur; no speculative `None` checks or fallback defaults.
- Narrow `try/except` with specific types; `logger.exception` inside `except` blocks.
- Comments explain why, never narrate the change; TODOs carry a GitHub handle.
- Reuse existing helpers; no speculative config options or abstractions.

## Code Review Rules

When reviewing code (or self-checking before submitting), apply
`docs/coding_standards.md`: review focus (Section 9.1), severity calibration
(Section 9.2), and AI-generated code signals (Section 9.5). That file is the
canonical standard; this file intentionally does not duplicate it.

When asked to review a PR, use the `pr-review` skill; before creating a PR,
self-check with the `pre-pr-check` skill (both under `.claude/skills/`).
