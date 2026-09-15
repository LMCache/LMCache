# Contributing 👍🎉

First off, thank you for taking the time to contribute! 🎉👍  
Check out the [online docs](https://docs.lmcache.ai/developer_guide/contributing.html) for a set of guidelines for contributing.

A summary of LMCache's current direction can be found at: [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

=======

## Becoming an LMCache Committer

To become a committer, you should:

- Have **more than 5 important features** merged.
- Have been **contributing for longer than 3 months**.
- Be [**nominated by an existing committer**](MAINTAINERS.md).

========

## Basics of LMCache dev

The default branch is `dev`. Base all new branches and pull requests against `dev`.

## Submitting a Pull Request

LMCache uses the fork-and-pull model: you work on a branch in your own fork and open a
pull request (PR) against `dev` in `LMCache/LMCache`. If the mechanics of that flow are new
to you, the [GitHub Workflow Guide](https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md)
from Kubernetes walks through it in detail.

### 1. Before you write code

- For a new feature or a bug you have found, [open an issue](https://github.com/LMCache/LMCache/issues/new)
  first to discuss the approach with maintainers. Small fixes (typos, docs, obvious bugs) can
  go straight to a PR.
- Keep each PR small and focused on one logical change. Break large work into a series of
  PRs that each stand on their own — see
  [Section 1.1 of the coding standard](docs/coding_standards.md#11-pr-scope-and-breakdown).

### 2. Branch and commit

```bash
# Fork the repo on GitHub, then clone your fork and add the upstream remote
git clone git@github.com:<your-username>/LMCache.git
cd LMCache
git remote add upstream https://github.com/LMCache/LMCache.git

# Always branch from an up-to-date dev
git fetch upstream
git switch -c my-change upstream/dev
```

Every commit must carry a `Signed-off-by` trailer certifying that you agree to the
[Developer Certificate of Origin](DCO). Use `-s` and git adds it for you:

```bash
git commit -s -m "[Doc] Describe the change"
```

If you forget, `git commit --amend -s` fixes the last commit and
`git rebase --signoff upstream/dev` fixes a whole branch.

### 3. Check your work locally

Run the same checks CI runs, before you push:

```bash
pre-commit run --all-files   # ruff, isort, mypy, codespell, clang-format

pytest -xvs --ignore=tests/disagg \
  --ignore=tests/v1/multiprocess/ \
  --ignore=tests/v1/distributed/ \
  --ignore=tests/skipped \
  --ignore=tests/v1/storage_backend/test_eic.py
```

If you touched anything under `docs/`, also confirm the docs build cleanly — the Sphinx
build must finish with no errors or warnings:

```bash
cd docs && make clean && make html
```

See [Testing](#testing) and [Linting & Code Quality](#linting--code-quality) below for the
details, and [`docs/coding_standards.md`](docs/coding_standards.md) for the full standard
your PR will be reviewed against.

### 4. Open the pull request

Push the branch to your fork and open a PR against `dev`.

- **Title** — prefix it with the type of change so the change set is easy to scan:
  `[Bugfix]`, `[Build]`, `[CI]`, `[Core]`, `[Doc]`, `[Misc]`, `[Model]`, `[Test]`.
  Include every prefix that applies if the change spans categories.
- **Description** — fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md): what the PR
  does and why it is needed, anything reviewers should look at first, and the checkboxes for
  user-facing docs and unit tests.
- **Link the issue** — `Fixes #1234` closes the issue when the PR merges; `Refs #1234` links
  it for context without closing it.
- **Open it as a draft** if you want early feedback on unfinished work, and mark it ready for
  review when it is done.

Opening the PR triggers the code-quality and test workflows. Both must pass before a
maintainer will merge; if one fails, read the job log, push a fix, and the checks re-run.

### 5. Ask for a review in Slack

Opening a PR does not notify a particular person. [`CODEOWNERS`](.github/CODEOWNERS)
automatically requests review from the owners of the paths you touched, but the reliable way
to get a human looking at your change is to ask for one:

> **After you open your PR, post the link in `#pr-reviews` in the
> [LMCache Slack workspace](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA)**,
> with a one-line summary of what it does.

This is what puts a human in the loop who knows your PR exists and cares that it lands. If
you would rather not use Slack, comment on your own PR and `@`-mention the `CODEOWNERS` for
the paths you changed — the goal is simply that a specific person knows the PR is waiting.

Please do follow up if your PR goes quiet: a PR with no activity for 60 days is marked
`stale` and closed 30 days after that.

### 6. Review, revisit, merge

- Maintainers review against [Section 9 of the coding standard](docs/coding_standards.md#9-code-review-process).
  Reading it first tells you what they will look for.
- Address feedback by pushing additional commits to the same branch (each one signed off).
  Maintainers **squash and merge**, so your PR ends up as a single commit regardless of how
  many commits it contains — there is no need to tidy your history.
- Re-request review once you have addressed the comments, and nudge `#pr-reviews` again if
  the PR stalls.
- If your branch falls behind `dev` and CI can no longer run cleanly, update it from
  `upstream/dev` and push again.

## Python Environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies:

```bash
# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install vllm/sglang/other engines

# Install LMCache. Requires nvcc. Use --no-build-isolation to ensure torch compatibility.
uv pip install -e . --no-build-isolation

# For ROCm build
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

C++/CUDA files use clang-format (Google style, 80-col). Rust code in `rust/` uses `cargo fmt` and `cargo clippy`.

All Python files require an `# SPDX-License-Identifier: Apache-2.0` header as the first line.

### Import Ordering

Imports must follow this section-heading convention:

```python
# Standard
import os

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig

# Local
from .utils import helper
```

### SLF (Private Member Access)

SLF lint rules are currently enforced by CI only in `lmcache/v1/multiprocess/` and `lmcache/v1/distributed/`. However, **all new code should follow SLF discipline regardless of location** — never access private members (prefixed with `_`) of other classes. Treat this as a project-wide coding standard for any new or modified code.

## Coding Conventions

### Type Hints

All functions and methods must have type hints for arguments and return values.

### Docstrings

Every public function and method must have a clear docstring covering:
- What the function does
- Arguments (with types and descriptions)
- Return values
- Raised exceptions (if any)
- Additional notes when behavior is non-obvious

### Encapsulation

Never access private members (prefixed with `_`) of other classes. Interact only through their public API.

### Code Organization

- **Module-level helper functions** go at the top of the file (after imports, before classes).
- **Private/helper methods** within a class go at the end of the class, after all public methods.
