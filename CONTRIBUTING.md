# Contributing 👍🎉

First off, thank you for taking the time to contribute! 🎉👍  
Check out the [online docs](https://docs.lmcache.ai/developer_guide/contributing.html) for a set of guidelines for contributing.

New to LMCache? Start at the [2026 onboarding umbrella issue](https://github.com/LMCache/LMCache/issues/3372) — it walks you through your first PR step by step.

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
  --ignore=tests/inprocess/v1/storage_backend/test_eic.py
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

- **Title** — start with a tag saying what kind of change it is, so the commit log stays easy
  to scan. Two forms are in wide use on `dev`, and either is fine:
  - a bracketed type — `[Bugfix]`, `[Build]`, `[CI]`, `[Core]`, `[Doc]`, `[Misc]`,
    `[Model]`, `[Test]`
  - a conventional-commit type — `fix:`, `feat:`, `docs:`, `ci:`, `refactor:`, `perf:`,
    `test:`, `chore:`, `build:`

  Naming the component as well helps reviewers spot their area quickly — as a second bracket
  (`[MP][Bugfix] ...`) or as a conventional-commit scope (`fix(mp): ...`). Include every type
  that applies if the change spans categories.
- **Description** — fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md): what the PR
  does and why it is needed, anything reviewers should look at first, and the checkboxes for
  user-facing docs and unit tests.
- **Link the issue** — `Fixes #1234` closes the issue when the PR merges; `Refs #1234` links
  it for context without closing it.
- **Open every PR as a draft.** A draft lets CI and the author catch problems before the PR
  asks maintainers for review. Follow the self-review gate below before marking it ready.

### 5. Self-review before marking the PR ready

The author owns the first review. Before clicking **Ready for review**:

- Review every line in the **Files changed** tab, including generated or agent-assisted
  changes, and fix anything you would ask another contributor to fix.
- Run the relevant local checks and record the commands and results in the PR description.
  Read the draft CI results too; a skipped check is not a passing check.
- Confirm that the description still matches the final diff, links the relevant issue, and
  calls out compatibility concerns, missing coverage, and user-facing documentation.
- Confirm that every commit has a DCO `Signed-off-by` trailer.
- Complete the **Before marking ready for review** checklist in the PR description.

Only the submitting human should mark the PR ready. For AI-assisted changes, the human must
understand and be able to defend the change end to end; a review performed only by the agent
that wrote the change does not satisfy this gate.

#### AI-assisted update loop

Use `/create-pr-explainer` in Claude Code or `$create-pr-explainer` in Codex
to generate the reviewer guide. Codex discovers the repository skill through
`.agents/skills/create-pr-explainer`, a symlink to
[the shared workflow](.claude/skills/create-pr-explainer/SKILL.md).
Both tools use that file for content, validation, and publication requirements.

For a PR containing AI-generated code, the same gate applies after every AI-assisted update,
not just when the PR is first opened:

1. Move the PR back to draft before asking reviewers to look at the new code.
2. Re-evaluate and refresh the PR title and description against the new head, including
   claims, validation results, compatibility notes, and known gaps.
3. Regenerate the HTML reviewer explainer and replace its attachment or link in the PR.
4. Rerun relevant checks, then have the submitting human review every new or changed line.
5. Mark the PR ready again only after the checklist reflects that latest human review.

The HTML explainer should help a reviewer who has no prior context understand the change. It
must be self-contained and include:

- a visual overview and a file-by-file explanation of the meaningful code changes;
- end-to-end call-chain diagrams, including ownership boundaries and extension points;
- a test map explaining what regression each test protects and why the test belongs at that
  level;
- compatibility and performance risks, validation evidence, missing coverage, and focused
  questions for reviewers.

Write the explainer at the depth of a reviewer-facing technical article, not a decorative
summary. A non-trivial PR should normally include at least three substantive visuals: a
before/after architecture view, an end-to-end sequence or data-flow view, and a
responsibility/extension-boundary or test-coverage view. Tie diagrams and annotated code
walkthroughs to real symbols, files, inputs, outputs, state transitions, error paths, and
owners. Explain motivation, important alternatives, compatibility and performance effects,
fallback or rollout behavior, remaining gaps, and a concrete suggested review order.

When motion would materially clarify a call chain or state transition, add a lightweight
animated or step-through diagram with visible pause, replay, or step controls. It must have
a complete static fallback, respect `prefers-reduced-motion`, avoid distracting auto-looping,
and work offline. Screenshots and decorative graphics do not count as substantive visuals;
small or documentation-only changes may use fewer diagrams when the explainer states why.

The AI tool must upload the `.html` file and put its link directly in the PR description. A
GitHub attachment that downloads the HTML for local viewing is acceptable. Do not post a
separate comment or wrap one HTML file in a zip archive. Do not create or enable Pages,
another repository, a branch, a gist, or a different hosting resource for the explainer.

Embed CSS, SVG diagrams, and other assets in the file; remove secrets, machine-local paths,
and links to artifacts reviewers cannot access. The explainer supplements the PR description
and source diff; it does not replace either one.

The single HTML file must contain complete English and Chinese versions with a visible
one-click language switch. Keep the two versions synchronized: each must explain the same
code changes, call chains, test rationale, compatibility and performance risks, validation
gaps, and reviewer questions. The switch must work when the downloaded file is opened
locally, without a server, build step, external assets, or an online translation service.
Translate diagram labels, captions, animation controls, alt text, and reviewer guidance in
addition to the prose.

### 6. What CI runs on your PR

Opening or updating a PR triggers a set of GitHub Actions workflows, plus a DCO check and a
Buildkite pipeline that report back as PR checks. The path filters mostly separate
operator-only changes from everything else, so a docs-only PR still runs most of the matrix.

| Check | Runs on | Covers |
|---|---|---|
| **DCO** | every PR | every commit carries a `Signed-off-by` trailer |
| **Code Quality** | every PR outside `operator/` | `pre-commit run --all-files` — ruff, ruff-format, isort, mypy, codespell, clang-format, and the repo's local hooks (SPDX headers, banned APIs) |
| **Test** | PRs to `dev` and `release-**` outside `operator/` | CPU-only unit tests, `pytest -m "not (cuda or musa or xpu or npu or neuron)"`, on Python 3.10–3.13 |
| **CPU device** | every PR outside `operator/` | server benchmarks and vLLM end-to-end tests on CPU, on `ubuntu-22.04` and `macos-latest` |
| **CodeQL** | PRs to `dev` | static analysis of the Python code and the Actions workflows |
| **Operator CI** | PRs touching `operator/` | the Go operator's own build and test suite |
| **actionlint** | PRs touching `.github/workflows/` | workflow-file linting |
| **PR Full Build** | PRs carrying the `full` label | builds the sdist and the CUDA, CLI, and cu129 wheels. The label is applied automatically when auto-merge is enabled on a PR, so it is normally a maintainer's doing rather than something you add |

GPU coverage does not run in GitHub Actions. It runs on Buildkite — unit tests on a GPU
runner plus the vLLM integration suite — and posts its own `buildkite/...` check, so read
that one too rather than only the green Actions checks.

All of this runs on draft PRs as well, so you can open a draft and let CI tell you what is
broken before you ask anyone to look.

If a check fails, open its log from the **Checks** tab on the PR, reproduce and fix it
locally, and push again. Every push re-runs the checks and cancels the previous in-flight
run for that PR.

### 7. Ask for a review in Slack

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

### 8. Review, revisit, merge

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
  --ignore=tests/inprocess/v1/storage_backend/test_eic.py

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

### Pre-commit Setup

Install the pre-commit hooks before making changes:

```bash
pre-commit install
```

To run all checks locally before pushing:

```bash
pre-commit run --all-files
```

### DCO Sign-Off

All commits must include a **Developer Certificate of Origin (DCO) sign-off**. Use `git commit -s` to sign off automatically:

```bash
git commit -s -m "your commit message"
```

PRs without DCO sign-off cannot be merged. The sign-off certifies that you have the right to submit the code under the project's license.

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
from lmcache.inprocess.v1.config import LMCacheEngineConfig

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
