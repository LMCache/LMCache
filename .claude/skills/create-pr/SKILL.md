---
name: create-pr
description: Create a draft GitHub pull request from the current branch to the upstream repo, using the repo's PR template
allowed-tools: Bash, Read, Glob, Grep, Agent, mcp__github__create_pull_request
argument-hint: "[base_branch] [--title 'PR title']"
---

# Create Pull Request

Create a draft GitHub pull request from the current branch to the upstream repository.

## Arguments

`$ARGUMENTS` may contain:
- First positional arg: base branch to merge into (default: auto-detect from upstream's default branch)
- `--title 'PR title'`: override the PR title (default: auto-generated from changes)

## Steps

### 1. Gather context

Run these in parallel:

```bash
# Check remotes — identify the fork remote and the upstream remote
git remote -v

# Current branch and tracking info
git branch -vv --contains HEAD

# Untracked/unstaged changes (warn user if dirty)
git status -s

# Default branch of the upstream repo
gh api repos/<UPSTREAM_OWNER>/<UPSTREAM_REPO> --jq '.default_branch'
```

Determine:
- **Fork remote**: the remote pointing to the user's fork (usually `local` or `origin` if the user's fork)
- **Upstream remote**: the remote pointing to the upstream repo (look for the canonical org/repo)
- **Fork owner**: extracted from the fork remote URL (e.g., `alice` from `git@github.com:alice/LMCache.git`)
- **Upstream owner/repo**: extracted from the upstream remote URL
- **Base branch**: from `$ARGUMENTS` or the upstream repo's default branch

### 2. Check the branch is pushed

Verify the current branch is pushed to the fork remote. If not, warn the user and stop — do NOT push without explicit permission.

### 3. Find and read the PR template

```
Glob pattern: .github/PULL_REQUEST_TEMPLATE.md
Glob pattern: .github/PULL_REQUEST_TEMPLATE/**/*.md
Glob pattern: .github/pull_request_template.md
```

Read the template to understand required sections.

### 4. Analyze changes

```bash
# Commits on this branch vs the base
git log --oneline <upstream_remote>/<base_branch>..HEAD

# Diff stat for summary
git diff <upstream_remote>/<base_branch>..HEAD --stat
```

If there are many commits, also read the full diff to understand the changes:
```bash
git diff <upstream_remote>/<base_branch>..HEAD
```

Run `/pre-pr-check --scope branch` and address or report its findings before
continuing. A clean agent review does not replace the submitting human's own
line-by-line review.

### 5. Generate the reviewer explainer

For AI-assisted code changes, generate a self-contained HTML file that a
reviewer can open without repository context or local assets. Include:

- a visual overview and explanations of meaningful code changes;
- end-to-end call chains, ownership boundaries, and extension points;
- a test map explaining the necessity and regression target of each changed
  test;
- compatibility and performance risks, validation evidence, gaps, and focused
  reviewer questions.

Build this as a reviewer-facing technical article, not a decorative summary.
For a non-trivial change, normally include at least three substantive visuals:
a before/after architecture view, an end-to-end sequence or data-flow view, and
a responsibility/extension-boundary or test-coverage view. Connect every visual
and annotated code walkthrough to real files and symbols, inputs and outputs,
state transitions, error paths, and owners. Explain motivation, rejected or
important alternatives, compatibility and performance effects, rollout or
fallback behavior, remaining gaps, and a suggested review order. Screenshots or
decorative graphics do not count; use fewer diagrams for a small change only
when the explainer says why that is sufficient.

When motion materially clarifies a call chain or state transition, add a
lightweight animated or step-through diagram with visible pause, replay, or
step controls. Provide a complete static fallback, respect
`prefers-reduced-motion`, avoid distracting auto-looping, and keep it usable
offline.

Put complete English and Chinese versions in the same HTML file, with a visible
one-click language switch. Keep both views semantically synchronized across
code changes, call chains, test rationale, risks, evidence, gaps, and reviewer
questions. The switch must work from a downloaded local file without a server,
build step, external assets, or an online translation service. Do not create
separate language-specific files or rely on automatic browser translation.
Translate diagram labels, captions, animation controls, alt text, and reviewer
guidance as well as the prose.

Use embedded CSS and SVG diagrams. Remove secrets, machine-local paths, and
inaccessible links. Keep the file outside the repository unless the user asks
to commit it. Upload the `.html` file and put its link directly in the PR
description. A GitHub attachment that downloads the file for local viewing is
acceptable. Do not post a separate comment or archive one HTML file in a zip.
Never create or enable Pages, another repository, a branch, a gist, or a
different hosting resource for the explainer.

### 6. Draft the PR

- **Title**: Use `--title` from arguments if provided. Otherwise, generate a concise title (under 70 chars) summarizing the change.
- **Body**: Fill in the PR template sections based on the changes. Focus on:
  - **Why** the change is needed (motivation, problem being solved)
  - **User-facing changes** (new CLI args, changed behavior, breaking changes)
  - Do NOT enumerate per-file changes — reviewers can read the diff
  - Keep it concise and informative

Show the drafted title and body to the user and ask for confirmation before creating the PR.

### 7. Create the PR

Use the `mcp__github__create_pull_request` tool:
- `owner`: upstream repo owner
- `repo`: upstream repo name
- `head`: `<fork_owner>:<branch_name>`
- `base`: the base branch
- `title`: the PR title
- `body`: the PR body
- `draft`: true

### 8. Report

Print the PR URL and the HTML explainer location so the user can review both.

## Important Notes

- Never push code or create commits — this skill only creates the PR from already-pushed branches.
- Always show the PR title and body to the user for confirmation before creating.
- If the working tree is dirty, warn the user that uncommitted changes won't be included.
- Respect the PR template structure — fill in all sections, use checkboxes where the template has them.
- Always create the PR as a draft. Never mark it ready for review; only the
  submitting human may do that after completing the PR template's self-review
  checklist.
- On every later AI-assisted code update, convert the PR back to draft first,
  then refresh the title, description, and HTML explainer for the new head.
  Never treat an earlier human review as covering later AI-generated changes.
