---
name: create-pr
description: Stage a draft pull request in the contributor's fork, verify fork CI, and promote it to an upstream draft only with explicit approval
allowed-tools: Bash, Read, Glob, Grep, Agent, mcp__github__create_pull_request
argument-hint: "[base_branch] [--title 'PR title']"
---

# Create Pull Request

Stage a draft pull request in the contributor's fork and verify its GitHub
Actions checks before offering to promote the same head to an upstream draft.
Never use an upstream pull request as the first CI run.

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
- **Base SHA**: the full commit SHA currently at the upstream base branch
- **Fork target branch**: `dev-<short-sha>`, where `<short-sha>` is the first
  eight hexadecimal characters of the base SHA

Confirm through the GitHub API that the presumed fork is actually a fork of the
upstream repository. Also confirm that GitHub Actions are enabled in the fork.

### 2. Create the immutable fork target and check the feature branch

Fetch the upstream base branch and record its full SHA. Create
`dev-<short-sha>` in the fork at that exact commit:

```bash
git push <fork_remote> <full_base_sha>:refs/heads/dev-<short-sha>
```

If the target branch already exists, verify that it points to the same full
SHA and reuse it. A name collision pointing to another commit is an error:
stop and report both SHAs. Never update or force-push a staging target branch,
and do not modify the fork's moving `dev` branch.

Verify the current feature branch is pushed to the fork remote. If not, warn
the user and stop — do NOT push without explicit permission.

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

### 5. Open or find the fork staging PR

Look for an existing pull request in the fork with:

- `owner` and `repo`: the fork repository;
- `head`: the current feature branch;
- `base`: the `dev-<short-sha>` fork target branch.

If none exists, draft a concise title and body using the repository template,
show them to the user, and ask for confirmation. Then create the pull request in
the fork with `draft: true`. This staging pull request is the CI admission gate;
do not publish the reviewer explainer yet.

### 6. Verify fork CI

Wait for every applicable GitHub Actions check owned by the fork to reach a
terminal state. Report skipped and unavailable checks separately. Do not count
checks from an older head SHA, and do not treat a check that never started as a
pass.

If a check fails, keep the change in the fork, fix it through the normal coding
workflow, and rerun the checks. If Actions are disabled or a workflow depends
on unavailable secrets, environments, self-hosted runners, or external CI,
report the limitation and ask the user whether to promote despite the gap.

Fork CI is preliminary evidence only. It does not replace upstream required
checks or external systems such as Buildkite.

### 7. Request promotion approval

After the fork checks pass, report:

- the fork pull request URL;
- the exact tested head SHA;
- each applicable check and its result;
- checks that can run only upstream.

Ask the user for explicit approval to promote that exact SHA. Do not create the
upstream pull request in the same step that creates the fork staging pull
request, and do not interpret the original request to create a PR as approval
to skip this promotion checkpoint.

### 8. Generate the reviewer explainer

Run `/create-pr-explainer` and use its validated artifact and publication
details. Do not duplicate or weaken that skill's requirements in this workflow.

### 9. Draft the upstream PR

- **Title**: Use `--title` from arguments if provided. Otherwise, generate a concise title (under 70 chars) summarizing the change.
- **Body**: Fill in the PR template sections based on the changes. Focus on:
  - **Why** the change is needed (motivation, problem being solved)
  - **User-facing changes** (new CLI args, changed behavior, breaking changes)
  - Do NOT enumerate per-file changes — reviewers can read the diff
  - Keep it concise and informative

Update the title and body for upstream reviewers. Include the fork staging PR,
the tested head SHA, the fork check results, and any checks unavailable until
upstream promotion. Show the draft to the user and ask for confirmation before
creating the upstream pull request.

### 10. Create the upstream PR

Use the `mcp__github__create_pull_request` tool:
- `owner`: upstream repo owner
- `repo`: upstream repo name
- `head`: `<fork_owner>:<branch_name>`
- `base`: the base branch
- `title`: the PR title
- `body`: the PR body
- `draft`: true

### 11. Report

Print both pull request URLs, the promoted head SHA, the fork check summary, and
the HTML explainer location so the user can review them.

## Important Notes

- Never push feature code or create commits. The only push this skill performs
  is creation of the verified `dev-<short-sha>` target ref at the exact
  upstream base SHA; the feature branch must already be pushed.
- Always show the PR title and body to the user for confirmation before creating.
- If the working tree is dirty, warn the user that uncommitted changes won't be included.
- Respect the PR template structure — fill in all sections, use checkboxes where the template has them.
- Always create both the fork staging PR and the upstream PR as drafts. Never
  mark either ready for review; only the submitting human may do that after
  completing the PR template's self-review checklist.
- On every later AI-assisted code update, convert the PR back to draft first,
  then refresh the title, description, and HTML explainer for the new head.
  Never treat an earlier human review as covering later AI-generated changes.
