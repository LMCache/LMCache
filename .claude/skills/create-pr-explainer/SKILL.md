---
name: create-pr-explainer
description: Generate, validate, and publish the required self-contained bilingual HTML reviewer guide for an AI-assisted LMCache pull request. Use when creating an AI-assisted PR or updating its code, title, description, validation evidence, or explainer for a new head.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
---

# Create PR Explainer

Create one reviewer-facing HTML artifact that lets a maintainer understand and
review an AI-assisted change without prior project context. Keep it tied to one
exact PR head; an earlier artifact never covers later AI-generated changes.

## 1. Gather exact evidence

Record the current head SHA, base ref, commit list, changed files, and full diff.
Read the affected implementations, callers, tests, and relevant architecture or
contribution guidance. For an existing PR, also read its current title, body,
review state, unresolved threads, and validation status.

Do not infer call chains from filenames alone. Trace real symbols, inputs,
outputs, state transitions, ownership, extension points, and failure paths.

## 2. Build a detailed technical narrative

The explainer is a technical review article, not a decorative summary. Cover:

- the motivation, previous behavior, new behavior, and important alternatives;
- an annotated walkthrough of meaningful code changes with accessible links to
  the exact revision;
- end-to-end call or data flows, including responsibility boundaries, inputs,
  outputs, state transitions, error paths, and independently replaceable parts;
- why every added or changed test is necessary, what regression it protects,
  why it belongs at that level, and what remains uncovered;
- compatibility, performance, rollout or fallback behavior, validation
  evidence, known gaps, risks, and focused reviewer questions;
- a concrete suggested review order for a maintainer without prior context.

Avoid invented detail and say when evidence is unavailable. Do not expose
secrets, credentials, machine-local paths, or links reviewers cannot access.

## 3. Make the explanation visual

For a non-trivial change, normally include at least three substantive visuals:

1. a before/after architecture or behavior view;
2. an end-to-end sequence, call-chain, state-machine, or data-flow view;
3. a responsibility/extension-boundary, test-coverage, or risk view.

Tie every visual to real code and review decisions. Screenshots, stock art, and
decorative graphics do not count. A small or documentation-only change may use
fewer diagrams only when the artifact explains why that is sufficient.

When motion materially improves a call chain or state transition, add a
lightweight animated or step-through diagram with visible pause, replay, or
step controls. Provide a complete static fallback, respect
`prefers-reduced-motion`, avoid distracting auto-looping, and keep every
essential fact available without animation.

## 4. Put both languages in one offline file

The same HTML file must contain complete English and Chinese explanations with
a visible one-click language switch. Keep both views semantically synchronized
across prose, diagram labels, captions, controls, alt text, code explanations,
test rationale, risks, and reviewer guidance.

Do not create separate language files or rely on browser translation, a server,
a build step, remote fonts, external assets, or an online translation service.
Embed CSS, SVG, and any required JavaScript so the downloaded file works by
itself. Preserve readable static content when JavaScript or motion is disabled.

## 5. Validate before publication

At minimum:

- parse the HTML and confirm it contains no missing local dependencies;
- open it locally and exercise English/Chinese switching;
- verify responsive layout, keyboard-accessible controls, static fallbacks,
  reduced-motion behavior, and any pause/replay/step controls;
- audit every path, URL, symbol, revision, command, artifact name, and result
  from a reviewer's perspective;
- confirm the explained SHA, title, claims, diagrams, and validation evidence
  match the current head.

Treat a stale or incomplete explainer as a review blocker.

## 6. Publish without extra resources

Keep the generated file outside the repository unless the user explicitly asks
to commit it. Upload the single `.html` file and put its attachment link directly
in the PR description. A download-for-local-viewing attachment is acceptable.

Do not post a separate comment, zip a single HTML file, or create or enable
Pages, another repository, a branch, a gist, or any other hosting resource.

For every later AI-assisted code update:

1. move the PR back to Draft before presenting the new code to reviewers;
2. regenerate and validate the explainer for the new head;
3. replace the PR's title, description, explainer link, and validation evidence;
4. leave the human line-by-line review checkbox incomplete; only the submitting
   human may mark that revision Ready for review.

## 7. Report

Return the exact explained SHA, local artifact path, published attachment URL,
validation performed, known gaps, and confirmation that the PR remains Draft.
