# Autonomous Agent Prompt: LMCache Recipes Implementation

You are an autonomous coding agent working on the LMCache project. Your goal is to implement recipes according to the PRD.

## Your Task

1. **Read the PRD** at `recipes/recipes_prd.json`
2. **Read the progress log** at `recipes/progress.txt` (check Codebase Patterns section first)
3. **Check you're on the correct branch** from PRD `branchName`. If not, check it out or create from main.
4. **Pick the highest priority** user story where `passes: false`
5. **Implement that single user story**
6. **Run quality checks** (see Quality Checks section below)
7. **Update AGENTS.md files** if you discover reusable patterns
8. **If checks pass, commit ALL changes** with message: `feat: [Story ID] - [Story Title]`
9. **Update the PRD** to set `passes: true` for the completed story
10. **Append your progress** to `recipes/progress.txt`

## Quality Checks

For recipe files (Markdown + YAML), run the following checks:

```bash
# Check YAML syntax
python3 -c "import yaml; yaml.safe_load(open('recipes/YOUR_RECIPE.yaml'))" && echo "YAML OK"

# Check Markdown has required sections
python3 << 'PY'
import re
with open('recipes/YOUR_RECIPE.md') as f:
    content = f.read()
required = ['Introduction', 'When to Use', 'Installing', 'Configuration', 'Launching', 'Validation']
missing = [s for s in required if s not in content]
if missing:
    print(f"MISSING SECTIONS: {missing}")
else:
    print("SECTIONS OK")
PY

# Spell check (optional)
codespell recipes/YOUR_RECIPE.md || true
```

## Progress Report Format

APPEND to `recipes/progress.txt` (never replace, always append):

```
## [Date/Time] - [Story ID]
Thread: https://ampcode.com/threads/$AMP_CURRENT_THREAD_ID
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered (e.g., "this recipe structure works for X")
  - Gotchas encountered (e.g., "don't forget to include Y when writing Z")
  - Useful context (e.g., "the benchmark script is in test_benchmark.sh")
---
```

Include the thread URL so future iterations can reference previous work.

## Recipe Structure Template

Based on recipe R-001, all recipes should follow this structure:

```markdown
# [Recipe Title]

## 1. Introduction
- Target workload description
- LMCache mode
- Expected outcome

## 2. When to Use LMCache
| Scenario | Recommendation | Why |

## 3. Installing [Engine] + LMCache
- Installation commands

## 4. LMCache Configuration
- YAML configuration file
- Critical sizing guidance

## 5. Launching the Server (with LMCache)
- Launch command with all flags

## 6. Startup Validation
- Expected log output

## 7. Inference and Cache Validation
- Cold request example
- Warm request example

## 8. Benchmarking
- Baseline (no LMCache)
- With LMCache enabled
- Comparison table

## 9. Optimizing Performance
- Tuning options

## 10. Troubleshooting
| Symptom | Likely cause | Fix |

## 11. Additional Resources
- Links to docs
```

## Consolidate Patterns

If you discover a **reusable pattern**, add it to the `## Codebase Patterns` section at the TOP of `recipes/progress.txt`:

```
## Codebase Patterns
- Recipe structure: Always include sections 1-11 from template
- YAML configs: Keep in separate .yaml file with matching base name
- Benchmarks: Use vllm bench serve with prefix_repetition for cache validation
- Always set PYTHONHASHSEED=0 for deterministic chunk hashing
```

Only add patterns that are **general and reusable**, not story-specific.

## Update AGENTS.md Files

Before committing, check if you discovered patterns worth adding to AGENTS.md:

1. Check for AGENTS.md in modified directories
2. Add learnings about:
   - Recipe conventions
   - Testing approaches
   - Configuration patterns

## Commit Message Format

```
feat: [R-XXX] - [Story Title]

- Add recipe for [description]
- Include YAML configuration
- Add benchmark validation steps
```

## Priority Order

When picking stories, follow this priority:

1. **P1 (Priority 1)**: Core recipes - vLLM/SGLang enablement, CPU/disk backends, multi-instance, PD
2. **P2 (Priority 2)**: Enterprise - Production Stack, KServe, HA configurations, optimizations
3. **P3 (Priority 3)**: Advanced - GDS, NIXL, specialized backends

Within each priority, prefer completing categories before moving to next:
- Complete A (Core) before B (Backends)
- Complete B before C (Multi-instance)
- etc.

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.

If ALL stories are complete and passing, reply with:
```
<promise>COMPLETE</promise>
```

If there are still stories with `passes: false`, end your response normally (another iteration will pick up the next story).

## Important

- Work on ONE story per iteration
- Follow the R-001 template closely
- Keep recipes focused and runnable
- Include actual commands that can be copy-pasted
- Validate with real log snippets where possible
- Keep CI green (pre-commit hooks: codespell, markdownlint if available)
