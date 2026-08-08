# Reproducing the cache-policy benchmark suite with Docker

This is the Docker-only path for installing and reproducing the
cache-policy evaluation: no local Python, conda, or system package
installs needed beyond Docker itself. See
[`README.md`](README.md) for the conda-based equivalent and the full
suite documentation; this file only covers the Docker workflow.

**What "reproduce" means here, precisely:**

- The deterministic reproducibility check (step 2 below) and every
  number in `results/admission_control/admission_vs_lfu_paired.json`
  and the other committed JSON result files are **bit-for-bit
  reproducible** -- pure-Python arithmetic over fixed seeds and
  already-committed input data, with no dependence on host fonts, OS,
  or matplotlib version.
- The **figure PNGs** (`report/figures/*.png`) are rendered from those
  same numbers, and `requirements-figures.txt` fully pins matplotlib and
  every one of its rendering-relevant transitive dependencies
  (`fonttools`, `pillow`, `kiwisolver`, `contourpy`, etc. -- not just
  matplotlib/numpy themselves). **This is necessary but was not, by
  itself, sufficient**: testing found that a `docker build` reusing
  Docker's layer cache could still produce a build whose figures differed
  from a `--no-cache` build with the exact same pinned dependency
  versions installed (confirmed via `pip freeze` -- same versions, still
  different pixels on ~5-10% of each figure). Three independent
  `docker build --no-cache` runs, by contrast, agreed byte-for-byte with
  each other every time. **Always pass `--no-cache` when reproducibility
  matters** (step 1 below) -- a plain `docker build` that reuses cached
  layers is not a reliable way to reproduce these figures, even though
  it reliably reproduces everything else in this suite.
- `report/cache_policy_evaluation_report.pdf` is the final submitted
  report and is **not** rebuilt by this image or any script here -- see
  "Report and reproducibility" in `README.md`.

## 1. Build the image

From the repository root:

```bash
docker build --no-cache -f benchmarks/cache_policy/Dockerfile -t lmcache-cache-policy-bench .
```

`--no-cache` is required, not optional, if you intend to reproduce the
committed figures exactly -- see the note above. It costs a full
re-download of the CPU torch wheel (a few minutes), but a cached rebuild
has been observed to silently produce different figure pixels despite
installing identical pinned package versions.

This installs `requirements/common.txt`, `requirements/test.txt`, and
the fully-pinned `requirements-figures.txt` on top of `python:3.11-slim`
-- matching `environment.yml`'s Python version and this repo's own
pinned figure-rendering versions. `lmcache` itself is never `pip
install`-ed; it's imported via `PYTHONPATH=/app` (set in the image),
exactly like the conda-based instructions in `README.md`.

## 2. Run the deterministic-reproducibility check (inside the container)

This is the image's **default command** -- running the container with
no extra arguments runs it automatically:

```bash
docker run --rm lmcache-cache-policy-bench
```

Expected output ends with:

```
PASS: all <N> results were bit-for-bit identical across two independent process runs.
```

This runs entirely inside the container (`verify_reproducibility.py`
spawns its own two worker subprocesses inside the same container), needs
no external data or volume mounts, and is the fastest way to prove the
image's environment reproduces the simulator's behavior deterministically
before trusting anything downstream of it.

## 3. Run the stats + comparison-script tests (inside the container)

```bash
docker run --rm lmcache-cache-policy-bench \
    pytest tests/benchmarks/test_stats.py tests/benchmarks/test_compare_admission_vs_lfu.py -v
```

Expect 21 passed.

## 4. Rebuild the report figures (inside the container, output copied back to the host)

Mount `results/` and `report/` so the regenerated files land back in
your checkout instead of disappearing when the container exits:

```bash
docker run --rm \
    -v "$(pwd)/benchmarks/cache_policy/results:/app/benchmarks/cache_policy/results" \
    -v "$(pwd)/benchmarks/cache_policy/report:/app/benchmarks/cache_policy/report" \
    lmcache-cache-policy-bench \
    ./benchmarks/cache_policy/reproduce_figures.sh
```

This writes a fresh
`results/admission_control/admission_vs_lfu_paired.json` and all 10 PNGs
under `report/figures/` -- from already-committed JSON only, no sweeps
are re-run (see `report/FIGURE_SOURCES.md` for what feeds each figure).

**Verify nothing unexpected changed:**

```bash
git status
git diff --stat benchmarks/cache_policy/results/admission_control/admission_vs_lfu_paired.json
```

The JSON diff should be empty, and `git status` should show **no changes
at all** -- verified end to end with this exact command against a fresh
`docker build --no-cache` image: every one of the 10 figures and the
JSON came out byte-identical to what's already committed, and running
this same command a second time against the same image reproduced that
result again. If you built with `--no-cache` and still see a figure PNG
reported as modified, check whether the *numbers* changed by diffing the
JSON the figure is sourced from (see `report/FIGURE_SOURCES.md`) before
assuming a real reproducibility failure -- but also double check you
actually used `--no-cache`, since that's the one variable observed to
matter.

## 5. (Optional, slow) Full experiment rerun, including real ShareGPT data

The default image does not include the ~230 MB ShareGPT corpus (it's
git-ignored and not fetched during the build). To run the full sweep or
the real-data tier inside the container, mount the corpus in after
preparing it on the host per README.md's "Real-data (ShareGPT) testing"
section, then run the full-path commands from README.md's "Report and
reproducibility" section via `docker run ... bash -c "..."` (see the
multi-line example in the Dockerfile's own header comment). Budget
significant extra time -- this tier is intentionally excluded from the
default `docker run` check.

## Troubleshooting

- **Build fails downloading torch**: `requirements/common.txt` pulls a
  CPU-only torch wheel from `download.pytorch.org`; a flaky network mid-build
  will show as a `pip install` failure, not a code problem. Retry the
  build.
- **`docker run` figure step fails to find `lmcache`**: confirm you
  didn't override `PYTHONPATH` in your `docker run` invocation --  the
  image sets `PYTHONPATH=/app` itself and no additional flags are needed.
- **Figures look different from `report/figures/` already in the repo**:
  expected per the pixel-vs-numbers note above; compare the JSON in
  `report/FIGURE_SOURCES.md`'s "Source(s)" column instead.
