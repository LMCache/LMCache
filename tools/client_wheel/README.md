# lmcache-client build

`build.py` builds the `lmcache-client` wheel from a staging copy of the tracked
files that `manifest.toml` selects. The root build of the full `lmcache` wheel
is not touched. CI runs everything below in `build_client_artifacts.yml`.

| File | Role |
| --- | --- |
| `manifest.toml` | What the wheel ships, what it keeps out, and the gaps the checks accept |
| `scope.py` | Reads the manifest; every other script selects files through it |
| `pyproject.toml`, `setup.py` | Static metadata and build glue for the staging root; `build.py` adds the root `[build-system]` |
| `build.py` | Stages the files and builds the wheel |
| `audit.py` | Checks the wheel's files, METADATA, entry points, native libraries and compat shims |
| `check_imports.py` | Static import closure, and per-module imports and smokes in a clean venv (`--runtime`) |
| `check_install.py` | Replacement/uninstall matrix and the subset check against the full wheel |
| `check_e2e.py` | A real request between a full-wheel server and a client-wheel venv; the second request must reuse the cache |

A tracked file under `lmcache/` is shipped if it is a compat shim or, with
`--boundary transitional`, a `[[transitional]]` file. Otherwise it is shipped
if it matches `trees` or `files` and no `[[exclude]]` entry.

## Changing what the wheel ships

**Adding code.** Add a line to `files`: a path, or a directory ending in `/`.
Declare a new third-party import in `pyproject.toml`, and add a `[smokes]`
entry if the code must do more than import. A top-level import that is not
declared fails the static check with `undeclared NAME imported at top level`.

**Removing code.** Delete its lines from `files`, then the dependencies, smokes
and compat shims only it used. A shipped module that still imports it fails the
static check with `not shipped`; a dependency left behind fails it with
`dependency NAME is imported by no module of the wheel`; a smoke left behind
fails the clean-venv import. Removing `lmcache/integration/vllm/` also means
removing the `client-vllm-cpu` job, which otherwise exits with `nothing to import`.

**Changing a requirement.** A requirement that differs from the full wheel's
(`requirements/common.txt` + `cuda13_core.txt`) needs a reason in
`[full_wheel_differences]`, and a reason needs a difference. The audit step
fails on PRs that edit `tools/client_wheel/` and warns on the others.

**Keeping a shared file out.** Add an `[[exclude]]` entry with `owner`, `reason`
and `paths`; it wins over `trees` and `files`. A client module that still
imports the file fails the static check with `excluded, not in boundary_gaps
(owner: reason)`. Lazy and typing imports that may stay go in `boundary_gaps`
as `"source kind target"`. A directory in `scanned` can only be excluded whole.

**Shipping server code.** A selected file under `deny` fails the build with
`in a deny area`. Add it to `trees` or `files`, take its area out of `deny`,
and drop its files from `[[transitional]]`.

The static check cannot see modules loaded by name. `scanned` covers the two
known scanned packages; `platform/_device_detect.py`,
`platform/base/ipc_wrapper.py`, `transport/grpc/descriptors.py` and
`integration/request_telemetry/factory.py` also load modules by name and are
not covered.

## Server files the client still needs

`[[transitional]]` lists them, each group with the change that removes it.
Only `build.py --boundary transitional` may ship them; that wheel is a
prototype, not a release artifact. When a migration lands, the ratchet step
(`check_imports.py --ratchet`) reports the group that is no longer needed.

Once the migrations are done, remove:

| What | Where | Reminder |
| --- | --- | --- |
| `[[transitional]]` groups, then the dependencies the static check names | manifest, `pyproject.toml` | ratchet, fatal on PRs that edit `tools/client_wheel/`; `dependency NAME is imported by no module of the wheel` |
| the `#6310` `[[exclude]]` entry and `boundary_gaps` | manifest | a stale gap fails the ratchet |
| `runtime_gaps` (#6573) | manifest | the runtime check exits 3 and warns |
| `lmcache/security/http.py` in `files` (#6573) | manifest | none |
| `custom_types.py`, from `[[transitional]]` to `[[compat_shims]]` (#6541) | manifest | the superset step's `grep custom_types.py` fails |
| `\|\| true` of the strict-wheel import step | workflow | none |
| the steps and the artifact upload of the transitional wheel | workflow | none |
| `--boundary`, `--ratchet`, `transitional_paths` | scripts | none |

## Running locally

```bash
NO_GPU_EXT=1 python tools/client_wheel/build.py --no-isolation --boundary transitional --version 0.0.0.dev0 --out dist/client
```

```bash
python tools/client_wheel/audit.py dist/client/*.whl --staging dist/client/STAGING_MANIFEST.json --boundary transitional
```

```bash
python tools/client_wheel/check_imports.py --source-tree . --boundary transitional
```

With the wheel installed in a venv:

```bash
python tools/client_wheel/check_imports.py --wheel dist/client/*.whl --runtime venv/bin/python
```

```bash
python tools/client_wheel/check_e2e.py --server-python FULL_VENV/bin/python --client-python CLIENT_VENV/bin/python
```
