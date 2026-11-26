# Module Dependencies

LMCache code is split into seven parts. SDK and integration are separate entry
points. Both explicitly depend on client, common, platform and transport, and
neither may import the other. Each part may import the parts its arrows lead to,
directly or through other parts. The rules live in one file,
`tools/module_deps.yaml`, and `tools/check_module_deps.py` enforces them in
pre-commit and CI.

```mermaid
flowchart BT
    platform --> common
    transport --> common
    transport -.-> platform
    server --> common
    server --> platform
    server --> transport
    client --> common
    client --> platform
    client --> transport
    sdk --> client
    sdk --> common
    sdk --> platform
    sdk --> transport
    integration --> client
    integration --> common
    integration --> platform
    integration --> transport
```

The dotted arrow means transport uses devices such as IPC wrappers on demand. The
check treats it like a solid arrow, and fails if this diagram and the rules file
disagree. It also fails if the arrows form a cycle.

Imports within a part are always allowed. Server and client never import each
other, and no lower part imports SDK or integration. SDK and integration must not
import server implementations, including `lmcache.v1.distributed`,
`lmcache.v1.multiprocess`, `lmcache.v1.mp_coordinator` and
`lmcache.v1.storage_backend`. Shared types and pure helpers belong in common;
engine-specific capture and adapter code belongs in integration.

Server is every lmcache module that is not in another part or explicitly outside
the tree. `lmcache.cli` is outside the tree and is not checked. SDK is checked,
including lazy and type-only imports.

Which packages make up each part, and where the files the diagram does not draw
belong, is in `tools/module_deps.yaml`.

## What counts as an import

- Every `import` and `from ... import`, including relative imports, imports
  inside functions, and imports inside `if TYPE_CHECKING:` blocks.
- Attribute chains on an imported `lmcache`, such as `lmcache.v1.client.x.run()`.
- `importlib.import_module(...)` and `__import__(...)` when the target is a
  literal or an f-string whose literal prefix lies inside one part.

## Exceptions

`exceptions` in the rules file lists the places that break the tree, by part or
package, for example `transport -> client`. It is empty today. Delete a line once
its imports are fixed; the check fails on a line that matches nothing. Do not add a
line to get a new import past the check. Move the shared code into a part both
sides may import instead, usually common.

## Limits

- A new import inside an excepted part or package passes.
- Dynamic imports whose target is computed at run time, `importlib.util`,
  `runpy`, `exec`, `sys.modules` and entry points are not checked.
- Draft pull requests skip pre-commit, so the check runs once a PR is ready for
  review.
