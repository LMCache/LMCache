# Module Dependencies

LMCache code is split into six parts. Each part may import the parts its arrows
lead to, directly or through other parts: integration may import common, since
integration depends on client and client on common. The rules live in one file,
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
    integration --> client
```

The dotted arrow means transport uses devices such as IPC wrappers on demand. The
check treats it like a solid arrow, and fails if this diagram and the rules file
disagree.

Imports within a part are always allowed. Server and client never import each
other, and nothing imports integration. Server is every lmcache module that is
not in another part. Modules with their own name outside the diagram, such as
`lmcache.cli` and `lmcache.sdk`, are not server and are not checked.

Which packages make up each part, and where the files the diagram does not draw
belong, is in `tools/module_deps.yaml`.

## What counts as an import

- Every `import` and `from ... import`, including relative imports, imports
  inside functions, and imports inside `if TYPE_CHECKING:` blocks.
- Attribute chains on an imported `lmcache`, such as `lmcache.v1.client.x.run()`.
- `importlib.import_module(...)` and `__import__(...)` when the target is a
  literal or an f-string whose literal prefix lies inside one part.

## Exceptions

`exceptions` in the rules file lists the places that break the tree today, by
part or package, for example `transport -> client`. Delete a line once its
imports are fixed; the check fails on a line that matches nothing. Do not add a
line to get a new import past the check. Move the shared code into a part both
sides may import instead, usually common.

## Limits

- A new import inside an excepted part or package passes.
- Dynamic imports whose target is computed at run time, `importlib.util`,
  `runpy`, `exec`, `sys.modules` and entry points are not checked.
- Draft pull requests skip pre-commit, so the check runs once a PR is ready for
  review.
