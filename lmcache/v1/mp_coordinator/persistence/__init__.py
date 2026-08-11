# SPDX-License-Identifier: Apache-2.0
"""What the coordinator keeps across a restart.

Two artifacts, because they agree on nothing that matters:

- The **directory checkpoint** (``checkpoint.py`` over
  ``snapshot_codec.py``) — large, binary, derived from the event stream,
  rewritten whole on a timer, and safe to lose.
- The **metadata document** (``metadata_persister.py``) — small, JSON, operator
  intent that nothing can rebuild, written when it changes.

Both sit on the one storage contract in ``store.py``, so a local file
today and an object store later are a single class apart.

Verbs follow the layer, so a name says which one you are at:

- ``capture`` / ``restore`` — live state ↔ a transferable value
- ``open_read`` / ``open_write`` — byte streams over stored bytes
- ``load`` / ``save`` — a whole artifact, storage included

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""
