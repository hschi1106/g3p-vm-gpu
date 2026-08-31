# Native Refactor Evidence

This directory records the evidence and responsibility mapping used while the
native-only refactor is in progress. It is descriptive, not normative:
language and wire behavior remains owned by [`spec/`](../../spec/).

- [`baseline.md`](baseline.md): reproducible pre-refactor build, test, parity,
  performance, quality, and CLI observations
- [`python-migration.md`](python-migration.md): ownership of every Python source
  and test module before removal
- [`inventory.md`](inventory.md): duplicated node metadata, tools, documents,
  binaries, and compatibility surfaces

The records describe commit `5f2b840724cfd06d3ec70c1515e7ea827d265e15`.
Later stages must update a record when they change the ownership or disposition
it assigns.
