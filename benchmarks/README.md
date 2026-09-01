# Benchmark Manifests

This directory stores compact, reviewable evidence for release and refactor
gates. Raw run logs, generated populations, datasets, and profiler captures are
artifacts and remain outside version control.

- `spec_freeze.json` records hashes of normative specifications.
- `simple_exp_*.json` records fixed-population performance comparisons.
- `psb*.json` records PSB support, quality, and performance comparisons.

Every committed manifest must identify its inputs, configuration, relevant
commit(s), raw timing or quality measurements, and gate result. The commands
that produce manifests are documented in [`../tools/README.md`](../tools/README.md);
the experiment constraints are defined in
[`../docs/guides/experiment-protocol.md`](../docs/guides/experiment-protocol.md).
Do not hand-edit a measured result to make a gate pass.

