# MeshCompute repository instructions

Before generating or modifying code, read `/INSTRUCTIONS.md` and `/AGENTS.md`.

Preserve these invariants:

- true multi-peer inference is a first-class requirement,
- direct P2P data plane after rendezvous,
- topology-aware scheduling,
- signed public model manifests,
- no arbitrary public model code,
- tool credentials stay in trusted agent/tool environments,
- model, harness, backend, and tool providers remain pluggable,
- every protocol object is versioned,
- every network operation has cancellation and timeout behavior,
- changes require tests,
- performance claims require benchmarks.

Do not spend substantial effort on mobile, memory, or UI before the multi-peer inference POC works.
