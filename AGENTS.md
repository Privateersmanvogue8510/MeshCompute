# Coding Agent Entry Point

Read these files before changing architecture:

1. `INSTRUCTIONS.md`
2. `ARCHITECTURE.md`
3. `PROTOCOL.md`
4. `SECURITY.md`
5. `ROADMAP.md`

Critical invariant:

> MeshCompute is a true distributed inference project. Do not reduce it to ordinary load balancing across independent model servers.

Start with Milestones 0 through 3 in `INSTRUCTIONS.md`.

Also read `.agent/MEMORY.md` (resume point, decisions, gotchas from previous agent
sessions) and `CHANGELOG.md` (what actually shipped) before assuming what exists.

Keep changes testable, benchmarked, versioned, and small enough to review.

Never place user tool credentials on public inference workers.
