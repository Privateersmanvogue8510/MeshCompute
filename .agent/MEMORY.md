# .agent/MEMORY.md — MeshCompute (agent scratch memory)

## Resume point (2026-09-14) — audit pass COMMITTED + PUSHED as mr-tbot; README rewritten (disclaimer, per-OS install, donation, community); LICENSE (Apache-2.0) added; codedatda.casa + mr-tbot.com portfolios updated from MAIN-NET. Next: verify on a real Windows/macOS/NVIDIA box; kill-switch/revocation design (see README Roadmap and hardening)
- Fable pass over Sonnet/Opus-built Phase-1 code. /auto-audit loop + /auto-agent-eco.
- Baseline: 44 tests pass (`.venv/bin/python -m pytest -q`).
- Targets: Linux x86_64 CPU (host), Linux CUDA, macOS arm64/x64 (Metal), Windows x64 (CUDA + CPU), Docker (control-plane/gateway), systemd.

## Decisions
- NVIDIA-only for GPU; CPU/RAM/storage contribution counts on every OS.
- Model updates: "model channels" — same model class, new version; nodes swap P2P/HF, delete old file.

## Gotchas
- `runtime_installer._build_from_source` shells out to `aw run` (dev-box tool) — breaks on any other machine.
- Windows llama.cpp assets are `.zip` (not tar.gz); Linux CUDA asset name must be verified live.
- `deploy/nodes.local.yaml`, `deploy/identities/*.key.json` are gitignored, present locally — never commit.
- Commit/push ONLY as mr-tbot (repo-local git config already set).

## Progress log (Fable pass)
- Rewrote runtime_installer (asset matrix win/mac/linux, zip+tar, accel cascade cuda-build>vulkan>cpu, pinned smoke hash, state.json channels, storage budget), swarm (handshake, resume by re-hash, no chunk-cache double write), daemon (model picker, fixed port 8099, seeding, P2P fetch, channel watch + changeover + delete old), control-plane (signed register/heartbeat, issued nonces, plan-bound receipts, path alias, manifest rescan + version guard, pools, relay auth), CLI (--model/--quant/--port/--accel/--max-storage/--update-poll, node models, manifest pin), sysinfo.py (cross-platform), install.ps1.
- Red-team (Opus) 40 findings; gateway/CLI + scheduler/ledger/frames/transport fixes delegated to 2 Opus agents (user asked: no more subagents after these).
- Live demo running: control plane on :8080 (scratch DB/manifests), node1 :8099 seeds public/smollm2-360m, node2 (MESH_HOME=scratch/demo/mesh2) :8098 should leech from node1.
- Cheaper path chosen: did NOT download the 24 GB qwen files to pin its hash (`mesh manifest pin` exists; operator runs it once).
- Live verified: node2 fetched channel v1 from node1 P2P (259/259 chunks, verified); pushes v1->v2->v3->v4 swapped both nodes live + deleted old file, but both hit HF each wave (race). Fixed with ORIGIN-LEADER election on the tracker clock (RendezvousPeers.origin_leader = first node to want the hash; others wait up to WAIT_MAX_S=1800 then fetch P2P; node re-announces as seeder after swap). Wave v5 verified: one HF download, the other node got it P2P. Gateway->scheduler->live node verified ('Pong.', x-mesh headers).
- Full suite: 120 passed. Lint clean (ruff --isolated --select E,F,W --line-length 110). Docker image builds + imports.
- Not committed. User must approve commit/push (mr-tbot identity).
