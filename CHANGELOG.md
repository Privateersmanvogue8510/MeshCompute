# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project does not yet use semantic version tags — `pyproject.toml` is pinned at
`0.1.0` throughout Phase 1.

## [Unreleased] — Phase 1 POC, audit pass 2 (2026-09-14)

Correctness/security pass over the Phase-1 code plus the model-distribution
feature set. Everything below is verified on the Linux host; Windows/macOS
paths are implemented + unit-tested but not yet run on real machines.

### Added

- **Cross-platform install + engine.** `scripts/install.ps1` (Windows) beside
  `scripts/install.sh`; `runtime_installer.py` now selects the upstream
  `llama.cpp` prebuilt per OS/arch/accelerator (Linux CPU/Vulkan, macOS Metal,
  Windows CUDA 12.4 + cudart bundle / CPU, zip or tar), verifies the binary
  runs, and degrades down a cascade (`cuda-build` → `vulkan` → `cpu`) instead of
  failing. The CUDA source build is opt-in (`--accel cuda-build`) and no longer
  shells out to a dev-box-only tool. Metal gets `-ngl 999`; Vulkan multi-GPU
  gets `GGML_VK_VISIBLE_DEVICES`.
- **Cross-platform host signals** (`node/daemon/meshcompute_node/sysinfo.py`):
  RAM total/available, free disk, user-idle seconds and AC-power on Linux, macOS
  and Windows (stdlib only). Idle-only contribution now gates on real input
  idle time on every OS instead of only X11.
- **Model channels.** A catalog alias is a channel; the control plane re-scans
  its manifest dir every 30 s, publishes a bumped `version` as the channel's
  current manifest, refuses version regressions, and exposes `version` +
  `size_bytes` on `/api/v1/models`. `GET /api/v1/models/{alias}` now works for
  aliases containing `/` (it never did).
- **Node model selection + organic updates** (`daemon.py`): `mesh node start
  --model <alias>` or an interactive picker (remembered in
  `~/.mesh/models/state.json`); `mesh node models` lists channels with sizes
  and fit. A node polls its channel (`--update-poll`, default 300 s) and on a
  new version downloads in the background, swaps the engine on the same fixed
  port (`--port`, default 8099), deletes the superseded files, re-posts its
  capability, and rolls back if the new engine fails to start.
- **Peer-to-peer model distribution wired into the daemon.** Nodes announce
  what they seed / want to the rendezvous tracker, dial seeders over QUIC, and
  fetch every chunk BLAKE3-verified before touching Hugging Face; every node
  seeds every file it holds. An update wave elects one **origin leader** per
  file on the tracker (first node to want it); the others wait for it to seed
  instead of all hitting Hugging Face — found live when two nodes both went to
  origin for the same version twice in a row. Swarm wire protocol now has a per-stream
  handshake, resumes by re-hashing chunks already on disk (no second chunk
  cache — a 24 GB model costs 24 GB), and reports bytes served.
- **Storage as a contribution.** `--max-storage` (default 100 GB) bounds cached
  model files, refuses over-budget downloads, and is advertised as
  `storage_share_bytes`. Orphaned model files from crashed runs are swept at
  start.
- `mesh manifest pin <yaml>` fills real `size_bytes` + BLAKE3
  `artifact_root_hash` from local files; `public/smollm2-360m` is a new, fully
  pinned CPU-friendly channel. `MESH_HOME` relocates a node's state.
- Pools endpoints (`GET/POST /api/v1/pools`) the CLI already exposed.
- Protocol (additive, `PROTOCOL_VERSION` unchanged): `RegisterRequest.signature_b64`,
  `HeartbeatRequest.signature_b64` + `peer_rtt_ms`, `ScheduleRequest.executable_strategies`,
  `RendezvousAnnounce.wanted_manifest_hashes`, `ModelView.version/size_bytes`.

### Fixed (from a red-team pass; each has a regression test)

- Anyone could re-register another node's id with their own key; node ids are
  now derived from the key and registration is signed.
- Heartbeats were unsigned: a dead node could be kept "online" and its free RAM
  spoofed. Now signed with the registered key.
- Work receipts were credited with no plan check and any nonce: 3 million
  credits in three HTTP calls. Receipts now require a plan the scheduler issued,
  membership in that plan, a nonce the control plane handed to that node, and
  GPU time that fits the receipt's own window; credit uses the same formula the
  on-chain ledger verifies (they disagreed).
- Ledger: a `correction`/`consumption` entry could mint an arbitrary positive
  balance; positive deltas now require a receipt-backed `verified_work` entry.
- `mesh manifest sign/verify` signed the hash string while the control plane
  signed the body; the CLI now uses the control plane's scheme and verifies both.
- Scheduler offered a pipeline split even when the two nodes could not jointly
  hold the model (the "infeasible" path was unreachable with 2+ nodes); tensor
  and speculative plans ignored queue depth and node capabilities; node↔node
  links were never populated so tensor/speculative plans could never be chosen
  through the API. All four fixed; node RTTs come from measured QUIC pings.
- Gateway: advertised multi-peer strategies it executed as single calls (now
  501 for non-single plans), forwarded a model alias without checking the node
  serves it (now 409 with the node's real model list; nodes start `llama-server
  --alias <channel>`), turned control-plane 404s into 500s, masked control-plane
  errors on `/v1/models`, never re-read `deploy/nodes.local.yaml`, kept sessions
  forever (now capped at 500, `DELETE` added), and `mesh api serve --control-url`
  set an env var the gateway didn't read.
- `Frame.decode` enforced the payload cap only before decompression (zip bomb);
  cap now applies to the decompressed size and is 64 MiB. Ping streams leaked
  one QUIC stream per RTT sample; framing errors were dropped silently.
- `LlamaCppBackend.start` raised `AttributeError` instead of the engine's real
  exit code/log tail; health timeout raised to 10 min for 24 GB models.
- Rendezvous `connect()` was never signed (always 400); relay websocket accepted
  any session id with unbounded queues.
- Docker Compose gateway had no route table mounted (no request could
  complete); `scripts/register_external_backend.py` now signs and labels its
  hardware figures as operator-declared.

## [Unreleased] — Phase 1 POC

The core P2P distributed inference proof-of-concept: build, run, and measure.

### Added

- **Standalone node daemon** (`node/daemon/meshcompute_node/daemon.py`,
  `mesh node start`): self-installs `llama.cpp` (prebuilt release binary where
  one exists, source build otherwise), downloads a GGUF model, launches it
  locally, exposes an OpenAI-compatible endpoint, and — when a control plane is
  reachable — registers, advertises a signed capability, and heartbeats. Falls
  back cleanly to a local-only SOLO mode when no control plane is reachable.
- **Contribution / idle policy engine** (`node/daemon/meshcompute_node/contribution.py`):
  idle-only gating on user input / CPU / GPU / AC power, plus resource budgets
  (CPU thread count, VRAM percent, RAM cap) enforced before the inference engine
  launches. Exposed via `mesh node start --idle-only/--idle-minutes/
  --pause-on-activity/--max-vram/--max-cpu/--max-ram/--require-ac-power`.
- **Control plane** (`apps/control-plane/meshcompute_control/app.py`,
  `uvicorn meshcompute_control.app:app`): node registration, signed capability
  intake, heartbeats, signed model manifest catalog (loaded from
  `models/manifests/*.yaml` at startup), topology-aware scheduling
  (`/api/v1/schedule`), work-receipt/credit ledger, and rendezvous
  announce/connect/relay endpoints.
- **Topology-aware scheduler** (`apps/control-plane/meshcompute_control/scheduler.py`,
  `topology.py`): scores single/pipeline/replica execution plans from measured
  RTT, VRAM, RAM, and live queue depth; logs a human-readable decision trace with
  every rejected alternative. Verified in `tests/scheduler/` to pick different
  plans under different simulated network conditions, including routing to an
  idle replica over a busy faster one.
- **Gateway** (`apps/gateway/meshcompute_gateway/app.py`, `mesh api serve`):
  OpenAI-compatible `/v1/models` and `/v1/chat/completions` (streaming and
  non-streaming), plus a native event-sourced session API
  (`/api/v1/sessions/...`). Enforces the tool boundary — backend requests are
  built from an explicit inference-only field whitelist, never from client
  headers/cookies/credentials.
- **CLI** (`apps/cli/meshcompute_cli/main.py`, the `mesh` entry point): `login`,
  `models`, `chat` (streaming REPL), `bench` (TTFT/decode tok-per-sec report),
  `node start`/`node status`, `api serve`, `pool list`/`pool create`,
  `manifest sign`/`manifest verify`.
- **QUIC peer transport** (`node/daemon/meshcompute_node/transport_quic.py`,
  built on `aioquic`): encrypted, multiplexed direct connections with an
  application-level PING/PONG RTT convention; NAT hole-punching is best-effort,
  relay fallback is a stub.
- **Rendezvous** (`apps/control-plane/meshcompute_control/rendezvous.py`,
  `node/daemon/meshcompute_node/rendezvous_client.py`): internet-native peer
  announce/connect with a WebSocket relay fallback path — no VPN, no tailnet.
- **Content-addressed model chunk swarm** (`node/runtime/meshcompute_runtime/swarm.py`):
  BitTorrent-style BLAKE3-verified chunk transfer with resume support and
  rejection of corrupt chunks.
- **Protocol layer** (`packages/protocol/meshcompute_protocol/`): Ed25519 node
  identity, signed model manifests (`mesh manifest sign`/`verify`), signed
  capability records, and wire frame encoding — the frozen HTTP/data contract
  every other component implements against.
- **Public model catalog, first target**: `public/qwen3.8-27b-fable`
  (`models/manifests/public-qwen3.8-27b-fable.yaml`) — a 27B qwen35 GGUF model
  with pinned upstream revision, tokenizer, chat template, and tool parser.
- **`scripts/register_external_backend.py`**: bridges an already-running
  OpenAI-compatible endpoint (e.g. an existing LM Studio/vLLM/`llama-server`) into
  the mesh as a registered, capability-advertising node — the path used to
  benchmark the real GPU pod below.
- **Decentralized credit ledger** (`packages/protocol/meshcompute_protocol/ledger.py`,
  `docs/DECENTRALIZATION.md`): a blockchain-style, no-single-owner ledger for
  compute credit — BLAKE3 hash-linked blocks with a Merkle root of entries,
  Ed25519-signed by their producer, independently verifiable from genesis
  (including re-verifying the signed `WorkReceipt` behind every credited entry),
  chain-wide double-credit prevention via challenge-nonce tracking, and
  longest-valid-chain fork choice. Deliberately no proof-of-work, no mining, no
  token — credits stay internal and non-transferable. The block/chain data model,
  validation, and fork choice are built and tested
  (`tests/protocol/test_ledger.py`, 6 tests); gossiping blocks between peers over
  QUIC is the next step and is not implemented yet — today the control plane's
  SQLite ledger is the only running instance.
- **Multi-GPU node support**: real per-GPU enumeration (index, VRAM, NVLINK
  peering) via `nvidia-smi`, a per-GPU on/off selection shared between capability
  reporting and resource budgeting (`node/daemon/meshcompute_node/contribution.py`'s
  `parse_gpu_devices`/`per_gpu_free_vram_bytes`), and `llama-server` launch-arg
  construction for tensor-splitting one model across the selected cards
  (`node/runtime/meshcompute_runtime/backends/llamacpp.py`'s `gpu_launch_args`:
  `CUDA_VISIBLE_DEVICES`, `--split-mode`, `--main-gpu`, `--tensor-split`
  proportional to each GPU's free VRAM). `CapabilityRecord` sums VRAM across all
  of a node's GPUs, so the scheduler already treats a multi-GPU box (e.g. a
  2×RTX 3090 pod) as one larger pod. Exposed on the CLI as
  `mesh node start --gpu-devices/--split-mode/--tensor-split`. Untested so far
  against real multi-GPU hardware in this repo — the launch-arg construction
  itself is unit-tested against a captured `nvidia-smi` sample.
- **docs/PHYSICS.md** and **docs/BENCHMARKS.md**: the mechanisms behind "more
  peers = faster" (replica load-balancing, speculative decoding, prefill
  parallelism, LAN-class tensor/pipeline split, swarm distribution), the one hard
  case that doesn't get faster (per-token WAN layer-splitting) and why, and real
  measured numbers including a live cross-Pacific WAN run (Hong Kong client → Las
  Vegas 2×RTX 3090 pod, 255 ms RTT, 36.5 tok/s steady-state decode).
- **44 automated tests** (`python -m pytest`) across the protocol (including the
  ledger), scheduler, and security/tool-boundary layers, all passing.

### Known limits (tracked, not silently missing)

- (superseded above) `mesh node start --model` and the model picker now exist.
- No prebuilt Linux+CUDA `llama.cpp` release exists upstream; Linux+NVIDIA
  uses the Vulkan prebuilt unless `--accel cuda-build` is given.
- Multi-NAT hole-punching between two independently-NATed real hosts is
  implemented but untested at that scale; the relay fallback is a stub.
- The `llama.cpp` RPC layer-split path (`--rpc`) is implemented in the backend
  adapter but not yet wired to the scheduler's `ExecutionPlan.peer_chain`
  automatically — it's a manual runbook today (`docs/DEPLOYMENT.md`).
- The gateway resolves a scheduled node_id to a backend URL via
  `deploy/nodes.local.yaml`, maintained by the operator — there is no automatic
  bridge yet from "a node registered with the control plane" to "the gateway can
  route to it."
- The credit ledger has no peer gossip/replication yet (single in-process
  instance today).

See `ROADMAP.md` for what's next (Phase 1.5 topology-aware scheduler hardening,
Phase 2 public platform/accounts/credits).
