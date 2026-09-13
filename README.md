# MeshCompute

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](#requirements)
[![Status: Proof of concept](https://img.shields.io/badge/Status-Proof%20of%20concept-orange.svg)](#status-what-is-real-today)
[![Linux · macOS · Windows](https://img.shields.io/badge/Platforms-Linux%20%C2%B7%20macOS%20%C2%B7%20Windows-555.svg)](#install)
[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue.svg?logo=paypal)](https://www.paypal.com/donate/?business=7DQWLBARMM3FE&no_recurring=0&item_name=Support+the+development+and+growth+of+innovative+MR_TBOT+projects.&currency_code=USD)

**A peer-to-peer AI inference network. Proof of concept.**

One command turns a spare machine — Linux, macOS or Windows, with or without an
NVIDIA GPU — into a node that fetches a model from its peers, serves it locally
through an OpenAI-compatible endpoint, and lends idle CPU, RAM, storage and GPU
time to everyone else on the mesh. Models are distributed BitTorrent-style,
model updates roll across the network organically, and contribution is tracked
on a signed, verifiable ledger. No VPN, no tailnet, no account with anyone.

- [Read this first: the disclaimer](#-read-this-before-you-run-it)
- [Install (Linux · macOS · Windows)](#install)
- [First run](#first-run-start-a-node)
- [Contribution controls](#contribution-controls)
- [Model channels and updates](#model-channels-and-updates)
- [Run your own mesh](#run-your-own-mesh)
- [Security posture](#security-posture)
- [Known limits](#known-limits)
- [Troubleshooting](#troubleshooting)
- [Roadmap and hardening](#roadmap-and-hardening)
- [Contributing](#contributing)
- [Support the project](#support-the-project)

---

## ⚠️ Read this before you run it

**MeshCompute is a proof of concept.** It is not a product, it is not hardened,
and it should not be run on a machine you cannot afford to have compromised.
More hardening is coming to address the concerns below; none of it is done yet.

**Running AI models on a peer-to-peer network raises questions that nobody has
good answers to yet, and this project does not pretend to.** Please read all of
this and decide for yourself.

**What you are actually doing when you run a node**

- You download and execute model weights that came from other participants.
  Every file is BLAKE3-verified against a manifest signed by the network
  operator, so what you run is what the operator published — but you are
  trusting that operator, and you are running an unsandboxed inference process
  (`llama-server`) as your own user.
- You serve inference for strangers. The requests that reach your node come
  from the network's scheduler, not from people you know.
- Your machine opens a UDP port for the peer data plane and talks to other
  machines on the open internet.
- Prompts and outputs are **not end-to-end encrypted** in Phase 1. The operator
  of whichever node serves a request can read that request. Never send secrets,
  credentials, or private data through a public pool.

**The rogue-AI concern — a genuinely unexplored problem**

A model, or an agent built on top of one, that runs across many volunteer
machines with no single party able to stop it is a safety concern that the
field has barely started to examine. We think it is one of the most important
open questions about systems like this one, and we want to be honest about
where this project stands:

- Phase 1 has a central control plane (scheduler, catalog, rendezvous). That is
  a de-facto choke point: take it down and no new work is scheduled and no new
  model versions are published.
- But nodes also run in **SOLO mode** with no control plane at all, peers
  exchange model data directly, and any node can be pointed at any control
  plane. **There is no network-wide kill switch today.** A node operator can
  always stop their own node (`Ctrl-C`, kill the process, delete `~/.mesh`).
  Nobody can stop everyone's.
- No agent tooling (web browsing, code execution, MCP tools) runs on public
  nodes in Phase 1. Nodes execute a single, catalog-pinned language model and
  nothing else. That is a deliberate limit, not an accident.

What we intend to do about it, in the order we intend to do it — see
[Roadmap and hardening](#roadmap-and-hardening):

1. A **centralized kill switch**: a signed network-halt message that the
   control plane can issue and that every node honours, including nodes that
   are mid-request, with a cool-down before they may rejoin.
2. **Channel-level model revocation**: publishing a "revoked" version of a
   channel makes every follower unload and delete that model.
3. **Capability gating**: no agent tooling on public nodes without explicit,
   per-node, per-capability opt-in that the scheduler can see and audit.
4. **Audit trail**: every unit of work already produces a signed receipt;
   these will be queryable so that what ran where, and when, can be
   reconstructed after the fact.

If you work on AI safety, distributed-systems security, or have thought hard
about any of this, **we want your input** — open an issue or a discussion.

**The boring but important part**

- You are responsible for complying with the laws of your jurisdiction, with
  the license of every model you choose to run or serve, and with the terms of
  wherever you got it (the public catalog points at Hugging Face).
- This software is provided under the [Apache-2.0 license](LICENSE) with **no
  warranty of any kind**. If it breaks your machine, eats your disk, or serves
  something you did not expect, that risk is yours.
- Do not run it on a machine that holds data you would not want sitting next to
  an internet-exposed service. A spare box, a VM, or a dedicated user account
  is the right place for it today.

---

## What it does

- **One command, any desktop OS.** `mesh node start` detects your hardware,
  self-installs the right upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
  build (CPU, Vulkan, CUDA or Metal), lets you pick a **model channel**, gets
  the model, and exposes a local OpenAI-compatible endpoint. No LM Studio, no
  Ollama, no admin rights, nothing installed outside the checkout and `~/.mesh`.
- **Every machine counts.** An NVIDIA GPU is used when present. A machine with
  no GPU still contributes CPU threads, RAM, and disk (it caches and seeds model
  files to other peers), and still gets its own local endpoint.
- **Peer-to-peer model distribution.** The second node to join a channel fetches
  the model from the first over QUIC in BLAKE3-verified chunks and seeds it on.
  Hugging Face is only the fallback. Downloads resume; corrupt chunks are
  rejected and re-fetched from another peer.
- **Model channels with organic updates.** The operator pushes a new version of
  a channel; every node following it downloads the new files — from peers first
  — swaps its engine on the same port, and **deletes the old version** so
  machines never fill with stale weights. One node per update wave pays the
  origin download; the rest wait for it and fetch peer-to-peer.
- **Contribute on your terms.** Idle-only by default, with caps on CPU, VRAM,
  RAM and storage, AC-power gating for laptops, and per-card GPU selection on
  multi-GPU boxes. Pausing contribution never takes away your own local
  endpoint.
- **Topology-aware scheduling.** The control plane picks between
  single / replica / pipeline / tensor / speculative plans from measured RTT,
  VRAM, queue depth and what each node advertises it can do, and logs why.
- **Signed everything.** Node ids derive from Ed25519 keys; registration,
  heartbeats, capabilities, rendezvous and work receipts are signed; receipts
  are credited only against a plan the scheduler issued; manifests are signed
  and content-addressed.
- **A verifiable credit ledger, not a coin.** BLAKE3 hash-linked, Ed25519-signed
  blocks of verified receipts, independently checkable from genesis, with no
  mint path outside receipt-backed work. No proof-of-work, no token, nothing
  transferable.

## Status: what is real today

Everything in the left column has a runtime path in this repo and has been
run, with the log lines quoted in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Verified on real hardware | Implemented, not yet verified on real hardware |
|---|---|
| Linux x86_64, CPU-only node: install, model picker, download, local endpoint | **Windows** (CUDA 12.4 and CPU builds) |
| Peer-to-peer model fetch between two nodes (259/259 chunks, verified) | **macOS** (Metal, Apple Silicon and Intel) |
| Channel updates v1 → v5: engine swap on the same port, old file deleted, one origin download per wave | **Linux + NVIDIA** (Vulkan prebuilt, opt-in CUDA source build) |
| Control plane + gateway + node end-to-end over a real cross-Pacific WAN (255 ms RTT), 36.5 tok/s decode | Multi-GPU tensor split (unit-tested only) |
| Signed register / heartbeat / receipts rejected when unsigned or forged | NAT hole-punching across two real NATs (LAN and loopback verified; relay is a stub) |
| Docker image builds; both services import | |
| 120 automated tests, lint clean | |

The first run on a real Windows, macOS or NVIDIA machine **is** the
verification for that platform. If that is you, please
[report the `[mesh] engine ready:` line](#contributing).

---

## Install

### Requirements

| | |
|---|---|
| **Python** | 3.12 or newer, on `PATH` |
| **git** | to clone the repo |
| **Disk** | ~0.5 GB for the engine and the small default model; 24 GB+ if you follow a large channel. `--max-storage` caps what a node keeps. |
| **GPU** | Optional. NVIDIA only for inference acceleration today. No GPU = CPU inference plus RAM and storage contribution. |
| **Network** | Outbound HTTPS (GitHub releases, Hugging Face). One UDP port for the peer data plane (any free port by default). |

Nothing needs admin or root. The installers create `.venv/` inside the checkout
and `pip install -e .` into it; the node keeps everything else under `~/.mesh`
(`%USERPROFILE%\.mesh` on Windows). Uninstalling is deleting those two folders.

### Linux

Debian / Ubuntu (24.04 ships Python 3.12):

```bash
sudo apt install -y python3.12 python3.12-venv git
git clone https://github.com/mr-tbot/MeshCompute.git
cd MeshCompute
./scripts/install.sh
source .venv/bin/activate
mesh node start
```

Fedora: `sudo dnf install -y python3.12 git`. Arch: `sudo pacman -S python git`.
Other distros: any Python ≥ 3.12 works; the installer looks for `python3.12`,
`python3`, then `python`.

**Linux + NVIDIA GPU.** The node uses the upstream **Vulkan** build of
`llama.cpp`, which runs on the NVIDIA driver with no CUDA toolkit. It needs the
Vulkan loader from your distro:

```bash
sudo apt install -y libvulkan1          # Debian / Ubuntu
sudo dnf install -y vulkan-loader        # Fedora
sudo pacman -S vulkan-icd-loader         # Arch
```

Without it the node says so and falls back to the CPU build. For CUDA-native
speed, `mesh node start --accel cuda-build` compiles `llama.cpp` once (needs
`git`, `cmake`, a C++ toolchain and `nvcc`); upstream ships no Linux+CUDA
prebuilt. Alpine / musl is not supported (upstream builds are glibc).

### macOS

Apple Silicon or Intel:

```bash
xcode-select --install                   # git, if you don't have it
brew install python@3.12                 # or the installer from python.org
git clone https://github.com/mr-tbot/MeshCompute.git
cd MeshCompute
./scripts/install.sh
source .venv/bin/activate
mesh node start
```

The node uses the upstream macOS build with **Metal** on and every layer
offloaded to the GPU. Nothing to configure. Idle detection uses `IOHIDSystem`;
AC-power detection uses `pmset`.

### Windows

Windows 10 / 11, x64 or arm64, PowerShell 5.1 or 7:

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
git clone https://github.com/mr-tbot/MeshCompute.git
cd MeshCompute
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
.\.venv\Scripts\Activate.ps1
mesh node start
```

`-ExecutionPolicy Bypass` applies to that one command only; nothing is changed
system-wide. If `mesh` is not found after activation, run
`.\.venv\Scripts\mesh.exe node start`.

**Windows + NVIDIA GPU.** The node downloads the upstream **CUDA 12.4** build
plus its `cudart` DLL bundle. Needs an NVIDIA driver ≥ 550 (check with
`nvidia-smi`). No GPU, or arm64: the upstream CPU build is used. Idle detection
uses `GetLastInputInfo`; AC power uses `GetSystemPowerStatus`.

### What gets installed where, per platform

| Platform | Engine build the node fetches | Where it lives |
|---|---|---|
| Linux x64 / arm64, no GPU | upstream `bin-ubuntu-<arch>` (CPU) | `~/.mesh/runtime/` |
| Linux + NVIDIA | upstream `bin-ubuntu-vulkan-<arch>`; `--accel cuda-build` compiles CUDA | `~/.mesh/runtime/` |
| macOS arm64 / x64 | upstream `bin-macos-<arch>`, Metal on | `~/.mesh/runtime/` |
| Windows x64 + NVIDIA | upstream `bin-win-cuda-12.4-x64` + `cudart` bundle | `%USERPROFILE%\.mesh\runtime\` |
| Windows x64 / arm64, no GPU | upstream `bin-win-cpu-<arch>` | `%USERPROFILE%\.mesh\runtime\` |

Model files go to `~/.mesh/models/`, the node identity to `~/.mesh/identity/`,
engine logs to `~/.mesh/logs/`. `MESH_HOME=/some/path` relocates all of it
(useful for a second node on one box or a bigger disk).

The daemon prints which build it ended up with:

```
[mesh] engine ready: vulkan build b10948 at /home/you/.mesh/runtime/...
```

`--accel {auto,cuda,cuda-build,vulkan,metal,cpu}` or `MESH_ACCEL=...` forces a
choice. An unavailable choice degrades down the cascade
(`cuda-build → vulkan → cpu` on Linux) and says so.

---

## First run: start a node

```bash
mesh node start
```

What happens, in order:

1. **Hardware detection and engine install** (table above). Verified to run,
   then cached; nothing is compiled unless you ask.
2. **Pick a model channel.** On a terminal you get the catalog:

   ```
   Available model channels:
     1. public/qwen3.8-27b-fable  v1  24.0 GB  [needs 24 GB, you have 30]  Qwen3.8-27B ...
     2. public/smollm2-360m       v1   0.3 GB  [fits]  SmolLM2 360M Instruct (CPU smoke channel)
     0. built-in smoke model (SmolLM2-360M, CPU-friendly)
   Select model to contribute to and use [0]:
   ```

   The choice is saved in `~/.mesh/models/state.json`; later starts don't ask.
   `--model <alias>` skips the prompt (`--model smoke` = the built-in tiny
   model). `mesh node models` lists channels with size and whether they fit.
3. **Get the model.** From peers seeding that channel first (BLAKE3-verified
   chunks over QUIC), Hugging Face as the fallback (resumable; `HF_TOKEN` is
   honoured for gated repos). Every file is verified against the signed manifest
   before use.
4. **Start the engine** on a fixed local port (`8099`; `--port` to change):

   ```
   [mesh] Local endpoint ready: http://127.0.0.1:8099/v1  (point your OpenAI client here)
   ```
5. **Register with a control plane** at `http://127.0.0.1:8080` (`--control-url`
   to change). None reachable → **SOLO mode**: your local endpoint still works,
   you are just not contributing or receiving channel updates.

### Talk to it

Any OpenAI client works against the local endpoint:

```bash
curl http://127.0.0.1:8099/v1/models
curl http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "public/smollm2-360m", "messages": [{"role": "user", "content": "Say hi in 3 words."}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8099/v1", api_key="unused")
print(client.chat.completions.create(model="public/smollm2-360m",
      messages=[{"role": "user", "content": "hello"}]).choices[0].message.content)
```

### Stop, restart, uninstall

- **Stop:** `Ctrl-C` in the terminal. The engine is stopped with it.
- **Restart:** `mesh node start` again. The engine and model are cached; the
  saved channel is reused; nothing is downloaded twice.
- **Run as a service:** Linux unit files are in `infra/systemd/` (see
  [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)). Windows and macOS service
  wrappers are not shipped yet; run it in a terminal or a `tmux`/`screen`
  session.
- **Uninstall:** delete the `MeshCompute` checkout and `~/.mesh`
  (`%USERPROFILE%\.mesh` on Windows). Nothing else was touched.

---

## Contribution controls

By default a node donates a share of the machine to the network **only once
the machine is idle**. Every flag below exists (`mesh node start --help`):

| Flag | Default | Meaning |
|---|---|---|
| `--model ALIAS` | ask / saved | Channel to contribute to and use; `smoke` = built-in tiny model |
| `--quant ID` | manifest's first | Quantization from the manifest (e.g. `Q6_K-MAX`) |
| `--port N` | `8099` | Local OpenAI endpoint port (stable across model updates) |
| `--quic-port N` | `0` (any) | UDP port of the peer data plane |
| `--accel` | `auto` | Engine build: `auto`, `cuda`, `cuda-build`, `vulkan`, `metal`, `cpu` |
| `--idle-only` / `--no-idle-only` | `--idle-only` | Only contribute while this machine is idle |
| `--idle-minutes N` | `10` | Minutes without keyboard/mouse input before contributing |
| `--pause-on-activity` / `--no-pause-on-activity` | on | Pause the instant input is detected |
| `--max-vram N` | `85` | Max % of VRAM to contribute |
| `--max-cpu N` | `50` | Max % of CPU (→ `--threads` of the engine) |
| `--max-ram N` | unset | Max RAM in GB (bounds the context size) |
| `--max-storage N` | `100` | Max GB of model files kept and seeded |
| `--require-ac-power` | off | Laptops: only contribute on mains power |
| `--gpu-devices` | `auto` | Which NVIDIA cards to donate, e.g. `0,1` or `0` |
| `--split-mode {layer,row}`, `--tensor-split` | `layer`, proportional | Multi-GPU split |
| `--update-poll N` | `300` | Seconds between channel-update checks |
| `--backend lmstudio --backend-url URL` | – | Register an existing OpenAI-compatible server instead of the embedded engine |
| `--pool NAME` | `public` | Pool to contribute to |
| `--control-url URL` | `http://127.0.0.1:8080` | Control plane |

Idle detection is real on all three OSes (`node/daemon/meshcompute_node/sysinfo.py`):
X11 `xprintidle` on Linux (Wayland has no input signal; the CPU/GPU/AC gates
still apply), `IOHIDSystem` on macOS, `GetLastInputInfo` on Windows. Being
"paused" only stops advertising availability to the *network* — your own local
endpoint keeps working. The same file-based settings are documented in
[`CONFIGURATION.md`](CONFIGURATION.md).

---

## Model channels and updates

A catalog alias such as `public/smollm2-360m` is a **channel**: a class of
model whose *current version* the catalog publishes. You follow one channel per
node. When the operator pushes a new version of that channel — a newer build,
a better quantization, the same model class — every node following it, on its
next poll (`--update-poll`, default 5 min):

1. downloads the new files in the background — from peers that already have
   them, else Hugging Face — while the old engine keeps serving;
2. suspends its network heartbeat (the scheduler routes around it), stops the
   old engine and starts the new one on the **same port**;
3. **deletes the old version's files**, so a machine following a channel holds
   exactly one copy. Your local endpoint is down for one engine load (seconds
   for a small model, about a minute for a 24 GB one).

If the new engine fails to start, the node restarts the old version and keeps
its files. A catalog offering a *lower* version than a node runs is ignored,
and the control plane refuses to publish version regressions.

An update wave does not stampede the origin. The rendezvous tracker names one
**origin leader** per file (the node that started wanting it first, on the
tracker's clock) and tells everyone else which peer is already fetching it.
Followers wait (up to 30 min, scaled to file size) for the leader to finish and
seed, then fetch from it peer-to-peer. What that looks like in a node's log:

```
[mesh] channel public/smollm2-360m: v4 -> v5 (03fe7ba8f538) — downloading in the background
[mesh] peer nd_e7e236db2 is already fetching SmolLM2-360M-Instruct-Q4_K_M.gguf from origin; waiting to get it from that peer instead
[mesh] fetching SmolLM2-360M-Instruct-Q4_K_M.gguf from peer nd_e7e236db2 @ 10.x.x.x:41159 (rtt 0.93 ms)
[mesh]   SmolLM2-360M-Instruct-Q4_K_M.gguf: 259/259 chunks
[mesh] SmolLM2-360M-Instruct-Q4_K_M.gguf obtained from peers (verified)
[mesh] channel public/smollm2-360m: now serving v5; deleted 1 old file(s): ['SmolLM2-360M-Instruct-Q4_K_S.gguf']
```

Channels are for **new versions of the same model class**. Replacing a channel
with an entirely different model is technically the same operation, but every
follower downloads the full new model, so operators treat that as a new channel.

---

## Run your own mesh

A mesh is a control plane, a gateway, and any number of nodes on any machines
that can reach the control plane.

**1. Control plane** (default `127.0.0.1:8080`):

```bash
uvicorn meshcompute_control.app:app --host 127.0.0.1 --port 8080
```

It loads every manifest in `models/manifests/*.yaml`, signs them, and re-scans
the directory every 30 s. Dropping a manifest with a higher `version` in there
**is** how you push a channel update.

**2. Gateway** (default `127.0.0.1:8081`), the OpenAI-compatible front door:

```bash
mesh api serve
```

**3. Nodes**, anywhere:

```bash
mesh node start --model public/smollm2-360m --control-url http://<control-plane>:8080
```

The second and later nodes fetch the model from the first ones (watch for
`fetching ... from peer` and `obtained from peers (verified)`), and every node
seeds what it holds.

**4. Wire nodes into the gateway's routing table.** The scheduler picks *which*
node serves a request; the gateway resolves that node id to an HTTP endpoint
from `deploy/nodes.local.yaml` (gitignored). Copy the template and add each
node's `node_id` (printed at start) and `backend_url`:
`cp deploy/nodes.example.yaml deploy/nodes.local.yaml`. Edits are picked up
without restarting the gateway.

**5. Use it:**

```bash
mesh models
mesh chat --model public/smollm2-360m
mesh bench --model public/smollm2-360m --tokens 128
```

Point any OpenAI client at `http://127.0.0.1:8081/v1`. `mesh chat` prints the
scheduler's `X-Mesh-Strategy` / `X-Mesh-Path` per turn.

**Docker Compose** for the control plane and gateway (nodes run on contributor
machines): `docker compose -f infra/compose/docker-compose.yml up -d`.
**systemd** units for all three: `infra/systemd/`. Environment variables, the
routing-table format, external backends, and the step-by-step for pushing a
channel update are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Pushing a model update (operator)

1. Edit the channel's manifest in `models/manifests/`: bump `version`, point
   at the new file. `mesh manifest pin <yaml>` fills in size and BLAKE3 from a
   local copy of the file.
2. Save it into the control plane's manifest directory. Within 30 s it logs
   `channel <alias> -> v<N> (<hash>)`.
3. Nodes move over on their next poll, one origin download per wave, old files
   deleted. That's it.

### Other CLI commands

```bash
mesh node models                  # channels, sizes, fits-here, installed version
mesh node status                  # nodes known to the control plane
mesh pool list / mesh pool create NAME [--private]
mesh manifest pin PATH.yaml       # fill real BLAKE3 + sizes from local files
mesh manifest sign PATH.yaml      # -> PATH.signed.json (same scheme the control plane uses)
mesh manifest verify PATH.signed.json
mesh login --token <TOKEN>        # Phase 1: auth is a stub
```

---

## Security posture

What is protected today:

- Node identity is an Ed25519 key; the node id is derived from it, so a node id
  cannot be claimed without the key.
- Registration, heartbeats, capability reports, rendezvous announces and
  connect requests are signed and rejected otherwise.
- Work receipts are credited only when they reference a plan the scheduler
  actually issued, come from a node in that plan, carry a nonce the control
  plane handed out, and pass a plausibility check on the claimed GPU time.
- The ledger has no mint path outside receipt-backed work; corrections can only
  go down.
- Model manifests are signed by the control plane and every file is
  content-addressed (BLAKE3); a peer that serves a bad chunk is dropped and the
  chunk re-fetched elsewhere.
- The gateway refuses (409) to stream from a node that reports a different
  model than the one requested, and refuses (501) execution plans it cannot
  honour rather than silently degrading.

What is **not** protected today, and you should assume:

- Prompts and completions are readable by the node that serves them and by
  anyone on the path between gateway and node (plain HTTP inside your
  deployment).
- The control plane is fully trusted: it signs the catalog, issues plans, and
  keeps the ledger. Compromise it and you control what the network runs.
- The inference engine is not sandboxed beyond being a separate process.
- Authentication on the gateway is a stub (`mesh login` stores a token; nothing
  enforces it yet).
- Hole-punching is best-effort and the relay data path is a stub, so a node
  behind NAT may not be reachable by peers even though it can reach them.

The full threat model is in [`SECURITY.md`](SECURITY.md).

---

## Known limits

- **Linux + NVIDIA runs the Vulkan build by default.** CUDA-native needs
  `--accel cuda-build` and a toolkit. Vulkan multi-GPU selection uses
  `GGML_VK_VISIBLE_DEVICES` with the same indices as `nvidia-smi`, which holds
  on single-vendor boxes.
- **AMD and Intel GPUs are not used for inference yet.** Those machines take the
  CPU path and still contribute CPU, RAM and storage.
- **Windows and macOS are not yet verified on real hardware in this repo.** The
  asset selection, extraction and host-signal code paths are unit-tested against
  upstream release names and the documented OS APIs; the first real run is the
  verification.
- **The gateway executes single-node plans only.** Multi-peer plans (pipeline,
  tensor, speculative) are scheduled but refused at the gateway until the QUIC
  inference session lands. Splitting one token's decode across a high-latency
  WAN is deliberately not attempted — [`docs/PHYSICS.md`](docs/PHYSICS.md)
  explains why.
- **No automatic node → gateway bridge.** The operator maintains
  `deploy/nodes.local.yaml`.
- **Public catalog hashes.** `public/smollm2-360m` is fully pinned.
  `public/qwen3.8-27b-fable` still carries `PENDING` hashes until someone with
  the 24 GB files runs `mesh manifest pin`.
- **One channel per node.** Following two channels means running two nodes
  (use `MESH_HOME` and `--port`).

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| `python3.12 not found` / installer refuses | Install Python ≥ 3.12 (commands per OS above) and re-run the installer. It is idempotent. |
| `mesh: command not found` | Activate the venv (`source .venv/bin/activate` / `.\.venv\Scripts\Activate.ps1`) or call `.venv/bin/mesh` directly. |
| Node says it fell back to the CPU build on a GPU box | Linux: install `libvulkan1` (or your distro's Vulkan loader). Windows: update the NVIDIA driver to ≥ 550. Force a build with `--accel` to see the exact failure. |
| Model download is slow or fails from Hugging Face | Gated repo → `export HF_TOKEN=...`. Downloads resume; just restart. Peers are tried first when the tracker knows any. |
| `[needs N GB, you have M]` in the picker | Free up disk, raise `--max-storage` (default 100 GB), or pick a channel that fits. The node refuses to exceed its storage budget or free space (1 GB margin). |
| Port `8099` already in use | `--port <other>`; the port is stable across updates. |
| Peers never connect to me | Your UDP port is not reachable from outside (NAT). Set `--quic-port N` and forward it, or accept that you download from peers but they cannot download from you. |
| Registered but never receives work | Contribution is idle-gated by default: `--no-idle-only` to test, or wait `--idle-minutes`. On Wayland there is no idle signal; the other gates still apply. |
| Something else | Run with the same flags again and open an issue with the full `[mesh]` log. |

---

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Copy-pasteable setup, solo node or full mesh, every flag |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Operator runbook: env vars, routing table, channel pushes, systemd, Compose, security posture |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | Why "more peers = faster" is real, and why per-token WAN splitting isn't |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Measured numbers and the log lines behind every claim above |
| [`docs/DECENTRALIZATION.md`](docs/DECENTRALIZATION.md) | The credit ledger: what's on-chain, what isn't, and why there's no coin |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Contribution / idle / resource config reference |
| [`PROTOCOL.md`](PROTOCOL.md) | Wire protocol: QUIC framing, swarm chunk exchange, rendezvous |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`API.md`](API.md) · [`SECURITY.md`](SECURITY.md) · [`MODEL_CATALOG.md`](MODEL_CATALOG.md) | Design documents |
| [`ROADMAP.md`](ROADMAP.md) · [`CHANGELOG.md`](CHANGELOG.md) | What's built vs. planned; what shipped when |

---

## Roadmap and hardening

Phase 1 (this repo, working): standalone nodes on every desktop OS, P2P model
distribution, channel updates, signed identities and receipts, the ledger, the
scheduler, single-node execution through the gateway.

Next, roughly in order — the full list is in [`ROADMAP.md`](ROADMAP.md):

- **Kill switch and revocation** (see the disclaimer above): signed network
  halt honoured by every node; channel-level model revocation; a public
  status page that shows whether a halt is in force.
- **QUIC inference session** so multi-peer plans (pipeline / tensor /
  speculative) actually execute, not just get scheduled.
- **Node → gateway bridge**: nodes advertise their endpoint; no hand-maintained
  routing table.
- **NAT traversal that works across two real NATs** and a relay data path for
  when it doesn't.
- **Ledger gossip** between nodes so the credit chain is not held by the
  control plane alone.
- **Capability gating and audit** before any agent tooling touches public
  nodes.
- **Real-hardware verification** on Windows, macOS and NVIDIA boxes — this is
  where you come in.

---

## Contributing

This is one developer plus AI tooling, shipping a proof of concept in public.
Community input is what turns it into something safe enough to use. Ways to
help, most valuable first:

1. **Run it on hardware we don't have.** Windows, macOS, any NVIDIA box, a
   multi-GPU box. Open an issue titled with your OS and GPU and paste the
   `[mesh] engine ready:` line and anything that broke. That single line
   moves a platform from "implemented" to "verified".
2. **Think about the safety problem.** If you have a view on how a network like
   this should be stoppable, auditable, or gated, open a discussion. This is
   the part we most want to get right and most want help with.
3. **Code.** Pull requests are welcome. The codebase is Python 3.12; the test
   suite and linter are the bar:

   ```bash
   ./scripts/install.sh && source .venv/bin/activate
   python -m pytest -q                                  # 120 tests
   ruff check --select E,F,W --line-length 110 .        # lint
   ```

   Good first areas: the Windows/macOS service wrappers, the node → gateway
   bridge, parallel-stream swarm downloads, a `mesh node stop`. Read
   [`AGENTS.md`](AGENTS.md) for repo conventions (it is written for AI coding
   agents, and it works for humans too).
4. **Docs and translations.** If a step above did not work as written, that is
   a bug in the README. Say so.

Please **don't** open pull requests that add agent tooling (web, code
execution, MCP) to public nodes until the capability gating exists. See the
disclaimer for why.

---

## Support the project

MeshCompute is built by **one developer** with the help of AI tools. No
corporate sponsor, no VC — nights, weekends, and community feedback. If this
project is useful or interesting to you, buying us a coffee keeps the lights on
and the GPUs spinning. Every contribution, whatever the size, goes straight
into development, testing hardware, and keeping this free and open source.

### Buy us a coffee (PayPal)

[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue.svg?logo=paypal&style=for-the-badge)](https://www.paypal.com/donate/?business=7DQWLBARMM3FE&no_recurring=0&item_name=Support+the+development+and+growth+of+innovative+MR_TBOT+projects.&currency_code=USD)

[**Click here to donate via PayPal**](https://www.paypal.com/donate/?business=7DQWLBARMM3FE&no_recurring=0&item_name=Support+the+development+and+growth+of+innovative+MR_TBOT+projects.&currency_code=USD)

### Crypto

| Currency | Address |
|----------|---------|
| **BTC** | `bc1qalnp0xze5t9nner2754k2pj7yjhkrt3uzvzdvt` |
| **ETH** | `0xAd640c506f5d2368cAF420a117380820C0C5F61C` |
| **XRP** | `rpciwKrQSaRZ1UjPunH8vLJhoM2s4NaYoL` |
| **DOGE** | `DM79aRx58J6RYuWakHjiELWbNJkTTDj1cv` |

Or contribute code, hardware test results, or a hard question about safety —
all of it counts. Thank you to everyone who tests, files issues, and spreads the
word.

---

## License

[Apache-2.0](LICENSE). `llama.cpp` is MIT-licensed and downloaded at runtime
from its upstream releases; models are subject to their own licenses.

---

**Made with <3 by [codedatda.casa](https://codedatda.casa)** — the development
collective behind [MESH-API](https://github.com/mr-tbot/mesh-api),
[SofaCode](https://sofacode.app) and friends. More at
[mr-tbot.com](https://mr-tbot.com).
