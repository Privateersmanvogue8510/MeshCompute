# Quickstart

Every command below is checked against `apps/cli/meshcompute_cli/main.py` — if a
flag isn't listed here, it doesn't exist yet.

## 1. Install (Linux, macOS, Windows)

Prerequisite: **Python 3.12+** and `git`. No GPU required — a CPU-only machine
contributes CPU, RAM and storage; an NVIDIA GPU is used when present.

| OS | Install | Then |
|---|---|---|
| **Linux** | `git clone https://github.com/mr-tbot/MeshCompute.git && cd MeshCompute && ./scripts/install.sh` | `source .venv/bin/activate` |
| **macOS** (Apple Silicon or Intel) | same as Linux | `source .venv/bin/activate` |
| **Windows** (PowerShell) | `git clone https://github.com/mr-tbot/MeshCompute.git; cd MeshCompute; powershell -ExecutionPolicy Bypass -File scripts\install.ps1` | `.\.venv\Scripts\Activate.ps1` |

Both installers only create `.venv/` inside the checkout and `pip install -e .`
into it — nothing system-wide, no admin. (Manual equivalent:
`python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .`.)

### What the node installs on first run, per platform

`mesh node start` downloads a prebuilt `llama.cpp` release for your machine
into `~/.mesh/runtime/` (`%USERPROFILE%\.mesh\runtime\` on Windows), verifies
it runs, and caches it. Nothing is compiled unless you ask.

| Platform | Engine build used | Notes |
|---|---|---|
| Linux x64/arm64, no GPU | upstream `bin-ubuntu-<arch>` CPU build | glibc distros; Alpine/musl not supported |
| Linux + NVIDIA | upstream **Vulkan** build (runs on the NVIDIA driver, no CUDA toolkit needed) | upstream ships no Linux+CUDA prebuilt. For CUDA-native speed: `--accel cuda-build` compiles once (needs `git`, `cmake`, a C++ toolchain and `nvcc`) |
| Linux + NVIDIA without `libvulkan1` | falls back to the CPU build | install your distro's `libvulkan1` to get the GPU path |
| macOS arm64 / x64 | upstream macOS build, **Metal** on (all layers offloaded) | |
| Windows x64 + NVIDIA | upstream **CUDA 12.4** build + its `cudart` DLL bundle | driver ≥ 550 |
| Windows x64/arm64, no GPU | upstream `win-cpu-<arch>` build | |

The daemon prints which build it ended up with, e.g.
`[mesh] engine ready: cpu build b10948 at ...`. `--accel {auto,cuda,cuda-build,vulkan,metal,cpu}`
or `MESH_ACCEL=...` forces a choice; an unavailable choice degrades down the
cascade (cuda-build → vulkan → cpu) and says so.

`MESH_HOME` relocates everything the node stores (engine, models, identity,
logs) away from `~/.mesh` — useful for a second node on one box or a bigger disk.

---

## 2. Standalone node (no control plane, no other service)

```bash
mesh node start
```

1. Detects your accelerator and installs the engine (table above).
2. **Picks a model channel.** On a terminal it shows the catalog and asks:
   ```
   Available model channels:
     1. public/qwen3.8-27b-fable  v1  24.0 GB  [needs 24 GB, you have 30]  Qwen3.8-27B ...
     2. public/smollm2-360m  v1  0.3 GB  [fits]  SmolLM2 360M Instruct (CPU smoke channel)
     0. built-in smoke model (SmolLM2-360M, CPU-friendly)
   Select model to contribute to and use [0]:
   ```
   The choice is remembered in `~/.mesh/models/state.json`; later starts don't
   ask. `--model <alias>` skips the prompt (`--model smoke` = built-in tiny model);
   `mesh node models` lists channels with size and whether they fit here.
3. **Gets the model.** Peers seeding that channel first (BLAKE3-verified chunks
   over QUIC), Hugging Face as the fallback (resumable; `HF_TOKEN` honoured).
   Every file is verified against the signed manifest before it's used.
4. Starts `llama-server` on a **fixed local port** (`8099`, `--port` to change) and prints:
   ```
   [mesh] Local endpoint ready: http://127.0.0.1:8099/v1  (point your OpenAI client here)
   ```
5. Registers with a control plane at `http://127.0.0.1:8080` (`--control-url`
   to change). None reachable → **SOLO mode**: the local endpoint still works,
   you're just not contributing or following channel updates.

Talk to it with any OpenAI client, or `curl`:

```bash
curl http://127.0.0.1:8099/v1/models
curl http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "public/smollm2-360m", "messages": [{"role": "user", "content": "Say hi in 3 words."}]}'
```

### Model channels and updates

A catalog alias (`public/smollm2-360m`) is a **channel**: a class of model whose
current version the catalog publishes. When the operator pushes a new version of
the same channel (see `docs/DEPLOYMENT.md`), every node following it, on its next
poll (`--update-poll`, default 300 s):

1. downloads the new files in the background — from peers that already have
   them, else Hugging Face — while the old engine keeps serving;
2. suspends its network heartbeat (the scheduler routes around it), stops the
   old engine and starts the new one on the **same port**;
3. deletes the old version's files, so a machine following a channel holds
   exactly one copy. The local endpoint is down for one engine load (seconds for
   a small model, ~a minute for a 24 GB one).

If the new engine fails to start, the node restarts the old version and keeps
its files; a catalog offering a *lower* version than a node runs is ignored.

An update wave doesn't stampede the origin: the rendezvous tracker names one
**origin leader** per file (the node that started wanting it first, on the
tracker's clock) and tells everyone else which peers are already fetching it.
Followers wait (up to 30 min, scaled to the file size) for the leader to finish
and seed, then fetch from it peer-to-peer. One node per wave pays the origin
download. Polls are jittered ±30 %.

### Contribution / resource flags

`mesh node start` donates a share of this machine to the network once idle, by
default. Every flag below is real:

| Flag | Default | Meaning |
|---|---|---|
| `--model ALIAS` | ask / saved | Channel to contribute to and use; `smoke` = built-in tiny model |
| `--quant ID` | manifest's first | Quantization from the manifest (e.g. `Q6_K-MAX`) |
| `--port N` | `8099` | Local OpenAI endpoint port (stable across updates) |
| `--quic-port N` | `0` (any) | UDP port of the peer data plane |
| `--accel` | `auto` | Engine build (see table above) |
| `--idle-only` / `--no-idle-only` | `--idle-only` | Only contribute while this machine is idle |
| `--idle-minutes N` | `10` | Minutes without keyboard/mouse input before contributing |
| `--pause-on-activity` / `--no-pause-on-activity` | on | Pause the instant input is detected |
| `--max-vram N` | `85` | Max % of VRAM to contribute |
| `--max-cpu N` | `50` | Max % of CPU (→ `--threads` of the engine) |
| `--max-ram N` | unset | Max RAM, GB (bounds the context size) |
| `--max-storage N` | `100` | Max GB of model files kept/seeded; advertised as `storage_share_bytes` |
| `--require-ac-power` | off | Laptops: only on mains |
| `--gpu-devices` | `auto` | Which NVIDIA cards to donate, e.g. `0,1` or `0` |
| `--split-mode {layer,row}`, `--tensor-split` | `layer`, proportional | Multi-GPU split |
| `--update-poll N` | `300` | Seconds between channel-update checks |
| `--backend lmstudio --backend-url URL` | – | Register an existing OpenAI server instead of the embedded engine |
| `--pool`, `--control-url` | `public`, `http://127.0.0.1:8080` | |

Idle detection is real on all three OSes (`node/daemon/meshcompute_node/sysinfo.py`):
X11 `xprintidle` on Linux (Wayland: no input signal, CPU/GPU/AC gates still
apply), `IOHIDSystem` on macOS, `GetLastInputInfo` on Windows. Being "paused"
only stops advertising availability to the *network* — your own local `/v1`
endpoint keeps working.

### Known limits (Phase 1)

- **Linux + NVIDIA runs the Vulkan build by default.** CUDA-native needs
  `--accel cuda-build` and a toolkit. Vulkan multi-GPU selection uses
  `GGML_VK_VISIBLE_DEVICES` with the same indices as `nvidia-smi`, which holds
  on single-vendor boxes.
- **Windows and macOS are not yet verified on real hardware in this repo.**
  The asset selection, extraction and host-signal code paths are unit-tested
  against the upstream release names and documented OS APIs; the first run on a
  real Windows/macOS machine is the verification. Please report the
  `[mesh] engine ready:` line.
- **Public catalog hashes.** `public/smollm2-360m` is fully pinned (BLAKE3).
  `public/qwen3.8-27b-fable` still carries `PENDING` hashes — see the comment
  at the top of its manifest for the one-command pin.
- **Multi-GPU tensor-split is unit-tested, not run on a real multi-GPU box here.**

---

## 3. Run the mesh: control plane + gateway + nodes

**1. Control plane** (default `127.0.0.1:8080`):

```bash
uvicorn meshcompute_control.app:app --host 127.0.0.1 --port 8080
```

It loads every manifest in `models/manifests/*.yaml`, signs them, and
re-scans the directory every 30 s — dropping a manifest with a higher `version`
in there **is** how you push a channel update.

**2. Gateway** (default `127.0.0.1:8081`):

```bash
mesh api serve
```

**3. Nodes** — as many as you like, on any machine that can reach the control plane:

```bash
mesh node start --model public/smollm2-360m --control-url http://<control-plane>:8080
```

The second and later nodes fetch the model **from the first ones** (watch for
`[mesh] fetching ... from peer ...` and `obtained from peers (verified)`), and
every node seeds what it holds.

**4. Wire nodes into the gateway's routing table.** The scheduler picks *which*
node serves a request; the gateway resolves that node_id to an HTTP endpoint from
`deploy/nodes.local.yaml` (gitignored). Copy the template and add each node's
`node_id` (printed at start) and `backend_url` (its local endpoint **without**
`/v1`): `cp deploy/nodes.example.yaml deploy/nodes.local.yaml`. Edits are picked
up without restarting the gateway. Full shape: **docs/DEPLOYMENT.md**.

**5. Use it:**

```bash
mesh models
mesh chat --model public/smollm2-360m
mesh bench --model public/smollm2-360m --tokens 128
```

`mesh chat` prints the scheduler's `X-Mesh-Strategy`/`X-Mesh-Path` per turn. The
gateway only accepts single-node plans in Phase 1 and refuses (409) to stream
from a node that reports a different model than the one you asked for.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="unused")
for chunk in client.chat.completions.create(model="public/smollm2-360m",
        messages=[{"role": "user", "content": "hello"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

### Other CLI commands

```bash
mesh node models                  # channels, sizes, fits-here, installed version
mesh node status                  # nodes known to the control plane
mesh pool list / mesh pool create NAME [--private]
mesh manifest pin PATH.yaml       # fill real BLAKE3 + sizes from local files
mesh manifest sign PATH.yaml      # -> PATH.signed.json (same scheme the control plane uses)
mesh manifest verify PATH.signed.json
mesh login --token <TOKEN>        # Phase-1: auth is a stub
```

Operating in production (env vars, systemd, Docker Compose, channel pushes,
security posture): **docs/DEPLOYMENT.md**. The physics of "more peers = faster":
**docs/PHYSICS.md**. Measured numbers: **docs/BENCHMARKS.md**.
