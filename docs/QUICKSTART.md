# Quickstart

Two real, working flows. Both are copy-pasteable. Every command below is checked
against `apps/cli/meshcompute_cli/main.py` — if a flag isn't listed here, it doesn't
exist yet.

Prerequisite for both flows: Python 3.12+.

```bash
git clone https://github.com/mr-tbot/MeshCompute.git
cd MeshCompute
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

(Or run `scripts/install.sh`, which does the same three steps and checks your Python
version first.)

---

## A. Standalone node (no control plane, no other service)

This is the whole product in one command. It self-installs `llama.cpp`, downloads a
model, serves it locally, and exposes an OpenAI-compatible endpoint:

```bash
mesh node start
```

What happens, in order (see `node/daemon/meshcompute_node/daemon.py`):

1. Detects your accelerator (`cuda` / `metal` / `cpu`).
2. Downloads and caches a `llama-server` build for your platform (prebuilt release
   where one exists; source build otherwise — see **Known limits** below).
3. Downloads a small CPU-friendly model (`bartowski/SmolLM2-360M-Instruct-GGUF`) to
   `~/.mesh/models/` and caches it — this is the Phase-1 smoke default. (The CLI does
   not yet expose a `--model` flag to pick a different one at start time; see
   **Known limits**.)
4. Starts `llama-server` and prints the local endpoint, e.g.:
   ```
   [mesh] Local endpoint ready: http://127.0.0.1:53214/v1  (point your OpenAI client here)
   ```
5. Tries to register with a control plane at `http://127.0.0.1:8080` (override with
   `--control-url`). If none is reachable, it prints and runs in **SOLO mode** — the
   local endpoint still works, you're just not contributing to a network.

Talk to it immediately with any OpenAI client, or `curl` (replace the port with the
one printed above):

```bash
curl http://127.0.0.1:53214/v1/models

curl http://127.0.0.1:53214/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Say hi in 3 words."}]}'
```

### Contribution / idle flags

`mesh node start` donates a share of this machine to the network once idle, by
default. Every flag below is real (`apps/cli/meshcompute_cli/main.py`):

```bash
mesh node start \
  --idle-only \
  --idle-minutes 10 \
  --pause-on-activity \
  --max-vram 85 \
  --max-cpu 50 \
  --max-ram 32 \
  --require-ac-power
```

| Flag | Default | Meaning |
|---|---|---|
| `--idle-only` / `--no-idle-only` | `--idle-only` | Only contribute to the network while this machine is idle |
| `--idle-minutes N` | `10` | Minutes of no user input required before contribution starts |
| `--pause-on-activity` / `--no-pause-on-activity` | `--pause-on-activity` | Pause contribution the instant user input is detected again |
| `--max-vram N` | `85` | Max % of VRAM to contribute |
| `--max-cpu N` | `50` | Max % of CPU to contribute |
| `--max-ram N` | unset (no cap) | Max RAM, in GB, to contribute |
| `--require-ac-power` / `--no-require-ac-power` | `--no-require-ac-power` | Only contribute while on AC power (laptops) |
| `--backend {llamacpp,lmstudio}` | `llamacpp` | Inference backend — `llamacpp` self-installs; `lmstudio` requires `--backend-url` pointing at an already-running OpenAI-compatible server |
| `--pool` | `public` | Pool ID to join |
| `--gpu-devices` | `auto` | Which GPUs to donate: `auto` (every detected card), or explicit indices, e.g. `0,1` or `0` |
| `--split-mode {layer,row}` | `layer` | How to split a model across multiple selected GPUs (passed to `llama-server --split-mode`) |
| `--tensor-split` | unset (proportional to free VRAM) | Explicit per-GPU split proportions, e.g. `3,1` |
| `--control-url` | `http://127.0.0.1:8080` | Control-plane URL |

Being "paused" only stops advertising availability to the *network* — your own
local `/v1` endpoint keeps working the whole time (`contribution.py`). These map
1:1 onto the `availability`/`compute` sections of `CONFIGURATION.md`; see
**docs/DEPLOYMENT.md** for how a `deploy/nodes.local.yaml` entry mirrors them.

### Known limits (Phase 1)

- **No `--model` flag yet.** `mesh node start` always resolves to the CPU smoke
  model above. The daemon itself (`daemon.async_main`) accepts a `model=` argument
  and can resolve any alias from the control-plane catalog (e.g.
  `public/qwen3.8-27b-fable`), but the CLI doesn't forward it yet. To serve a
  specific catalog model today, either drive the daemon module directly
  (`python -c "from meshcompute_node.daemon import run; run(model='public/qwen3.8-27b-fable')"`)
  or register an existing OpenAI-compatible server that already has it loaded — see
  **docs/DEPLOYMENT.md**.
- **CUDA build.** No prebuilt Linux+CUDA `llama-server` release exists upstream
  today, so a CUDA node builds `llama.cpp` from source on first run (slower first
  start, cached after).
- **Multi-GPU.** `mesh node start --gpu-devices 0,1` selects specific cards
  (default `auto` = every detected GPU); `--split-mode`/`--tensor-split` control
  how a model is split across them. The scheduler already sums VRAM across a
  node's selected GPUs and treats a multi-GPU box (e.g. a 2×RTX 3090 NVLINK pod)
  as one larger pod. Untested on real multi-GPU hardware in this repo so far
  (no GPU box available to this checkout) — the launch-arg construction itself
  is unit-tested against a captured `nvidia-smi` sample
  (`node/runtime/meshcompute_runtime/backends/llamacpp.py`).

---

## B. Run the mesh: control-plane + gateway + a node

This is the full architecture: a control plane (identity, catalog, scheduling), a
gateway (the trusted OpenAI-compatible surface), and one or more nodes.

**1. Start the control plane** (default `127.0.0.1:8080`):

```bash
uvicorn meshcompute_control.app:app --host 127.0.0.1 --port 8080
```

It loads every manifest in `models/manifests/*.yaml` at startup — including
`public/qwen3.8-27b-fable`, the first published catalog target.

**2. Start the gateway** (default `127.0.0.1:8081`), in another terminal:

```bash
mesh api serve
```

(`mesh api serve --port 8081 --control-url http://127.0.0.1:8080` to be explicit;
both are already the defaults.)

**3. Start a node**, in another terminal:

```bash
mesh node start
```

Note the `[mesh] node identity: <node_id>` and `[mesh] Local endpoint ready: ...`
lines it prints.

**4. Wire the node into the gateway's routing table.**

The control-plane's scheduler picks *which* node_id should serve a request, but the
gateway resolves that node_id to an actual HTTP endpoint from
`deploy/nodes.local.yaml` (gitignored — never committed, no real IPs in this public
repo). Copy the template and fill in the node_id and endpoint you just saw printed:

```bash
cp deploy/nodes.example.yaml deploy/nodes.local.yaml
```

Edit the `nodes:` entry so its `node_id` is the one your node printed, and
`backend_url` is its local endpoint **without** the `/v1` suffix (e.g.
`http://127.0.0.1:53214`). Full shape and every field: **docs/DEPLOYMENT.md**.

**5. Use it:**

```bash
mesh models
mesh chat --model public/qwen3.8-27b-fable
mesh bench --model public/qwen3.8-27b-fable --tokens 128
```

`mesh models` lists whatever the gateway/control-plane currently serve.
`mesh chat` is an interactive streaming REPL against `/v1/chat/completions`
(Ctrl-C cancels a turn, Ctrl-D quits) that also prints the scheduler's
`X-Mesh-Strategy`/`X-Mesh-Path` for each turn. `mesh bench` streams one fixed
prompt and reports TTFT, total time, and decode tokens/sec.

> Because the CLI can't yet start a node loaded with a specific catalog model (see
> **Known limits** above), `mesh chat --model public/qwen3.8-27b-fable` only
> succeeds once *some* registered node can actually serve that model — either a
> big-enough node started via the daemon's `model=` argument directly, or an
> existing OpenAI-compatible endpoint (e.g. a GPU box already running the model)
> registered with `scripts/register_external_backend.py` (see
> **docs/DEPLOYMENT.md**). Against a plain `mesh node start` CPU box, chat against
> whatever model id it actually reports from `mesh models` instead (the smoke
> model, `local`).

### Point any OpenAI client at the gateway

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="unused")
resp = client.chat.completions.create(
    model="public/qwen3.8-27b-fable",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

Any client that speaks the OpenAI Chat Completions API works — the gateway never
looks at the client's own `Authorization` header when building the backend request
(`apps/gateway/meshcompute_gateway/app.py`, `SECURITY.md` tool boundary).

### Other CLI commands

```bash
mesh login --token <TOKEN>        # saves a token to ~/.mesh/config.json (Phase-1: auth is a stub)
mesh node status                  # list nodes known to the control plane
mesh pool list                    # list pools
mesh pool create NAME [--private]
mesh manifest sign PATH.yaml      # sign a ModelManifest -> PATH.signed.json
mesh manifest verify PATH.signed.json
```

More on operating this in production (env vars, systemd, Docker Compose, the
RPC layer-split runbook, security posture): **docs/DEPLOYMENT.md**. The physics
behind "more peers = faster" and its one hard limit: **docs/PHYSICS.md**. Measured
numbers: **docs/BENCHMARKS.md**.
