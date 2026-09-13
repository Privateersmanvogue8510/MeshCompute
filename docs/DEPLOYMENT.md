# Deployment / Operator Runbook

This is the operator-facing companion to `docs/QUICKSTART.md`. It covers
environment variables, the `deploy/nodes.local.yaml` shape, the optional
external-OpenAI-endpoint path, the multi-node `llama.cpp` RPC layer-split runbook,
contribution/idle config, and the security posture you're operating under.

**This is a public repo.** Nothing in this file, or anything you commit, may
contain a real IP, hostname, or credential. Every example below uses a
placeholder (`HOST`, `PEER_B_HOST`, etc.). Real values belong only in
`deploy/nodes.local.yaml` and `deploy/identities/*.key.json` — both are
`.gitignore`d.

## Processes

MeshCompute Phase 1 is three independent processes plus however many contributor
nodes join:

| Process | Default bind | Run with |
|---|---|---|
| Control plane | `127.0.0.1:8080` | `uvicorn meshcompute_control.app:app --host 127.0.0.1 --port 8080` |
| Gateway | `127.0.0.1:8081` | `mesh api serve` (wraps `uvicorn meshcompute_gateway.app:app`) |
| Node daemon | n/a (contributes; also serves a local `/v1`) | `mesh node start` |

There is no `mesh control-plane serve` command — the control plane is a plain
FastAPI app, run directly with `uvicorn` (see the `Run with:` line at the top of
`apps/control-plane/meshcompute_control/app.py`). The gateway has a CLI wrapper
(`mesh api serve`) because it's the process end users are expected to run and
point clients at.

## Environment variables

### Control plane — `MESH_CP_*` (`apps/control-plane/meshcompute_control/app.py`, `Settings`)

| Var | Default | Meaning |
|---|---|---|
| `MESH_CP_BIND` | `127.0.0.1:8080` | Informational only today — actually bind with `uvicorn ... --host --port` (the app reads this for its own `__main__` fallback, not when launched via `uvicorn <module>:app`) |
| `MESH_CP_DB` | `var/control.db` | SQLite path for node/pool/manifest/receipt state |
| `MESH_CP_MANIFEST_DIR` | `models/manifests` | Directory of signed model manifest YAMLs loaded at startup |

The control plane also reads (not env-configured) `deploy/nodes.local.yaml` at
startup, best-effort, to seed measured RTT-from-here per node_id (the
`measured_rtt_ms_from_here` field — see below). A missing or malformed file is
never fatal; the scheduler just falls back to a pessimistic WAN default for that
link.

Node identity for the control plane itself lives at
`deploy/identities/control-plane.key.json` (auto-created on first run,
gitignored).

### Gateway — mostly `MESH_GW_*` (`apps/gateway/meshcompute_gateway/settings.py`, `Settings`)

| Var | Default | Meaning |
|---|---|---|
| `MESH_GW_BIND` | `127.0.0.1:8081` | Informational — same caveat as `MESH_CP_BIND`; `mesh api serve --port N` is what actually controls the bind port |
| `MESH_GW_CONTROL_URL` | `http://127.0.0.1:8080` | Where the gateway sends `/api/v1/schedule` and `/api/v1/models` requests |
| `MESH_GW_DIRECT_BACKEND_URL` | unset | **Dev/test escape hatch only.** When set, and the control plane is unreachable, the gateway routes every request straight to this OpenAI-compatible URL instead of failing. Never rely on this in front of real users — it bypasses the scheduler entirely |
| `NODES_LOCAL_FILE` | `deploy/nodes.local.yaml` | Path to the node routing table below. **Note:** unlike the other three, this field has no `MESH_GW_` prefix in code — the env var really is `NODES_LOCAL_FILE`, not `MESH_GW_NODES_LOCAL_FILE` |

`mesh api serve` also takes `--control-url` and sets `MESH_CONTROL_URL` (not
`MESH_GW_CONTROL_URL`) for the CLI's own use — that's a separate variable read by
`apps/cli/meshcompute_cli/config.py`, not by the gateway process itself. If you
start the gateway with `mesh api serve --control-url ...`, the CLI forwards it via
`MESH_CONTROL_URL`; if you run `uvicorn meshcompute_gateway.app:app` directly
instead, set `MESH_GW_CONTROL_URL` yourself.

### Node daemon

No `MESH_*` env vars — every knob is a CLI flag or a keyword argument to
`meshcompute_node.daemon.run()`/`async_main()`. See `docs/QUICKSTART.md` for the
full flag table, and `CONFIGURATION.md` for the conceptual config shape they
implement.

## `deploy/nodes.local.yaml`

Committed template: `deploy/nodes.example.yaml` (safe — no real IPs). Copy it:

```bash
cp deploy/nodes.example.yaml deploy/nodes.local.yaml
```

It serves two independent readers:

- **The control plane** reads `nodes[].node_id` + `nodes[].measured_rtt_ms_from_here`
  to seed the topology graph with a known link RTT from "here" (the control
  plane's vantage point) to that node, instead of the scheduler's pessimistic WAN
  default.
- **The gateway** reads `nodes[].node_id` (or `.name` as a fallback key) +
  `nodes[].backend_url` (+ optional `.backend`, default `lmstudio`) to map a
  scheduler-selected node_id to the actual OpenAI-compatible HTTP endpoint it
  proxies to (`apps/gateway/meshcompute_gateway/settings.py::load_node_endpoints`).

This is a real Phase-1 seam worth understanding: registering a node with the
control plane (`mesh node start`) makes it visible to the *scheduler*, but the
gateway only knows how to actually reach a node's HTTP endpoint via this file.
There's no automatic bridge from "a node registered itself" to "the gateway can
route to it" yet — you (the operator) put the mapping in
`deploy/nodes.local.yaml` yourself. `backend_url` must not have a trailing `/v1`
(the backend adapter appends the OpenAI paths itself).

Shape (placeholders only — never fill real values in a committed file):

```yaml
control_plane:
  bind: "127.0.0.1:8080"
  db_path: "var/control.db"
  manifest_dir: "models/manifests"

gateway:
  bind: "127.0.0.1:8081"
  control_url: "http://127.0.0.1:8080"

rendezvous:
  public_url: "https://rendezvous.example.org"
  stun_servers:
    - "stun.l.google.com:19302"
    - "stun1.l.google.com:19302"

nodes:
  - name: my-gpu-node
    node_id: "<node_id printed by `mesh node start`>"
    role: worker
    backend: llamacpp
    backend_url: "http://HOST:PORT"        # no real IP in a committed file — ever
    model: public/qwen3.8-27b-fable
    pool: public
    idle_only: true
    max_vram_percent: 85
    measured_rtt_ms_from_here: 5           # ms, as measured from the control plane
```

See `deploy/nodes.example.yaml` for the fully-commented version, including the
optional external-endpoint entry described next.

## Optional: registering an existing OpenAI-compatible endpoint

MeshCompute's default and recommended path is the standalone node
(`mesh node start`) — it needs nothing else. But if you already run an
OpenAI-compatible server (LM Studio, vLLM, an existing `llama-server`) and want
the mesh to route to it as-is, use `scripts/register_external_backend.py`
instead of standing up a second daemon:

```bash
# deploy/nodes.local.yaml must already have one `nodes[0]` entry with
# backend_url pointing at your running OpenAI-compatible server, e.g.:
#   nodes:
#     - name: my-existing-endpoint
#       backend: lmstudio          # or whatever your server identifies as
#       backend_url: "http://HOST:PORT"
#       pool: public
#       gpus: 2
#       vram_bytes_each: 24000000000

python scripts/register_external_backend.py
```

It measures the endpoint's real steady-state decode tok/s and TTFT, creates (or
reuses) a signed node identity at `deploy/identities/external-pod.key.json`,
registers + advertises a signed capability against the control plane using that
live measurement, and writes the resulting `node_id` back into
`deploy/nodes.local.yaml` so the gateway can find it. This is exactly how the
WAN benchmark in `docs/BENCHMARKS.md` was produced — the 2×RTX 3090 pod runs its
own OpenAI-compatible server; this script is the bridge, not a second node
daemon.

This path is explicitly optional and secondary: an existing OpenAI endpoint is an
*external backend*, not a requirement. The product default is a standalone node
that never needs LM Studio/Ollama/vLLM at all.

## Multi-node `llama.cpp` RPC layer-split runbook

For a model too large for one GPU but small enough to split pipeline-style across
two machines *on a low-latency link* — this is genuinely a per-request speedup,
not a WAN split (see `docs/PHYSICS.md` §4: the scheduler only admits this when
the link is LAN-class, `is_lan_class()`: ≤5 ms RTT, ≥500 Mbps, direct — same rack,
same LAN, or same cloud region. Do **not** attempt this over the open internet;
that's the exact case `docs/PHYSICS.md` measures as ~39x slower, not faster).

This is a manual operator runbook today — the scheduler's `ExecutionPlan` already
reserves the `pipeline` strategy and `LlamaCppBackend` already accepts
`rpc_servers`/`--rpc`, but wiring a scheduler-selected peer chain straight into a
live `--rpc` launch is a Phase-1.5 TODO (`node/runtime/meshcompute_runtime/backends/llamacpp.py`,
`rpc_split_args`). Until then, run it by hand:

**On peer B** (the machine donating extra layers — no model file needed on this
side, `rpc-server` just executes ops it's handed):

```bash
# rpc-server ships in the same llama.cpp release archive `mesh node start`
# already downloaded on peer B once (via runtime_installer.ensure_llama_server).
# Find it next to that cached llama-server:
find ~/.mesh/runtime -name rpc-server

~/.mesh/runtime/<tag>-<os>-<accel>-<arch>/**/rpc-server --host 0.0.0.0 --port 50052
```

**On peer A** (the machine actually serving the request), launch `llama-server`
pointed at peer B over `--rpc`:

```bash
llama-server -m <path-to-model.gguf> --host 127.0.0.1 --port 8090 \
  -ngl 999 --rpc PEER_B_HOST:50052
```

Or, driving it through the Python adapter instead of the raw binary:

```python
from meshcompute_runtime.backends.llamacpp import LlamaCppBackend

backend = LlamaCppBackend(
    server_bin, model_path, n_gpu_layers=999,
    rpc_servers=["PEER_B_HOST:50052"],   # LlamaCppBackend.rpc_split_args() shows the flag shape
)
```

Replace `PEER_B_HOST` with the real address in your local, gitignored
configuration only — never in a committed file or example.

## Contribution / idle configuration

The CLI flags on `mesh node start` (`--idle-only`, `--idle-minutes`,
`--pause-on-activity`, `--max-vram`, `--max-cpu`, `--max-ram`,
`--require-ac-power`) are the runtime knobs for exactly the policy
`CONFIGURATION.md` describes conceptually under `availability`/`compute`/
`thermal`. See `docs/QUICKSTART.md` for the full flag-to-default table and
`CONFIGURATION.md` for the target end-state config shape (a `mesh-node.yaml`
config file is on the roadmap; today the CLI flags are the real, working
interface — see `node/daemon/meshcompute_node/contribution.py` for exactly how
each percentage becomes a thread count / VRAM budget / RAM cap enforced before
the engine launches).

## Security posture

From `SECURITY.md` (full detail there) — the parts that matter operationally:

- **Public-swarm privacy is not confidential compute.** A worker running
  inference can observe activations and prompt-derived state. Don't market or
  treat public-pool inference as private; recommend a private pool for sensitive
  data. The gateway never logs prompt bodies (`messages`/`content`) anywhere —
  verify this holds if you fork/extend it.
- **Tools and secrets never reach a public inference worker.** The gateway
  builds the backend's `ChatRequest` from an explicit whitelist of inference-only
  fields and never reads the client's `Authorization` header (or any other
  header) to construct it — see `_build_backend_request` in
  `apps/gateway/meshcompute_gateway/app.py`. Keep any tool credentials/API
  keys/cookies entirely inside your trusted client or gateway-side harness; never
  pass them through to a worker.
- **The control plane is not the inference data path.** It never sees prompt
  bodies and holds no tool credentials — its job is auth, registration,
  rendezvous, catalog, and scheduling metadata only.
- **Node identity keys are Ed25519, self-generated, gitignored.**
  `deploy/identities/*.key.json` and anything matching `*.key.json` /
  `*_id_ed25519*` / `credentials*.json` in `.gitignore` must never be committed;
  rotate any key that leaks.
- **Model manifests are signed and pinned** (revision, tokenizer, tool parser);
  the public pool never runs arbitrary upstream model code — GGUF weights only.
- Deploying behind a reverse proxy/TLS terminator, adding per-user rate limits,
  and sandboxing the worker process (container, read-only model files, restricted
  egress) are still your responsibility in Phase 1 — see `SECURITY.md`'s
  "Worker sandbox" and "Denial of service" sections; none of this is automated
  by the code yet.

## Running under systemd / Docker Compose

Unit files: `infra/systemd/mesh-control-plane.service`,
`infra/systemd/mesh-gateway.service`, `infra/systemd/mesh-node.service` — edit the
`User=`, `WorkingDirectory=`, and venv path placeholders before installing.

Compose (control plane + gateway only — nodes normally run on contributor
machines, not in this compose file): `infra/compose/docker-compose.yml`.

```bash
docker compose -f infra/compose/docker-compose.yml up -d
```

See the comments in both for what to edit for your environment.
