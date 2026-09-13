"""The standalone MeshCompute node daemon (INSTRUCTIONS §2.1, §7, §9, §10).

One program a user launches. It installs its own inference engine, lets the
user pick a model channel from the catalog, obtains that model from peers
(BLAKE3-verified chunks over QUIC) or Hugging Face, runs it, exposes a local
OpenAI-compatible endpoint, seeds the model to other peers, contributes its
measured compute share to the network, and — when the catalog advances a
channel to a new version — downloads the new file in the background, swaps
the engine over, and deletes the old file so the machine never fills with junk.

`mesh node start` (apps/cli/meshcompute_cli/main.py) calls run()/async_main().
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Callable

import httpx
import yaml

from meshcompute_protocol import (
    CapabilityRecord,
    ContributionPolicy,
    HeartbeatRequest,
    ModelManifest,
    NodeIdentity,
    RegisterRequest,
    SignedCapability,
    SignedManifest,
)
from meshcompute_runtime.backends.base import BackendAdapter, ChatRequest
from meshcompute_runtime.backends.lmstudio import OpenAICompatBackend
from meshcompute_runtime.backends.llamacpp import LlamaCppBackend, gpu_launch_args
from meshcompute_runtime import runtime_installer as ri
from meshcompute_runtime.swarm import leech_file, serve_chunk_stream

from .capability import measure_capability, ram_free_bytes
from .contribution import (
    ContributionController,
    bound_ctx_size,
    cpu_thread_budget,
    per_gpu_free_vram_bytes,
    ram_budget_bytes,
    total_ram_bytes,
    vram_budget_bytes,
)
from .rendezvous_client import RendezvousClient
from .stun import discover_reflexive
from .sysinfo import disk_free_bytes
from .transport_base import PeerAddress
from .transport_quic import QuicTransport

DEFAULT_CONTROL_URL = "http://127.0.0.1:8080"          # matches apps/cli/meshcompute_cli/config.py
DEFAULT_IDENTITY_PATH = ri.MESH_HOME / "identity.json"
DEFAULT_LOCAL_PORT = 8099                              # stable local /v1 endpoint port
HEARTBEAT_INTERVAL_S = 30.0
UPDATE_POLL_S = 300.0                                  # how often to check the channel
_REPO_MANIFEST_DIR = Path(__file__).resolve().parents[3] / "models" / "manifests"


def _log(msg: str) -> None:
    print(f"[mesh] {msg}", flush=True)


# --------------------------------------------------------------------------- catalog / model choice
async def fetch_catalog(control_url: str) -> list[ModelManifest]:
    """Every manifest the control plane serves, else the repo's own
    models/manifests/*.yaml (same files the control plane loads), else []."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{control_url}/api/v1/models")
            resp.raise_for_status()
            out = []
            for view in resp.json():
                r = await client.get(f"{control_url}/api/v1/models/{view['id']}")
                if r.status_code == 200:
                    out.append(SignedManifest.model_validate(r.json()).manifest)
            if out:
                return out
    except (httpx.HTTPError, ValueError):
        pass
    out = []
    if _REPO_MANIFEST_DIR.is_dir():
        for path in sorted(_REPO_MANIFEST_DIR.glob("*.yaml")):
            try:
                out.append(ModelManifest.model_validate(yaml.safe_load(path.read_text())))
            except Exception:  # a broken local manifest must not kill the node
                _log(f"skipping unreadable manifest {path.name}")
    return out


async def fetch_manifest(control_url: str, alias: str) -> ModelManifest | None:
    """One channel's CURRENT manifest from the control plane (None if unreachable
    or unknown). This is the poll the update loop runs."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{control_url}/api/v1/models/{alias}")
        if resp.status_code == 200:
            return SignedManifest.model_validate(resp.json()).manifest
    except (httpx.HTTPError, ValueError):
        pass
    return None


def fits(manifest: ModelManifest, budget_bytes: int) -> bool:
    return ri.model_target(manifest).total_bytes <= budget_bytes


def choose_model_interactively(catalog: list[ModelManifest], budget_bytes: int,
                               ask: Callable[[str], str] = input) -> ModelManifest | None:
    """Numbered picker on a TTY. Enter = smoke default (None)."""
    print("\nAvailable model channels:")
    for i, m in enumerate(catalog, 1):
        size = ri.model_target(m).total_bytes / 1e9
        fit = "fits" if fits(m, budget_bytes) else f"needs {size:.0f} GB, you have {budget_bytes/1e9:.0f}"
        print(f"  {i}. {m.id}  v{m.version}  {size:.1f} GB  [{fit}]  {m.display_name}")
    print("  0. built-in smoke model (SmolLM2-360M, CPU-friendly)")
    while True:
        raw = ask("Select model to contribute to and use [0]: ").strip()
        if raw in ("", "0"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(catalog):
            return catalog[int(raw) - 1]
        print("  enter a number from the list")


async def resolve_model(model: str | None, control_url: str, budget_bytes: int, *,
                        interactive: bool | None = None):
    """Pick what to run. Order: explicit `model` alias -> saved choice in
    state.json -> interactive picker (TTY) -> smoke default. "smoke" forces
    the default. Returns a ModelManifest or ri.SMOKE_MODEL."""
    state = ri.load_state()
    if model == "smoke":
        return ri.SMOKE_MODEL
    alias = model or state.get("selected_alias")
    if alias:
        catalog = {m.id: m for m in await fetch_catalog(control_url)}
        if alias in catalog:
            if model and state.get("selected_alias") != model:
                state["selected_alias"] = model
                ri.save_state(state)
            return catalog[alias]
        _log(f"model '{alias}' not in the catalog at {control_url} (or locally); "
             f"{'falling back to the smoke model' if model else 'pick again'}")
        if model:
            return ri.SMOKE_MODEL
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        catalog = await fetch_catalog(control_url)
        if catalog:
            choice = choose_model_interactively(catalog, budget_bytes)
            state["selected_alias"] = choice.id if choice else None
            ri.save_state(state)
            return choice or ri.SMOKE_MODEL
    _log("no model selected; running the built-in smoke model "
         "(pass --model <alias>, or run interactively to pick from the catalog)")
    return ri.SMOKE_MODEL


# --------------------------------------------------------------------------- P2P model transfer
def _lan_addrs(port: int) -> list[str]:
    addrs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet sent; picks the default-route interface
        addrs.append(f"{s.getsockname()[0]}:{port}")
        s.close()
    except OSError:
        pass
    addrs.append(f"127.0.0.1:{port}")
    return addrs


class PeerSwarm:
    """Owns the node's QUIC transport for model distribution: seeds every file
    in state.json to inbound peers, and fetches files from peers the rendezvous
    tracker says are seeding them. This is the BitTorrent part."""

    def __init__(self, identity: NodeIdentity, control_url: str, transport: QuicTransport) -> None:
        self.identity = identity
        self.control_url = control_url
        self.transport = transport
        self.rendezvous = RendezvousClient(control_url)
        self.reflexive_addr: str | None = None
        self.bytes_served = 0
        self.peer_rtts: dict[str, float] = {}     # node_id -> measured QUIC RTT (ms)
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    # --- seeding ---------------------------------------------------------------
    async def seed_forever(self) -> None:
        async for conn in self.transport.accept():
            self._spawn(self._serve_conn(conn))

    async def _serve_conn(self, conn) -> None:
        closed = asyncio.ensure_future(conn.wait_closed())
        try:
            while not closed.done():
                nxt = asyncio.ensure_future(conn.accept_stream())
                done, _ = await asyncio.wait({nxt, closed}, return_when=asyncio.FIRST_COMPLETED)
                if nxt not in done:
                    nxt.cancel()
                    break
                stream = nxt.result()
                self._spawn(self._serve_stream(stream))
        finally:
            if closed.done() and not closed.cancelled():
                closed.exception()   # aioquic resolves it with ConnectionError; consume it
            else:
                closed.cancel()

    async def _serve_stream(self, stream) -> None:
        try:
            self.bytes_served += await serve_chunk_stream(stream, ri.seed_index(ri.load_state()))
        except Exception as e:  # a bad peer must never take the seeder down
            _log(f"seed stream ended: {e}")

    # --- rendezvous ---------------------------------------------------------------
    async def announce(self, wanted: list[str] | None = None):
        seeding = sorted({h for (h, _f) in ri.seed_index(ri.load_state())})
        if self.reflexive_addr is None:
            self.reflexive_addr = await discover_reflexive(0)  # public IP, best-effort
        return await self.rendezvous.announce(
            self.identity, local_addrs=_lan_addrs(self.transport.local_quic_port),
            reflexive_addr=self.reflexive_addr, quic_port=self.transport.local_quic_port,
            seeding_hashes=seeding, wanted_hashes=wanted or [])

    # --- leeching ---------------------------------------------------------------
    # How long to wait for another node that is already fetching the same file
    # before going to origin ourselves: enough for it to finish at a modest
    # origin rate, capped. Origin is never more than this far away.
    WAIT_RATE_BPS = 2 << 20
    WAIT_MAX_S = 1800.0
    WAIT_POLL_S = 15.0

    async def _try_seeders(self, seeders, dest: Path, manifest_hash: str, shard: dict) -> bool:
        for peer in seeders:
            addrs = list(peer.local_addrs) + ([peer.reflexive_addr] if peer.reflexive_addr else [])
            for addr in addrs:
                host, _, port_s = addr.rpartition(":")
                try:
                    port = int(port_s) if port_s else peer.quic_port
                    conn = await self.transport.dial(PeerAddress(node_id=peer.node_id, host=host,
                                                                 port=port), timeout=8.0)
                except Exception:
                    continue
                try:
                    with contextlib.suppress(Exception):
                        self.peer_rtts[peer.node_id] = round(await conn.rtt_ms(), 2)
                    stream = await conn.open_stream()
                    _log(f"fetching {shard['rfilename']} from peer {peer.node_id[:12]} @ {addr} "
                         f"(rtt {self.peer_rtts.get(peer.node_id, '?')} ms)")
                    last = [0.0]

                    def progress(done: int, total: int) -> None:
                        if time.monotonic() - last[0] > 5 or done == total:
                            last[0] = time.monotonic()
                            _log(f"  {shard['rfilename']}: {done}/{total} chunks")

                    await leech_file(stream, dest, manifest_hash, shard, progress_cb=progress)
                    await stream.close()
                    return True
                except Exception as e:
                    _log(f"peer {peer.node_id[:12]} failed ({e}); trying next")
                    break   # same peer's other addresses will fail the same way
                finally:
                    with contextlib.suppress(Exception):
                        await conn.close()
        return False

    async def fetch(self, dest: Path, manifest_hash: str, shard: dict) -> bool:
        """ri.SwarmFetch: get the whole verified file from a seeder. If nobody
        seeds it yet, the tracker names ONE origin leader for it (the node that
        started wanting it first); everyone else waits for that node to finish
        and seed. One node per update wave pays the origin download, the rest
        get it peer-to-peer. Returns False = this node goes to origin."""
        deadline = time.monotonic() + min(self.WAIT_MAX_S,
                                          max(60.0, shard["size_bytes"] / self.WAIT_RATE_BPS))
        waited_for: str | None = None
        while True:
            try:
                resp = await self.announce(wanted=[manifest_hash])
            except httpx.HTTPError as e:
                _log(f"rendezvous unreachable ({e}); no peers for {shard['rfilename']}")
                return False
            seeders = [p for p in resp.peers
                       if p.node_id != self.identity.node_id and p.role != "leecher"]
            if seeders and await self._try_seeders(seeders, dest, manifest_hash, shard):
                return True
            leader = resp.origin_leader
            if leader in (None, self.identity.node_id) or time.monotonic() > deadline:
                if waited_for:
                    _log(f"peer {waited_for[:12]} did not finish {shard['rfilename']} in time; "
                         f"using origin")
                elif not seeders:
                    _log(f"no peer is seeding {shard['rfilename']} yet; fetching it from origin "
                         f"for the network")
                return False
            if waited_for != leader:
                waited_for = leader
                _log(f"peer {leader[:12]} is already fetching {shard['rfilename']} from origin; "
                     f"waiting to get it from that peer instead")
            await asyncio.sleep(self.WAIT_POLL_S)

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            with contextlib.suppress(BaseException):
                await t


# --------------------------------------------------------------------------- control-plane calls
def signed_body(identity: NodeIdentity, model) -> dict:
    """model_dump with `signature_b64` filled in over the rest of the body —
    the scheme the control plane verifies for register/heartbeat/connect."""
    body = model.model_dump(mode="json", exclude={"signature_b64"})
    body["signature_b64"] = identity.sign_json(body)
    return body


async def _try_register(control_url: str, identity: NodeIdentity, pool_id: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{control_url}/api/v1/nodes/register",
                json=signed_body(identity, RegisterRequest(
                    node_id=identity.node_id, public_b64=identity.public_b64, pool_ids=[pool_id])))
            resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


async def _post_capability(control_url: str, identity: NodeIdentity, backend: BackendAdapter,
                           policy: ContributionPolicy, gpu_devices, storage_share_bytes: int,
                           ) -> CapabilityRecord:
    record = await measure_capability(
        identity, backend, {"contribution_policy": policy.model_dump(), "gpu_devices": gpu_devices})
    # Cap the ADVERTISED free_vram to the configured share and attach the full
    # policy, so the scheduler never over-places beyond what this contributor
    # actually chose to donate. Storage share = what the swarm may keep here.
    vbudget = vram_budget_bytes(record.total_vram_bytes(), policy.max_vram_percent)
    capped_gpus = [g.model_copy(update={"free_vram_bytes": min(g.free_vram_bytes, vbudget)})
                   for g in record.gpus]
    record = record.model_copy(update={"gpus": capped_gpus, "contribution_policy": policy,
                                       "storage_share_bytes": storage_share_bytes})
    signed = SignedCapability(record=record, public_b64=identity.public_b64,
                              signature_b64=identity.sign_json(record.model_dump(mode="json")))
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{control_url}/api/v1/nodes/{identity.node_id}/capabilities",
                                 json=signed.model_dump(mode="json"))
        resp.raise_for_status()
    return record


class NodeRuntime:
    """Mutable per-daemon state the background loops share: the live backend
    (swapped on a channel update) and whether we are advertising availability."""

    def __init__(self) -> None:
        self.backend: BackendAdapter | None = None
        self.engine: LlamaCppBackend | None = None
        self.updating = False          # True during a changeover -> heartbeats pause
        self.channel: str | None = None
        self.manifest_hash: str = ""
        self.version: int = 0


async def _heartbeat_loop(control_url: str, identity: NodeIdentity, interval_s: float,
                          stop_event: asyncio.Event, controller: ContributionController,
                          rt: NodeRuntime, swarm: "PeerSwarm | None" = None) -> None:
    """While PAUSED (idle policy) or updating, skip the POST — the control
    plane's heartbeat timeout ages a silent node to offline, i.e. "unavailable"
    with no protocol change."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        announced: bool | None = None
        while not stop_event.is_set():
            available = controller.is_available() and not rt.updating
            if available != announced:
                why = "updating model" if rt.updating else controller.state.value
                _log(f"contribution {'ACTIVE' if available else 'PAUSED (' + why + ')'} — "
                     f"heartbeats {'resumed' if available else 'suspended'}")
                announced = available
            if available:
                cached = sorted({ch["manifest_hash"] for ch in ri.load_state()["channels"].values()})
                hb = HeartbeatRequest(node_id=identity.node_id, ram_free_bytes=ram_free_bytes(),
                                      cached_manifest_hashes=cached,
                                      peer_rtt_ms=dict(swarm.peer_rtts) if swarm else {})
                with contextlib.suppress(httpx.HTTPError):
                    await client.post(f"{control_url}/api/v1/nodes/{identity.node_id}/heartbeat",
                                      json=signed_body(identity, hb))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)


# --------------------------------------------------------------------------- engine launch + changeover
class EngineLauncher:
    """Everything needed to (re)launch llama-server for a model file on the
    node's fixed local port, with the contributor's resource budget applied."""

    def __init__(self, engine: ri.Engine, port: int, policy: ContributionPolicy, ctx_size: int,
                 gpu_devices, split_mode: str, tensor_split: str | None) -> None:
        self.engine, self.port, self.policy, self.ctx_size = engine, port, policy, ctx_size
        self.gpu_devices, self.split_mode, self.tensor_split = gpu_devices, split_mode, tensor_split

    def build(self, model_path: str, model_id: str) -> LlamaCppBackend:
        thread_budget = cpu_thread_budget(self.policy.max_cpu_percent)
        ram_budget = ram_budget_bytes(total_ram_bytes(), self.policy.max_ram_gb)
        ctx_size = bound_ctx_size(self.ctx_size, ram_budget)
        free_vram = per_gpu_free_vram_bytes(self.gpu_devices)
        selected = [{"index": i, "free_vram_bytes": free_vram[i]} for i in sorted(free_vram)]
        env, args = gpu_launch_args(selected, split_mode=self.split_mode,
                                    tensor_split=self.tensor_split, accel=self.engine.accel)
        _log(f"resource budget: threads={thread_budget} ctx_size={ctx_size} "
             f"gpus={[g['index'] for g in selected] or 'none'} accel={self.engine.accel} "
             f"launch_env={env} launch_args={args}")
        return LlamaCppBackend(self.engine.path, model_path, model_id=model_id, host="127.0.0.1",
                               port=self.port, ctx_size=ctx_size,
                               extra_args=["--threads", str(thread_budget), "--alias", model_id, *args],
                               extra_env=env)


async def changeover(rt: NodeRuntime, launcher: EngineLauncher, new_manifest: ModelManifest, *,
                     swarm_fetch: ri.SwarmFetch | None, max_storage_bytes: int | None,
                     cache_dir: str | Path | None = None) -> bool:
    """Move a running node from the channel's old version to `new_manifest`:
      1. fetch + verify the new files while the old engine keeps serving,
      2. pause network availability, stop old, start new on the SAME port,
      3. on success delete the old version's files; on failure restart the old
         engine (its files are untouched) and report False.
    Downtime on the local endpoint = one engine load."""
    alias = new_manifest.id
    cache_dir = Path(cache_dir or ri.DEFAULT_MODEL_CACHE)
    _log(f"channel {alias}: v{rt.version} -> v{new_manifest.version} "
         f"({new_manifest.manifest_hash()[:12]}) — downloading in the background")
    try:
        new_path, target = await ri.ensure_model(new_manifest, cache_dir, swarm_fetch=swarm_fetch,
                                                 max_storage_bytes=max_storage_bytes, log=_log)
    except Exception as e:
        _log(f"channel {alias}: update download failed ({e}); staying on v{rt.version}")
        return False

    rt.updating = True
    old_engine = rt.engine
    try:
        new_engine = launcher.build(new_path, alias)
        if old_engine is not None:
            old_engine.stop()
        try:
            await new_engine.start()
        except Exception as e:
            _log(f"channel {alias}: new engine failed to start ({e}); restarting v{rt.version}")
            if old_engine is not None:
                old = launcher.build(old_engine.model_path, alias)
                await old.start()
                rt.engine = rt.backend = old
            return False
        rt.engine = rt.backend = new_engine
        state = ri.load_state(cache_dir)
        stale = ri.record_channel(state, alias, target.manifest_hash, target.version,
                                  ri.channel_files(cache_dir, target))
        ri.save_state(state, cache_dir)
        removed = ri.delete_files(stale, cache_dir)
        rt.manifest_hash, rt.version = target.manifest_hash, target.version
        _log(f"channel {alias}: now serving v{target.version}; "
             f"deleted {len(removed)} old file(s): {[Path(p).name for p in removed]}")
        return True
    finally:
        rt.updating = False


async def _update_loop(rt: NodeRuntime, launcher: EngineLauncher, control_url: str, poll_s: float,
                       stop_event: asyncio.Event, *, swarm_fetch, max_storage_bytes,
                       on_swapped: Callable[[], object] | None = None) -> None:
    """Poll the channel's manifest; when the catalog advances it, swap over.
    A lower version than we run is ignored (rollback is an explicit operator
    action, not something a stale control plane can trigger by accident)."""
    while not stop_event.is_set():
        with contextlib.suppress(asyncio.TimeoutError):   # jitter spreads an update wave out
            await asyncio.wait_for(stop_event.wait(), timeout=poll_s * random.uniform(0.7, 1.3))
        if stop_event.is_set() or rt.channel is None or rt.updating:
            continue
        m = await fetch_manifest(control_url, rt.channel)
        if m is None or m.manifest_hash() == rt.manifest_hash:
            continue
        if m.version < rt.version:
            _log(f"channel {rt.channel}: catalog offers v{m.version} < running v{rt.version}; ignoring")
            continue
        try:
            ok = await changeover(rt, launcher, m, swarm_fetch=swarm_fetch,
                                  max_storage_bytes=max_storage_bytes)
        except Exception as e:   # keep polling; the node may recover next round
            _log(f"channel {rt.channel}: changeover crashed ({e}); will retry next poll")
            continue
        if ok and on_swapped is not None:
            with contextlib.suppress(Exception):
                await on_swapped()


# --------------------------------------------------------------------------- main flow
async def async_main(*, backend: str = "llamacpp", backend_url: str | None = None,
                     model: str | None = None, pool_id: str = "public", pool: str | None = None,
                     control_url: str | None = None, port: int | None = None,
                     quic_port: int = 0, accel: str | None = None, quant: str | None = None,
                     idle_only: bool = True, idle_minutes_before_start: int = 10,
                     pause_on_user_activity: bool = True, require_ac_power: bool = False,
                     max_vram_percent: int = 85, max_vram: int | None = None,
                     max_cpu_percent: int = 50, max_cpu: int | None = None,
                     max_ram_gb: int | None = None, max_storage_gb: int | None = 100,
                     max_gpu_percent: int = 90, gpu_temperature_limit_c: int = 82,
                     allow_public_pool: bool = True,
                     gpu_devices: str | list[int] | None = "auto", split_mode: str = "layer",
                     tensor_split: str | None = None, update_poll_s: float = UPDATE_POLL_S,
                     smoke: bool = False, identity_path: str | None = None, ctx_size: int = 4096,
                     heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
                     interactive: bool | None = None, **_ignored) -> None:
    pool_id = pool or pool_id
    max_vram_percent = max_vram if max_vram is not None else max_vram_percent
    max_cpu_percent = max_cpu if max_cpu is not None else max_cpu_percent
    control_url = (control_url or DEFAULT_CONTROL_URL).rstrip("/")
    max_storage_bytes = None if max_storage_gb is None else max_storage_gb * (1 << 30)

    policy = ContributionPolicy(
        idle_only=idle_only, max_gpu_percent=max_gpu_percent, max_vram_percent=max_vram_percent,
        max_cpu_percent=max_cpu_percent, allow_public_pool=allow_public_pool,
        idle_minutes_before_start=idle_minutes_before_start,
        pause_on_user_activity=pause_on_user_activity, require_ac_power=require_ac_power,
        max_ram_gb=max_ram_gb, gpu_temperature_limit_c=gpu_temperature_limit_c)
    controller = ContributionController(policy)
    await controller.start()
    _log(f"contribution policy: idle_only={policy.idle_only} max_cpu={policy.max_cpu_percent}% "
         f"max_vram={policy.max_vram_percent}% "
         f"max_ram={policy.max_ram_gb if policy.max_ram_gb is not None else 'unset'}GB "
         f"max_storage={max_storage_gb if max_storage_gb is not None else 'unset'}GB "
         f"require_ac_power={policy.require_ac_power} -> initial state {controller.state.value}")
    if controller.last_snapshot is not None:
        s = controller.last_snapshot
        _log(f"idle signals: user_idle_s={s.user_idle_s} cpu%={s.cpu_percent:.1f} "
             f"gpu%={s.gpu_percent} on_ac={s.on_ac}")

    identity = NodeIdentity.load_or_create(identity_path or DEFAULT_IDENTITY_PATH)
    _log(f"node identity: {identity.node_id}")

    rt = NodeRuntime()
    launcher: EngineLauncher | None = None
    swarm: PeerSwarm | None = None
    seed_task: asyncio.Task | None = None
    transport = QuicTransport(identity, bind_port=quic_port)
    await transport.start()

    if backend == "llamacpp":
        engine = await ri.ensure_llama_server(accel=accel, log=_log)
        _log(f"engine ready: {engine.accel} build {engine.tag} at {engine.path}")

        # Budget for "does this model fit here": donated VRAM if any, else RAM.
        vram = vram_budget_bytes(sum(per_gpu_free_vram_bytes(gpu_devices).values()), max_vram_percent)
        budget = vram or ram_budget_bytes(total_ram_bytes(), max_ram_gb)
        chosen = await resolve_model(model, control_url, budget, interactive=interactive)

        swarm = PeerSwarm(identity, control_url, transport)
        seed_task = asyncio.create_task(swarm.seed_forever())
        model_path, target = await ri.ensure_model(chosen, swarm_fetch=swarm.fetch,
                                                   max_storage_bytes=max_storage_bytes,
                                                   quant_id=quant, log=_log)
        _log(f"model ready: {target.alias} v{target.version} -> {model_path}")

        local_port = port or DEFAULT_LOCAL_PORT
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", local_port)) == 0:
                probe2 = socket.socket()
                probe2.bind(("127.0.0.1", 0))
                local_port = probe2.getsockname()[1]
                probe2.close()
                _log(f"port {port or DEFAULT_LOCAL_PORT} is busy; using {local_port} instead")
        launcher = EngineLauncher(engine, local_port, policy, ctx_size, gpu_devices, split_mode,
                                  tensor_split)
        rt.engine = rt.backend = launcher.build(model_path, target.alias)
        await rt.engine.start()
        state = ri.load_state()
        stale = ri.record_channel(state, target.alias, target.manifest_hash, target.version,
                                  ri.channel_files(ri.DEFAULT_MODEL_CACHE, target))
        # A node follows ONE channel: whatever it served before (an older version,
        # or a different channel the user switched away from) is junk now.
        for other in [a for a in state["channels"] if a != target.alias]:
            stale += [f["path"] for f in state["channels"].pop(other)["files"]]
        ri.save_state(state)
        if stale:
            removed = ri.delete_files(stale)
            _log(f"deleted {len(removed)} superseded model file(s): {[Path(p).name for p in removed]}")
        orphans = ri.sweep_orphans(state)   # leftovers of crashed runs / older layouts
        if orphans:
            _log(f"removed {len(orphans)} orphaned model file(s): {[Path(p).name for p in orphans]}")
        rt.channel = target.alias if not isinstance(chosen, ri.ModelSpec) else None
        rt.manifest_hash, rt.version = target.manifest_hash, target.version
        endpoint = f"{rt.engine.base_url}/v1"
    else:
        if not backend_url:
            raise SystemExit(f"--backend {backend} requires --backend-url")
        rt.backend = OpenAICompatBackend(backend_url, backend_name=backend)
        endpoint = f"{backend_url.rstrip('/')}/v1"
        seed_task = None

    _log(f"Local endpoint ready: {endpoint}  (point your OpenAI client here)")

    if smoke:
        models = await rt.backend.list_models()
        req = ChatRequest(model=models[0] if models else "local",
                          messages=[{"role": "user", "content": "Say hello in exactly three words."}],
                          max_tokens=32, temperature=0.2, stream=True)
        text = "".join([c.text async for c in rt.backend.chat_stream(req) if c.text])
        _log(f"smoke self-test completion: {text!r}")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, AttributeError):
            loop.add_signal_handler(sig, stop_event.set)

    tasks: list[asyncio.Task] = []
    storage_share = min(max_storage_bytes or disk_free_bytes(ri.DEFAULT_MODEL_CACHE.parent),
                        disk_free_bytes(ri.DEFAULT_MODEL_CACHE.parent))

    async def post_capability() -> None:
        record = await _post_capability(control_url, identity, rt.backend, policy, gpu_devices,
                                        storage_share)
        _log(f"posted signed capability: decode {record.benchmark.decode_tokens_per_sec} tok/s, "
             f"backends={record.backends}, storage_share={storage_share/1e9:.0f} GB")
        if swarm is not None:
            with contextlib.suppress(httpx.HTTPError):
                await swarm.announce()   # now seeding the new version, wanting nothing

    try:
        if await _try_register(control_url, identity, pool_id):
            await post_capability()
            if swarm is not None:
                with contextlib.suppress(httpx.HTTPError):
                    peers = await swarm.announce()
                    _log(f"announced to rendezvous (quic :{transport.local_quic_port}, "
                         f"reflexive {swarm.reflexive_addr}); {len(peers.peers)} peer(s) share content")
            tasks.append(asyncio.create_task(_heartbeat_loop(
                control_url, identity, heartbeat_interval_s, stop_event, controller, rt, swarm)))
            if launcher is not None and rt.channel is not None:
                tasks.append(asyncio.create_task(_update_loop(
                    rt, launcher, control_url, update_poll_s, stop_event,
                    swarm_fetch=swarm.fetch if swarm else None, max_storage_bytes=max_storage_bytes,
                    on_swapped=post_capability)))
                _log(f"watching channel {rt.channel} for updates every {update_poll_s:.0f}s")
            _log(f"registered with control plane at {control_url}; contributing to pool '{pool_id}'")
        else:
            _log(f"control plane at {control_url} unreachable; running in SOLO mode "
                 f"(local endpoint only — not contributing to the network)")
        if smoke:
            return
        await stop_event.wait()
    finally:
        _log("shutting down...")
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(BaseException):
                await t
        if seed_task is not None:
            seed_task.cancel()
            with contextlib.suppress(BaseException):
                await seed_task
        if swarm is not None:
            await swarm.stop()
        await transport.stop()
        await controller.stop()
        if rt.engine is not None:
            rt.engine.stop()
        _log("stopped.")


def run(**opts) -> None:
    asyncio.run(async_main(**opts))


if __name__ == "__main__":
    run(smoke=True, model="smoke")
