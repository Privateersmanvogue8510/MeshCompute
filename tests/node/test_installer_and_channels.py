"""Engine asset selection per target OS, model cache state (channels /
versions / cleanup), storage budget, and the daemon's changeover logic — all
pure or against fakes, no network, no engine (INSTRUCTIONS §23)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from meshcompute_protocol import ContributionPolicy, ModelManifest
from meshcompute_runtime import runtime_installer as ri

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- engine assets
@pytest.mark.parametrize("system,accel,machine,expected", [
    ("Linux", "cpu", "x86_64", "bin-ubuntu-x64.tar.gz"),
    ("Linux", "cpu", "aarch64", "bin-ubuntu-arm64.tar.gz"),
    ("Linux", "vulkan", "x86_64", "bin-ubuntu-vulkan-x64.tar.gz"),
    ("Linux", "cuda", "x86_64", None),                       # no upstream Linux CUDA prebuilt
    ("Darwin", "metal", "arm64", "bin-macos-arm64.tar.gz"),
    ("Darwin", "metal", "x86_64", "bin-macos-x64.tar.gz"),
    ("Windows", "cuda", "AMD64", "bin-win-cuda-12.4-x64.zip"),
    ("Windows", "cpu", "AMD64", "bin-win-cpu-x64.zip"),
    ("Windows", "cpu", "ARM64", "bin-win-cpu-arm64.zip"),
    ("Windows", "vulkan", "AMD64", "bin-win-vulkan-x64.zip"),
])
def test_asset_suffix_matches_upstream_release_names(system, accel, machine, expected):
    assert ri.asset_suffix(system, accel, machine) == expected


@pytest.mark.parametrize("system,accel,cascade", [
    ("Linux", "cuda", ["vulkan", "cpu"]),
    ("Linux", "cuda-build", ["cuda-build", "vulkan", "cpu"]),
    ("Linux", "cpu", ["cpu"]),
    ("Windows", "cuda", ["cuda", "vulkan", "cpu"]),
    ("Windows", "cpu", ["cpu"]),
    ("Darwin", "cuda", ["metal"]),
    ("Darwin", "metal", ["metal"]),
])
def test_accelerator_cascade_never_dead_ends(system, accel, cascade):
    assert ri._cascade(system, accel) == cascade


def test_detect_accel_honours_env_override(monkeypatch):
    monkeypatch.setenv("MESH_ACCEL", "cpu")
    assert ri.detect_accel() == "cpu"
    monkeypatch.setenv("MESH_ACCEL", "cuda-build")
    assert ri.detect_accel() == "cuda-build"


# --------------------------------------------------------------------------- model targets
def _manifest(version=1, rfilename="SmolLM2-360M-Instruct-Q4_K_M.gguf", size=270590880):
    raw = yaml.safe_load((REPO / "models/manifests/public-smollm2-360m.yaml").read_text())
    raw["version"] = version
    raw["quantizations"][0]["files"][0]["rfilename"] = rfilename
    raw["quantizations"][0]["files"][0]["size_bytes"] = size
    return ModelManifest.model_validate(raw)


def test_model_target_uses_first_quant_and_versioned_dest(tmp_path):
    m = _manifest()
    t = ri.model_target(m)
    assert t.alias == "public/smollm2-360m" and t.version == 1 and t.total_bytes == 270590880
    dest = ri.model_dest(tmp_path, t, t.files[0]["rfilename"])
    assert dest.parent.name.startswith("v1-")
    assert dest.parent.parent.name == "bartowski--SmolLM2-360M-Instruct-GGUF"
    # a new version lands in a different directory: never overwrites the running file
    t2 = ri.model_target(_manifest(version=2))
    assert ri.model_dest(tmp_path, t2, t2.files[0]["rfilename"]).parent != dest.parent
    with pytest.raises(ValueError, match="no quantization"):
        ri.pick_quant(m, "Q9_NOPE")


def test_smoke_model_is_pinned():
    assert len(ri.SMOKE_MODEL.artifact_root_hash) == 64 and ri.SMOKE_MODEL.size_bytes > 0


def test_channel_state_records_versions_and_returns_stale_files(tmp_path):
    st = ri.load_state(tmp_path)
    t1 = ri.model_target(_manifest(version=1))
    f1 = ri.channel_files(tmp_path, t1)
    assert ri.record_channel(st, t1.alias, t1.manifest_hash, 1, f1) == []
    t2 = ri.model_target(_manifest(version=2, rfilename="SmolLM2-360M-Instruct-Q4_K_S.gguf", size=259915680))
    stale = ri.record_channel(st, t2.alias, t2.manifest_hash, 2, ri.channel_files(tmp_path, t2))
    assert stale == [f1[0]["path"]]
    ri.save_state(st, tmp_path)
    assert ri.load_state(tmp_path)["channels"][t2.alias]["version"] == 2
    assert ri.installed_bytes(st) == 259915680


def test_delete_files_refuses_paths_outside_cache(tmp_path):
    inside = tmp_path / "cache" / "repo" / "v1-x" / "m.gguf"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    inside.with_suffix(".gguf.part").write_bytes(b"y")
    outside = tmp_path / "elsewhere.gguf"
    outside.write_bytes(b"z")
    removed = ri.delete_files([str(inside), str(outside)], tmp_path / "cache")
    assert removed == [str(inside)]
    assert not inside.exists() and not inside.with_suffix(".gguf.part").exists()
    assert not inside.parent.exists()          # empty version dir pruned
    assert outside.exists()


def test_sweep_orphans_only_touches_model_files(tmp_path):
    st = ri.load_state(tmp_path)
    owned = tmp_path / "repo" / "v1-a" / "owned.gguf"
    orphan = tmp_path / "repo" / "v0-old" / "old.gguf"
    part = tmp_path / "repo" / "v0-old" / "old2.gguf.part"
    other = tmp_path / "notes.txt"
    for p in (owned, orphan, part, other):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"1")
    st["channels"]["c"] = {"manifest_hash": "h", "version": 1,
                          "files": [{"rfilename": "owned.gguf", "path": str(owned), "size_bytes": 1,
                                     "artifact_root_hash": "PENDING", "chunk_bytes": 1}]}
    removed = ri.sweep_orphans(st, tmp_path)
    assert sorted(removed) == sorted([str(orphan), str(part)])
    assert owned.exists() and other.exists() and not orphan.parent.exists()
    assert list(ri.seed_index(st)) == [("h", "owned.gguf")]


def test_check_storage_enforces_budget_and_disk(tmp_path):
    st = ri.load_state(tmp_path)
    st["channels"]["c"] = {"manifest_hash": "h", "version": 1,
                          "files": [{"rfilename": "a", "path": "/x", "size_bytes": 60 << 30,
                                     "artifact_root_hash": "PENDING", "chunk_bytes": 1}]}
    ri.check_storage(tmp_path, 10 << 30, st, max_storage_bytes=100 << 30)
    with pytest.raises(RuntimeError, match="storage budget"):
        ri.check_storage(tmp_path, 50 << 30, st, max_storage_bytes=100 << 30)
    with pytest.raises(RuntimeError, match="not enough disk"):
        ri.check_storage(tmp_path, 1 << 60, st, max_storage_bytes=None)


async def test_ensure_model_swarm_first_then_origin_and_verifies(tmp_path, monkeypatch):
    m = _manifest(version=1, rfilename="tiny.gguf", size=5)
    m.quantizations[0].files[0].artifact_root_hash = ri.blake3.blake3(b"hello").hexdigest()
    calls = []

    async def good_swarm(dest, manifest_hash, shard):
        calls.append("swarm")
        dest.write_bytes(b"hello")
        return True

    path, target = await ri.ensure_model(m, tmp_path, swarm_fetch=good_swarm, log=lambda s: None)
    assert Path(path).read_bytes() == b"hello" and calls == ["swarm"]

    # a peer delivering wrong bytes is not trusted: origin is used instead
    m2 = _manifest(version=2, rfilename="tiny2.gguf", size=5)
    m2.quantizations[0].files[0].artifact_root_hash = ri.blake3.blake3(b"hello").hexdigest()

    async def bad_swarm(dest, manifest_hash, shard):
        dest.write_bytes(b"wrong")
        return True

    async def fake_origin(repo, revision, filename, dest, root_hash, log):
        calls.append("origin")
        dest.write_bytes(b"hello")

    monkeypatch.setattr(ri, "_download_hf_file", fake_origin)
    path2, _ = await ri.ensure_model(m2, tmp_path, swarm_fetch=bad_swarm, log=lambda s: None)
    assert Path(path2).read_bytes() == b"hello" and calls[-1] == "origin"

    # cached + verified -> neither swarm nor origin is called again
    calls.clear()
    await ri.ensure_model(m2, tmp_path, swarm_fetch=bad_swarm, log=lambda s: None)
    assert calls == []


# --------------------------------------------------------------------------- daemon changeover
class _FakeEngine:
    """Stands in for LlamaCppBackend: records start/stop, can be told to fail."""
    started: list[str] = []
    fail_on: set[str] = set()

    def __init__(self, model_path: str, port: int):
        self.model_path, self.port, self.running = model_path, port, False

    async def start(self):
        if Path(self.model_path).name in _FakeEngine.fail_on:
            raise RuntimeError("boom")
        self.running = True
        _FakeEngine.started.append(self.model_path)

    def stop(self):
        self.running = False


class _FakeLauncher:
    port = 8099

    def build(self, model_path, model_id):
        return _FakeEngine(model_path, self.port)


async def test_changeover_swaps_engine_deletes_old_and_rolls_back_on_failure(tmp_path, monkeypatch):
    from meshcompute_node import daemon

    monkeypatch.setattr(ri, "DEFAULT_MODEL_CACHE", tmp_path)
    _FakeEngine.started, _FakeEngine.fail_on = [], set()

    async def fetch(dest, manifest_hash, shard):
        dest.write_bytes(b"model-bytes")
        return True

    v1 = _manifest(version=1, rfilename="v1.gguf", size=11)
    v2 = _manifest(version=2, rfilename="v2.gguf", size=11)
    v3 = _manifest(version=3, rfilename="v3.gguf", size=11)
    for m in (v1, v2, v3):
        m.quantizations[0].files[0].artifact_root_hash = "PENDING-test"

    # node running v1
    p1, t1 = await ri.ensure_model(v1, tmp_path, swarm_fetch=fetch, log=lambda s: None)
    st = ri.load_state(tmp_path)
    ri.record_channel(st, t1.alias, t1.manifest_hash, 1, ri.channel_files(tmp_path, t1))
    ri.save_state(st, tmp_path)
    rt = daemon.NodeRuntime()
    rt.engine = rt.backend = _FakeEngine(p1, 8099)
    await rt.engine.start()
    rt.channel, rt.manifest_hash, rt.version = t1.alias, t1.manifest_hash, 1

    ok = await daemon.changeover(rt, _FakeLauncher(), v2, swarm_fetch=fetch, max_storage_bytes=None,
                                 cache_dir=tmp_path)
    assert ok and rt.version == 2 and rt.engine.model_path.endswith("v2.gguf") and rt.engine.running
    assert not Path(p1).exists(), "old version's file must be deleted after a successful swap"
    assert Path(rt.engine.model_path).exists()
    assert ri.load_state(tmp_path)["channels"][t1.alias]["version"] == 2
    assert rt.updating is False

    # v3's engine fails to start -> v2 restarted, v2 file kept, state unchanged
    _FakeEngine.fail_on = {"v3.gguf"}
    ok = await daemon.changeover(rt, _FakeLauncher(), v3, swarm_fetch=fetch, max_storage_bytes=None,
                                 cache_dir=tmp_path)
    assert not ok and rt.version == 2 and rt.engine.model_path.endswith("v2.gguf") and rt.engine.running
    assert ri.load_state(tmp_path)["channels"][t1.alias]["version"] == 2
    assert rt.updating is False


async def test_update_loop_ignores_lower_version(monkeypatch):
    from meshcompute_node import daemon

    rt = daemon.NodeRuntime()
    rt.channel, rt.manifest_hash, rt.version = "public/smollm2-360m", "running-hash", 5
    older = _manifest(version=4)
    swapped = []

    async def fake_fetch_manifest(control_url, alias):
        return older

    async def fake_changeover(*a, **k):
        swapped.append(1)
        return True

    monkeypatch.setattr(daemon, "fetch_manifest", fake_fetch_manifest)
    monkeypatch.setattr(daemon, "changeover", fake_changeover)
    stop = asyncio.Event()
    task = asyncio.create_task(daemon._update_loop(rt, None, "http://cp.test", 0.01, stop,
                                                   swarm_fetch=None, max_storage_bytes=None))
    await asyncio.sleep(0.1)
    stop.set()
    await task
    assert swapped == [] and rt.version == 5


# --------------------------------------------------------------------------- model choice
async def test_resolve_model_prefers_flag_then_saved_choice_then_smoke(tmp_path, monkeypatch):
    from meshcompute_node import daemon

    monkeypatch.setattr(ri, "DEFAULT_MODEL_CACHE", tmp_path)
    catalog = [_manifest()]

    async def fake_catalog(control_url):
        return catalog

    monkeypatch.setattr(daemon, "fetch_catalog", fake_catalog)
    chosen = await daemon.resolve_model("public/smollm2-360m", "http://cp.test", 10**12, interactive=False)
    assert chosen.id == "public/smollm2-360m"
    assert ri.load_state(tmp_path)["selected_alias"] == "public/smollm2-360m"   # remembered

    again = await daemon.resolve_model(None, "http://cp.test", 10**12, interactive=False)
    assert again.id == "public/smollm2-360m"                                       # reused

    smoke = await daemon.resolve_model("smoke", "http://cp.test", 10**12, interactive=False)
    nope = await daemon.resolve_model("public/nope", "http://cp.test", 10**12, interactive=False)
    assert smoke is ri.SMOKE_MODEL and nope is ri.SMOKE_MODEL

    picked = daemon.choose_model_interactively(catalog, 10**12, ask=lambda _: "1")
    assert picked is catalog[0]
    assert daemon.choose_model_interactively(catalog, 10**12, ask=lambda _: "") is None
    assert daemon.fits(catalog[0], 10**6) is False


def test_gpu_launch_args_per_accelerator():
    from meshcompute_runtime.backends.llamacpp import gpu_launch_args
    assert gpu_launch_args([], accel="metal") == ({}, ["-ngl", "999"])
    assert gpu_launch_args([], accel="cpu") == ({}, [])
    env, args = gpu_launch_args([{"index": 1, "free_vram_bytes": 1 << 30}], accel="vulkan")
    assert env == {"CUDA_VISIBLE_DEVICES": "1", "GGML_VK_VISIBLE_DEVICES": "1"} and args == ["-ngl", "999"]
    two = [{"index": 0, "free_vram_bytes": 24 << 30}, {"index": 1, "free_vram_bytes": 12 << 30}]
    env, args = gpu_launch_args(two)
    assert env == {"CUDA_VISIBLE_DEVICES": "0,1"} and args[args.index("--tensor-split") + 1] == "24576,12288"


def test_contribution_policy_fields_survive_capability(monkeypatch):
    """The daemon attaches the FULL policy to the advertised record (not just 5 fields)."""
    from meshcompute_node import daemon

    policy = ContributionPolicy(idle_only=False, max_ram_gb=7, require_ac_power=True,
                                idle_minutes_before_start=3)
    captured = {}

    async def fake_measure(identity, backend, config):
        from meshcompute_protocol import CapabilityRecord
        return CapabilityRecord(node_id=identity.node_id, backends=["llamacpp"])

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): ...
        async def post(self, url, json):
            captured["record"] = json["record"]

            class R:
                def raise_for_status(self): ...
            return R()

    monkeypatch.setattr(daemon, "measure_capability", fake_measure)
    monkeypatch.setattr(daemon.httpx, "AsyncClient", _Client)
    from meshcompute_protocol import NodeIdentity
    asyncio.run(daemon._post_capability("http://cp.test", NodeIdentity.generate(), None, policy, "auto",
                                        storage_share_bytes=5 << 30))
    pol = captured["record"]["contribution_policy"]
    assert pol["max_ram_gb"] == 7 and pol["require_ac_power"] is True
    assert pol["idle_minutes_before_start"] == 3
    assert captured["record"]["storage_share_bytes"] == 5 << 30
