"""Control-plane privilege-boundary tests (SECURITY.md: falsified capabilities,
replayed receipts, untrusted manifests). Runs against a real FastAPI app + a
throwaway sqlite DB — no network, no live pod.
"""

from __future__ import annotations

from pathlib import Path

from meshcompute_protocol import (
    CapabilityRecord, GpuInfo, HeartbeatRequest, MeasuredWork, NodeIdentity, RegisterRequest,
    WorkReceipt,
)

QWEN = "public/qwen3.8-27b-fable"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _signed(ident: NodeIdentity, model) -> dict:
    body = model.model_dump(mode="json", exclude={"signature_b64"})
    body["signature_b64"] = ident.sign_json(body)
    return body


def _register(client, ident: NodeIdentity, pool_ids=None):
    return client.post("/api/v1/nodes/register", json=_signed(ident, RegisterRequest(
        node_id=ident.node_id, public_b64=ident.public_b64, pool_ids=pool_ids or ["public"])))


def _heartbeat(client, ident: NodeIdentity, **fields):
    return client.post(f"/api/v1/nodes/{ident.node_id}/heartbeat",
                       json=_signed(ident, HeartbeatRequest(node_id=ident.node_id, **fields)))


def _big_capability(client, ident: NodeIdentity):
    record = CapabilityRecord(node_id=ident.node_id, backends=["llamacpp"], strategies=["single", "replica"],
                              gpus=[GpuInfo(vendor="nvidia", model="test", vram_bytes=48_000_000_000,
                                            free_vram_bytes=48_000_000_000)])
    cap = {"record": record.model_dump(mode="json"), "public_b64": ident.public_b64,
           "signature_b64": ident.sign_json(record.model_dump(mode="json"))}
    r = client.post(f"/api/v1/nodes/{ident.node_id}/capabilities", json=cap)
    assert r.status_code == 200, r.text


def _schedule(client, node: NodeIdentity) -> dict:
    """Register a node big enough for the qwen manifest, heartbeat it (signed),
    and get a real plan for it — the prerequisite for a creditable receipt."""
    _big_capability(client, node)
    hb = _heartbeat(client, node, ram_free_bytes=64_000_000_000)
    assert hb.status_code == 200, hb.text
    plan = client.post("/api/v1/schedule", json={"model_id": QWEN, "client_node_id": "client"})
    assert plan.status_code == 200, plan.text
    return {"plan": plan.json(), "nonce": hb.json()["challenge_nonce"]}


def _receipt_body(ident: NodeIdentity, plan: dict, nonce: str, receipt_id: str = "r1", **work) -> dict:
    receipt = WorkReceipt(receipt_id=receipt_id, session_id="s1", plan_id=plan["plan_id"],
                          node_id=ident.node_id, model_manifest_hash=plan["model_manifest_hash"],
                          role="single", started_at=1000.0, finished_at=1002.0,
                          measured_work=MeasuredWork(**(work or {"tokens_generated": 10})),
                          challenge_nonce=nonce)
    return {"receipt": receipt.model_dump(mode="json"), "public_b64": ident.public_b64,
            "signature_b64": ident.sign_json(receipt.model_dump(mode="json"))}


def test_register_then_valid_capability_accepted(cp_client):
    ident = NodeIdentity.generate()
    r = _register(cp_client, ident)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    record = CapabilityRecord(node_id=ident.node_id, backends=["lmstudio"], strategies=["single"])
    signature = ident.sign_json(record.model_dump(mode="json"))
    cap = {"record": record.model_dump(mode="json"), "public_b64": ident.public_b64,
           "signature_b64": signature}

    r2 = cp_client.post(f"/api/v1/nodes/{ident.node_id}/capabilities", json=cap)
    assert r2.status_code == 200
    assert r2.json()["ok"] is True


def test_capability_with_wrong_signature_rejected(cp_client):
    ident = NodeIdentity.generate()
    _register(cp_client, ident)

    record = CapabilityRecord(node_id=ident.node_id, backends=["lmstudio"])
    wrong_ident = NodeIdentity.generate()  # signs with the WRONG key
    bad_signature = wrong_ident.sign_json(record.model_dump(mode="json"))
    cap = {"record": record.model_dump(mode="json"), "public_b64": ident.public_b64,
           "signature_b64": bad_signature}

    r = cp_client.post(f"/api/v1/nodes/{ident.node_id}/capabilities", json=cap)
    assert r.status_code == 400


def test_capability_whose_pubkey_does_not_derive_node_id_rejected(cp_client):
    ident = NodeIdentity.generate()
    _register(cp_client, ident)

    other = NodeIdentity.generate()
    record = CapabilityRecord(node_id=ident.node_id, backends=["lmstudio"])
    signature = other.sign_json(record.model_dump(mode="json"))
    # public_b64 is legit (belongs to `other`) but does not derive ident.node_id
    cap = {"record": record.model_dump(mode="json"), "public_b64": other.public_b64,
           "signature_b64": signature}

    r = cp_client.post(f"/api/v1/nodes/{ident.node_id}/capabilities", json=cap)
    assert r.status_code == 400


def test_work_receipt_replay_rejected(cp_client):
    ident = NodeIdentity.generate()
    _register(cp_client, ident)
    ctx = _schedule(cp_client, ident)

    r1 = cp_client.post("/api/v1/work-receipts", json=_receipt_body(ident, ctx["plan"], ctx["nonce"]))
    assert r1.status_code == 200, r1.text
    assert r1.json() == {"ok": True, "credited": 10.0}   # credit_for_receipt: tokens generated

    # same nonce again, even under a fresh receipt_id -> replay (nonce consumed).
    r2 = cp_client.post("/api/v1/work-receipts",
                        json=_receipt_body(ident, ctx["plan"], ctx["nonce"], receipt_id="r2"))
    assert r2.status_code == 400
    assert "not issued" in r2.text


def test_work_receipt_needs_issued_nonce_real_plan_and_membership(cp_client):
    ident = NodeIdentity.generate()
    _register(cp_client, ident)
    ctx = _schedule(cp_client, ident)

    forged = _receipt_body(ident, ctx["plan"], "made-up-nonce")
    assert cp_client.post("/api/v1/work-receipts", json=forged).status_code == 400

    no_plan = _receipt_body(ident, {**ctx["plan"], "plan_id": "NO_SUCH_PLAN"}, ctx["nonce"])
    assert cp_client.post("/api/v1/work-receipts", json=no_plan).status_code == 400

    # a different registered node can't claim work from a plan it wasn't in
    other = NodeIdentity.generate()
    _register(cp_client, other)
    other_nonce = _heartbeat(cp_client, other).json()["challenge_nonce"]
    stolen = _receipt_body(other, ctx["plan"], other_nonce)
    r = cp_client.post("/api/v1/work-receipts", json=stolen)
    assert r.status_code == 400 and "not part of that plan" in r.text

    # implausible work: 1e9 ms of GPU time inside a 2-second receipt window
    huge = _receipt_body(ident, ctx["plan"], ctx["nonce"], gpu_active_ms=1e9)
    r = cp_client.post("/api/v1/work-receipts", json=huge)
    assert r.status_code == 400 and "plausible" in r.text


def test_register_rejects_unsigned_and_foreign_key(cp_client):
    victim = NodeIdentity.generate()
    assert _register(cp_client, victim).status_code == 200

    unsigned = cp_client.post("/api/v1/nodes/register", json={
        "node_id": victim.node_id, "public_b64": victim.public_b64, "pool_ids": ["public"]})
    assert unsigned.status_code == 400

    attacker = NodeIdentity.generate()
    takeover = RegisterRequest(node_id=victim.node_id, public_b64=attacker.public_b64)
    r = cp_client.post("/api/v1/nodes/register", json=_signed(attacker, takeover))
    assert r.status_code == 400   # node_id must derive from the key

    # victim still owns its record: its signed capability is accepted
    record = CapabilityRecord(node_id=victim.node_id, backends=["lmstudio"])
    cap = {"record": record.model_dump(mode="json"), "public_b64": victim.public_b64,
           "signature_b64": victim.sign_json(record.model_dump(mode="json"))}
    assert cp_client.post(f"/api/v1/nodes/{victim.node_id}/capabilities", json=cap).status_code == 200


def test_heartbeat_must_be_signed_by_registered_key(cp_client):
    ident = NodeIdentity.generate()
    _register(cp_client, ident)
    assert _heartbeat(cp_client, ident, ram_free_bytes=1).status_code == 200

    spoof = cp_client.post(f"/api/v1/nodes/{ident.node_id}/heartbeat",
                           json={"node_id": ident.node_id, "ram_free_bytes": 10**15})
    assert spoof.status_code == 400

    other = NodeIdentity.generate()
    wrong_key = _signed(other, HeartbeatRequest(node_id=ident.node_id, ram_free_bytes=10**15))
    assert cp_client.post(f"/api/v1/nodes/{ident.node_id}/heartbeat", json=wrong_key).status_code == 400


def test_get_model_by_alias_with_slash(cp_client):
    r = cp_client.get(f"/api/v1/models/{QWEN}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["manifest"]["id"] == QWEN and body["manifest_hash"] and body["signature_b64"]
    assert cp_client.get("/api/v1/models/public/does-not-exist").status_code == 404


def test_pools_list_and_create(cp_client):
    assert {"id": "public", "private": False} in cp_client.get("/api/v1/pools").json()
    assert cp_client.post("/api/v1/pools", json={"id": "studio", "private": True}).status_code == 200
    assert cp_client.post("/api/v1/pools", json={"id": "studio", "private": True}).status_code == 409
    assert cp_client.post("/api/v1/pools", json={"id": "bad/id"}).status_code == 400
    assert {"id": "studio", "private": True} in cp_client.get("/api/v1/pools").json()


def test_manifest_rescan_publishes_new_version_and_refuses_rollback(monkeypatch, tmp_path):
    import shutil

    import yaml
    from fastapi.testclient import TestClient

    src = REPO_ROOT / "models" / "manifests" / "public-smollm2-360m.yaml"
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    shutil.copy(src, mdir / "smollm2.yaml")
    monkeypatch.setenv("MESH_CP_DB", str(tmp_path / "control.db"))
    monkeypatch.setenv("MESH_CP_MANIFEST_DIR", str(mdir))
    from meshcompute_control.app import create_app

    with TestClient(create_app()) as client:
        views = {m["id"]: m for m in client.get("/api/v1/models").json()}
        assert views["public/smollm2-360m"]["version"] == 1
        assert views["public/smollm2-360m"]["size_bytes"] == 270590880

        # operator pushes v2 (new file + hash) by editing the YAML in place
        data = yaml.safe_load((mdir / "smollm2.yaml").read_text())
        data["version"] = 2
        data["quantizations"][0]["files"][0]["rfilename"] = "SmolLM2-360M-Instruct-Q4_K_S.gguf"
        data["quantizations"][0]["files"][0]["size_bytes"] = 259915680
        data["quantizations"][0]["files"][0]["artifact_root_hash"] = "3" * 64
        (mdir / "smollm2.yaml").write_text(yaml.safe_dump(data))
        assert client.app.state.scan_manifests() == ["public/smollm2-360m"]
        m = client.get("/api/v1/models/public/smollm2-360m").json()["manifest"]
        assert m["version"] == 2 and m["quantizations"][0]["files"][0]["size_bytes"] == 259915680

        # a stale v1 file dropped back in must NOT roll the channel back
        shutil.copy(src, mdir / "smollm2.yaml")
        assert client.app.state.scan_manifests() == []
        assert client.get("/api/v1/models/public/smollm2-360m").json()["manifest"]["version"] == 2


def test_models_endpoint_serves_signed_qwen_manifest(cp_client):
    r = cp_client.get("/api/v1/models")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()}
    assert "public/qwen3.8-27b-fable" in by_id
    assert by_id["public/qwen3.8-27b-fable"]["signed"] is True
    assert by_id["public/qwen3.8-27b-fable"]["manifest_hash"]
