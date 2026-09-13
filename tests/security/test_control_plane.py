"""Control-plane privilege-boundary tests (SECURITY.md: falsified capabilities,
replayed receipts, untrusted manifests). Runs against a real FastAPI app + a
throwaway sqlite DB — no network, no live pod.
"""

from __future__ import annotations

from meshcompute_protocol import CapabilityRecord, NodeIdentity, WorkReceipt


def _register(client, ident: NodeIdentity, pool_ids=None):
    return client.post("/api/v1/nodes/register", json={
        "node_id": ident.node_id, "public_b64": ident.public_b64,
        "pool_ids": pool_ids or ["public"],
    })


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

    receipt = WorkReceipt(receipt_id="r1", session_id="s1", plan_id="p1", node_id=ident.node_id,
                          model_manifest_hash="h1", role="single", challenge_nonce="nonce-1")
    signature = ident.sign_json(receipt.model_dump(mode="json"))
    body = {"receipt": receipt.model_dump(mode="json"), "public_b64": ident.public_b64,
            "signature_b64": signature}

    r1 = cp_client.post("/api/v1/work-receipts", json=body)
    assert r1.status_code == 200
    assert r1.json()["ok"] is True

    # same node_id + same challenge_nonce, even under a fresh receipt_id -> replay.
    replay_receipt = receipt.model_copy(update={"receipt_id": "r2"})
    replay_signature = ident.sign_json(replay_receipt.model_dump(mode="json"))
    replay_body = {"receipt": replay_receipt.model_dump(mode="json"),
                   "public_b64": ident.public_b64, "signature_b64": replay_signature}

    r2 = cp_client.post("/api/v1/work-receipts", json=replay_body)
    assert r2.status_code == 409


def test_models_endpoint_serves_signed_qwen_manifest(cp_client):
    r = cp_client.get("/api/v1/models")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()}
    assert "public/qwen3.8-27b-fable" in by_id
    assert by_id["public/qwen3.8-27b-fable"]["signed"] is True
    assert by_id["public/qwen3.8-27b-fable"]["manifest_hash"]
