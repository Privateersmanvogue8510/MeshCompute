"""Gateway routing/plan-validation behaviour: how a control-plane answer maps onto
an HTTP status, which plans this Phase-1 gateway refuses to execute, and that it
only routes to a node actually serving the requested model. All outbound HTTP is
respx-mocked; no network, no live pod.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest
import respx

MODEL = "public/qwen3.8-27b-fable"

PLAN_JSON = {
    "plan_id": "plan_test",
    "model_manifest_hash": "hash1",
    "model_id": MODEL,
    "strategy": "single",
    "peer_chain": None,      # filled in per-test with the mocked node_id
    "context_length": 0,
}

SSE_BODY = (
    'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)


@pytest.fixture(autouse=True)
def _clear_models_cache():
    """router._models_cache is keyed by backend URL with a 60s TTL, so it would
    otherwise leak a previous test's model list into the next one."""
    from meshcompute_gateway import router as gw_router

    gw_router._models_cache.clear()
    yield
    gw_router._models_cache.clear()


def _chat(client, stream=False):
    return client.post("/v1/chat/completions",
                       json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                             "stream": stream})


# --- A: control-plane status -> gateway status ------------------------------------

def test_control_plane_404_becomes_gateway_404(gateway_client, gateway_env):
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(404, text="unknown model"))
        response = _chat(gateway_client)

    assert response.status_code == 404
    assert MODEL in response.json()["detail"]


def test_control_plane_400_becomes_gateway_502(gateway_client, gateway_env):
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(400, text="bad request"))
        response = _chat(gateway_client)

    # must be a mapped 502, never an httpx.HTTPStatusError escaping as a 500
    assert response.status_code == 502


def test_control_plane_500_stays_503(gateway_client, gateway_env):
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(500, text="infeasible"))
        response = _chat(gateway_client)

    assert response.status_code == 503


def test_schedule_request_asks_only_for_single_strategy(gateway_client, gateway_env):
    import json

    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=True) as router:
        schedule_route = router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MODEL}]}))
        router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        assert _chat(gateway_client).status_code == 200

    sent = json.loads(schedule_route.calls.last.request.content)
    assert sent["executable_strategies"] == ["single"]


# --- B: plans this gateway cannot execute -----------------------------------------

def test_multi_peer_pipeline_plan_is_501_and_never_calls_a_backend(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "strategy": "pipeline",
            "peer_chain": [gateway_env["node_id"], "nd_other"]}
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        backend_route = router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        models_route = router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MODEL}]}))
        response = _chat(gateway_client)

    assert response.status_code == 501
    assert "pipeline" in response.json()["detail"] and "2 peers" in response.json()["detail"]
    assert not backend_route.called and not models_route.called
    # and no header may claim a strategy we didn't run
    assert "x-mesh-strategy" not in response.headers


# --- C: node must actually serve the requested model ------------------------------

def test_backend_serving_a_different_model_is_409(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "public/some-other-model"}]}))
        backend_route = router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        response = _chat(gateway_client)

    assert response.status_code == 409
    assert "public/some-other-model" in response.json()["detail"]
    assert not backend_route.called


def test_backend_serving_the_requested_model_is_200(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MODEL}]}))
        router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        response = _chat(gateway_client)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello"


def test_backend_reporting_no_models_is_allowed_through(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []}))
        router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        response = _chat(gateway_client)

    assert response.status_code == 200


# --- E: /v1/models ----------------------------------------------------------------

def test_models_control_plane_error_is_502(gateway_client, gateway_env):
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{gateway_env['control_url']}/api/v1/models").mock(
            return_value=httpx.Response(500, text="boom"))
        response = gateway_client.get("/v1/models")

    assert response.status_code == 502
    assert "500" in response.json()["detail"]


def test_models_falls_back_only_when_control_plane_is_unreachable(gateway_client, gateway_env):
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{gateway_env['control_url']}/api/v1/models").mock(
            side_effect=httpx.ConnectError("no route"))
        response = gateway_client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"]  # dev_model_catalog() from models/manifests


def test_models_passes_through_version_and_size(gateway_client, gateway_env):
    payload = [{"id": MODEL, "manifest_hash": "h", "version": 7, "size_bytes": 123}]
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{gateway_env['control_url']}/api/v1/models").mock(
            return_value=httpx.Response(200, json=payload))
        response = gateway_client.get("/v1/models")

    entry = response.json()["data"][0]
    assert entry["id"] == MODEL and entry["object"] == "model"
    assert entry["version"] == 7 and entry["size_bytes"] == 123


# --- D: nodes.local.yaml re-read on edit ------------------------------------------

def test_edited_nodes_file_is_picked_up_without_cache_clear(gateway_client, gateway_env):
    import yaml

    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MODEL}]}))
        router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        assert _chat(gateway_client).status_code == 200  # first load caches the file

    # operator adds a second worker while the gateway keeps running
    nodes_file = os.environ["NODES_LOCAL_FILE"]
    new_url = "http://backend2.test"
    with open(nodes_file, "w") as fh:
        yaml.safe_dump({"nodes": [
            {"node_id": gateway_env["node_id"], "backend_url": gateway_env["backend_url"],
             "backend": "lmstudio"},
            {"node_id": "nd_test_2", "backend_url": new_url, "backend": "lmstudio"},
        ]}, fh)
    future = time.time() + 2
    os.utime(nodes_file, (future, future))

    plan2 = {**PLAN_JSON, "peer_chain": ["nd_test_2"]}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan2))
        router.get(f"{new_url}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MODEL}]}))
        router.post(f"{new_url}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))
        response = _chat(gateway_client)

    assert response.status_code == 200
    assert response.headers["x-mesh-path"] == "nd_test_2"


# --- F: bounded session store + DELETE --------------------------------------------

def test_session_store_is_capped_and_evicts_the_oldest(gateway_client):
    from meshcompute_gateway import app as gw_app

    gw_app._sessions.clear()
    gw_app._events.clear()
    gw_app._cancel_events.clear()

    first = gateway_client.post("/api/v1/sessions", json={"model_id": MODEL}).json()["session_id"]
    for _ in range(gw_app.MAX_SESSIONS):
        assert gateway_client.post("/api/v1/sessions", json={"model_id": MODEL}).status_code == 200

    assert len(gw_app._sessions) <= gw_app.MAX_SESSIONS
    assert gateway_client.get(f"/api/v1/sessions/{first}").status_code == 404
    assert first not in gw_app._events and first not in gw_app._cancel_events


def test_delete_session_removes_it(gateway_client):
    from meshcompute_gateway import app as gw_app

    session_id = gateway_client.post("/api/v1/sessions",
                                     json={"model_id": MODEL}).json()["session_id"]
    assert gateway_client.delete(f"/api/v1/sessions/{session_id}").status_code == 200
    assert gateway_client.get(f"/api/v1/sessions/{session_id}").status_code == 404
    assert gateway_client.delete(f"/api/v1/sessions/{session_id}").status_code == 404
    assert session_id not in gw_app._events


def test_harness_field_is_bound_not_silently_dropped():
    """The CLI sends --harness; pydantic ignores extra keys, so the field must exist
    on the model or the value vanishes before Phase 2.5 can ever use it."""
    from meshcompute_gateway.app import ChatCompletionRequest

    body = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                 harness="claude-code")
    assert body.harness == "claude-code"


# --- G: `mesh api serve` wiring ---------------------------------------------------

def test_api_serve_exports_gateway_control_url_and_host(monkeypatch):
    import uvicorn
    from click.testing import CliRunner
    from meshcompute_cli.main import cli

    # setenv first so monkeypatch restores both vars the command overwrites directly
    monkeypatch.setenv("MESH_CONTROL_URL", "http://stale.test")
    monkeypatch.setenv("MESH_GW_CONTROL_URL", "http://stale.test")
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))

    result = CliRunner().invoke(cli, ["api", "serve", "--control-url", "http://cp.test",
                                      "--host", "0.0.0.0", "--port", "9999"])

    assert result.exit_code == 0, result.output
    # the gateway's settings read MESH_GW_CONTROL_URL, not MESH_CONTROL_URL
    assert os.environ["MESH_GW_CONTROL_URL"] == "http://cp.test"
    assert os.environ["MESH_CONTROL_URL"] == "http://cp.test"
    assert captured["host"] == "0.0.0.0" and captured["port"] == 9999
