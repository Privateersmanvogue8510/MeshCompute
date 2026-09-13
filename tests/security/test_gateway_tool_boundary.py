"""Gateway tool-credential boundary (SECURITY.md inference-worker boundary,
INSTRUCTIONS §2.4): a client's Authorization header must never reach the
inference backend. All outbound HTTP is respx-mocked; no network, no live pod.
"""

from __future__ import annotations

import httpx
import pytest
import respx

PLAN_JSON = {
    "plan_id": "plan_test",
    "model_manifest_hash": "hash1",
    "model_id": "public/qwen3.8-27b-fable",
    "strategy": "single",
    "peer_chain": None,  # filled in per-test with the mocked node_id
    "context_length": 0,
}

SSE_BODY = (
    'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)


@pytest.fixture(autouse=True)
def _clear_models_cache():
    """router._models_cache is keyed by backend URL with a 60s TTL — clear it so the
    /v1/models probe below is actually exercised (and never served a stale list)."""
    from meshcompute_gateway import router as gw_router

    gw_router._models_cache.clear()
    yield
    gw_router._models_cache.clear()


def test_authorization_header_never_reaches_backend(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}

    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "public/qwen3.8-27b-fable"}]}))
        backend_route = router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))

        response = gateway_client.post(
            "/v1/chat/completions",
            json={"model": "public/qwen3.8-27b-fable",
                  "messages": [{"role": "user", "content": "hi"}], "stream": False},
            headers={"Authorization": "Bearer super-secret-user-token"},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello world"

    backend_request = backend_route.calls.last.request
    assert "authorization" not in backend_request.headers
    assert "Bearer" not in backend_request.headers.get("cookie", "")

    # X-Mesh-* response headers must be set (INSTRUCTIONS §8/§22: expose the plan).
    assert response.headers["x-mesh-plan-id"] == "plan_test"
    assert response.headers["x-mesh-strategy"] == "single"
    assert response.headers["x-mesh-path"] == gateway_env["node_id"]


def test_streaming_chat_also_withholds_authorization(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}

    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "public/qwen3.8-27b-fable"}]}))
        backend_route = router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))

        response = gateway_client.post(
            "/v1/chat/completions",
            json={"model": "public/qwen3.8-27b-fable",
                  "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers={"Authorization": "Bearer super-secret-user-token"},
        )

    assert response.status_code == 200
    backend_request = backend_route.calls.last.request
    assert "authorization" not in backend_request.headers
    assert response.headers["x-mesh-plan-id"] == "plan_test"
