"""Gateway /v1/chat/completions end-to-end against a respx-mocked control-plane +
OpenAI-compatible backend (no network, no live pod)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

PLAN_JSON = {
    "plan_id": "plan_test",
    "model_manifest_hash": "hash1",
    "model_id": "public/qwen3.8-27b-fable",
    "strategy": "single",
    "peer_chain": None,
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


def test_streaming_chat_completions_is_openai_shaped_sse(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "public/qwen3.8-27b-fable"}]}))
        router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))

        response = gateway_client.post(
            "/v1/chat/completions",
            json={"model": "public/qwen3.8-27b-fable",
                  "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    raw_events = [e for e in response.text.replace("\r\n", "\n").split("\n\n") if e.strip()]
    data_lines = [e[len("data: "):] for e in raw_events if e.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"

    chunk_payloads = [json.loads(d) for d in data_lines[:-1]]
    assert chunk_payloads  # got at least one real chunk before [DONE]
    for chunk in chunk_payloads:
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk and "delta" in chunk["choices"][0]
    assert chunk_payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_non_streaming_chat_completions_aggregates_to_one_completion(gateway_client, gateway_env):
    plan = {**PLAN_JSON, "peer_chain": [gateway_env["node_id"]]}
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{gateway_env['control_url']}/api/v1/schedule").mock(
            return_value=httpx.Response(200, json=plan))
        router.get(f"{gateway_env['backend_url']}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "public/qwen3.8-27b-fable"}]}))
        router.post(f"{gateway_env['backend_url']}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_BODY))

        response = gateway_client.post(
            "/v1/chat/completions",
            json={"model": "public/qwen3.8-27b-fable",
                  "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] >= 1
