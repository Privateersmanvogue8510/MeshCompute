"""OpenAICompatBackend.chat_stream against a respx-mocked OpenAI-compatible
endpoint (LM Studio / llama.cpp shaped SSE) — no network."""

from __future__ import annotations

import httpx
import pytest
import respx
from meshcompute_runtime.backends import ChatRequest, OpenAICompatBackend

BACKEND_URL = "http://backend.test"


async def _collect(backend: OpenAICompatBackend, req: ChatRequest):
    return [chunk async for chunk in backend.chat_stream(req)]


@pytest.mark.asyncio
async def test_chat_stream_parses_deltas_and_stops_on_done():
    sse_body = (
        'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":" there"},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
        # anything after [DONE] must never be parsed/yielded.
        'data: {"choices":[{"delta":{"content":"should not appear"}}]}\n\n'
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{BACKEND_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse_body))

        backend = OpenAICompatBackend(BACKEND_URL)
        req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
        chunks = await _collect(backend, req)

    texts = [c.text for c in chunks if c.text]
    assert texts == ["Hi", " there"]
    assert "should not appear" not in "".join(texts)
    assert chunks[-1].finish_reason == "stop"
    assert all(c.finish_reason is None for c in chunks[:-1])


@pytest.mark.asyncio
async def test_chat_stream_honours_tool_call_delta():
    sse_body = (
        'data: {"choices":[{"delta":{"tool_calls":[{"id":"c1","function":'
        '{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        'data: [DONE]\n\n'
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{BACKEND_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse_body))

        backend = OpenAICompatBackend(BACKEND_URL)
        req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
        chunks = await _collect(backend, req)

    assert len(chunks) == 1
    assert chunks[0].tool_call_delta == {"id": "c1", "function": {"name": "search", "arguments": "{}"}}
    assert chunks[0].finish_reason == "tool_calls"
