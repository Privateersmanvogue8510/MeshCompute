"""LM Studio / OpenAI-compatible backend adapter.

Wraps any OpenAI-compatible chat endpoint (LM Studio, llama.cpp server, vLLM, ...).
This is the adapter that lets MeshCompute serve the 2x3090 pod's already-loaded
Qwen3.8-27B model over the internet on day one: the pod runs LM Studio's server,
this adapter proxies to it, and the gateway exposes it through the mesh API.

Cancellation: closing the async generator aborts the upstream HTTP stream.
Timeout: httpx timeouts bound every phase; TTFT has its own read timeout.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import BackendAdapter, BackendCapabilities, ChatRequest, TokenChunk


class OpenAICompatBackend(BackendAdapter):
    def __init__(self, base_url: str, *, backend_name: str = "lmstudio",
                 api_key: str | None = None, connect_timeout: float = 10.0,
                 read_timeout: float = 300.0, supports_tools: bool = True,
                 supports_vision: bool = False, max_context: int = 32768) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend_name = backend_name
        self.api_key = api_key
        self._timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._caps = BackendCapabilities(
            id=f"{backend_name}@{self.base_url}", backends=[backend_name],
            supports_streaming=True, supports_tools=supports_tools,
            supports_vision=supports_vision, supports_pipeline_split=False,
            max_context=max_context)

    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                r = await c.get(f"{self.base_url}/v1/models", headers=self._headers())
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/v1/models", headers=self._headers())
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]

    def _build_body(self, req: ChatRequest) -> dict:
        body: dict = {"model": req.model, "messages": req.messages, "stream": req.stream}
        # Map optional sampling params; only send what's set (avoid overriding backend defaults).
        for k in ("max_tokens", "temperature", "top_p", "presence_penalty", "stop"):
            v = getattr(req, k, None)
            if v is not None:
                body[k] = v
        # llama.cpp / LM Studio accept these outside the OpenAI core spec:
        for k in ("top_k", "min_p", "repetition_penalty"):
            v = getattr(req, k, None)
            if v is not None:
                body[k] = v
        if req.tools:
            body["tools"] = req.tools
        body.update(req.extra)
        return body

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[TokenChunk]:
        body = self._build_body(req)
        body["stream"] = True
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            async with c.stream("POST", f"{self.base_url}/v1/chat/completions",
                                 headers=self._headers(), json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    tool_delta = delta.get("tool_calls")
                    finish = choice.get("finish_reason")
                    if text or tool_delta or finish:
                        yield TokenChunk(text=text,
                                         finish_reason=finish,
                                         tool_call_delta=tool_delta[0] if tool_delta else None)
