"""MeshCompute gateway (INSTRUCTIONS §13): the trusted, user-facing API.

Turns a chat/session request into a scheduler plan (router.py) then streams tokens
from the selected worker's OpenAI-compatible backend (node/runtime backends).

SECURITY (SECURITY.md tool boundary, INSTRUCTIONS §2.4 — CRITICAL):
Public inference workers must never receive tool credentials, API keys, cookies, or
the client's Authorization header. This module never reads request.headers to build
a backend call; `_build_backend_request` whitelists ONLY inference fields onto the
backend's ChatRequest. Tools/secrets stay in this trusted gateway zone.

Also: do not log prompt bodies (SECURITY.md, INSTRUCTIONS §18) — no logging of
`messages`/`content` is added anywhere in this module.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from meshcompute_runtime.backends import BackendAdapter, ChatRequest, TokenChunk
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .router import SchedulerUnavailable, resolve_backend
from .settings import dev_model_catalog, get_settings

app = FastAPI(title="MeshCompute Gateway", version="0.1.0")


def create_app() -> FastAPI:
    return app


# --------------------------------------------------------------------------- helpers

def _estimate_prompt_tokens(messages: list[dict]) -> int:
    # ponytail: chars/4 heuristic, no real tokenizer wired up yet (Phase-1). Upgrade
    # to the model's actual tokenizer if scheduling/usage accuracy matters later.
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, chars // 4)


def _build_backend_request(*, model: str, messages: list[dict], stream: bool,
                           max_tokens: int | None = None, temperature: float | None = None,
                           top_p: float | None = None, top_k: int | None = None,
                           min_p: float | None = None, repetition_penalty: float | None = None,
                           presence_penalty: float | None = None,
                           stop: list[str] | str | None = None,
                           tools: list[dict] | None = None) -> ChatRequest:
    """Build the backend request from ONLY inference fields (see module docstring).
    Never pass through client headers/cookies/credentials here."""
    if isinstance(stop, str):
        stop = [stop]
    return ChatRequest(model=model, messages=messages, stream=stream, max_tokens=max_tokens,
                       temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                       repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                       stop=stop, tools=tools)


def _chunk_obj(completion_id: str, model: str, delta: dict, finish_reason: str | None) -> dict:
    return {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def _sse_chunks(gen: AsyncIterator[TokenChunk], completion_id: str, model: str,
                      content_out: list[str], cancel_event: asyncio.Event | None = None,
                      ) -> AsyncIterator[str]:
    """Shared SSE relay for /v1/chat/completions and native session messages: turns
    backend TokenChunks into OpenAI chat.completion.chunk JSON lines, appends text
    into `content_out` (so a caller can record the full reply), and ends with
    '[DONE]'. Honours client disconnect / explicit cancel by closing `gen`."""
    sent_role = False
    try:
        async for tc in gen:
            if cancel_event is not None and cancel_event.is_set():
                break
            delta: dict[str, Any] = {}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            if tc.text:
                delta["content"] = tc.text
                content_out.append(tc.text)
            if tc.tool_call_delta:
                delta["tool_calls"] = [tc.tool_call_delta]
            yield json.dumps(_chunk_obj(completion_id, model, delta, tc.finish_reason))
            if tc.finish_reason:
                break
        yield "[DONE]"
    except asyncio.CancelledError:
        raise  # client disconnected — let it propagate, don't swallow
    except Exception as e:  # backend blew up mid-stream: tell the client, then close
        yield json.dumps({"error": {"message": str(e)}})
    finally:
        await gen.aclose()  # cancel the upstream HTTP stream, always


async def _aggregate_chat(backend: BackendAdapter, req: ChatRequest,
                          ) -> tuple[str, str | None, list[dict] | None]:
    content = []
    finish_reason = None
    tool_calls: list[dict] | None = None
    async for tc in backend.chat_stream(req):
        if tc.text:
            content.append(tc.text)
        if tc.tool_call_delta:
            tool_calls = (tool_calls or [])
            tool_calls.append(tc.tool_call_delta)
        if tc.finish_reason:
            finish_reason = tc.finish_reason
    return "".join(content), finish_reason, tool_calls


def _mesh_headers(plan) -> dict[str, str]:
    return {"X-Mesh-Plan-Id": plan.plan_id, "X-Mesh-Strategy": plan.strategy.value,
            "X-Mesh-Path": ",".join(plan.peer_chain)}


# --------------------------------------------------------------------------- /v1 (OpenAI-compatible)

@app.get("/v1/models")
async def list_models():
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.mesh_gw_control_url}/api/v1/models")
            resp.raise_for_status()
            models = resp.json()
    except httpx.HTTPError:
        models = dev_model_catalog()  # dev/test fallback: control-plane not reachable
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "owned_by": "meshcompute"} for m in models]}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    tools: list[dict] | None = None


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    try:
        plan, backend = await resolve_backend(
            model_id=body.model, client_node_id="client", ctx=0,
            prompt_tokens=_estimate_prompt_tokens(body.messages))
    except SchedulerUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    headers = _mesh_headers(plan)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    req = _build_backend_request(model=body.model, messages=body.messages, stream=True,
                                 max_tokens=body.max_tokens, temperature=body.temperature,
                                 top_p=body.top_p, top_k=body.top_k, min_p=body.min_p,
                                 repetition_penalty=body.repetition_penalty,
                                 presence_penalty=body.presence_penalty, stop=body.stop,
                                 tools=body.tools)

    if body.stream:
        gen = backend.chat_stream(req)
        return EventSourceResponse(
            _sse_chunks(gen, completion_id, body.model, []), headers=headers)

    try:
        content, finish_reason, tool_calls = await _aggregate_chat(backend, req)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"backend error: {e}")

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    prompt_tokens = _estimate_prompt_tokens(body.messages)
    completion_tokens = max(1, len(content) // 4)
    completion = {
        "id": completion_id, "object": "chat.completion", "created": int(time.time()),
        "model": body.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
    return JSONResponse(completion, headers=headers)


# --------------------------------------------------------------------------- native sessions
# Phase-1 in-memory, event-sourced session store (INSTRUCTIONS §19). Not durable across
# restarts and not shared across gateway processes — Phase 2 swaps this for persisted
# storage behind the same functions without changing the event/session shape.
_sessions: dict[str, dict] = {}
_events: dict[str, list[dict]] = {}
_cancel_events: dict[str, asyncio.Event] = {}


class CreateSessionRequest(BaseModel):
    model_id: str
    harness_id: str = "native"
    pool_id: str = "public"
    tools: list[str] = Field(default_factory=list)
    memory_mode: str = "session"


class SessionMessageRequest(BaseModel):
    content: str
    role: str = "user"
    stream: bool = True


def _append_event(session: dict, content_type: str, payload: dict) -> dict:
    event = {"event_id": f"evt_{uuid.uuid4().hex[:16]}", "session_id": session["session_id"],
             "timestamp": time.time(), "harness_id": session["harness_id"],
             "model_id": session["model_id"], "content_type": content_type, "payload": payload}
    _events[session["session_id"]].append(event)
    return event


def _session_messages(session_id: str) -> list[dict]:
    return [{"role": e["payload"]["role"], "content": e["payload"]["content"]}
            for e in _events[session_id] if e["content_type"].endswith("_message")]


def _get_session_or_404(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.post("/api/v1/sessions")
async def create_session(body: CreateSessionRequest):
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    session = {"session_id": session_id, "model_id": body.model_id, "harness_id": body.harness_id,
              "pool_id": body.pool_id, "tools": body.tools, "memory_mode": body.memory_mode,
              "created_at": time.time(), "status": "active"}
    _sessions[session_id] = session
    _events[session_id] = []
    return session


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    return _get_session_or_404(session_id)


@app.post("/api/v1/sessions/{session_id}/messages")
async def post_session_message(session_id: str, body: SessionMessageRequest):
    session = _get_session_or_404(session_id)
    _append_event(session, "user_message", {"role": body.role, "content": body.content})
    messages = _session_messages(session_id)

    try:
        plan, backend = await resolve_backend(
            model_id=session["model_id"], client_node_id="client", ctx=0,
            prompt_tokens=_estimate_prompt_tokens(messages))
    except SchedulerUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    headers = _mesh_headers(plan)
    req = _build_backend_request(model=session["model_id"], messages=messages, stream=True)
    cancel_event = _cancel_events.setdefault(session_id, asyncio.Event())
    cancel_event.clear()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.stream:
        async def _stream_and_record():
            gen = backend.chat_stream(req)
            content_parts: list[str] = []
            try:
                async for piece in _sse_chunks(gen, completion_id, session["model_id"],
                                               content_parts, cancel_event):
                    yield piece
            finally:
                _append_event(session, "assistant_message",
                              {"role": "assistant", "content": "".join(content_parts)})

        return EventSourceResponse(_stream_and_record(), headers=headers)

    try:
        content, finish_reason, _tool_calls = await _aggregate_chat(backend, req)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"backend error: {e}")
    _append_event(session, "assistant_message", {"role": "assistant", "content": content})
    return JSONResponse({"session_id": session_id, "content": content,
                         "finish_reason": finish_reason or "stop"}, headers=headers)


@app.post("/api/v1/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    _get_session_or_404(session_id)
    _cancel_events.setdefault(session_id, asyncio.Event()).set()
    return {"ok": True, "session_id": session_id, "status": "cancel_requested"}


@app.get("/api/v1/sessions/{session_id}/events")
async def get_session_events(session_id: str):
    _get_session_or_404(session_id)
    return {"session_id": session_id, "events": _events[session_id]}


# --------------------------------------------------------------------------- harnesses
# Stub registry (INSTRUCTIONS §2.3/§13): harness adapters beyond "native" are Phase 2.5.
# The endpoint/shape is stable now; a real adapter registers by appending here.
_HARNESSES = [
    {"id": "native", "display_name": "MeshCompute Native", "version": "0.1.0",
     "capabilities": ["chat", "streaming"], "resumability": False},
]


@app.get("/api/v1/harnesses")
async def list_harnesses():
    return _HARNESSES


@app.get("/healthz")
async def healthz():
    return {"ok": True}
