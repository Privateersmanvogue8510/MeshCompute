"""Backend adapter contract (INSTRUCTIONS §26 rule 14, §6).

A backend runs assigned model work. The rest of MeshCompute talks to this stable
interface and does not care whether it is llama.cpp, an OpenAI-compatible endpoint,
vLLM or SGLang. New backends ADVERTISE capabilities (rule 9) rather than being
switched on by device name.

Backends never receive tool credentials or user secrets (SECURITY.md tool boundary);
they receive only model data and the inference request.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    stream: bool = True
    stop: list[str] | None = None
    tools: list[dict] | None = None
    extra: dict = field(default_factory=dict)   # backend-specific passthrough


@dataclass
class TokenChunk:
    text: str = ""
    finish_reason: str | None = None
    tool_call_delta: dict | None = None


@dataclass
class BackendCapabilities:
    id: str
    backends: list[str]              # e.g. ["lmstudio"] or ["llamacpp"]
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_pipeline_split: bool = False   # can this backend split layers across peers?
    max_context: int = 0


class BackendAdapter(abc.ABC):
    """One inference backend on one worker."""

    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abc.abstractmethod
    async def health(self) -> bool:
        """Is the backend reachable and a model loaded?"""

    @abc.abstractmethod
    async def list_models(self) -> list[str]: ...

    @abc.abstractmethod
    def chat_stream(self, req: ChatRequest) -> AsyncIterator[TokenChunk]:
        """Stream tokens. MUST honour cancellation (async generator close) and a
        timeout — every network path has cancellation + timeout (rule 12)."""
        ...

    async def benchmark(self, model: str) -> dict:
        """Measured decode/prefill tok/s. Default: run a tiny generation and time it.
        Never trust self-reported numbers elsewhere; this is the measured source."""
        import time
        req = ChatRequest(model=model,
                          messages=[{"role": "user", "content": "Count: 1 2 3 4 5 6 7 8"}],
                          max_tokens=32, temperature=0.7, stream=True)
        t0 = time.monotonic()
        ttft = None
        n = 0
        async for chunk in self.chat_stream(req):
            if chunk.text:
                if ttft is None:
                    ttft = time.monotonic() - t0
                n += 1
        dt = time.monotonic() - t0
        decode = (n / (dt - ttft)) if (ttft and dt > ttft and n) else 0.0
        return {"decode_tokens_per_sec": round(decode, 2),
                "ttft_s": round(ttft or 0.0, 3), "sampled_tokens": n}
