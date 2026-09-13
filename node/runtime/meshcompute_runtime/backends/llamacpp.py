"""LlamaCppBackend — the embedded, standalone inference engine.

INSTRUCTIONS §2.1: no third-party service (LM Studio/Ollama). llama.cpp is a
library/binary WE download and drive ourselves (runtime_installer.py), the
same way an app bundles ffmpeg. This module owns the child-process lifecycle;
it reuses OpenAICompatBackend (backends/lmstudio.py) to talk to the spawned
`llama-server` over HTTP, since llama-server already speaks OpenAI — no need
to reimplement chat/streaming parsing for a backend that happens to be local.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator

from .base import BackendAdapter, BackendCapabilities, ChatRequest, TokenChunk
from .lmstudio import OpenAICompatBackend

DEFAULT_LOG_DIR = Path.home() / ".mesh" / "logs"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppBackend(BackendAdapter):
    """Manages one local `llama-server` child process loading one GGUF model."""

    def __init__(self, server_bin: str, model_path: str, *, model_id: str = "local",
                 host: str = "127.0.0.1", port: int | None = None, ctx_size: int = 4096,
                 n_gpu_layers: int = 0, rpc_servers: list[str] | None = None,
                 mmproj_path: str | None = None, extra_args: list[str] | None = None,
                 log_dir: str | Path = DEFAULT_LOG_DIR) -> None:
        self.server_bin = server_bin
        self.model_path = model_path
        self.model_id = model_id
        self.host = host
        self.port = port or _free_port()
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.rpc_servers = rpc_servers or []
        self.mmproj_path = mmproj_path
        self.extra_args = extra_args or []
        self.log_path = Path(log_dir) / f"llama-server-{self.port}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self._openai = OpenAICompatBackend(
            f"http://{self.host}:{self.port}", backend_name="llamacpp",
            supports_tools=True, supports_vision=bool(mmproj_path), max_context=ctx_size)
        self._caps = BackendCapabilities(
            id=f"llamacpp@{self.host}:{self.port}", backends=["llamacpp"],
            supports_streaming=True, supports_tools=True, supports_vision=bool(mmproj_path),
            supports_pipeline_split=True,  # llama.cpp's rpc-server can split layers across peers
            max_context=ctx_size)

    def capabilities(self) -> BackendCapabilities:
        return self._caps

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _argv(self) -> list[str]:
        argv = [self.server_bin, "-m", self.model_path, "--host", self.host,
                "--port", str(self.port), "-c", str(self.ctx_size)]
        if self.n_gpu_layers:
            argv += ["-ngl", str(self.n_gpu_layers)]
        if self.rpc_servers:
            argv += ["--rpc", ",".join(self.rpc_servers)]
        if self.mmproj_path:
            argv += ["--mmproj", self.mmproj_path]
        argv += self.extra_args
        return argv

    async def start(self, *, health_timeout: float = 120.0) -> None:
        """Launch the child process and wait until it answers /v1/models."""
        self._log_file = open(self.log_path, "wb")
        self._proc = subprocess.Popen(self._argv(), stdout=self._log_file,
                                       stderr=subprocess.STDOUT)
        deadline = time.monotonic() + health_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                tail = self.log_path.read_text(errors="replace")[-4000:]
                self.stop()
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}); "
                    f"log tail:\n{tail}")
            if await self._openai.health():
                return
            await asyncio.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server did not become healthy within {health_timeout}s "
                            f"(see {self.log_path})")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    async def health(self) -> bool:
        return await self._openai.health()

    async def list_models(self) -> list[str]:
        models = await self._openai.list_models()
        return models or [self.model_id]

    def chat_stream(self, req: ChatRequest) -> AsyncIterator[TokenChunk]:
        return self._openai.chat_stream(req)

    @staticmethod
    def rpc_split_args(rpc_peers: list[str]) -> list[str]:
        """Multi-node pipeline-split launch helper (INSTRUCTIONS §4.1): pass
        ["host:port", ...] of running llama.cpp `rpc-server`/`ggml-rpc-server`
        peers and llama-server offloads layers to them via --rpc.
        TODO(phase-1.5): not exercised on this single-box proof; wire real peer
        rpc-server addresses from the scheduler's ExecutionPlan.peer_chain.
        """
        return ["--rpc", ",".join(rpc_peers)] if rpc_peers else []


if __name__ == "__main__":
    import asyncio as _asyncio

    from meshcompute_runtime.runtime_installer import SMOKE_MODEL, detect_accel, ensure_llama_server, ensure_model

    async def _demo() -> None:
        accel = detect_accel()
        server_bin = await ensure_llama_server(accel=accel)
        model_path = await ensure_model(SMOKE_MODEL)
        backend = LlamaCppBackend(server_bin, model_path, ctx_size=2048,
                                   n_gpu_layers=999 if accel == "cuda" else 0)
        print(f"starting llama-server on {backend.base_url} (log: {backend.log_path})")
        await backend.start()
        assert await backend.health()
        models = await backend.list_models()
        assert models, "expected at least one model id from /v1/models"

        req = ChatRequest(model=models[0],
                           messages=[{"role": "user", "content": "Say hello in exactly three words."}],
                           max_tokens=32, temperature=0.2, stream=True)
        text = "".join([c.text async for c in backend.chat_stream(req) if c.text])
        assert text.strip(), "expected a non-empty completion from the locally-loaded model"
        print(f"completion: {text!r}")

        bench = await backend.benchmark(models[0])
        print(f"benchmark: {bench}")
        backend.stop()
        print("backends/llamacpp.py self-check PASSED")

    _asyncio.run(_demo())
