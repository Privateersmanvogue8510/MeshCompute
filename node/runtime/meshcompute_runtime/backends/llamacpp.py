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
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator

from .base import BackendAdapter, BackendCapabilities, ChatRequest, TokenChunk
from .lmstudio import OpenAICompatBackend

DEFAULT_LOG_DIR = Path(os.environ.get("MESH_HOME") or (Path.home() / ".mesh")) / "logs"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gpu_launch_args(selected_gpus: list[dict], *, split_mode: str = "layer",
                     tensor_split: str | None = None,
                     main_gpu: int = 0, accel: str = "cuda") -> tuple[dict[str, str], list[str]]:
    """Pure helper (no subprocess, no I/O — unit-testable with zero GPUs
    present): given the GPUs a contributor selected to donate, return
    (env, args) to launch llama-server across exactly those cards.

    `selected_gpus` is a list of {"index": int, "free_vram_bytes": int}
    (see contribution.py's per_gpu_free_vram_bytes), in the order they should
    appear in CUDA_VISIBLE_DEVICES.

      - CPU / no GPUs selected: ({}, []) — no CUDA env, no -ngl. Unchanged.
      - One GPU: CUDA_VISIBLE_DEVICES="<index>", -ngl 999. No split flags —
        nothing to split across a single card. Unchanged single-GPU behavior.
      - 2+ GPUs (the NVLINK-pod case — docs/PHYSICS.md #4, INSTRUCTIONS §4.2):
        CUDA_VISIBLE_DEVICES="<i0>,<i1>,...", -ngl 999, --split-mode,
        --main-gpu, and a --tensor-split proportional to each selected GPU's
        free VRAM (so heterogeneous cards get a fair share) unless the caller
        pins an explicit `tensor_split` string.

    `main_gpu` and the --tensor-split order both refer to CUDA_VISIBLE_DEVICES's
    remapped 0..N-1 view (the order of `selected_gpus`), not physical indices —
    that's how CUDA_VISIBLE_DEVICES remapping works regardless of which
    physical cards were chosen.
    TODO(phase-1.5): real multi-GPU execution and NVLINK peer-to-peer transfer
    behavior are untestable on this CPU-only box; this only proves the
    argument-building logic, not runtime behavior on real hardware.
    """
    if accel == "metal":
        # Apple Silicon: unified memory, one device — offload everything.
        return {}, ["-ngl", "999"]
    if not selected_gpus:
        return {}, []

    indices = [str(g["index"]) for g in selected_gpus]
    env = {"CUDA_VISIBLE_DEVICES": ",".join(indices)}
    if accel == "vulkan":
        # Linux+NVIDIA without a CUDA build runs the Vulkan backend; its device
        # filter is GGML_VK_VISIBLE_DEVICES (indices in Vulkan's enumeration
        # order, which matches nvidia-smi order on single-vendor boxes).
        env["GGML_VK_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    args = ["-ngl", "999"]

    if len(selected_gpus) > 1:
        if tensor_split is None:
            free_mib = [max(1, g.get("free_vram_bytes", 0) // (1 << 20)) for g in selected_gpus]
            tensor_split = ",".join(str(m) for m in free_mib)
        args += ["--split-mode", split_mode, "--main-gpu", str(main_gpu), "--tensor-split", tensor_split]
    return env, args


class LlamaCppBackend(BackendAdapter):
    """Manages one local `llama-server` child process loading one GGUF model."""

    def __init__(self, server_bin: str, model_path: str, *, model_id: str = "local",
                 host: str = "127.0.0.1", port: int | None = None, ctx_size: int = 4096,
                 n_gpu_layers: int = 0, rpc_servers: list[str] | None = None,
                 mmproj_path: str | None = None, extra_args: list[str] | None = None,
                 extra_env: dict[str, str] | None = None,
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
        self.extra_env = extra_env or {}  # e.g. CUDA_VISIBLE_DEVICES — see gpu_launch_args()
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

    async def start(self, *, health_timeout: float = 600.0) -> None:
        """Launch the child process and wait until it answers /v1/models.
        Default timeout is generous: a 24 GB GGUF can take minutes to page in
        from a cold disk."""
        self._log_file = open(self.log_path, "wb")
        env = {**os.environ, **self.extra_env} if self.extra_env else None
        self._proc = subprocess.Popen(self._argv(), stdout=self._log_file,
                                       stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + health_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                tail = self.log_path.read_text(errors="replace")[-4000:]
                self.stop()
                raise RuntimeError(f"llama-server exited early (code {code}); log tail:\n{tail}")
            if await self._openai.health():
                return
            await asyncio.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server did not become healthy within {health_timeout}s "
                            f"(see {self.log_path})")

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

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

    from meshcompute_runtime.runtime_installer import (
        SMOKE_MODEL, detect_accel, ensure_llama_server, ensure_model,
    )

    # --- gpu_launch_args: pure arg-builder proof, no GPU required ---
    cpu_env, cpu_args = gpu_launch_args([])
    assert cpu_env == {} and cpu_args == [], "CPU/no-GPU path must add no CUDA env and no -ngl"

    one_gpu = [{"index": 0, "free_vram_bytes": 24 * (1 << 30)}]
    env1, args1 = gpu_launch_args(one_gpu)
    assert env1 == {"CUDA_VISIBLE_DEVICES": "0"}
    assert args1 == ["-ngl", "999"], "single GPU: no split-mode/tensor-split — nothing to split"

    # Simulated 2x RTX 3090 NVLINK pod (docs/PHYSICS.md #4).
    two_gpus = [{"index": 0, "free_vram_bytes": 24 * (1 << 30)},
                {"index": 1, "free_vram_bytes": 23 * (1 << 30)}]
    env2, args2 = gpu_launch_args(two_gpus)
    assert env2 == {"CUDA_VISIBLE_DEVICES": "0,1"}
    assert "-ngl" in args2 and "999" in args2
    assert args2[args2.index("--split-mode") + 1] == "layer"
    assert "--tensor-split" in args2 and "24576,23552" in args2  # proportional to free VRAM (MiB)
    print(f"2-GPU NVLINK simulated launch: env={env2} args={args2}")

    env_gpu1_only, args_gpu1_only = gpu_launch_args([{"index": 1, "free_vram_bytes": 23 * (1 << 30)}])
    assert env_gpu1_only == {"CUDA_VISIBLE_DEVICES": "1"}
    assert args_gpu1_only == ["-ngl", "999"]

    env_row, args_row = gpu_launch_args(two_gpus, split_mode="row", tensor_split="1,1")
    assert args_row[args_row.index("--split-mode") + 1] == "row"
    assert args_row[args_row.index("--tensor-split") + 1] == "1,1"

    assert gpu_launch_args([], accel="metal") == ({}, ["-ngl", "999"]), "Metal must offload"
    env_vk, _ = gpu_launch_args(one_gpu, accel="vulkan")
    assert env_vk["GGML_VK_VISIBLE_DEVICES"] == "0" and env_vk["CUDA_VISIBLE_DEVICES"] == "0"

    print(f"single-GPU launch: env={env1} args={args1}")
    print(f"gpu_devices=[1] launch: env={env_gpu1_only} args={args_gpu1_only}")
    print(f"CPU launch: env={cpu_env} args={cpu_args}")
    print("gpu_launch_args self-check PASSED")

    async def _demo() -> None:
        engine = await ensure_llama_server(accel=detect_accel())
        model_path, _target = await ensure_model(SMOKE_MODEL)
        _env, gargs = gpu_launch_args([], accel=engine.accel)
        backend = LlamaCppBackend(engine.path, model_path, ctx_size=2048, extra_args=gargs)
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
