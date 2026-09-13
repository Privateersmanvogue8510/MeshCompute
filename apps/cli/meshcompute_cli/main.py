"""mesh — thin CLI client over the MeshCompute gateway + control-plane HTTP APIs.

Talks OpenAI-compatible + native HTTP to services that may not be running (or,
during Phase 1 co-development, may not even be importable yet). Every network
call is wrapped so a down/missing service prints one clean line, never a
traceback. Gateway/daemon internals (meshcompute_gateway, meshcompute_node) are
imported lazily inside the commands that need them for the same reason.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

from .config import get_control_url, get_gateway_url, get_token, load_config, save_config

console = Console()
err_console = Console(stderr=True)

BENCH_PROMPT = (
    "Explain, step by step, why the sum of the first n positive integers equals "
    "n(n+1)/2, then write a short Python function that computes it."
)


def _auth_headers() -> dict[str, str]:
    tok = get_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _human_bytes(n: int) -> str:
    if n <= 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}PB"


def _die(msg: str) -> None:
    err_console.print(f"[red]{msg}[/red]")
    sys.exit(1)


@click.group()
def cli() -> None:
    """mesh — CLI for the MeshCompute peer-to-peer inference network."""


# --------------------------------------------------------------------------- login
@cli.command()
@click.option("--token", prompt=True, hide_input=True, help="API token to persist.")
def login(token: str) -> None:
    """Save an auth token to ~/.mesh/config.json (Phase 1: auth is a stub)."""
    cfg = load_config()
    cfg["token"] = token
    save_config(cfg)
    console.print(f"[green]saved token to {Path.home() / '.mesh' / 'config.json'}[/green]")


# -------------------------------------------------------------------------- models
@cli.command()
def models() -> None:
    """List models the gateway currently serves."""
    url = get_gateway_url()
    try:
        r = httpx.get(f"{url}/v1/models", headers=_auth_headers(), timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as e:
        _die(f"could not reach gateway at {url} ({e}); is `mesh api serve` running?")
        return
    body = r.json()
    items = body.get("data", []) if isinstance(body, dict) else body
    table = Table(title=f"models @ {url}")
    table.add_column("id")
    for m in items:
        table.add_row(m.get("id", str(m)) if isinstance(m, dict) else str(m))
    console.print(table)


# ---------------------------------------------------------------------------- chat
@cli.command()
@click.option("--model", required=True, help="Model id to chat with.")
@click.option("--harness", default="native", help="Execution harness (default: native).")
@click.option("--system", default=None, help="Optional system prompt.")
def chat(model: str, harness: str, system: str | None) -> None:
    """Interactive streaming chat REPL against /v1/chat/completions."""
    url = get_gateway_url()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    console.print(f"[dim]model={model} harness={harness} — Ctrl-C cancels a turn, Ctrl-D quits.[/dim]")

    while True:
        try:
            user_text = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not user_text.strip():
            continue
        messages.append({"role": "user", "content": user_text})

        payload = {"model": model, "harness": harness, "messages": messages, "stream": True}
        assistant_text = ""
        try:
            with httpx.stream("POST", f"{url}/v1/chat/completions", json=payload,
                               headers=_auth_headers(), timeout=None) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    err_console.print(f"[red]gateway error: {e}[/red]")
                    messages.pop()
                    continue

                strategy = resp.headers.get("X-Mesh-Strategy")
                path = resp.headers.get("X-Mesh-Path")
                if strategy or path:
                    console.print(f"[dim]strategy={strategy or '-'} path={path or '-'}[/dim]")

                console.print("[bold magenta]mesh>[/bold magenta] ", end="")
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
                        assistant_text += piece
                console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]cancelled[/yellow]")
            messages.pop()
            continue
        except httpx.HTTPError as e:
            err_console.print(f"[red]could not reach gateway at {url} ({e})[/red]")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": assistant_text})


# ---------------------------------------------------------------------------- bench
@cli.command()
@click.option("--model", required=True, help="Model id to benchmark.")
@click.option("--tokens", default=128, type=int, help="max_tokens to request.")
def bench(model: str, tokens: int) -> None:
    """Stream one prompt and report TTFT, time-to-completion, and decode tokens/sec."""
    url = get_gateway_url()
    payload = {
        "model": model, "stream": True, "max_tokens": tokens,
        "messages": [{"role": "user", "content": BENCH_PROMPT}],
    }
    t0 = time.monotonic()
    t_first = None
    n_tokens = 0
    n_chars = 0
    try:
        with httpx.stream("POST", f"{url}/v1/chat/completions", json=payload,
                           headers=_auth_headers(), timeout=None) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if piece:
                    if t_first is None:
                        t_first = time.monotonic()
                    n_tokens += 1  # one streamed content chunk counted as one token
                    n_chars += len(piece)
    except httpx.HTTPError as e:
        _die(f"could not reach gateway at {url} ({e}); is `mesh api serve` running?")
        return
    t_end = time.monotonic()

    if t_first is None:
        err_console.print("[yellow]no content was streamed back (empty response)[/yellow]")
        return

    ttft = t_first - t0
    total = t_end - t0
    decode_window = max(t_end - t_first, 1e-9)
    table = Table(title=f"bench: {model}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("time to first chunk (TTFT)", f"{ttft:.3f}s")
    table.add_row("time to completion", f"{total:.3f}s")
    table.add_row("streamed tokens", str(n_tokens))
    table.add_row("streamed chars", str(n_chars))
    table.add_row("decode tokens/sec", f"{n_tokens / decode_window:.2f}")
    console.print(table)
    console.print("[dim]note: reasoning models emit a <think> block first, so TTFT-to-first-"
                  "content can be high even when decode is fast.[/dim]")


# ---------------------------------------------------------------------------- node
@cli.group()
def node() -> None:
    """Node daemon controls."""


@node.command("status")
def node_status() -> None:
    """List known nodes from the control plane."""
    url = get_control_url()
    try:
        r = httpx.get(f"{url}/api/v1/nodes", headers=_auth_headers(), timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as e:
        _die(f"could not reach control plane at {url} ({e}); is the control plane running?")
        return
    table = Table(title=f"nodes @ {url}")
    for col in ("node_id", "online", "pool", "vram_free", "backends"):
        table.add_column(col)
    for n in r.json():
        cap = (n.get("capability") or {}).get("record") or {}
        gpus = cap.get("gpus") or []
        vram_free = sum(g.get("free_vram_bytes", 0) for g in gpus)
        table.add_row(
            n.get("node_id", "-"),
            str(n.get("online", "-")),
            ",".join(n.get("pool_ids") or []) or "-",
            _human_bytes(vram_free),
            ",".join(cap.get("backends") or []) or "-",
        )
    console.print(table)


@node.command("start")
@click.option("--backend", type=click.Choice(["lmstudio", "llamacpp"]), default="llamacpp")
@click.option("--backend-url", default=None, help="URL of the local inference backend.")
@click.option("--pool", "pool_id", default="public")
@click.option("--idle-only/--no-idle-only", default=True,
              help="Only contribute to the network while this machine is idle.")
@click.option("--idle-minutes", "idle_minutes_before_start", default=10, type=int,
              help="Minutes of no user input required before contributing.")
@click.option("--pause-on-activity/--no-pause-on-activity", "pause_on_user_activity", default=True,
              help="Pause contribution the moment user input is detected again.")
@click.option("--max-vram", "max_vram_percent", default=85, type=int, help="Max VRAM percent to contribute.")
@click.option("--max-cpu", "max_cpu_percent", default=50, type=int, help="Max CPU percent to contribute.")
@click.option("--max-ram", "max_ram_gb", default=None, type=int,
              help="Max RAM, in GB, to contribute (unset = no explicit cap).")
@click.option("--require-ac-power/--no-require-ac-power", default=False,
              help="Only contribute while on AC power (laptops).")
@click.option("--gpu-devices", default="auto",
              help='Which GPUs to donate: "auto" (all detected), or explicit indices '
                   'e.g. "0,1" or "0".')
@click.option("--split-mode", type=click.Choice(["layer", "row"]), default="layer",
              help="How to split a model across multiple selected GPUs.")
@click.option("--tensor-split", default=None,
              help='Explicit per-GPU split proportions, e.g. "3,1" (default: '
                   "proportional to each selected GPU's free VRAM).")
@click.option("--control-url", default=None)
def node_start(backend: str, backend_url: str | None, pool_id: str, idle_only: bool,
               idle_minutes_before_start: int, pause_on_user_activity: bool,
               max_vram_percent: int, max_cpu_percent: int, max_ram_gb: int | None,
               require_ac_power: bool, gpu_devices: str, split_mode: str,
               tensor_split: str | None, control_url: str | None) -> None:
    """Start the node daemon (contributes capacity to the mesh)."""
    try:
        from meshcompute_node import daemon as node_daemon
    except ImportError:
        console.print(
            "[yellow]node daemon not yet available in this build; run: "
            "mesh node start (again) once node/daemon/meshcompute_node/daemon.py lands.[/yellow]"
        )
        sys.exit(0)

    control = control_url or get_control_url()
    opts = dict(backend=backend, backend_url=backend_url, pool_id=pool_id,
                idle_only=idle_only, idle_minutes_before_start=idle_minutes_before_start,
                pause_on_user_activity=pause_on_user_activity,
                max_vram_percent=max_vram_percent, max_cpu_percent=max_cpu_percent,
                max_ram_gb=max_ram_gb, require_ac_power=require_ac_power,
                gpu_devices=gpu_devices, split_mode=split_mode, tensor_split=tensor_split,
                control_url=control)
    if hasattr(node_daemon, "run"):
        node_daemon.run(**opts)
    elif hasattr(node_daemon, "async_main"):
        import asyncio
        asyncio.run(node_daemon.async_main(**opts))
    else:
        _die("meshcompute_node.daemon has neither run() nor async_main(); cannot start.")


# ----------------------------------------------------------------------------- api
@cli.group()
def api() -> None:
    """Gateway server controls."""


@api.command("serve")
@click.option("--port", default=8081, type=int)
@click.option("--control-url", default=None)
def api_serve(port: int, control_url: str | None) -> None:
    """Launch the gateway (OpenAI-compatible + native HTTP surface)."""
    try:
        import uvicorn
        import meshcompute_gateway.app  # noqa: F401 — import-check before handing off to uvicorn
    except ImportError as e:
        console.print(
            f"[yellow]gateway not yet available in this build ({e}); try again once "
            "apps/gateway/meshcompute_gateway/app.py lands.[/yellow]"
        )
        sys.exit(0)

    import os
    os.environ["MESH_CONTROL_URL"] = control_url or get_control_url()
    console.print(f"[dim]starting gateway on 127.0.0.1:{port} (control={os.environ['MESH_CONTROL_URL']})[/dim]")
    uvicorn.run("meshcompute_gateway.app:app", host="127.0.0.1", port=port)


# ---------------------------------------------------------------------------- pool
@cli.group()
def pool() -> None:
    """Pool management."""


@pool.command("list")
def pool_list() -> None:
    """List pools known to the control plane."""
    url = get_control_url()
    try:
        r = httpx.get(f"{url}/api/v1/pools", headers=_auth_headers(), timeout=10)
    except httpx.HTTPError as e:
        _die(f"could not reach control plane at {url} ({e})")
        return
    if r.status_code == 404:
        console.print("[yellow]pools not available yet[/yellow]")
        return
    try:
        r.raise_for_status()
    except httpx.HTTPError as e:
        _die(f"control plane error: {e}")
        return
    table = Table(title=f"pools @ {url}")
    table.add_column("id")
    table.add_column("private")
    for p in r.json():
        if isinstance(p, dict):
            table.add_row(str(p.get("id", "-")), str(p.get("private", "-")))
        else:
            table.add_row(str(p), "-")
    console.print(table)


@pool.command("create")
@click.argument("name")
@click.option("--private", is_flag=True, default=False)
def pool_create(name: str, private: bool) -> None:
    """Create a pool."""
    url = get_control_url()
    try:
        r = httpx.post(f"{url}/api/v1/pools", json={"id": name, "private": private},
                        headers=_auth_headers(), timeout=10)
    except httpx.HTTPError as e:
        _die(f"could not reach control plane at {url} ({e})")
        return
    if r.status_code == 404:
        console.print("[yellow]pools not available yet[/yellow]")
        return
    try:
        r.raise_for_status()
    except httpx.HTTPError as e:
        _die(f"control plane error: {e}")
        return
    console.print(f"[green]pool created:[/green] {name}")


# ------------------------------------------------------------------------ manifest
@cli.group()
def manifest() -> None:
    """Sign and verify model manifests."""


@manifest.command("sign")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--key", default="deploy/identities/cli.key.json", help="Node identity key file (created if missing).")
@click.option("--out", "out_path", default=None, help="Output SignedManifest json path.")
def manifest_sign(path: str, key: str, out_path: str | None) -> None:
    """Sign a YAML ModelManifest, writing a SignedManifest json."""
    import yaml
    from meshcompute_protocol import ModelManifest, NodeIdentity, SignedManifest

    raw = yaml.safe_load(Path(path).read_text())
    mf = ModelManifest.model_validate(raw)
    manifest_hash = mf.manifest_hash()

    ident = NodeIdentity.load_or_create(key)
    signature_b64 = ident.sign(manifest_hash.encode("utf-8"))

    signed = SignedManifest(manifest=mf, manifest_hash=manifest_hash,
                             public_b64=ident.public_b64, signature_b64=signature_b64)

    out = Path(out_path) if out_path else Path(path).with_suffix(".signed.json")
    out.write_text(signed.model_dump_json(indent=2))
    console.print(f"[green]manifest_hash:[/green] {manifest_hash}")
    console.print(f"[dim]signed by {ident.node_id} -> {out}[/dim]")


@manifest.command("verify")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def manifest_verify(path: str) -> None:
    """Verify a SignedManifest's signature and that manifest_hash matches the content."""
    from meshcompute_protocol import SignedManifest
    from meshcompute_protocol import verify as verify_signature

    try:
        signed = SignedManifest.model_validate_json(Path(path).read_text())
    except Exception as e:  # malformed json/schema -> clean FAIL, not a traceback
        console.print(f"[red]FAIL[/red] could not parse SignedManifest: {e}")
        sys.exit(1)

    recomputed = signed.manifest.manifest_hash()
    hash_ok = recomputed == signed.manifest_hash
    sig_ok = verify_signature(signed.public_b64, signed.manifest_hash.encode("utf-8"), signed.signature_b64)

    if hash_ok and sig_ok:
        console.print(f"[green]OK[/green] manifest_hash={signed.manifest_hash}")
        return

    reasons = []
    if not hash_ok:
        reasons.append(f"manifest_hash mismatch (recomputed {recomputed})")
    if not sig_ok:
        reasons.append("signature invalid")
    console.print(f"[red]FAIL[/red] {'; '.join(reasons)}")
    sys.exit(1)


if __name__ == "__main__":
    cli()
