"""Resolve a chat/session request to (ExecutionPlan, BackendAdapter).

Normal path: POST the control-plane's /api/v1/schedule (ScheduleRequest ->
ExecutionPlan), take plan.peer_chain[0] (the winning node — INSTRUCTIONS §8 topology
scheduler), and look up that node's OpenAI-compatible endpoint from
deploy/nodes.local.yaml (gitignored — SECURITY.md: no private IPs in committed code).

Dev/test fallback: the control-plane's /schedule endpoint doesn't exist yet in this
repo (Milestone 5). So when the control-plane is unreachable, and
MESH_GW_DIRECT_BACKEND_URL (or the sole node in nodes.local.yaml) is configured, we
synthesize a trivial single-node ExecutionPlan and route straight to that backend.
This lets the gateway be verified end-to-end before the control-plane exists. It is
clearly marked and never masks a real scheduler decision (a 500/infeasible response
from a control-plane that IS reachable is a hard error, not a fallback trigger).
"""

from __future__ import annotations

import uuid

import httpx
from meshcompute_protocol import ExecutionPlan, ScheduleRequest, Strategy
from meshcompute_runtime.backends import BackendAdapter, OpenAICompatBackend

from .settings import NodeEndpoint, get_settings, load_node_endpoints


class SchedulerUnavailable(Exception):
    """No feasible plan / no way to reach a backend. app.py maps this to HTTP 503."""


# Backend adapters are cheap (just an httpx base_url + name) but we cache by URL so
# concurrent requests to the same worker reuse one adapter instance.
_backend_cache: dict[str, BackendAdapter] = {}


def _backend_for(url: str, backend_name: str) -> BackendAdapter:
    key = url.rstrip("/")
    backend = _backend_cache.get(key)
    if backend is None:
        backend = OpenAICompatBackend(base_url=key, backend_name=backend_name)
        _backend_cache[key] = backend
    return backend


async def _schedule_via_control_plane(model_id: str, pool_id: str, client_node_id: str,
                                       context_length: int, prompt_tokens: int) -> ExecutionPlan | None:
    """Returns None if the control-plane can't be reached at all (fallback candidate).
    Raises SchedulerUnavailable for a control-plane that responded but rejected the
    request (infeasible plan, bad request, etc.) — that's a real answer, not a gap."""
    settings = get_settings()
    req = ScheduleRequest(model_id=model_id, pool_id=pool_id, client_node_id=client_node_id,
                          context_length=context_length, prompt_tokens=prompt_tokens)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{settings.mesh_gw_control_url}/api/v1/schedule",
                                     json=req.model_dump(mode="json"))
    except httpx.TransportError:
        return None  # control-plane unreachable -> caller may try the dev fallback
    if resp.status_code == 500:
        raise SchedulerUnavailable(f"scheduler reports an infeasible plan: {resp.text}")
    resp.raise_for_status()
    return ExecutionPlan.model_validate(resp.json())


def _direct_fallback() -> tuple[NodeEndpoint, str] | None:
    settings = get_settings()
    if settings.mesh_gw_direct_backend_url:
        return NodeEndpoint(node_id="dev-direct", backend_url=settings.mesh_gw_direct_backend_url), \
            "MESH_GW_DIRECT_BACKEND_URL"
    nodes = load_node_endpoints()
    if len(nodes) == 1:
        ep = next(iter(nodes.values()))
        return ep, "the sole node in deploy/nodes.local.yaml"
    return None


async def resolve_backend(model_id: str, client_node_id: str, ctx: int,
                          prompt_tokens: int, pool_id: str = "public",
                          ) -> tuple[ExecutionPlan, BackendAdapter]:
    plan = await _schedule_via_control_plane(model_id, pool_id, client_node_id, ctx, prompt_tokens)

    if plan is None:
        fallback = _direct_fallback()
        if fallback is None:
            raise SchedulerUnavailable(
                "control-plane unreachable and no dev fallback configured "
                "(set MESH_GW_DIRECT_BACKEND_URL, or leave exactly one node in "
                "deploy/nodes.local.yaml)")
        ep, source = fallback
        plan = ExecutionPlan(
            plan_id=f"dev-{uuid.uuid4().hex[:8]}",
            model_manifest_hash="dev-unsigned",
            model_id=model_id,
            strategy=Strategy.SINGLE,
            peer_chain=[ep.node_id],
            context_length=ctx,
            decision_trace=[f"DEV FALLBACK (no control-plane reachable): routed directly "
                            f"to {ep.node_id} via {source}."],
        )
        return plan, _backend_for(ep.backend_url, ep.backend)

    node_id = plan.peer_chain[0]
    ep = load_node_endpoints().get(node_id)
    if ep is None:
        raise SchedulerUnavailable(
            f"scheduler chose node {node_id!r} but it has no backend_url in "
            f"deploy/nodes.local.yaml")
    return plan, _backend_for(ep.backend_url, ep.backend)
