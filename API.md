# MeshCompute API and Agent Surface

## API goals

A user should be able to consume MeshCompute without using the first-party UI.

Expose:
- OpenAI-compatible endpoints,
- native MeshCompute endpoints,
- MCP,
- ACP-facing agent integration,
- SDKs.

## OpenAI-compatible surface

Initial target:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
```

Support streaming.

Model IDs should be stable platform aliases pointing to signed manifest revisions.

Example:

```text
public/qwen3-coder-30b
private/my-pool/my-model
```

Expose actual revision/hash in response metadata.

## Native sessions

Example conceptual API:

```text
POST /api/v1/sessions
GET  /api/v1/sessions/{id}
POST /api/v1/sessions/{id}/messages
POST /api/v1/sessions/{id}/cancel
POST /api/v1/sessions/{id}/harness
GET  /api/v1/sessions/{id}/events
```

Create session body:

```json
{
  "model_id": "public/qwen3-coder-30b",
  "harness_id": "native",
  "pool_id": "public",
  "workspace_id": "optional",
  "tools": ["web.search", "web.fetch"],
  "memory_mode": "session"
}
```

## Harness registry

```text
GET /api/v1/harnesses
```

Return:
- ID,
- display name,
- version,
- capabilities,
- supported tools,
- resumability,
- state-migration support,
- editor integration support.

A user can set a default harness, but a session may override it.

## Node API

```text
POST /api/v1/nodes/register
POST /api/v1/nodes/{id}/heartbeat
POST /api/v1/nodes/{id}/capabilities
GET  /api/v1/nodes/{id}/assignments
POST /api/v1/work-receipts
```

Prefer outbound node connections so contributors do not need manual inbound firewall rules.

## Pools

```text
GET  /api/v1/pools
POST /api/v1/pools
POST /api/v1/pools/{id}/members
GET  /api/v1/pools/{id}/models
```

Public pool behavior is platform-controlled.

Private pool behavior is owner-controlled.

## Credits

```text
GET /api/v1/credits/balance
GET /api/v1/credits/ledger
GET /api/v1/contributions
```

Ledger entries are immutable append-only records with correction entries rather than destructive edits.

## Tool API

Tools are represented through the agent gateway and MCP where possible.

Required first-party logical tools:
- `web.search`
- `web.fetch`
- `browser.open`
- `browser.click`
- `browser.type`
- `files.read`
- `files.write`
- `shell.exec`
- `git.status`
- `git.diff`
- `git.commit`

High-impact tools require user approval policy.

## Web results and citations

Search results should carry:
- title,
- canonical URL,
- snippet,
- source/provider,
- retrieved timestamp.

Fetched pages should preserve source provenance so the harness can cite them.

## ACP

Provide an ACP-compatible agent endpoint or adapter so editors can connect to the same session/harness layer.

Do not make the VS Code experience depend on a private MeshCompute-only editor protocol.
