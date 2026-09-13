# MeshCompute Security and Privacy

## Threat model

Assume:
- public workers can be malicious,
- public clients can be malicious,
- peers can falsify capabilities,
- peers can replay receipts,
- model artifacts can be poisoned,
- web pages can contain prompt injection,
- users may accidentally send secrets into a public swarm.

## Primary boundaries

### Inference worker boundary

Public workers must never receive:
- platform database credentials,
- user API keys,
- SSH keys,
- Git credentials,
- browser session cookies,
- MCP secrets,
- unrestricted filesystem access.

### Model boundary

Public models use signed manifests and non-executable weight formats.

Do not enable arbitrary upstream model Python code in the public pool.

Pin:
- model revision,
- tokenizer,
- tool parser,
- inference backend version where behavior is sensitive.

### Tool boundary

Tools execute in:
- the user's trusted local agent process, or
- a platform-managed sandbox.

Never execute tool calls directly on random GPU peers.

## Public-swarm privacy

Transport encryption is necessary but not sufficient.

A worker involved in inference may observe:
- activations,
- model-specific intermediate state,
- and potentially prompt-derived information.

Therefore:
- do not claim end-to-end private inference on the public swarm,
- avoid prompt logging,
- recommend private pools for sensitive code/data,
- provide clear UI disclosure before first public-swarm use.

## Tool prompt injection

Web and repository content is untrusted.

The agent gateway must:
- label external content as data,
- keep system/tool policy separate,
- require approval for dangerous actions,
- prevent retrieved content from silently changing tool permissions,
- scope credentials per tool,
- log tool decisions.

## Worker sandbox

Run inference runtime under least privilege.

Prefer:
- container or sandbox boundary,
- read-only model files,
- bounded scratch space,
- no host filesystem mounts by default,
- restricted network egress,
- resource limits,
- no privileged container,
- no Docker socket.

## Update security

Node daemon updates should be signed.

Model catalog updates should be signed independently.

A compromised model registry key must not automatically grant software-update authority.

## Credit abuse

Protect against:
- fake uptime,
- fake GPU specs,
- replayed work,
- self-dealing loops,
- artificial chunk traffic,
- colluding peers.

Credit only work accepted by a real execution plan or verified storage/distribution policy.

## Denial of service

Apply:
- per-user rate limits,
- per-peer connection limits,
- bounded frame sizes,
- bounded queues,
- admission control,
- challenge cost for suspicious registration bursts,
- execution ticket expiration.

## Private pools

Private pools can raise the trust level but do not remove security requirements.

Support:
- explicit peer allowlists,
- organization ownership,
- revocation,
- separate model catalog,
- separate accounting policy,
- optional direct-only networking.

## Security milestones

POC:
- encrypted P2P,
- signed identities,
- signed model manifests,
- no arbitrary model code,
- basic sandbox,
- basic tool separation.

Beta:
- reputation,
- anti-replay,
- benchmark challenges,
- sandbox hardening,
- prompt injection controls,
- audit logs.

Later:
- confidential-compute experiments,
- hardware attestation,
- privacy-preserving inference research.
