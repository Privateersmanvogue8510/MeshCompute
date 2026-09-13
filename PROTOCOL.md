# MeshCompute Protocol Design

## Protocol layers

MeshCompute separates:

1. **Control protocol**
   - registration,
   - capability updates,
   - model/pool metadata,
   - scheduling,
   - credits,
   - reputation.

2. **Peer transport**
   - encrypted QUIC,
   - discovery,
   - NAT traversal,
   - relay fallback,
   - multiplexed streams.

3. **Inference protocol**
   - session establishment,
   - tensor metadata,
   - activation transfer,
   - KV ownership,
   - cancellation,
   - result streaming.

4. **Model distribution protocol**
   - signed manifests,
   - content-addressed chunks,
   - availability,
   - chunk requests,
   - verification.

Do not merge these into one opaque socket protocol.

## Peer identity

Each node generates an Ed25519 keypair.

`node_id` derives from the public key.

Private keys:
- remain on the node,
- use OS secure storage where available,
- are never sent to the control plane.

Use signed capability announcements and signed work receipts.

## Rendezvous

A node can learn peers from:
- control-plane rendezvous,
- LAN discovery,
- DHT,
- cached known peers.

The public service may issue short-lived peer connection tickets to prevent arbitrary unauthorized sessions.

## NAT traversal

Attempt in order:
1. direct known-address QUIC,
2. hole-punched direct QUIC,
3. trusted relay.

Record the resulting path type because it changes scheduling cost.

## Session handshake

Conceptual fields:

```text
InferenceSessionOpen
  protocol_version
  plan_id
  session_id
  model_manifest_hash
  strategy
  shard_assignment
  context_parameters
  peer_chain
  expiration
  signed_ticket
```

Each peer validates:
- signature,
- expiry,
- model hash,
- assigned role,
- available resources.

## Activation frame

Conceptual binary header:

```text
ActivationFrame
  protocol_version
  session_id
  request_id
  step
  microbatch
  tensor_id
  dtype
  shape
  compression
  payload_length
  checksum
```

Payload follows the header.

Compression is optional and benchmark-driven. Never assume compression helps every activation shape.

## Backpressure

Every data stream must expose:
- bounded queues,
- receiver window,
- cancellation,
- timeout,
- max frame size.

A slow node must not cause unbounded memory growth on upstream peers.

## KV-cache ownership

In pipeline mode, each shard owner keeps the KV for its layers.

KV state is keyed by:
- session,
- sequence,
- layer range,
- cache generation.

Do not move full KV state between peers on every token.

## Session affinity

Keep a generation on the same execution path whenever possible.

Path changes are expensive because KV state is distributed.

## Model chunk protocol

A manifest describes immutable artifacts.

Conceptual chunk identifier:

```text
chunk_id = BLAKE3(model_manifest_hash || file_path || offset || chunk_bytes)
```

Peers advertise chunk sets compactly.

Support:
- range/chunk requests,
- parallel sources,
- resume,
- hash verify before commit,
- peer scoring,
- origin fallback.

## Work receipts

Every completed assignment produces a signed receipt containing:

```text
WorkReceipt
  receipt_id
  session_id
  plan_id
  node_id
  model_manifest_hash
  role
  measured_work
  started_at
  finished_at
  outcome
  challenge_nonce
  signature
```

The accounting service validates receipts before crediting contribution.

## Versioning

Every public protocol object has a version.

Backward-compatible fields may be added.

Breaking changes require a new protocol version and explicit compatibility behavior.

Never infer a protocol version from application version alone.
