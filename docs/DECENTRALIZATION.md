# Decentralization & the credit ledger

MeshCompute aims to be **decentralized as much as is practical**. This document says
what that means concretely, what is on-chain vs off-chain, and why.

## What must not depend on one owner

The weakest centralization point is the **credit/accounting ledger** — who earned and
spent compute credit, and node reputation. If one operator's database is the sole
authority, that operator can rewrite balances, censor contributors, or simply
disappear. That is the thing to decentralize first.

## The decentralized ledger (blockchain-style, no coin)

`packages/protocol/meshcompute_protocol/ledger.py` implements a chain of signed
blocks:

- **Blocks** batch verified work-receipts into ledger entries and link to the
  previous block by a **BLAKE3 hash** (a chain). Each block is signed (Ed25519) by
  its producer, and carries a **Merkle root** of its entries.
- **Independently verifiable.** Anyone can validate the whole chain from genesis with
  no trusted server: prev-hash links, Merkle roots, producer signatures, and — the
  important part — **each entry's underlying signed WorkReceipt** are all re-checked.
  A credit is only valid if the work that earned it has a valid signed receipt.
- **Double-credit proof.** Every receipt's `challenge_nonce` may be credited at most
  once across the entire chain. Reusing a nonce, forging a receipt signature, or
  inflating a credit beyond what the receipt's measured work warrants are all
  rejected (see `tests/protocol/test_ledger.py`).
- **Replicated peer-to-peer.** Nodes gossip blocks over the existing QUIC transport;
  a joining peer verifies from genesis and adopts the best chain. The control-plane
  becomes **one replica/producer among many**, not the authority — its SQLite table
  is a cached view of the replicated chain.
- **Fork choice**: longest valid chain, ties broken by lowest block hash
  (deterministic). This is adequate without adversarial miners and is a **single
  function** (`pick_best_chain`) that a BFT/weighted-work ordering layer can replace
  later without changing the block format.

### Deliberately NOT included

- **No proof-of-work / mining.** Validity (a valid signed receipt behind every
  credit), not wasted computation, gates acceptance. This is "proof of verified
  work," not PoW.
- **No token / no transferable cryptocurrency.** Credits are internal,
  non-transferable compute units. The original spec (INSTRUCTIONS §Product-principle-4,
  rule 6) warned against a speculative coin; a decentralized *ledger* delivers the
  decentralization without becoming one. A tradeable token would be a separate,
  explicit product decision — it is not needed for the compute network to work.

## What stays OFF-chain, and why

- **The inference data plane** (activations, tokens, KV, model chunks) — on the P2P
  QUIC / BitTorrent-style layer. Consensus per token would destroy latency; there is
  nothing to gain from ordering activations globally.
- **Rendezvous and scheduling** — latency-sensitive and ephemeral. Peers still
  discover each other and form pods directly; the chain only records the *accounting*
  outcome (signed receipts) after the fact.

## Reconciling with the original spec

INSTRUCTIONS.md said "do not introduce blockchain … in the initial design" and "do
not add blockchain unless explicitly requested." The project owner has now explicitly
requested a decentralized, blockchain-style ledger. This module honors that request
while keeping the spec's real intent intact: **no speculative coin, no on-chain
inference, decentralization where it removes a trusted single owner.**

## Roadmap

- **Now (Phase 1):** verifiable block/chain data model + validation + fork choice +
  balances (built, tested). Control-plane still writes the convenience SQLite ledger
  in parallel.
- **Next:** wire block production from verified receipts and gossip replication over
  the QUIC transport (a `LEDGER_SYNC` packet class), make the control-plane's ledger a
  derived view of the chain, expose `mesh credits` from the chain.
- **Later (if adversarial multi-operator federation demands it):** swap fork-choice
  for a BFT ordering layer (e.g. CometBFT-style validators); the block format and
  verification stay.
