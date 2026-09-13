"""Decentralized credit ledger — blockchain-style, no single owner (INSTRUCTIONS §11).

The original spec said "no blockchain in the initial design"; the project owner has
since explicitly asked for a decentralized, blockchain-style ledger. This module is
that layer, built to be maximally decentralized WITHOUT becoming a speculative coin:

  - Blocks of verified work-receipts, each block linked to the previous by a BLAKE3
    hash (a chain), signed by its producer (Ed25519).
  - Every block is INDEPENDENTLY VERIFIABLE by anyone: the producer signature, the
    Merkle root of entries, the prev-hash link, and — crucially — each entry's
    underlying signed WorkReceipt all verify without trusting any server.
  - Double-credit / replay is prevented across the whole chain by tracking each
    receipt's challenge_nonce (a nonce may be credited at most once, ever).
  - NO proof-of-work, NO mining, NO token. Ordering is "longest valid chain, ties
    broken by lowest block hash" — adequate without adversarial miners, and the
    block structure lets a BFT ordering layer replace fork-choice later without
    touching the data model. Credits are NON-transferable internal compute units.

Only the ACCOUNTING layer is on-chain. Activations/tokens/model-chunks stay on the
P2P data plane (consensus per token would destroy latency). Rendezvous and
scheduling stay off-chain (latency-sensitive). The control-plane becomes one
replica/producer among many, not the authority.

Timestamps are passed in by callers (never generated in-module) so verification is
deterministic and reproducible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .content import blake3_hex
from .identity import NodeIdentity, canonical_json, node_id_from_public, verify_json, verify
from .receipts import SignedReceipt, Outcome
from .versioning import PROTOCOL_VERSION

GENESIS_PREV = "0" * 64


class ChainError(Exception):
    """A block or chain failed validation. The message says exactly why."""


class LedgerEntry(BaseModel):
    """One credit event, derived from a verified work-receipt (earn) or a spend."""
    entry_id: str                 # unique; for earns == the receipt_id
    account: str                  # node_id being credited (earn) or debited (spend)
    delta: float                  # >0 earn, <0 spend/consumption
    reason: str                   # verified_work | consumption | correction
    ref_receipt_id: str = ""
    challenge_nonce: str = ""     # anti double-credit; unique across the whole chain
    # the signed receipt travels with the entry so any validator re-verifies it
    receipt: SignedReceipt | None = None

    def entry_hash(self) -> str:
        # hash excludes the (large, re-derivable) receipt payload
        return blake3_hex(canonical_json({
            "entry_id": self.entry_id, "account": self.account, "delta": self.delta,
            "reason": self.reason, "ref_receipt_id": self.ref_receipt_id,
            "challenge_nonce": self.challenge_nonce}))


def merkle_root(entries: list[LedgerEntry]) -> str:
    """BLAKE3 Merkle root over entry hashes. Empty -> hash of empty. Odd level
    duplicates the last node (standard). Order is the block's entry order."""
    if not entries:
        return blake3_hex(b"")
    level = [bytes.fromhex(e.entry_hash()) for e in entries]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_h(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0].hex()


def _h(b: bytes) -> bytes:
    import blake3
    return blake3.blake3(b).digest()


# --- credit formula (auditable, calibratable — INSTRUCTIONS §11) -------------
def credit_for_receipt(sr: SignedReceipt) -> float:
    """Measured, not claimed. Phase-1 rule: credit = tokens generated, else GPU-
    active seconds. Multipliers (demand/reliability/scarcity) are Phase-2 and slot
    in here. Failed/cancelled work earns nothing."""
    r = sr.receipt
    if r.outcome not in (Outcome.OK, Outcome.PARTIAL):
        return 0.0
    w = r.measured_work
    if w.tokens_generated:
        return float(w.tokens_generated)
    return round(w.gpu_active_ms / 1000.0, 3)


class BlockHeader(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    index: int
    prev_hash: str
    timestamp: float               # epoch seconds, stamped by producer (passed in)
    merkle_root: str
    producer_node_id: str
    entry_count: int


class Block(BaseModel):
    header: BlockHeader
    entries: list[LedgerEntry] = Field(default_factory=list)
    producer_public_b64: str = ""
    producer_signature_b64: str = ""

    def block_hash(self) -> str:
        return blake3_hex(canonical_json(self.header.model_dump(mode="json")))

    @classmethod
    def produce(cls, *, identity: NodeIdentity, index: int, prev_hash: str,
                timestamp: float, entries: list[LedgerEntry]) -> "Block":
        header = BlockHeader(
            index=index, prev_hash=prev_hash, timestamp=timestamp,
            merkle_root=merkle_root(entries), producer_node_id=identity.node_id,
            entry_count=len(entries))
        sig = identity.sign_json(header.model_dump(mode="json"))
        return cls(header=header, entries=entries,
                   producer_public_b64=identity.public_b64, producer_signature_b64=sig)


def verify_block(block: Block, *, expected_prev_hash: str, expected_index: int,
                 seen_nonces: set[str], require_receipts: bool = True) -> None:
    """Raise ChainError if the block is invalid. Mutates seen_nonces with this
    block's nonces on success (so the caller detects cross-block double-credit)."""
    h = block.header
    if h.protocol_version != PROTOCOL_VERSION:
        raise ChainError(f"block {h.index}: protocol_version {h.protocol_version}")
    if h.index != expected_index:
        raise ChainError(f"block index {h.index} != expected {expected_index}")
    if h.prev_hash != expected_prev_hash:
        raise ChainError(f"block {h.index}: prev_hash mismatch (fork/tamper)")
    if h.entry_count != len(block.entries):
        raise ChainError(f"block {h.index}: entry_count {h.entry_count} != {len(block.entries)}")
    if merkle_root(block.entries) != h.merkle_root:
        raise ChainError(f"block {h.index}: merkle_root mismatch (entries tampered)")
    # producer signature + identity binding
    if node_id_from_public(_pub(block.producer_public_b64)) != h.producer_node_id:
        raise ChainError(f"block {h.index}: producer key does not derive producer_node_id")
    if not verify_json(block.producer_public_b64, h.model_dump(mode="json"),
                       block.producer_signature_b64):
        raise ChainError(f"block {h.index}: bad producer signature")
    # per-entry validation
    local_nonces: set[str] = set()
    for e in block.entries:
        if e.reason == "verified_work":
            if require_receipts:
                if e.receipt is None:
                    raise ChainError(f"entry {e.entry_id}: verified_work without receipt")
                sr = e.receipt
                # receipt signature + ownership
                if node_id_from_public(_pub(sr.public_b64)) != sr.receipt.node_id:
                    raise ChainError(f"entry {e.entry_id}: receipt key/node_id mismatch")
                if not verify_json(sr.public_b64, sr.receipt.model_dump(mode="json"),
                                   sr.signature_b64):
                    raise ChainError(f"entry {e.entry_id}: bad receipt signature")
                if sr.receipt.node_id != e.account:
                    raise ChainError(f"entry {e.entry_id}: credited account != receipt node")
                if sr.receipt.receipt_id != e.ref_receipt_id:
                    raise ChainError(f"entry {e.entry_id}: ref_receipt_id mismatch")
                expected = credit_for_receipt(sr)
                if abs(expected - e.delta) > 1e-6:
                    raise ChainError(
                        f"entry {e.entry_id}: delta {e.delta} != credit_for_receipt {expected}")
            # double-credit prevention across the whole chain AND within the block
            nonce = e.challenge_nonce
            if not nonce:
                raise ChainError(f"entry {e.entry_id}: verified_work missing challenge_nonce")
            if nonce in seen_nonces or nonce in local_nonces:
                raise ChainError(f"entry {e.entry_id}: nonce reused (double-credit)")
            local_nonces.add(nonce)
        elif e.reason in ("consumption", "correction"):
            pass  # spends/corrections: Phase-2 rules (balance checks, authorization)
        else:
            raise ChainError(f"entry {e.entry_id}: unknown reason {e.reason!r}")
    seen_nonces |= local_nonces


def _pub(public_b64: str):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from .identity import _unb64
    return Ed25519PublicKey.from_public_bytes(_unb64(public_b64))


class Chain:
    """An append-only, fully-verifiable chain of ledger blocks, replicated per-node.
    No single instance is authoritative — peers gossip blocks and adopt the best
    valid chain (see pick_best_chain)."""

    def __init__(self, blocks: list[Block] | None = None) -> None:
        self.blocks: list[Block] = []
        self._seen_nonces: set[str] = set()
        if blocks:
            for b in blocks:
                self.append(b)

    @classmethod
    def new(cls, *, identity: NodeIdentity, timestamp: float) -> "Chain":
        c = cls()
        genesis = Block.produce(identity=identity, index=0, prev_hash=GENESIS_PREV,
                                timestamp=timestamp, entries=[])
        c.append(genesis)
        return c

    @property
    def head_hash(self) -> str:
        return self.blocks[-1].block_hash() if self.blocks else GENESIS_PREV

    @property
    def height(self) -> int:
        return len(self.blocks)

    def append(self, block: Block) -> None:
        verify_block(block, expected_prev_hash=self.head_hash,
                     expected_index=len(self.blocks), seen_nonces=self._seen_nonces)
        self.blocks.append(block)

    def verify_full(self) -> None:
        """Re-verify the entire chain from genesis (what a joining peer does)."""
        seen: set[str] = set()
        prev = GENESIS_PREV
        for i, b in enumerate(self.blocks):
            verify_block(b, expected_prev_hash=prev, expected_index=i, seen_nonces=seen)
            prev = b.block_hash()

    def balances(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for b in self.blocks:
            for e in b.entries:
                out[e.account] = round(out.get(e.account, 0.0) + e.delta, 6)
        return out

    def used_nonces(self) -> set[str]:
        return set(self._seen_nonces)


def pick_best_chain(chains: list[Chain]) -> Chain:
    """Fork choice: the longest VALID chain; ties broken by lowest head hash
    (deterministic). Invalid chains are discarded. A BFT/weighted-work ordering
    can replace this function later without changing the block format."""
    valid: list[Chain] = []
    for c in chains:
        try:
            c.verify_full()
            valid.append(c)
        except ChainError:
            continue
    if not valid:
        raise ChainError("no valid chain among candidates")
    return sorted(valid, key=lambda c: (-c.height, c.head_hash))[0]
