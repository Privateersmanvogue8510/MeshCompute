"""Decentralized ledger (blockchain-style) tests — no single owner, verifiable.

Covers: chain growth + balances, tamper detection (merkle), cross-block double-
credit prevention (nonce reuse), forged receipt rejection, inflated-credit
rejection, and longest-valid-chain fork choice. Deterministic (fixed timestamps).
"""

from __future__ import annotations

import copy

import pytest

from meshcompute_protocol import (
    NodeIdentity, Chain, Block, LedgerEntry, ChainError,
    WorkReceipt, SignedReceipt, MeasuredWork, Outcome, credit_for_receipt, pick_best_chain,
)

TS = 1789330000.0


def _sr(node: NodeIdentity, rid: str, nonce: str, tokens: int) -> SignedReceipt:
    r = WorkReceipt(receipt_id=rid, session_id="s", plan_id="p", node_id=node.node_id,
                    model_manifest_hash="mh", role="single",
                    measured_work=MeasuredWork(tokens_generated=tokens),
                    started_at=TS, finished_at=TS + 1, outcome=Outcome.OK, challenge_nonce=nonce)
    return SignedReceipt(receipt=r, public_b64=node.public_b64,
                         signature_b64=node.sign_json(r.model_dump(mode="json")))


def _entry(sr: SignedReceipt) -> LedgerEntry:
    return LedgerEntry(entry_id=sr.receipt.receipt_id, account=sr.receipt.node_id,
                       delta=credit_for_receipt(sr), reason="verified_work",
                       ref_receipt_id=sr.receipt.receipt_id,
                       challenge_nonce=sr.receipt.challenge_nonce, receipt=sr)


@pytest.fixture
def producer():
    return NodeIdentity.generate()


@pytest.fixture
def worker():
    return NodeIdentity.generate()


def test_genesis_and_credit(producer, worker):
    c = Chain.new(identity=producer, timestamp=TS)
    assert c.height == 1
    c.append(Block.produce(identity=producer, index=1, prev_hash=c.head_hash,
                           timestamp=TS + 10, entries=[_entry(_sr(worker, "r1", "n1", 100))]))
    assert c.balances()[worker.node_id] == 100.0
    c.verify_full()  # independently re-verifiable from genesis


def test_tamper_detected(producer, worker):
    c = Chain.new(identity=producer, timestamp=TS)
    c.append(Block.produce(identity=producer, index=1, prev_hash=c.head_hash,
                           timestamp=TS + 10, entries=[_entry(_sr(worker, "r1", "n1", 100))]))
    bad = copy.deepcopy(c)
    bad.blocks[1].entries[0].delta = 999999.0
    with pytest.raises(ChainError):
        bad.verify_full()


def test_double_credit_rejected(producer, worker):
    c = Chain.new(identity=producer, timestamp=TS)
    c.append(Block.produce(identity=producer, index=1, prev_hash=c.head_hash,
                           timestamp=TS + 10, entries=[_entry(_sr(worker, "r1", "n1", 100))]))
    dup = Block.produce(identity=producer, index=2, prev_hash=c.head_hash, timestamp=TS + 20,
                        entries=[_entry(_sr(worker, "r2", "n1", 50))])  # same nonce n1
    with pytest.raises(ChainError):
        c.append(dup)


def test_forged_receipt_rejected(producer, worker):
    c = Chain.new(identity=producer, timestamp=TS)
    sr = _sr(worker, "r1", "n1", 10)
    sr.signature_b64 = sr.signature_b64[:-4] + "AAAA"
    blk = Block.produce(identity=producer, index=1, prev_hash=c.head_hash, timestamp=TS + 10,
                        entries=[_entry(sr)])
    with pytest.raises(ChainError):
        c.append(blk)


def test_inflated_delta_rejected(producer, worker):
    c = Chain.new(identity=producer, timestamp=TS)
    e = _entry(_sr(worker, "r1", "n1", 10))
    e.delta = 10000.0
    with pytest.raises(ChainError):
        c.append(Block.produce(identity=producer, index=1, prev_hash=c.head_hash,
                               timestamp=TS + 10, entries=[e]))


def test_fork_choice_longest_valid(producer, worker):
    short = Chain.new(identity=producer, timestamp=TS)
    short.append(Block.produce(identity=producer, index=1, prev_hash=short.head_hash,
                              timestamp=TS + 10, entries=[_entry(_sr(worker, "r1", "n1", 5))]))
    longer = Chain.new(identity=producer, timestamp=TS)
    longer.append(Block.produce(identity=producer, index=1, prev_hash=longer.head_hash,
                               timestamp=TS + 10, entries=[_entry(_sr(worker, "r5", "n5", 5))]))
    longer.append(Block.produce(identity=producer, index=2, prev_hash=longer.head_hash,
                               timestamp=TS + 20, entries=[_entry(_sr(worker, "r6", "n6", 5))]))
    assert pick_best_chain([short, longer]) is longer


def test_positive_correction_rejected(producer, worker):
    """Only receipt-backed verified work may MINT credit. A "correction" that adds
    credit needs an authorization scheme Phase 1 does not have, so it is invalid."""
    c = Chain.new(identity=producer, timestamp=TS)
    bad = Block.produce(identity=producer, index=1, prev_hash=c.head_hash, timestamp=TS + 10,
                        entries=[LedgerEntry(entry_id="fake", account=worker.node_id,
                                             delta=1e9, reason="correction")])
    with pytest.raises(ChainError, match="verified_work"):
        c.append(bad)

    # and it must not slip past a full re-verification either
    c.blocks.append(bad)
    with pytest.raises(ChainError, match="verified_work"):
        c.verify_full()


def test_negative_consumption_accepted(producer, worker):
    c = Chain.new(identity=producer, timestamp=TS)
    c.append(Block.produce(identity=producer, index=1, prev_hash=c.head_hash, timestamp=TS + 10,
                           entries=[_entry(_sr(worker, "r1", "n1", 100))]))
    c.append(Block.produce(identity=producer, index=2, prev_hash=c.head_hash, timestamp=TS + 20,
                           entries=[LedgerEntry(entry_id="spend1", account=worker.node_id,
                                                delta=-40.0, reason="consumption")]))
    assert c.balances()[worker.node_id] == 60.0
    c.verify_full()
