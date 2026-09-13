"""SQLite persistence for the control plane (stdlib sqlite3 — no ORM, sqlalchemy
is not installed). One connection, WAL mode, migrations are CREATE TABLE IF NOT
EXISTS at open (single schema version so far).

Timestamps are never generated in here — callers (app.py) pass `time.time()` at
the call site, matching the protocol package's own rule that pure objects don't
stamp their own clock.

The ledger is append-only by construction: there is no update/delete method,
only append_ledger(). Corrections are new rows with their own reason, never edits
(INSTRUCTIONS §11).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: one connection + one lock. Good enough for a single-process
        # POC control plane; move to a pool if concurrent write throughput
        # ever matters.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    public_b64 TEXT NOT NULL,
                    pool_ids TEXT NOT NULL,
                    last_seen REAL NOT NULL DEFAULT 0,
                    online INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    node_id TEXT PRIMARY KEY,
                    signed_capability TEXT NOT NULL,
                    updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS manifests (
                    alias TEXT PRIMARY KEY,
                    manifest_hash TEXT NOT NULL,
                    signed_manifest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    receipt_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    json TEXT NOT NULL,
                    UNIQUE (node_id, nonce)
                );
                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    delta REAL NOT NULL,
                    reason TEXT NOT NULL,
                    ref_receipt TEXT NOT NULL,
                    ts REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    plan TEXT NOT NULL,
                    created REAL NOT NULL
                );
                """
            )

    # --- nodes ---------------------------------------------------------------
    def upsert_node(self, node_id: str, public_b64: str, pool_ids: list[str],
                     last_seen: float, online: bool = True) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO nodes (node_id, public_b64, pool_ids, last_seen, online) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(node_id) DO UPDATE SET "
                "public_b64=excluded.public_b64, pool_ids=excluded.pool_ids, "
                "last_seen=excluded.last_seen, online=excluded.online",
                (node_id, public_b64, json.dumps(pool_ids), last_seen, int(online)),
            )

    def touch_node(self, node_id: str, last_seen: float, online: bool = True) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE nodes SET last_seen=?, online=? WHERE node_id=?",
                (last_seen, int(online), node_id),
            )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
        return _node_row(row) if row else None

    def list_nodes(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [_node_row(r) for r in rows]

    # --- capabilities ----------------------------------------------------------
    def put_capability(self, node_id: str, signed_capability: dict, updated: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO capabilities (node_id, signed_capability, updated) "
                "VALUES (?, ?, ?) ON CONFLICT(node_id) DO UPDATE SET "
                "signed_capability=excluded.signed_capability, updated=excluded.updated",
                (node_id, json.dumps(signed_capability), updated),
            )

    def get_capability(self, node_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT signed_capability FROM capabilities WHERE node_id=?", (node_id,)
            ).fetchone()
        return json.loads(row["signed_capability"]) if row else None

    # --- manifests ---------------------------------------------------------------
    def put_manifest(self, alias: str, manifest_hash: str, signed_manifest: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO manifests (alias, manifest_hash, signed_manifest) "
                "VALUES (?, ?, ?) ON CONFLICT(alias) DO UPDATE SET "
                "manifest_hash=excluded.manifest_hash, signed_manifest=excluded.signed_manifest",
                (alias, manifest_hash, json.dumps(signed_manifest)),
            )

    def get_manifest(self, alias: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT signed_manifest FROM manifests WHERE alias=?", (alias,)
            ).fetchone()
        return json.loads(row["signed_manifest"]) if row else None

    def get_manifest_by_hash(self, manifest_hash: str) -> dict | None:
        """ScheduleRequest.model_id may be an alias or a manifest_hash — support both."""
        with self._lock:
            row = self._conn.execute(
                "SELECT signed_manifest FROM manifests WHERE manifest_hash=?", (manifest_hash,)
            ).fetchone()
        return json.loads(row["signed_manifest"]) if row else None

    def list_manifests(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT alias, manifest_hash, signed_manifest FROM manifests"
            ).fetchall()
        return [
            {
                "alias": r["alias"],
                "manifest_hash": r["manifest_hash"],
                "signed_manifest": json.loads(r["signed_manifest"]),
            }
            for r in rows
        ]

    # --- receipts (anti-replay ledger source) -------------------------------
    def put_receipt(self, receipt_id: str, node_id: str, plan_id: str, nonce: str,
                     data: dict) -> bool:
        """Insert a receipt. Returns False if receipt_id OR (node_id, nonce) was
        already used — both are UNIQUE constraints, so this is the single
        mechanism that both dedupes and rejects challenge_nonce replay."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO receipts (receipt_id, node_id, plan_id, nonce, json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (receipt_id, node_id, plan_id, nonce, json.dumps(data)),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    # --- ledger (append-only) -------------------------------------------------
    def append_ledger(self, node_id: str, delta: float, reason: str, ref_receipt: str,
                       ts: float) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO ledger (node_id, delta, reason, ref_receipt, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (node_id, delta, reason, ref_receipt, ts),
            )
        return cur.lastrowid

    def balance(self, node_id: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(delta), 0) AS bal FROM ledger WHERE node_id=?", (node_id,)
            ).fetchone()
        return row["bal"]

    # --- sessions (stored execution plans) --------------------------------------
    def put_session(self, session_id: str, plan: dict, created: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (session_id, plan, created) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET plan=excluded.plan, created=excluded.created",
                (session_id, json.dumps(plan), created),
            )

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT plan FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return json.loads(row["plan"]) if row else None


def _node_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "node_id": row["node_id"],
        "public_b64": row["public_b64"],
        "pool_ids": json.loads(row["pool_ids"]),
        "last_seen": row["last_seen"],
        "online": bool(row["online"]),
    }


def _demo() -> None:
    """ponytail self-check: run `python store.py` to sanity-test CRUD + the
    ledger/receipt replay guard against a throwaway temp DB."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(f"{d}/t.db")
        s.upsert_node("nd_a", "pub_a", ["public"], 100.0)
        assert s.get_node("nd_a")["pool_ids"] == ["public"]
        s.touch_node("nd_a", 200.0)
        assert s.get_node("nd_a")["last_seen"] == 200.0

        s.put_capability("nd_a", {"record": {"node_id": "nd_a"}}, 1.0)
        assert s.get_capability("nd_a")["record"]["node_id"] == "nd_a"

        s.put_manifest("public/x", "hash1", {"manifest_hash": "hash1"})
        assert s.get_manifest("public/x")["manifest_hash"] == "hash1"
        assert s.get_manifest_by_hash("hash1")["manifest_hash"] == "hash1"
        assert len(s.list_manifests()) == 1

        assert s.put_receipt("r1", "nd_a", "plan1", "nonce1", {"ok": True}) is True
        # replay: same node_id + nonce must be rejected even with a new receipt_id
        assert s.put_receipt("r2", "nd_a", "plan1", "nonce1", {"ok": True}) is False
        # duplicate receipt_id must also be rejected
        assert s.put_receipt("r1", "nd_a", "plan1", "nonce2", {"ok": True}) is False

        s.append_ledger("nd_a", 1.5, "work_receipt", "r1", 10.0)
        s.append_ledger("nd_a", 0.5, "work_receipt", "r1-correction", 11.0)
        assert s.balance("nd_a") == 2.0

        s.put_session("sess1", {"plan_id": "plan1"}, 5.0)
        assert s.get_session("sess1")["plan_id"] == "plan1"
    print("store.py self-check OK")


if __name__ == "__main__":
    _demo()
