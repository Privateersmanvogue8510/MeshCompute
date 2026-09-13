"""Protocol versioning.

Every public protocol object carries an explicit version (INSTRUCTIONS §26 rule 8,
PROTOCOL.md "Versioning"). Never infer a protocol version from application version.
Backward-compatible fields may be added within a version; breaking changes bump it.
"""

from __future__ import annotations

# Wire protocol version for control/inference/peer objects.
PROTOCOL_VERSION = 1

# Minimum MeshCompute runtime a manifest may require (semver string).
MIN_RUNTIME_VERSION = "0.1.0"


def is_compatible(peer_version: int) -> bool:
    """A peer speaking an equal or lower major protocol version is compatible.

    Phase 1 has a single version, so this is exact-match. It exists so callers
    route through one check now and the negotiation policy has one place to grow.
    """
    return peer_version == PROTOCOL_VERSION
