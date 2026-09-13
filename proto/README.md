# proto/ — reference protocol IDL

These `.proto` files are the **canonical, versioned schema** for the MeshCompute
wire protocol, intended for generating cross-language SDKs (Python is the Phase-1
implementation; a TypeScript SDK is Phase-2 per INSTRUCTIONS §6).

Phase-1 note: the Python services currently exchange the control objects as JSON
(FastAPI/pydantic — see `packages/protocol/meshcompute_protocol/`) and the
peer data plane uses the compact binary framing in `frames.py`. These `.proto`
definitions mirror those types field-for-field so a future gRPC/protobuf transport
or a non-Python SDK stays wire-compatible. Every message carries `protocol_version`
(INSTRUCTIONS §26 rule 8). Keep these in lockstep with the pydantic models.
