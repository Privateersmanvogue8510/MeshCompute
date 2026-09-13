"""MeshCompute gateway — the user-facing API and trusted orchestration layer
(INSTRUCTIONS §13). Turns a chat/session request into a scheduler plan and streams
tokens back from the selected worker's backend."""

from .app import create_app

__all__ = ["create_app"]
