# Design References

Research snapshot: 2026-09-14.

These references inform the architecture. They are not dependencies unless the implementation explicitly adopts them.

## Distributed inference

### exo

Current exo documentation describes:
- peer-to-peer device clustering,
- topology-aware model partitioning,
- ChatGPT-compatible API,
- support for multiple partitioning strategies,
- low-latency interconnect optimization.

Repository:
https://github.com/exo-explore/exo

### Petals

Petals is a direct precedent for "BitTorrent-style" distributed LLM inference and private/public swarms.

Repository:
https://github.com/bigscience-workshop/petals

Important caution:
Petals is architecturally informative, but its release history is old enough that MeshCompute should not assume it is the best modern runtime base.

## Model target

Qwen3-Coder official repository:
https://github.com/QwenLM/Qwen3-Coder

The official docs list Qwen3-Coder-30B-A3B-Instruct and note that function calling depends on the appropriate tool parser in SGLang/vLLM.

## MCP

Current specification baseline:
2026-07-28

Specification/blog:
https://blog.modelcontextprotocol.io/posts/2026-07-28/

The 2026-07-28 release moved the protocol core to a stateless request model and changed assumptions from earlier MCP versions.

## ACP

Agent Client Protocol:
https://zed.dev/acp

Use ACP as an editor-facing interoperability target so agent/harness integration is not locked to one IDE.

## Mobile

Android NNAPI migration guidance:
https://developer.android.com/ndk/guides/neuralnetworks/migration-guide

NNAPI is deprecated. Current Android guidance points developers toward maintained on-device runtimes and GPU paths.

Apple Background Tasks:
https://developer.apple.com/documentation/backgroundtasks

Apple supports continued processing and, on supported configurations, background GPU/inference resources. Treat availability as OS-controlled rather than guaranteed always-on compute.
