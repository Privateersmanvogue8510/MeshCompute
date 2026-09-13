# Public Model Catalog Policy

## Why the catalog is small

Tool calling, tokenization, chat templates, quantization, and runtime behavior vary by model.

The public network should optimize a small set of models extremely well rather than support hundreds badly.

## Initial target

Use one version-pinned Qwen3-Coder 30B-class model as the first engineering target if current license, runtime, and tool parser tests pass.

Candidate baseline:
- Qwen3-Coder-30B-A3B-Instruct
- exact upstream revision pinned
- platform-tested quantization(s)
- platform-tested SGLang/vLLM parser
- configured context smaller than theoretical maximum for early POC if required by KV/cache limits

The public alias might be:

```text
public/qwen3-coder-30b
```

The alias resolves to an immutable manifest hash.

## Later public slots

Do not add new models until the current slot has:
- benchmark coverage,
- tool-call conformance,
- cancellation tests,
- shard distribution tests,
- quantization quality tests,
- context tests,
- license review.

Suggested conceptual slots:
- `public/fast`
- `public/general`
- `public/code`
- `public/reasoning`

The actual models behind these aliases can evolve through explicit versioning.

## Manifest requirements

```yaml
id: public/qwen3-coder-30b
version: 1
upstream:
  provider: Qwen
  repository: exact-upstream-reference
  revision: exact-commit-or-content-hash
license:
  id: exact-license
runtime:
  min_meshcompute: 0.1.0
  backends:
    - sglang
    - vllm
tokenizer:
  revision: exact
chat_template:
  revision: exact
tool_parser:
  id: exact-parser
quantizations:
  - id: example
    artifact_root_hash: example
parallelism:
  pipeline: true
  tensor: conditional
  expert: conditional
  speculative_draft: false
```

This is illustrative, not final schema.

## Private catalog

Private pools may use custom models.

Private model owners accept responsibility for:
- license,
- model code,
- safety,
- parser behavior,
- resource use.

Even private manifests should be content-addressed and signed.
