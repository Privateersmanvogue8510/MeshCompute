# Node Configuration

## Design goals

A contributor must be able to install the worker, set limits, and understand what the node is doing.

No mysterious resource grabbing.

## Example configuration

```yaml
node:
  display_name: studio-gpu-01
  auto_start: true

pool:
  public: true
  private_pool_ids: []

compute:
  gpu_devices: auto
  max_gpu_utilization_percent: 90
  max_vram_percent: 85
  max_cpu_percent: 50
  max_ram_gb: 32

storage:
  cache_path: ./mesh-cache
  max_cache_gb: 200
  seed_model_chunks: true

network:
  max_upload_mbps: 500
  max_download_mbps: 500
  allow_relay: true
  allow_lan_discovery: true

availability:
  idle_only: true
  idle_minutes_before_start: 10
  pause_on_user_activity: true
  require_ac_power: false

thermal:
  enabled: true
  gpu_temperature_limit_c: 82

privacy:
  public_inference: true
  private_only: false
```

## Required controls

Contributor UI must include:
- start/stop immediately,
- public/private toggle,
- per-GPU toggle,
- max VRAM,
- max utilization,
- max storage,
- bandwidth cap,
- idle-only mode,
- schedule,
- temperature cutoff,
- contribution history,
- earned credits,
- current model shards,
- current sessions and resource usage.

Stopping contribution must stop accepting new work immediately and drain or cancel current work according to the user's selected policy.
