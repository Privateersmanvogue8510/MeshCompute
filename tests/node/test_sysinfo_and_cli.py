"""Host signals (cross-platform module, run on whatever this host is) and the
CLI surfaces that must exist for the documented install/run flow."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from meshcompute_node import sysinfo

REPO = Path(__file__).resolve().parents[2]


def test_sysinfo_never_raises_and_is_sane():
    total, free = sysinfo.total_ram_bytes(), sysinfo.free_ram_bytes()
    assert total > 0 and 0 <= free <= total
    assert sysinfo.disk_free_bytes(".") > 0
    assert sysinfo.disk_free_bytes("/definitely/not/here") == 0
    assert 0.0 <= sysinfo.cpu_load_percent() <= 100.0
    assert sysinfo.user_idle_seconds() is None or sysinfo.user_idle_seconds() >= 0
    assert sysinfo.on_ac_power() in (True, False, None)


def test_node_start_exposes_documented_flags():
    from meshcompute_cli.main import cli

    out = CliRunner().invoke(cli, ["node", "start", "--help"]).output
    for flag in ("--model", "--quant", "--port", "--quic-port", "--accel", "--max-storage",
                 "--update-poll", "--gpu-devices", "--idle-only", "--max-vram", "--max-cpu", "--max-ram"):
        assert flag in out, flag


def test_manifest_pin_rewrites_hashes_from_local_files(tmp_path):
    import blake3
    import yaml
    from meshcompute_cli.main import cli

    src = REPO / "models/manifests/public-smollm2-360m.yaml"
    text = src.read_text().replace(
        '"6cb0ba7f2d5e00ad4445c162c9e7e3f2d2e620f10656b86b1df1f6b1797e509d"', '"PENDING-x"')
    text = text.replace("size_bytes: 270590880", "size_bytes: 0")
    yml = tmp_path / "m.yaml"
    yml.write_text(text)
    cache = tmp_path / "cache" / "repo" / "v1"
    cache.mkdir(parents=True)
    f = cache / "SmolLM2-360M-Instruct-Q4_K_M.gguf"
    f.write_bytes(b"pretend-gguf-bytes")

    r = CliRunner().invoke(cli, ["manifest", "pin", str(yml), "--cache", str(tmp_path / "cache")])
    assert r.exit_code == 0, r.output
    data = yaml.safe_load(yml.read_text())
    shard = data["quantizations"][0]["files"][0]
    assert shard["size_bytes"] == len(b"pretend-gguf-bytes")
    assert shard["artifact_root_hash"] == blake3.blake3(b"pretend-gguf-bytes").hexdigest()
    assert data["id"] == "public/smollm2-360m"   # everything else untouched


def test_install_scripts_exist_for_every_desktop_os():
    assert (REPO / "scripts/install.sh").is_file()
    assert (REPO / "scripts/install.ps1").is_file()
    assert "3.12" in (REPO / "scripts/install.ps1").read_text()
