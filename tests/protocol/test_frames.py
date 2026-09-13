"""Frame encode/decode invariants (PROTOCOL.md, INSTRUCTIONS.md §23 protocol tests)."""

from __future__ import annotations

import zlib

import pytest
from meshcompute_protocol import Compression, Dtype, Frame, MAX_NDIM, MAX_PAYLOAD, PacketClass
from meshcompute_protocol import frames as frames_mod


def _frame(**overrides) -> Frame:
    defaults = dict(
        packet_class=PacketClass.ACTIVATION,
        request_id=42,
        session_id="sess-1",
        step=3,
        microbatch=1,
        tensor_id=7,
        dtype=Dtype.F16,
        shape=(2, 3, 4),
        payload=b"\x01\x02\x03\x04" * 10,
    )
    defaults.update(overrides)
    return Frame(**defaults)


def test_roundtrip_preserves_payload_shape_and_ids():
    original = _frame()
    decoded = Frame.decode(original.encode())

    assert decoded.payload == original.payload
    assert decoded.shape == original.shape
    assert decoded.packet_class == original.packet_class
    assert decoded.dtype == original.dtype
    assert decoded.request_id == original.request_id
    assert decoded.tensor_id == original.tensor_id
    assert decoded.step == original.step
    assert decoded.microbatch == original.microbatch
    # session_id itself isn't on the wire (only its crc, for fast routing) —
    # confirm the crc matches rather than expecting the string to survive.
    assert decoded._sid_crc == zlib.crc32(original.session_id.encode("utf-8")) & 0xFFFFFFFF


def test_zlib_compression_roundtrip():
    payload = b"A" * 10_000  # highly compressible
    original = _frame(payload=payload, compression=Compression.ZLIB)
    encoded = original.encode()

    assert len(encoded) < len(payload)  # compression actually happened on the wire
    decoded = Frame.decode(encoded)
    assert decoded.payload == payload
    assert decoded.compression == Compression.NONE  # decode yields the plain bytes


def test_corrupt_payload_raises_value_error():
    original = _frame(shape=())  # empty shape -> payload is the tail of the buffer
    buf = bytearray(original.encode())
    buf[-1] ^= 0xFF  # flip the last payload byte
    with pytest.raises(ValueError, match="crc"):
        Frame.decode(bytes(buf))


def test_payload_over_max_raises_value_error(monkeypatch):
    monkeypatch.setattr(frames_mod, "MAX_PAYLOAD", 16)
    f = _frame(payload=b"x" * 17, compression=Compression.NONE)
    with pytest.raises(ValueError, match="MAX_PAYLOAD"):
        f.encode()


def test_ndim_over_max_raises_value_error():
    f = _frame(shape=tuple(range(1, MAX_NDIM + 2)), payload=b"x")
    with pytest.raises(ValueError, match="MAX_NDIM"):
        f.encode()


def test_wrong_protocol_version_raises_value_error():
    encoded = bytearray(_frame().encode())
    encoded[0] = 99  # protocol_version is the first header byte
    with pytest.raises(ValueError, match="protocol_version"):
        Frame.decode(bytes(encoded))


def test_zlib_bomb_over_max_decompressed_raises_value_error(monkeypatch):
    """A few hundred compressed bytes can expand past the cap. The cap must be
    enforced on the DECOMPRESSED size too, not just what arrived on the wire."""
    encoded = _frame(payload=b"A" * 100_000, compression=Compression.ZLIB).encode()
    monkeypatch.setattr(frames_mod, "MAX_PAYLOAD", 4096)
    assert len(encoded) < 4096  # small on the wire...

    with pytest.raises(ValueError, match="decompressed payload exceeds cap"):
        Frame.decode(encoded)  # ...but 100kB once inflated


def test_max_payload_constant_is_64_mib():
    assert MAX_PAYLOAD == 64 * 1024 * 1024
