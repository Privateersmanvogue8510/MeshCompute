"""Binary activation framing (PROTOCOL.md "Activation frame", "Data-plane packet classes").

Large tensor payloads must NOT go through JSON. This defines a compact, versioned
binary header for activation / KV / token packets. The header is fixed-layout; the
payload follows. Compression is optional and benchmark-driven (never assumed to help).

Header layout (big-endian), 40 bytes fixed + variable shape:
  u8   protocol_version
  u8   packet_class     (see PacketClass)
  u8   dtype            (see Dtype)
  u8   compression      (see Compression)
  u32  session_id_crc   (crc of session id string, for fast routing/validation)
  u64  request_id
  u32  step
  u16  microbatch
  u16  tensor_id
  u8   ndim
  u8   reserved
  u16  payload_crc16    (of payload for corruption detection)
  u32  payload_length
  u32[ndim] shape
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum

from .versioning import PROTOCOL_VERSION


class PacketClass(IntEnum):
    CONTROL = 0
    ACTIVATION = 1
    KV_OP = 2
    TOKEN_RESULT = 3
    MODEL_CHUNK = 4
    TELEMETRY = 5
    CANCEL = 6
    WORK_RECEIPT = 7


class Dtype(IntEnum):
    F32 = 0
    F16 = 1
    BF16 = 2
    I32 = 3
    U8 = 4
    RAW = 255


class Compression(IntEnum):
    NONE = 0
    ZLIB = 1


_FIXED = struct.Struct(">BBBB I Q I H H B B H I")  # 32 bytes before shape
MAX_NDIM = 8
MAX_PAYLOAD = 256 * 1024 * 1024  # 256 MiB hard cap (backpressure / DoS bound)


@dataclass
class Frame:
    packet_class: PacketClass
    request_id: int = 0
    session_id: str = ""
    step: int = 0
    microbatch: int = 0
    tensor_id: int = 0
    dtype: Dtype = Dtype.RAW
    shape: tuple[int, ...] = ()
    payload: bytes = b""
    compression: Compression = Compression.NONE
    protocol_version: int = PROTOCOL_VERSION
    _sid_crc: int = field(default=0, repr=False)

    def encode(self) -> bytes:
        payload = self.payload
        if self.compression == Compression.ZLIB:
            payload = zlib.compress(payload)
        if len(payload) > MAX_PAYLOAD:
            raise ValueError(f"payload {len(payload)} exceeds MAX_PAYLOAD {MAX_PAYLOAD}")
        if len(self.shape) > MAX_NDIM:
            raise ValueError(f"ndim {len(self.shape)} exceeds MAX_NDIM {MAX_NDIM}")
        sid_crc = zlib.crc32(self.session_id.encode("utf-8")) & 0xFFFFFFFF
        pcrc = zlib.crc32(payload) & 0xFFFF
        head = _FIXED.pack(
            self.protocol_version, int(self.packet_class), int(self.dtype),
            int(self.compression), sid_crc, self.request_id, self.step,
            self.microbatch, self.tensor_id, len(self.shape), 0, pcrc, len(payload))
        shape_bytes = struct.pack(f">{len(self.shape)}I", *self.shape)
        return head + shape_bytes + payload

    @classmethod
    def decode(cls, buf: bytes) -> "Frame":
        if len(buf) < _FIXED.size:
            raise ValueError("frame too short")
        (pv, pc, dt, comp, sid_crc, rid, step, mb, tid, ndim, _res,
         pcrc, plen) = _FIXED.unpack_from(buf, 0)
        if pv != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version {pv}")
        if ndim > MAX_NDIM:
            raise ValueError("ndim too large")
        if plen > MAX_PAYLOAD:
            raise ValueError("payload_length exceeds cap")
        off = _FIXED.size
        shape = struct.unpack_from(f">{ndim}I", buf, off)
        off += 4 * ndim
        payload = buf[off:off + plen]
        if len(payload) != plen:
            raise ValueError("truncated payload")
        if (zlib.crc32(payload) & 0xFFFF) != pcrc:
            raise ValueError("payload crc mismatch (corruption)")
        if comp == Compression.ZLIB:
            payload = zlib.decompress(payload)
        f = cls(packet_class=PacketClass(pc), request_id=rid, step=step, microbatch=mb,
                tensor_id=tid, dtype=Dtype(dt), shape=tuple(shape), payload=payload,
                compression=Compression.NONE, protocol_version=pv)
        f._sid_crc = sid_crc
        return f
