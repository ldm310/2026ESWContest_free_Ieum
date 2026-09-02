
from __future__ import annotations

import struct
from collections.abc import Iterator


AUDIO_UDP_ADDR = ("127.0.0.1", 50007)
DIRECTION_UDP_ADDR = ("127.0.0.1", 50008)


AUDIO_HEADER = struct.Struct("<III")
DIRECTION_PACKET = struct.Struct("<3d")
MAX_AUDIO_PAYLOAD_BYTES = 5_600


def iter_audio_packets(
    sequence_id: int,
    audio_bytes: bytes,
    max_payload_bytes: int = MAX_AUDIO_PAYLOAD_BYTES,
) -> Iterator[bytes]:

    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes는 0보다 커야 합니다.")

    total = len(audio_bytes)
    for offset in range(0, total, max_payload_bytes):
        payload = audio_bytes[offset : offset + max_payload_bytes]
        yield AUDIO_HEADER.pack(sequence_id, total, offset) + payload


def unpack_audio_packet(packet: bytes) -> tuple[int, int, int, bytes]:

    if len(packet) < AUDIO_HEADER.size:
        raise ValueError("오디오 UDP 패킷이 헤더보다 짧습니다.")
    sequence_id, total, offset = AUDIO_HEADER.unpack_from(packet)
    return sequence_id, total, offset, packet[AUDIO_HEADER.size :]


def pack_direction(direction: tuple[float, float, float]) -> bytes:

    return DIRECTION_PACKET.pack(*direction)


def unpack_direction(packet: bytes) -> tuple[float, float, float]:

    if len(packet) != DIRECTION_PACKET.size:
        raise ValueError("방향 UDP 패킷 길이가 올바르지 않습니다.")
    return DIRECTION_PACKET.unpack(packet)
