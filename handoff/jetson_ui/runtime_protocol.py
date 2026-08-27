"""실시간 DOA 프로세스와 자막 UI 프로세스 사이의 UDP 규격."""

from __future__ import annotations

import struct
from collections.abc import Iterator


AUDIO_UDP_ADDR = ("127.0.0.1", 50007)
DIRECTION_UDP_ADDR = ("127.0.0.1", 50008)
SENTENCE_END_UDP_ADDR = ("127.0.0.1", 50009)

# DOA·Beamforming 결과를 0.25초마다 STT에 전달한다. StreamingSTT의
# partial 요청 간격도 0.25초이므로 upstream에서 더 큰 블록을 만들어 화면
# 갱신을 늦추지 않도록 두 프로세스가 이 값을 공유한다.
AUDIO_CHUNK_SECONDS = 0.25

# sequence_id, 전체 바이트 길이, 현재 조각의 시작 offset
AUDIO_HEADER = struct.Struct("<III")
DIRECTION_PACKET = struct.Struct("<3d")
MAX_AUDIO_PAYLOAD_BYTES = 5_600

# 문장 종료 신호는 오디오 packet과 섞지 않고 별도 UDP port로 전달한다.
# 고정 payload를 검증해 우연히 들어온 다른 datagram이 final을 만들지 않게 한다.
SENTENCE_END_PACKET = b"STT_SENTENCE_END_V1"


def iter_audio_packets(
    sequence_id: int,
    audio_bytes: bytes,
    max_payload_bytes: int = MAX_AUDIO_PAYLOAD_BYTES,
) -> Iterator[bytes]:
    """한 mono 오디오 블록을 순서 식별자가 포함된 UDP 패킷으로 분할한다."""

    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes는 0보다 커야 합니다.")

    total = len(audio_bytes)
    for offset in range(0, total, max_payload_bytes):
        payload = audio_bytes[offset : offset + max_payload_bytes]
        yield AUDIO_HEADER.pack(sequence_id, total, offset) + payload


def unpack_audio_packet(packet: bytes) -> tuple[int, int, int, bytes]:
    """UDP 오디오 패킷의 헤더와 payload를 분리한다."""

    if len(packet) < AUDIO_HEADER.size:
        raise ValueError("오디오 UDP 패킷이 헤더보다 짧습니다.")
    sequence_id, total, offset = AUDIO_HEADER.unpack_from(packet)
    return sequence_id, total, offset, packet[AUDIO_HEADER.size :]


def pack_direction(direction: tuple[float, float, float]) -> bytes:
    """방향 단위벡터를 고정 길이 UDP 패킷으로 직렬화한다."""

    return DIRECTION_PACKET.pack(*direction)


def unpack_direction(packet: bytes) -> tuple[float, float, float]:
    """고정 길이 UDP 패킷에서 방향 단위벡터를 복원한다."""

    if len(packet) != DIRECTION_PACKET.size:
        raise ValueError("방향 UDP 패킷 길이가 올바르지 않습니다.")
    return DIRECTION_PACKET.unpack(packet)


def pack_sentence_end() -> bytes:
    """외부 문장 종료 검출기가 전송할 고정 payload를 반환한다."""

    return SENTENCE_END_PACKET


def unpack_sentence_end(packet: bytes) -> None:
    """문장 종료 payload를 검증한다.

    Args:
        packet: 외부 DOA·Beamforming 모듈에서 받은 UDP payload.

    Raises:
        ValueError: 현재 프로토콜의 문장 종료 신호가 아닌 경우.
    """

    if packet != SENTENCE_END_PACKET:
        raise ValueError("문장 종료 UDP 패킷이 올바르지 않습니다.")
