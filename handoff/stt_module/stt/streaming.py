"""외부 mono 음성 chunk를 partial/final 자막으로 변환하는 Streaming STT API."""

from __future__ import annotations

import logging
import queue
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from stt.config import BEAM_SIZE, HOTWORDS, INITIAL_PROMPT, LANGUAGE, SAMPLE_RATE
from stt.model import ModelManager, get_model
from stt.types import STTResult


DEFAULT_SILENCE_THRESHOLD = 0.01
DEFAULT_PARTIAL_INTERVAL_SECONDS = 0.25
DEFAULT_SILENCE_DURATION_SECONDS = 0.45
DEFAULT_PRE_ROLL_SECONDS = 0.3
DEFAULT_MIN_SPEECH_DURATION_SECONDS = 0.3
DEFAULT_PREVIEW_SECONDS = 4.0
WARM_UP_AUDIO_SECONDS = 0.25

PARTIAL_BEAM_SIZE = 1
PARTIAL_TEMPERATURE = 0.0
PARTIAL_TASK = "transcribe"
PARTIAL_CONDITION_ON_PREVIOUS_TEXT = False
PARTIAL_VAD_FILTER = False
STREAMING_FINAL_VAD_FILTER = False

AudioArray = NDArray[np.float32]
ResultCallback = Callable[[STTResult], None]

_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[StreamingSTT callback 오류] %(message)s"))
    _LOGGER.addHandler(_handler)
    _LOGGER.setLevel(logging.ERROR)
    _LOGGER.propagate = False

# 여러 StreamingSTT 객체가 생성돼도 Singleton Whisper 모델에는 동시에 한
# thread만 진입한다. 일반 사용에서는 객체 하나를 장기간 재사용하는 것을 권장한다.
_MODEL_INFERENCE_LOCK = threading.Lock()


def normalize_text(text: str) -> str:
    """문자열의 앞뒤 및 연속 공백을 정리한다."""

    return " ".join(text.split())


def longest_common_prefix(previous: str, current: str) -> str:
    """두 문자열이 공유하는 가장 긴 prefix를 반환한다."""

    prefix_length = 0
    for previous_character, current_character in zip(previous, current):
        if previous_character != current_character:
            break
        prefix_length += 1
    return previous[:prefix_length]


def merge_partial(stable_prefix: str, current_text: str) -> str:
    """안정 prefix와 최신 partial 후보를 중복 없이 합친다."""

    stable_prefix = normalize_text(stable_prefix)
    current_text = normalize_text(current_text)
    if not stable_prefix:
        return current_text
    if current_text.startswith(stable_prefix):
        return current_text
    if stable_prefix.startswith(current_text):
        return stable_prefix
    overlap = longest_common_prefix(stable_prefix, current_text)
    return normalize_text(stable_prefix + current_text[len(overlap) :])


def resample_linear(
    audio: AudioArray,
    source_rate: int,
    target_rate: int,
) -> AudioArray:
    """외부 의존성 없이 mono 오디오를 선형 보간 리샘플링한다.

    Args:
        audio: mono float32 입력 배열.
        source_rate: 입력 샘플레이트.
        target_rate: 출력 샘플레이트.

    Returns:
        target_rate에 맞춘 새 float32 배열.

    Raises:
        ValueError: 샘플레이트가 양수가 아닌 경우.
    """

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("샘플레이트는 0보다 커야 합니다.")
    if audio.size == 0 or source_rate == target_rate:
        return audio.astype(np.float32, copy=True)

    output_length = max(1, round(audio.size * target_rate / source_rate))
    if audio.size == 1:
        return np.full(output_length, audio[0], dtype=np.float32)

    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.arange(output_length, dtype=np.float64) * (
        source_rate / target_rate
    )
    target_positions = np.minimum(target_positions, audio.size - 1)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def prepare_audio_array(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> AudioArray:
    """지원 입력을 검증하고 mono float32 target-rate 배열로 변환한다.

    빈 배열은 유효한 no-op 입력으로 간주해 빈 float32 배열을 반환한다.
    다채널 NumPy 배열은 의도치 않은 채널 혼합을 막기 위해 거부한다.
    """

    if not isinstance(audio, np.ndarray):
        raise TypeError("audio는 numpy.ndarray여야 합니다.")
    if audio.ndim == 2:
        if audio.shape[1] != 1:
            raise ValueError("NumPy 오디오는 mono 또는 (N, 1) 형식이어야 합니다.")
        audio = audio[:, 0]
    elif audio.ndim != 1:
        raise ValueError("NumPy 오디오는 1차원 mono 배열이어야 합니다.")

    if audio.size == 0:
        return np.empty(0, dtype=np.float32)

    if audio.dtype == np.int16:
        prepared = audio.astype(np.float32) / 32_768.0
    elif np.issubdtype(audio.dtype, np.floating):
        prepared = audio.astype(np.float32)
    else:
        raise ValueError("지원하는 dtype은 float32, float64, int16입니다.")

    if not np.all(np.isfinite(prepared)):
        raise ValueError("오디오에 NaN 또는 Inf가 포함되어 있습니다.")
    return resample_linear(prepared, source_rate, target_rate)


class _StablePartialTracker:
    """여러 결과에서 반복 확인된 prefix를 보수적으로 유지한다."""

    def __init__(self) -> None:
        self.reset()

    def update(self, text: str) -> str:
        """최신 partial을 반영해 표시할 안정화 문자열을 반환한다."""

        current = normalize_text(text)
        if not current:
            return ""
        common_prefix = longest_common_prefix(self._previous, current)

        if common_prefix:
            if self._candidate and common_prefix.startswith(self._candidate):
                self._confirmations += 1
            else:
                self._confirmations = 1
            self._candidate = common_prefix
            if self._confirmations >= 2 and len(common_prefix) >= len(self._stable):
                self._stable = common_prefix

        if self._stable and not current.startswith(self._stable):
            self._conflicts += 1
            if self._conflicts >= 2:
                self._stable = common_prefix
                self._candidate = common_prefix
                self._confirmations = 0
                self._conflicts = 0
        else:
            self._conflicts = 0

        self._previous = current
        return merge_partial(self._stable, current)

    def reset(self) -> None:
        """새 발화를 위해 안정화 상태를 초기화한다."""

        self._previous = ""
        self._stable = ""
        self._candidate = ""
        self._confirmations = 0
        self._conflicts = 0


class _SampleBuffer:
    """가변 길이 chunk를 sample 개수 기준 rolling window로 보관한다."""

    def __init__(self, max_samples: int) -> None:
        self._max_samples = max(0, max_samples)
        self._chunks: deque[AudioArray] = deque()
        self._sample_count = 0

    def append(self, chunk: AudioArray) -> None:
        """chunk를 추가하고 오래된 sample을 정확한 길이만큼 제거한다."""

        if self._max_samples == 0 or chunk.size == 0:
            return
        self._chunks.append(chunk)
        self._sample_count += chunk.size
        while self._sample_count > self._max_samples and self._chunks:
            excess = self._sample_count - self._max_samples
            oldest = self._chunks[0]
            if oldest.size <= excess:
                self._chunks.popleft()
                self._sample_count -= oldest.size
            else:
                self._chunks[0] = oldest[excess:].copy()
                self._sample_count -= excess

    def snapshot(self) -> AudioArray:
        """현재 rolling window의 독립된 연속 배열을 반환한다."""

        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(tuple(self._chunks)).astype(np.float32, copy=False)

    def clear(self) -> None:
        """모든 sample을 제거한다."""

        self._chunks.clear()
        self._sample_count = 0


class _Stats:
    """StreamingSTT thread들이 공유하는 통계 저장소."""

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._lock = threading.Lock()
        self.partial_count = 0
        self.partial_total_ms = 0.0
        self.partial_max_ms = 0.0
        self.final_count = 0
        self.final_total_ms = 0.0
        self.final_max_ms = 0.0
        self.error_count = 0
        self.max_audio_queue_size = 0
        self.skipped_partial_requests = 0
        self.audio_samples_received = 0

    def record_audio(self, sample_count: int, queue_size: int) -> None:
        """정규화 후 받은 sample 수와 Queue 최대 길이를 기록한다."""

        with self._lock:
            self.audio_samples_received += sample_count
            self.max_audio_queue_size = max(self.max_audio_queue_size, queue_size)

    def record_latency(self, result_type: Literal["partial", "final"], latency: float) -> None:
        """partial 또는 final latency를 누적한다."""

        with self._lock:
            if result_type == "partial":
                self.partial_count += 1
                self.partial_total_ms += latency
                self.partial_max_ms = max(self.partial_max_ms, latency)
            else:
                self.final_count += 1
                self.final_total_ms += latency
                self.final_max_ms = max(self.final_max_ms, latency)

    def record_error(self) -> None:
        """STT 처리 오류를 한 건 추가한다."""

        with self._lock:
            self.error_count += 1

    def record_skipped_partial(self) -> None:
        """교체 또는 무효화된 partial 요청을 한 건 추가한다."""

        with self._lock:
            self.skipped_partial_requests += 1

    def as_dict(self, current_queue_size: int) -> dict[str, int | float]:
        """외부 변경이 내부 상태에 영향을 주지 않는 통계 복사본을 반환한다."""

        with self._lock:
            return {
                "partial_count": self.partial_count,
                "final_count": self.final_count,
                "error_count": self.error_count,
                "average_partial_latency_ms": (
                    self.partial_total_ms / self.partial_count
                    if self.partial_count
                    else 0.0
                ),
                "max_partial_latency_ms": self.partial_max_ms,
                "average_final_latency_ms": (
                    self.final_total_ms / self.final_count
                    if self.final_count
                    else 0.0
                ),
                "max_final_latency_ms": self.final_max_ms,
                "max_audio_queue_size": self.max_audio_queue_size,
                "current_audio_queue_size": current_queue_size,
                "skipped_partial_requests": self.skipped_partial_requests,
                "audio_samples_received": self.audio_samples_received,
                "audio_seconds_received": self.audio_samples_received / self._sample_rate,
            }


class _ResultEmitter:
    """sequence와 callback 예외 격리를 담당한다."""

    def __init__(self, callback: ResultCallback, stats: _Stats) -> None:
        self._callback = callback
        self._stats = stats
        self._sequence_lock = threading.Lock()
        self._sequence_id = 0

    def emit(
        self,
        result_type: Literal["partial", "final", "error"],
        text: str,
        latency_ms: float,
        utterance_id: int,
        is_final: bool,
        error: str | None = None,
    ) -> None:
        """불변 결과를 만든 뒤 내부 lock 밖에서 사용자 callback을 호출한다."""

        with self._sequence_lock:
            self._sequence_id += 1
            sequence_id = self._sequence_id

        result = STTResult(
            type=result_type,
            text=text,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
            sequence_id=sequence_id,
            utterance_id=utterance_id,
            is_final=is_final,
            error=error,
        )
        try:
            self._callback(result)
        except Exception:
            _LOGGER.exception("사용자 on_result callback 실행에 실패했습니다.")


def _normalize_segments(segments: Any) -> str:
    """faster-whisper segment를 문자열 하나로 결합한다."""

    return normalize_text("".join(segment.text for segment in segments))


def _run_model_transcription(
    model: Any,
    audio_input: AudioArray | str,
    beam_size: int,
    is_partial: bool,
) -> str:
    """partial/final 설정으로 모델을 호출하고 지연 segment를 소비한다."""

    options: dict[str, Any] = {
        "language": LANGUAGE,
        "beam_size": beam_size,
        # StreamingSTT가 RMS와 침묵 길이로 발화를 이미 잘라낸다. 여기서
        # Silero VAD를 다시 적용하면 문장 앞뒤 음절이 잘리거나 final이
        # 느려질 수 있으므로 partial/final 모두 중복 필터를 사용하지 않는다.
        "vad_filter": (
            PARTIAL_VAD_FILTER if is_partial else STREAMING_FINAL_VAD_FILTER
        ),
        "without_timestamps": True,
    }
    if INITIAL_PROMPT:
        options["initial_prompt"] = INITIAL_PROMPT
    if HOTWORDS:
        options["hotwords"] = HOTWORDS
    if is_partial:
        options.update(
            task=PARTIAL_TASK,
            temperature=PARTIAL_TEMPERATURE,
            condition_on_previous_text=PARTIAL_CONDITION_ON_PREVIOUS_TEXT,
        )
    segments, _ = model.transcribe(audio_input, **options)
    return _normalize_segments(segments)


def _write_temporary_wav(audio: AudioArray, sample_rate: int) -> Path:
    """NumPy 직접 입력 미지원 시 사용할 mono PCM WAV를 작성한다."""

    pcm_audio = (np.clip(audio, -1.0, 1.0) * 32_767).astype("<i2")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with wave.open(str(temporary_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _transcribe_audio(
    model: Any,
    audio: AudioArray,
    sample_rate: int,
    beam_size: int,
    is_partial: bool,
) -> str:
    """전역 lock 아래 NumPy 직접 추론 후 필요할 때 WAV로 fallback한다."""

    with _MODEL_INFERENCE_LOCK:
        try:
            return _run_model_transcription(model, audio, beam_size, is_partial)
        except (TypeError, AttributeError):
            temporary_path = _write_temporary_wav(audio, sample_rate)
            try:
                return _run_model_transcription(
                    model, str(temporary_path), beam_size, is_partial
                )
            finally:
                temporary_path.unlink(missing_ok=True)


@dataclass
class _InferenceRequest:
    """단일 추론 worker에 전달하는 오디오 snapshot."""

    utterance_id: int
    audio: AudioArray
    completion: threading.Event | None = None


class _InferenceScheduler:
    """최신 partial 하나와 우선 final 요청을 단일 thread에서 처리한다."""

    def __init__(
        self,
        final_model: Any,
        partial_model: Any,
        sample_rate: int,
        emitter: _ResultEmitter,
        stats: _Stats,
    ) -> None:
        self._final_model = final_model
        self._partial_model = partial_model
        self._sample_rate = sample_rate
        self._emitter = emitter
        self._stats = stats
        self._condition = threading.Condition()
        self._latest_partial: _InferenceRequest | None = None
        self._final_requests: deque[_InferenceRequest] = deque()
        self._active_utterance_id: int | None = None
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._tracker = _StablePartialTracker()
        self._last_partial = ""

    def start(self) -> None:
        """단일 Whisper 추론 thread를 시작한다."""

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="streaming-stt-inference",
            daemon=False,
        )
        self._thread.start()

    def submit_partial(self, request: _InferenceRequest) -> None:
        """대기 partial을 최신 snapshot으로 교체한다."""

        with self._condition:
            if self._stop_requested:
                return
            if self._latest_partial is not None:
                self._stats.record_skipped_partial()
            self._latest_partial = request
            self._active_utterance_id = request.utterance_id
            self._condition.notify()

    def submit_final(self, request: _InferenceRequest) -> None:
        """대기 partial을 취소하고 final을 우선 목록에 넣는다."""

        with self._condition:
            if self._stop_requested:
                if request.completion is not None:
                    request.completion.set()
                return
            if self._latest_partial is not None:
                self._latest_partial = None
                self._stats.record_skipped_partial()
            if self._active_utterance_id == request.utterance_id:
                self._active_utterance_id = None
            self._final_requests.append(request)
            self._condition.notify()

    def cancel_partial(self, utterance_id: int) -> None:
        """짧은 잡음 발화에 속한 partial을 무효화한다."""

        with self._condition:
            if (
                self._latest_partial is not None
                and self._latest_partial.utterance_id == utterance_id
            ):
                self._latest_partial = None
                self._stats.record_skipped_partial()
            if self._active_utterance_id == utterance_id:
                self._active_utterance_id = None

    def stop(self) -> None:
        """대기 partial은 취소하고 모든 final을 처리한 뒤 종료한다."""

        with self._condition:
            self._stop_requested = True
            if self._latest_partial is not None:
                self._latest_partial = None
                self._stats.record_skipped_partial()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _worker_loop(self) -> None:
        """항상 final을 먼저 선택하며 Whisper를 직렬 실행한다."""

        while True:
            with self._condition:
                while (
                    not self._final_requests
                    and self._latest_partial is None
                    and not self._stop_requested
                ):
                    self._condition.wait()
                if self._final_requests:
                    request = self._final_requests.popleft()
                    request_type = "final"
                elif self._latest_partial is not None:
                    request = self._latest_partial
                    self._latest_partial = None
                    request_type = "partial"
                elif self._stop_requested:
                    return
                else:
                    continue

            if request_type == "final":
                self._process_final(request)
            else:
                self._process_partial(request)

    def _process_partial(self, request: _InferenceRequest) -> None:
        """partial을 추론하고 여전히 현재 발화일 때만 callback한다."""

        started_at = time.perf_counter()
        try:
            text = _transcribe_audio(
                self._partial_model,
                request.audio,
                self._sample_rate,
                PARTIAL_BEAM_SIZE,
                True,
            )
        except Exception as exc:
            latency = (time.perf_counter() - started_at) * 1_000
            self._stats.record_error()
            self._emitter.emit(
                "error", "", latency, request.utterance_id, False, str(exc)
            )
            return

        latency = (time.perf_counter() - started_at) * 1_000
        self._stats.record_latency("partial", latency)
        with self._condition:
            is_current = self._active_utterance_id == request.utterance_id
        if not is_current:
            self._stats.record_skipped_partial()
            return

        displayed = self._tracker.update(text)
        if displayed and displayed != self._last_partial:
            self._last_partial = displayed
            self._emitter.emit(
                "partial", displayed, latency, request.utterance_id, False
            )

    def _process_final(self, request: _InferenceRequest) -> None:
        """전체 발화를 final 설정으로 한 번 인식하고 callback한다."""

        self._tracker.reset()
        self._last_partial = ""
        started_at = time.perf_counter()
        try:
            text = _transcribe_audio(
                self._final_model,
                request.audio,
                self._sample_rate,
                BEAM_SIZE,
                False,
            )
        except Exception as exc:
            latency = (time.perf_counter() - started_at) * 1_000
            self._stats.record_error()
            self._emitter.emit(
                "error", "", latency, request.utterance_id, True, str(exc)
            )
        else:
            latency = (time.perf_counter() - started_at) * 1_000
            self._stats.record_latency("final", latency)
            self._emitter.emit(
                "final", text, latency, request.utterance_id, True
            )
        finally:
            if request.completion is not None:
                request.completion.set()


class _AudioProcessor:
    """RMS 발화 감지와 preview/전체 발화 buffer를 관리한다."""

    def __init__(
        self,
        sample_rate: int,
        partial_interval: float,
        silence_threshold: float,
        silence_duration: float,
        preview_seconds: float,
        pre_roll_seconds: float,
        min_speech_duration: float,
        scheduler: _InferenceScheduler,
    ) -> None:
        self._sample_rate = sample_rate
        self._partial_samples_required = max(1, round(partial_interval * sample_rate))
        self._silence_threshold = silence_threshold
        self._silence_samples_required = max(1, round(silence_duration * sample_rate))
        self._minimum_speech_samples = max(1, round(min_speech_duration * sample_rate))
        self._scheduler = scheduler

        self._pre_roll = _SampleBuffer(round(pre_roll_seconds * sample_rate))
        self._preview = _SampleBuffer(round(preview_seconds * sample_rate))
        self._utterance: list[AudioArray] = []
        self._is_speaking = False
        self._utterance_id = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._samples_since_partial = 0

    def process(self, audio: AudioArray) -> None:
        """한 chunk를 발화 상태와 두 buffer에 반영한다."""

        is_voice = calculate_rms(audio) > self._silence_threshold
        if not self._is_speaking:
            if is_voice:
                self._start_utterance(audio)
            else:
                self._pre_roll.append(audio)
            return

        self._utterance.append(audio)
        self._preview.append(audio)
        self._samples_since_partial += audio.size
        if is_voice:
            self._speech_samples += audio.size
            self._silence_samples = 0
        else:
            self._silence_samples += audio.size

        if self._samples_since_partial >= self._partial_samples_required:
            self._scheduler.submit_partial(
                _InferenceRequest(self._utterance_id, self._preview.snapshot())
            )
            self._samples_since_partial = 0
        if self._silence_samples >= self._silence_samples_required:
            self.finalize()

    def finalize(self, completion: threading.Event | None = None) -> None:
        """침묵 또는 flush 시 현재 유효 발화를 final 요청으로 제출한다."""

        if not self._is_speaking:
            if completion is not None:
                completion.set()
            return

        utterance = self._utterance
        utterance_id = self._utterance_id
        speech_samples = self._speech_samples
        full_audio = np.concatenate(tuple(utterance)).astype(np.float32, copy=False)

        pre_roll_samples = self._pre_roll._max_samples
        trailing = full_audio[-pre_roll_samples:] if pre_roll_samples else full_audio[:0]
        self._reset(trailing)

        if speech_samples < self._minimum_speech_samples:
            self._scheduler.cancel_partial(utterance_id)
            if completion is not None:
                completion.set()
            return
        self._scheduler.submit_final(
            _InferenceRequest(utterance_id, full_audio.copy(), completion)
        )

    def _start_utterance(self, first_voice_chunk: AudioArray) -> None:
        """pre-roll을 포함해 새 발화를 시작한다."""

        self._utterance_id += 1
        pre_roll = self._pre_roll.snapshot()
        self._utterance = [pre_roll, first_voice_chunk] if pre_roll.size else [first_voice_chunk]
        self._preview.clear()
        if pre_roll.size:
            self._preview.append(pre_roll)
        self._preview.append(first_voice_chunk)
        self._pre_roll.clear()
        self._is_speaking = True
        self._speech_samples = first_voice_chunk.size
        self._silence_samples = 0
        self._samples_since_partial = 0

    def _reset(self, trailing: AudioArray) -> None:
        """발화 상태를 초기화하고 다음 시작을 위한 pre-roll을 남긴다."""

        self._is_speaking = False
        self._utterance = []
        self._preview.clear()
        self._pre_roll.clear()
        self._pre_roll.append(trailing)
        self._speech_samples = 0
        self._silence_samples = 0
        self._samples_since_partial = 0


def calculate_rms(audio: AudioArray) -> float:
    """mono float32 chunk의 RMS 값을 계산한다."""

    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


@dataclass
class _FlushCommand:
    """Audio Queue 순서를 보존하는 동기 flush 제어 메시지."""

    completed: threading.Event


class StreamingSTT:
    """외부 음성 chunk를 thread-safe하게 한국어 partial/final 결과로 변환한다.

    빈 NumPy 배열이나 빈 PCM bytes는 오류 없이 무시한다. 내부 기준
    샘플레이트와 다른 입력은 NumPy 선형 보간으로 자동 리샘플링한다.
    """

    def __init__(
        self,
        on_result: ResultCallback,
        sample_rate: int = SAMPLE_RATE,
        partial_interval: float = DEFAULT_PARTIAL_INTERVAL_SECONDS,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        silence_duration: float = DEFAULT_SILENCE_DURATION_SECONDS,
        preview_seconds: float = DEFAULT_PREVIEW_SECONDS,
        pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS,
        min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION_SECONDS,
    ) -> None:
        """Streaming STT 설정과 lifecycle 상태를 초기화한다."""

        if not callable(on_result):
            raise TypeError("on_result는 호출 가능한 함수여야 합니다.")
        numeric_settings = {
            "sample_rate": sample_rate,
            "partial_interval": partial_interval,
            "silence_duration": silence_duration,
            "preview_seconds": preview_seconds,
            "min_speech_duration": min_speech_duration,
        }
        if any(value <= 0 for value in numeric_settings.values()):
            raise ValueError("sample_rate와 시간 설정은 0보다 커야 합니다.")
        if silence_threshold < 0 or pre_roll_seconds < 0:
            raise ValueError("silence_threshold와 pre_roll_seconds는 0 이상이어야 합니다.")

        self._sample_rate = sample_rate
        self._partial_interval = partial_interval
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._preview_seconds = preview_seconds
        self._pre_roll_seconds = pre_roll_seconds
        self._min_speech_duration = min_speech_duration

        self._stats = _Stats(sample_rate)
        self._emitter = _ResultEmitter(on_result, self._stats)
        self._state_condition = threading.Condition()
        self._state: Literal["created", "starting", "running", "stopping", "stopped"] = "created"
        self._audio_queue: queue.Queue[AudioArray | _FlushCommand] = queue.Queue()
        self._audio_stop = threading.Event()
        self._audio_thread: threading.Thread | None = None
        self._scheduler: _InferenceScheduler | None = None
        self._processor: _AudioProcessor | None = None

    def start(self) -> None:
        """Singleton 모델 warm-up과 두 worker thread를 한 번만 시작한다.

        중복 호출은 이미 실행 중이면 그대로 반환한다. start가 동시에 진행
        중이면 완료를 기다린다. stop 이후에는 같은 객체를 재시작할 수 없다.
        """

        with self._state_condition:
            while self._state == "starting":
                self._state_condition.wait()
            if self._state == "running":
                return
            if self._state in {"stopping", "stopped"}:
                raise RuntimeError("중지된 StreamingSTT 객체는 다시 시작할 수 없습니다.")
            self._state = "starting"

        try:
            # partial은 반응성이 높은 tiny, final은 정확도가 높은 기본 모델을
            # 각각 한 번만 로드해 이후 모든 발화에서 재사용한다.
            model = get_model()
            partial_model = ModelManager.get_partial_model()
            silent_audio = np.zeros(
                round(self._sample_rate * WARM_UP_AUDIO_SECONDS), dtype=np.float32
            )
            _transcribe_audio(
                partial_model,
                silent_audio,
                self._sample_rate,
                PARTIAL_BEAM_SIZE,
                True,
            )
            if partial_model is not model:
                _transcribe_audio(
                    model,
                    silent_audio,
                    self._sample_rate,
                    BEAM_SIZE,
                    False,
                )

            scheduler = _InferenceScheduler(
                model,
                partial_model,
                self._sample_rate,
                self._emitter,
                self._stats,
            )
            processor = _AudioProcessor(
                self._sample_rate,
                self._partial_interval,
                self._silence_threshold,
                self._silence_duration,
                self._preview_seconds,
                self._pre_roll_seconds,
                self._min_speech_duration,
                scheduler,
            )
            scheduler.start()
            self._scheduler = scheduler
            self._processor = processor
            self._audio_stop.clear()
            self._audio_thread = threading.Thread(
                target=self._audio_worker_loop,
                name="streaming-stt-audio",
                daemon=False,
            )
            self._audio_thread.start()
        except Exception as exc:
            # 모델 로딩 이후 warm-up 또는 worker 시작이 실패했을 수 있으므로
            # 부분적으로 시작된 리소스를 남기지 않고 다음 start 재시도를 허용한다.
            if self._scheduler is not None:
                self._scheduler.stop()
                self._scheduler = None
            ModelManager.unload_model()
            self._stats.record_error()
            self._emitter.emit("error", "", 0.0, 0, False, str(exc))
            with self._state_condition:
                self._state = "created"
                self._state_condition.notify_all()
            raise RuntimeError(f"StreamingSTT 시작에 실패했습니다. 원인: {exc}") from exc

        with self._state_condition:
            self._state = "running"
            self._state_condition.notify_all()

    def push_audio(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
    ) -> None:
        """NumPy 음성 chunk를 정규화·리샘플링해 처리 Queue에 추가한다."""

        source_rate = self._sample_rate if sample_rate is None else sample_rate
        prepared = prepare_audio_array(audio, source_rate, self._sample_rate)
        if prepared.size == 0:
            return
        with self._state_condition:
            self._require_running()
            self._audio_queue.put_nowait(prepared)
            self._stats.record_audio(prepared.size, self._audio_queue.qsize())

    def push_pcm16(
        self,
        pcm_bytes: bytes,
        sample_rate: int | None = None,
        channels: int = 1,
    ) -> None:
        """little-endian signed PCM16 bytes를 mono float32로 변환해 입력한다."""

        if not isinstance(pcm_bytes, bytes):
            raise TypeError("pcm_bytes는 bytes여야 합니다.")
        if channels <= 0:
            raise ValueError("channels는 1 이상이어야 합니다.")
        frame_width = 2 * channels
        if len(pcm_bytes) % frame_width != 0:
            raise ValueError("PCM16 byte 길이가 채널별 완전한 frame과 맞지 않습니다.")
        if not pcm_bytes:
            return

        pcm = np.frombuffer(pcm_bytes, dtype="<i2").reshape(-1, channels)
        mono = pcm.astype(np.float32).mean(axis=1) / 32_768.0
        self.push_audio(mono, sample_rate)

    def flush(self) -> None:
        """외부 문장 종료 신호를 받아 현재 발화를 즉시 final로 확정한다.

        Queue에 먼저 들어온 음성을 모두 반영한 뒤 final callback이 끝날 때까지
        기다린다. DOA·Beamforming 모듈이 문장 종료를 검출하면 침묵 timeout을
        기다리지 않고 이 메서드를 호출하는 것이 권장된다.
        """

        completed = threading.Event()
        with self._state_condition:
            self._require_running()
            self._audio_queue.put_nowait(_FlushCommand(completed))
            self._stats.record_audio(0, self._audio_queue.qsize())
        completed.wait()

    def stop(self) -> None:
        """입력을 닫고 Queue, final, 두 worker, 모델을 순서대로 정리한다."""

        with self._state_condition:
            while self._state == "starting":
                self._state_condition.wait()
            if self._state == "stopped":
                return
            if self._state == "stopping":
                while self._state != "stopped":
                    self._state_condition.wait()
                return
            if self._state == "created":
                self._state = "stopped"
                self._state_condition.notify_all()
                return
            self._state = "stopping"

        self._audio_stop.set()
        if self._audio_thread is not None:
            self._audio_thread.join()
            self._audio_thread = None
        if self._scheduler is not None:
            self._scheduler.stop()
        ModelManager.unload_model()

        with self._state_condition:
            self._state = "stopped"
            self._state_condition.notify_all()

    def get_stats(self) -> dict[str, int | float]:
        """현재 성능 통계의 새 dict를 thread-safe하게 반환한다."""

        return self._stats.as_dict(self._audio_queue.qsize())

    def _require_running(self) -> None:
        """호출 시점이 running이 아니면 명확한 RuntimeError를 발생시킨다."""

        if self._state == "running":
            return
        if self._state == "created":
            raise RuntimeError("StreamingSTT.start()를 먼저 호출해야 합니다.")
        if self._state == "starting":
            raise RuntimeError("StreamingSTT 모델 준비가 아직 완료되지 않았습니다.")
        raise RuntimeError("중지 중이거나 중지된 StreamingSTT에는 음성을 넣을 수 없습니다.")

    def _audio_worker_loop(self) -> None:
        """입력 Queue를 순서대로 소비하고 flush 제어 메시지를 처리한다."""

        assert self._processor is not None
        while not self._audio_stop.is_set() or not self._audio_queue.empty():
            try:
                item = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if isinstance(item, _FlushCommand):
                    self._processor.finalize(item.completed)
                else:
                    self._processor.process(item)
            except Exception as exc:
                self._stats.record_error()
                self._emitter.emit("error", "", 0.0, 0, False, str(exc))
                if isinstance(item, _FlushCommand):
                    item.completed.set()
            finally:
                self._audio_queue.task_done()

        # stop()은 마지막 입력까지 소비한 뒤 남은 유효 발화를 final로 제출한다.
        self._processor.finalize()
