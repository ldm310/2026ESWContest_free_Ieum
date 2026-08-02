# Streaming STT Input Contract

## 권장 입력

- 데이터 형식: `np.ndarray`
- dtype: `np.float32`
- shape: `(N,)`
- 채널: mono
- sample rate: 16000Hz
- 값 범위: `-1.0`~`1.0`
- chunk 길이: 20ms~500ms
- chunk 순서: 녹음된 시간 순서
- DOA와 Beamforming을 완료한 이후의 음성을 입력
- 서로 다른 화자의 음성을 하나의 `StreamingSTT` 인스턴스에 섞지 않음

16000Hz에서 `(1600,)` 배열은 100ms mono 음성입니다.

```python
stt.push_audio(audio_chunk, sample_rate=16000)
```

`(N, 1)`, float64, int16 NumPy 입력도 변환하여 처리합니다. 빈 배열은 무시합니다. NaN, Inf 또는 임의 다채널 NumPy 배열은 `ValueError`입니다. 기준과 다른 샘플레이트는 NumPy 선형 보간으로 변환합니다.

## PCM16 입력 계약

- 형식: signed 16-bit PCM
- byte order: little-endian
- sample rate: 16000Hz 권장
- channels: 기본값 1
- 기본 연결 계약: Beamforming 이후 mono PCM

```python
stt.push_pcm16(
    pcm_bytes,
    sample_rate=16000,
    channels=1,
)
```

다채널 PCM16도 `channels`를 명시하면 프레임별 평균으로 mono 변환합니다. 이 기능은 호환 편의를 위한 것이며, 권장 입력은 Beamforming을 마친 mono입니다. byte 길이는 `2 * channels`의 배수여야 합니다.

## 출력 계약

모든 결과는 변경할 수 없는 `STTResult`로 callback에 전달됩니다.

```python
STTResult(
    type="partial",
    text="안녕하세요",
    latency_ms=1032.51,
    timestamp="2026-08-02T15:30:00+09:00",
    sequence_id=1,
    utterance_id=1,
    is_final=False,
    error=None,
)
```

- `partial`: 발화 중 화면을 갱신하는 임시 자막
- `final`: 침묵 또는 `flush()`로 확정된 최종 자막
- `error`: 모델 또는 입력 처리 중 발생한 오류

`sequence_id`는 결과가 발생할 때마다 증가하며 `utterance_id`는 새 발화마다 증가합니다. partial은 UI 갱신용, final은 확정 기록용으로 사용하는 것을 권장합니다. 사용자 callback 예외는 내부 worker를 종료시키지 않습니다.

## Lifecycle 계약

1. `StreamingSTT(on_result=...)` 생성
2. `start()`로 모델 로딩, warm-up, worker 시작
3. `push_audio()` 또는 `push_pcm16()`을 시간 순서대로 호출
4. 입력 전환·화자 전환·종료 시 필요하면 `flush()` 호출
5. 반드시 `stop()` 호출

`start()`와 `stop()`의 중복 호출은 안전합니다. start 전 또는 stop 후 입력은 `RuntimeError`입니다. `flush()`는 현재 발화의 final callback 완료까지 기다립니다.
