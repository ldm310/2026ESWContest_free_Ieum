# Streaming Korean STT Module

DOA·Beamforming 이후 mono 음성을 받아 faster-whisper 기반 한국어 partial/final 결과를 callback으로 제공하는 장치 독립 모듈입니다.

## 1. 담당 범위

이 패키지는 음성 chunk의 발화 감지, rolling partial, 전체 발화 final, 모델 Singleton 사용과 통계를 담당합니다. DOA, Beamforming, 물리 마이크, GUI와 최종 저장은 연동 측에서 담당합니다.

## 2. 전체 파이프라인

```text
4채널 마이크 → DOA → Beamforming → mono 음성
    → StreamingSTT → partial/final STTResult → UI 또는 최종 시스템
```

## 3. 폴더 구조

```text
stt_module/
├── stt/                # 핵심 API와 모델 설정
├── examples/           # NumPy, WAV, PCM16 연결 예제
├── evaluation/         # Windows/macOS TTS 30문장 생성 및 CER/WER 평가
├── tests/              # 실제 모델 없는 자동 테스트
├── sample_data/        # 사용자 WAV 배치 위치 안내
├── requirements-core.txt
├── INTERFACE_CONTRACT.md
├── JETSON_UBUNTU_SSH_SETUP.md
├── TROUBLESHOOTING.md
└── VERSION.txt
```

## 4. 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -c "from stt import StreamingSTT, STTResult; print('import 성공')"
```

SSH 접속 후 Jetson Ubuntu 터미널에서 GitHub 코드를 직접 받고 실행하는 방법은
[JETSON_UBUNTU_SSH_SETUP.md](JETSON_UBUNTU_SSH_SETUP.md)를 참고하십시오.

## 5. 핵심 API

```python
from stt import StreamingSTT, STTResult

stt = StreamingSTT(on_result=handle_result)
stt.start()
stt.push_audio(audio_chunk, sample_rate=16000)
stt.push_pcm16(pcm_bytes, sample_rate=16000, channels=1)
stt.flush()
stats = stt.get_stats()
stt.stop()
```

`start()`는 빠른 partial용 `tiny`와 정확한 final용 `small` 모델을 한 번씩 로딩하고 warm-up합니다. `push_*()`는 thread-safe하며 final 요청은 partial보다 우선합니다. `stop()`은 Queue와 worker를 정리하고 두 모델 참조를 해제합니다.

## 6. 입력 계약

권장 입력은 mono, float32, 16000Hz, `(N,)`, 값 범위 `-1.0`~`1.0`, chunk 20ms~500ms입니다. chunk는 시간 순서대로 전달하고 서로 다른 화자의 음성을 한 인스턴스에 섞지 마십시오. 상세 내용은 [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md)를 참고하십시오.

## 7. 출력 계약

callback은 불변 `STTResult`를 받습니다. `type`은 `partial`, `final`, `error` 중 하나입니다. `sequence_id`, `utterance_id`, timezone 포함 `timestamp`, `latency_ms`, `is_final`, `error`가 함께 제공됩니다.

## 8. NumPy 연동 예제

```python
def handle_result(result: STTResult) -> None:
    if result.type == "partial":
        update_ui(result.text)
    elif result.type == "final":
        save_confirmed_subtitle(result.text)

stt = StreamingSTT(on_result=handle_result)
stt.start()
try:
    for beamformed_audio in beamforming_output_stream:
        stt.push_audio(beamformed_audio, sample_rate=16000)
    stt.flush()
finally:
    stt.stop()
```

`beamforming_output_stream`, `update_ui`, `save_confirmed_subtitle`은 준형님 시스템에서 제공할 연동 지점입니다. [numpy_input_example.py](examples/numpy_input_example.py)에도 최소 골격이 있습니다.

## 9. PCM16 연동 예제

```python
stt.push_pcm16(pcm_bytes, sample_rate=16000, channels=1)
```

다채널 PCM16 평균 변환도 지원하지만 권장 계약은 Beamforming 이후 mono입니다. 실행 예제는 [pcm16_input_example.py](examples/pcm16_input_example.py)입니다.

## 10. partial/final 사용 방법

- `partial`: 말하는 중 자주 바뀌므로 화면 갱신에 사용
- `final`: 침묵 또는 flush로 확정되므로 저장·전송에 사용
- `error`: 원인을 기록하고 입력 또는 환경 상태 점검에 사용

partial 요청은 무제한으로 쌓이지 않고 최신 snapshot 하나만 유지합니다. final은 대기 partial보다 우선하며 Whisper 추론은 동시에 하나만 실행됩니다.

## 11. flush() 사용 시점

입력 stream 종료, 화자 전환, 외부 문장 종료 신호 또는 프로그램 단계 전환 시 호출하십시오. 현재 발화가 없으면 즉시 반환하며, 발화가 있으면 final callback 완료까지 기다립니다.

## 12. 통계 조회

```python
stats = stt.get_stats()
print(stats["average_partial_latency_ms"])
print(stats["max_audio_queue_size"])
```

반환값은 내부 상태와 분리된 새 dict입니다.

## 13. Jetson Ubuntu 직접 설치

Jetson에서 GitHub clone/pull, `project_main` 적용, 가상환경, CUDA 확인, 테스트와
실행 명령은 [JETSON_UBUNTU_SSH_SETUP.md](JETSON_UBUNTU_SSH_SETUP.md)에
단계별로 정리했습니다.

## 14. 테스트 실행

```bash
python -m unittest discover -s tests -v
```

29개 테스트는 실제 Whisper 모델과 마이크 없이 입력 변환, lifecycle, callback,
flush, 통계, partial/final 모델 분리와 평가 데이터·지표 계산을 검증합니다.

## 15. 실행 가능한 WAV 예제

```bash
python examples/streaming_api_example.py sample_data/sample.wav
python examples/pcm16_input_example.py sample_data/sample.wav
```

권장 WAV는 mono signed 16-bit PCM입니다. 개인 녹음은 패키지에 포함하지 않습니다.

## 16. Jetson 연결 지점

```text
beamformed_audio
    ↓
stt.push_audio(beamformed_audio, sample_rate=16000)
```

준형님 확인 범위는 Jetson 설치, CUDA/CTranslate2 호환 설정, DOA, Beamforming, 실제 마이크와 UI입니다. 이 패키지는 검증되지 않은 Jetson CUDA 버전이나 설치 명령을 추측하지 않습니다.

## 17. 현재 제한사항

- 검증 환경: macOS, Python 3.12.13, faster-whisper 1.2.1, NumPy 2.5.1, CPU 추론
- Jetson 실제 장치 실행은 아직이며 Mac 로컬과 정적 호환성만 점검했습니다.
- 발화 종료는 외부 VAD가 아닌 RMS 임계값 방식입니다.
- 다른 샘플레이트는 간단한 NumPy 선형 보간을 사용합니다.
- 전체 발화는 final 정확도를 위해 침묵 또는 flush까지 메모리에 유지됩니다.
- CPU에서는 partial/final latency가 실시간 입력 주기보다 길 수 있습니다.

## 18. 문의 또는 확인해야 할 사항

- Beamforming 출력의 정확한 dtype, shape, sample rate와 chunk 주기
- 환경별 무음 RMS에 맞춘 `silence_threshold`
- 화자 전환 시 `flush()` 호출 주체
- Jetson GPU의 `int8_float16` 실행 성능
- Jetson CUDA와 CTranslate2의 실제 호환 조합
- UI에서 partial 교체와 final 확정 결과를 처리하는 방식
