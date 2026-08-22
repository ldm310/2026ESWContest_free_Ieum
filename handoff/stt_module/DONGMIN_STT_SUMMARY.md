# 동민 담당 작업: 한국어 실시간 STT

## 한 줄 요약

**준형님 모듈이 정리한 음성을 받아, 말하는 중에는 임시 자막을 보여주고 말이 끝나면 최종 자막을 전달하는 기능을 만들었습니다.**

## 전체 프로젝트에서 담당한 부분

```text
4채널 마이크
    ↓
DOA: 소리가 들어오는 방향 찾기
    ↓
Beamforming: 원하는 방향의 목소리만 강조하기
    ↓
[동민 담당] 음성을 한국어 글자로 변환하기
    ↓
모니터에 자막 표시 또는 결과 저장
```

제가 담당한 부분은 **STT(Speech-to-Text)** 입니다.

DOA, Beamforming, 실제 마이크 제어와 모니터 UI는 제 담당 코드에 포함하지 않았습니다.

## 만든 기능

- `faster-whisper`를 이용한 한국어 음성 인식
- 말하는 중간 결과인 Partial 자막 제공
- 말이 끝난 뒤 확정된 Final 자막 제공
- Beamforming 결과를 받을 수 있는 NumPy 음성 입력 지원
- PCM16 bytes 입력 지원
- Whisper 모델을 한 번만 불러와 계속 재사용
- 입력이 밀리지 않는지 확인할 수 있는 Queue 및 지연시간 통계 제공
- 오류가 발생해도 원인을 callback으로 전달

## Partial과 Final이란?

사용자가 다음 문장을 말한다고 가정합니다.

```text
오늘 회의를 시작하겠습니다.
```

말하는 도중에는 화면의 문장이 다음처럼 계속 바뀔 수 있습니다.

```text
오늘
오늘 회의를
오늘 회의를 시작하겠습니다
```

이처럼 **말하는 중에 빠르게 보여주는 임시 결과가 Partial 자막**입니다.

사용자가 말을 멈추면 다음처럼 문장이 확정됩니다.

```text
오늘 회의를 시작하겠습니다.
```

이처럼 **말이 끝난 뒤 저장하거나 전송할 수 있는 확정 결과가 Final 자막**입니다.

```text
Partial → 화면에 임시로 표시
Final   → 최종 자막으로 확정·저장
```

## 팀 코드와 연결하는 방법

준형님 Beamforming 코드에서 다음 형태의 음성을 전달하면 됩니다.

| 항목 | 권장 형식 |
|---|---|
| 채널 | mono(1채널) |
| Sample rate | 16,000 Hz |
| Python 자료형 | `numpy.ndarray` |
| dtype | `np.float32` |
| shape | `(N,)` |
| 값 범위 | `-1.0` ~ `1.0` |

핵심 연결 코드는 한 줄입니다.

```python
stt.push_audio(beamformed_audio, sample_rate=16000)
```

전체 사용 흐름은 다음과 같습니다.

```python
from stt import STTResult, StreamingSTT


def handle_result(result: STTResult) -> None:
    if result.type == "partial":
        print("화면에 임시 표시:", result.text)
    elif result.type == "final":
        print("최종 자막 저장:", result.text)
    elif result.type == "error":
        print("오류:", result.error)


stt = StreamingSTT(on_result=handle_result)
stt.start()

# Beamforming에서 음성 chunk가 나올 때마다 호출합니다.
stt.push_audio(beamformed_audio, sample_rate=16000)

# 외부에서 문장 종료를 판단하거나 입력을 끝낼 때 호출합니다.
stt.flush()
stt.stop()
```

## 현재까지 확인한 결과

| 확인 항목 | 결과 |
|---|---|
| MacBook 내장 마이크 입력 | 성공 |
| 한국어 Partial 자막 | 출력 확인 |
| 한국어 Final 자막 | 출력 확인 |
| Callback overflow | 0회 |
| 최대 Audio Queue | 1 |
| 자동 테스트 | 20개 통과 |

현재 Mac CPU 실행에서 한 전문용어 WAV의 Partial 추론은 평균 약
`213.27ms`, Final 추론은 약 `1,568.70ms`였습니다. 합성 한국어 30문장
평가에서는 CER `11.26%`, WER `18.14%`, 전문용어 정확도 `95.45%`가
측정됐습니다. 합성 음성 결과이므로 실제 Jetson·ReSpeaker 성능과는 구분합니다.

## 아직 확인해야 하는 부분

- Jetson Orin Nano에서 실제 설치 및 실행
- Jetson CUDA와 CTranslate2의 호환 여부
- Beamforming 출력의 실제 dtype, shape, sample rate, chunk 크기
- 문장 종료를 STT 내부 침묵 감지로 처리할지 외부에서 `flush()`할지 결정
- UI에서 Partial 자막을 교체하고 Final 자막을 확정하는 방법
- 평가용 정답 음성을 이용한 CER/WER 및 전문 용어 정확도 측정

## 팀원에게 필요한 핵심 내용

```text
입력: Beamforming이 끝난 16kHz mono 음성
처리: faster-whisper 기반 한국어 STT
출력: 말하는 중 Partial / 말이 끝나면 Final
연결: stt.push_audio(beamformed_audio, sample_rate=16000)
남은 일: Jetson 실행, Beamforming 실제 연동, UI 연결, 정확도 평가
```

## 관련 문서

- `README.md`: 설치 및 전체 사용법
- `INTERFACE_CONTRACT.md`: 정확한 입력·출력 규격
- `WINDOWS_SETUP.md`: Windows 설치 방법
- `TROUBLESHOOTING.md`: 오류 해결 방법
- `STT_EVALUATION_PLAN.md`: 성능 평가 기준
