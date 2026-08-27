# Whisper 기반 한국어 실시간 STT 구조 및 입출력 명세서

- 작성일: 2026-08-22
- 기준 브랜치: `feature/dongmin`
- 기준 코드: `13b1ddb` (`fix: reduce streaming STT hallucinations`)

## 1. 문서 목적

본 문서는 Jetson Orin Nano용 한국어 실시간 자막 시스템에서 Whisper를
어떻게 사용했는지 설명한다. 특히 다음 네 가지를 명확히 구분한다.

1. OpenAI Whisper 원본 모델의 아키텍처
2. 현재 프로젝트가 Whisper를 가져와 실행하는 방법
3. Whisper 내부에서 수정한 부분과 수정하지 않은 부분
4. 외부 모듈, Whisper 모델, UI 사이의 입력·출력 계약

## 2. 핵심 결론

현재 프로젝트는 **WhisperLive 소스코드를 사용하거나 수정한 것이 아니다.**
OpenAI가 학습한 Whisper 사전학습 모델을 `faster-whisper`로 실행합니다.

다음 부분은 수정하지 않았습니다.

- Whisper Encoder·Decoder 아키텍처
- Multi-Head Attention 구조
- Tokenizer 어휘 및 특수 토큰
- `tiny`, `small` 모델 가중치
- Whisper 학습 데이터와 학습 방법
- Loss function과 모델 parameter

다음 부분은 프로젝트에서 직접 구현했습니다.

- DOA·Beamforming 후 mono 음성을 받는 API
- Partial과 Final을 구분하는 Rolling Buffer 방식
- `tiny` Partial과 `small` Final 모델 운영
- Singleton·Lazy Loading 모델 관리
- 적응형 RMS 발화 감지와 Final VAD
- Beamforming 음량·DC offset·peak 보정
- Queue, worker thread, 최신 Partial 유지와 Final 우선 처리
- 신뢰도 및 전문용어 환각 필터
- UI에 전달하는 `STTResult` 메타데이터

따라서 본 시스템은 **Whisper 모델을 재학습한 것이 아니라, Whisper 추론 전·후
파이프라인을 실시간 임베디드 시스템에 맞게 설계한 것**입니다.

## 3. Whisper 원본 모델 아키텍처

Whisper는 음성 특징을 입력받아 텍스트 token을 순차적으로 생성하는
**Transformer Encoder–Decoder 기반 sequence-to-sequence 모델**입니다. OpenAI는
한 모델에서 다국어 음성 인식, 영어 번역, 언어 식별 등을 특수 token으로
구분하여 학습했습니다.

```text
16 kHz mono waveform
        ↓
Log-Mel Spectrogram
        ↓
Convolution stem + positional encoding
        ↓
Transformer Audio Encoder
        ↓
Encoded audio representation
        ↓
Transformer Text Decoder
        ↓
Text token sequence
        ↓
Tokenizer decode
        ↓
Transcription text
```

### 3.1 음성 특징 추출

OpenAI Whisper 기준 음성 전처리는 다음 값을 사용합니다.

| 항목 | Whisper 기준 |
|---|---:|
| Sample rate | 16,000 Hz |
| 기본 오디오 창 | 30초 |
| 30초 sample 수 | 480,000 |
| FFT 크기 | 400 sample = 25 ms |
| Hop length | 160 sample = 10 ms |
| Mel channel | `tiny`, `small` 모델은 80 |
| 특징 | Log-Mel Spectrogram |

입력 waveform은 Short-Time Fourier Transform과 Mel filter bank를 거쳐
`(80, time_frames)` 형태의 Log-Mel Spectrogram으로 변환됩니다. 현재
프로젝트는 이 변환 수식과 Mel filter bank를 수정하지 않았습니다.

### 3.2 Audio Encoder

Audio Encoder는 Log-Mel 특징을 음성의 문맥 표현으로 변환합니다.

1. Convolution layer가 인접한 시간 특징을 결합합니다.
2. Positional encoding으로 음성 프레임의 순서를 표현합니다.
3. Transformer self-attention block이 전체 음성 문맥을 분석합니다.
4. Encoder의 출력은 음성 문맥을 담은 hidden representation입니다.

### 3.3 Text Decoder

Text Decoder는 Encoder 출력과 이전에 생성한 token을 사용해 다음 token을
예측합니다. 다음과 같은 특수 token이 함께 사용됩니다.

- 인식 시작 token
- 언어 token: 현재 시스템은 한국어 `ko`
- 작업 token: 번역이 아닌 `transcribe`
- Timestamp 사용 여부 token
- 무음 가능성 token
- 이전 문맥 또는 Initial Prompt token

Decoder는 autoregressive 방식이므로 token을 한 번에 모두 출력하지 않고,
이전 token을 참고해 다음 token을 반복 생성합니다.

### 3.4 현재 사용하는 모델 규모

| 용도 | 모델 | 약 parameter 수 | Encoder/Decoder layer | Width | Attention head |
|---|---|---:|---:|---:|---:|
| Partial | `tiny` | 39M | 4 | 384 | 6 |
| Final | `small` | 244M | 12 | 768 | 12 |

`tiny`는 실시간 화면 갱신 속도를, `small`은 Final 정확도를 우선합니다.
이 두 모델은 다시 학습하거나 layer를 변경한 모델이 아닙니다.

## 4. Whisper 모델을 가져오는 방법

### 4.1 라이브러리 설치

`requirements-core.txt`를 통해 `faster-whisper`와 `numpy`를 설치합니다.

```bash
pip install -r requirements-core.txt
```

`faster-whisper`는 OpenAI Whisper를 CTranslate2 추론 엔진으로 재구현한
라이브러리입니다. 현재 프로젝트는 `openai-whisper` Python package를
직접 import하지 않으며 `WhisperLive` package도 import하지 않습니다.

### 4.2 모델 이름에서 가중치까지

`stt/config.py`에서 역할별 모델 이름을 정의합니다.

```python
MODEL_SIZE = "small"
PARTIAL_MODEL_SIZE = "tiny"
```

`stt/model.py`에서 최초 `get_model()` 호출 시 다음을 실행합니다.

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    model_size,
    device=get_device(),
    compute_type=get_compute_type(),
)
```

`WhisperModel("small")`은 faster-whisper의 모델 mapping에 따라 Hugging Face의
`Systran/faster-whisper-small`을, `WhisperModel("tiny")`는
`Systran/faster-whisper-tiny`를 다운로드합니다. 다운로드된 가중치는
CTranslate2 형식이며 기본적으로 Hugging Face cache에 저장됩니다.

현재 Mac 개발 환경의 실제 cache는 다음과 같습니다.

```text
~/.cache/huggingface/hub/
├── models--Systran--faster-whisper-small    약 464 MB
└── models--Systran--faster-whisper-tiny     약 75 MB
```

모델 가중치는 Git 저장소에 포함되지 않습니다. Jetson에서 처음 실행할
때는 인터넷에서 각 모델을 다운로드하거나, 미리 다운로드한 로컬 CTranslate2
모델 경로를 사용해야 합니다.

### 4.3 장치와 연산 형식

| 실행 환경 | `device` | `compute_type` |
|---|---|---|
| CUDA 가용 | `cuda` | `int8_float16` |
| CUDA 불가 | `cpu` | `int8` |

이 변경은 Whisper 네트워크의 layer를 바꾸는 것이 아니라 실행 시 가중치와
연산 정밀도를 조정하는 추론 최적화입니다.

### 4.4 Singleton과 Lazy Loading

`ModelManager` 클래스는 모델을 최초 요청 시에만 생성하고 이후에는 같은
객체를 재사용합니다.

```text
첫 get_model() 호출
  → 모델 다운로드/로딩
  → 메모리에 보관

이후 get_model() 호출
  → 기존 객체 반환
```

Thread lock으로 동시 최초 로딩을 보호하며, `unload_model()` 호출 시
메모리 참조를 해제합니다. 메모리에서 해제해도 디스크 cache는 삭제하지
않으므로 다음 실행에서 재사용할 수 있습니다.

## 5. 원본 Whisper와 현재 STT 코드 비교

| 구분 | OpenAI Whisper 기본 사용 | 현재 프로젝트 | 변경 위치 |
|---|---|---|---|
| 실행체 | PyTorch `openai-whisper` | CTranslate2 `faster-whisper` | 추론 엔진 선택 |
| 모델 | 한 모델을 일반적으로 선택 | Partial `tiny`, Final `small` | 우리 모델 관리 코드 |
| 입력 | 음성 파일 또는 waveform | Beamforming mono chunk·PCM16 | 우리 입력 API |
| 실시간 방식 | 전체 파일을 30초 window로 처리 | 최근 4초 Partial + 전체 발화 Final | 우리 Rolling Buffer |
| 언어 | 자동 탐지 가능 | `ko` 고정 | 추론 option |
| 탐색 | 사용자 설정 | Partial beam 1, Final beam 3 | 추론 option |
| VAD | 선택적 | 적응형 RMS + Final VAD | 우리 전처리/추론 option |
| Prompt | 선택적 | Final에만 제한적 적용 | 추론 option |
| Hotwords | 선택적 | 실시간 모델에 미전달 | 우리 안전 정책 |
| Timestamp | 출력 가능 | `without_timestamps=True` | 추론 option |
| 출력 | text·segment·언어 정보 | `STTResult` callback | 우리 출력 API |
| 화자 식별 | Whisper 기본 기능 아님 | STT 출력에 화자 ID 없음 | DOA/UI에서 별도 처리 |
| Fine-tuning | 가능하지만 별도 학습 필요 | 실시하지 않음 | 모델 가중치 원본 유지 |

### 5.1 우리가 Whisper 내부에서 건드린 부분

**Whisper 네트워크 아키텍처 내부를 직접 수정한 부분은 없습니다.**

다만 Whisper/faster-whisper가 제공하는 추론 option을 다음과 같이
설정했습니다.

| option | Partial | Final | 이유 |
|---|---:|---:|---|
| `language` | `ko` | `ko` | 한국어 고정, 언어 탐지 비용 절감 |
| `beam_size` | 1 | 3 | Partial 속도, Final 정확도 절충 |
| `vad_filter` | `False` | `True` | Partial 반응성, Final 잡음 억제 |
| `initial_prompt` | 미사용 | 전문 문맥 | Partial 환각 방지 |
| `hotwords` | 미사용 | 미사용 | 잡음에서 전문용어 환각 방지 |
| `condition_on_previous_text` | `False` | `False` | 이전 오인식 누적 방지 |
| `without_timestamps` | `True` | `True` | 자막 문자열 생성에 집중 |
| `task` | `transcribe` | 기본 transcribe | 영어 번역이 아닌 한국어 복사 |

위 설정은 모델 parameter를 다시 학습하거나 영구적으로 바꾸는 것이
아닙니다. 각 추론 요청의 decoding 방법만 조정합니다.

### 5.2 Whisper 밖에서 직접 추가한 부분

1. **입력 정규화**: dtype·shape·sample rate 변환, DC 제거, peak 보정
2. **발화 감지**: 시작 1초 잡음 calibration과 적응형 RMS 문턱
3. **Partial 생성**: 0.25초 간격으로 최근 4초 음성 재추론
4. **Final 생성**: 침묵 또는 `flush()` 시 전체 발화 재추론
5. **중복 방지**: 서로 충돌하는 Partial을 이어 붙이지 않고 최신 결과로 교체
6. **안전 필터**: Final segment의 `avg_logprob`, `no_speech_prob` 확인
7. **환각 필터**: 음성 없이 BEM·MVDR·CTranslate2 등이 나열·반복되는 결과 제거
8. **모델 재사용**: Partial·Final 모델을 프로그램 실행 중 한 번만 로드
9. **비동기 처리**: audio queue와 inference worker로 마이크 callback 블로킹 방지
10. **결과 metadata**: 지연시간, 시각, 발화 ID, 순서 ID, 오류 원인 추가

## 6. 전체 시스템 데이터 흐름

```text
ReSpeaker 4채널 PCM
        ↓  [외부 모듈: STT 범위 아님]
DOA: 음원 방향 추정
        ↓  [외부 모듈: STT 범위 아님]
MVDR/BEM Beamforming: 특정 방향 음성 강조
        ↓
16 kHz mono waveform chunk
        ↓  [우리 코드]
입력 검증·resample·DC/peak 보정
        ↓  [우리 코드]
잡음 calibration·RMS 발화 감지·Rolling Buffer
        ↓  [faster-whisper]
Log-Mel 특징 추출
        ↓  [Whisper 모델]
Transformer Encoder–Decoder 추론
        ↓  [faster-whisper]
segment text·신뢰도·무음 확률
        ↓  [우리 코드]
신뢰도·환각 필터·Partial 중복 억제
        ↓
STTResult callback
        ↓  [UI]
화면 Partial 교체 / Final 자막 확정
```

DOA, Beamforming, 화자 ID, 카메라 사람 표시는 Whisper 모델의 입출력이
아닙니다. 이 정보는 외부 모듈과 UI가 별도로 결합해야 합니다.

## 7. STT 외부 입력 명세

### 7.1 권장 NumPy 입력

| 항목 | 명세 |
|---|---|
| API | `stt.push_audio(audio_chunk, sample_rate=16000)` |
| 자료형 | `numpy.ndarray` |
| dtype | `np.float32` 권장 |
| shape | `(N,)` mono |
| channel | Beamforming 완료 후 1채널 |
| sample rate | 16,000 Hz 권장 |
| 값 범위 | `-1.0` ~ `1.0` |
| chunk 길이 | 20~500 ms 권장 |
| 입력 순서 | 실제 녹음 시간 순서 |

16 kHz에서 100 ms chunk의 예시는 다음과 같습니다.

```python
audio_chunk.shape == (1600,)
audio_chunk.dtype == np.float32
```

호환 입력으로 `(N, 1)`, float64, int16 NumPy도 지원합니다. 입력 sample rate가
16 kHz가 아니면 NumPy 선형 보간으로 16 kHz로 변환합니다. NaN·Inf·임의
다채널 NumPy 입력은 `ValueError`입니다.

### 7.2 PCM16 bytes 입력

| 항목 | 명세 |
|---|---|
| API | `stt.push_pcm16(pcm_bytes, sample_rate=16000, channels=1)` |
| encoding | Signed 16-bit PCM |
| byte order | Little-endian |
| channel | 1채널 권장 |
| sample rate | 16,000 Hz 권장 |

다채널 PCM16은 `channels`를 명시하면 프레임별 평균으로 mono 변환하지만,
기본 계약은 Beamforming이 완료된 mono 입력입니다.

### 7.3 입력 전처리

Whisper에 전달하기 전 우리 코드가 다음을 처리합니다.

1. mono shape과 유한한 숫자인지 검증
2. float32 `-1.0~1.0` 기준 변환
3. 필요한 경우 16 kHz로 resample
4. 긴 chunk의 평균값을 빼 DC offset 제거
5. peak가 1.0을 넘으면 전체 chunk를 같은 비율로 축소
6. `-1.0~1.0` 범위로 clip
7. 시작 1초 배경 잡음 RMS calibration
8. `max(0.01, noise_floor × 2.5)`를 기본 발화 문턱으로 사용

시작 1초 calibration 동안은 배경음만 입력하는 것을 권장합니다.

## 8. Whisper 모델 직전 입력 명세

우리 코드는 faster-whisper에 다음 두 형태 중 하나를 전달합니다.

1. 16 kHz mono float32 NumPy waveform
2. 호환 필요 시 임시 mono PCM WAV 경로

faster-whisper 내부에서 waveform을 Log-Mel Spectrogram으로 변환한 후
CTranslate2 Whisper Encoder에 전달합니다. 즉, 외부 연동 모듈이 Mel
Spectrogram을 직접 만들 필요는 없습니다.

### Partial 입력

- 발화 중 최근 4초 waveform
- 기본 0.25초 간격으로 스냅샷 생성
- `tiny`, beam 1, VAD 비활성화
- Partial은 확정 결과가 아니므로 다음 추론에서 변경될 수 있음

### Final 입력

- 발화 시작부터 침묵 또는 `flush()`까지의 전체 waveform
- `small`, beam 3, VAD 활성화
- Final에만 프로젝트 전문 문맥 Initial Prompt 적용
- 화자·방향 전환 신호가 오면 외부에서 `flush()` 호출 권장

## 9. faster-whisper 원시 출력 명세

`model.transcribe()`는 대략 다음과 같은 두 값을 반환합니다.

```python
segments, transcription_info = model.transcribe(audio_input, **options)
```

### 9.1 Segment

`segments`는 lazy iterable/generator입니다. 각 segment에는 다음과 같은 정보가
포함될 수 있습니다.

| 필드 | 의미 | 현재 코드의 사용 |
|---|---|---|
| `text` | 인식 문자열 | 순서대로 결합 |
| `start`, `end` | segment 시간 범위 | UI 출력에는 미사용 |
| `tokens` | Decoder가 생성한 token ID | 직접 미사용 |
| `avg_logprob` | segment 평균 log probability | Final 신뢰도 필터 |
| `no_speech_prob` | 무음일 확률 | Final 신뢰도 필터 |
| `compression_ratio` | 반복 출력 판단에 사용 가능한 값 | 현재 직접 미사용 |

현재 Final 필터 기준은 다음과 같습니다.

```text
avg_logprob >= -1.2
no_speech_prob <= 0.8
```

기준을 통과한 `segment.text`를 순서대로 결합하고 앞뒤·중복 공백을
정리합니다. 한국어가 없는 상태에서 프롬프트 전문용어만 반복·나열된
결과는 빈 문자열로 처리합니다.

### 9.2 TranscriptionInfo

faster-whisper는 감지 언어, 언어 확률, 음성 길이, VAD 후 길이,
적용된 추론 option 등을 담은 `TranscriptionInfo`도 반환합니다. 현재
StreamingSTT API는 언어를 `ko`로 고정하므로 이 객체를 UI에 직접
전달하지 않습니다.

## 10. 현재 STT 공개 출력 명세

원시 segment를 정리한 후 UI·외부 모듈에는 변경 불가능한
`STTResult` 객체를 callback으로 전달합니다.

```python
@dataclass(frozen=True)
class STTResult:
    type: Literal["partial", "final", "error"]
    text: str
    latency_ms: float
    timestamp: str
    sequence_id: int
    utterance_id: int
    is_final: bool
    error: str | None = None
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `type` | `partial`, `final`, `error` | 결과 종류 |
| `text` | `str` | 인식된 한국어 자막; 오류·인식 없음은 `""` |
| `latency_ms` | `float` | 해당 Whisper 추론 소요 시간(ms) |
| `timestamp` | `str` | Timezone을 포함한 ISO 8601 생성 시각 |
| `sequence_id` | `int` | Callback 결과마다 증가하는 순번 |
| `utterance_id` | `int` | 발화마다 증가하는 ID |
| `is_final` | `bool` | Final 결과 또는 Final 오류 여부 |
| `error` | `str \| None` | 오류 원인; 정상이면 `None` |

### 10.1 Partial 출력 예시

```python
STTResult(
    type="partial",
    text="오늘 회의를",
    latency_ms=225.08,
    timestamp="2026-08-22T18:30:00+09:00",
    sequence_id=4,
    utterance_id=1,
    is_final=False,
    error=None,
)
```

Partial은 임시 결과입니다. UI는 이전 Partial을 아래에 쌓지 않고 같은 발화의
최신 `text`로 교체해야 합니다.

### 10.2 Final 출력 예시

```python
STTResult(
    type="final",
    text="오늘 회의를 시작하겠습니다.",
    latency_ms=1421.18,
    timestamp="2026-08-22T18:30:02+09:00",
    sequence_id=5,
    utterance_id=1,
    is_final=True,
    error=None,
)
```

Final은 해당 발화의 확정 결과입니다. UI는 임시 Partial을 제거하고 Final을
자막 기록에 한 번만 추가해야 합니다.

### 10.3 Error 출력 예시

```python
STTResult(
    type="error",
    text="",
    latency_ms=0.0,
    timestamp="2026-08-22T18:30:03+09:00",
    sequence_id=6,
    utterance_id=1,
    is_final=False,
    error="StreamingSTT 시작에 실패했습니다. ...",
)
```

## 11. Whisper가 출력하지 않는 정보

현재 Whisper/STT 출력만으로는 다음을 알 수 없습니다.

- 카메라에 나타난 사람의 좌표
- 말한 사람의 화자 ID
- 음원의 DOA 각도
- Beamforming 방향
- 실제 화자 이름
- 카메라 사람과 마이크 화자가 같은지의 연결 정보

따라서 UI의 `화자 1`, `화자 2`, `화자 3` 색상·테두리·활성 상태는
DOA·카메라·화자 매핑 모듈이 STT 결과와 결합해야 합니다. STT는 해당 화자의
음성이 올바른 순서로 입력된다는 가정 아래 자막 문자열만 생성합니다.

## 12. 학습·Fine-tuning 여부

현재 모델은 사전학습된 `tiny`, `small` 가중치를 그대로 사용합니다.

| 항목 | 현재 여부 |
|---|---|
| 추가 학습 | 안 함 |
| Fine-tuning | 안 함 |
| LoRA/Adapter | 안 함 |
| Tokenizer 확장 | 안 함 |
| Vocabulary 추가 | 안 함 |
| 모델 layer 변경 | 안 함 |
| Weight pruning | 안 함 |
| CTranslate2 양자화 추론 | 사용 |

`initial_prompt`, beam size, VAD, 신뢰도 필터는 학습이 아니라 **추론 설정
및 후처리**입니다.

## 13. 현재 구조의 장점과 제한

### 장점

- WhisperLive server·WebSocket 구조 없이 Jetson 프로세스 내부에서 직접 호출 가능
- DOA·Beamforming mono 출력과 간단한 `push_audio()` API로 연결
- `tiny` Partial과 `small` Final로 반응성·정확도 분리
- CTranslate2 양자화로 Jetson GPU 메모리 사용량 절감 가능
- 실시간 전문용어 환각과 Partial 중복 자막 억제

### 제한

- Whisper는 token 단위 autoregressive 모델이므로 완전한 frame-level streaming 모델은 아님
- Partial은 최근 음성을 반복 추론하는 방식이므로 계속 변경될 수 있음
- Final은 전체 발화를 재추론하므로 발화가 길수록 latency가 증가
- Whisper 기본 모델만으로는 화자 구분과 카메라 사람 매핑 불가
- 실제 Jetson·ReSpeaker·Beamforming 환경에서 다시 정확도·지연시간 측정 필요
- `requirements-core.txt`가 현재 정확한 package version을 고정하지 않아 최종 배포 전
  재현성을 위한 version pinning 필요

## 14. STT 성능 개선 과정과 정량적 결과

### 14.1 평가 조건과 주의사항

설정 변경에 따른 정확도와 속도를 비교하기 위해 macOS 한국어 TTS로 생성한
30문장을 사용했습니다.

| 항목 | 평가 조건 |
|---|---|
| 장치 | Mac CPU |
| 추론 라이브러리 | faster-whisper 1.2.1 |
| 연산 형식 | CPU `int8` |
| 언어 | `ko` |
| 음성 수 | 30개 |
| 총 음성 길이 | 103.775초 |
| 문장 구성 | 일반 10개, 전문용어 10개, 긴 문장·숫자·영문 혼합 10개 |
| 주 정확도 지표 | CER |
| 보조 지표 | WER, 전문용어 정확도, 문장 완전 일치율 |
| 속도 지표 | 평균·P95 Final latency, RTF |

합성 음성은 발음이 깨끗하기 때문에 실제 ReSpeaker·Beamforming 환경보다
성능이 높게 나올 수 있습니다. 따라서 아래 결과는 모델·설정 비교용
기준선이며 실제 Jetson 최종 성능을 보장하는 수치가 아닙니다. 또한 실행마다
CPU 상태가 다르므로 수십 ms 정도의 latency 차이는 측정 오차를 포함합니다.

### 14.2 초기 상태와 문제 인식

초기에는 Final `small`, beam size 5, Final Initial Prompt 설정을 사용했습니다.
전문용어는 잘못 인식되거나 일반 한국어로 변환되었고, Mac CPU에서 Final
자막이 확정되기까지 약 1.5초가 필요했습니다.

| 초기 기준선 | 결과 |
|---|---:|
| Raw CER | 13.16% |
| 전문용어 정규화 CER | 9.29% |
| 전문용어 정확도 | 72.73% |
| 평균 Final latency | 1,489.85 ms |
| RTF | 0.4307 |

`medium`, beam size 5도 비교했지만 합성 30문장에서 CER 17.10%, 평균
Final latency 4,066.47 ms, RTF 1.1756으로 측정됐습니다. 이 평가에서는
`medium`이 `small`보다 정확하지 않았고 약 2.73배 느렸으므로 Final 모델은
`small`로 유지했습니다.

### 14.3 1차 성능 개선

처음에는 정확도와 반응 속도를 높이기 위해 다음과 같이 변경했습니다.

1. Partial 전용 `tiny` 모델을 추가했습니다.
2. Partial은 beam 1, 0.25초 간격, 최근 4초 음성으로 설정했습니다.
3. Final `small`의 beam size를 5에서 3으로 줄였습니다.
4. CUDA에서 `int8_float16`, CPU에서 `int8`을 사용하도록 했습니다.
5. 전문용어 정확도를 높이기 위해 Initial Prompt와 Hotwords를 같이 적용했습니다.

| 항목 | 초기 | 1차 개선 | 변화 |
|---|---:|---:|---:|
| Raw CER | 13.16% | 11.26% | 1.90%p 개선 |
| 정규화 CER | 9.29% | 7.84% | 1.45%p 개선 |
| 전문용어 정확도 | 72.73% | 95.45% | 22.72%p 개선 |
| 평균 Final latency | 1,489.85 ms | 1,462.90 ms | 26.95 ms(1.81%) 감소 |
| RTF | 0.4307 | 0.4229 | 0.0078 감소 |

합성 음성 평가만 보면 1차 개선으로 정확도와 latency가 모두 개선된 것처럼
보였습니다. 특히 전문용어 정확도 95.45%는 유의미한 개선으로 판단되었습니다.

### 14.4 실제 UI에서 발견된 trouble

현장 UI에서 사람이 해당 단어를 말하지 않았는데도 다음과 같은 자막이
반복 표시되는 문제가 발생했습니다.

```text
BEM, MVDR, BEM ...
CTranslate2 ...
RTF, BEM ...
```

이 문제는 일반적인 문자 인코딩 오류가 아니라 **실제 음성에 없는 단어를
Whisper가 생성한 환각**이었습니다. 원인 우선순위는 다음과 같았습니다.

1. **Hotwords 과도한 편향**: 전문용어를 강제한 설정이 잡음에서도 해당 단어를 선택하게 했습니다.
2. **낮은 고정 RMS 문턱**: 고정값 0.01보다 큰 Beamforming 배경 잡음이 음성으로 판단됐습니다.
3. **Partial·Final VAD 동시 비활성화**: 추론 직전에 잡음 구간을 제거할 보조 장치가 없었습니다.
4. **Beamforming mono 신호 상태**: DC offset, 낮거나 과도한 음량, 잡음·클리핑을 STT 직전에서 보정하지 않았습니다.
5. **Final 신뢰도 필터 부재**: `avg_logprob`, `no_speech_prob`가 나빠도 문자열이 UI에 저장됐습니다.
6. **Partial 결과 병합 방식**: 서로 다른 임시 가설을 이어 붙여 중복·비문을 만들 수 있었습니다.

즉, 1차 개선의 높은 전문용어 정확도는 깨끗한 합성 음성에서는 이점이었지만,
실제 잡음 환경에서는 거짓 전문용어 자막을 만드는 위험이었습니다. 따라서
95.45%를 현장 신뢰성까지 반영한 최종 성능으로 보고할 수 없었습니다.

### 14.5 2차 수정: 실제 환경 안전성 개선

환각을 억제하면서 정상 한국어 음성을 보존하기 위해 다음을 수정했습니다.

1. 실시간 Partial·Final 모델에 `hotwords`를 전달하지 않도록 했습니다.
2. Initial Prompt는 전체 발화 문맥을 볼 수 있는 Final에만 제한적으로 사용했습니다.
3. 프로그램 시작 후 1초간 배경 잡음 RMS를 calibration하도록 했습니다.
4. 발화 문턱을 `max(0.01, noise_floor × 2.5)`로 자동 조정하도록 했습니다.
5. Partial VAD는 반응속도를 위해 끄고, Final VAD는 활성화했습니다.
6. Beamforming chunk에서 DC offset을 제거하고 peak 범위를 보정했습니다.
7. Final segment에 `avg_logprob >= -1.2`, `no_speech_prob <= 0.8` 필터를 적용했습니다.
8. 한국어 없이 프롬프트 전문용어만 반복·나열된 Final을 빈 결과로 처리했습니다.
9. 서로 다른 Partial 가설은 문자열로 이어 붙이지 않고 최신 가설로 교체했습니다.
10. 오프라인 평가에도 실제 StreamingSTT Final과 같은 VAD·신뢰도·환각 정책을 동일하게 적용했습니다.

이 작업도 Whisper Encoder·Decoder나 가중치를 변경한 것이 아니라 **Whisper 입력
전처리, 추론 option, 출력 후처리를 변경한 것**입니다.

### 14.6 2차 수정 후 검증 결과

#### 합성 한국어 30문장

| 항목 | 1차 정확도 우선 | 2차 현재 안전 설정 | 변화 |
|---|---:|---:|---:|
| Raw CER | 11.26% | 13.43% | 2.17%p 증가 |
| Raw WER | 18.14% | 18.63% | 0.49%p 증가 |
| 정규화 CER | 7.84% | 11.61% | 3.77%p 증가 |
| 전문용어 정확도 | 95.45% | 72.73% | 22.72%p 감소 |
| 문장 완전 일치율 | 46.67% | 46.67% | 변화 없음 |
| 평균 Final latency | 1,462.90 ms | 1,421.18 ms | 41.72 ms(2.85%) 감소 |
| P95 Final latency | 별도 기록 없음 | 1,575.89 ms | 현재 기준선 확보 |
| RTF | 0.4229 | 0.4108 | 0.0121(2.86%) 감소 |
| 빈 결과 | 0 | 0 | 변화 없음 |
| 프롬프트 전문용어 환각 | 현장에서 발생 | 0 | 실험 기준 억제 |

현재 설정은 깨끗한 합성 음성의 CER과 전문용어 정확도는 1차 설정보다
나빠졌습니다. 이는 Hotwords가 정답 전문용어를 강제로 선택하던 효과를 제거한
결과입니다. 반면 운영 중 잘못된 전문용어 자막이 누적되는 문제를 억제했고,
Final 평균 latency는 41.72 ms 감소했습니다.

#### 잡음·실제 모델 흐름 검증

| 검증 항목 | 결과 |
|---|---:|
| RMS 0.02 지속 Gaussian 잡음 입력 | Partial·Final 0건 |
| BEM·MVDR·CTranslate2 나열 환각 단위 테스트 | 차단 성공 |
| 정상 전문용어 WAV Final | 정답 문장 출력 |
| 정상 WAV Partial 평균 추론시간 | 약 225.08 ms |
| 정상 WAV Final 추론시간 | 약 1,680.05 ms |
| Callback 오류 후 worker 유지 | 성공 |
| 전체 자동화 테스트 | 33개 통과 |

정상 전문용어 WAV의 Final 출력은 다음과 같았습니다.

```text
Jetson Orin Nano에서 한국어 실시간 자막 시스템을 실행합니다.
```

### 14.7 결과 해석

이번 개선의 핵심은 **합성 음성의 최대 정확도보다 실제 운영 신뢰성을
우선한 것**입니다. Hotwords를 제거했기 때문에 전문용어 benchmark 수치는
낮아졌지만, 말하지 않은 전문용어가 UI 자막으로 쌓이는 더 큰 실패를
차단했습니다.

초기 기준선과 현재 안전 설정을 직접 비교하면 다음과 같습니다.

| 항목 | 초기 | 현재 | 최종 판단 |
|---|---:|---:|---|
| Raw CER | 13.16% | 13.43% | 0.27%p 증가; 비슷한 수준 |
| 전문용어 정확도 | 72.73% | 72.73% | 유지 |
| 평균 Final latency | 1,489.85 ms | 1,421.18 ms | 68.67 ms(4.61%) 감소 |
| RTF | 0.4307 | 0.4108 | 0.0199(4.62%) 감소 |
| 잡음 환각 방지 | 부족 | 적응형 RMS·VAD·필터 | 개선 |

따라서 현재 설정은 초기와 비슷한 정확도를 유지하면서 평균 추론시간을
줄이고, 실제 UI에서 확인된 전문용어 환각을 억제한 설정으로 평가할 수
있습니다. 다만 Final 0.5초 이내 목표는 Mac CPU에서는 달성하지 못했으며,
Jetson GPU와 실제 ReSpeaker·Beamforming 입력에서 P95 latency와 CER를 다시 측정해야
합니다.

### 14.8 보고서용 성능 개선 요약문

> 초기 STT는 faster-whisper `small`, beam size 5, Initial Prompt 설정으로
> 합성 한국어 30문장에서 CER 13.16%, 전문용어 정확도 72.73%, 평균 Final
> latency 1,489.85 ms를 기록하였다. 반응속도와 전문용어 인식률을 높이기
> 위해 `tiny` Partial, `small` Final, beam size 3, Initial Prompt와 Hotwords를 적용한
> 1차 개선을 수행하였다. 합성 음성에서 CER은 11.26%로 낮아지고 전문용어
> 정확도는 95.45%로 높아졌지만, 실제 UI에서 음성에 없는 BEM·MVDR·CTranslate2
> 등이 반복 출력되는 환각 문제가 발견되었다. 원인은 Hotwords 과도한 편향,
> 고정 RMS 문턱, Final VAD·신뢰도 필터 부재로 판단하였다. 2차 수정에서
> 실시간 Hotwords를 제거하고, 시작 1초 잡음 calibration, 적응형 RMS, Final VAD,
> Beamforming 신호 보정, 신뢰도·환각 필터를 적용하였다. 현재 설정은 CER
> 13.43%, 전문용어 정확도 72.73%, 평균 Final latency 1,421.18 ms, RTF 0.4108을
> 기록하였고 RMS 0.02 지속 잡음에서 잘못된 자막 0건과 전체 자동화 테스트
> 33개 통과를 확인하였다. 결과적으로 초기와 비슷한 정확도를 유지하면서 Final
> latency를 약 4.61% 줄이고 현장 자막 환각 안전성을 개선하였다.

## 15. 보고서용 최종 요약

> 본 시스템은 WhisperLive 소스코드를 직접 사용하지 않고,
> OpenAI Whisper 사전학습 모델을 CTranslate2 기반 faster-whisper로 실행하였다.
> Whisper의 Transformer Encoder–Decoder 구조, tokenizer, 학습 가중치는 수정하지
> 않았으며, Jetson Orin Nano와 DOA·Beamforming 출력에 맞추어 입력 정규화,
> Partial·Final 이중 추론, Singleton 모델 관리, 적응형 잡음 감지,
> VAD·신뢰도·환각 필터와 callback 인터페이스를 직접 구현하였다.
> STT 입력은 Beamforming을 완료한 16 kHz mono float32 waveform이고, 출력은
> 한국어 Partial·Final 문자열과 latency, timestamp, sequence·utterance ID를 포함한
> `STTResult`이다. 화자 ID와 카메라 좌표는 Whisper 출력이 아니며 외부
> DOA·카메라·UI 모듈에서 별도로 결합한다.

## 16. 참고 자료

- OpenAI Whisper 논문: [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
- OpenAI Whisper 공식 저장소: [openai/whisper](https://github.com/openai/whisper)
- OpenAI Whisper 오디오 전처리: [whisper/audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py)
- OpenAI Whisper 모델 구조: [whisper/model.py](https://github.com/openai/whisper/blob/main/whisper/model.py)
- OpenAI Whisper 모델 카드: [model-card.md](https://github.com/openai/whisper/blob/main/model-card.md)
- faster-whisper 공식 저장소: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- faster-whisper 모델 mapping·다운로드: [faster_whisper/utils.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py)

## 17. 관련 프로젝트 파일

- `stt/config.py`: 모델 크기, 언어, beam, 장치·연산 설정
- `stt/model.py`: Singleton·Lazy Loading·모델 해제
- `stt/streaming.py`: 입력 전처리, Partial·Final, Queue, VAD, 필터
- `stt/types.py`: `STTResult` 출력 계약
- `INTERFACE_CONTRACT.md`: 외부 모듈 연동 요약
