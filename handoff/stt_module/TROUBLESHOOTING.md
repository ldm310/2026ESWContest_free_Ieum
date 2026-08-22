# Troubleshooting

## 1. `No module named 'faster_whisper'`

원인: 현재 Python 환경에 핵심 패키지가 설치되지 않았거나 다른 인터프리터를 사용 중입니다.

```bash
python -m pip show faster-whisper
python -m pip install -r requirements-core.txt
python -c "import faster_whisper; print(faster_whisper.__version__)"
```

## 2. `No module named 'numpy'`

원인: NumPy가 현재 가상환경에 설치되지 않았습니다.

```bash
python -m pip show numpy
python -m pip install -r requirements-core.txt
```

## 3. 모델 최초 다운로드가 오래 걸림

`start()`는 최초 실행 시 모델을 내려받고 warm-up합니다. 네트워크와 디스크 상태에 따라 오래 걸릴 수 있습니다. 모델 캐시가 생성됐는지와 네트워크 연결을 확인하고 첫 실행을 완료하십시오. 진행 중 프로세스를 반복 종료하면 다음 실행에서 다시 확인 작업이 발생할 수 있습니다.

## 4. CUDA를 찾지 못함

현재 장치 선택 결과를 확인합니다.

```bash
python -c "from stt.config import get_device; print(get_device())"
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

`cpu`가 출력되면 현재 CTranslate2 환경에서 CUDA 장치를 사용할 수 없다는 뜻입니다. Windows 또는 Jetson의 드라이버·CUDA·CTranslate2 호환 조합은 해당 장치 환경에서 확인해야 하며 이 패키지는 특정 CUDA 버전을 가정하지 않습니다.

## 5. CTranslate2 compute type 오류

```bash
python -c "from stt.config import get_device, get_compute_type; print(get_device(), get_compute_type())"
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

CPU에서는 기본 `int8`, CUDA에서는 기본 `int8_float16`입니다. 설치된
CTranslate2와 장치가 해당 형식을 지원하는지 확인하십시오. 오류 전문을
보존한 뒤 환경 호환성을 먼저 점검합니다.

## 6. 입력 배열 shape 오류

권장 shape은 `(N,)`, 허용 가능한 추가 형태는 `(N, 1)`입니다.

```python
print(audio_chunk.shape)
```

`(N, 2)` 같은 다채널 NumPy 배열은 자동 평균하지 않습니다. Beamforming 또는 명시적인 mono 변환 후 입력하십시오.

## 7. 입력 배열 dtype 오류

```python
print(audio_chunk.dtype)
```

지원 dtype은 float32, float64, int16입니다. 권장 변환은 `audio_chunk.astype(np.float32)`이며 정수형을 직접 float로 바꿀 때는 값 범위도 함께 정규화해야 합니다.

## 8. NaN 또는 Inf 입력 오류

```python
print(np.all(np.isfinite(audio_chunk)))
```

DOA 또는 Beamforming 계산에서 0으로 나누기나 발산이 발생했는지 확인하십시오. STT 입력 직전에 유한 값만 전달해야 합니다.

## 9. sample rate 오류

```python
print(source_sample_rate)
```

샘플레이트는 0보다 큰 정수여야 하며 권장값은 16000Hz입니다. 다른 값은 선형 보간되지만 입력 메타데이터가 실제 음성과 다르면 속도와 인식 결과가 왜곡됩니다.

## 10. partial이 느림

```python
print(stt.get_stats())
```

`average_partial_latency_ms`, `skipped_partial_requests`, `max_audio_queue_size`를 확인하십시오. CPU 추론이 갱신 주기보다 느리면 오래된 partial snapshot은 최신 값으로 교체됩니다. 모델 크기, 장치 선택, 입력 preview 길이를 실제 장치 성능에 맞춰 검토하십시오.

## 11. final latency가 큼

final은 정확도를 위해 전체 발화를 다시 처리합니다. 발화가 길수록 지연시간이 증가합니다. `average_final_latency_ms`와 실제 발화 길이를 함께 확인하십시오. 이 모듈은 정확한 문장 보존을 위해 고정 최대 길이로 발화를 자르지 않습니다.

## 12. 결과 callback이 호출되지 않음

- `start()`가 성공했는지 확인
- 입력 RMS가 `silence_threshold`를 넘는지 확인
- partial은 `partial_interval`만큼 음성이 쌓였는지 확인
- final은 침묵을 기다리거나 `flush()`를 호출
- callback 내부 예외가 stderr에 기록됐는지 확인

```python
print(stt.get_stats())
```

## 13. 침묵 감지가 너무 빠름

사람 음성이 임계값 아래로 떨어지는 경우입니다. `silence_threshold`를 낮추거나 `silence_duration`을 늘리십시오. 환경 RMS를 측정한 뒤 작은 폭으로 조절하십시오.

## 14. 침묵 감지가 너무 늦음

배경 소음이 임계값보다 큰 경우입니다. `silence_threshold`를 높이거나 `silence_duration`을 줄이십시오. Beamforming 출력의 노이즈 레벨을 먼저 확인하십시오.

## 15. PowerShell 가상환경 실행 정책 오류

현재 PowerShell 프로세스에만 실행을 허용합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

시스템 전체 정책을 변경할 필요는 없습니다.
