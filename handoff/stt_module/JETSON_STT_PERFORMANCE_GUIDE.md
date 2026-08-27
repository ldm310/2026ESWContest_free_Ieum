# 준형님용 Jetson STT 성능 설정 가이드

이 문서는 Jetson Orin Nano에서 현재 한국어 실시간 자막 코드를 실행할 때
**정확도와 반응속도를 함께 확보하기 위한 설정·검증 순서**를 정리한다.
대상 코드는 GitHub `develop` 브랜치의 `handoff/jetson_ui`와
`handoff/stt_module`이다.

- 최종 업데이트: 2026-08-27
- 대상 장치: Jetson Orin Nano, ReSpeaker 4 Mic Array, 1024×600 모니터
- 현재 입력: 16kHz 4채널 → BEM/DOA/MVDR → mono float32 → StreamingSTT
- 중요: 이 문서의 Mac 수치는 코드 회귀 검사용이며 Jetson GPU 성능이 아니다.

## 1. 결론: 먼저 사용할 권장 조합

아래 값은 현재 코드에 이미 기본값으로 반영되어 있다. Jetson 첫 실행에서는
모델이나 beam 값을 임의로 바꾸지 말고 이 조합으로 기준 성능부터 측정한다.

| 구분 | 권장값 | 저장소에서 변경할 위치 | 목적 |
|---|---|---|---|
| Partial 모델 | `tiny` | `handoff/stt_module/stt/config.py`의 `PARTIAL_MODEL_SIZE` | 말하는 중 빠른 임시 자막 |
| Partial beam | `1` | `handoff/stt_module/stt/streaming.py`의 `PARTIAL_BEAM_SIZE` | 탐색량 최소화 |
| Partial 요청 간격 | `0.25초` | `handoff/stt_module/stt/streaming.py`의 `DEFAULT_PARTIAL_INTERVAL_SECONDS` | 빠른 화면 갱신 |
| Beamforming 전달 단위 | `0.25초` | `handoff/jetson_ui/runtime_protocol.py`의 `AUDIO_CHUNK_SECONDS` | upstream 0.5초 대기 제거 |
| Partial 오디오 범위 | 최근 `4초` | `handoff/stt_module/stt/streaming.py`의 `DEFAULT_PREVIEW_SECONDS` | 긴 발화의 반복 연산 제한 |
| Final 모델 | `small` | `handoff/stt_module/stt/config.py`의 `MODEL_SIZE` | 한국어 정확도와 Jetson 부하 절충 |
| Final beam | `3` | `handoff/stt_module/stt/config.py`의 `BEAM_SIZE` | beam 5보다 빠르고 beam 1보다 정확도 보존 |
| 언어 | `ko` | `handoff/stt_module/stt/config.py`의 `LANGUAGE` | 언어 감지 시간 제거 |
| GPU 연산 | `int8_float16` | `handoff/stt_module/stt/config.py`의 `get_device()`, `get_compute_type()` | GPU 메모리·속도 절충 |
| Final VAD | 사용 | `handoff/stt_module/stt/streaming.py`의 `STREAMING_FINAL_VAD_FILTER`, `STREAMING_FINAL_VAD_PARAMETERS` | 잡음 구간과 무음 환각 억제 |
| 문장 종료 | 외부 신호 즉시 `flush()` | `handoff/jetson_ui/runtime_protocol.py`, `stage6_caption_ui.py` | 내부 침묵 대기 제거 |
| 침묵 fallback | `0.45초` | `handoff/stt_module/stt/streaming.py`의 `DEFAULT_SILENCE_DURATION_SECONDS` | 외부 신호 누락 시에도 Final 생성 |
| 입력 | Beamforming 후 mono `float32`, 16kHz | `handoff/stt_module/stt/config.py`의 `SAMPLE_RATE`, `handoff/jetson_ui/realtime_doa.py`의 `SAMPLE_RATE`, `handoff/jetson_ui/stage6_caption_ui.py`의 `STT_SAMPLE_RATE` | Whisper 입력 계약 유지 |
| 모델 관리 | `tiny`, `small` 각각 1회 로딩 | `handoff/stt_module/stt/model.py`의 `ModelManager` | 매 발화 재로딩 방지 |

Jetson의 `~/project_main`에 전달본을 복사한 뒤에는 실제 실행 파일 경로가 다음과
같이 바뀐다.

| GitHub 저장소 원본 | Jetson 실제 실행 경로 |
|---|---|
| `handoff/stt_module/stt/config.py` | `~/project_main/stt/config.py` |
| `handoff/stt_module/stt/streaming.py` | `~/project_main/stt/streaming.py` |
| `handoff/stt_module/stt/model.py` | `~/project_main/stt/model.py` |
| `handoff/jetson_ui/realtime_doa.py` | `~/project_main/realtime_doa.py` |
| `handoff/jetson_ui/runtime_protocol.py` | `~/project_main/runtime_protocol.py` |
| `handoff/jetson_ui/stage6_caption_ui.py` | `~/project_main/stage6_caption_ui.py` |

정식 변경은 GitHub 저장소 원본에서 하고 커밋한 뒤 `~/project_main`에 다시
복사하는 것이 원칙이다. `~/project_main`만 직접 수정하면 다음 업데이트 때
덮어써질 수 있다. `ModelManager`의 Singleton 구조는 성능 설정값이 아니므로
일반적인 모델·beam 비교 과정에서는 수정하지 않는다.

샘플레이트는 세 파일의 값이 모두 일치해야 한다. 현재 파이프라인 계약은
16kHz로 확정되어 있으므로 모델 비교 과정에서는 변경하지 않는다.

현재 코드에는 다음 항목이 이미 구현되어 있으므로 같은 기능을 별도 프로세스나
새 모델 객체로 다시 만들지 않는다.

- `ModelManager`: `tiny`, `small` 모델을 실행 중 각각 한 번만 로드
- `StreamingSTT.start()`: 두 모델을 0.25초 무음으로 사전 warm-up
- `_InferenceScheduler`: 대기 Partial 폐기, Final 우선 처리
- `_AudioProcessor`: 문장 끝 무음에서 새 Partial 요청 차단
- `RealtimeReceiver`: 로컬 UDP `127.0.0.1:50009`의 문장 종료 신호 수신
- `RuntimeState`: Partial 화자를 유지하고 Final에서 활성 화자 표시 해제

Mac 합성 음성 30문장에서는 `medium + beam 5`가 `small + beam 5`보다
정확하지 않았고 약 2.73배 느렸다. 따라서 Jetson에서도 처음부터 `medium`이나
`large`로 올리지 않는다. 실제 ReSpeaker 녹음으로 CER가 개선되는 것이 확인될
때만 큰 모델을 후보로 삼는다.

## 2. 최신 develop 코드 준비

SSH로 Jetson에 접속한 뒤 Jetson Ubuntu 터미널에서 실행한다.

먼저 지연시간 개선 커밋이 `develop`에 병합됐는지 GitHub에서 확인한다. 아직
병합 전이면 임의로 오래된 `develop`을 실행하지 말고 팀에서 지정한
`feature/dongmin` 커밋을 사용한다. 브랜치 이름보다 **실제 커밋과 아래 상수값**을
확인하는 것이 중요하다.

```bash
cd ~/Embedded-project
git status
git switch develop
git pull origin develop
git log -1 --oneline
```

`git status`에 준형님이 직접 수정한 파일이 표시되면 먼저 커밋하거나 별도로
보관한다. 변경사항이 있는 상태에서 강제로 덮어쓰지 않는다.

받은 코드가 이번 최종본인지 확인한다.

```bash
cd ~/Embedded-project
grep -n "AUDIO_CHUNK_SECONDS = 0.25" \
  handoff/jetson_ui/runtime_protocol.py
grep -n "SENTENCE_END_UDP_ADDR" \
  handoff/jetson_ui/runtime_protocol.py
grep -n "if is_voice and self._samples_since_partial" \
  handoff/stt_module/stt/streaming.py
```

세 명령이 모두 해당 줄을 출력해야 한다. 나오지 않으면 이번 지연시간 개선
코드가 포함되지 않은 브랜치 또는 커밋이다.

최신 전달본을 실제 실행 폴더에 적용한다.

`~/project_main`에 준형님이 직접 수정한 DOA·Beamforming 코드가 있다면 먼저
아래 세 파일의 차이를 확인한다. 전달본 복사는 같은 이름의 파일을 덮어쓰므로
검토 없이 실행하지 않는다.

```bash
diff -u ~/project_main/realtime_doa.py \
  ~/Embedded-project/handoff/jetson_ui/realtime_doa.py || true
diff -u ~/project_main/runtime_protocol.py \
  ~/Embedded-project/handoff/jetson_ui/runtime_protocol.py || true
diff -u ~/project_main/stage6_caption_ui.py \
  ~/Embedded-project/handoff/jetson_ui/stage6_caption_ui.py || true
```

필요하면 기존 파일을 먼저 백업한다.

```bash
mkdir -p ~/project_main/backup_before_stt_update
cp -p ~/project_main/realtime_doa.py \
  ~/project_main/runtime_protocol.py \
  ~/project_main/stage6_caption_ui.py \
  ~/project_main/backup_before_stt_update/ 2>/dev/null || true
```

```bash
mkdir -p ~/project_main
cp -R ~/Embedded-project/handoff/jetson_ui/. ~/project_main/
cp -R ~/Embedded-project/handoff/stt_module/stt ~/project_main/
cp ~/Embedded-project/handoff/stt_module/requirements-core.txt \
  ~/project_main/requirements-stt.txt
```

복사 후 실행 폴더도 같은 값인지 다시 확인한다.

```bash
grep -n "AUDIO_CHUNK_SECONDS = 0.25" ~/project_main/runtime_protocol.py
grep -n "SENTENCE_END_UDP_ADDR" ~/project_main/runtime_protocol.py
```

다음 파일은 Git에 없으므로 별도로 준비해야 한다.

```bash
test -f ~/project_main/bem_table_reduced.h5 \
  && echo "BEM 테이블 확인" \
  || echo "BEM 테이블이 없습니다"
```

## 3. Jetson 전력·냉각 설정

추론 속도 비교 중에는 전력 모드와 클럭이 달라지지 않게 고정한다. 모드 번호는
JetPack과 보드 설정에 따라 다를 수 있으므로 `0`이라고 가정하지 않는다.

```bash
sudo /usr/sbin/nvpmodel -q
```

출력에서 `MAXN` 또는 `MAXN SUPER`의 실제 ID를 확인한 다음 적용한다.

```bash
MAXN_MODE_ID=0  # 위 출력에 표시된 실제 ID로 반드시 변경
sudo /usr/sbin/nvpmodel -m "$MAXN_MODE_ID"
sudo jetson_clocks
sudo jetson_clocks --show
```

`jetson_clocks`는 일정한 최대 성능으로 벤치마크할 때 사용한다. 전력 소모와
발열이 증가하므로 정격 전원과 활성 냉각 팬이 필요하다. 온도가 계속 상승하거나
클럭이 낮아지면 모델 설정 전에 냉각부터 해결한다.

두 번째 SSH 터미널에서 자원 상태를 계속 확인한다.

```bash
tegrastats --interval 1000
```

확인 항목은 RAM 사용량, GPU 사용률, 온도, 스로틀링과 SWAP 사용량이다. SWAP이
계속 증가하면 지연시간 변동이 커질 수 있으므로 다른 무거운 프로그램을 먼저
종료한다.

현재 프로세스는 Partial `tiny`와 Final `small` 모델을 동시에 메모리에 유지하고,
Chromium UI·카메라·BEM/MVDR도 함께 실행한다. 모델 로딩 직후와 10문장 처리 후
두 번 이상 `tegrastats`를 기록한다. 다음 상태에서는 모델 크기를 올리지 않는다.

- RAM이 지속적으로 부족하거나 SWAP이 증가함
- 온도 상승 뒤 GPU/CPU 클럭이 내려감
- GPU 사용률은 낮은데 CPU 한 코어가 계속 포화됨
- `realtime_doa.py`에서 오래된 chunk 폐기 경고가 반복됨

발표용으로 전력 모드를 고정했다면 재부팅 후 설정이 유지됐다고 가정하지 말고
매 실행 전에 `nvpmodel -q`와 `jetson_clocks --show`를 다시 확인한다.

## 4. 가상환경과 CUDA 확인

기본 설치 절차는 `JETSON_UBUNTU_SSH_SETUP.md`를 따른다.

```bash
cd ~/project_main
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
```

최신 faster-whisper/CTranslate2는 GPU 실행 시 CUDA 12와 cuDNN 9 조합을
기본 대상으로 한다. JetPack의 CUDA·cuDNN 버전과 설치된 CTranslate2가 맞지
않으면 GPU가 있어도 모델 로딩이 실패할 수 있다. 버전을 임의로 섞기 전에
아래 결과를 먼저 기록한다.

```bash
python -c "import faster_whisper, ctranslate2; print('faster-whisper', faster_whisper.__version__); print('CTranslate2', ctranslate2.__version__)"
python -c "import numpy, sounddevice, h5py, scipy; print('필수 import 성공')"
nvcc --version || true
dpkg -l | grep -E 'cudnn|cuda' | head -30
```

현재 코드가 실제로 CUDA와 `int8_float16`을 선택하는지 확인한다.

```bash
python - <<'PY'
import ctranslate2
from stt.config import get_compute_type, get_device

device = get_device()
compute_type = get_compute_type()
print("CUDA device count:", ctranslate2.get_cuda_device_count())
print("selected device:", device)
print("selected compute type:", compute_type)
if device == "cuda":
    supported = ctranslate2.get_supported_compute_types("cuda")
    print("supported CUDA compute types:", sorted(supported))
    if "int8_float16" not in supported:
        raise SystemExit("이 Jetson 런타임은 int8_float16을 지원하지 않습니다.")
else:
    raise SystemExit("CUDA를 인식하지 못해 CPU로 설정되었습니다.")
PY
```

정상 기준은 다음과 같다.

```text
CUDA device count: 1 이상
selected device: cuda
selected compute type: int8_float16
supported CUDA compute types: ... int8_float16 ...
```

`cpu int8`로 나오면 그대로 성능 측정을 진행하지 않는다. JetPack, CUDA,
cuDNN 및 AArch64용 CTranslate2 설치 상태를 먼저 해결해야 한다.

설치가 성공했더라도 실제 모델 생성 단계에서 CUDA library 오류가 날 수 있다.
따라서 import 성공과 `get_device()` 출력만으로 완료 처리하지 말고 다음 절의
`tiny`·`small` 실제 로딩까지 확인한다. 패키지 버전은 Jetson에서 정상 로딩된
조합을 기록하고 발표 직전에 전체 업그레이드하지 않는다.

## 5. 모델 사전 다운로드와 warm-up

`tiny`와 `small`은 첫 실행 때 Hugging Face에서 CTranslate2 변환 모델을 받을 수
있다. 발표 직전에 다운로드하지 말고 인터넷이 연결된 준비 단계에서 두 모델을
미리 로드한다.

```bash
cd ~/project_main
source .venv/bin/activate
CT2_VERBOSE=1 python - <<'PY'
from stt.model import ModelManager

final_model = ModelManager.get_model()
partial_model = ModelManager.get_partial_model()
print("Final small 모델 준비 완료:", type(final_model).__name__)
print("Partial tiny 모델 준비 완료:", type(partial_model).__name__)
ModelManager.unload_model()
PY
```

실제 프로그램의 `StreamingSTT.start()`도 두 모델을 다시 메모리에 한 번만 올리고
0.25초 무음으로 warm-up한다. 프로그램 실행 직후 모델 준비가 끝나기 전에 말한
결과는 성능 측정에서 제외한다.

모델 다운로드 캐시는 Jetson 사용자 계정에 저장되므로 다른 계정이나 `sudo`로
실행하면 다시 다운로드할 수 있다. 실제 발표 계정과 동일한 사용자로 사전
다운로드한다. 인터넷이 끊긴 현장에서도 시작되는지 네트워크를 끈 상태로 한 번
재실행해 확인한다.

프로그램 시작 시 모델 로딩과 warm-up이 끝난 뒤에야 UI 수신·카메라 단계가
시작된다. 첫 화면이 늦게 보이더라도 여러 번 실행하지 말고 터미널의 모델 로딩
오류를 먼저 확인한다. 성능 측정에서는 다음을 제외한다.

- 모델 다운로드가 발생한 첫 실행
- `StreamingSTT.start()` warm-up 시간
- 첫 번째 사용자 발화에 초기화 시간이 섞였다고 의심되는 결과

## 6. ReSpeaker·Beamforming 입력 확인

모델 크기를 바꾸는 것보다 입력 계약을 지키는 것이 우선이다.

```text
ReSpeaker 4채널 음성
  → DOA/BEM
  → MVDR Beamforming
  → mono float32, 16000Hz, -1.0~1.0
  → StreamingSTT.push_audio()
```

현재 `realtime_doa.py`는 입력 장치가 최소 6채널을 제공한다고 가정하고 채널
`[1, 2, 3, 4]`를 실제 마이크로 선택한다. 장치 번호와 채널 수를 확인한다.

```bash
cd ~/project_main
source .venv/bin/activate
python realtime_doa.py --list-devices
```

반드시 확인할 사항:

- 선택한 ReSpeaker 장치의 입력 채널이 6개 이상인지 확인한다.
- 실제 마이크 배선 순서가 코드의 `[1, 2, 3, 4]`와 일치해야 한다.
- `bem_table_reduced.h5`가 현재 박스·마이크 위치와 같은 조건에서 생성됐는지
  확인한다.
- Beamforming mono에 `NaN`, `Inf`, 심한 clipping 또는 큰 DC offset이 없어야
  한다.
- 프로그램 시작 후 첫 1초는 주변 배경 잡음만 입력한다. 이 구간은 적응형 RMS
  발화 문턱을 보정하는 시간이다.
- 시작 1초 안에 바로 말하면 첫 음절이 누락되거나 잡음 문턱이 잘못 잡힐 수 있다.

이번 최종 코드의 Beamforming 전달 단위는 `0.25초`, 대기 Queue는 최대 2개이므로
과부하가 발생하면 약 0.5초보다 오래된 입력을 버리고 최신 입력을 유지한다.
DOA·BEM·MVDR 한 번의 처리시간이 지속적으로 250ms를 넘으면 정확한 자막 입력이
유실될 수 있다. 터미널 로그와 함께 다음을 확인한다.

```text
목표: DOA·BEM·MVDR chunk 처리 P95 < 250ms
목표: 오래된 chunk 폐기 경고 0건
목표: callback overflow 0건
```

`0.25초`로 줄인 뒤 DOA 방향이 불안정하거나 음성이 끊기면 즉시 Queue를 키우지
말고, 같은 10문장으로 `0.25초`와 이전 `0.5초`를 A/B 비교한다. 실시간 목표를
위해서는 0.25초가 우선이지만 DOA 정확도 또는 오디오 연속성이 무너지면 준형님
모듈의 연산 최적화가 먼저 필요하다.

지속적으로 다음 메시지가 나오면 STT보다 DOA·MVDR 처리가 입력 속도를 따라가지
못하는 상태다.

```text
[오디오 경고] 처리 지연으로 가장 오래된 chunk를 버리고 최신 입력을 유지합니다.
```

이때 Queue 크기부터 늘리면 과거 음성이 쌓여 자막이 더 늦어진다. 먼저 GPU/CPU
사용률, BEM 계산시간과 발열을 확인한다.

## 7. 권장 실행 명령

장치 이름 또는 번호를 확인한 뒤 실행한다.

```bash
cd ~/project_main
source .venv/bin/activate
export PYTHON_BIN="$PWD/.venv/bin/python"
export AUDIO_DEVICE="ReSpeaker"
export CAMERA_MODE="auto"
export UVC_DEVICE="0"
export BEM_TABLE="$PWD/bem_table_reduced.h5"
./run_jetson_ui.sh
```

시작 후 다음 순서로 확인한다.

1. 마이크·카메라 상태 원이 초록색인지 확인한다.
2. `stt_ready`가 된 뒤 1초 동안 말하지 않고 배경 잡음을 보정한다.
3. 한 사람이 2~5초 길이의 한국어 문장을 말한다.
4. 말하는 중 Partial이 기존 문장을 이어 붙이지 않고 교체되는지 확인한다.
5. 문장 종료 신호를 보냈을 때 터미널에 아래 로그가 한 번 나오는지 확인한다.

   ```text
   [문장 종료] 외부 신호를 받아 Final 처리를 시작합니다.
   ```

6. 말이 끝난 뒤 Final이 한 번만 추가되고 마지막 음절이 잘리지 않는지 확인한다.
7. 외부 신호를 일부러 보내지 않은 문장도 0.45초 침묵 fallback으로 Final이
   생성되는지 확인한다.
8. 다른 화자가 말할 때 활성 화자 테두리와 자막 색이 함께 바뀌는지 확인한다.
9. `Ctrl+C` 종료 시 traceback 없이 두 프로세스가 종료되는지 확인한다.

## 8. 0.5초 목표에 대한 현재 구조의 주의점

현재 `latency_ms`는 **Whisper 추론 함수가 실행된 시간**이다. 사용자가 말을
끝낸 시점부터 화면에 표시될 때까지의 전체 시간은 다음 네 구간을 포함한다.

```text
발화 종료 검출 대기 + 입력/추론 Queue 대기 + Whisper 추론 + UI 전달
```

현재 통합 코드에는 다음 지연시간 개선이 반영되어 있다.

1. `realtime_doa.py`는 `CHUNK_SEC=0.25초`로 Beamforming 결과를 보낸다.
2. `stage6_caption_ui.py`는 로컬 UDP 문장 종료 신호를 수신해 동일한
   `StreamingSTT.flush()`를 즉시 호출한다.
3. 문장 끝 무음에서는 새 Partial 요청을 만들지 않아 Final worker 선점을 줄인다.
4. 외부 신호가 없을 때는 기존 0.45초 침묵 검사를 안전한 fallback으로 사용한다.

준형님 모듈의 문장 종료 지점에는 다음 전송 코드를 연결해야 한다.

```python
import socket

from runtime_protocol import SENTENCE_END_UDP_ADDR, pack_sentence_end

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(pack_sentence_end(), SENTENCE_END_UDP_ADDR)
```

실제 통합에서는 문장마다 socket을 새로 만들지 말고 프로세스 시작 시 한 번 만든
socket을 재사용한다. 종료 시점마다 `sendto()`만 한 번 호출한다.

외부 종료 신호 계약은 다음을 반드시 지켜야 한다.

1. 문장 중간의 짧은 쉼에는 보내지 않는다.
2. 한 utterance에 정확히 한 번만 보낸다.
3. 해당 문장의 마지막 Beamforming 오디오 chunk를 모두 전송한 **뒤** 보낸다.
4. 신호 payload나 port를 문자열로 직접 복제하지 말고 `runtime_protocol.py`의
   `pack_sentence_end()`와 `SENTENCE_END_UDP_ADDR`를 import한다.
5. 전송지는 loopback `127.0.0.1:50009`로 유지한다. 외부 네트워크에 공개할
   필요가 없다.
6. 문장 종료 검출기가 비활성화되거나 실패해도 신호를 임의 반복 전송하지 않는다.
   신호가 없으면 STT 내부 0.45초 침묵 fallback이 처리한다.

오디오와 종료 신호는 서로 다른 UDP port와 수신 thread를 사용한다. localhost에서
동작하더라도 마지막 오디오보다 종료 신호가 먼저 처리되면 마지막 음절이 다음
utterance로 넘어가거나 잘릴 수 있다. Jetson 실측에서 마지막 음절 누락이 보이면
송신 순서를 먼저 확인한다. 그래도 재현되면 종료 패킷에 마지막 `sequence_id`를
포함하고 UI가 해당 audio sequence 처리 완료 후 flush하는 protocol 보강이
필요하다. 현재 패킷에는 sequence barrier가 없다.

수신 경로만 수동으로 확인할 때는 UI가 실행 중인 별도 SSH 터미널에서 다음을
한 번 실행한다. 이 명령은 문장 종료 검출기를 대신하는 최종 운용 방식이 아니라
UDP 연결 확인용이다.

```bash
cd ~/project_main
source .venv/bin/activate
python -c "import socket; from runtime_protocol import SENTENCE_END_UDP_ADDR, pack_sentence_end; sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.sendto(pack_sentence_end(), SENTENCE_END_UDP_ADDR); sock.close()"
```

코드 경로가 준비되었더라도 **Partial·Final 전체 체감시간 0.5초 이내는 Jetson
실측 전까지 보장할 수 없다.** DOA 정확도와 오디오 유실이 없는지 확인하고,
Partial 추론이 이미 실행 중이면 해당 호출은 강제 중단할 수 없으므로 Partial
P95가 0.25초 요청 간격보다 긴지도 반드시 기록한다.

## 9. Jetson에서 기록할 합격 기준

모델 로딩이 포함된 첫 결과와 warm-up 결과는 제외하고 동일 문장 10개 이상을
측정한다. 평균만 보지 말고 P95와 실패 건수를 함께 남긴다.

| 항목 | 1차 목표 | 측정 방법 |
|---|---:|---|
| Partial 추론 P95 | `500ms 이하` | `STTResult.latency_ms` |
| Final 추론 P95 | `500ms 이하` | `STTResult.latency_ms` |
| 체감 Final P95 | `500ms 이하` 목표 | 발화 종료 신호부터 UI 표시까지 별도 측정 |
| Beamforming chunk 처리 P95 | `250ms 미만` | chunk 처리 시작·전송 시간 로그 |
| 종료 신호 처리 | utterance당 `1회` | `[문장 종료]` 로그 개수 |
| RTF | `1.0 미만` | 총 추론시간 / 총 음성길이 |
| 일반 한국어 CER | 이전 설정 이하 | 동일 실제 녹음과 정답으로 비교 |
| 빈 Final | `0건` 목표 | 10개 이상 발화 |
| 잡음 환각 | `0건` 목표 | 배경 잡음만 1분 입력 |
| 중복 Final | `0건` | utterance별 Final 개수 확인 |
| 마지막 음절 누락 | `0건` | 외부 flush 사용 10문장 직접 확인 |
| Queue drop/overflow | `0건` | 터미널 경고 확인 |

`latency_ms <= 500`만 만족해도 침묵 대기와 upstream chunk 때문에 체감시간은
500ms를 넘을 수 있다. 보고서에는 추론 지연과 end-to-caption 지연을 분리해서
기록한다.

각 측정에는 최소한 다음 환경 정보를 함께 남긴다. 환경이 다르면 숫자를 직접
비교하지 않는다.

```text
Git commit / JetPack / CUDA / CTranslate2 / faster-whisper
nvpmodel mode / jetson_clocks 사용 여부 / 온도
모델·beam·compute type / 문장 ID / 음성 길이
Partial latency / Final latency / end-to-caption latency / CER
Queue drop / overflow / 빈 결과 / 중복 결과
```

실제 녹음과 정답 JSON이 준비돼 있으면 같은 Jetson에서 CER/WER/RTF를 계산한다.

```bash
cd ~/Embedded-project/handoff/stt_module
~/project_main/.venv/bin/python -m evaluation.evaluate_dataset \
  --manifest /절대/경로/ground_truth.json \
  --model-size small \
  --output evaluation/results/jetson_small.json
```

## 10. 결과에 따른 조정 순서

한 번에 하나만 바꾸고 같은 WAV와 같은 전력 모드로 재측정한다.

### A. CUDA가 잡히지 않을 때

코드 설정을 바꾸지 말고 JetPack·CUDA·cuDNN·CTranslate2 호환부터 해결한다.
CPU `int8` 결과는 Jetson GPU 성능 결과로 기록하지 않는다.

### B. Partial이 500ms보다 느릴 때

1. `tiny`, beam 1이 실제로 사용되는지 확인한다.
2. GPU 스로틀링과 Queue drop을 확인한다.
3. Partial P95가 250ms보다 길면 요청 간격 `0.25초`가 과도할 수 있다.
   `0.35초`와 A/B 비교해 대기 요청과 Final 간섭이 줄어드는지 본다.
4. 현재 Beamforming 전달값 `CHUNK_SEC=0.25`가 실제로 적용됐는지 확인한다.
5. DOA·MVDR chunk 처리 P95가 250ms를 넘으면 STT 주기보다 upstream을 먼저
   최적화한다.

### C. Final이 500ms보다 느릴 때

1. 먼저 외부 문장 종료 신호에서 즉시 `flush()`되는지 확인한다.
2. `small + beam 3`의 추론 P95를 측정한다.
3. 초과하면 같은 실제 WAV로 `beam 1`을 시험한다.
4. beam 1의 CER 증가가 1%p 이내일 때만 속도 우선 설정으로 채택한다.
5. 그래도 느리면 `base + beam 3`을 후보로 측정하되 CER 저하를 함께 기록한다.

`beam 1` 또는 `base`는 자동 채택값이 아니다. 동일한 실제 ReSpeaker 평가셋에서
Final P95 개선과 CER 변화를 모두 확인한 뒤 선택한다. 여러 값을 동시에 바꾸면
어떤 변경이 정확도 또는 속도에 영향을 줬는지 알 수 없다.

### D. 정확도가 낮을 때

1. 모델부터 키우지 말고 BEM 테이블, 채널 순서, 음량, clipping과 DOA를 확인한다.
2. Beamforming 전 mono와 Beamforming 후 mono를 같은 문장으로 비교한다.
3. 일반 한국어 실제 녹음의 CER를 주 지표로 사용한다.
4. Final `initial_prompt`는 현재처럼 짧게 유지한다.
5. 실시간 `hotwords`는 다시 켜지 않는다. 잡음에서 BEM·MVDR 등의 단어가
   강제로 출력되는 문제가 이미 확인됐다.
6. `medium`은 실제 Jetson CER 개선이 수치로 확인된 경우에만 채택한다.

### E. 잡음이 자막으로 나올 때

1. 시작 1초가 실제 배경 잡음 구간인지 확인한다.
2. Beamforming 출력 RMS와 peak를 확인한다.
3. Final VAD와 현재 환각 필터를 끄지 않는다.
4. 지속 오검출이면 `noise_threshold_multiplier`를 `2.5 → 3.0` 후보로 시험한다.
5. 작은 목소리가 누락되면 반대로 `2.5 → 2.0` 후보를 시험한다.

문턱값은 현장 소음과 마이크 gain에 따라 달라지므로 고정 정답이 아니다. 변경
전후에 일반 음성 누락률과 잡음 환각률을 함께 측정해야 한다.

### F. 통합 단계에서 증상이 발생할 때

| 증상 | 먼저 확인할 항목 | 금지할 임시 대응 |
|---|---|---|
| `[문장 종료]` 로그가 없음 | 준형님 모듈의 `sendto()`, port 50009, 최신 `runtime_protocol.py` | 침묵시간을 바로 0.1초로 낮추기 |
| Final이 두 번 보임 | utterance당 종료 신호 전송 횟수, UI 중복 프로세스 | 결과 문자열만 임의 삭제하기 |
| 마지막 음절이 잘림 | 마지막 audio 전송 후 종료 신호 순서, sequence 처리 | pre-roll만 무작정 늘리기 |
| Partial은 빠른데 Final이 늦음 | Final P95, 실행 중 Partial, beam, CUDA, 외부 flush | Partial 주기를 0.15초로 더 줄이기 |
| 자막이 과거 음성을 따라감 | chunk 폐기 경고, DOA 처리 P95, Queue 크기 | Queue를 크게 늘리기 |
| `cpu int8`로 실행됨 | CTranslate2 CUDA 인식과 설치 조합 | CPU 결과를 Jetson GPU 결과로 제출하기 |
| CUDA OOM·프로세스 종료 | tiny+small+Chromium 전체 RAM, SWAP, 다른 프로세스 | 곧바로 medium·large 모델 사용하기 |
| 잡음에서 기술용어 반복 | 실시간 hotwords 비활성화, Final VAD, 잡음 보정 | hotwords 목록을 더 늘리기 |

동일 프로그램이 두 번 실행되면 UDP port bind 오류 또는 중복 자막이 생길 수 있다.
재실행 전 남아 있는 프로세스를 확인한다.

```bash
pgrep -af 'realtime_doa.py|stage6_caption_ui.py'
ss -lunp | grep -E '50007|50008|50009' || true
```

## 11. 음성 데이터를 더 추가해야 하는가

**추가하는 것이 좋다.** 다만 WAV 파일을 폴더에 넣는 것만으로 Whisper 모델의
정확도가 자동으로 올라가지는 않는다. 현재 코드는 사전학습된 faster-whisper
모델을 추론에만 사용하므로 새 음성을 스스로 학습하지 않는다.

추가 데이터의 첫 번째 목적은 다음과 같다.

1. 같은 음성으로 모델·beam·VAD·RMS 문턱을 공정하게 비교한다.
2. ReSpeaker, 거리, 방향, 소음과 Beamforming이 CER에 미치는 영향을 찾는다.
3. 자주 틀리는 일반 한국어 표현을 확인해 입력 처리와 설정을 조정한다.
4. 데이터가 충분히 쌓인 뒤 Fine-tuning 필요 여부를 판단한다.

현재 프로젝트는 전문용어보다 일반 한국어가 중요하므로 첫 실제 평가셋은 다음
비율을 권장한다.

| 문장 종류 | 문장 수 | 예시 목적 |
|---|---:|---|
| 일반 대화·회의 문장 | 20개 | 일상적인 한국어 정확도 평가 |
| 긴 문장·숫자·영문 혼합 | 5개 | 문장 길이와 혼합 표기 오류 확인 |
| 프로젝트 용어 포함 | 5개 | Jetson, DOA 등 최소 필수 용어 확인 |
| 합계 | 30개 | 첫 기준선 |

한 사람이 30문장만 녹음하는 것보다 **3명이 같은 30문장을 말해 총 90개**를
만드는 것이 좋다. 각 화자의 파일에는 정면·측면, 0.5m·1m·2m, 조용한 환경·
생활 소음 환경이 골고루 포함되게 한다. 시간이 부족하면 먼저 30개로 파이프라인을
확인하고, 최종 판단 전 90개까지 늘린다.

같은 발화에 대해 가능하면 다음 두 파일을 함께 보관한다.

- ReSpeaker 원본 다채널 음성: DOA·Beamforming 문제 재현용
- BEM/MVDR 처리 후 16kHz mono 음성: 실제 STT 성능 평가용

정답표에는 최소한 다음 정보를 기록한다. `category` 외의 환경 정보는 현재
평가기가 무시하더라도 나중에 거리·방향별 결과를 분석할 때 필요하다.

```json
{
  "id": "speaker01_front_1m_000001",
  "category": "general",
  "audio_path": "wav/beamformed/speaker01_front_1m_000001.wav",
  "text": "오늘 회의는 오후 세 시에 시작하겠습니다.",
  "speaker_id": "speaker01",
  "distance_m": 1.0,
  "angle_deg": 0,
  "noise_condition": "quiet"
}
```

데이터는 `튜닝용 60개`와 `최종 시험용 30개`처럼 분리한다. beam, VAD 또는
RMS 문턱을 고를 때는 튜닝용만 사용하고, 최종 시험용은 설정을 확정한 뒤 한 번만
측정한다. 같은 문장이 양쪽에 섞이면 실제보다 성능이 좋아 보일 수 있다.

합성 한국어 음성 30개는 코드가 정상 동작하는지와 설정 A/B 비교에는 유용하지만
발음과 음질이 깨끗해 현장 성능을 대신할 수 없다. 최종 성능 판단에는 반드시
실제 ReSpeaker·박스·BEM/MVDR 환경에서 녹음한 데이터를 사용한다.

공개 한국어 음성 데이터는 일반적인 발음 범위를 넓히는 보조 평가나 향후
Fine-tuning에는 도움이 될 수 있다. 그러나 현재 시스템의 주요 실패 원인이
마이크 배열, 잔향, 방향과 Beamforming이라면 공개 데이터보다 **우리 장치에서
직접 수집한 60~90개 음성**이 설정 개선에 더 직접적이다.

Fine-tuning을 하려면 평가용과 별도의 학습 데이터가 필요하며, Whisper 학습 후
CTranslate2 형식으로 다시 변환해야 한다. 단순히 현재 `sample_data`나 평가
폴더에 WAV를 추가하는 것은 Fine-tuning이 아니다. 먼저 실제 평가 데이터로
오류 원인을 수치화하고, 입력·설정 조정만으로 부족할 때 별도 Fine-tuning 단계로
진행한다.

개인 음성과 대용량 WAV는 현재 `.gitignore`에 의해 Git에서 제외된다. GitHub에는
정답 JSON, 수집 조건과 평가 결과만 올리고, 원본 음성은 팀 공유 저장소나 외장
저장장치에 별도로 보관한다.

## 12. 현재 로컬 기준선과 해석 방법

2026-08-27 최종 코드의 Mac 로컬 회귀 결과는 다음과 같다.

| 검증 | 결과 |
|---|---:|
| Streaming STT 자동 테스트 | 38개 통과 |
| Jetson UI·통합 계약 테스트 | 6개 통과 |
| Python·셸 문법 검사 | 통과 |
| 실제 UDP 종료 신호 | 동일 `StreamingSTT.flush()` 1회 호출 |
| 잘못된 종료 패킷 | 무시 |
| 문장 끝 무음의 신규 Partial | 생성하지 않음 |

동일한 10ms 모의 추론에서 종료 경로만 비교한 결과는 다음과 같다.

| 경로 | 발화 종료 후 Final callback |
|---|---:|
| 0.45초 침묵 fallback, 0.25초 chunk | 518.81ms |
| 외부 종료 신호 즉시 `flush()` | 12.86ms |
| 제거된 제어 대기 | 505.95ms |

이 수치는 **Whisper 실제 처리속도가 12.86ms라는 뜻이 아니다.** 같은 10ms 가짜
추론을 사용해 외부 종료 신호가 침묵·chunk 대기를 제거하는지만 검증한 결과다.

Zeroth-Korean과 FLEURS의 실제 사람 낭독음성 100개를 `small`, CPU `int8`,
beam 3으로 다시 평가한 결과는 다음과 같다.

| 지표 | 이전 동일 100문장 | 최종 코드 재평가 |
|---|---:|---:|
| CER | 8.22% | 8.22% |
| WER | 24.33% | 24.33% |
| 평균 파일 추론 | 1,817.04ms | 1,816.06ms |
| P95 파일 추론 | 2,226.91ms | 2,108.94ms |
| RTF | 0.1887 | 0.1886 |
| 빈 결과·오류·중복 | 0건 | 0건 |

CER·WER가 동일하므로 이번 제어 경로 개선에 따른 정확도 회귀는 발견되지 않았다.
오프라인 파일 평가기는 0.25초 chunk나 외부 flush를 사용하지 않으므로 latency
차이는 실행 편차이며 지연시간 개선 효과로 주장하지 않는다. 보고서의 최종 성능은
Jetson과 실제 ReSpeaker/BEM/MVDR 입력으로 다시 측정해야 한다.

## 13. 발표 전 최종 체크리스트

- [ ] `develop` 최신 커밋을 pull했다.
- [ ] pull한 파일에서 `AUDIO_CHUNK_SECONDS = 0.25`를 확인했다.
- [ ] `runtime_protocol.py`에 `SENTENCE_END_UDP_ADDR`가 있다.
- [ ] BEM 테이블이 현재 박스와 마이크 배열에 맞는다.
- [ ] Jetson이 MAXN 계열 모드이며 냉각 팬이 정상 작동한다.
- [ ] `get_device()` 결과가 `cuda`이다.
- [ ] CUDA 지원 연산 목록에 `int8_float16`이 있다.
- [ ] `tiny`와 `small` 모델을 미리 다운로드했다.
- [ ] ReSpeaker 입력 채널과 `[1, 2, 3, 4]` 매핑을 확인했다.
- [ ] 실행 직후 1초 동안 배경 잡음을 보정했다.
- [ ] 준형님 모듈이 마지막 audio chunk 뒤 종료 신호를 한 번 보낸다.
- [ ] `[문장 종료]` 로그가 문장마다 정확히 한 번 나온다.
- [ ] 일반 한국어 10문장으로 CER와 지연시간을 기록했다.
- [ ] 잡음만 1분 입력했을 때 환각 자막이 없다.
- [ ] Queue drop, callback overflow와 GPU 스로틀링이 없다.
- [ ] 0.25초 chunk에서도 DOA 정확도와 마지막 음절이 유지된다.
- [ ] Partial/Final 중복이 없고 화자 표시가 Final에서 정상 해제된다.
- [ ] `Ctrl+C`로 traceback 없이 종료된다.

## 14. 참고 문서

- 프로젝트 설치 절차: `JETSON_UBUNTU_SSH_SETUP.md`
- STT 입력·출력 계약: `INTERFACE_CONTRACT.md`
- 평가 방법과 기존 수치: `STT_EVALUATION_PLAN.md`
- 전체 Whisper 구조와 개선 이력: `WHISPER_STT_ARCHITECTURE_REPORT.md`
- [NVIDIA Jetson Orin Nano 전력 모드와 tegrastats](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/howto.html)
- [CTranslate2 양자화와 지원 compute type](https://opennmt.net/CTranslate2/quantization.html)
- [CTranslate2 `get_supported_compute_types`](https://opennmt.net/CTranslate2/python/ctranslate2.get_supported_compute_types.html)
- [faster-whisper GPU 요구사항과 모델 로딩](https://github.com/SYSTRAN/faster-whisper)

## 15. 최종 권장 판단

Jetson 첫 기준선은 **`tiny Partial + small Final + CUDA int8_float16 + Final beam
3 + 0.25초 Beamforming mono + 외부 종료 신호 flush`**로 잡는다. 정확도를
높이기 위해 무조건 큰 모델을 사용하지 말고, 입력 품질과 GPU 실행 여부를 먼저
확인한다. Final VAD와 현재 잡음·환각 필터는 유지한다.

성능 수치가 목표에 미달할 때는 `CUDA 확인 → 전력·발열 확인 → 입력·BEM 확인 →
0.25초 chunk 처리 확인 → 외부 flush 송신 순서 확인 → beam 조정 → 모델 크기
조정` 순서로 진행한다. 이 순서를 지켜야 속도 저하의 원인과 정확도 저하의
원인을 구분할 수 있다.

현재 로컬 코드 검증은 완료됐지만 다음 세 항목은 Jetson에서만 최종 확정할 수
있다.

1. `tiny`와 `small` 동시 로딩 시 실제 GPU 메모리 여유
2. DOA·BEM·MVDR 0.25초 chunk의 처리 P95와 방향 정확도
3. 외부 종료 신호부터 UI Final 표시까지의 end-to-caption P95

이 세 결과가 없으면 “Jetson Final 0.5초 이내 달성”으로 보고하지 않는다.
