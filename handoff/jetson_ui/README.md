# Jetson 1024×600 실시간 자막 UI 전달본

Jetson Orin Nano의 10.1인치 1024×600 모니터에서 DOA·화자 인식·한국어
STT 상태와 자막을 표시하기 위한 통합 파일이다. 이 폴더는 준형님이 관리하는
`project_main`에 덮어쓰는 전달용 구성이다.

## 상태 원 규칙

- 마이크 또는 카메라가 연결되면 초록색 원
- 마이크 또는 카메라가 연결되지 않으면 빨간색 원
- UI 서버가 준비되면 오른쪽 시스템 상태가 초록색 원
- 장치 상태는 `audio_connected`, `camera_connected` 값에 따라 자동 갱신

## 포함 파일

- `stage6_caption_ui.py`: 카메라·화자·STT 상태와 UI 서버 관리
- `realtime_doa.py`: DOA·MVDR 결과를 UI 서버로 전달
- `runtime_protocol.py`: 오디오·방향·외부 문장 종료 UDP 데이터 규격
- `run_jetson_ui.sh`: Jetson 실행 진입점
- `ui/`: 1024×600 정적 화면
- `tests/test_caption_ui.py`: UI 서버와 Jetson 전체화면 옵션 검사
- `requirements-jetson.txt`: 필요한 Python 패키지

## project_main에 적용

이 폴더의 파일을 BEM 테이블과 기존 DOA 모듈이 있는 `project_main` 루트에
복사한다. `bem_table_reduced.h5`, Whisper 모델, 가상환경과 실행 결과는 Git에
포함하지 않는다.

```bash
cp -R handoff/jetson_ui/* /path/to/project_main/
cd /path/to/project_main

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements-jetson.txt

chmod +x run_jetson_ui.sh
./run_jetson_ui.sh
```

장치 없이 화면만 확인할 때는 다음 명령을 사용한다.

```bash
./run_jetson_ui.sh --demo
```

## 외부 문장 종료 신호 연결

준형님 모듈이 문장 종료를 확정한 시점에 다음 패킷을 한 번 보내면 UI
프로세스가 사용 중인 동일한 `StreamingSTT` 객체의 `flush()`를 즉시 호출한다.
내부 0.45초 침묵 검사는 종료 신호가 없을 때의 fallback으로 그대로 유지된다.

```python
import socket

from runtime_protocol import SENTENCE_END_UDP_ADDR, pack_sentence_end

control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
control_socket.sendto(pack_sentence_end(), SENTENCE_END_UDP_ADDR)
```

오디오·Beamforming 전달 주기는 StreamingSTT의 Partial 요청 주기와 동일한
`0.25초`이다. 실제 Jetson에서는 이 변경 후에도 DOA 정확도와 오디오 유실이
없는지 확인해야 한다.

## 검증 결과

- 1024×600 화면 너비와 높이에 정확히 맞음
- 가로·세로 스크롤 없음
- 미연결 마이크·카메라 빨간 원 표시 확인
- 연결 시 초록 원으로 전환하는 코드 검사
- 브라우저 JavaScript 오류 0건
- Python 자동 테스트 UI·통합 계약 6개, Streaming STT 38개 통과
- partial 동안 해당 화자 카드·얼굴 상자만 활성화
- final 또는 녹음 중지 시 활성 화자 표시 해제
- 진행 중인 발화는 DOA 후보가 바뀌어도 최초 화자 번호 유지
- 외부 문장 종료 신호가 동일 StreamingSTT 객체를 한 번만 flush하는 계약 확인
- 문장 끝 무음에서 새 Partial 요청을 만들지 않아 Final 우선 처리 확인
- Ctrl+C 종료 시 traceback 없이 정상 종료

실제 카메라·ReSpeaker·HDMI 출력은 Jetson 장치에서 마지막으로 확인해야 한다.
