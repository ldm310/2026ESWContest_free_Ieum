# proj_main — 실시간 한국어 다화자 강의 자막 시스템

Jetson Orin Nano Super 8GB / ROS2 Humble / 완전 오프라인.

이 폴더 하나로 돌아간다. `~/Embedded-project`, `~/emb_repo`, `~/project_main` 을
참조하지 않는다. 팀원 코드(STT 엔진, 자막 UI)는 패키지 안에 사본으로 들어 있다.

---

## 1. 빠른 시작

```bash
proj start          # 전체 파이프라인 기동 (12초 뒤 노드 목록 출력)
proj status         # 프로세스·노드·토픽·포트·자원
proj stop           # 정지
```

`proj` 는 `~/proj_main/bin/proj` 이고 `~/.bashrc` 의 PATH 에 들어 있다.
SSH 로 붙으면 바로 쓸 수 있다.

### 화면

| 주소 | 내용 |
|---|---|
| `http://localhost:8765` | 자막 UI — 카메라 영상, 얼굴 박스, 실시간 자막 |
| `http://localhost:8770` | DOA 디버그 UI — 극좌표 방향, 시간축, 슬롯 활성도, 파이프라인 단계 |

둘 다 127.0.0.1 에만 바인딩한다. 원격에서 보려면 터널을 뚫는다.

```bash
ssh -N -L 8765:localhost:8765 -L 8770:localhost:8770 jetson
```

---

## 2. proj 명령

```
proj start [인자…]   기동. launch 인자를 그대로 넘긴다
proj stop            정지
proj status          프로세스·노드·토픽·포트·RAM·GPU
proj log             실행 로그 실시간 (Ctrl+C 로 빠져나옴)
proj last [N]        최근 N줄 (기본 60)
proj tracks          /sound_source_tracks 흘려보기
proj debug           /doa_debug 한 번 (아래 표 참고)
proj caption         /caption 흘려보기
proj hz TOPIC        발행 주기
proj echo TOPIC      한 번 출력
proj weights         설치된 가중치의 epoch·dev MAE
```

로그는 `~/proj_main/log/run_YYYYmmdd_HHMMSS.log` 에 쌓인다.

---

## 3. 실행 인자

`proj start 이름:=값` 형태로 넘긴다. 여러 개 가능.

### 자주 쓰는 것

```bash
proj start activity_threshold:=0.5        # 덜 민감하게
proj start model_size:=medium             # STT 정확도↑ 속도↓
proj start camera:=false                  # 카메라 없이
proj start debug_ui:=false                # 디버그 UI 끄고
proj start raw_audio:=true                # STT 입력을 마이크 원본으로 (모델 의심될 때 대조)
```

### 전체 기본값

| 인자 | 기본값 | 설명 |
|---|---|---|
| `activity_threshold` | `0.30` | 이 값 이상이어야 트랙 후보 |
| `confirm_chunks` | `2` | 연속 몇 청크 잡혀야 트랙 확정 |
| `coast_chunks` | `32` | 놓쳐도 몇 청크까지 트랙 유지. 128 ms/청크 = 4.1초. 짧으면 문장 사이 쉼마다 track_id 가 갈려 자막이 조각난다 |
| `model_size` | `small` | Whisper 크기. **medium 은 VRAM 부족으로 죽는다** (DOA 모델·카메라와 8 GB 공유) |
| `stt_device` | `cuda` | **cpu 로 두면 죽는다** (아래 6장) |
| `stt_compute_type` | `float16` | |
| `silence_threshold` | `0.050` | STT 무음 판정 |
| `audio_gain` | `3.0` | STT 입력 이득 |
| `raw_audio` | `false` | **false=모델 분리 출력(기본)**, true=마이크 원본 ch0(대조군) |
| `camera` | `true` | |
| `yaw_offset_deg` | `0.0` | 마이크 배열↔카메라 방위 정렬 보정(도). 실측값 11.6 으로 실행 |
| `debug_ui` | `true` | |
| `port` / `debug_port` | `8765` / `8770` | |

### 코드 기본값 (launch 인자로는 안 나와 있음)

```
azimuth_min_deg     30.0     학습 범위. 이 밖의 트랙은 발행하지 않음
azimuth_max_deg    150.0
elevation_min_deg  -10.0
elevation_max_deg   35.0
use_face_limit      true     카메라 얼굴 수를 화자 수 상한으로
face_hold_sec        3.0     최근 이만큼의 최대 얼굴 수를 상한으로
face_timeout_sec     2.0     카메라가 이보다 오래 끊기면 상한 미적용
                             얼굴 0명도 상한 미적용 (판단 근거 없음으로 본다)
speaking_threshold   0.30    이 값 이상이면 UI 에서 "말하는 중" 강조
speaking_hold_sec    0.6     짧은 끊김에 강조가 깜빡이지 않게
model_enabled       true     추론 on/off (UI 버튼으로도 바꿈)
```

---

## 4. 시스템 구성

```
ReSpeaker 4Mic (6ch 16kHz)
   └─ doa_separation ── /sound_source_tracks ──┬─ camera_node ── /faces ─┐
        방향·활성도·분리                        │   (DOA로 얼굴 매칭)      │
        └─ /separated_audio ── stt_bridge ── /caption ── ui_bridge :8765 ─┘
        └─ /doa_debug ─────────────────────── doa_debug :8770
RealSense D455 ── camera_node ── /camera/compressed
```

### 패키지

| 패키지 | 역할 |
|---|---|
| `captioning_msgs` | 메시지 정의 |
| `doa_separation_node` | 마이크 캡처, DOA, 화자 분리, 트래킹 |
| `camera_node` | RealSense, 얼굴 검출(BlazeFace), DOA-얼굴 매칭 |
| `stt_bridge_node` | Whisper 자막. `stt_module/` 에 STT 엔진 사본 |
| `ui_bridge_node` | 자막 UI. `jetson_ui/` 에 화면 코드 사본 |
| `doa_debug_node` | DOA 디버그 UI |

### 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/sound_source_tracks` | `SoundSourceTracks` | 트랙별 방향·활성도·수명, 음원 개수 확률 |
| `/separated_audio` | `SeparatedAudio` | 트랙별 mono. `raw_audio` 에 따라 원본 또는 분리 출력 |
| `/faces` | `Faces` | 얼굴 박스. `n_detected` 는 화면 전체 검출 수 |
| `/caption` | `Caption` | 자막 (부분/확정) |
| `/camera/compressed` | `CompressedImage` | 카메라 영상 |
| `/doa_debug` | `Float32MultiArray` | 디버그 배열 (아래) |
| `/model_enabled` | `Bool` | 추론 on/off |
| `/stt_raw_audio` | `Bool` | true=원본 ch0, false=분리 출력 |
| `/caption_reset` | `Empty` | 자막·STT 세션 초기화 |

---

## 5. 좌표 규약 — **절대 바꾸지 말 것**

```
패널 0, 1, 2, 3  =  마이크 1, 2, 3, 4      항등. 순서를 섞지 않는다.
mic_channels = [1, 2, 3, 4]                장치 6채널 중 원시 마이크
PANEL_INDICES = [0, 1, 2, 3]               데이터 생성 쪽도 항등
```

```
+x 오른쪽    +y 정면    +z 위
azimuth = atan2(y, x)

az   0  = 오른쪽
az  90  = 정면          ← 카메라가 보는 방향
az 180  = 왼쪽
az 270  = 뒤

elevation = asin(z)     0 = 수평
```

BEM 표(`p_total_devicefin4k_az360el181_chief_4panel.h5`)도 같은 규약이다.
표의 `doa_grid` 인덱스는 `az_idx * 181 + el_idx` 이고, `q=16380` 이 `[0,1,0]` = 정면이다.

학습 데이터 범위는 **방위각 30~150도, 고도각 -10~35도** 다. 카메라 화각(89도, az 45~135)에
좌우 15도씩 여유를 준 값이다. 이 밖의 방향은 학습된 적이 없어 트랙으로 올리지 않는다.

---

## 6. 자주 겪는 문제

### stt_bridge 가 뜨자마자 죽는다

```
ValueError: Requested int8 compute type, but the target device or backend
            do not support efficient int8 computation.
```

aarch64 ctranslate2 는 **CPU int8 을 지원하지 않는다.** `stt_device:=cuda`,
`stt_compute_type:=float16` 으로 둔다 (기본값이 이미 그렇다).
CPU 로 돌리려면 `stt_compute_type:=float32` 를 써야 하는데 실시간이 안 나온다.

### doa_separation 이 FileNotFoundError 로 죽는다

```
FileNotFoundError: .../weights/jointnet_v4.pt
```

가중치 파일이 없다. `src/doa_separation_node/weights/` 에 넣고 다시 빌드한다.

```bash
cd ~/proj_main/ros2_ws && colcon build --symlink-install --packages-select doa_separation_node
```

### 첫 발화 자막이 8초쯤 늦게 뜬다

모델 예열이 안 된 것이다. 기동 로그에 `Whisper small 예열 완료 N.Ns (cuda)` 가
찍혀야 한다. 없으면 예열이 실패해 첫 오디오 콜백에서 로드되고, 그동안 ROS
executor 가 멈춰 오디오가 버려진다.

### stt_bridge 가 CUDA out of memory 로 죽는다

`model_size:=medium` 을 쓴 경우다. small 이하로 내린다.

### 마이크를 못 찾는다

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

`ReSpeaker 4 Mic Array (UAC1.0)` 가 6채널로 보여야 한다. USB 를 다시 꽂는다.

### 아무도 말 안 하는데 트랙이 뜬다 / 안 뜬다

`proj debug` 로 배열을 본다. 활성도(인덱스 5~7)와 문턱(12)을 비교한다.
너무 민감하면 `activity_threshold` 를 올리고, 안 잡히면 내린다.

### 포트가 이미 쓰이고 있다

```bash
proj stop
ss -tlnp | grep -E "8765|8770"
```

---

## 7. `/doa_debug` 배열 배치

25개 float. 순서가 바뀌면 디버그 UI 가 어긋나므로 발행 쪽과 함께 고쳐야 한다.

| 인덱스 | 내용 |
|---|---|
| 0–3 | 마이크 1~4 RMS |
| 4 | 기준 채널 RMS |
| 5–7 | 슬롯 1~3 활성도 (모델 원본) |
| 8–11 | 음원 개수 확률 0~3개 (얼굴 상한 적용 시 뒤쪽은 0) |
| 12 | `activity_threshold` |
| 13 | `level_gate` |
| 14 | 레벨 게이트 열림 (1/0) |
| 15 | 워밍업 완료 (1/0) |
| 16 | 문턱 넘은 슬롯 수 |
| 17 | 추적 중 트랙 수 (미확정 포함) |
| 18 | 확정 트랙 수 |
| 19 | 최고 연속 히트 수 |
| 20 | `confirm_chunks` |
| 21 | 얼굴 상한 (-1 = 미적용) |
| 22 | 얼굴 상한으로 버린 트랙 수 |
| 23 | 방위각/고도각 범위 밖으로 버린 트랙 수 |
| 24 | 실제 발행한 트랙 수 |

---

## 8. 모델 교체

가중치는 `src/doa_separation_node/weights/` 에 둔다. 기본 파일명은 `jointnet_v4.pt` 다.

```bash
# 노트북에서
rsync -a jointnet_v4.pt jetson:~/proj_main/ros2_ws/src/doa_separation_node/weights/

# 젯슨에서
cd ~/proj_main/ros2_ws
colcon build --symlink-install --packages-select doa_separation_node
proj stop && proj start
proj weights          # epoch 과 dev MAE 확인
```

다른 파일을 쓰려면 `proj start weights:=/절대/경로.pt`.

**주의**: `rsync --delete` 로 `src/` 를 통째로 덮으면 노트북 쪽에 없는 가중치가 지워진다.
가중치도 노트북 소스 트리에 같이 두고 동기화한다.

---

## 9. 화면 조작

두 UI 모두 같은 토픽으로 명령을 보낸다. 한쪽에서 바꾸면 다른 쪽 표시도 따라간다.

### 자막 UI (:8765) 상단

| 버튼 | 동작 |
|---|---|
| **DOA 모델** | 추론 on/off. 끄면 트랙·분리 음성 발행 중단, 노드는 생존. 켤 때 순환 상태·트래커 초기화 |
| **STT 입력** | `원본 ch0` ↔ `분리 출력` 전환 |
| **자막 리셋** | 화면 자막 비우고 STT 세션 전부 종료 |

### 얼굴 박스 색

```
화자 1  파랑 #2878f0      화자 2  초록 #20b45a      화자 3  주황 #ff9818
미상    회색 점선          아직 한 번도 말한 적 없는 얼굴
말하는 중                 같은 색, 테두리 6px + 흰 링 + "말하는 중" 배지
```

번호는 화면 왼쪽부터 매긴다. **한 번이라도 소리 트랙이 붙은 자리에만** 번호가 생긴다.
얼굴에 매칭되지 않은 소리(화각 밖 발화, 얼굴 검출 실패)는 화자를 특정하지 않고 `미상` 이다.

---

## 10. 빌드

```bash
cd ~/proj_main/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

전부 다시 짓기:

```bash
rm -rf build install log && colcon build --symlink-install
```

`--symlink-install` 이라 파이썬 소스는 다시 빌드하지 않아도 반영된다. 다만
`setup.py` 의 `data_files` 로 설치되는 것(가중치, vendor 디렉터리, launch, UI 자산)은
바꾸면 다시 빌드해야 한다.

파이썬은 시스템 것을 쓴다 (torch 2.8.0 CUDA, ctranslate2 4.6.0, sounddevice, mediapipe).
`~/proj_main/venv` 는 비어 있고 쓰지 않는다.

---

## 11. 자립 확인

바깥 폴더를 참조하지 않는지 확인한다. 아무것도 안 나와야 한다.

```bash
grep -rn "Embedded-project\|emb_repo\|project_main" ~/proj_main/ros2_ws/src
```
