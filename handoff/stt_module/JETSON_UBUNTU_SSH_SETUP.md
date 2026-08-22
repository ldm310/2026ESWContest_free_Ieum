# Jetson Ubuntu SSH Setup

이 문서는 Windows 또는 다른 PC에서 SSH로 Jetson Orin Nano에 접속한 뒤,
**Jetson의 Ubuntu 터미널에서** GitHub `feature/dongmin` 브랜치를 직접 받아
실행하는 절차다. 아래 모든 Linux 명령은 SSH 접속 후 Jetson 터미널에서
실행한다.

## 1. 기본 도구 설치

Jetson 터미널에서 실행한다.

```bash
sudo apt update
sudo apt install -y git python3-pip python3-venv portaudio19-dev libsndfile1
```

## 2. GitHub 코드 받기

처음 한 번만 저장소를 clone한다.

```bash
cd ~
git clone -b feature/dongmin https://github.com/ldm310/Embedded-project.git
cd ~/Embedded-project
```

이미 clone한 저장소가 있으면 현재 변경사항을 먼저 확인하고 갱신한다.

```bash
cd ~/Embedded-project
git status
git switch feature/dongmin
git pull origin feature/dongmin
```

`git status`에 직접 수정한 파일이 나타나면 덮어쓰지 말고 먼저 별도로
보관하거나 커밋해야 한다.

## 3. project_main에 전달본 적용

기존 DOA·Beamforming 코드와 BEM 테이블이 있는 `~/project_main`에 UI와 STT
전달본을 복사한다. 기존 프로젝트 파일은 삭제하지 않는다.

```bash
cd ~/Embedded-project
mkdir -p ~/project_main
cp -R handoff/jetson_ui/. ~/project_main/
cp -R handoff/stt_module/stt ~/project_main/
cp handoff/stt_module/requirements-core.txt ~/project_main/requirements-stt.txt
```

다음 파일은 Git에 포함되지 않으므로 `~/project_main`에 별도로 있어야 한다.

```text
bem_table_reduced.h5
```

파일을 확인한다.

```bash
test -f ~/project_main/bem_table_reduced.h5 \
  && echo "BEM 테이블 확인" \
  || echo "BEM 테이블이 없습니다"
```

## 4. 가상환경과 패키지 설치

Jetson에 설치된 OpenCV 등 시스템 패키지를 함께 사용하도록
`--system-site-packages`를 적용한다.

```bash
cd ~/project_main
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
```

터미널을 다시 열었다면 다음 명령으로 가상환경에 들어간다.

```bash
cd ~/project_main
source .venv/bin/activate
```

## 5. STT와 CUDA 확인

```bash
cd ~/project_main
python -c "import numpy, faster_whisper, ctranslate2; print('필수 패키지 import 성공')"
python -c "from stt.config import get_device, get_compute_type; print(get_device(), get_compute_type())"
```

Jetson GPU가 정상 인식되면 두 번째 명령의 예상 출력은 다음과 같다.

```text
cuda int8_float16
```

`cpu int8`이 출력되면 코드를 실행하기 전에 JetPack, CUDA, cuDNN과
CTranslate2 호환 상태를 확인한다.

## 6. 자동 테스트

같은 가상환경 Python으로 STT와 UI 테스트를 실행한다.

```bash
cd ~/Embedded-project/handoff/stt_module
~/project_main/.venv/bin/python -m unittest discover -s tests -v

cd ~/Embedded-project/handoff/jetson_ui
~/project_main/.venv/bin/python -m unittest discover -s tests -v
```

자동 테스트는 실제 ReSpeaker·카메라·GPU 추론을 대신하지 않는다.

## 7. 화면만 먼저 확인

장치를 연결하지 않고 1024×600 UI만 확인한다.

```bash
cd ~/project_main
chmod +x run_jetson_ui.sh
./run_jetson_ui.sh --demo
```

## 8. 실제 장치 실행

ReSpeaker, 카메라, HDMI 모니터와 BEM 테이블을 준비한 뒤 실행한다.

```bash
cd ~/project_main
source .venv/bin/activate
./run_jetson_ui.sh
```

장치 이름이 자동으로 잡히지 않으면 먼저 입력 장치를 확인한다.

```bash
python realtime_doa.py --list-devices
```

준형님 모듈이 문장 종료를 검출하는 지점에서는 현재 발화를 바로 확정하도록
동일한 `StreamingSTT` 객체의 `flush()`를 호출해야 한다.

## 9. 이후 GitHub 업데이트 반영

새 커밋을 받을 때마다 저장소를 갱신한 뒤 전달본을 다시 복사한다.

```bash
cd ~/Embedded-project
git status
git switch feature/dongmin
git pull origin feature/dongmin

cp -R handoff/jetson_ui/. ~/project_main/
cp -R handoff/stt_module/stt ~/project_main/
cp handoff/stt_module/requirements-core.txt ~/project_main/requirements-stt.txt
```

패키지 목록이 변경된 경우에만 가상환경에서 다시 설치한다.

```bash
cd ~/project_main
source .venv/bin/activate
python -m pip install -r requirements-jetson.txt
```

## 10. 종료

실행 중 `Ctrl+C`를 누르면 UI와 DOA 프로세스가 순서대로 종료된다. 가상환경만
나가려면 다음 명령을 실행한다.

```bash
deactivate
```
