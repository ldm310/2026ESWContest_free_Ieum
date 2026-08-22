# Windows Setup

Windows 실제 실행 검증은 아직 수행하지 않았습니다. 아래 절차는 PowerShell과 Python 3.12를 기준으로 한 정적 호환 설치 절차이며, 최종 확인은 준형님 Windows 환경에서 필요합니다.

## 1. GitHub에서 Windows로 받기

Git for Windows가 설치된 PowerShell에서 처음 한 번만 clone합니다.

```powershell
cd "$HOME\Documents"
git clone -b feature/dongmin https://github.com/ldm310/Embedded-project.git
cd .\Embedded-project
```

이미 clone한 폴더가 있으면 새로 받지 않고 다음 명령으로 갱신합니다.

```powershell
cd "$HOME\Documents\Embedded-project"
git switch feature/dongmin
git pull origin feature/dongmin
```

`git pull` 전에 직접 수정한 파일이 있으면 `git status`로 확인하고 별도로
보관해야 합니다. 평가용 WAV와 결과 JSON은 Git에 포함되지 않습니다.

## 2. 프로젝트 폴더 이동

```powershell
cd .\handoff\stt_module
```

## 3. 가상환경 생성

```powershell
py -3.12 -m venv .venv
```

## 4. 가상환경 실행

```powershell
.\.venv\Scripts\Activate.ps1
```

실행 정책 오류가 발생하면 현재 PowerShell 프로세스에만 임시 허용합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. pip 업그레이드

```powershell
python -m pip install --upgrade pip
```

## 6. 핵심 패키지 설치

```powershell
python -m pip install -r requirements-core.txt
```

## 7. import 확인

```powershell
python -c "from stt import StreamingSTT, STTResult; print('STT 모듈 import 성공')"
```

import만으로 모델 다운로드, worker 시작 또는 추론이 실행되지 않아야 정상입니다.

## 8. 자동 테스트

```powershell
python -m unittest discover -s tests -v
```

## 9. WAV 예제 실행

```powershell
python examples\streaming_api_example.py sample_data\sample.wav
python examples\pcm16_input_example.py sample_data\sample.wav
```

예제와 모듈은 `pathlib.Path`를 사용하므로 Windows 백슬래시와 한글 경로를 처리할 수 있습니다. 다만 모델 파일 다운로드, CPU 추론, Windows 장치 연동은 준형님 환경에서 실제 확인해야 합니다.

## 10. Windows 한국어 TTS 평가 데이터 생성

Windows 설정에서 **시간 및 언어 → 언어 및 지역 → 한국어 → 언어 옵션**으로
이동해 한국어 `음성` 기능을 먼저 설치합니다. 이후 프로젝트의
`handoff\stt_module` 폴더에서 실행합니다.

- [Microsoft Windows 언어 기능 설치 안내](https://support.microsoft.com/en-us/surface/remove-unwanted-keyboard-or-language-from-windows-11-883fbe1c-fcf8-44bc-ba42-a834f486c058)
- [Microsoft System.Speech WAV 출력 명세](https://learn.microsoft.com/en-us/dotnet/api/system.speech.synthesis.speechsynthesizer.setoutputtowavefile)

```powershell
python -m evaluation.generate_evaluation_data
```

다음 구성이 자동 생성됩니다.

```text
evaluation\generated\
├── audio\                 # 16kHz mono PCM WAV 30개
└── ground_truth.json      # WAV별 정답 문장과 카테고리
```

문장은 일반 문장 10개, 프로젝트 용어 10개, 숫자·영문 혼합 문장 10개입니다.
이미 생성된 파일을 다시 만들 때만 `--overwrite`를 사용합니다.

```powershell
python -m evaluation.generate_evaluation_data --overwrite
```

한국어 음성 미설치 오류가 발생하면 임의의 다른 언어 음성으로 생성하지 말고
Windows 한국어 음성 기능을 설치한 뒤 다시 실행합니다.

## 11. Small·Medium STT 평가

현재 production 설정과 같은 `language=ko`, `beam_size=3`, initial prompt와
hotwords로 평가하며 모델은 한 번만 로드합니다.

```powershell
python -m evaluation.evaluate_dataset --model-size small
python -m evaluation.evaluate_dataset --model-size medium
```

결과 경로는 다음과 같습니다.

```text
evaluation\results\small_result.json
evaluation\results\medium_result.json
```

결과에는 CER, WER, 전문용어 정규화 CER, 전문용어 정확도, 문장 일치율,
평균·P50·P95·최대 latency, RTF, 빈 결과, 오류 및 중복 ID가 기록됩니다.
합성 음성 점수는 모델 설정 비교용이며 실제 마이크 성능으로 보고하지 않습니다.

## 12. Windows에서 Jetson으로 전달

Windows OpenSSH Client가 활성화되어 있고 Jetson과 같은 네트워크에 있을 때
PowerShell에서 전달 폴더를 복사합니다. 아래 사용자명과 IP는 실제 값으로
교체합니다.

```powershell
cd "$HOME\Documents\Embedded-project"
ssh jetson@JETSON_IP "mkdir -p ~/handoff ~/project_main"
scp -r .\handoff\stt_module jetson@JETSON_IP:~/handoff/
scp -r .\handoff\jetson_ui jetson@JETSON_IP:~/handoff/
```

Jetson 터미널에서 production 폴더에 적용합니다.

```bash
mkdir -p ~/project_main
cp -R ~/handoff/jetson_ui/. ~/project_main/
cp -R ~/handoff/stt_module/stt ~/project_main/
```

Jetson에서도 동일한 합성 평가 파일을 사용하려면 Windows에서 생성한
`evaluation\generated` 폴더를 별도로 복사합니다.

```powershell
scp -r .\handoff\stt_module\evaluation\generated jetson@JETSON_IP:~/project_main/evaluation/
```
