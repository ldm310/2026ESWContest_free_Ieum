# Windows Setup

Windows 실제 실행 검증은 아직 수행하지 않았습니다. 아래 절차는 PowerShell과 Python 3.12를 기준으로 한 정적 호환 설치 절차이며, 최종 확인은 준형님 Windows 환경에서 필요합니다.

## 1. 프로젝트 폴더 이동

```powershell
cd "전달받은\stt_module"
```

## 2. 가상환경 생성

```powershell
py -3.12 -m venv .venv
```

## 3. 가상환경 실행

```powershell
.\.venv\Scripts\Activate.ps1
```

실행 정책 오류가 발생하면 현재 PowerShell 프로세스에만 임시 허용합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. pip 업그레이드

```powershell
python -m pip install --upgrade pip
```

## 5. 핵심 패키지 설치

```powershell
python -m pip install -r requirements-core.txt
```

## 6. import 확인

```powershell
python -c "from stt import StreamingSTT, STTResult; print('STT 모듈 import 성공')"
```

import만으로 모델 다운로드, worker 시작 또는 추론이 실행되지 않아야 정상입니다.

## 7. 자동 테스트

```powershell
python -m unittest discover -s tests -v
```

## 8. WAV 예제 실행

```powershell
python examples\streaming_api_example.py sample_data\sample.wav
python examples\pcm16_input_example.py sample_data\sample.wav
```

예제와 모듈은 `pathlib.Path`를 사용하므로 Windows 백슬래시와 한글 경로를 처리할 수 있습니다. 다만 모델 파일 다운로드, CPU 추론, Windows 장치 연동은 준형님 환경에서 실제 확인해야 합니다.
