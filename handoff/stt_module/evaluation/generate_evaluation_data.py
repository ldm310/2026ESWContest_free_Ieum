"""Windows 또는 macOS 내장 한국어 TTS로 평가 WAV와 정답표를 생성한다."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_SENTENCES = EVALUATION_ROOT / "sentences.json"
DEFAULT_OUTPUT = EVALUATION_ROOT / "generated"
EXPECTED_CATEGORY_COUNTS = {"general": 10, "domain": 10, "mixed": 10}


def load_sentences(path: Path) -> list[dict[str, str]]:
    """평가 문장 JSON을 읽고 30문장 구성을 검증한다.

    Args:
        path: 문장 목록 JSON 경로.

    Returns:
        ``id``, ``category``, ``text``가 포함된 문장 목록.

    Raises:
        ValueError: 항목 형식, ID 또는 카테고리 구성이 잘못된 경우.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("문장 JSON의 최상위 값은 리스트여야 합니다.")

    sentences: list[dict[str, str]] = []
    identifiers: set[str] = set()
    category_counts = {category: 0 for category in EXPECTED_CATEGORY_COUNTS}
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{index}번째 문장 항목은 객체여야 합니다.")
        identifier = str(item.get("id", "")).strip()
        category = str(item.get("category", "")).strip()
        text = str(item.get("text", "")).strip()
        if not identifier or not text or category not in EXPECTED_CATEGORY_COUNTS:
            raise ValueError(f"{index}번째 문장의 id/category/text를 확인해 주세요.")
        if identifier in identifiers:
            raise ValueError(f"중복 문장 ID입니다: {identifier}")
        identifiers.add(identifier)
        category_counts[category] += 1
        sentences.append({"id": identifier, "category": category, "text": text})

    if category_counts != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(
            "평가 문장은 general/domain/mixed 각 10개여야 합니다. "
            f"현재 구성: {category_counts}"
        )
    return sentences


def _powershell_path() -> str:
    """사용 가능한 Windows PowerShell 실행 파일을 반환한다."""

    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("Windows PowerShell을 찾을 수 없습니다.")
    return executable


def _windows_script() -> str:
    """System.Speech를 호출하는 PowerShell 스크립트를 반환한다."""

    return r'''param(
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [Parameter(Mandatory=$true)][string]$Text,
    [string]$VoiceName = "",
    [int]$Rate = 0
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $voices = @($synth.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -eq "ko-KR"
    })
    if ($VoiceName) {
        $synth.SelectVoice($VoiceName)
    } elseif ($voices.Count -gt 0) {
        $synth.SelectVoice($voices[0].VoiceInfo.Name)
    } else {
        throw "한국어 음성(ko-KR)이 없습니다. Windows 언어 옵션에서 한국어 음성 기능을 설치하세요."
    }
    $synth.Rate = $Rate
    $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
        16000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono
    )
    $synth.SetOutputToWaveFile($OutputPath, $format)
    $synth.Speak($Text)
    $synth.SetOutputToNull()
} finally {
    $synth.Dispose()
}
'''


def generate_windows_wav(
    text: str,
    output_path: Path,
    voice: str | None,
    rate: int,
) -> str:
    """Windows System.Speech로 16kHz mono PCM WAV를 생성한다."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        suffix=".ps1",
        delete=False,
    ) as script_file:
        script_file.write(_windows_script())
        script_path = Path(script_file.name)
    try:
        command = [
            _powershell_path(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-OutputPath",
            str(output_path),
            "-Text",
            text,
            "-Rate",
            str(rate),
        ]
        if voice:
            command.extend(["-VoiceName", voice])
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Windows 한국어 TTS 생성에 실패했습니다: {output_path.name}"
        ) from exc
    finally:
        script_path.unlink(missing_ok=True)
    return voice or "Windows 기본 ko-KR 음성"


def generate_macos_wav(
    text: str,
    output_path: Path,
    voice: str | None,
    rate: int,
) -> str:
    """로컬 검증을 위해 macOS 내장 TTS로 16kHz mono PCM WAV를 생성한다."""

    say_path = shutil.which("say")
    afconvert_path = shutil.which("afconvert")
    if say_path is None or afconvert_path is None:
        raise RuntimeError("macOS say 또는 afconvert 명령을 찾을 수 없습니다.")

    selected_voice = voice or "Yuna"
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temporary:
        aiff_path = Path(temporary.name)
    try:
        subprocess.run(
            [
                say_path,
                "-v",
                selected_voice,
                "-r",
                str(rate),
                "-o",
                str(aiff_path),
                text,
            ],
            check=True,
        )
        subprocess.run(
            [
                afconvert_path,
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                str(aiff_path),
                str(output_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"macOS 한국어 TTS 생성에 실패했습니다: {text}") from exc
    finally:
        aiff_path.unlink(missing_ok=True)
    return selected_voice


def validate_wav(path: Path) -> dict[str, int | float]:
    """생성 WAV가 16kHz mono signed 16-bit PCM인지 확인한다."""

    with wave.open(str(path), "rb") as wav_file:
        metadata: dict[str, int | float] = {
            "sample_rate": wav_file.getframerate(),
            "channels": wav_file.getnchannels(),
            "sample_width_bits": wav_file.getsampwidth() * 8,
            "duration_seconds": round(
                wav_file.getnframes() / wav_file.getframerate(), 3
            ),
        }
    expected = (
        metadata["sample_rate"],
        metadata["channels"],
        metadata["sample_width_bits"],
    )
    if expected != (16_000, 1, 16) or metadata["duration_seconds"] <= 0:
        raise ValueError(f"지원하지 않는 WAV 형식입니다: {path} ({metadata})")
    return metadata


def generate_dataset(args: argparse.Namespace) -> Path:
    """선택한 운영체제 TTS로 전체 평가 데이터와 manifest를 생성한다."""

    sentences = load_sentences(args.sentences)
    output_root = args.output.resolve()
    audio_root = output_root / "audio"
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"출력 폴더가 비어 있지 않습니다: {output_root}. --overwrite를 사용하세요."
        )
    audio_root.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    if system == "Windows":
        generator = generate_windows_wav
        rate = args.windows_rate
        engine = "windows-system-speech"
    elif system == "Darwin":
        generator = generate_macos_wav
        rate = args.macos_rate
        engine = "macos-say"
    else:
        raise RuntimeError("자동 TTS 생성은 Windows와 macOS에서만 지원합니다.")

    items: list[dict[str, Any]] = []
    selected_voice = ""
    for index, sentence in enumerate(sentences, start=1):
        output_path = audio_root / f"{sentence['id']}.wav"
        if args.overwrite:
            output_path.unlink(missing_ok=True)
        selected_voice = generator(
            sentence["text"], output_path, args.voice, rate
        )
        metadata = validate_wav(output_path)
        items.append(
            {
                **sentence,
                "audio_path": output_path.relative_to(output_root).as_posix(),
                **metadata,
            }
        )
        print(f"[생성 {index:02d}/{len(sentences)}] {output_path.name}")

    manifest = {
        "schema_version": 1,
        "purpose": "synthetic-regression-only",
        "tts_engine": engine,
        "voice": selected_voice,
        "items": items,
    }
    manifest_path = output_root / "ground_truth.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[완료] {len(items)}개 WAV와 정답표: {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """평가 데이터 생성 명령행 인자를 반환한다."""

    parser = argparse.ArgumentParser(
        description="Windows/macOS 내장 한국어 TTS 평가 데이터 생성"
    )
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voice", help="설치된 TTS 음성 이름")
    parser.add_argument("--windows-rate", type=int, default=0, choices=range(-10, 11))
    parser.add_argument("--macos-rate", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """명령행 설정으로 평가 데이터를 생성한다."""

    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
