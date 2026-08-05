"""프로젝트 전역에서 사용하는 STT 설정값.

이 모듈은 설정만 정의하며 faster-whisper의 ``WhisperModel``을 생성하거나
모델 파일을 불러오지 않는다.
"""


def get_device() -> str:
    """실행 환경을 확인해 faster-whisper가 사용할 장치를 반환한다.

    Returns:
        CUDA 장치를 사용할 수 있으면 ``"cuda"``, 그렇지 않으면 ``"cpu"``.
    """
    return "cpu"  # Jetson Nano는 CUDA를 지원하지만, Whisper 모델은 CPU에서만 실행한다.

    # faster-whisper의 실행 엔진인 CTranslate2를 우선 사용해 확인한다.
    # 패키지가 아직 설치되지 않은 초기 프로젝트 상태도 고려해 선택적으로
    # 불러오며, 장치 조회가 실패하면 PyTorch를 이용한 확인으로 넘어간다.
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass

    # 개발 환경에 PyTorch만 설치된 경우를 위한 보조 확인 방법이다.
    # PyTorch도 없다면 CUDA를 사용할 수 없는 환경으로 안전하게 판단한다.
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass

    # CUDA 런타임이나 지원 장치를 확인할 수 없는 모든 환경에서는 CPU를
    # 기본값으로 사용해 설정 모듈을 안전하게 불러올 수 있도록 한다.
    return "cpu"


def get_compute_type() -> str:
    """현재 장치에 적합한 faster-whisper 연산 형식을 반환한다.

    CUDA에서는 빠른 반정밀도 연산인 ``float16``을 사용하고, CPU에서는
    메모리 사용량과 처리 효율을 고려한 ``int8`` 양자화를 사용한다.

    Returns:
        CUDA 환경이면 ``"float16"``, CPU 환경이면 ``"int8"``.
    """

    return "float16" if get_device() == "cuda" else "int8"


# 사용할 Whisper 모델의 크기 또는 버전 이름이다. ``tiny``, ``base``,
# ``small``, ``medium``, ``large-v3`` 등 faster-whisper가 지원하는 값을
# 지정할 수 있다. 정확도와 Jetson의 자원 사용량 사이의 균형을 고려해
# 기본값은 small로 설정한다.
MODEL_SIZE: str = "small"

# 입력 음성을 한국어로 명시해 자동 언어 감지 비용을 줄인다.
LANGUAGE: str = "ko"

# 디코딩 시 유지할 후보 수. 값이 클수록 탐색은 넓어지지만 처리 시간이 늘어난다.
BEAM_SIZE: int = 5

# 무음 구간을 제외해 불필요한 추론을 줄이기 위한 VAD 필터 사용 여부이다.
VAD_FILTER: bool = True

# STT 입력 오디오의 표준 샘플링 주파수(Hz)이다.
SAMPLE_RATE: int = 16_000

# 향후 자막 변환 결과를 저장할 프로젝트 루트 기준 상대 경로이다.
OUTPUT_JSON: str = "output/subtitle.json"
