

def get_device() -> str:

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass

    return "cpu"


def get_compute_type() -> str:

    return "float16" if get_device() == "cuda" else "int8"


MODEL_SIZE: str = "small"


LANGUAGE: str = "ko"


BEAM_SIZE: int = 5


VAD_FILTER: bool = True


SAMPLE_RATE: int = 16_000


OUTPUT_JSON: str = "output/subtitle.json"
