
from __future__ import annotations

import gc
from threading import Lock
from typing import TYPE_CHECKING, ClassVar

from stt.config import MODEL_SIZE, get_compute_type, get_device

if TYPE_CHECKING:

    from faster_whisper import WhisperModel


__all__ = ["ModelManager", "get_model"]


class ModelManager:

    _model: ClassVar[WhisperModel | None] = None

    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def get_model(cls) -> WhisperModel:

        if cls._model is not None:
            return cls._model

        with cls._lock:
            if cls._model is not None:
                return cls._model

            device = get_device()
            compute_type = get_compute_type()

            try:

                from faster_whisper import WhisperModel

                cls._model = WhisperModel(
                    MODEL_SIZE,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as exc:

                raise RuntimeError(
                    "faster-whisper 모델 로딩에 실패했습니다. "
                    f"model_size={MODEL_SIZE!r}, device={device!r}, "
                    f"compute_type={compute_type!r}. 원인: {exc}"
                ) from exc

            return cls._model

    @classmethod
    def unload_model(cls) -> None:

        with cls._lock:
            if cls._model is None:
                return

            model = cls._model
            cls._model = None

            cls._release_resources(model)
            del model

            gc.collect()

    @staticmethod
    def _release_resources(model: WhisperModel) -> None:

        _ = model


def get_model() -> WhisperModel:

    return ModelManager.get_model()
