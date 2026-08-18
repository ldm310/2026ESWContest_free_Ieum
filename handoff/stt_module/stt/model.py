"""faster-whisper 모델을 한 번만 로드하고 공유하는 모듈."""

from __future__ import annotations

import gc
from threading import Lock
from typing import TYPE_CHECKING, ClassVar

from stt.config import MODEL_SIZE, PARTIAL_MODEL_SIZE, get_compute_type, get_device

if TYPE_CHECKING:
    # 정적 타입 검사에만 필요한 import이다. 실제 패키지 import와 무거운 모델
    # 로딩은 최초 get_model() 호출 시점까지 지연한다.
    from faster_whisper import WhisperModel


__all__ = ["ModelManager", "get_model"]


class ModelManager:
    """프로그램 전체에서 단일 Whisper 모델의 생명주기를 관리한다.

    모델 참조와 동기화 객체를 클래스 내부에 보관하므로 별도의 관리자
    인스턴스를 만들 필요가 없다. 모든 호출자는 동일한 모델 상태를 공유한다.
    """

    # None은 모델이 아직 로드되지 않았거나 명시적으로 해제된 상태를 뜻한다.
    _model: ClassVar[WhisperModel | None] = None
    _partial_model: ClassVar[WhisperModel | None] = None

    # 최초 로딩과 모델 해제를 동일한 잠금으로 보호한다. 여러 백엔드 요청이
    # 동시에 들어와도 모델 생성과 해제가 서로 겹치지 않는다.
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def get_model(cls) -> WhisperModel:
        """공유할 faster-whisper 모델 인스턴스를 반환한다.

        최초 호출에서만 설정값에 따라 ``WhisperModel``을 생성한다. 로딩이
        끝난 이후의 모든 호출은 메모리에 보관된 동일한 인스턴스를 반환한다.

        Returns:
            프로그램 전체에서 공유하는 ``WhisperModel`` 인스턴스.

        Raises:
            RuntimeError: faster-whisper를 불러올 수 없거나 모델 생성에 실패한 경우.
        """

        # 대부분의 호출은 잠금을 획득하지 않고 이미 생성된 객체를 바로 반환한다.
        if cls._model is not None:
            return cls._model

        # 최초 호출이 동시에 들어올 수 있으므로 잠금 안에서 상태를 다시
        # 확인한다. 앞선 스레드가 생성했다면 해당 객체를 그대로 반환한다.
        with cls._lock:
            if cls._model is not None:
                return cls._model

            cls._model = cls._load_model(MODEL_SIZE, role="final")

            return cls._model

    @classmethod
    def get_partial_model(cls) -> WhisperModel:
        """실시간 preview용 경량 Whisper 모델을 반환한다.

        partial 모델도 프로그램 실행 중 한 번만 로드한다. 설정에서 final과
        같은 크기를 지정하면 객체를 중복 생성하지 않고 기본 모델을 공유한다.

        Returns:
            partial 자막 추론에 사용할 공유 모델 인스턴스.
        """

        if PARTIAL_MODEL_SIZE == MODEL_SIZE:
            return cls.get_model()
        if cls._partial_model is not None:
            return cls._partial_model

        with cls._lock:
            if cls._partial_model is None:
                cls._partial_model = cls._load_model(
                    PARTIAL_MODEL_SIZE,
                    role="partial",
                )
            return cls._partial_model

    @staticmethod
    def _load_model(model_size: str, role: str) -> WhisperModel:
        """지정한 역할의 faster-whisper 모델을 현재 장치에 로드한다."""

        device = get_device()
        compute_type = get_compute_type()
        try:
            # import까지 지연해 설정 확인만으로 모델 런타임이 로드되지 않게 한다.
            from faster_whisper import WhisperModel

            return WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper 모델 로딩에 실패했습니다. "
                f"role={role!r}, model_size={model_size!r}, device={device!r}, "
                f"compute_type={compute_type!r}. 원인: {exc}"
            ) from exc

    @classmethod
    def unload_model(cls) -> None:
        """현재 모델에 대한 내부 참조를 해제한다.

        로드된 모델이 없으면 아무 작업도 하지 않는다. 모델 참조를 제거하고
        가비지 컬렉션을 요청하는 최소한의 정리만 수행하며, 향후 TensorRT나
        Fine-Tuning 모델의 전용 해제 절차는 ``_release_resources()``에
        확장할 수 있다.
        """

        with cls._lock:
            if cls._model is None and cls._partial_model is None:
                return

            # 같은 객체를 공유하는 설정까지 고려해 중복 해제를 방지한다.
            models = []
            if cls._model is not None:
                models.append(cls._model)
            if cls._partial_model is not None and cls._partial_model is not cls._model:
                models.append(cls._partial_model)
            cls._model = None
            cls._partial_model = None

            for model in models:
                cls._release_resources(model)
            models.clear()

            # Python 객체가 더 이상 참조되지 않을 때 CTranslate2가 소유한
            # CPU/GPU 리소스도 정리될 수 있도록 즉시 가비지 컬렉션을 요청한다.
            gc.collect()

    @staticmethod
    def _release_resources(model: WhisperModel) -> None:
        """모델 백엔드별 추가 리소스 정리를 위한 확장 지점을 제공한다.

        현재 faster-whisper 모델에는 별도의 공용 종료 메서드가 없으므로
        최소 정리만 수행한다. 향후 TensorRT 컨텍스트나 별도 GPU 버퍼를
        사용하면 이 메서드에 명시적인 해제 절차를 추가할 수 있다.

        Args:
            model: 메모리에서 해제할 모델 인스턴스.
        """

        # 현재는 객체 참조 해제와 가비지 컬렉션만으로 정리하므로 의도적으로
        # 추가 작업을 수행하지 않는다.
        _ = model


def get_model() -> WhisperModel:
    """기존 외부 API를 통해 공유 모델 인스턴스를 반환한다.

    Returns:
        프로그램 전체에서 공유하는 ``WhisperModel`` 인스턴스.

    Raises:
        RuntimeError: faster-whisper를 불러올 수 없거나 모델 생성에 실패한 경우.
    """

    return ModelManager.get_model()
