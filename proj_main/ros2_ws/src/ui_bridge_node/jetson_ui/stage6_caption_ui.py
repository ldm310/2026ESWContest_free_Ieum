#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import shutil
import signal
import socket
import subprocess
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from runtime_protocol import (
    AUDIO_UDP_ADDR,
    DIRECTION_UDP_ADDR,
    unpack_audio_packet,
    unpack_direction,
)


PROJECT_ROOT = Path(__file__).resolve().parent
UI_ROOT = PROJECT_ROOT / "ui"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STT_SAMPLE_RATE = 16_000
MAX_CAPTIONS = 7
MAX_FACES = 3
MATCH_THRESHOLD_DEG = 22.0
UVC_HORIZONTAL_FOV_DEG = 69.0

SPEAKERS = (
    {"id": 1, "name": "화자 1", "color": "#2878f0"},
    {"id": 2, "name": "화자 2", "color": "#20b45a"},
    {"id": 3, "name": "화자 3", "color": "#ff9818"},
)


@dataclass(frozen=True)
class Caption:

    speaker_id: int
    text: str
    time: str
    latency_ms: float


class RuntimeState:

    def __init__(self, demo: bool = False) -> None:
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._frame: bytes | None = None
        self._frame_version = 0
        self._faces: list[dict[str, Any]] = []
        self._active_speaker_id: int | None = None
        self._direction = (1.0, 0.0, 0.0)
        self._captions: deque[Caption] = deque(maxlen=MAX_CAPTIONS)
        self._partial: dict[str, Any] | None = None
        self._utterance_speakers: dict[int, int] = {}
        self._status: dict[str, Any] = {
            "demo": demo,
            "recording": True,
            "diarization": True,
            "model": True,
            "stt_raw": False,
            "camera_connected": False,
            "audio_connected": False,
            "doa_ready": False,
            "stt_ready": demo,
            "error": None,
        }

    def set_status(self, **changes: Any) -> None:

        with self._lock:
            self._status.update(changes)

    def toggle(self, key: str) -> bool:

        if key not in {"recording", "diarization", "model", "stt_raw"}:
            raise ValueError(f"지원하지 않는 제어 항목입니다: {key}")
        with self._lock:
            self._status[key] = not bool(self._status[key])
            if key == "recording" and not self._status[key]:
                self._partial = None
            return bool(self._status[key])

    def clear_captions(self) -> None:

        with self._lock:
            self._captions.clear()
            self._partial = None
            self._utterance_speakers.clear()

    def is_recording(self) -> bool:

        with self._lock:
            return bool(self._status["recording"])

    def set_direction(self, direction: tuple[float, float, float]) -> None:

        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-9:
            return
        with self._lock:
            self._direction = tuple(value / norm for value in direction)
            self._status["doa_ready"] = True

    def direction(self) -> tuple[float, float, float]:

        with self._lock:
            return tuple(self._direction)

    def set_frame(self, jpeg: bytes) -> None:

        with self._frame_condition:
            self._frame = jpeg
            self._frame_version += 1
            self._frame_condition.notify_all()

    def wait_for_frame(
        self,
        previous_version: int,
        timeout: float,
    ) -> tuple[bytes | None, int]:

        with self._frame_condition:
            if self._frame_version == previous_version:
                self._frame_condition.wait(timeout=timeout)
            return self._frame, self._frame_version

    def update_faces(
        self,
        faces: list[dict[str, Any]],
        active_speaker_id: int | None,
    ) -> None:

        with self._lock:
            self._faces = faces
            self._active_speaker_id = active_speaker_id

    def active_speaker_id(self) -> int:

        with self._lock:
            return self._active_speaker_id or 0

    def handle_stt_result(self, result: Any) -> None:

        with self._lock:
            result_type = getattr(result, "type", "error")
            utterance_id = int(getattr(result, "utterance_id", 0))
            speaker_id = self._utterance_speakers.setdefault(
                utterance_id,
                self._active_speaker_id or 0,
            )

            if result_type == "partial":
                self._partial = {
                    "speaker_id": speaker_id,
                    "text": str(getattr(result, "text", "")),
                    "latency_ms": float(getattr(result, "latency_ms", 0.0)),
                }
                return

            if result_type == "final":
                text = str(getattr(result, "text", "")).strip()
                if text:
                    self._captions.append(
                        Caption(
                            speaker_id=speaker_id,
                            text=text,
                            time=datetime.now().astimezone().strftime("%H:%M:%S"),
                            latency_ms=round(
                                float(getattr(result, "latency_ms", 0.0)),
                                2,
                            ),
                        )
                    )
                self._partial = None
                self._utterance_speakers.pop(utterance_id, None)
                return

            error = str(getattr(result, "error", "알 수 없는 STT 오류"))
            self._status["error"] = error
            if bool(getattr(result, "is_final", False)):
                self._partial = None
                self._utterance_speakers.pop(utterance_id, None)

    def add_demo_caption(self, speaker_id: int, text: str) -> None:

        with self._lock:
            self._active_speaker_id = speaker_id
            self._captions.append(
                Caption(
                    speaker_id=speaker_id,
                    text=text,
                    time=datetime.now().astimezone().strftime("%H:%M:%S"),
                    latency_ms=0.0,
                )
            )
            self._partial = {
                "speaker_id": speaker_id,
                "text": "실시간 자막 입력을 기다리는 중입니다…",
                "latency_ms": 0.0,
            }

    def snapshot(self) -> dict[str, Any]:

        with self._lock:
            return {
                "speakers": list(SPEAKERS),
                "faces": [dict(face) for face in self._faces],
                "active_speaker_id": self._active_speaker_id,
                "captions": [asdict(caption) for caption in self._captions],
                "partial": dict(self._partial) if self._partial else None,
                "status": dict(self._status),
            }


class StreamingSTTService:

    def __init__(self, state: RuntimeState) -> None:
        self._state = state
        self._stt: Any | None = None

    def start(self) -> None:

        try:
            from stt import StreamingSTT

            self._stt = StreamingSTT(on_result=self._state.handle_stt_result)
            self._stt.start()
        except Exception as exc:
            self._state.set_status(stt_ready=False, error=f"STT 시작 실패: {exc}")
            raise RuntimeError(f"STT 시작에 실패했습니다: {exc}") from exc
        self._state.set_status(stt_ready=True)

    def push_audio(self, audio: Any) -> None:

        if self._stt is not None:
            self._stt.push_audio(audio, sample_rate=STT_SAMPLE_RATE)

    def stop(self) -> None:

        if self._stt is None:
            return
        try:
            self._stt.stop()
        finally:
            self._stt = None
            try:
                from stt.model import ModelManager

                ModelManager.unload_model()
            except Exception:
                pass


class RealtimeReceiver:

    def __init__(
        self,
        state: RuntimeState,
        stt_service: StreamingSTTService,
        stop_event: threading.Event,
    ) -> None:
        self._state = state
        self._stt_service = stt_service
        self._stop_event = stop_event
        self._threads: list[threading.Thread] = []

    def start(self) -> None:

        self._threads = [
            threading.Thread(target=self._audio_loop, name="ui-audio-udp", daemon=True),
            threading.Thread(target=self._direction_loop, name="ui-direction-udp", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:

        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()

    def _audio_loop(self) -> None:
        import numpy as np

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind(AUDIO_UDP_ADDR)
        buffers: dict[int, dict[str, Any]] = {}
        try:
            while not self._stop_event.is_set():
                try:
                    packet, _ = sock.recvfrom(65_535)
                except TimeoutError:
                    continue
                try:
                    sequence_id, total, offset, payload = unpack_audio_packet(packet)
                except ValueError:
                    continue

                entry = buffers.setdefault(
                    sequence_id,
                    {"total": total, "parts": {}, "created": time.monotonic()},
                )
                entry["parts"][offset] = payload
                received = sum(len(part) for part in entry["parts"].values())
                if received >= total:
                    data = b"".join(
                        entry["parts"][part_offset]
                        for part_offset in sorted(entry["parts"])
                    )[:total]
                    buffers.pop(sequence_id, None)
                    mono = np.frombuffer(data, dtype=np.float32).copy()
                    self._state.set_status(audio_connected=True)
                    if mono.size and self._state.is_recording():
                        self._stt_service.push_audio(mono)

                now = time.monotonic()
                for stale_id in [
                    key
                    for key, value in buffers.items()
                    if now - float(value["created"]) > 3.0
                ]:
                    buffers.pop(stale_id, None)
        finally:
            sock.close()

    def _direction_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind(DIRECTION_UDP_ADDR)
        try:
            while not self._stop_event.is_set():
                try:
                    packet, _ = sock.recvfrom(256)
                except TimeoutError:
                    continue
                try:
                    self._state.set_direction(unpack_direction(packet))
                except ValueError:
                    continue
        finally:
            sock.close()


class OpenCVFaceDetector:

    def __init__(self, cv2_module: Any) -> None:
        cascade_path = (
            Path(cv2_module.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        self._cv2 = cv2_module
        self._detector = cv2_module.CascadeClassifier(str(cascade_path))
        if self._detector.empty():
            raise RuntimeError(f"OpenCV 얼굴 검출 모델을 열 수 없습니다: {cascade_path}")

    def detect(self, frame: Any) -> list[tuple[int, int, int, int]]:

        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        boxes = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(48, 48),
        )
        ordered = sorted(
            (tuple(int(value) for value in box) for box in boxes),
            key=lambda box: box[0] + box[2] / 2,
        )
        return ordered[:MAX_FACES]


class CameraWorker:

    def __init__(
        self,
        state: RuntimeState,
        stop_event: threading.Event,
        camera_mode: str,
        uvc_device: int,
    ) -> None:
        self._state = state
        self._stop_event = stop_event
        self._camera_mode = camera_mode
        self._uvc_device = uvc_device
        self._thread: threading.Thread | None = None

    def start(self) -> None:

        self._thread = threading.Thread(
            target=self._run,
            name="ui-camera",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import cv2

            detector = OpenCVFaceDetector(cv2)
            if self._camera_mode in {"auto", "realsense"}:
                try:
                    self._run_realsense(cv2, detector)
                    return
                except Exception as exc:
                    if self._camera_mode == "realsense":
                        raise
                    print(f"[카메라] RealSense 연결 실패, UVC로 전환: {exc}", flush=True)
            self._run_uvc(cv2, detector)
        except Exception as exc:
            self._state.set_status(
                camera_connected=False,
                error=f"카메라 시작 실패: {exc}",
            )
            print(f"[카메라 오류] {exc}", flush=True)

    def _run_realsense(self, cv2: Any, detector: OpenCVFaceDetector) -> None:
        import numpy as np

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2가 설치되어 있지 않습니다.") from exc

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        self._state.set_status(camera_connected=True, error=None)
        frame_index = 0
        boxes: list[tuple[int, int, int, int]] = []

        try:
            while not self._stop_event.is_set():
                frames = align.process(pipeline.wait_for_frames(timeout_ms=2_000))
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                depth_m = (
                    np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                )
                frame_index += 1
                if frame_index % 3 == 1:
                    boxes = detector.detect(frame)
                intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                directions = [
                    self._realsense_face_direction(
                        rs,
                        intrinsics,
                        depth_m,
                        box,
                    )
                    for box in boxes
                ]
                self._publish_frame(cv2, frame, boxes, directions)
        finally:
            pipeline.stop()
            self._state.set_status(camera_connected=False)

    def _run_uvc(self, cv2: Any, detector: OpenCVFaceDetector) -> None:
        capture = cv2.VideoCapture(self._uvc_device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"USB 카메라 장치 {self._uvc_device}를 열 수 없습니다.")

        self._state.set_status(camera_connected=True, error=None)
        frame_index = 0
        boxes: list[tuple[int, int, int, int]] = []
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("USB 카메라 프레임을 읽지 못했습니다.")
                frame_index += 1
                if frame_index % 3 == 1:
                    boxes = detector.detect(frame)
                height, width = frame.shape[:2]
                directions = [self._uvc_face_direction(box, width, height) for box in boxes]
                self._publish_frame(cv2, frame, boxes, directions)
        finally:
            capture.release()
            self._state.set_status(camera_connected=False)

    def _publish_frame(
        self,
        cv2: Any,
        frame: Any,
        boxes: list[tuple[int, int, int, int]],
        directions: list[tuple[float, float, float] | None],
    ) -> None:
        height, width = frame.shape[:2]
        active_index = self._match_active_face(directions)
        faces = []
        for index, (x, y, box_width, box_height) in enumerate(boxes):
            speaker_id = index + 1
            faces.append(
                {
                    "speaker_id": speaker_id,
                    "x": round(x / width * 100, 3),
                    "y": round(y / height * 100, 3),
                    "width": round(box_width / width * 100, 3),
                    "height": round(box_height / height * 100, 3),
                    "active": active_index == index,
                }
            )
        active_speaker_id = active_index + 1 if active_index is not None else None
        self._state.update_faces(faces, active_speaker_id)

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self._state.set_frame(encoded.tobytes())

    def _match_active_face(
        self,
        face_directions: list[tuple[float, float, float] | None],
    ) -> int | None:
        if not face_directions:
            return None
        query = self._state.direction()
        candidates = [
            (index, self._angular_error(direction, query))
            for index, direction in enumerate(face_directions)
            if direction is not None
        ]
        if not candidates:
            return None
        index, error = min(candidates, key=lambda item: item[1])
        return index if error <= MATCH_THRESHOLD_DEG else None

    @staticmethod
    def _angular_error(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> float:
        dot = sum(a * b for a, b in zip(first, second))
        return math.degrees(math.acos(max(-1.0, min(1.0, dot))))

    @staticmethod
    def _uvc_face_direction(
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float]:
        x, y, width, height = box
        center_x = x + width / 2
        center_y = y + height / 2
        azimuth = math.radians(
            -((center_x / frame_width) - 0.5) * UVC_HORIZONTAL_FOV_DEG
        )
        elevation = math.radians(-((center_y / frame_height) - 0.5) * 50.0)
        return (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )

    @staticmethod
    def _realsense_face_direction(
        rs: Any,
        intrinsics: Any,
        depth_m: Any,
        box: tuple[int, int, int, int],
    ) -> tuple[float, float, float] | None:
        import numpy as np

        x, y, width, height = box
        center_x = min(max(int(x + width / 2), 0), depth_m.shape[1] - 1)
        center_y = min(max(int(y + height / 2), 0), depth_m.shape[0] - 1)
        patch = depth_m[
            max(0, center_y - 8) : min(depth_m.shape[0], center_y + 8),
            max(0, center_x - 8) : min(depth_m.shape[1], center_x + 8),
        ]
        valid = patch[patch > 0]
        if not valid.size:
            return None
        distance = float(np.median(valid))
        camera_x, camera_y, camera_z = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [center_x, center_y],
            distance,
        )

        vector = np.array(
            [camera_z + 0.12, camera_x, -camera_y + 0.07],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return None
        normalized = vector / norm
        return tuple(float(value) for value in normalized)


class DemoEngine:

    _PHRASES = (
        "프로젝트 진행 상황을 공유하겠습니다.",
        "네, 먼저 말씀해 주세요.",
        "저는 일정 관련해서 질문이 있습니다.",
        "다음 주까지 완료하는 것으로 하죠.",
        "알겠습니다. 정리해서 공유하겠습니다.",
    )

    def __init__(self, state: RuntimeState, stop_event: threading.Event) -> None:
        self._state = state
        self._stop_event = stop_event
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ui-demo", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def _run(self) -> None:
        self._state.set_status(stt_ready=True, audio_connected=False, doa_ready=False)
        for index, phrase in enumerate(self._PHRASES):
            if self._stop_event.wait(0.25 if index == 0 else 0.7):
                return
            self._state.add_demo_caption((index % 3) + 1, phrase)
        while not self._stop_event.wait(3.0):
            pass


class CaptionHTTPServer(ThreadingHTTPServer):

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        state: RuntimeState,
        stop_event: threading.Event,
        control_hook: Any = None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.state = state
        self.stop_event = stop_event

        self.control_hook = control_hook


class UIRequestHandler(BaseHTTPRequestHandler):

    server: CaptionHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.server.state.snapshot())
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/camera.mjpg":
            self._stream_camera()
            return

        relative = "index.html" if path == "/" else path.lstrip("/")
        if relative not in {"index.html", "styles.css", "app.js"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(UI_ROOT / relative)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = str(payload.get("action", ""))
            if action == "reset_captions":
                self.server.state.clear_captions()
                value = True
            else:
                value = self.server.state.toggle(action)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        hook = getattr(self.server, "control_hook", None)
        if hook is not None:
            try:
                hook(action, value)
            except Exception:
                pass
        self._send_json({"ok": True, "value": value})

    def log_message(self, format_string: str, *args: Any) -> None:
        del format_string, args

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _stream_camera(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        try:
            while not self.server.stop_event.is_set():
                frame, version = self.server.state.wait_for_frame(version, timeout=1.0)
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Jetson 실시간 화자 자막 UI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--demo", action="store_true", help="장치 없이 UI만 검증")
    parser.add_argument(
        "--camera",
        choices=("auto", "realsense", "uvc"),
        default="auto",
        help="실제 모드의 카메라 백엔드",
    )
    parser.add_argument("--uvc-device", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def open_kiosk(url: str) -> subprocess.Popen[Any] | None:

    browser = next(
        (
            executable
            for name in (
                "chromium-browser",
                "chromium",
                "google-chrome",
                "google-chrome-stable",
            )
            if (executable := shutil.which(name))
        ),
        None,
    )
    if browser is None:
        print(f"[브라우저] Chromium을 찾지 못했습니다. 직접 열어 주세요: {url}")
        webbrowser.open(url)
        return None
    return subprocess.Popen(
        [
            browser,
            "--kiosk",
            "--start-fullscreen",
            "--window-size=1024,600",
            "--window-position=0,0",
            "--force-device-scale-factor=1",
            "--disable-pinch",
            "--noerrdialogs",
            "--disable-infobars",
            "--disable-session-crashed-bubble",
            "--disable-features=TranslateUI",
            f"--app={url}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def validate_runtime_files() -> None:

    required = [UI_ROOT / "index.html", UI_ROOT / "styles.css", UI_ROOT / "app.js"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("필수 UI 파일이 없습니다: " + ", ".join(missing))


def main() -> None:

    args = parse_args()
    validate_runtime_files()
    stop_event = threading.Event()
    state = RuntimeState(demo=args.demo)
    stt_service = StreamingSTTService(state)
    receiver: RealtimeReceiver | None = None
    camera: CameraWorker | None = None
    demo: DemoEngine | None = None
    browser_process: subprocess.Popen[Any] | None = None

    if args.demo:
        demo = DemoEngine(state, stop_event)
        demo.start()
    else:
        stt_service.start()
        receiver = RealtimeReceiver(state, stt_service, stop_event)
        receiver.start()
        camera = CameraWorker(
            state,
            stop_event,
            camera_mode=args.camera,
            uvc_device=args.uvc_device,
        )
        camera.start()

    server = CaptionHTTPServer((args.host, args.port), UIRequestHandler, state, stop_event)
    server.timeout = 0.5
    url = f"http://{args.host}:{args.port}"

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"[UI 시작] {url}", flush=True)
    print("[종료] Ctrl+C", flush=True)
    if not args.no_browser:
        browser_process = open_kiosk(url)

    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        if demo is not None:
            demo.stop()
        if camera is not None:
            camera.stop()
        if receiver is not None:
            receiver.stop()
        stt_service.stop()
        if browser_process is not None and browser_process.poll() is None:
            browser_process.terminate()
            try:
                browser_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                browser_process.kill()
        print("[종료] Jetson 실시간 자막 UI를 종료했습니다.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[UI 실행 오류] {exc}") from exc
