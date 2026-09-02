from __future__ import annotations

import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import CompressedImage

from std_msgs.msg import Bool, Empty

from captioning_msgs.msg import Caption, Faces, SoundSourceTracks

MAX_SLOTS = 3


@dataclass
class _Result:

    type: str
    text: str
    utterance_id: int
    is_final: bool
    latency_ms: float = 0.0
    error: str | None = None


class SlotTable:

    def __init__(self, size: int = MAX_SLOTS) -> None:
        self.size = size
        self.by_track: dict[int, int] = {}
        self.last_seen: dict[int, float] = {}

    def assign(self, track_id: int, now: float) -> int:
        slot = self.by_track.get(track_id)
        if slot is not None:
            self.last_seen[track_id] = now
            return slot
        used = set(self.by_track.values())
        free = [s for s in range(1, self.size + 1) if s not in used]
        if free:
            slot = free[0]
        else:
            victim = min(self.last_seen, key=self.last_seen.get)
            slot = self.by_track.pop(victim)
            self.last_seen.pop(victim, None)
        self.by_track[track_id] = slot
        self.last_seen[track_id] = now
        return slot

    def release_stale(self, now: float, timeout: float) -> None:
        for track_id in [t for t, seen in self.last_seen.items() if now - seen > timeout]:
            self.by_track.pop(track_id, None)
            self.last_seen.pop(track_id, None)


def _vendor_dir(name: str) -> str:
    try:
        share = Path(get_package_share_directory("ui_bridge_node"))
        candidate = share / name
        if candidate.is_dir():
            return str(candidate)
    except Exception:
        pass
    return str(Path(__file__).resolve().parent.parent / name)


class UiBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("ui_bridge")

        self.declare_parameter("ui_path", _vendor_dir("jetson_ui"))
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("open_browser", False)
        self.declare_parameter("slot_timeout_sec", 5.0)

        self.declare_parameter("speaking_threshold", 0.0)
        self.declare_parameter("speaking_hold_sec", 0.6)

        get = lambda name: self.get_parameter(name).value
        ui_path = get("ui_path")
        if ui_path not in sys.path:
            sys.path.insert(0, ui_path)
        import os
        os.chdir(ui_path)
        import stage6_caption_ui as ui

        self.ui = ui
        self.state = ui.RuntimeState(demo=False)
        self.state.set_status(audio_connected=True, doa_ready=False, stt_ready=True)
        self.slots = SlotTable()
        self.track_side: dict[int, int] = {}
        self.slot_timeout = float(get("slot_timeout_sec"))
        self.speaking_threshold = float(get("speaking_threshold"))
        self.speaking_hold = float(get("speaking_hold_sec"))

        self.speaking_seen: dict[int, float] = {}

        self.spoken_seats: set[int] = set()
        self.lock = threading.Lock()

        sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.VOLATILE,
                              history=QoSHistoryPolicy.KEEP_LAST, depth=50)
        self.create_subscription(SoundSourceTracks, "sound_source_tracks",
                                 self._on_tracks, sensor)
        self.create_subscription(Caption, "caption", self._on_caption, reliable)
        self.create_subscription(Faces, "faces", self._on_faces, sensor)
        self.create_subscription(CompressedImage, "camera/compressed", self._on_image, sensor)
        self.create_subscription(Empty, "caption_reset", self._on_reset, 10)

        self.reset_publisher = self.create_publisher(Empty, "caption_reset", 10)
        self.model_publisher = self.create_publisher(Bool, "model_enabled", 10)

        self.create_subscription(Bool, "model_enabled", self._on_model_state, 10)
        self.stt_raw_publisher = self.create_publisher(Bool, "stt_raw_audio", 10)
        self.create_subscription(Bool, "stt_raw_audio", self._on_stt_state, 10)

        self.stop_event = threading.Event()
        host, port = get("host"), int(get("port"))
        self.server = ui.CaptionHTTPServer((host, port), ui.UIRequestHandler,
                                           self.state, self.stop_event,
                                           self._on_ui_control)
        self.server.timeout = 0.5
        self.http_thread = threading.Thread(target=self._serve, daemon=True)
        self.http_thread.start()
        if bool(get("open_browser")):
            ui.open_kiosk(f"http://{host}:{port}")
        self.get_logger().info(f"UI http://{host}:{port}  (UI 코드 {ui_path})")

    def _serve(self) -> None:
        while not self.stop_event.is_set():
            self.server.handle_request()

    def _on_tracks(self, message: SoundSourceTracks) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.slots.release_stale(now, self.slot_timeout)
            loudest, best = None, -1.0
            for track in message.tracks:

                slot = self.track_side.get(int(track.track_id), 0)
                if track.activity > best:
                    best, loudest = track.activity, slot
                if slot and track.activity >= self.speaking_threshold:
                    self.speaking_seen[slot] = now
            for slot in [s for s, seen in self.speaking_seen.items()
                         if now - seen > self.speaking_hold]:
                self.speaking_seen.pop(slot, None)
            if message.tracks:
                direction = message.tracks[0].direction
                self.state.set_direction((direction.x, direction.y, direction.z))

            self.state.update_faces(self._faces_payload(), loudest or 0)

    def _on_faces(self, message: Faces) -> None:
        self.state.set_status(camera_connected=True)
        now = self.get_clock().now().nanoseconds / 1e9
        iw = float(message.image_width) or 1.0
        ih = float(message.image_height) or 1.0
        with self.lock:

            found = [f for f in message.faces if f.found]
            found.sort(key=lambda f: f.bbox[0] + f.bbox[2] / 2.0)
            payload = []
            for rank, face in enumerate(found):
                x, y, w, h = face.bbox

                seat = min(rank + 1, MAX_SLOTS)
                track_id = int(face.track_id)
                if track_id != 0:
                    self.track_side[track_id] = seat
                    self.spoken_seats.add(seat)
                speaker = seat if seat in self.spoken_seats else 0
                payload.append({
                    "speaker_id": speaker,
                    "x": 100.0 * x / iw, "y": 100.0 * y / ih,
                    "width": 100.0 * w / iw, "height": 100.0 * h / ih,
                    "distance": float(face.distance),
                })
            self.face_count = int(message.n_detected)
            self._faces = payload

        self.state.update_faces(self._faces_payload(), self.state.active_speaker_id())

    def _on_model_state(self, message: Bool) -> None:
        self.state.set_status(model=bool(message.data))

    def _on_stt_state(self, message: Bool) -> None:
        self.state.set_status(stt_raw=bool(message.data))

    def _on_ui_control(self, action: str, value: bool) -> None:
        if action == "model":
            self.model_publisher.publish(Bool(data=bool(value)))
            self.get_logger().info(f"화면에서 DOA 모델 {'켜기' if value else '끄기'}")
        elif action == "stt_raw":
            self.stt_raw_publisher.publish(Bool(data=bool(value)))
            self.get_logger().info(
                f"화면에서 STT 입력 → {'원본 채널' if value else '모델 분리 출력'}")
        elif action == "reset_captions":

            self.reset_publisher.publish(Empty())
            self.get_logger().info("화면에서 자막 리셋")

    def _on_reset(self, _message: Empty) -> None:
        state = self.state
        with state._lock:
            state._captions.clear()
            state._partial = None
            state._utterance_speakers.clear()
        with self.lock:
            self.slots = SlotTable()
            self.track_side.clear()
        self.get_logger().info("자막 리셋")

    def _on_image(self, message: CompressedImage) -> None:
        self.state.set_frame(bytes(message.data))
        self.state.set_status(camera_connected=True)

    def _faces_payload(self) -> list[dict]:
        now = self.get_clock().now().nanoseconds / 1e9
        speaking = {slot for slot, seen in self.speaking_seen.items()
                    if now - seen <= self.speaking_hold}
        return [{**face, "active": int(face["speaker_id"]) in speaking}
                for face in getattr(self, "_faces", [])]

    def _on_caption(self, message: Caption) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            slot = self.track_side.get(int(message.track_id), 0)

        unique_utterance = int(message.track_id) * 100000 + int(message.utterance_id)

        self.state.update_faces(self._faces_payload(), slot)
        self.state.handle_stt_result(_Result(
            type="final" if message.is_final else "partial",
            text=message.text,
            utterance_id=unique_utterance,
            is_final=message.is_final))

    def destroy_node(self) -> bool:
        self.stop_event.set()
        try:
            self.server.server_close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UiBridgeNode()
    signal.signal(signal.SIGINT, lambda *_: node.stop_event.set())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
