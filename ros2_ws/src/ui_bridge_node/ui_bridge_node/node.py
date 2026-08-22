from __future__ import annotations

import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import CompressedImage

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


class UiBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("ui_bridge")

        self.declare_parameter("ui_path", str(Path.home() / "emb_repo/handoff/jetson_ui"))
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("open_browser", False)
        self.declare_parameter("slot_timeout_sec", 5.0)

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
        self.slot_timeout = float(get("slot_timeout_sec"))
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

        self.stop_event = threading.Event()
        host, port = get("host"), int(get("port"))
        self.server = ui.CaptionHTTPServer((host, port), ui.UIRequestHandler,
                                           self.state, self.stop_event)
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
                slot = self.slots.assign(int(track.track_id), now)
                if track.activity > best:
                    best, loudest = track.activity, slot
            if message.tracks:
                direction = message.tracks[0].direction
                self.state.set_direction((direction.x, direction.y, direction.z))
            if loudest is not None:
                self.state.update_faces(self._faces_payload(), loudest)

    def _on_faces(self, message: Faces) -> None:
        self.state.set_status(camera_connected=True)
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            payload = []
            for face in message.faces:
                if not face.found:
                    continue
                x, y, w, h = face.bbox
                payload.append({
                    "speaker_id": self.slots.assign(int(face.track_id), now),
                    "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                    "distance": float(face.distance),
                })
            self._faces = payload
            self.state.update_faces(payload, self.state.active_speaker_id())

    def _on_image(self, message: CompressedImage) -> None:
        self.state.set_frame(bytes(message.data))
        self.state.set_status(camera_connected=True)

    def _faces_payload(self) -> list[dict]:
        return list(getattr(self, "_faces", []))

    def _on_caption(self, message: Caption) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            slot = self.slots.assign(int(message.track_id), now)

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
