from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from std_msgs.msg import Bool, Empty, Float32MultiArray

from captioning_msgs.msg import Faces, SoundSourceTracks

PAGE = (Path(__file__).resolve().parent / "index.html")


class Hub:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=20)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def publish(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self.lock:
            targets = list(self.clients)
        for q in targets:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass


def make_handler(hub: Hub, control):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            action = self.path.rsplit("/", 1)[-1].split("?")[0]
            ok = control(action)
            body = b'{"ok": true}' if ok else b'{"ok": false}'
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/events"):
                self._events()
            elif self.path in ("/", "/index.html"):
                self._page()
            else:
                self.send_error(404)

        def _page(self) -> None:
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    try:
                        line = q.get(timeout=10.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(b"data: " + line.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


class DoaDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("doa_debug")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8770)
        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)

        self.hub = Hub()
        self.latest = {"tracks": [], "count": [], "mic": [], "ref": 0.0,
                       "activity": [], "thr": 0.0, "gate": 0.0, "open": True,
                       "warm": False, "live": 0, "held": 0, "confirmed": 0,
                       "hits": 0, "need": 0, "published": 0,
                       "faces": 0, "faces_hold": 0, "model_on": True,
                       "face_limit": -1, "face_dropped": 0, "out_of_range": 0}

        self.face_window: list[tuple[float, int]] = []
        self.face_hold_sec = float(self.declare_parameter(
            "face_hold_sec", 3.0).value)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(SoundSourceTracks, "sound_source_tracks",
                                 self._on_tracks, sensor_qos)
        self.create_subscription(Float32MultiArray, "doa_debug",
                                 self._on_debug, sensor_qos)
        self.create_subscription(Faces, "faces", self._on_faces, sensor_qos)

        self.reset_publisher = self.create_publisher(Empty, "caption_reset", 10)
        self.model_publisher = self.create_publisher(Bool, "model_enabled", 10)
        self.model_on = True

        self.server = ThreadingHTTPServer((host, port),
                                          make_handler(self.hub, self._control))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.get_logger().info(f"DOA 디버그 UI http://{host}:{port}")

    def _control(self, action: str) -> bool:
        if action == "reset":
            self.reset_publisher.publish(Empty())
            self.get_logger().info("자막 리셋 요청")
            return True
        if action in ("model_on", "model_off"):
            self.model_on = (action == "model_on")
            self.model_publisher.publish(Bool(data=self.model_on))
            self.latest["model_on"] = self.model_on
            self.get_logger().info(f"모델 {'켜기' if self.model_on else '끄기'} 요청")
            return True
        return False

    def _on_tracks(self, message: SoundSourceTracks) -> None:
        self.latest["tracks"] = [
            {"id": int(t.track_id),
             "az": round(float(t.azimuth_deg), 2),
             "el": round(float(t.elevation_deg), 2),
             "act": round(float(t.activity), 3),
             "age": round(float(t.age_sec), 2)}
            for t in message.tracks]
        self.latest["count"] = [round(float(p), 4) for p in message.count_probability]
        self._emit()

    def _on_debug(self, message: Float32MultiArray) -> None:
        d = list(message.data)

        self.latest["mic"] = [round(float(v), 6) for v in d[0:4]]
        self.latest["ref"] = round(float(d[4]), 6) if len(d) > 4 else 0.0
        self.latest["activity"] = [round(float(v), 3) for v in d[5:8]]

        if len(d) >= 21:
            self.latest.update(
                thr=round(float(d[12]), 3), gate=round(float(d[13]), 6),
                open=bool(d[14] > 0.5), warm=bool(d[15] > 0.5),
                live=int(d[16]), held=int(d[17]), confirmed=int(d[18]),
                hits=int(d[19]), need=int(d[20]),
                face_limit=int(d[21]), face_dropped=int(d[22]),
                out_of_range=int(d[23]),
                published=int(d[24]) if len(d) > 24 else 0)

    def _on_faces(self, message: Faces) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        self.face_window.append((now, int(message.n_detected)))
        cutoff = now - self.face_hold_sec
        self.face_window = [(t, n) for t, n in self.face_window if t >= cutoff]
        self.latest["faces"] = int(message.n_detected)
        self.latest["faces_hold"] = max((n for _, n in self.face_window), default=0)

    def _emit(self) -> None:
        stamp = self.get_clock().now().nanoseconds / 1e9
        self.hub.publish({**self.latest, "t": round(stamp, 3)})

    def destroy_node(self) -> bool:
        try:
            self.server.shutdown()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DoaDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
