from __future__ import annotations

import math
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.optimize import linear_sum_assignment

from sensor_msgs.msg import CompressedImage

from captioning_msgs.msg import Face, Faces, SoundSourceTracks


SEAT_BGR = {0: (140, 140, 140), 1: (240, 120, 40), 2: (90, 180, 32), 3: (24, 152, 255)}


def _find_face_model(name: str = "blaze_face_short_range.tflite") -> str:
    from pathlib import Path
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(Path(get_package_share_directory("camera_node")) / "models" / name)
    except Exception:
        pass
    here = Path(__file__).resolve()
    candidates += [here.parents[1] / "models" / name,
                   here.parents[2] / "models" / name]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0]) if candidates else name


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera")

        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 800)
        self.declare_parameter("fps", 30)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("frame_id", "mic_array")

        self.declare_parameter("lateral_sign", 1.0)
        self.declare_parameter("yaw_offset_deg", 0.0)
        self.declare_parameter("cam_offset", [0.0, 0.0, 0.07])
        self.declare_parameter("nominal_distance_m", 3.0)

        self.declare_parameter("search_radius_deg", 20.0)

        self.declare_parameter("speaking_threshold", 0.0)

        self.declare_parameter("face_model", _find_face_model())
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("publish_image", True)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_quality", 70)

        get = lambda name: self.get_parameter(name).value
        self.width, self.height = int(get("width")), int(get("height"))
        self.frame_id = get("frame_id")
        self.search_radius_deg = float(get("search_radius_deg"))
        self.speaking_threshold = float(get("speaking_threshold"))
        self.publish_image = bool(get("publish_image"))
        self.image_width = int(get("image_width"))
        self.image_quality = int(get("image_quality"))
        self.lateral_sign = float(get("lateral_sign"))
        self.cam_offset = np.asarray(get("cam_offset"), dtype=np.float64)
        self.nominal_distance = float(get("nominal_distance_m"))
        yaw = math.radians(float(get("yaw_offset_deg")))
        self.yaw_cos, self.yaw_sin = math.cos(yaw), math.sin(yaw)

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=get("face_model")),
            min_detection_confidence=float(get("min_confidence")))
        self.detector = vision.FaceDetector.create_from_options(options)

        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, int(get("fps")))
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, int(get("fps")))
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intrinsics.fx, intrinsics.fy
        self.cx, self.cy = intrinsics.ppx, intrinsics.ppy

        self.tracks: list = []
        self.stamp = None
        self.lock = threading.Lock()

        sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(SoundSourceTracks, "sound_source_tracks", self._on_tracks, sensor)
        self.publisher = self.create_publisher(Faces, "faces", sensor)
        self.image_publisher = self.create_publisher(CompressedImage, "camera/compressed", sensor)
        self.create_timer(1.0 / float(get("publish_rate")), self._tick)

        horizontal = 2 * math.degrees(math.atan(self.width / 2 / self.fx))
        vertical = 2 * math.degrees(math.atan(self.height / 2 / self.fy))
        self.get_logger().info(
            f"RealSense {self.width}x{self.height}  화각 {horizontal:.1f}x{vertical:.1f}도  "
            f"lateral_sign {self.lateral_sign:+.0f}  렌즈오프셋 {list(self.cam_offset)}")
        self.get_logger().warn(
            "lateral_sign 미검증")

    def _on_tracks(self, message: SoundSourceTracks) -> None:
        with self.lock:
            self.tracks = [(int(t.track_id),
                            np.array([t.direction.x, t.direction.y, t.direction.z]),
                            float(t.activity))
                           for t in message.tracks]
            self.stamp = message.header.stamp

    def _camera_vector(self, direction: np.ndarray) -> np.ndarray:
        point = self.nominal_distance * direction - self.cam_offset
        x = point[0] * self.yaw_cos + point[1] * self.yaw_sin
        y = -point[0] * self.yaw_sin + point[1] * self.yaw_cos
        z = point[2]

        return np.array([self.lateral_sign * x, -z, y])

    def _project(self, direction: np.ndarray) -> tuple[float, float] | None:
        camera = self._camera_vector(direction)
        if camera[2] <= 1e-6:
            return None
        return (self.fx * camera[0] / camera[2] + self.cx,
                self.fy * camera[1] / camera[2] + self.cy)

    def _detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        boxes = []
        for detection in result.detections:
            box = detection.bounding_box
            boxes.append((int(box.origin_x), int(box.origin_y),
                          int(box.width), int(box.height)))
        return boxes

    def _tick(self) -> None:
        try:
            frames = self.align.process(self.pipeline.wait_for_frames(200))
        except Exception:
            return
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color:
            return
        image = np.asanyarray(color.get_data())

        boxes = self._detect(image)

        with self.lock:
            tracks, stamp = list(self.tracks), self.stamp

        radius = self.fx * math.tan(math.radians(self.search_radius_deg))
        matched: dict[int, int] = {}
        if tracks and boxes:
            cost = np.full((len(tracks), len(boxes)), 1e6)
            for i, (_, direction, _activity) in enumerate(tracks):
                point = self._project(direction)
                if point is None:
                    continue
                for j, (x, y, w, h) in enumerate(boxes):
                    distance = math.hypot(x + w / 2 - point[0], y + h / 2 - point[1])
                    if distance <= radius:
                        cost[i, j] = distance
            rows, columns = linear_sum_assignment(cost)
            matched = {int(r): int(c) for r, c in zip(rows, columns) if cost[r, c] < 1e6}

        message = Faces()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        if stamp is not None:
            message.source_stamp = stamp
        message.image_width, message.image_height = image.shape[1], image.shape[0]

        message.n_detected = len(boxes)

        for index, (track_id, _, _activity) in enumerate(tracks):
            face = Face()
            face.track_id = track_id
            face.face_id = 0
            if index in matched:
                x, y, w, h = boxes[matched[index]]
                face.found = True
                face.bbox = [x, y, w, h]
                face.distance = self._depth_at(depth, x + w // 2, y + h // 2)
            else:
                face.found = False
                face.bbox = [0, 0, 0, 0]
                face.distance = 0.0
            message.faces.append(face)

        used = set(matched.values())
        for index, (x, y, w, h) in enumerate(boxes):
            if index in used:
                continue
            face = Face()
            face.track_id = 0
            face.face_id = index + 1
            face.found = True
            face.bbox = [x, y, w, h]
            face.distance = self._depth_at(depth, x + w // 2, y + h // 2)
            message.faces.append(face)

        self.publisher.publish(message)

        if self.publish_image:
            overlay = self._overlay(image, tracks, boxes, matched)
            self._publish_image(overlay)

    def _overlay(self, image, tracks, boxes, matched):
        canvas = image.copy()
        scale = canvas.shape[1] / 640.0

        seat = {j: rank + 1 for rank, j in enumerate(
            sorted(range(len(boxes)), key=lambda j: boxes[j][0] + boxes[j][2] / 2))}
        speaking_box = {}
        for index, (_tid, _dir, activity) in enumerate(tracks):
            if index in matched and activity >= self.speaking_threshold:
                speaking_box[matched[index]] = activity
        for j, (x, y, w, h) in enumerate(boxes):
            number = seat.get(j, 0)
            colour = (SEAT_BGR[number] if j in matched.values()
                      else (140, 140, 140))
            thickness = max(2, int(4 * scale)) if j in speaking_box else 1
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, thickness)
            if j in matched.values() and number:
                label = f"화자 {number}" + (" 발화" if j in speaking_box else "")
                cv2.putText(canvas, label, (x, max(y - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return canvas

    def _publish_image(self, image) -> None:
        if self.image_width and image.shape[1] > self.image_width:
            scale = self.image_width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", image,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality])
        if not ok:
            return
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.format = "jpeg"
        message.data = buffer.tobytes()
        self.image_publisher.publish(message)

    def _depth_at(self, depth, x: int, y: int) -> float:
        if depth is None:
            return 0.0
        try:
            return float(depth.get_distance(int(x), int(y)))
        except Exception:
            return 0.0

    def destroy_node(self) -> bool:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
