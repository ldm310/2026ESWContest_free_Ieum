"""
stage2_faces.py  -  2단계: 얼굴마다 방위각 + 거리 붙이기  (mediapipe 1.0 새 방식)

무엇을 하나:
  - 영상에서 얼굴을 자동으로 찾는다 (MediaPipe Tasks API)
  - 얼굴마다 초록 네모를 그리고, 그 위에 'az +12.3 deg  1.85 m' 를 띄운다
      · az = 방위각. 카메라 정면 0도, 오른쪽 +, 왼쪽 -
      · m  = 그 사람까지의 거리(미터)

실행:  py -3.11 stage2_faces.py     (종료: 영상 창 클릭 후 ESC)

처음 한 번은 얼굴 검출 모델(약 200KB)을 인터넷에서 자동으로 받는다.
"""

import os
import urllib.request

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

W, H = 640, 480

# ── 0) 얼굴 검출 모델 준비 (없으면 자동 다운로드) ────────────────
MODEL_PATH = "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
if not os.path.exists(MODEL_PATH):
    print("얼굴 검출 모델 다운로드 중...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("다운로드 완료:", MODEL_PATH, flush=True)

# ── 1) 얼굴 검출기 만들기 (새 Tasks API) ───────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

# ── 2) RealSense 설정 (1단계와 동일) ────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)   # 컬러/뎁스 좌표 맞추기(정렬)


def face_distance_m(depth_m, cx, cy, half=8):
    """얼굴 중심 주변 작은 사각형에서 값이 있는 픽셀들의 '중앙값' 거리(미터).
    한 점만 쓰면 그 점이 하필 구멍(0)일 때 틀리므로 여러 점의 중앙값을 쓴다."""
    y0 = max(cy - half, 0); y1 = min(cy + half, H)
    x0 = max(cx - half, 0); x1 = min(cx + half, W)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * depth_scale   # 전체 거리맵(미터)

        intr = color_frame.profile.as_video_stream_profile().intrinsics

        # ── 얼굴 찾기 ──
        # MediaPipe는 RGB 순서 + mp.Image 형태를 원함
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        for det in result.detections:
            # 새 방식은 위치를 '픽셀'로 준다 (예전처럼 비율로 곱하지 않는다)
            bb = det.bounding_box
            bx, by, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height

            cx = min(max(bx + bw // 2, 0), W - 1)
            cy = min(max(by + bh // 2, 0), H - 1)

            dist = face_distance_m(depth_m, cx, cy)

            if dist > 0:
                # 역투영: 픽셀(cx,cy) + 거리 -> 실제 3D 좌표(X,Y,Z) 미터
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
                azimuth = np.degrees(np.arctan2(X, Z))   # 오른쪽 +, 왼쪽 -
                label = f"az {azimuth:+.1f} deg  {dist:.2f} m"
                dot_color = (0, 255, 255)
            else:
                label = "no depth"
                dot_color = (0, 0, 255)

            cv2.rectangle(color_image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(color_image, (cx, cy), 4, dot_color, -1)
            cv2.putText(color_image, label, (bx, max(by - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("faces", color_image)
        if cv2.waitKey(1) == 27:   # ESC
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()