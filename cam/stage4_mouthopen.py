"""
extra6_mouth_open.py  -  입 벌림/다물림 표시 (4단계 ASD의 핵심 부품)

무엇을 하나:
  - 얼굴에 점(랜드마크)을 찍어, 윗입술과 아랫입술의 세로 간격을 잰다.
  - 간격 / 얼굴크기 = '벌림 비율'로 계산해, 거리·사람과 무관하게 판단.
  - 벌리면 초록 'OPEN', 다물면 회색 'closed' 를 얼굴 옆에 표시.
  - depth로 그 사람까지 거리(m)도 함께 표시.

준비물: 이미 깔린 mediapipe면 충분. (모델 파일은 처음 실행 시 자동 다운로드)

실행:  py -3.11 extra6_mouth_open.py
조작:  w / s 로 판정 기준선(threshold) 조절, ESC 종료
"""
import os
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs

W, H = 640, 480

# 입이 이 비율보다 크게 벌어지면 'OPEN'. w/s로 실시간 조절 가능.
OPEN_RATIO = 0.35

# MediaPipe Face Mesh 랜드마크 번호 (468점 기준)
UPPER_LIP = 13     # 윗입술 안쪽 중앙
LOWER_LIP = 14     # 아랫입술 안쪽 중앙
LEFT_EYE = 33      # 왼눈 바깥
RIGHT_EYE = 263    # 오른눈 바깥 (두 눈 사이 거리 = 얼굴크기 기준)

# ── 얼굴 랜드마커 모델 (없으면 자동 다운로드) ──
MODEL_PATH = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("얼굴 랜드마커 모델 다운로드 중...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=3)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ── RealSense ──
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)


def dist_at(depth_m, px, py, half=8):
    y0 = max(py-half, 0); y1 = min(py+half, H)
    x0 = max(px-half, 0); x1 = min(px+half, W)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df:
            continue
        img = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            for lm in result.face_landmarks:
                # 랜드마크는 0~1 비율 -> 픽셀로
                def px(idx):
                    return np.array([lm[idx].x * W, lm[idx].y * H])

                upper = px(UPPER_LIP); lower = px(LOWER_LIP)
                le = px(LEFT_EYE); re = px(RIGHT_EYE)

                mouth_gap = np.linalg.norm(upper - lower)     # 입술 세로 간격(픽셀)
                face_size = np.linalg.norm(le - re)           # 두 눈 사이 간격(픽셀)
                ratio = mouth_gap / face_size if face_size > 0 else 0.0

                is_open = ratio > OPEN_RATIO

                # 입 중앙 지점 + 거리
                mouth_c = ((upper + lower) / 2).astype(int)
                d = dist_at(depth_m, mouth_c[0], mouth_c[1])

                # 표시
                color = (0, 220, 0) if is_open else (150, 150, 150)
                label = "OPEN" if is_open else "closed"
                cv2.circle(img, tuple(upper.astype(int)), 3, (0, 255, 255), -1)
                cv2.circle(img, tuple(lower.astype(int)), 3, (0, 255, 255), -1)
                cv2.line(img, tuple(upper.astype(int)), tuple(lower.astype(int)), color, 2)
                cv2.putText(img, f"{label}  r={ratio:.2f}", (mouth_c[0] + 15, mouth_c[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if d > 0:
                    cv2.putText(img, f"{d:.2f}m", (mouth_c[0] + 15, mouth_c[1] + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(img, f"OPEN if ratio > {OPEN_RATIO:.2f}  (w/s to tune)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("mouth open/closed", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('w'):
            OPEN_RATIO = min(OPEN_RATIO + 0.02, 1.0)
        elif key == ord('s'):
            OPEN_RATIO = max(OPEN_RATIO - 0.02, 0.05)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
