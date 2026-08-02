"""
stage3_match.py  -  3단계: 카메라 각도 <-> 마이크 각도 맞추기 + 발화자 매칭

무엇을 하나:
  - 2단계처럼 얼굴마다 각도/거리를 구한 뒤, 카메라 각도를 '마이크 각도'로 변환한다
  - 마이크가 알려준 각도(지금은 키보드로 흉내)에 가장 가까운 얼굴을
    빨간 박스 + 'SPEAKING' 으로 표시한다  (이게 '누가 말하나' 매칭 로직)

조작:
  a / d : 마이크 각도(파란 세로선)를 왼쪽 / 오른쪽으로 움직임
  s     : 지금 화면의 가장 큰 얼굴을 캘리브레이션용으로 기록(calib.csv)
  ESC   : 종료 (종료 시 calib.csv 저장)

실행:  py -3.11 stage3_match.py
"""

import os
import csv
import urllib.request

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

W, H = 640, 480

# ===================================================================
#  캘리브레이션 상수  ★ 3단계의 핵심 두 숫자 ★
#    mic_az = LATERAL_SIGN * cam_az + AZIMUTH_OFFSET_DEG
#  처음엔 아래 기본값(1, 0)으로 두고, 캘리브레이션으로 진짜 값을 찾아 여기에 적는다.
# ===================================================================
LATERAL_SIGN = 1.0          # +1 또는 -1 (좌우가 뒤집혀 있으면 -1)
AZIMUTH_OFFSET_DEG = 0.0    # 두 기준의 0도가 어긋난 만큼(도)

SPEAKING_THRESHOLD_DEG = 12.0   # 마이크 각도와 이 정도 이내면 '이 사람이 말한다'

# ── 얼굴 검출 모델 (없으면 자동 다운로드) ──
MODEL_PATH = "blaze_face_short_range.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
if not os.path.exists(MODEL_PATH):
    print("얼굴 검출 모델 다운로드 중...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

# ── RealSense ──
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)


def face_distance_m(depth_m, cx, cy, half=8):
    y0 = max(cy - half, 0); y1 = min(cy + half, H)
    x0 = max(cx - half, 0); x1 = min(cx + half, W)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def cam_az_to_mic_az(cam_az):
    """카메라 각도 -> 마이크 각도 (3단계의 변환식)"""
    return LATERAL_SIGN * cam_az + AZIMUTH_OFFSET_DEG


query_mic_az = 0.0     # 마이크가 "이 방향에서 소리"라고 알려준 값 (지금은 키보드로 흉내)
calib_rows = []        # 캘리브레이션 기록 (cam_az, 기준각도)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        # 얼굴마다 각도/거리 계산
        faces = []
        for det in result.detections:
            bb = det.bounding_box
            bx, by, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
            cx = min(max(bx + bw // 2, 0), W - 1)
            cy = min(max(by + bh // 2, 0), H - 1)
            dist = face_distance_m(depth_m, cx, cy)
            if dist <= 0:
                continue
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
            cam_az = np.degrees(np.arctan2(X, Z))
            faces.append({
                "box": (bx, by, bw, bh),
                "cam_az": cam_az,
                "mic_az": cam_az_to_mic_az(cam_az),
                "dist": dist,
                "area": bw * bh,
            })

        # 마이크 각도(query)에 가장 가까운 얼굴 = 발화자 후보
        speaker = None
        if faces:
            speaker = min(faces, key=lambda f: abs(f["mic_az"] - query_mic_az))
            if abs(speaker["mic_az"] - query_mic_az) > SPEAKING_THRESHOLD_DEG:
                speaker = None      # 너무 멀면 아무도 아님

        # 그리기
        for f in faces:
            bx, by, bw, bh = f["box"]
            is_spk = (f is speaker)
            color = (0, 0, 255) if is_spk else (0, 255, 0)   # 발화자=빨강, 나머지=초록
            cv2.rectangle(color_image, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(color_image, f"mic_az {f['mic_az']:+.1f}  {f['dist']:.2f}m",
                        (bx, max(by - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if is_spk:
                cv2.putText(color_image, "SPEAKING", (bx, by + bh + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 마이크 각도(query)를 화면에 파란 세로선으로 표시
        cam_az_q = (query_mic_az - AZIMUTH_OFFSET_DEG) / LATERAL_SIGN
        u = int(intr.ppx + intr.fx * np.tan(np.radians(cam_az_q)))
        u = min(max(u, 0), W - 1)
        cv2.line(color_image, (u, 0), (u, H), (255, 0, 0), 1)
        cv2.putText(color_image, f"mic query = {query_mic_az:+.1f} deg", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(color_image, "a/d: query   s: save calib   ESC: quit", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("stage3", color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC
            break
        elif key == ord('a'):
            query_mic_az -= 2
        elif key == ord('d'):
            query_mic_az += 2
        elif key == ord('s'):               # 캘리브레이션 기록
            if faces:
                big = max(faces, key=lambda f: f["area"])
                calib_rows.append((round(big["cam_az"], 2), round(query_mic_az, 2)))
                print(f"기록: cam_az={big['cam_az']:+.1f}  기준={query_mic_az:+.1f}"
                      f"  (총 {len(calib_rows)}개)", flush=True)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if calib_rows:
        with open("calib.csv", "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["cam_az", "ref_az"])
            w.writerows(calib_rows)
        print(f"calib.csv 저장 완료: {len(calib_rows)}행", flush=True)