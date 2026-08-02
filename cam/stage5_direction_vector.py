"""
stage3_direction_vector.py  -  카메라 출력을 '마이크 좌표계 3D 단위 방향벡터'로

바뀐 점 (이전 stage3_match 대비):
  - 각도(cam_az) 대신, 준형님 코드와 같은 형식인 3D 단위벡터 d_unit=[x,y,z]를 낸다.
  - 마이크 좌표계 규약(준형님 directions_to_angles 기준):
        x = 앞(정면),  y = 옆(왼/오른),  z = 위
        방위각 = atan2(y, x),  고도각 = arcsin(z)
  - 카메라 3D점(RealSense: X=오른, Y=아래, Z=앞)을 마이크 좌표계로 회전 후,
    마이크 중심 기준 오프셋(카메라가 위 +7cm)을 빼서 방향을 계산 → 단위벡터로 정규화.
  - 매칭은 각도 빼기가 아니라 벡터 사이 각(angular_error_deg)으로 한다(360도 문제 없음).

준형님 코드와의 접점:
  - 여기서 만든 d_unit 을 그의 estimate_bem_doa 결과 direction 과
    angular_error_deg(cam_d_unit, mic_d_unit) 로 바로 비교하면 된다.

실행:  py -3.11 stage3_direction_vector.py
조작:  a/d = 마이크 방위각(query) 좌우, w/x = 고도각 상하, s = 캘리브레이션 기록, ESC 종료
"""
import os
import csv
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs

W, H = 640, 480

# ===================================================================
#  캘리브레이션 값 ★ 3단계 핵심 ★
# ===================================================================
# (1) 축 부호: 카메라 오른쪽(+X_cam)이 마이크 +y(옆)와 같은 쪽이면 +1, 반대면 -1.
LATERAL_SIGN = 1.0
# (2) 마이크 중심 기준 '카메라 렌즈' 위치 (미터), 마이크 좌표계 [앞, 옆, 위].
#     그림 기준 기본값: 옆 0, 위 +0.07, 앞뒤 0.  실측 후 여기 숫자만 고치면 됨.
CAM_OFFSET = np.array([0.0, 0.0, 0.07])   # [앞(x), 옆(y), 위(z)]
# (3) 카메라가 마이크와 다른 방향을 보게 붙였다면 여기서 미세 회전(도). 정면이면 0.
YAW_OFFSET_DEG = 0.0

MATCH_THRESHOLD_DEG = 12.0   # 마이크 방향과 이 각도 이내면 '이 사람이 말한다'

# ── 얼굴 모델 (없으면 자동 다운로드) ──
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
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)


def face_distance_m(depth_m, cx, cy, half=8):
    y0 = max(cy-half, 0); y1 = min(cy+half, H)
    x0 = max(cx-half, 0); x1 = min(cx+half, W)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


# 카메라 좌표(X=오른,Y=아래,Z=앞) -> 마이크 좌표(x=앞,y=옆,z=위) 회전
_yaw = np.radians(YAW_OFFSET_DEG)
_cos, _sin = np.cos(_yaw), np.sin(_yaw)

def cam_point_to_mic_frame(Xc, Yc, Zc):
    """카메라 3D점을 마이크 좌표계로. 축 재배치 + 좌우부호 + 미세 yaw."""
    x = Zc                        # 앞 <- 카메라 Z
    y = LATERAL_SIGN * Xc         # 옆 <- 카메라 X (부호는 실측)
    z = -Yc                       # 위 <- 카메라 -Y (카메라 Y는 아래가 +)
    # 정면축(x)-옆축(y) 평면에서 미세 yaw 보정
    xr = x * _cos - y * _sin
    yr = x * _sin + y * _cos
    return np.array([xr, yr, z])


def to_unit_direction(p_mic):
    """마이크 중심 기준 방향 단위벡터. 카메라 오프셋을 빼서 시차 보정."""
    v = p_mic - CAM_OFFSET
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def angular_error_deg(a, b):
    """두 단위벡터 사이 각(도). 준형님 코드와 동일한 방식."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def query_to_unit(az_deg, el_deg):
    """키보드로 흉내내는 마이크 방향(방위각/고도각) -> 단위벡터."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


query_az, query_el = 0.0, 0.0
calib_rows = []

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df:
            continue
        img = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
        intr = cf.profile.as_video_stream_profile().intrinsics

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        faces = []
        for det in result.detections:
            bb = det.bounding_box
            bx, by, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
            cx = min(max(bx + bw//2, 0), W-1)
            cy = min(max(by + bh//2, 0), H-1)
            dist = face_distance_m(depth_m, cx, cy)
            if dist <= 0:
                continue
            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
            p_mic = cam_point_to_mic_frame(Xc, Yc, Zc)
            d_unit = to_unit_direction(p_mic)      # ★ 최종 출력: 마이크 좌표계 단위벡터
            faces.append({"box": (bx, by, bw, bh), "d_unit": d_unit,
                          "dist": dist, "area": bw*bh})

        # 마이크 방향(query)과 벡터 사이각으로 발화자 매칭
        q = query_to_unit(query_az, query_el)
        speaker = None
        if faces:
            speaker = min(faces, key=lambda f: angular_error_deg(f["d_unit"], q))
            if angular_error_deg(speaker["d_unit"], q) > MATCH_THRESHOLD_DEG:
                speaker = None

        for f in faces:
            bx, by, bw, bh = f["box"]
            spk = (f is speaker)
            color = (0, 0, 255) if spk else (0, 220, 0)
            ux, uy, uz = f["d_unit"]
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, 2)
            cv2.putText(img, f"d=[{ux:+.2f},{uy:+.2f},{uz:+.2f}] {f['dist']:.2f}m",
                        (bx, max(by-8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if spk:
                cv2.putText(img, "SPEAKING", (bx, by+bh+22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img, f"mic query az={query_az:+.0f} el={query_el:+.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img, "a/d:az  w/x:el  s:save  ESC:quit", (10, H-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("stage3 direction vector", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('a'): query_az -= 3
        elif key == ord('d'): query_az += 3
        elif key == ord('w'): query_el += 3
        elif key == ord('x'): query_el -= 3
        elif key == ord('s'):
            if faces:
                big = max(faces, key=lambda f: f["area"])
                ux, uy, uz = big["d_unit"]
                calib_rows.append((round(ux,4), round(uy,4), round(uz,4),
                                   round(query_az,1), round(query_el,1)))
                print(f"기록: d_unit=[{ux:+.2f},{uy:+.2f},{uz:+.2f}] "
                      f"query az={query_az:+.0f} el={query_el:+.0f} (총 {len(calib_rows)})",
                      flush=True)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if calib_rows:
        with open("calib_vec.csv", "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["ux", "uy", "uz", "ref_az", "ref_el"])
            w.writerows(calib_rows)
        print(f"calib_vec.csv 저장: {len(calib_rows)}행", flush=True)
