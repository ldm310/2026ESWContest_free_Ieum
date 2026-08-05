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
#  캘리브레이션 값  ★ 3단계 핵심 ★
# ===================================================================
# (1) 좌우 부호: 카메라 오른쪽(+X_cam)이 마이크 +y(옆)와 같은 쪽이면 +1, 반대면 -1.
#     실물 붙이고 준형님 값과 대조해 확정. (스피커 테스트로 30분)
LATERAL_SIGN = 1.0

# (2) 카메라 렌즈 위치 (미터), '박스 중심' 기준, 마이크 좌표계 [앞x, 옆y, 위z].
#     그림 기준값: 앞면(x=+0.12)의 중앙 위 7cm → [0.12, 0, 0.07]. 실측 후 수정.
#     주의: 발화자가 1.5m보다 멀면 이 값의 영향은 미미하다(먼저 0으로 두고 시작해도 됨).
CAM_POS_IN_BOX = np.array([0.12, 0.0, 0.07])   # [앞, 옆, 위]

# (3) 카메라가 정면과 다른 방향을 보게 붙였다면 미세 yaw 보정(도). 정면이면 0.
YAW_OFFSET_DEG = 0.0

MATCH_THRESHOLD_DEG = 12.0   # 마이크 방향과 이 각도 이내면 '이 사람이 말한다'

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


# 카메라 좌표(X=오른, Y=아래, Z=앞) -> 마이크 좌표(x=앞, y=옆, z=위) 회전
_yaw = np.radians(YAW_OFFSET_DEG)
_cos, _sin = np.cos(_yaw), np.sin(_yaw)

def cam_point_to_mic_axes(Xc, Yc, Zc):
    """카메라 3D점을 마이크 '축 방향'으로만 재배치(원점은 아직 카메라)."""
    x = Zc                     # 앞  <- 카메라 Z
    y = LATERAL_SIGN * Xc      # 옆  <- 카메라 X (부호는 실측)
    z = -Yc                    # 위  <- 카메라 -Y (카메라 Y는 아래가 +)
    xr = x * _cos - y * _sin   # 정면-옆 평면 미세 yaw 보정
    yr = x * _sin + y * _cos
    return np.array([xr, yr, z])


def direction_from_box_center(p_cam_in_mic_axes):
    """카메라 기준 점 -> 박스 중심 기준 방향 단위벡터.
    박스중심→발화자 = (카메라→발화자) + (박스중심→카메라).
    준형님 estimate.direction 과 같은 '박스 중심에서 본 방향'이 된다."""
    v = p_cam_in_mic_axes + CAM_POS_IN_BOX
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def angular_error_deg(a, b):
    """두 단위벡터 사이 각(도). doa.py 의 angular_error_deg 와 동일."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def query_to_unit(az_deg, el_deg):
    """키보드로 흉내내는 마이크 방향 -> 단위벡터 (마이크 좌표계 규약)."""
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
            p_axes = cam_point_to_mic_axes(Xc, Yc, Zc)
            d_unit = direction_from_box_center(p_axes)   # ★ 준형님 estimate.direction 과 같은 형식
            faces.append({"box": (bx, by, bw, bh), "d_unit": d_unit,
                          "dist": dist, "area": bw*bh})

        # ===============================================================
        #  발화자 매칭  (준형님 DOA 방향 q 와 얼굴 방향 d_unit 을 비교)
        # ---------------------------------------------------------------
        #  지금 (실물 마이크 없음): 키보드로 q 를 흉내낸다
        q = query_to_unit(query_az, query_el)
        #  통합 (준형님 실시간 래퍼 준비되면 위 줄을 지우고 아래로):
        #  q = estimate.direction        # estimate = estimate_bem_doa(csm, dictionary)
        # ===============================================================
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
