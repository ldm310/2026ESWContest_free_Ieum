"""
extra7_pose_skeleton.py  -  몸 관절 스켈레톤 트래킹 + 간단 제스처 인식

용어: 이건 '포인트클라우드'가 아니라 포즈 추정(pose estimation) /
      스켈레톤 트래킹. 관절 점 = 랜드마크, 이은 선 = 본(bone).

무엇을 하나:
  - 몸 전체 33개 관절에 점을 찍고, 뼈대로 잇는다.
  - 각 손목까지의 거리(m)를 depth로 표시.
  - 간단한 제스처 판정: 왼손 들기 / 오른손 들기 / 만세 / T포즈.

준비물: 이미 깔린 mediapipe면 충분. (모델은 처음 실행 시 자동 다운로드)

실행:  py -3.11 extra7_pose_skeleton.py     (ESC 종료)
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

# Pose 33점 중 자주 쓰는 관절 번호
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# 뼈대(어느 관절끼리 이을지) 목록
BONES = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

# ── Pose 모델 (없으면 자동 다운로드) ──
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
if not os.path.exists(MODEL_PATH):
    print("포즈 모델 다운로드 중...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
landmarker = vision.PoseLandmarker.create_from_options(options)

# ── RealSense ──
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)


def dist_at(depth_m, px, py, half=6):
    y0 = max(py-half, 0); y1 = min(py+half, H)
    x0 = max(px-half, 0); x1 = min(px+half, W)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def classify_gesture(P):
    """관절 픽셀좌표 P(dict)로 간단 제스처 판정. 화면 y는 위가 작다."""
    def up(wrist, shoulder):        # 손목이 어깨보다 위에 있으면 손 든 것
        return P[wrist][1] < P[shoulder][1] - 20

    l_up = up(L_WRIST, L_SHOULDER)
    r_up = up(R_WRIST, R_SHOULDER)

    # T포즈: 양 손목이 어깨와 비슷한 높이 + 좌우로 넓게 벌림
    span = abs(P[L_WRIST][0] - P[R_WRIST][0])
    level = (abs(P[L_WRIST][1] - P[L_SHOULDER][1]) < 40 and
             abs(P[R_WRIST][1] - P[R_SHOULDER][1]) < 40)
    t_pose = level and span > abs(P[L_SHOULDER][0] - P[R_SHOULDER][0]) * 1.8

    if l_up and r_up:
        return "BOTH HANDS UP (manse)"
    if t_pose:
        return "T-POSE"
    if l_up:
        return "LEFT HAND UP"
    if r_up:
        return "RIGHT HAND UP"
    return ""


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

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            # 33개 관절을 픽셀좌표로
            P = {idx: (int(lm[idx].x * W), int(lm[idx].y * H)) for idx in range(len(lm))}

            # 뼈대 그리기
            for a, b in BONES:
                cv2.line(img, P[a], P[b], (0, 200, 255), 2)
            # 관절 점 그리기
            for idx in range(len(lm)):
                cv2.circle(img, P[idx], 4, (0, 0, 255), -1)

            # 손목까지 거리 표시
            for wrist, name in [(L_WRIST, "L"), (R_WRIST, "R")]:
                d = dist_at(depth_m, P[wrist][0], P[wrist][1])
                if d > 0:
                    cv2.putText(img, f"{name} {d:.2f}m", (P[wrist][0]+8, P[wrist][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 제스처 판정
            g = classify_gesture(P)
            if g:
                cv2.putText(img, g, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 220, 0), 2)

        cv2.putText(img, "pose skeleton + gesture", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("pose skeleton", img)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
