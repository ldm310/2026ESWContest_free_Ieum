"""
extra1_distance_grid.py  -  화면을 격자로 나눠 각 칸의 거리 표시 (제일 기초)

무엇을 하나:
  - 컬러 영상 위에 3x3 격자를 그리고, 각 칸 중앙의 거리(미터)를 적는다.
  - "카메라가 공간을 어떻게 숫자로 보는가"를 눈으로 익히는 용도.

실행:  py -3.11 extra1_distance_grid.py     (종료: 창 클릭 후 ESC)
"""
import pyrealsense2 as rs
import numpy as np
import cv2

W, H = 640, 480
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df:
            continue
        img = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

        # 3x3 격자
        for i in range(1, 3):
            cv2.line(img, (W * i // 3, 0), (W * i // 3, H), (80, 80, 80), 1)
            cv2.line(img, (0, H * i // 3), (W, H * i // 3), (80, 80, 80), 1)

        # 각 칸 중앙의 거리
        for r in range(3):
            for c in range(3):
                cx = W * c // 3 + W // 6
                cy = H * r // 3 + H // 6
                patch = depth_m[cy-6:cy+6, cx-6:cx+6]
                valid = patch[patch > 0]
                d = float(np.median(valid)) if valid.size else 0.0
                text = f"{d:.2f}m" if d > 0 else "--"
                cv2.putText(img, text, (cx - 28, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("distance grid", img)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
