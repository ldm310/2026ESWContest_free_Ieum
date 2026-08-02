"""
extra2_nearest_point.py  -  가장 가까운 물체 찾아 표시 (손 추적처럼 동작)

무엇을 하나:
  - 매 프레임 전체 거리맵에서 '가장 가까운 지점'을 찾아 빨간 원으로 표시.
  - 카메라 앞에 손을 내밀면 손을 따라다닌다. (배경보다 손이 가까우므로)
  - 뒤에서 배운 것: 거리맵은 그냥 숫자 배열이라, min/argmin 같은 numpy로 다룰 수 있다.

실행:  py -3.11 extra2_nearest_point.py     (종료: 창 클릭 후 ESC)
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

        # 0(측정실패)은 아주 먼 값으로 바꿔서 '가장 가까운 점' 계산에서 제외
        d = depth_m.copy()
        d[d <= 0] = 99.0

        # 0.2m~2m 범위만 관심 (너무 가깝거나 먼 잡음 제거)
        d[d < 0.2] = 99.0
        d[d > 2.0] = 99.0

        min_val = d.min()
        if min_val < 99.0:
            cy, cx = np.unravel_index(np.argmin(d), d.shape)  # 가장 가까운 픽셀 위치
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), 3)
            cv2.putText(img, f"nearest {min_val:.2f}m", (cx - 40, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("nearest point", img)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
