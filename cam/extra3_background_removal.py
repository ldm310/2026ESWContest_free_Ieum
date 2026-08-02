"""
extra3_background_removal.py  -  거리로 배경 지우기 (그린스크린 없이 누끼)

무엇을 하나:
  - 정한 거리(기본 1.5m)보다 먼 부분을 회색으로 덮어, 가까운 사람만 남긴다.
  - w / s 키로 경계 거리를 조절.
  - 뒤에서 배운 것: 거리 조건 하나로 전경/배경을 나눌 수 있다(뎁스 카메라의 핵심 강점).

실행:  py -3.11 extra3_background_removal.py   (w/s: 경계 조절, ESC: 종료)
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

clip_m = 1.5   # 이 거리보다 멀면 배경 처리

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df:
            continue
        img = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

        # 배경 마스크: 거리가 0(모름)이거나 경계보다 먼 픽셀
        bg = (depth_m <= 0) | (depth_m > clip_m)
        gray = np.full_like(img, 90)             # 회색 배경
        out = np.where(bg[:, :, None], gray, img)

        cv2.putText(out, f"clip = {clip_m:.1f}m   (w/s to change)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("background removal", out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('w'):
            clip_m = min(clip_m + 0.1, 6.0)
        elif key == ord('s'):
            clip_m = max(clip_m - 0.1, 0.3)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
