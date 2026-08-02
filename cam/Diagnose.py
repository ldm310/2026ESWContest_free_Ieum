"""
diagnose.py  -  어디서 막히는지 찾는 진단용 스크립트

명령창(cmd)에서 실행:  python diagnose.py
화면에 찍힌 '마지막 번호'와 빨간 에러 글씨를 그대로 알려주세요.
"""

print("1. 스크립트 시작", flush=True)

import pyrealsense2 as rs
print("2. pyrealsense2 불러오기 성공 / 버전:", rs.__version__, flush=True)

import numpy as np
import cv2
print("3. numpy, cv2 불러오기 성공", flush=True)

# 파이썬이 카메라를 몇 개 인식하는지
ctx = rs.context()
devices = ctx.query_devices()
print("4. 파이썬이 인식한 카메라 수:", len(devices), flush=True)
for d in devices:
    print("     -", d.get_info(rs.camera_info.name),
          "/ 시리얼:", d.get_info(rs.camera_info.serial_number), flush=True)

# 카메라 통로 열기
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("5. start() 시도 중... (여기서 멈추면 카메라를 다른 프로그램이 붙잡고 있는 것)", flush=True)
pipe.start(cfg)
print("6. start() 성공", flush=True)

print("7. 프레임 대기 중... (여기서 계속 멈춰 있으면 realsense-viewer가 켜져 있는 것)", flush=True)
frames = pipe.wait_for_frames()
color = frames.get_color_frame()
print("8. 프레임 받음! 카메라 파이썬 연결 정상. 이미지 크기:",
      np.asanyarray(color.get_data()).shape, flush=True)

pipe.stop()
print("9. 정상 종료. 이제 week1_viewer.py 가 될 겁니다.", flush=True)