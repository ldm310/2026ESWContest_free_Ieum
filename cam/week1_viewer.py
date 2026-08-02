"""
week1_viewer.py  -  RealSense D455 1주차 뷰어

무엇을 하나:
  - 창 2개를 띄운다.  왼쪽('color') = 일반 영상, 오른쪽('depth') = 거리를 색으로 칠한 영상
  - 'color' 창에서 마우스를 움직이면, 그 지점의 실제 거리(미터)를 글씨로 보여준다

실행:  python week1_viewer.py     (종료: 영상 창을 클릭해 두고 ESC 키)
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# 마우스가 지금 가리키는 픽셀 좌표. 처음엔 화면 중앙(320,240)에서 시작.
mouse = [320, 240]

def on_mouse(event, x, y, flags, param):
    # 마우스가 움직일 때마다 좌표를 저장해 둔다
    mouse[0], mouse[1] = x, y


# ── 1) 카메라 통로(파이프라인) 열기 ─────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 영상
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)  # 거리 영상
pipeline.start(config)

# ── 2) 정렬 도구 (★ 로드맵에서 말한 'align') ────────────────────
# 컬러 센서와 뎁스 센서는 물리적으로 다른 위치라, 좌표를 맞춰줘야 한다.
# 이걸 안 켜면 마우스 위치와 거리값이 어긋난다. 첫날부터 켜는 이유.
align = rs.align(rs.stream.color)

# ── 3) 거리 영상을 보기 좋게 색으로 칠해주는 도구 ───────────────
# (가까우면 파랑, 멀면 빨강 같은 무지개색으로 알아서 칠해줌)
colorizer = rs.colorizer()

# ── 4) 'color' 창을 만들고, 그 위에서 마우스 움직임을 감지 ───────
cv2.namedWindow("color")
cv2.setMouseCallback("color", on_mouse)

try:
    while True:
        # 프레임(사진 한 장 묶음) 받기
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)          # ★ 정렬 적용
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue                            # 한 장이라도 비면 건너뜀

        # 프레임을 화면에 그릴 수 있는 이미지(숫자 배열)로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # 마우스 좌표가 화면(640x480) 밖으로 나가면 오류나므로 안쪽으로 가둔다
        x = min(max(mouse[0], 0), 639)
        y = min(max(mouse[1], 0), 479)

        # ── 그 지점의 실제 거리(미터) ──
        # get_distance()가 SDK 내부에서 알아서 '미터'로 준다. 단위 걱정 X.
        dist = depth_frame.get_distance(x, y)
        if dist > 0:
            label = f"({x},{y})  {dist:.2f} m"
        else:
            # 0은 '거리 0m'가 아니라 '측정 실패(모름)'. 검은머리/유리/역광/너무가까움에서 발생.
            label = f"({x},{y})  no depth"

        # 컬러 영상 위에 노란 점과 글씨 그리기
        # (주의: OpenCV 글씨는 한글을 못 그린다. 화면 글씨는 영어로.)
        cv2.circle(color_image, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(color_image, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ── 두 창 띄우기 ──
        cv2.imshow("color", color_image)
        cv2.imshow("depth", depth_image)

        # ESC(키 번호 27) 누르면 반복 종료
        if cv2.waitKey(1) == 27:
            break
finally:
    # 어떤 경우든 카메라를 깔끔히 끄고 창을 닫는다
    pipeline.stop()
    cv2.destroyAllWindows()