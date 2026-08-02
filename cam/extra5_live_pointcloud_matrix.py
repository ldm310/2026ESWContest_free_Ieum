"""
extra5_live_pointcloud_matrix.py
    실시간으로 부드럽게 도는 포인트클라우드 + 인물 옆 '무작위 글자 자막'

무엇을 하나:
  - 뎁스+컬러로 3D 점을 만들고, 창을 열어둔 채 매 프레임 점만 갈아끼워
    마우스로 도는 도중에도 영상이 계속 흐른다. (드래그=회전, 휠=확대)
  - 가장 가까운 인물(가까운 점 뭉치)을 추적해, 그 옆에 무작위 글자가
    한 줄씩 주르륵 쌓인다. 매트릭스 자막 느낌.
  - 글자는 '줄 수'와 '줄당 글자 수'에 상한이 있어 오래 켜둬도 용량이 안 는다.

준비물 설치(한 번만):
    py -3.11 -m pip install open3d

실행:
    py -3.11 extra5_live_pointcloud_matrix.py
    종료: 포인트클라우드 창에서 Q 키  (또는 명령창 Ctrl+C)
"""

import random
import string

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# ── 용량/성능 조절 손잡이 (여기 숫자만 바꾸면 됨) ──────────────────
W, H = 640, 480
STRIDE = 2          # 점 다운샘플: 2 = 2픽셀마다 1점 (클수록 가벼움)
DEPTH_TRUNC = 4.0   # 이 거리(m) 너머 점은 버림

MAX_LINES = 14      # 자막 최대 줄 수 (넘으면 위에서 한 줄씩 지움)
LINE_LEN = 22       # 한 줄당 글자 수 (고정)
NEW_CHARS_PER_FRAME = 2   # 매 프레임 마지막 줄에 덧붙는 글자 수 (타이핑 속도)
CHARSET = string.ascii_letters + string.digits + "!@#$%&*<>/\\|+=-"

# ── RealSense 시작 ────────────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

# 내부 파라미터 -> Open3D 형식
i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
o3d_intr = o3d.camera.PinholeCameraIntrinsic(W, H, i.fx, i.fy, i.ppx, i.ppy)
FLIP = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]])  # 위아래 보정


def make_pointcloud(color_bgr, depth_u16):
    """컬러/뎁스 -> 다운샘플된 Open3D 포인트클라우드"""
    color = o3d.geometry.Image(color_bgr[::STRIDE, ::STRIDE, ::-1].copy())      # BGR->RGB
    depth = o3d.geometry.Image(depth_u16[::STRIDE, ::STRIDE].copy())
    intr = o3d.camera.PinholeCameraIntrinsic(
        W // STRIDE, H // STRIDE,
        i.fx / STRIDE, i.fy / STRIDE, i.ppx / STRIDE, i.ppy / STRIDE)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1.0 / depth_scale,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    pcd.transform(FLIP)
    return pcd


def nearest_person_point(depth_m):
    """가장 가까운 인물 위치의 3D 좌표(대략)를 반환. 없으면 None."""
    d = depth_m.copy()
    d[(d <= 0.3) | (d > 3.0)] = 99.0     # 0.3~3m만 관심
    if d.min() >= 99.0:
        return None
    cy, cx = np.unravel_index(np.argmin(d), d.shape)
    z = float(np.median(d[cy-4:cy+4, cx-4:cx+4][d[cy-4:cy+4, cx-4:cx+4] < 99.0]))
    x = (cx - i.ppx) * z / i.fx
    y = (cy - i.ppy) * z / i.fy
    return np.array([x, -y, -z])          # FLIP과 같은 좌표계로


def rand_line(n):
    return "".join(random.choice(CHARSET) for _ in range(n))


def build_text_geometry(anchor_xyz, lines):
    """자막 글자들을 3D 텍스트로 만들어 리스트로 반환 (앵커 오른쪽에 세로로 쌓임)"""
    geoms = []
    ax, ay, az = anchor_xyz
    for row, text in enumerate(lines):
        t = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.0).to_legacy()
        t.paint_uniform_color([0.1, 0.9, 0.3])          # 초록
        t.scale(0.0016, center=t.get_center())
        t.translate((ax + 0.12, ay + 0.18 - row * 0.045, az), relative=False)
        geoms.append(t)
    return geoms


# ── 실시간 창 ─────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("live point cloud + matrix (drag=rotate, Q=quit)", width=1024, height=768)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
text_geoms = []
lines = [rand_line(LINE_LEN)]
first = True

print("실시간 포인트클라우드 시작. 마우스로 돌려보세요. 종료는 Q.", flush=True)
try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df:
            continue
        color_bgr = np.asanyarray(cf.get_data())
        depth_u16 = np.asanyarray(df.get_data())
        depth_m = depth_u16.astype(np.float32) * depth_scale

        # 점 좌표만 갈아끼움 (창은 유지)
        new = make_pointcloud(color_bgr, depth_u16)
        pcd.points = new.points
        pcd.colors = new.colors
        vis.update_geometry(pcd)

        # ── 무작위 글자 자막 갱신 ──
        last = lines[-1]
        if len(last) >= LINE_LEN:                 # 줄이 다 차면 새 줄 시작
            lines.append("")
        lines[-1] += rand_line(min(NEW_CHARS_PER_FRAME, LINE_LEN - len(lines[-1])))
        if len(lines) > MAX_LINES:                # 상한 넘으면 맨 위 줄 삭제(용량 고정)
            lines.pop(0)

        # 이전 텍스트 제거 후, 인물 옆에 다시 그림
        for g in text_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        text_geoms = []
        anchor = nearest_person_point(depth_m)
        if anchor is not None:
            text_geoms = build_text_geometry(anchor, lines)
            for g in text_geoms:
                vis.add_geometry(g, reset_bounding_box=False)

        if first:                                 # 첫 프레임만 시점 자동 맞춤
            vis.reset_view_point(True)
            first = False

        if not vis.poll_events():                 # 창 닫힘/Q -> 종료
            break
        vis.update_renderer()
finally:
    vis.destroy_window()
    pipeline.stop()
