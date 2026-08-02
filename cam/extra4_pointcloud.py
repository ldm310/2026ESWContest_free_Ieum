"""
extra4_pointcloud.py  -  3D 포인트클라우드 (욕심 프로젝트)

무엇을 하나:
  - 뎁스+컬러로 3D 점들을 만들고, Open3D 창에서 마우스로 돌려본다.
  - 마우스 드래그=회전, 휠=확대/축소.
  - 뒤에서 배운 것: 화면(2D 픽셀)이 아니라 '실제 3D 공간'으로 세상을 본다는 감각.

준비물 설치(한 번만):
    py -3.11 -m pip install open3d

실행:  py -3.11 extra4_pointcloud.py     (창 닫으면 다음 프레임으로. 완전 종료: 명령창에서 Ctrl+C)
"""
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

W, H = 640, 480
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# RealSense 내부 파라미터 -> Open3D 형식으로 변환
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
i = color_stream.get_intrinsics()
o3d_intr = o3d.camera.PinholeCameraIntrinsic(W, H, i.fx, i.fy, i.ppx, i.ppy)

print("포인트클라우드 창을 준비 중...  (창을 닫으면 다음 장면으로 갱신됩니다)", flush=True)
try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df:
            continue

        color_raw = o3d.geometry.Image(np.asanyarray(cf.get_data())[:, :, ::-1].copy())  # BGR->RGB
        depth_raw = o3d.geometry.Image(np.asanyarray(df.get_data()))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw,
            depth_scale=1.0 / profile.get_device().first_depth_sensor().get_depth_scale(),
            depth_trunc=4.0,                # 4m 넘는 점은 버림
            convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 위아래 보정

        # 한 장면을 띄우고, 창을 닫으면 while 루프가 다음 장면을 만든다
        o3d.visualization.draw_geometries([pcd], window_name="point cloud (drag=rotate)")
finally:
    pipeline.stop()
