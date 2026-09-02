# proj_main

Jetson Orin Nano, ROS 2 Humble

## 실행

```
proj start
proj ui
proj stop
proj rec
```

인자: `proj start activity_threshold:=0.3 yaw_offset_deg:=11.6`

## 화면

- http://localhost:8765
- http://localhost:8770

## 빌드

```
cd ros2_ws
colcon build --symlink-install
```

가중치: `ros2_ws/src/doa_separation_node/weights/jointnet_v4.pt`

## 패키지

| 패키지 | 역할 |
|---|---|
| doa_separation_node | 마이크, 방향, 분리, 트래킹 |
| camera_node | 얼굴 검출, 트랙 매칭 |
| stt_bridge_node | Whisper 자막 |
| ui_bridge_node | 자막 화면 |
| doa_debug_node | 디버그 화면 |
| captioning_msgs | 메시지 |
