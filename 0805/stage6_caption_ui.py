"""
stage6_caption_ui.py  -  화자별 자막 UI (카메라 매칭 + 실제 STT 결합)

두 가지 모드:
  USE_REAL_PIPELINE = False  → 가짜 STT + 키보드 방향 (마이크 없이 UI 테스트)
  USE_REAL_PIPELINE = True   → 준형님 realtime_doa.py (방향+빔포밍 mono) + 동민님 STT

진짜 모드 실행법 (마이크가 준형님 노트북에 있을 때, 같은 노트북에서):
  터미널 1:  py -3.11 realtime_doa.py          (방향 공유 + mono 오디오 UDP 송신)
  터미널 2:  py -3.11 stage6_caption_ui.py      (아래 USE_REAL_PIPELINE=True 로)

연결 구조 (진짜 모드):
  준형 4채널 → [realtime_doa] DOA→방향(공유변수) / 빔포밍→mono(UDP)
  → [stage6] 방향으로 발화자 매칭 + mono를 동민님 STT에 투입 → 자막을 그 사람에게

설치:
  py -3.11 -m pip install pyrealsense2 opencv-contrib-python mediapipe pillow numpy
  (진짜 STT 쓰려면 동민님 stt 폴더가 import 되게 두고 requirements-core.txt 설치)
"""
import os
import time
import queue
import random
import socket
import struct
import threading
import urllib.request
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs

# ============================================================
#  모드 스위치
# ============================================================
USE_REAL_PIPELINE = False    # True = 준형 realtime_doa + 동민 STT / False = 가짜(테스트)

W, H = 640, 480

# ── 캘리브레이션 값 (stage3와 동일) ──
LATERAL_SIGN = 1.0
CAM_POS_IN_BOX = np.array([0.12, 0.0, 0.07])
MATCH_THRESHOLD_DEG = 15.0

# 준형 realtime_doa.py 가 mono 오디오를 보내는 UDP 포트 (진짜 모드)
AUDIO_UDP_PORT = 50007
STT_SAMPLE_RATE = 16000

# ── 한글 폰트 ──
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
if _FONT_PATH is None:
    print("경고: 한글 폰트를 못 찾음. NanumGothic 설치 권장.", flush=True)
_font_cache = {}
def _font(size):
    if _FONT_PATH is None:
        return None
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]

def draw_korean(img_bgr, items):
    if not items or _FONT_PATH is None:
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, (x, y), color, size in items:
        draw.text((x, y), text, font=_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ============================================================
#  STT 소스 — 진짜(동민) 또는 가짜
# ============================================================
result_q: "queue.Queue" = queue.Queue()   # STTResult 를 여기로

if USE_REAL_PIPELINE:
    # 동민님 모듈 (stt 폴더가 import 경로에 있어야 함)
    from stt import StreamingSTT, STTResult

    def on_result(r):
        result_q.put(r)

    stt = StreamingSTT(on_result=on_result)
    stt.start()

    # 준형 realtime_doa.py 의 방향 공유값
    from realtime_doa import get_latest_direction

    # realtime_doa.py 가 UDP로 보내는 mono 오디오를 받아 STT에 push
    def _udp_audio_receiver():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", AUDIO_UDP_PORT))
        buffers = {}   # offset 재조립용
        while True:
            packet, _ = sock.recvfrom(65535)
            total, offset = struct.unpack("<II", packet[:8])
            payload = packet[8:]
            buffers.setdefault(total, {})[offset] = payload
            got = sum(len(v) for v in buffers[total].values())
            if got >= total:            # 한 조각 다 모임
                data = b"".join(buffers[total][o] for o in sorted(buffers[total]))
                buffers.pop(total, None)
                mono = np.frombuffer(data[:total], dtype=np.float32)
                if mono.size:
                    stt.push_audio(mono, sample_rate=STT_SAMPLE_RATE)
    threading.Thread(target=_udp_audio_receiver, daemon=True).start()

else:
    # ── 가짜 STT (마이크 없이 UI 테스트) ──
    @dataclass(frozen=True)
    class STTResult:
        type: str; text: str; utterance_id: int; sequence_id: int; is_final: bool

    _PHRASES = ["안녕하세요", "지금 회의 시작하겠습니다", "카메라 파트 진행 상황입니다",
                "마이크 방향이랑 잘 맞네요", "다음 안건으로 넘어가죠", "네 좋습니다",
                "그 부분은 확인해 볼게요", "자막이 실시간으로 나옵니다"]

    class FakeStreamingSTT:
        def __init__(self, on_result):
            self.on_result = on_result; self._stop = threading.Event(); self._t = None
        def start(self):
            self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        def stop(self): self._stop.set()
        def _run(self):
            uid = seq = 0
            while not self._stop.is_set():
                uid += 1; built = ""
                for w in random.choice(_PHRASES).split(" "):
                    if self._stop.is_set(): return
                    built = (built + " " + w).strip(); seq += 1
                    self.on_result(STTResult("partial", built, uid, seq, False)); time.sleep(0.45)
                seq += 1; self.on_result(STTResult("final", built, uid, seq, True)); time.sleep(1.2)

    def on_result(r): result_q.put(r)
    stt = FakeStreamingSTT(on_result=on_result); stt.start()
    def get_latest_direction():   # 가짜 모드에선 키보드 방향을 쓰므로 미사용
        return np.array([1.0, 0.0, 0.0])


# ── 얼굴 검출 모델 ──
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
    y0=max(cy-half,0); y1=min(cy+half,H); x0=max(cx-half,0); x1=min(cx+half,W)
    patch = depth_m[y0:y1, x0:x1]; valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0
def cam_point_to_mic_axes(Xc, Yc, Zc):
    return np.array([Zc, LATERAL_SIGN * Xc, -Yc])
def direction_from_box_center(p_axes):
    v = p_axes + CAM_POS_IN_BOX; n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
def angular_error_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
def query_to_unit(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


query_az, query_el = 0.0, 0.0
active_text = ""
transcript = []

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
            cx=min(max(bx+bw//2,0),W-1); cy=min(max(by+bh//2,0),H-1)
            dist = face_distance_m(depth_m, cx, cy)
            if dist <= 0: continue
            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
            d_unit = direction_from_box_center(cam_point_to_mic_axes(Xc, Yc, Zc))
            faces.append({"box": (bx,by,bw,bh), "d_unit": d_unit, "dist": dist})

        # 방향 q: 진짜 모드=준형 DOA, 가짜 모드=키보드
        if USE_REAL_PIPELINE:
            q = get_latest_direction()
        else:
            q = query_to_unit(query_az, query_el)

        speaker = None
        if faces:
            speaker = min(faces, key=lambda f: angular_error_deg(f["d_unit"], q))
            if angular_error_deg(speaker["d_unit"], q) > MATCH_THRESHOLD_DEG:
                speaker = None

        spk_az = None
        if speaker is not None:
            ux, uy, _ = speaker["d_unit"]
            spk_az = np.degrees(np.arctan2(uy, ux))
        while not result_q.empty():
            r = result_q.get()
            if r.type == "partial":
                active_text = r.text
            elif r.type == "final":
                tag = f"{spk_az:+.0f}°" if spk_az is not None else "?"
                transcript.append((tag, r.text)); transcript[:] = transcript[-4:]
                active_text = ""

        for f in faces:
            bx,by,bw,bh = f["box"]; spk = (f is speaker)
            cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(0,0,255) if spk else (0,220,0),2)

        mode_txt = "REAL" if USE_REAL_PIPELINE else "FAKE"
        cv2.putText(img, f"[{mode_txt}] dir az={np.degrees(np.arctan2(q[1],q[0])):+.0f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
        if not USE_REAL_PIPELINE:
            cv2.putText(img, "a/d/w/x: mic dir   ESC: quit", (10,H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1)

        korean_items = []
        if speaker is not None and active_text:
            bx,by,bw,bh = speaker["box"]
            tw = min(max(len(active_text)*16 + 20, 120), W-10)
            tx = min(max(bx+bw//2 - tw//2, 5), W-tw-5); ty = max(by-44, 5)
            ov = img.copy(); cv2.rectangle(ov,(tx,ty),(tx+tw,ty+34),(30,30,30),-1)
            img = cv2.addWeighted(ov,0.55,img,0.45,0)
            korean_items.append((active_text,(tx+10,ty+5),(120,255,180),22))
        if transcript:
            strip_h = 22*len(transcript)+16
            ov = img.copy(); cv2.rectangle(ov,(0,H-strip_h-28),(W,H-28),(20,20,20),-1)
            img = cv2.addWeighted(ov,0.5,img,0.5,0)
            for i,(tag,txt) in enumerate(transcript):
                korean_items.append((f"[{tag}] {txt}",(10,H-strip_h-20+i*22),(235,235,235),18))
        img = draw_korean(img, korean_items)

        cv2.imshow("stage6 caption UI", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('a'): query_az -= 3
        elif key == ord('d'): query_az += 3
        elif key == ord('w'): query_el += 3
        elif key == ord('x'): query_el -= 3
finally:
    try: stt.stop()
    except Exception: pass
    pipeline.stop()
    cv2.destroyAllWindows()
