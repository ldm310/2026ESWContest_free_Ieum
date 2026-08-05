"""
test_single_speaker.py  -  [실험용] 노트북 마이크 STT를 한 사람 얼굴 옆에 띄우기

이 실험이 하는 것:
  - 화면에서 가장 크게(=가까이) 잡힌 얼굴 1명을 대상으로 삼는다.
  - 노트북 내장 마이크(1채널)로 실제 음성을 받아 동민님 STT에 넣는다.
  - 그 사람 얼굴 옆에 partial(발화 중)을 실시간으로, final(확정)을 아래 기록으로 띄운다.

한계(의도된 것):
  - 노트북 마이크는 1채널이라 '방향'을 모른다. 그래서 발화자 매칭이 아니라
    '가장 가까운 한 사람'에게 자막을 붙인다. 방향 기반 매칭은 통합(True) 버전의 몫.

준비:
  - 이 폴더에 동민님 stt 폴더가 있어야 함 (import stt).
    구조 예:  Test_single_speaker_stt/
                 test_single_speaker.py
                 stt/            <- 동민님 모듈 통째로 복사
  - 동민님 requirements-core.txt 설치 필요.

설치:
  py -3.11 -m pip install pyrealsense2 opencv-contrib-python mediapipe pillow numpy sounddevice
  py -3.11 -m pip install -r stt/requirements-core.txt      # 동민님 STT 의존성

실행:  py -3.11 test_single_speaker.py     (ESC 종료)
"""
import os
import sys
import time
import queue
import threading
import urllib.request

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs

# 같은 폴더의 stt 패키지를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stt import StreamingSTT, STTResult      # 동민님 모듈

W, H = 640, 480
STT_SAMPLE_RATE = 16000
MIC_BLOCK = 1600                 # 100ms (16000 * 0.1)

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
    if _FONT_PATH is None: return None
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]
def draw_korean(img_bgr, items):
    if not items or _FONT_PATH is None:
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, (x, y), color, size in items:
        d.text((x, y), text, font=_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


_measure_img = Image.new("RGB", (10, 10))
_measure_draw = ImageDraw.Draw(_measure_img)

def _text_size(text, size):
    """글자의 픽셀 폭/높이를 실제 폰트로 측정."""
    f = _font(size)
    if f is None:
        return (len(text) * size, size)
    l, t, r, b = _measure_draw.textbbox((0, 0), text, font=f)
    return (r - l, b - t)

def _wrap(text, size, max_w):
    """max_w(픽셀) 안에 들어가도록 글자 단위로 줄바꿈한 리스트 반환."""
    lines, cur = [], ""
    for ch in text:
        if _text_size(cur + ch, size)[0] <= max_w or not cur:
            cur += ch
        else:
            lines.append(cur); cur = ch
    if cur:
        lines.append(cur)
    return lines

def draw_caption_bubble(img, face_box, text, color, size, max_w=200):
    """얼굴 옆(공간 없으면 반대쪽)에 자동 줄바꿈 말풍선을 그린다."""
    bx, by, bw, bh = face_box
    lines = _wrap(text, size, max_w)
    line_h = size + 6
    box_w = min(max((max(_text_size(l, size)[0] for l in lines) + 20), 60), W - 10)
    box_h = line_h * len(lines) + 12

    # 오른쪽에 공간 있으면 오른쪽, 없으면 왼쪽
    if bx + bw + 8 + box_w <= W - 5:
        tx = bx + bw + 8
    else:
        tx = max(bx - 8 - box_w, 5)
    ty = min(max(by, 5), H - box_h - 5)

    ov = img.copy()
    cv2.rectangle(ov, (tx, ty), (tx + box_w, ty + box_h), (25, 25, 25), -1)
    img = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
    items = [(ln, (tx + 10, ty + 6 + i * line_h), color, size) for i, ln in enumerate(lines)]
    return img, items

# ── STT: 결과를 큐로 ──
result_q: "queue.Queue[STTResult]" = queue.Queue()
def on_result(r): result_q.put(r)
stt = StreamingSTT(on_result=on_result)
stt.start()

# ── 노트북 마이크(1채널) → STT push ──
def _mic_callback(indata, frames, time_info, status):
    if status:
        print("[mic]", status, flush=True)
    mono = indata[:, 0].astype(np.float32).copy()   # 1채널
    stt.push_audio(mono, sample_rate=STT_SAMPLE_RATE)

mic_stream = sd.InputStream(samplerate=STT_SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=MIC_BLOCK, callback=_mic_callback)
mic_stream.start()

# ── 얼굴 검출 ──
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
align = rs.align(rs.stream.color)

active_text = ""          # partial (발화 중)
transcript = []           # 하단 확정 기록
last_final_text = ""      # 얼굴 옆에 잠깐 더 보여줄 확정 자막
last_final_time = 0.0
FINAL_HOLD_SEC = 4.0      # 확정 자막을 얼굴 옆에 몇 초 유지할지

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        if not cf:
            continue
        img = np.asanyarray(cf.get_data())

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        # 가장 큰 얼굴 1명 = 대상
        target = None
        for det in result.detections:
            bb = det.bounding_box
            box = (bb.origin_x, bb.origin_y, bb.width, bb.height)
            if target is None or box[2]*box[3] > target[2]*target[3]:
                target = box

        # STT 결과 소비
        while not result_q.empty():
            r = result_q.get()
            if r.type == "partial":
                active_text = r.text
            elif r.type == "final":
                if r.text.strip():
                    transcript.append(r.text); transcript[:] = transcript[-4:]
                    last_final_text = r.text
                    last_final_time = time.time()
                active_text = ""
            elif r.type == "error":
                print("[STT error]", r.error, flush=True)

        korean_items = []
        if target is not None:
            bx, by, bw, bh = target
            cv2.rectangle(img, (bx,by), (bx+bw,by+bh), (0,200,255), 2)
            # 얼굴 옆에 보여줄 글자: 말하는 중이면 partial(초록),
            # 아니면 방금 확정된 final 을 몇 초간 유지(노랑).
            side_text, side_color = "", None
            if active_text:
                side_text, side_color = active_text, (120,255,180)      # 발화 중=초록
            elif last_final_text and (time.time() - last_final_time) < FINAL_HOLD_SEC:
                side_text, side_color = last_final_text, (120,220,255)  # 확정=노랑
            if side_text:
                img, bubble_items = draw_caption_bubble(
                    img, (bx, by, bw, bh), side_text, side_color, size=18, max_w=200)
                korean_items.extend(bubble_items)
        else:
            cv2.putText(img, "얼굴 없음 (no face)", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 확정 기록 스트립
        if transcript:
            strip_h = 22*len(transcript)+16
            ov = img.copy(); cv2.rectangle(ov,(0,H-strip_h-28),(W,H-28),(20,20,20),-1)
            img = cv2.addWeighted(ov,0.5,img,0.5,0)
            for i,txt in enumerate(transcript):
                korean_items.append((txt,(10,H-strip_h-20+i*22),(235,235,235),16))

        cv2.putText(img, "[MIC] 노트북 마이크 -> STT   ESC: quit", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1)

        img = draw_korean(img, korean_items)
        cv2.imshow("Test: single speaker STT", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    try: stt.stop()
    except Exception: pass
    try: mic_stream.stop()
    except Exception: pass
    pipeline.stop()
    cv2.destroyAllWindows()