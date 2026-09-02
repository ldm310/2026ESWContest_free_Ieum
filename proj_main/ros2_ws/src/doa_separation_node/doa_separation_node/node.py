from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float32MultiArray
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from captioning_msgs.msg import (Faces, SeparatedAudio, SoundSourceTrack,
                                 SoundSourceTracks)

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from streaming_net import StreamConfig, StreamingJointNet, count_probability
from tracker import SourceTracker, TrackerConfig


def _find_weights(name: str) -> str:
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(Path(get_package_share_directory("doa_separation_node")) / "weights" / name)
    except Exception:
        pass
    here = Path(__file__).resolve()
    candidates += [here.parents[1] / "weights" / name,
                   here.parents[2] / "weights" / name,
                   here.parents[3] / "weights" / name]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0]) if candidates else name


def find_input_device(name: str, minimum_channels: int) -> int | None:
    import sounddevice as sd
    for index, info in enumerate(sd.query_devices()):
        if name.lower() in info["name"].lower() and info["max_input_channels"] >= minimum_channels:
            return index
    return None


class DoaSeparationNode(Node):
    def __init__(self) -> None:
        super().__init__("doa_separation")

        default_weights = _find_weights("jointnet_v4.pt")

        self.declare_parameter("weights", default_weights)
        self.declare_parameter("torch_device", "cuda")
        self.declare_parameter("frame_id", "mic_array")

        self.declare_parameter("device_name", "ArrayUAC10")
        self.declare_parameter("device_channels", 6)
        self.declare_parameter(
            "pulse_source",
            "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input")

        self.declare_parameter("mic_channels", [1, 2, 3, 4])
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("chunk_frames", 8)
        self.declare_parameter("input_blocksize", 512)

        self.declare_parameter("activity_threshold", 0.3)

        self.declare_parameter("model_enabled", True)
        self.declare_parameter("use_face_limit", True)
        self.declare_parameter("face_hold_sec", 3.0)
        self.declare_parameter("face_timeout_sec", 2.0)
        self.declare_parameter("azimuth_min_deg", 30.0)
        self.declare_parameter("azimuth_max_deg", 150.0)
        self.declare_parameter("elevation_min_deg", -10.0)
        self.declare_parameter("elevation_max_deg", 35.0)
        self.declare_parameter("gate_deg", 35.0)
        self.declare_parameter("smooth_halflife_sec", 0.6)
        self.declare_parameter("confirm_chunks", 3)
        self.declare_parameter("coast_chunks", 6)

        self.declare_parameter("publish_audio", True)
        self.declare_parameter("publish_during_warmup", False)
        self.declare_parameter("level_gate", 0.015)
        self.declare_parameter("speech_low_ratio", 0.5)
        self.declare_parameter("gate_hold_chunks", 8)
        self.declare_parameter("raw_audio", False)
        self.declare_parameter("raw_audio_channel", 0)

        get = lambda name: self.get_parameter(name).value
        self.frame_id = get("frame_id")
        self.sample_rate = int(get("sample_rate"))
        self.chunk_frames = int(get("chunk_frames"))
        self.mic_channels = [int(c) for c in get("mic_channels")]
        self.publish_audio = bool(get("publish_audio"))
        self.publish_during_warmup = bool(get("publish_during_warmup"))
        self.level_gate = float(get("level_gate"))
        self.speech_low_ratio = float(get("speech_low_ratio"))
        self.gate_hold_chunks = int(get("gate_hold_chunks"))
        self.model_enabled = bool(get("model_enabled"))
        self.use_face_limit = bool(get("use_face_limit"))
        self.face_hold_sec = float(get("face_hold_sec"))
        self.face_timeout_sec = float(get("face_timeout_sec"))
        self.face_window: list[tuple[float, int]] = []
        self.face_lock = threading.Lock()
        self.azimuth_min = float(get("azimuth_min_deg"))
        self.azimuth_max = float(get("azimuth_max_deg"))
        self.elevation_min = float(get("elevation_min_deg"))
        self.elevation_max = float(get("elevation_max_deg"))
        self.gate_hold = 0
        self.raw_audio = bool(get("raw_audio"))
        self.last_track_id = 1
        self.raw_audio_channel = int(get("raw_audio_channel"))

        if len(self.mic_channels) != 4:
            raise RuntimeError(f"mic_channels 는 4개여야 한다: {self.mic_channels}")

        self.net = StreamingJointNet(get("weights"), StreamConfig(
            sample_rate=self.sample_rate, chunk_frames=self.chunk_frames,
            device=get("torch_device")))
        self.chunk_samples = (self.chunk_frames * self.net.config.hop
                              * getattr(self.net, "decimate", 1))
        self.chunk_seconds = self.chunk_samples / self.sample_rate

        self.tracker = SourceTracker(TrackerConfig(
            activity_threshold=float(get("activity_threshold")),
            gate_deg=float(get("gate_deg")),
            smooth_halflife_sec=float(get("smooth_halflife_sec")),
            confirm_chunks=int(get("confirm_chunks")),
            coast_chunks=int(get("coast_chunks"))))

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.track_publisher = self.create_publisher(
            SoundSourceTracks, "sound_source_tracks", sensor_qos)
        self.create_subscription(Faces, "faces", self._on_faces, sensor_qos)
        self.create_subscription(Bool, "model_enabled", self._on_model_enabled, 10)

        self.create_subscription(Bool, "stt_raw_audio", self._on_stt_source, 10)

        self.state_publisher = self.create_publisher(Bool, "stt_raw_audio", 10)
        self.model_state_publisher = self.create_publisher(Bool, "model_enabled", 10)
        self.announce_timer = self.create_timer(2.0, self._announce_state)
        self.audio_publisher = self.create_publisher(
            SeparatedAudio, "separated_audio", sensor_qos)

        self.debug_publisher = self.create_publisher(
            Float32MultiArray, "doa_debug", sensor_qos)

        self.queue: queue.Queue = queue.Queue(maxsize=64)
        self.buffer = np.zeros((0, 4), dtype=np.float32)
        self.ref_buffer = np.zeros((0,), dtype=np.float32)
        self.ref_chunk = np.zeros((0,), dtype=np.float32)
        self.emitted_samples = 0
        self.origin_ns: int | None = None
        self.dropped = 0
        self.running = True

        warm = np.zeros((self.chunk_samples, 4), dtype=np.float32)
        for _ in range(12):
            self.net.push(warm)
        self.net.reset()

        self.stream = self._open_stream(get("device_name"),
                                        int(get("device_channels")),
                                        int(get("input_blocksize")),
                                        get("pulse_source"))
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"가중치 epoch {self.net.epoch}, 조각 {self.chunk_samples}샘플 "
            f"({1000 * self.chunk_seconds:.0f} ms), 마이크 채널 {self.mic_channels}"
            + ("  [원본 음성 모드 — 방향은 모델, STT 입력은 원본 채널 "
               + str(self.raw_audio_channel) + "]" if self.raw_audio else ""))

    def _open_stream(self, device_name: str, device_channels: int, blocksize: int,
                     pulse_source: str):
        import sounddevice as sd
        if pulse_source:
            os.environ["PULSE_SOURCE"] = pulse_source
            index = next((i for i, info in enumerate(sd.query_devices())
                          if info["name"] == "pulse"
                          and info["max_input_channels"] >= device_channels), None)
            if index is None:
                raise RuntimeError("pulse 입력장치를 찾지 못했다. "
                                   "pulse_source 를 비우고 device_name 으로 직접 찾게 하라.")
            self.get_logger().info(f"PulseAudio 소스 {pulse_source}")
        else:
            index = find_input_device(device_name, device_channels)
            if index is None:
                self.get_logger().warn(
                    f"'{device_name}' 을 찾지 못했다. 기본 입력장치를 쓴다. "
                    f"채널 순서가 다를 수 있으니 방향을 확인해야 한다.")
        if max(self.mic_channels) >= device_channels:
            raise RuntimeError(
                f"mic_channels {self.mic_channels} 가 장치 채널 수 {device_channels} 를 넘는다")
        stream = sd.InputStream(
            device=index, channels=device_channels, samplerate=self.sample_rate,
            dtype="float32", blocksize=blocksize, callback=self._audio_callback)
        stream.start()
        return stream

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.get_logger().warn(f"오디오 상태 {status}", throttle_duration_sec=5.0)
        arrival = self.get_clock().now().nanoseconds
        try:
            self.queue.put_nowait((indata[:, self.mic_channels].copy(),
                                   indata[:, 0].copy(), arrival, frames))
        except queue.Full:
            self.dropped += 1
            self.get_logger().error(
                f"큐가 찼다. 조각 {self.dropped}개 유실 — GRU 상태가 불연속이 된다.",
                throttle_duration_sec=5.0)

    def _work(self) -> None:
        while self.running:
            try:
                block, ref, arrival_ns, frames = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.origin_ns is None:
                self.origin_ns = arrival_ns - int(1e9 * frames / self.sample_rate)
            self.buffer = np.concatenate([self.buffer, block], axis=0)
            self.ref_buffer = np.concatenate([self.ref_buffer, ref], axis=0)
            while self.buffer.shape[0] >= self.chunk_samples:
                chunk = self.buffer[: self.chunk_samples]
                self.buffer = self.buffer[self.chunk_samples:]
                self.ref_chunk = self.ref_buffer[: self.chunk_samples]
                self.ref_buffer = self.ref_buffer[self.chunk_samples:]
                try:
                    self._process(chunk)
                except Exception as error:
                    self.get_logger().error(f"추론 실패: {error}")

    def _announce_state(self) -> None:
        self.state_publisher.publish(Bool(data=bool(self.raw_audio)))
        self.model_state_publisher.publish(Bool(data=bool(self.model_enabled)))
        self.announce_timer.cancel()

    def _on_stt_source(self, message: Bool) -> None:
        want = bool(message.data)
        if want == self.raw_audio:
            return
        self.raw_audio = want
        source = (f"원본 채널 {self.raw_audio_channel}" if want else "모델 분리 출력")
        self.get_logger().info(f"STT 입력 → {source}")

    def _on_model_enabled(self, message: Bool) -> None:
        want = bool(message.data)
        if want == self.model_enabled:
            return
        self.model_enabled = want
        if want:

            self.net.reset()
            self.tracker = SourceTracker(self.tracker.config)
        self.get_logger().info(f"모델 {'켜짐' if want else '꺼짐'}")

    def _process(self, chunk: np.ndarray) -> None:
        if not self.model_enabled:
            return
        result = self.net.push(np.ascontiguousarray(chunk))
        if result is None:
            return

        stamp_ns = self.origin_ns + int(1e9 * self.emitted_samples / self.sample_rate)
        self.emitted_samples += self.chunk_samples

        tracks = self.tracker.update(result["direction"], result["activity"],
                                     dt=self.chunk_seconds)
        gate_open = True
        if self.ref_chunk.shape[0]:
            level = float(np.sqrt((self.ref_chunk ** 2).mean()))
            if level >= self.level_gate:
                self.gate_hold = self.gate_hold_chunks
            elif self.gate_hold > 0:
                self.gate_hold -= 1
            else:
                tracks = []
                gate_open = False
        if not result["warm"] and not self.publish_during_warmup:
            return

        stamp = rclpy.time.Time(nanoseconds=stamp_ns).to_msg()

        n_before_range = len(tracks)
        tracks = [t for t in tracks
                  if self.azimuth_min <= t.azimuth_deg <= self.azimuth_max
                  and self.elevation_min <= t.elevation_deg <= self.elevation_max]
        n_out_of_range = n_before_range - len(tracks)

        n_before_face = len(tracks)
        face_limit = self._face_limit()
        if face_limit is not None and len(tracks) > face_limit:

            tracks = sorted(tracks, key=lambda t: -float(t.activity))[:face_limit]
        n_face_dropped = n_before_face - len(tracks)

        message = SoundSourceTracks()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        for track in tracks:
            entry = SoundSourceTrack()
            entry.track_id = int(track.track_id)
            entry.direction = Vector3(x=float(track.direction[0]),
                                      y=float(track.direction[1]),
                                      z=float(track.direction[2]))
            entry.azimuth_deg = float(track.azimuth_deg)
            entry.elevation_deg = float(track.elevation_deg)
            entry.activity = float(track.activity)
            entry.age_sec = float(track.age_sec)
            message.tracks.append(entry)
        counts = count_probability(result["activity"])
        if face_limit is not None and face_limit < len(counts) - 1:

            counts = counts[:face_limit + 1]
            total = float(counts.sum())
            counts = counts / total if total > 1e-12 else counts
        message.count_probability = [float(p) for p in counts]
        self.track_publisher.publish(message)

        debug = Float32MultiArray()
        mic_rms = np.sqrt((chunk ** 2).mean(axis=0))
        ref_rms = float(np.sqrt((self.ref_chunk ** 2).mean())) if self.ref_chunk.shape[0] else 0.0

        cfg = self.tracker.config

        n_live = int((np.asarray(result["activity"]) >= cfg.activity_threshold).sum())
        n_held = len(self.tracker.tracks)
        n_confirmed = sum(1 for t in self.tracker.tracks if t.confirmed)
        best_hits = max((t.hits for t in self.tracker.tracks), default=0)
        debug.data = ([float(v) for v in mic_rms] + [ref_rms]
                      + [float(v) for v in result["activity"]]

                      + (list(message.count_probability)
                         + [0.0] * 4)[:4]
                      + [float(cfg.activity_threshold),
                         float(self.level_gate),
                         1.0 if gate_open else 0.0,
                         1.0 if result["warm"] else 0.0,
                         float(n_live), float(n_held), float(n_confirmed),
                         float(best_hits), float(cfg.confirm_chunks),
                         float(-1 if face_limit is None else face_limit),
                         float(n_face_dropped), float(n_out_of_range),
                         float(len(tracks))])
        self.debug_publisher.publish(debug)

        if not self.publish_audio:
            return

        audio = SeparatedAudio()
        audio.header.stamp = stamp
        audio.header.frame_id = self.frame_id

        if self.raw_audio:
            if self.ref_chunk.shape[0] == chunk.shape[0]:
                mono = self.ref_chunk.astype(np.float32)
            elif 0 <= self.raw_audio_channel < chunk.shape[1]:
                mono = chunk[:, self.raw_audio_channel].astype(np.float32)
            else:
                mono = chunk.mean(axis=1).astype(np.float32)
            audio.sample_rate = int(self.sample_rate)
            audio.n_samples = int(mono.shape[0])
            audio.track_ids = [1]
            audio.samples = mono.tolist()
            self.audio_publisher.publish(audio)
            return

        mapping = self.tracker.slot_to_track()
        audio.sample_rate = int(self.net.config.sample_rate)
        audio.n_samples = int(result["audio"].shape[1])
        slots = [s for s in sorted(mapping) if self._is_speech(result["audio"][s])]
        audio.track_ids = [int(mapping[slot]) for slot in slots]
        if slots:
            audio.samples = result["audio"][slots].reshape(-1).astype(np.float32).tolist()
        else:
            audio.samples = []
        self.audio_publisher.publish(audio)

    def _is_speech(self, signal: np.ndarray) -> bool:
        if signal.size < 32:
            return False
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        total = float(spectrum.sum())
        if total <= 1e-20:
            return False
        cut = max(1, int(len(spectrum) * 1000.0 / (self.net.config.sample_rate / 2.0)))
        return float(spectrum[:cut].sum()) / total >= self.speech_low_ratio

    def _on_faces(self, message: Faces) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        with self.face_lock:
            self.face_window.append((now, int(message.n_detected)))
            cutoff = now - self.face_hold_sec
            self.face_window = [(t, n) for t, n in self.face_window if t >= cutoff]

    def _face_limit(self) -> int | None:
        if not self.use_face_limit:
            return None
        now = self.get_clock().now().nanoseconds / 1e9
        with self.face_lock:
            fresh = [n for t, n in self.face_window if now - t <= self.face_hold_sec]
            newest = max((t for t, _ in self.face_window), default=None)
        if newest is None or now - newest > self.face_timeout_sec:
            return None
        limit = max(fresh) if fresh else 0

        return limit if limit > 0 else None

    def destroy_node(self) -> bool:
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DoaSeparationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
