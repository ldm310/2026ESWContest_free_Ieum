from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from captioning_msgs.msg import SeparatedAudio, SoundSourceTrack, SoundSourceTracks

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

        default_weights = _find_weights("jointnet_h128b12_epoch30.pt")

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
        self.declare_parameter("chunk_frames", 16)
        self.declare_parameter("input_blocksize", 512)

        self.declare_parameter("activity_threshold", 0.3)
        self.declare_parameter("gate_deg", 35.0)
        self.declare_parameter("smooth_halflife_sec", 0.6)
        self.declare_parameter("confirm_chunks", 3)
        self.declare_parameter("coast_chunks", 6)

        self.declare_parameter("publish_audio", True)
        self.declare_parameter("publish_during_warmup", False)
        self.declare_parameter("bypass", False)
        self.declare_parameter("bypass_threshold", 0.008)
        self.declare_parameter("bypass_channel", 0)

        get = lambda name: self.get_parameter(name).value
        self.frame_id = get("frame_id")
        self.sample_rate = int(get("sample_rate"))
        self.chunk_frames = int(get("chunk_frames"))
        self.mic_channels = [int(c) for c in get("mic_channels")]
        self.publish_audio = bool(get("publish_audio"))
        self.publish_during_warmup = bool(get("publish_during_warmup"))
        self.bypass = bool(get("bypass"))
        self.bypass_threshold = float(get("bypass_threshold"))
        self.bypass_channel = int(get("bypass_channel"))

        if len(self.mic_channels) != 4:
            raise RuntimeError(f"mic_channels 는 4개여야 한다: {self.mic_channels}")

        self.net = StreamingJointNet(get("weights"), StreamConfig(
            sample_rate=self.sample_rate, chunk_frames=self.chunk_frames,
            device=get("torch_device")))
        self.chunk_samples = self.chunk_frames * self.net.config.hop
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
        self.audio_publisher = self.create_publisher(
            SeparatedAudio, "separated_audio", sensor_qos)

        self.queue: queue.Queue = queue.Queue(maxsize=64)
        self.buffer = np.zeros((0, 4), dtype=np.float32)
        self.emitted_samples = 0
        self.origin_ns: int | None = None
        self.dropped = 0
        self.running = True

        self.stream = self._open_stream(get("device_name"),
                                        int(get("device_channels")),
                                        int(get("input_blocksize")),
                                        get("pulse_source"))
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"가중치 epoch {self.net.epoch}, 조각 {self.chunk_samples}샘플 "
            f"({1000 * self.chunk_seconds:.0f} ms), 마이크 채널 {self.mic_channels}"
            + (f"  [우회 모드 — 모델 미사용, 문턱 {self.bypass_threshold}, "
               f"{'채널 ' + str(self.bypass_channel) if self.bypass_channel >= 0 else '4채널 평균'}]"
               if self.bypass else ""))

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
            self.queue.put_nowait((indata[:, self.mic_channels].copy(), arrival, frames))
        except queue.Full:
            self.dropped += 1
            self.get_logger().error(
                f"큐가 찼다. 조각 {self.dropped}개 유실 — GRU 상태가 불연속이 된다.",
                throttle_duration_sec=5.0)

    def _work(self) -> None:
        while self.running:
            try:
                block, arrival_ns, frames = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.origin_ns is None:
                self.origin_ns = arrival_ns - int(1e9 * frames / self.sample_rate)
            self.buffer = np.concatenate([self.buffer, block], axis=0)
            while self.buffer.shape[0] >= self.chunk_samples:
                chunk = self.buffer[: self.chunk_samples]
                self.buffer = self.buffer[self.chunk_samples:]
                try:
                    self._process(chunk)
                except Exception as error:
                    self.get_logger().error(f"추론 실패: {error}")

    def _process(self, chunk: np.ndarray) -> None:
        if self.bypass:
            self._process_bypass(chunk)
            return
        result = self.net.push(np.ascontiguousarray(chunk))
        if result is None:
            return

        stamp_ns = self.origin_ns + int(1e9 * self.emitted_samples / self.sample_rate)
        self.emitted_samples += self.chunk_samples

        tracks = self.tracker.update(result["direction"], result["activity"],
                                     dt=self.chunk_seconds)
        if not result["warm"] and not self.publish_during_warmup:
            return

        stamp = rclpy.time.Time(nanoseconds=stamp_ns).to_msg()

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
        message.count_probability = [float(p) for p in count_probability(result["activity"])]
        self.track_publisher.publish(message)

        if not self.publish_audio:
            return

        mapping = self.tracker.slot_to_track()
        audio = SeparatedAudio()
        audio.header.stamp = stamp
        audio.header.frame_id = self.frame_id
        audio.sample_rate = self.sample_rate
        audio.n_samples = int(result["audio"].shape[1])
        slots = sorted(mapping)
        audio.track_ids = [int(mapping[slot]) for slot in slots]
        if slots:
            audio.samples = result["audio"][slots].reshape(-1).astype(np.float32).tolist()
        else:
            audio.samples = []
        self.audio_publisher.publish(audio)

    def _process_bypass(self, chunk: np.ndarray) -> None:
        if 0 <= self.bypass_channel < chunk.shape[1]:
            mono = chunk[:, self.bypass_channel].astype(np.float32)
        else:
            mono = chunk.mean(axis=1).astype(np.float32)
        level = float(np.sqrt((mono ** 2).mean()))
        active = level >= self.bypass_threshold

        stamp_ns = self.origin_ns + int(1e9 * self.emitted_samples / self.sample_rate)
        self.emitted_samples += self.chunk_samples
        stamp = rclpy.time.Time(nanoseconds=stamp_ns).to_msg()

        message = SoundSourceTracks()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        if active:
            entry = SoundSourceTrack()
            entry.track_id = 1
            entry.direction = Vector3(x=1.0, y=0.0, z=0.0)
            entry.azimuth_deg = 0.0
            entry.elevation_deg = 0.0
            entry.activity = 1.0
            entry.age_sec = float(self.emitted_samples / self.sample_rate)
            message.tracks.append(entry)
        message.count_probability = [0.0, 1.0] if active else [1.0, 0.0]
        self.track_publisher.publish(message)

        if not self.publish_audio:
            return
        audio = SeparatedAudio()
        audio.header.stamp = stamp
        audio.header.frame_id = self.frame_id
        audio.sample_rate = self.sample_rate
        audio.n_samples = int(mono.shape[0])
        audio.track_ids = [1] if active else []
        audio.samples = mono.tolist() if active else []
        self.audio_publisher.publish(audio)

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
