"""Jetson 자막 UI의 정적 파일과 HTTP 계약 테스트."""

from __future__ import annotations

import json
import threading
import unittest
from http.client import HTTPConnection
from types import SimpleNamespace
from unittest.mock import Mock, patch

from stage6_caption_ui import (
    CaptionHTTPServer,
    RealtimeReceiver,
    RuntimeState,
    StreamingSTTService,
    UIRequestHandler,
    open_kiosk,
    validate_runtime_files,
)
from runtime_protocol import (
    AUDIO_CHUNK_SECONDS,
    SENTENCE_END_PACKET,
    pack_sentence_end,
    unpack_sentence_end,
)


class CaptionUITest(unittest.TestCase):
    """장치 없이도 실행 가능한 UI 서버 계약을 검증한다."""

    def setUp(self) -> None:
        self.stop_event = threading.Event()
        self.state = RuntimeState(demo=True)
        self.server = CaptionHTTPServer(
            ("127.0.0.1", 0),
            UIRequestHandler,
            self.state,
            self.stop_event,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = int(self.server.server_address[1])

    def tearDown(self) -> None:
        self.stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, bytes]:
        """테스트 서버에 요청하고 상태, Content-Type, 본문을 반환한다."""

        connection = HTTPConnection("127.0.0.1", self.port, timeout=2.0)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        result = (response.status, response.getheader("Content-Type", ""), payload)
        connection.close()
        return result

    def test_runtime_files_and_static_assets(self) -> None:
        validate_runtime_files()
        expected = {
            "/": ("text/html", "실시간 화자 인식"),
            "/styles.css": ("text/css", ".device-screen"),
            "/app.js": ("text/javascript", "pollState"),
        }
        for path, (content_type, marker) in expected.items():
            with self.subTest(path=path):
                status, actual_type, payload = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn(content_type, actual_type)
                self.assertIn(marker, payload.decode("utf-8"))

        _, _, html = self.request("GET", "/")
        rendered = html.decode("utf-8")
        self.assertIn('id="microphone-dot"', rendered)
        self.assertIn('id="camera-dot"', rendered)
        self.assertIn('id="runtime-dot"', rendered)
        self.assertEqual(rendered.count('class="speaker-speaking-badge"'), 3)
        self.assertNotIn("mic-status-icon", rendered)
        self.assertNotIn("camera-status-icon", rendered)

        _, _, stylesheet = self.request("GET", "/styles.css")
        css = stylesheet.decode("utf-8")
        self.assertIn(".connection-dot", css)
        self.assertIn("background: #ef4444", css)
        self.assertIn(".connection-dot.live", css)
        self.assertIn("background: #22bd5b", css)

        _, _, script = self.request("GET", "/app.js")
        javascript = script.decode("utf-8")
        self.assertIn("const POLL_INTERVAL_MS = 100;", javascript)
        self.assertIn('cameraDot.classList.toggle("live", connected)', javascript)
        self.assertIn('microphoneDot.classList.toggle("live", microphoneConnected)', javascript)
        self.assertIn("현재 말하는 중", javascript)

    def test_partial_activates_speaker_until_final(self) -> None:
        """partial 화자는 강조하고 같은 발화의 final에서 해제한다."""

        faces = [
            {"speaker_id": 1, "active": False},
            {"speaker_id": 2, "active": True},
            {"speaker_id": 3, "active": False},
        ]
        self.state.update_faces(faces, active_speaker_id=2)
        self.state.handle_stt_result(
            SimpleNamespace(
                type="partial",
                text="안녕하세요",
                latency_ms=120.0,
                utterance_id=10,
                is_final=False,
                error=None,
            )
        )

        partial_state = self.state.snapshot()
        self.assertEqual(partial_state["active_speaker_id"], 2)
        self.assertEqual(partial_state["partial"]["speaker_id"], 2)
        self.assertEqual(
            [face["speaker_id"] for face in partial_state["faces"] if face["active"]],
            [2],
        )

        # DOA 후보가 바뀌어도 진행 중인 utterance는 최초 화자로 유지한다.
        self.state.update_faces(faces, active_speaker_id=3)
        self.assertEqual(self.state.snapshot()["active_speaker_id"], 2)

        self.state.handle_stt_result(
            SimpleNamespace(
                type="final",
                text="안녕하세요.",
                latency_ms=310.0,
                utterance_id=10,
                is_final=True,
                error=None,
            )
        )
        final_state = self.state.snapshot()
        self.assertIsNone(final_state["active_speaker_id"])
        self.assertIsNone(final_state["partial"])
        self.assertFalse(any(face["active"] for face in final_state["faces"]))
        self.assertEqual(final_state["captions"][-1]["speaker_id"], 2)

        # 다음 발화는 새 DOA 후보인 화자 3으로 구분한다.
        self.state.handle_stt_result(
            SimpleNamespace(
                type="partial",
                text="다음 발화입니다",
                latency_ms=125.0,
                utterance_id=11,
                is_final=False,
                error=None,
            )
        )
        self.assertEqual(self.state.snapshot()["active_speaker_id"], 3)

        self.state.toggle("recording")
        stopped_state = self.state.snapshot()
        self.assertIsNone(stopped_state["active_speaker_id"])
        self.assertIsNone(stopped_state["partial"])

    def test_health_state_and_controls(self) -> None:
        status, _, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload), {"status": "ok"})

        status, _, payload = self.request("GET", "/api/state")
        snapshot = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertTrue(snapshot["status"]["recording"])
        self.assertEqual(len(snapshot["speakers"]), 3)

        body = json.dumps({"action": "recording"}).encode("utf-8")
        status, _, payload = self.request(
            "POST",
            "/api/control",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload), {"ok": True, "value": False})
        self.assertFalse(self.state.snapshot()["status"]["recording"])

    def test_external_sentence_end_flushes_same_streaming_instance(self) -> None:
        """외부 종료 신호는 UI가 사용 중인 STT 객체를 정확히 한 번 flush한다."""

        stt_service = StreamingSTTService(self.state)
        streaming_stt = Mock()
        stt_service._stt = streaming_stt
        receiver = RealtimeReceiver(self.state, stt_service, self.stop_event)

        self.assertTrue(receiver._handle_sentence_end_packet(pack_sentence_end()))
        streaming_stt.flush.assert_called_once_with()

        self.assertFalse(receiver._handle_sentence_end_packet(b"invalid"))
        streaming_stt.flush.assert_called_once_with()

    def test_sentence_end_protocol_and_chunk_interval(self) -> None:
        """0.25초 오디오 계약과 고정 문장 종료 payload를 검증한다."""

        self.assertEqual(AUDIO_CHUNK_SECONDS, 0.25)
        self.assertEqual(pack_sentence_end(), SENTENCE_END_PACKET)
        self.assertIsNone(unpack_sentence_end(SENTENCE_END_PACKET))
        with self.assertRaises(ValueError):
            unpack_sentence_end(b"unknown-command")

    @patch("stage6_caption_ui.subprocess.Popen")
    @patch("stage6_caption_ui.shutil.which")
    def test_kiosk_uses_jetson_resolution_flags(
        self,
        which: Mock,
        popen: Mock,
    ) -> None:
        which.side_effect = lambda name: "/usr/bin/chromium" if name == "chromium" else None
        open_kiosk("http://127.0.0.1:8765")
        command = popen.call_args.args[0]
        self.assertIn("--kiosk", command)
        self.assertIn("--window-size=1024,600", command)
        self.assertIn("--window-position=0,0", command)
        self.assertIn("--force-device-scale-factor=1", command)
        self.assertIn("--app=http://127.0.0.1:8765", command)


if __name__ == "__main__":
    unittest.main()
