"""Jetson 자막 UI의 정적 파일과 HTTP 계약 테스트."""

from __future__ import annotations

import json
import threading
import unittest
from http.client import HTTPConnection
from unittest.mock import Mock, patch

from stage6_caption_ui import (
    CaptionHTTPServer,
    RuntimeState,
    UIRequestHandler,
    open_kiosk,
    validate_runtime_files,
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
        self.assertIn('cameraDot.classList.toggle("live", connected)', javascript)
        self.assertIn('microphoneDot.classList.toggle("live", microphoneConnected)', javascript)

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
