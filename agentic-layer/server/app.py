from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from server.agent import respond
    from server.config import AGENTIC_ROOT, Settings
else:
    from .agent import respond
    from .config import AGENTIC_ROOT, Settings


UI_ROOT = AGENTIC_ROOT / "ui"


class AgenticRequestHandler(BaseHTTPRequestHandler):
    server_version = "AgenticLayer/0.1"

    def log_message(self, format: str, *args: object) -> None:
        if Settings.log_level != "silent":
            super().log_message(format, *args)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "providers": {
                        "primary": Settings.primary_provider,
                        "fallback": Settings.fallback_provider,
                        "gemini_configured": Settings.provider_configured("gemini"),
                        "groq_configured": Settings.provider_configured("groq"),
                    },
                }
            )
            return

        requested = "index.html" if parsed.path in {"/", ""} else unquote(parsed.path.lstrip("/"))
        file_path = (UI_ROOT / requested).resolve()
        if not str(file_path).startswith(str(UI_ROOT.resolve())):
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid path")
            return
        self._send_file(file_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            message = str(payload.get("message", "")).strip()
            mode = str(payload.get("mode", "ask")).strip().lower()
            provider = str(payload.get("provider", "auto")).strip().lower()

            if not message:
                self._send_json({"error": "Message is required."}, HTTPStatus.BAD_REQUEST)
                return

            result = respond(message=message, ui_mode=mode, provider=provider)
            self._send_json(
                {
                    "answer": result.answer,
                    "provider": result.provider,
                    "mode": result.mode,
                    "fallback_used": result.fallback_used,
                }
            )
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    server = ThreadingHTTPServer((Settings.host, Settings.port), AgenticRequestHandler)
    print(f"Agentic Layer running at http://{Settings.host}:{Settings.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Agentic Layer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
