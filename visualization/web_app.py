"""注释
命令示例:
python visualization/web_app.py
python visualization/web_app.py --host 0.0.0.0 --port 8000

参数含义:
- `--host`: 本地 HTTP 服务监听地址。
- `--port`: 本地 HTTP 服务端口。

逻辑说明:
本文件提供 ReAct Web 可视化的本地后端。它只暴露轻量 API 和静态页面，不重写 ReAct 决策逻辑；所有运行控制都交给
`visualization.web_runtime.WebRuntimeManager`。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__:
    from visualization import web_runtime
else:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_PARENT = CURRENT_DIR.parent
    if str(PROJECT_PARENT) not in sys.path:
        sys.path.insert(0, str(PROJECT_PARENT))
    from visualization import web_runtime


VIS_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = VIS_ROOT / "web_static"
RUNTIME_ROOT = VIS_ROOT / "runtime"
RUN_MANAGER = web_runtime.WebRuntimeManager()
SURROGATE_MANAGER = web_runtime.SurrogateRuntimeManager()


def build_combined_state() -> dict:
    payload = RUN_MANAGER.get_state()
    payload["surrogate"] = SURROGATE_MANAGER.get_state()
    return payload


class VisualizationHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_ROOT / "index.html")
            return
        if parsed.path == "/api/options":
            self._send_json(RUN_MANAGER.get_options())
            return
        if parsed.path == "/api/request/preview":
            params = parse_qs(parsed.query)
            request_key = (params.get("request_key") or [""])[0]
            try:
                self._send_json(RUN_MANAGER.preview_request(request_key))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/run/state":
            self._send_json(build_combined_state())
            return
        if parsed.path == "/api/surrogate/state":
            self._send_json(SURROGATE_MANAGER.get_state())
            return
        if parsed.path == "/api/run/events":
            self._serve_sse_events()
            return
        if parsed.path.startswith("/runtime/"):
            relative = parsed.path.removeprefix("/runtime/").lstrip("/")
            self._serve_file(RUNTIME_ROOT / relative)
            return
        if parsed.path.startswith("/static/"):
            relative = parsed.path.removeprefix("/static/").lstrip("/")
            self._serve_file(STATIC_ROOT / relative)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_ROOT / "index.html", head_only=True)
            return
        if parsed.path.startswith("/runtime/"):
            relative = parsed.path.removeprefix("/runtime/").lstrip("/")
            self._serve_file(RUNTIME_ROOT / relative, head_only=True)
            return
        if parsed.path.startswith("/static/"):
            relative = parsed.path.removeprefix("/static/").lstrip("/")
            self._serve_file(STATIC_ROOT / relative, head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run/start":
            payload = self._read_json_body()
            try:
                response = RUN_MANAGER.start_run(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response)
            return
        if parsed.path == "/api/run/stop":
            self._send_json(RUN_MANAGER.stop_run())
            return
        if parsed.path == "/api/run/reset":
            self._send_json(RUN_MANAGER.reset_run())
            return
        if parsed.path == "/api/surrogate/train":
            payload = self._read_json_body()
            try:
                response = SURROGATE_MANAGER.start_train(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response)
            return
        if parsed.path == "/api/surrogate/eval":
            payload = self._read_json_body()
            try:
                response = SURROGATE_MANAGER.start_eval(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response)
            return
        if parsed.path == "/api/surrogate/samples":
            payload = self._read_json_body()
            try:
                response = web_runtime.resolve_surrogate_samples_payload(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response)
            return
        if parsed.path == "/api/surrogate/stop":
            self._send_json(SURROGATE_MANAGER.stop())
            return
        if parsed.path == "/api/surrogate/reset":
            self._send_json(SURROGATE_MANAGER.reset())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path, head_only: bool = False) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        body = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _serve_sse_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_payload = ""
        try:
            while True:
                payload = build_combined_state()
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                if encoded != last_payload:
                    self.wfile.write(f"event: state\n".encode("utf-8"))
                    self.wfile.write(f"data: {encoded}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_payload = encoded
                else:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), VisualizationHandler)
    local_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    startup_payload = {
        "local_url": f"http://{local_host}:{args.port}",
        "listen_host": args.host,
        "listen_port": args.port,
        "python": str(RUN_MANAGER.python_executable),
        "external_access_hint": "Use your platform-mapped public URL or public_ip:port; 0.0.0.0 is only the bind address.",
    }
    print(
        json.dumps(startup_payload, ensure_ascii=False)
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
