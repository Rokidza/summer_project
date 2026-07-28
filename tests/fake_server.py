"""A real HTTP server speaking just enough of the OpenAI chat-completions API.

The harness talks only HTTP to optillm, so its tests point the OpenAI client at
this instead of a real vLLM+optillm stack. It is a genuine socket server, not a
mock: nothing patches the client, and the assertions are about what came back
over the wire.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer


@dataclass
class FakeInferenceServer:
    """Serves scripted responses and records the requests it received."""

    replies: list = field(default_factory=list)
    """Each entry is a dict (a chat-completion body) or an int HTTP error status."""
    default_reply: dict | None = None
    requests: list = field(default_factory=list)

    _server: HTTPServer | None = None
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def next_reply(self) -> dict | int:
        with self._lock:
            if self.replies:
                return self.replies.pop(0)
        return self.default_reply if self.default_reply is not None else completion()

    def record(self, payload: dict) -> None:
        with self._lock:
            self.requests.append(payload)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "FakeInferenceServer":
        server = self._server = HTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def completion(
    content: str = "The answer is \\boxed{4}",
    *,
    completion_tokens: int = 12,
    total_tokens: int | None = 30,
    router_approach: str | None = None,
    llm_calls: int | None = None,
) -> dict:
    """A chat-completion body, shaped the way optillm returns them."""
    usage: dict = {"completion_tokens": completion_tokens}
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
    if router_approach is not None:
        body["optillm_router_approach"] = router_approach
    if llm_calls is not None:
        body["optillm_llm_calls"] = llm_calls
    return body


def _make_handler(fake: FakeInferenceServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            length = int(self.headers.get("Content-Length", 0))
            fake.record(json.loads(self.rfile.read(length) or b"{}"))

            reply = fake.next_reply()
            if isinstance(reply, int):
                self._respond(reply, {"error": {"message": "scripted failure"}})
            else:
                self._respond(200, reply)

        def _respond(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass  # keep the test output readable

    return Handler
