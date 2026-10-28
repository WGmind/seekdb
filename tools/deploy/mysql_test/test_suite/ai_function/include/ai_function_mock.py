#!/usr/bin/env python3
"""Deterministic mock of the AI model services used by the ai_function mysqltests.

Endpoints
  OpenAI compatible
    POST /compatible-mode/v1/chat/completions
    POST /v1/embeddings
  Ollama native
    POST /api/generate
    POST /api/embed
    POST /api/embeddings

Every request is appended to <state>/events.jsonl together with the full
request body, so the tests (or a human) can assert what the server side really
received: rendered prompt, temperature/max_tokens, Authorization header, ...

Usage
  ai_function_mock.py start [PORT]   start in background (default port 8000)
  ai_function_mock.py stop  [PORT]   stop the background server
  ai_function_mock.py serve [PORT]   run in the foreground
  ai_function_mock.py reset [PORT]   truncate the event log
  ai_function_mock.py dump  [PORT]   pretty print the recorded events
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

DEFAULT_PORT = 8000
DEFAULT_DIM = 768


def state_dir(port):
    return "/tmp/seekdb_ai_mock_%d" % port


def state_path(port, name):
    return os.path.join(state_dir(port), name)


def read_events(port):
    events = []
    path = state_path(port, "events.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def stop_server(port):
    pid_path = state_path(port, "server.pid")
    if not os.path.exists(pid_path):
        return
    try:
        with open(pid_path, "r", encoding="ascii") as stream:
            pid = int(stream.read().strip())
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
    except (ValueError, ProcessLookupError):
        pass
    try:
        os.unlink(pid_path)
    except FileNotFoundError:
        pass


def start_server(port):
    os.makedirs(state_dir(port), exist_ok=True)
    stop_server(port)
    with open(state_path(port, "events.jsonl"), "w", encoding="utf-8"):
        pass
    output = open(state_path(port, "server.out"), "a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "serve", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    with open(state_path(port, "server.pid"), "w", encoding="ascii") as stream:
        stream.write(str(process.pid))
    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                return
        except OSError:
            if process.poll() is not None:
                break
            time.sleep(0.02)
    raise RuntimeError("ai_function mock did not start on port %d" % port)


def text_of(value):
    if value is None:
        return ""
    return str(value)


def chat_content(system_text, user_text):
    """Deterministic assistant answer derived from the request."""
    if "empty-response" in user_text:
        return ""
    if "base-10 integer" in system_text:
        return "7"
    if "valid JSON value" in system_text:
        return json.dumps({"category": "mock", "input": user_text}, ensure_ascii=False)
    return "mock: " + user_text


def embedding_vector(index, dims):
    return [float((index + 1) % 8) * 0.125] * dims


def serve(port):
    lock = threading.Lock()
    counter = {"seq": 0}
    log_path = state_path(port, "events.jsonl")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def send_json(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send_json(200, {"status": "ok", "service": "ai_function_mock"})

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(size)
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            parsed = urlparse(self.path)
            path = parsed.path

            messages = body.get("messages") or []
            system_text = ""
            user_text = ""
            for message in messages:
                role = message.get("role")
                if role == "system":
                    system_text = text_of(message.get("content"))
                elif role == "user":
                    user_text = text_of(message.get("content"))
            if not user_text and body.get("prompt") is not None:
                user_text = text_of(body.get("prompt"))
            if not user_text and body.get("input") is not None:
                value = body.get("input")
                if isinstance(value, list):
                    user_text = "|".join(text_of(item) for item in value)
                else:
                    user_text = text_of(value)

            status = 200
            if "always-fail" in path or "always-fail" in user_text:
                status = 400
                response = {"error": {"message": "injected failure", "type": "mock_error"}}
            elif path.endswith("/embeddings") or path.endswith("/embed"):
                dims = body.get("dimensions")
                try:
                    dims = int(dims)
                except (TypeError, ValueError):
                    dims = DEFAULT_DIM
                if dims <= 0:
                    dims = DEFAULT_DIM
                if "dim-mismatch" in user_text:
                    # deliberately answer with a different dimension than asked
                    dims = 4
                inputs = body.get("input")
                if not isinstance(inputs, list):
                    inputs = [inputs] if inputs is not None else [""]
                if path.endswith("/api/embeddings"):
                    # legacy Ollama single embedding endpoint
                    response = {"embedding": embedding_vector(0, dims)}
                elif path.endswith("/api/embed"):
                    response = {
                        "model": text_of(body.get("model")),
                        "embeddings": [
                            embedding_vector(i, dims) for i, _ in enumerate(inputs)
                        ],
                        "total_duration": 1,
                        "prompt_eval_count": 1,
                    }
                else:
                    response = {
                        "object": "list",
                        "model": text_of(body.get("model")),
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": embedding_vector(i, dims),
                            }
                            for i, _ in enumerate(inputs)
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                    if "reverse-order" in user_text:
                        # answer the inputs in reverse order, the "index" field
                        # still tells the client where each vector belongs
                        response["data"] = list(reversed(response["data"]))
            elif path.endswith("/rerank"):
                query = text_of(body.get("query"))
                documents = body.get("documents") or []
                if not isinstance(documents, list):
                    documents = [documents]
                scored = []
                for pos, doc in enumerate(documents):
                    doc_text = doc if isinstance(doc, str) else json.dumps(
                        doc, ensure_ascii=False, sort_keys=True)
                    if query and query.lower() in doc_text.lower():
                        score = 1.0
                    else:
                        # deterministic, strictly decreasing with position
                        score = 0.5 - 0.01 * pos
                    scored.append((score, pos))
                # descending score, stable on the original index
                scored.sort(key=lambda item: (-item[0], item[1]))
                if "reverse-results" in query:
                    # exercise the client's index mapping: answer out of order
                    scored = list(reversed(scored))
                top_n = body.get("top_n")
                if isinstance(top_n, int) and top_n > 0:
                    scored = scored[:top_n]
                response = {
                    "id": "rerank-mock",
                    "results": [
                        {
                            "index": pos,
                            "relevance_score": float(score),
                            "document": documents[pos],
                        }
                        for score, pos in scored
                    ],
                    "meta": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
                }
            elif path.endswith("/api/generate"):
                content = chat_content(system_text, user_text)
                response = {
                    "model": text_of(body.get("model")),
                    "created_at": "2026-01-01T00:00:00Z",
                    "response": content,
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
            else:
                # OpenAI compatible chat completion
                content = chat_content(system_text, user_text)
                response = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 1767225600,
                    "model": text_of(body.get("model")),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }

            with lock:
                counter["seq"] += 1
                event = {
                    "seq": counter["seq"],
                    "path": path,
                    "status": status,
                    "authorization": self.headers.get("Authorization", ""),
                    "system": system_text,
                    "user": user_text,
                    "body": body,
                }
                with open(log_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.send_json(status, response)

    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def dump_events(port):
    for event in read_events(port):
        print(json.dumps(event, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    if mode == "start":
        start_server(port)
        print("ai_function mock listening on 0.0.0.0:%d" % port)
    elif mode == "stop":
        stop_server(port)
        print("ai_function mock stopped (port %d)" % port)
    elif mode == "serve":
        serve(port)
    elif mode == "reset":
        with open(state_path(port, "events.jsonl"), "w", encoding="utf-8"):
            pass
        print("event log reset")
    elif mode == "dump":
        dump_events(port)
    else:
        raise SystemExit("unknown mode: " + mode)


if __name__ == "__main__":
    main()
