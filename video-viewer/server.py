#!/usr/bin/env python3
"""Eval video viewer — auto-discovers MP4s and serves a modern playback UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
STACK_ROOT = HERE.parent.parent  # leatherback-stack
INDEX = HERE / "index.html"

SKIP_DIRS = {
    ".venv",
    ".git",
    "node_modules",
    "__pycache__",
    ".cursor",
    "IsaacLab",
    "extscache",
    "extscore",
}

VIDEO_SOURCES = (
    ("eval", STACK_ROOT / "eval-nav" / "logs"),
    ("play", STACK_ROOT / "task-franka-pickplace-multibase" / "best_policy" / "videos"),
)


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _parse_eval_meta(rel: Path) -> dict:
    """Extract eval run folder and scene from path like logs/.../eval_run_*/videos/eval/*.mp4."""
    parts = rel.parts
    run = None
    scene = None
    for i, p in enumerate(parts):
        if p.startswith("eval_run_"):
            run = p
        if p == "eval" and i + 1 < len(parts):
            scene = parts[-1]
    return {"run": run, "scene": scene}


def discover_videos() -> list[dict]:
    videos: list[dict] = []
    seen: set[str] = set()

    for source, root in VIDEO_SOURCES:
        if not root.is_dir():
            continue
        for path in root.rglob("*.mp4"):
            if _should_skip(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(STACK_ROOT).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            meta = _parse_eval_meta(path.relative_to(STACK_ROOT))
            videos.append(
                {
                    "id": rel,
                    "url": f"/video/{rel}",
                    "path": rel,
                    "name": path.stem,
                    "filename": path.name,
                    "source": source,
                    "run": meta["run"],
                    "modified": stat.st_mtime,
                    "modified_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": stat.st_size,
                    "size_human": _human_size(stat.st_size),
                }
            )

    videos.sort(key=lambda v: v["modified"], reverse=True)
    return videos


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class Handler(BaseHTTPRequestHandler):
    server_version = "EvalVideoViewer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        if args and isinstance(args[0], str) and args[0].startswith("GET /api/"):
            return
        super().log_message(fmt, *args)

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _safe_path(self, rel: str) -> Path | None:
        rel = unquote(rel).lstrip("/")
        if ".." in rel or rel.startswith("/"):
            return None
        full = (STACK_ROOT / rel).resolve()
        try:
            full.relative_to(STACK_ROOT.resolve())
        except ValueError:
            return None
        return full

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._send_file(INDEX, "text/html; charset=utf-8")

        if path == "/api/videos":
            return self._send_json({"videos": discover_videos(), "scanned_at": time.time()})

        if path.startswith("/video/"):
            rel = path[len("/video/") :]
            full = self._safe_path(rel)
            if full is None or not full.is_file() or full.suffix.lower() != ".mp4":
                self.send_error(404)
                return
            return self._send_file(full, "video/mp4")

        self.send_error(404)

    def _send_file(self, full: Path, content_type: str) -> None:
        if not full.is_file():
            self.send_error(404)
            return
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval video viewer server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    mimetypes.init()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[EvalVideoViewer] Serving UI  → http://{args.host}:{args.port}/")
    print(f"[EvalVideoViewer] Scan root → {STACK_ROOT}")
    print(f"[EvalVideoViewer] Found     → {len(discover_videos())} video(s)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[EvalVideoViewer] Stopped.")


if __name__ == "__main__":
    main()
