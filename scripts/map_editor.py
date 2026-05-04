from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
EDITOR_DIR = ROOT / "tools" / "map_editor"
MAP_DIR = ROOT / "configs" / "maps"
TASK_DIR = ROOT / "configs" / "task"


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()).strip("._")
    if not cleaned:
        raise ValueError("Map name is required.")
    return cleaned[:80]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def map_path(name: str) -> Path:
    return MAP_DIR / f"{safe_name(name)}.yaml"


def build_task_config(payload: dict) -> dict:
    base_name = safe_name(payload.get("base_task", "open_clutter"))
    base_path = TASK_DIR / f"{base_name}.yaml"
    if not base_path.exists():
        raise ValueError(f"Unknown base task: {base_name}")
    config = load_yaml(base_path)
    map_name = safe_name(payload["name"])
    obstacles = payload.get("obstacles", [])
    config["name"] = map_name
    config["arena"]["half_size"] = float(payload.get("arena_half", config["arena"]["half_size"]))
    config["obstacles"]["count"] = [len(obstacles), len(obstacles)]
    config["map"] = {
        "enabled": True,
        "name": map_name,
        "base_task": base_name,
        "start": [
            float(payload["start"]["x"]),
            float(payload["start"]["y"]),
            float(payload["start"].get("yaw", 0.0)),
        ],
        "goal": [
            float(payload["goal"]["x"]),
            float(payload["goal"]["y"]),
        ],
        "jitter": {
            "enabled": bool(payload.get("jitter", {}).get("enabled", False)),
            "start_std": float(payload.get("jitter", {}).get("start_std", 0.0)),
            "goal_std": float(payload.get("jitter", {}).get("goal_std", 0.0)),
            "yaw_std": float(payload.get("jitter", {}).get("yaw_std", 0.0)),
        },
        "obstacles": normalize_obstacles(obstacles),
    }
    return config


def normalize_obstacles(obstacles: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for idx, item in enumerate(obstacles):
        shape = str(item.get("shape", "cylinder"))
        if shape == "box":
            half_x = float(item.get("half_x", 0.2))
            half_y = float(item.get("half_y", 0.2))
            normalized.append(
                {
                    "id": str(item.get("id", f"box_{idx}")),
                    "shape": "box",
                    "x": float(item.get("x", 0.0)),
                    "y": float(item.get("y", 0.0)),
                    "half_x": half_x,
                    "half_y": half_y,
                    "radius": float(item.get("radius", max(half_x, half_y))),
                    "yaw": float(item.get("yaw", 0.0)),
                    "kind": str(item.get("kind", "box")),
                }
            )
        else:
            radius = float(item.get("radius", 0.2))
            normalized.append(
                {
                    "id": str(item.get("id", f"cylinder_{idx}")),
                    "shape": "cylinder",
                    "x": float(item.get("x", 0.0)),
                    "y": float(item.get("y", 0.0)),
                    "radius": radius,
                    "half_x": radius,
                    "half_y": radius,
                }
            )
    return normalized


class MapEditorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(EDITOR_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[map-editor] " + fmt % args + "\n")

    def send_json(self, status: int, data: dict | list) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/base-tasks":
            tasks = sorted(item.stem for item in TASK_DIR.glob("*.yaml"))
            self.send_json(200, {"tasks": tasks})
            return
        if path == "/api/maps":
            maps = sorted(item.stem for item in MAP_DIR.glob("*.yaml"))
            self.send_json(200, {"maps": maps})
            return
        if path.startswith("/api/maps/"):
            name = unquote(path.removeprefix("/api/maps/"))
            file_path = map_path(name)
            if not file_path.exists():
                self.send_json(404, {"error": "Map not found."})
                return
            self.send_json(200, {"name": file_path.stem, "path": str(file_path), "config": load_yaml(file_path)})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/maps":
            self.send_json(404, {"error": "Unknown endpoint."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            config = build_task_config(payload)
            file_path = map_path(payload["name"])
            write_yaml(file_path, config)
            self.send_json(200, {"ok": True, "name": file_path.stem, "path": str(file_path), "config": config})
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/maps/"):
            self.send_json(404, {"error": "Unknown endpoint."})
            return
        try:
            file_path = map_path(unquote(path.removeprefix("/api/maps/")))
            if file_path.exists():
                file_path.unlink()
            self.send_json(200, {"ok": True})
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), MapEditorHandler)
    print(f"Map editor: http://{args.host}:{args.port}")
    print(f"Maps are written to: {MAP_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
