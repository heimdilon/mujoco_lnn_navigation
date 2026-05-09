from scripts.map_editor import build_task_config


def test_map_editor_payload_builds_fixed_task_config():
    cfg = build_task_config(
        {
            "name": "unit_custom",
            "base_task": "sparse_goal",
            "arena_half": 3.0,
            "start": {"x": -1.0, "y": -1.0, "yaw": 0.5},
            "goal": {"x": 1.0, "y": 1.0},
            "obstacles": [
                {"id": "wall", "shape": "box", "kind": "wall", "x": 0.0, "y": 0.0, "half_x": 0.1, "half_y": 0.6, "yaw": 0.75},
            ],
            "jitter": {"enabled": False},
        }
    )
    assert cfg["name"] == "unit_custom"
    assert cfg["arena"]["half_size"] == 3.0
    assert cfg["obstacles"]["count"] == [1, 1]
    assert cfg["map"]["base_task"] == "sparse_goal"
    assert cfg["map"]["start"] == [-1.0, -1.0, 0.5]
    assert cfg["map"]["goal"] == [1.0, 1.0]
    assert cfg["map"]["obstacles"][0]["shape"] == "box"
    assert cfg["map"]["obstacles"][0]["kind"] == "wall"
    assert cfg["map"]["obstacles"][0]["yaw"] == 0.75


def test_map_editor_payload_persists_safety_config():
    cfg = build_task_config(
        {
            "name": "unit_custom_safety",
            "base_task": "sparse_goal",
            "arena_half": 3.0,
            "start": {"x": -1.0, "y": -1.0, "yaw": 0.0},
            "goal": {"x": 1.0, "y": 1.0},
            "obstacles": [],
            "jitter": {"enabled": False},
            "safety": {
                "robot_radius": 0.1,
                "goal_radius": 0.3,
                "obstacle_min_gap": 0.2,
            },
        }
    )
    assert cfg["robot"]["radius"] == 0.1
    assert cfg["goal"]["radius"] == 0.3
    assert cfg["obstacles"]["min_clearance"] == 0.2


def test_map_editor_exposes_reset_controls():
    html = __import__("pathlib").Path("tools/map_editor/index.html").read_text(encoding="utf-8")
    app = __import__("pathlib").Path("tools/map_editor/app.js").read_text(encoding="utf-8")
    assert 'id="clearObstacles"' in html
    assert 'id="newMap"' in html
    assert "/app.js?v=" in html
    assert 'addEventListener("click", clearObstacles)' in app
    assert 'addEventListener("click", newBlankMap)' in app


def test_map_editor_exposes_safety_analysis():
    html = __import__("pathlib").Path("tools/map_editor/index.html").read_text(encoding="utf-8")
    app = __import__("pathlib").Path("tools/map_editor/app.js").read_text(encoding="utf-8")
    server = __import__("pathlib").Path("scripts/map_editor.py").read_text(encoding="utf-8")
    assert 'id="robotRadius"' in html
    assert 'id="nearestGap"' in html
    assert 'id="robotCorridor"' in html
    assert 'id="safetyWarnings"' in html
    assert "function obstaclePairClearance" in app
    assert "function analyzeMap" in app
    assert "robot_radius" in app
    assert 'path.startswith("/api/base-tasks/")' in server
