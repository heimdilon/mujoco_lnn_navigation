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
