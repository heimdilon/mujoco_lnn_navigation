from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.utils.map_generation import MAP_TYPES, build_generated_map, build_map_gallery_image, validate_map_config


def test_generated_map_families_are_valid():
    base = load_task_config("configs/task/open_clutter.yaml")
    for index, map_type in enumerate(MAP_TYPES):
        cfg = build_generated_map(base, f"unit_{map_type}", map_type, seed=700 + index, difficulty="easy", validation_resolution=0.22)
        result = validate_map_config(cfg, resolution=0.22)
        assert result.valid, result.reason
        assert cfg["map"]["enabled"] is True
        assert cfg["map"]["generated"]["type"] == map_type
        assert len(cfg["map"]["obstacles"]) >= 1
        assert result.waypoint_count >= 1


def test_generated_map_gallery_image_is_built():
    base = load_task_config("configs/task/open_clutter.yaml")
    configs = [
        build_generated_map(base, "unit_gallery_wall_gap", "wall_gap", seed=900, difficulty="easy", validation_resolution=0.22),
        build_generated_map(base, "unit_gallery_zigzag", "zigzag", seed=901, difficulty="easy", validation_resolution=0.22),
    ]
    image = build_map_gallery_image(configs, columns=2, cell_size=220)
    assert image.size == (440, 220)
    assert image.getbbox() is not None
