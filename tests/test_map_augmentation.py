import subprocess
import sys

import yaml

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.utils.map_augmentation import MapAugmentationSettings, build_augmented_map
from mujoco_lnn_nav.utils.map_generation import validate_map_config


def test_augmented_manual_map_is_valid():
    base = load_task_config("configs/maps/example_manual.yaml")
    settings = MapAugmentationSettings(
        start_goal_jitter=0.10,
        obstacle_jitter=0.03,
        obstacle_scale_jitter=0.02,
        validation_resolution=0.22,
        max_attempts=40,
    )
    cfg, result = build_augmented_map(base, "example_manual_aug_unit", seed=1234, settings=settings)

    assert result.valid, result.reason
    assert validate_map_config(cfg, resolution=0.22).valid
    assert cfg["map"]["base_map"] == "example_manual"
    assert cfg["map"]["augmented"]["source"] == "example_manual"
    assert len(cfg["map"]["obstacles"]) == len(base["map"]["obstacles"])
    assert cfg["map"]["start"][:2] != base["map"]["start"][:2]


def test_augment_maps_script_writes_split_without_holdout_training(tmp_path):
    split_path = tmp_path / "split.yaml"
    out_dir = tmp_path / "augmented"
    split_out = tmp_path / "augmented_split.yaml"
    gallery = tmp_path / "gallery.png"
    split_path.write_text(
        yaml.safe_dump(
            {
                "name": "unit_split",
                "seed": 77,
                "train_maps": ["configs/maps/example_manual.yaml"],
                "holdout_maps": ["configs/maps/custom_map_03.yaml"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/augment_maps.py",
            "--split-config",
            str(split_path),
            "--variants-per-map",
            "1",
            "--out",
            str(out_dir),
            "--split-out",
            str(split_out),
            "--gallery",
            str(gallery),
            "--validation-resolution",
            "0.22",
        ],
        check=True,
    )

    augmented_split = yaml.safe_load(split_out.read_text(encoding="utf-8"))
    assert augmented_split["holdout_maps"] == ["configs/maps/custom_map_03.yaml"]
    assert "configs/maps/custom_map_03.yaml" not in augmented_split["train_maps"]
    assert len(augmented_split["train_maps"]) == 2
    assert gallery.exists()
