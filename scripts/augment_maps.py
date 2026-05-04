from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mujoco_lnn_nav.config import load_task_config, load_yaml
from mujoco_lnn_nav.utils.map_augmentation import MapAugmentationSettings, build_augmented_map
from mujoco_lnn_nav.utils.map_generation import render_map_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create validated augmented variants from manual training maps.")
    parser.add_argument("--split-config", default="configs/splits/custom8_seed25462877008.yaml")
    parser.add_argument("--variants-per-map", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--split-out", default=None)
    parser.add_argument("--gallery", default=None)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--no-include-original", action="store_true")
    parser.add_argument("--start-goal-jitter", type=float, default=0.22)
    parser.add_argument("--yaw-jitter", type=float, default=0.45)
    parser.add_argument("--obstacle-jitter", type=float, default=0.07)
    parser.add_argument("--obstacle-scale-jitter", type=float, default=0.04)
    parser.add_argument("--obstacle-yaw-jitter", type=float, default=0.05)
    parser.add_argument("--validation-resolution", type=float, default=0.16)
    return parser.parse_args()


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def relative_repo_path(path: str | Path) -> str:
    path = repo_path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def default_name(split_config: dict[str, Any], variants_per_map: int) -> str:
    base = str(split_config.get("name", "manual_split"))
    return f"{base}_aug_v{variants_per_map}"


def main() -> None:
    args = parse_args()
    if args.variants_per_map < 1:
        raise ValueError("--variants-per-map must be at least 1.")

    split_path = repo_path(args.split_config)
    split_config = load_yaml(split_path)
    split_name = default_name(split_config, args.variants_per_map)
    seed = int(args.seed if args.seed is not None else split_config.get("seed", 0))
    out_dir = repo_path(args.out or f"configs/maps/augmented/{split_name}")
    split_out = repo_path(args.split_out or f"configs/splits/{split_name}.yaml")
    gallery_path = repo_path(args.gallery or f"results/map_gallery/{split_name}.png")
    include_original = not args.no_include_original

    settings = MapAugmentationSettings(
        start_goal_jitter=args.start_goal_jitter,
        yaw_jitter=args.yaw_jitter,
        obstacle_jitter=args.obstacle_jitter,
        obstacle_scale_jitter=args.obstacle_scale_jitter,
        obstacle_yaw_jitter=args.obstacle_yaw_jitter,
        validation_resolution=args.validation_resolution,
    )

    train_maps = [relative_repo_path(path) for path in split_config.get("train_maps", [])]
    holdout_maps = [relative_repo_path(path) for path in split_config.get("holdout_maps", [])]
    if not train_maps:
        raise ValueError(f"No train_maps found in {split_path}.")

    generated_paths: list[str] = []
    gallery_configs: list[dict[str, Any]] = []
    for map_index, train_map in enumerate(train_maps):
        base_cfg = load_task_config(repo_path(train_map))
        gallery_configs.append(base_cfg)
        base_name = str(base_cfg.get("map", {}).get("name", base_cfg.get("name", f"map_{map_index:02d}")))
        for variant_idx in range(args.variants_per_map):
            variant_name = f"{base_name}_aug_{variant_idx:02d}"
            variant_seed = seed + map_index * 10_000 + variant_idx * 101
            cfg, result = build_augmented_map(base_cfg, variant_name, variant_seed, settings=settings)
            out_path = out_dir / f"{variant_name}.yaml"
            write_yaml(out_path, cfg)
            generated_paths.append(relative_repo_path(out_path))
            gallery_configs.append(cfg)
            print(
                f"{relative_repo_path(out_path)} source={base_name} "
                f"obstacles={len(cfg['map']['obstacles'])} path={result.path_length:.2f}m"
            )

    augmented_split = {
        "name": split_name,
        "source_split": relative_repo_path(split_path),
        "seed": seed,
        "description": "Augmented training split generated from manual maps; holdout maps are copied unchanged.",
        "train_maps": [*train_maps, *generated_paths] if include_original else generated_paths,
        "holdout_maps": holdout_maps,
        "augmentation": {
            "variants_per_map": args.variants_per_map,
            "include_original_train_maps": include_original,
            **settings.to_dict(),
        },
        "notes": [
            "Only train_maps from source_split are augmented.",
            "Holdout maps are never augmented or used during training.",
            "Pure policy evaluation must leave auto-waypoints disabled.",
        ],
    }
    write_yaml(split_out, augmented_split)
    render_map_gallery(gallery_configs, gallery_path, columns=args.columns)

    print(f"Split: {relative_repo_path(split_out)}")
    print(f"Gallery: {relative_repo_path(gallery_path)}")
    print(f"Train maps: {len(augmented_split['train_maps'])}; holdout maps: {len(holdout_maps)}")


if __name__ == "__main__":
    main()
