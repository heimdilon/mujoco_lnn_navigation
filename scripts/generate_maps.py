from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.utils.map_generation import MAP_TYPES, build_generated_map, render_map_gallery, validate_map_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validated fixed-map YAML configs for MuJoCo navigation.")
    parser.add_argument("--type", choices=[*MAP_TYPES, "all"], default="all", help="Map family to generate.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--count", type=int, default=12, help="Number of maps to generate.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--prefix", default="gen", help="Output map name prefix.")
    parser.add_argument("--base-task", default="configs/task/open_clutter.yaml")
    parser.add_argument("--out", default="configs/maps/generated", help="Output directory for YAML maps.")
    parser.add_argument("--gallery", default="results/map_gallery/generated_maps.png", help="Optional PNG gallery path.")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--validation-resolution", type=float, default=0.16)
    parser.add_argument("--no-gallery", action="store_true")
    return parser.parse_args()


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def selected_type(requested: str, index: int) -> str:
    if requested != "all":
        return requested
    return MAP_TYPES[index % len(MAP_TYPES)]


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1.")

    base_config = load_task_config(args.base_task)
    out_dir = Path(args.out)
    generated: list[dict] = []
    for idx in range(args.count):
        map_type = selected_type(args.type, idx)
        name = f"{args.prefix}_{map_type}_{args.difficulty}_{idx:03d}"
        cfg = build_generated_map(
            base_config=base_config,
            map_name=name,
            map_type=map_type,
            seed=args.seed + idx,
            difficulty=args.difficulty,
            validation_resolution=args.validation_resolution,
        )
        result = validate_map_config(cfg, resolution=args.validation_resolution)
        if not result.valid:
            raise RuntimeError(f"Generated invalid map {name}: {result.reason}")
        path = out_dir / f"{name}.yaml"
        write_yaml(path, cfg)
        generated.append(cfg)
        print(
            f"{path} type={map_type} difficulty={args.difficulty} "
            f"obstacles={len(cfg['map']['obstacles'])} path={result.path_length:.2f}m"
        )

    if not args.no_gallery:
        render_map_gallery(generated, args.gallery, columns=args.columns)
        print(f"Gallery: {Path(args.gallery)}")


if __name__ == "__main__":
    main()
