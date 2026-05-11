from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate several policy classes on the same manual maps.")
    parser.add_argument("--map-configs", nargs="+", required=True)
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train" / "bc_gru_manual_maps.yaml"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["mlp", "cfc", "gru", "lstm"],
        choices=["mlp", "cfc", "cfc_deep", "gru", "lstm"],
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dagger-iterations", type=int, default=2)
    parser.add_argument("--dagger-rollouts-per-map", type=int, default=2)
    parser.add_argument("--dagger-epochs", type=int, default=25)
    parser.add_argument("--dagger-noise", type=float, default=0.04)
    parser.add_argument("--dagger-expert-mix", type=float, default=0.25)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--goal-observation-max", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        handle.write(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    print(process.stdout, end="")


def read_summary(path: Path, policy: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = dict(row)
            row["policy"] = policy
            rows.append(row)
    return rows


def numeric(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def write_comparison(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "map",
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_steps",
        "mean_final_distance",
        "output_dir",
        "gif",
        "png",
    ]
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    policy_rows = []
    for policy in sorted({row["policy"] for row in rows}):
        matching = [row for row in rows if row["policy"] == policy]
        policy_rows.append(
            {
                "policy": policy,
                "maps": len(matching),
                "mean_success_rate": sum(numeric(row["success_rate"]) for row in matching) / max(1, len(matching)),
                "mean_collision_rate": sum(numeric(row["collision_rate"]) for row in matching) / max(1, len(matching)),
                "mean_timeout_rate": sum(numeric(row["timeout_rate"]) for row in matching) / max(1, len(matching)),
                "mean_steps": sum(numeric(row["mean_steps"]) for row in matching) / max(1, len(matching)),
                "mean_final_distance": sum(numeric(row["mean_final_distance"]) for row in matching) / max(1, len(matching)),
            }
        )
    with (output_dir / "policy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "policy",
            "maps",
            "mean_success_rate",
            "mean_collision_rate",
            "mean_timeout_rate",
            "mean_steps",
            "mean_final_distance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(policy_rows)

    (output_dir / "comparison_summary.json").write_text(
        json.dumps({"maps": rows, "policies": policy_rows}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "results" / args.run_name
    log_path = output_dir / "commands.log"
    all_rows: list[dict] = []

    for policy in args.policies:
        train_run = f"{args.run_name}/{policy}_train"
        eval_run = f"{args.run_name}/{policy}_eval"
        checkpoint = ROOT / "results" / args.run_name / f"{policy}_train" / "latest.pt"

        if not args.eval_only and not (args.skip_existing and checkpoint.exists()):
            train_cmd = [
                str(PYTHON),
                "scripts\\train_bc.py",
                "--map-configs",
                *args.map_configs,
                "--train-config",
                args.train_config,
                "--run-name",
                train_run,
                "--policy",
                policy,
                "--epochs",
                str(args.epochs),
                "--save-interval",
                "20",
                "--dagger-iterations",
                str(args.dagger_iterations),
                "--dagger-rollouts-per-map",
                str(args.dagger_rollouts_per_map),
                "--dagger-epochs",
                str(args.dagger_epochs),
                "--dagger-noise",
                str(args.dagger_noise),
                "--dagger-expert-mix",
                str(args.dagger_expert_mix),
                "--device",
                args.device,
                "--no-final-eval",
            ]
            if args.hidden_size is not None:
                train_cmd.extend(["--hidden-size", str(args.hidden_size)])
            if args.learning_rate is not None:
                train_cmd.extend(["--learning-rate", str(args.learning_rate)])
            print(f"\n=== train {policy} ===")
            run_command(train_cmd, log_path)

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint missing for {policy}: {checkpoint}")

        eval_cmd = [
            str(PYTHON),
            "scripts\\batch_evaluate.py",
            "--map-configs",
            *args.map_configs,
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(args.episodes),
            "--run-name",
            eval_run,
            "--max-steps",
            str(args.max_steps),
            "--goal-observation-max",
            str(args.goal_observation_max),
            "--device",
            args.device,
        ]
        if args.no_gif:
            eval_cmd.append("--no-gif")
        if args.no_png:
            eval_cmd.append("--no-png")
        print(f"\n=== eval {policy} ===")
        run_command(eval_cmd, log_path)
        all_rows.extend(read_summary(ROOT / "results" / args.run_name / f"{policy}_eval" / "summary.csv", policy))

    write_comparison(all_rows, output_dir)
    print(f"\ncomparison: {output_dir / 'comparison_summary.csv'}")
    print(f"policy summary: {output_dir / 'policy_summary.csv'}")


if __name__ == "__main__":
    main()
