import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


BENCHMARK_CONFIGS = {
    "deepfakebench": "configs/test_deepfakebench.yaml",
    "universal": "configs/test_universal.yaml",
}


def build_config(base_config, args):
    config = copy.deepcopy(base_config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")

    config["weight_path"] = str(checkpoint)
    config["test_dataset"] = args.datasets
    config["output_dir"] = str(
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs") / f"eval_{args.benchmark}_reproduced"
    )
    if args.device:
        config["device"] = args.device
    if args.prototypes_path:
        config["prototypes_path"] = str(Path(args.prototypes_path).expanduser().resolve())

    if args.benchmark == "deepfakebench":
        config["dataset_mode"] = "deepfakebench"
        config["data_json_folder"] = str(Path(args.data_json_folder).expanduser().resolve())
        config["base_data_path"] = str(Path(args.base_data_path).expanduser().resolve())
    else:
        config["dataset_mode"] = "universal"
        config.setdefault("universal", {})
        config["universal"]["data_root"] = str(Path(args.data_root).expanduser().resolve())

    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a released checkpoint without editing YAML files."
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_CONFIGS),
        required=True,
        help="Benchmark suite to evaluate.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the released .pt file.")
    parser.add_argument("--config", help="Optional test YAML override.")
    parser.add_argument("--datasets", default="all", help="Dataset name, list, or 'all'.")
    parser.add_argument("--device", help="Override device, for example cuda:0.")
    parser.add_argument("--output-dir", help="Directory for generated config and metrics.")
    parser.add_argument("--prototypes-path", help="Optional prototypes sidecar file.")
    parser.add_argument(
        "--data-json-folder",
        help="DeepfakeBench dataset JSON folder.",
    )
    parser.add_argument(
        "--base-data-path",
        help="DeepfakeBench root used to resolve relative frame paths.",
    )
    parser.add_argument(
        "--data-root",
        help="UniversalFakeDetect root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the config and print the command without running evaluation.",
    )
    args = parser.parse_args(argv)

    if args.benchmark == "deepfakebench":
        if not args.data_json_folder or not args.base_data_path:
            parser.error(
                "--data-json-folder and --base-data-path are required for the deepfakebench benchmark"
            )
    elif not args.data_root:
        parser.error("--data-root is required for the universal benchmark")
    return args


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config or repo_root / BENCHMARK_CONFIGS[args.benchmark])
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = build_config(base_config, args)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = output_dir / "test_config.yaml"
    generated_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    command = [
        sys.executable,
        str(repo_root / "test.py"),
        "--config",
        str(generated_config),
        "--datasets",
        args.datasets,
    ]
    if args.device:
        command.extend(["--device", args.device])

    print("Generated config:", generated_config)
    print("Running:", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=repo_root, check=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
