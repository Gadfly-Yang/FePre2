import argparse
import copy

import pytest

from scripts.reproduce import build_config


def _args(**overrides):
    defaults = {
        "benchmark": "deepfakebench",
        "checkpoint": "/tmp/model.pt",
        "config": None,
        "data_json_folder": "/tmp/json",
        "base_data_path": "/tmp/data",
        "data_root": None,
        "datasets": "all",
        "device": None,
        "output_dir": "/tmp/out",
        "prototypes_path": None,
    }
    values = {**defaults, **overrides}
    return argparse.Namespace(**values)


def test_build_config_sets_deepfakebench_paths(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    base = {
        "dataset_mode": "universal",
        "weight_path": "old.pt",
        "output_dir": "outputs/old",
        "test_dataset": "Celeb-DF-v2",
    }

    config = build_config(base, _args(checkpoint=str(checkpoint)))

    assert config["dataset_mode"] == "deepfakebench"
    assert config["weight_path"] == str(checkpoint)
    assert config["data_json_folder"] == "E:\\tmp\\json"
    assert config["base_data_path"] == "E:\\tmp\\data"
    assert config["test_dataset"] == "all"
    assert config["output_dir"] == "E:\\tmp\\out"
    assert base["dataset_mode"] == "universal"


def test_build_config_sets_universal_data_root(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    base = {
        "dataset_mode": "deepfakebench",
        "weight_path": "old.pt",
        "universal": {"data_root": "old/root", "test_generators": "all"},
    }

    config = build_config(
        base,
        _args(
            benchmark="universal",
            checkpoint=str(checkpoint),
            data_root=str(tmp_path / "universal"),
        ),
    )

    assert config["dataset_mode"] == "universal"
    assert config["weight_path"] == str(checkpoint)
    assert config["universal"]["data_root"] == str(tmp_path / "universal")
    assert config["universal"]["test_generators"] == "all"


def test_build_config_rejects_missing_checkpoint():
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        build_config({}, _args(checkpoint="/tmp/missing.pt"))
