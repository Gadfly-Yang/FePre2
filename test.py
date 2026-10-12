import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import (
    UNIVERSAL_DIFFUSION_PAIRS,
    UNIVERSAL_TEST_ORDER,
    load_universal_diffusion_records,
    load_universal_records,
    make_processor,
    split_dataset_names,
)
from model import ClipMlpClassifier, resolve_visual_model_config
from train import evaluate, make_loader, update_prototypes


DEEPFAKEBENCH_DATASETS = [
    "FaceForensics++",
    "FF-DF",
    "FF-F2F",
    "FF-FS",
    "FF-NT",
    "FaceShifter",
    "DeepFakeDetection",
    "Celeb-DF-v1",
    "Celeb-DF-v2",
    "DFDC",
    "DFDCP",
    "UADFV",
]

# FF++ sub-variants excluded from "all" because FaceForensics++ already covers them.
_DEEPFAKEBENCH_EXCLUDE_FROM_ALL = {"FF-DF", "FF-F2F", "FF-FS", "FF-NT"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed CLIPMLPa checkpoint on all DeepfakeBench or UniversalFakeDetect datasets."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "test_deepfakebench.yaml"),
        help="Test YAML config. It should contain weight_path and dataset settings.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to output_dir/deepfakebench_test_results.json.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names. Defaults to test_dataset in config, or all when set to 'all'.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    config["config_path"] = str(path)
    return config


def find_weight_path(config):
    candidates = [
        config.get("weight_path"),
        config.get("checkpoint_path"),
        config.get("test_weight_path"),
    ]
    if config.get("weight_dir") and config.get("weight_name"):
        candidates.append(str(Path(config["weight_dir"]) / str(config["weight_name"])))

    for candidate in candidates:
        if candidate:
            return Path(candidate).expanduser()
    raise ValueError("test config must define weight_path, checkpoint_path, test_weight_path, or weight_dir + weight_name")


def load_checkpoint(weight_path, device):
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    return torch.load(weight_path, map_location=device)


def merge_config(test_config, checkpoint):
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    merged = copy.deepcopy(checkpoint_config)
    merged.update(test_config)
    return merged


def build_model(config, checkpoint, device):
    lora_cfg = config.get("lora", {}) or {}
    ae_cfg = config.get("ae", {}) or {}
    ae_type = str(ae_cfg.get("type", "ae")).strip().lower()
    if ae_type not in {"ae", "vae"}:
        raise ValueError(f"ae.type must be 'ae' or 'vae', got {ae_type}")

    visual_cfg = resolve_visual_model_config(config)
    model = ClipMlpClassifier(
        clip_model=config.get("clip_model", "openai/clip-vit-large-patch14"),
        visual_model_type=visual_cfg["type"],
        visual_model_name=visual_cfg["name"],
        feature_pooling=visual_cfg["feature_pooling"],
        visual_hidden_size=(config.get("visual_model", {}) or {}).get("hidden_size"),
        feature_layer=config.get("feature_layer", "final"),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        dropout=float(config.get("dropout", 0.1)),
        freeze_clip=bool(config.get("freeze_clip", True)),
        normalize_clip_features=bool(config.get("normalize_clip_features", True)),
        use_lora=bool(lora_cfg.get("enabled", True)),
        lora_r=int(lora_cfg.get("r", 2)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.25)),
        lora_target_modules=lora_cfg.get("target_modules"),
        cache_dir=config.get("cache_dir"),
        local_files_only=bool(config.get("local_files_only", False)),
        use_ae=bool(ae_cfg.get("enabled", False)),
        ae_type=ae_type,
    ).to(device)

    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _tensor_or_none(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _prototype_value(prototypes, str_key, int_key):
    if str_key in prototypes:
        return prototypes[str_key]
    if int_key in prototypes:
        return prototypes[int_key]
    return None


def _prototype_dict_from_payload(data):
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("prototypes"), dict):
        return data["prototypes"]
    if isinstance(data.get("best_prototypes"), dict):
        return data["best_prototypes"]
    if isinstance(data.get("final_prototypes"), dict):
        return data["final_prototypes"]
    return data


def _prototypes_from_payload(data, device):
    prototypes = _prototype_dict_from_payload(data)
    if not isinstance(prototypes, dict):
        return None
    real = _tensor_or_none(_prototype_value(prototypes, "0", 0), device)
    fake = _tensor_or_none(_prototype_value(prototypes, "1", 1), device)
    if real is None or fake is None:
        return None
    return {0: real, 1: fake}


def load_prototypes_from_json(path, device):
    if not path:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        return None
    with path.open("r") as handle:
        data = json.load(handle)
    return _prototypes_from_payload(data, device)


def load_prototypes_from_checkpoint(checkpoint, device):
    """Extract prototypes embedded inside a training checkpoint.

    Training checkpoints saved after this feature was added contain a
    ``prototypes`` key directly in the top-level dict.
    """
    if not isinstance(checkpoint, dict):
        return None
    return _prototypes_from_payload(checkpoint, device)


def load_prototypes_from_pt(path, device):
    """Load prototypes from a torch.save'd .prototypes.pt file.

    This is the primary format produced by train.py at the end of training.
    """
    if not path:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return _prototypes_from_payload(data, device)


def _prototype_pt_path(weight_path):
    """Return the companion .prototypes.pt path for a given weight file."""
    w = Path(weight_path)
    return w.parent / w.name.replace(".pt", ".prototypes.pt")


def default_metrics_path(config, weight_path):
    if config.get("prototypes_path"):
        return config["prototypes_path"]
    # 1) Companion .prototypes.pt next to the weight file (written by train.py)
    pt_path = _prototype_pt_path(weight_path)
    if pt_path.exists():
        return str(pt_path)
    # 2) Legacy .prototypes.json sidecar
    sidecar = Path(weight_path).with_suffix(".prototypes.json")
    if sidecar.exists():
        return str(sidecar)
    return None


@torch.no_grad()
def estimate_prototypes(model, config, processor, device):
    proto_config = copy.deepcopy(config)
    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()

    if dataset_mode == "universal":
        universal_cfg = config.get("universal", {}) or {}
        data_root = universal_cfg.get("data_root")
        if not data_root:
            raise ValueError("universal.data_root must be set in the config to estimate UniversalFakeDetect prototypes")
        proto_dataset = config.get("prototype_dataset")
        if proto_dataset:
            records = load_universal_records(
                data_root, split="train",
                categories=proto_dataset,
            )
        else:
            records = load_universal_records(
                data_root, split="train",
                categories=universal_cfg.get("train_categories", "all"),
                exclude=list(universal_cfg.get("train_exclude", [])),
            )
        from torch.utils.data import DataLoader
        from data import FrameDataset, build_clip_transform, collate_frames

        transform = build_clip_transform(processor, train=False, image_size=config.get("image_size"))
        dataset = FrameDataset(records, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("eval_batch_size", 32)),
            shuffle=False,
            num_workers=int(config.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_frames,
        )
    else:
        proto_dataset = proto_config.get("prototype_dataset", proto_config.get("train_dataset"))
        if not proto_dataset:
            raise ValueError("No prototypes found. Set prototypes_path or provide prototype_dataset/train_dataset to estimate prototypes.")
        proto_config["train_dataset"] = proto_dataset
        proto_config["train_split"] = proto_config.get("prototype_split", proto_config.get("train_split", "train"))
        loader, records = make_loader(
            proto_config,
            split=proto_config["train_split"],
            train=True,
            processor=processor,
        )

    prototypes = {0: None, 1: None}
    momentum = float(proto_config.get("proto_momentum", 0.99))
    model.eval()

    from tqdm import tqdm
    for batch in tqdm(loader, desc="Estimating prototypes", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["binary_label"].to(device, non_blocking=True)
        outputs = model(images)
        update_prototypes(prototypes, outputs["embedding"].detach(), labels, momentum)
    if prototypes[0] is None or prototypes[1] is None:
        raise RuntimeError(f"Could not estimate both real/fake prototypes from records")
    return prototypes


# ---------------------------------------------------------------------------
#  Dataset name helpers
# ---------------------------------------------------------------------------

def available_deepfakebench_datasets(config):
    folder = Path(config["data_json_folder"])
    available = []
    for name in DEEPFAKEBENCH_DATASETS:
        if (folder / f"{name}.json").exists():
            available.append(name)
    return available


def available_universal_datasets(config):
    """Return all testable dataset keys following C2P-CLIP ordering."""
    return [key for key, _, _ in UNIVERSAL_TEST_ORDER]


def requested_datasets(config, cli_value=None):
    value = cli_value if cli_value is not None else config.get("test_dataset", "all")
    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()

    if dataset_mode == "universal":
        if isinstance(value, str) and value.strip().lower() in {"all", "universal", "universal_all"}:
            datasets = available_universal_datasets(config)
        else:
            datasets = split_dataset_names(value)
    else:
        if isinstance(value, str) and value.strip().lower() in {"all", "deepfakebench", "deepfakebench_all"}:
            datasets = available_deepfakebench_datasets(config)
            # Exclude FF++ sub-variants: FaceForensics++ already covers FF-DF/F2F/FS/NT
            datasets = [d for d in datasets if d not in _DEEPFAKEBENCH_EXCLUDE_FROM_ALL]
        else:
            datasets = split_dataset_names(value)

    if not datasets:
        raise ValueError("No test datasets requested or found")
    return datasets


# ---------------------------------------------------------------------------
#  Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(model, config, dataset_name, processor, prototypes, device):
    eval_config = copy.deepcopy(config)
    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()

    if dataset_mode == "universal":
        loader, records = make_loader(
            eval_config,
            split="test",
            train=False,
            processor=processor,
            include_video_name=True,
            dataset_names=dataset_name,
        )
    else:
        eval_config["test_dataset"] = dataset_name
        loader, records = make_loader(
            eval_config,
            split=eval_config.get("test_split", "test"),
            train=False,
            processor=processor,
            include_video_name=True,
        )
    metrics = evaluate(model, loader, prototypes, device)
    metrics["records"] = len(records)
    return metrics


def format_metric(value):
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _metric_delta(expected, actual):
    if expected is None or actual is None:
        return None
    return float(actual) - float(expected)


def _checkpoint_reference_metrics(checkpoint, dataset_mode, datasets):
    if not isinstance(checkpoint, dict):
        return None, None
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        return None, None
    if str(dataset_mode).strip().lower() == "universal":
        if len(datasets) > 1:
            return metrics, "checkpoint aggregate"
        per_dataset = metrics.get("per_dataset")
        if isinstance(per_dataset, dict) and datasets and datasets[0] in per_dataset:
            return per_dataset[datasets[0]], f"checkpoint per_dataset[{datasets[0]}]"
        return metrics, "checkpoint aggregate"
    return metrics, "checkpoint metrics"


def print_metric_deltas(reference_metrics, actual_metrics, label):
    if not reference_metrics:
        return {}
    deltas = {}
    print(f"[reproduce] comparing against {label}")
    for key in ("loss", "acc", "auc", "video_auc", "ap"):
        if key not in reference_metrics or key not in actual_metrics:
            continue
        delta = _metric_delta(reference_metrics.get(key), actual_metrics.get(key))
        if delta is None:
            continue
        deltas[key] = delta
        print(
            f"[reproduce] {key}: checkpoint={format_metric(reference_metrics.get(key))} "
            f"standalone={format_metric(actual_metrics.get(key))} delta={delta:+.6g}"
        )
    return deltas


def main():
    args = parse_args()
    test_config = load_config(args.config)
    device = args.device or test_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = find_weight_path(test_config)
    checkpoint = load_checkpoint(weight_path, device)
    config = merge_config(test_config, checkpoint)
    config["device"] = device

    model = build_model(config, checkpoint, device)
    visual_cfg = resolve_visual_model_config(config)
    processor = make_processor(
        visual_cfg["name"],
        model_type=visual_cfg["type"],
        cache_dir=config.get("cache_dir"),
        local_files_only=bool(config.get("local_files_only", False)),
    )

    # ---- load prototypes: checkpoint-embedded > companion .prototypes.pt > .json > error ----
    prototypes = load_prototypes_from_checkpoint(checkpoint, device)
    if prototypes is not None:
        prototype_source = f"checkpoint ({weight_path})"
        print(f"[prototypes] Loaded from checkpoint")
    else:
        proto_path = default_metrics_path(config, weight_path)
        if proto_path and str(proto_path).endswith(".pt"):
            prototypes = load_prototypes_from_pt(proto_path, device)
        elif proto_path:
            prototypes = load_prototypes_from_json(proto_path, device)
        if prototypes is not None:
            prototype_source = str(proto_path)
            print(f"[prototypes] Loaded from {proto_path}")
        else:
            raise FileNotFoundError(
                "No saved prototypes found for this weight. Expected either prototypes embedded "
                f"in the checkpoint, or companion file: {_prototype_pt_path(weight_path)}"
            )

    datasets = requested_datasets(config, args.datasets)
    results = {
        "config": str(args.config),
        "weight_path": str(weight_path),
        "prototype_source": prototype_source,
        "device": str(device),
        "dataset_mode": str(config.get("dataset_mode", "deepfakebench")),
        "datasets": {},
    }

    print(f"weight: {weight_path}")
    print(f"dataset_mode: {config.get('dataset_mode', 'deepfakebench')}")
    print(f"prototype_source: {prototype_source}")
    print(f"datasets ({len(datasets)}): {', '.join(datasets)}")

    from tqdm import tqdm

    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()

    actual_for_compare = None
    if dataset_mode == "universal":
        # C2P-CLIP grouped output
        prev_group = None
        per_dataset = {}
        universal_tasks = [(k, d, g) for k, d, g in UNIVERSAL_TEST_ORDER if k in datasets]
        for key, display, group in tqdm(universal_tasks, desc="Datasets"):
            try:
                metrics = evaluate_dataset(model, config, key, processor, prototypes, device)
            except (ValueError, FileNotFoundError):
                continue
            per_dataset[key] = metrics
            results["datasets"][key] = metrics
            if group != prev_group:
                print(f"\n  --- {group}-based ---")
                prev_group = group
            print(
                f"  {display:<18s} "
                f"records={metrics['records']:<6d} "
                f"acc={format_metric(metrics.get('acc'))}  "
                f"auc={format_metric(metrics.get('auc'))}  "
                f"ap={format_metric(metrics.get('ap'))}"
            )
        # Group averages
        for grp in ("GAN", "Diffusion"):
            grp_results = {k: m for k, m in per_dataset.items() if any(k == key and g == grp for key, _, g in UNIVERSAL_TEST_ORDER)}
            grp_aucs = [m["auc"] for m in grp_results.values() if m.get("auc") is not None]
            grp_accs = [m["acc"] for m in grp_results.values() if m.get("acc") is not None]
            grp_aps = [m["ap"] for m in grp_results.values() if m.get("ap") is not None]
            if grp_aucs:
                print(f"  {'[' + grp + ' avg]':<18s} acc={sum(grp_accs)/len(grp_accs):.4f}  auc={sum(grp_aucs)/len(grp_aucs):.4f}  ap={sum(grp_aps)/len(grp_aps):.4f}  datasets={len(grp_aucs)}")
        aucs = [m["auc"] for m in per_dataset.values() if m.get("auc") is not None]
        accs = [m["acc"] for m in per_dataset.values() if m.get("acc") is not None]
        aps = [m["ap"] for m in per_dataset.values() if m.get("ap") is not None]
        losses = [m["loss"] for m in per_dataset.values() if m.get("loss") is not None]
        actual_for_compare = {
            "acc": sum(accs) / len(accs) if accs else None,
            "auc": sum(aucs) / len(aucs) if aucs else None,
            "ap": sum(aps) / len(aps) if aps else None,
            "loss": sum(losses) / len(losses) if losses else None,
            "per_dataset": per_dataset,
        }
        results["aggregate"] = actual_for_compare
        print(
            f"  {'[AGGREGATE]':<18s} acc={format_metric(actual_for_compare['acc'])}  "
            f"auc={format_metric(actual_for_compare['auc'])}  ap={format_metric(actual_for_compare['ap'])}  "
            f"datasets={len(per_dataset)}"
        )
        print()
    else:
        for dataset_name in tqdm(datasets, desc="Datasets"):
            metrics = evaluate_dataset(model, config, dataset_name, processor, prototypes, device)
            results["datasets"][dataset_name] = metrics
            if len(datasets) == 1:
                actual_for_compare = metrics
            print(
                f"[{dataset_name}] "
                f"records={metrics['records']} "
                f"acc={format_metric(metrics.get('acc'))} "
                f"auc={format_metric(metrics.get('auc'))} "
                f"ap={format_metric(metrics.get('ap'))} "
                f"video_auc={format_metric(metrics.get('video_auc'))}"
            )

    reference_metrics, reference_label = _checkpoint_reference_metrics(checkpoint, dataset_mode, datasets)
    if actual_for_compare is not None and reference_metrics is not None:
        results["reproduction_delta"] = print_metric_deltas(reference_metrics, actual_for_compare, reference_label)
    elif reference_metrics is not None:
        print("[reproduce] skipped metric delta comparison: request exactly one DeepfakeBench dataset or use Universal aggregate")

    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()
    default_name = "universal_test_results.json" if dataset_mode == "universal" else "deepfakebench_test_results.json"
    output_path = Path(args.output or Path(config.get("output_dir", PROJECT_ROOT / "outputs")) / default_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
