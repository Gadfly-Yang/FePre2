import argparse
import copy
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
except ImportError:
    accuracy_score = average_precision_score = roc_auc_score = None

try:
    from .data import (
        BalancedRealFakeBatchSampler,
        FrameDataset,
        UNIVERSAL_DIFFUSION_PAIRS,
        UNIVERSAL_TEST_ORDER,
        build_clip_transform,
        collate_frames,
        load_deepfake_records,
        load_universal_diffusion_records,
        load_universal_records,
        make_processor,
        split_dataset_names,
    )
    from .losses import (
        GlobalFisherStats,
        compute_fisher_ratio,
        entropy_sparse_fisher_loss,
        fisher_distribution,
        fisher_topk_overlap,
        global_fisher_loss,
        l_sep_loss,
        sparse_fisher_metrics,
        supervised_contrastive_loss,
        symmetric_kl_distribution,
    )
    from .model import ClipMlpClassifier, resolve_visual_model_config
except ImportError:
    from data import (
        BalancedRealFakeBatchSampler,
        FrameDataset,
        UNIVERSAL_DIFFUSION_PAIRS,
        UNIVERSAL_TEST_ORDER,
        build_clip_transform,
        collate_frames,
        load_deepfake_records,
        load_universal_diffusion_records,
        load_universal_records,
        make_processor,
        split_dataset_names,
    )
    from losses import (
        GlobalFisherStats,
        compute_fisher_ratio,
        entropy_sparse_fisher_loss,
        fisher_distribution,
        fisher_topk_overlap,
        global_fisher_loss,
        l_sep_loss,
        sparse_fisher_metrics,
        supervised_contrastive_loss,
        symmetric_kl_distribution,
    )
    from model import ClipMlpClassifier, resolve_visual_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP + MLP with prototype classification and optional LoRA")
    parser.add_argument("--config", default="configs/deepfakebench.yaml", help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    config["config_path"] = str(path)
    return config


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_running_loss():
    return {
        "loss": 0.0,
        "l_contrast": 0.0,
        "l_sep": 0.0,
        "l_gsep": 0.0,
        "l_sparse": 0.0,
        "l_entropy": 0.0,
        "l_ae": 0.0,
        "l_vae_kl": 0.0,
        "l_view_consistency": 0.0,
        "view_topk_overlap": 0.0,
        "view_topk_overlap_count": 0,
        "count": 0,
    }


def _make_running_sparse_metrics():
    return {
        "d_eff": 0.0,
        "gini": 0.0,
        "topk_mass_ratio": 0.0,
        "zero_ratio": 0.0,
        "max_ratio": 0.0,
        "count": 0,
    }


def _average_running_values(running, keys):
    count = max(1, running.get("count", 0))
    return {key: f"{running[key] / count:.4f}" for key in keys}


def _format_metric(value):
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _format_count(value):
    return f"{int(value):,}"


def _print_training_summary(
    *,
    config,
    model,
    device,
    started_at,
    train_records,
    test_records,
    optimizer_name,
    lr,
    weight_decay,
    output_dir,
    weight_dir,
    save_best_auc_weight,
    loss_weights,
    ae_cfg,
    lora_cfg,
    sparse_cfg,
    embedding_key,
    contrast_feature_key,
    vae_bottleneck_contrastive_enabled,
):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_ratio = 100.0 * trainable_params / max(1, total_params)

    active_losses = [(name, weight) for name, weight in loss_weights if float(weight) != 0.0]
    disabled_losses = [name for name, weight in loss_weights if float(weight) == 0.0]

    print("\n" + "=" * 72)
    print("Training run summary")
    print("=" * 72)
    print(f"start_time           : {started_at}")
    print(f"config               : {config.get('config_path')}")
    print(f"device               : {device}")
    print(f"seed                 : {config.get('seed', 42)}")
    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()
    print(f"dataset_mode         : {dataset_mode}")
    if dataset_mode == "universal":
        universal_cfg = config.get("universal", {}) or {}
        print(f"universal.data_root  : {universal_cfg.get('data_root')}")
        print(f"universal.train_categories: {universal_cfg.get('train_categories', 'all')}")
        print(f"universal.test_generators : {universal_cfg.get('test_generators', 'all')}")
        print(f"universal.include_diffusion_test: {universal_cfg.get('include_diffusion_test', True)}")
    else:
        print(f"train/test dataset   : {config.get('train_dataset')} ({config.get('train_split', 'train')}) -> {config.get('test_dataset')} ({config.get('test_split', 'test')})")
        print(f"data_json_folder     : {config.get('data_json_folder')}")
        print(f"base_data_path       : {config.get('base_data_path')}")
    print(f"records              : train={_format_count(len(train_records))}, test={_format_count(len(test_records))}")
    print(f"visual_model        : type={config.get('visual_model_type_resolved', 'clip')}, name={config.get('visual_model_name_resolved', config.get('clip_model', 'openai/clip-vit-large-patch14'))}")
    print(f"feature_pooling     : {config.get('feature_pooling_resolved', 'default')}")
    print(f"clip_model           : {config.get('clip_model', 'openai/clip-vit-large-patch14')}")
    print(f"feature_layer        : {config.get('feature_layer', 'final')}")
    print(f"embedding/hidden dim : {config.get('embedding_dim', 128)} / {config.get('hidden_dim', 512)}")
    print(f"freeze_clip          : {bool(config.get('freeze_clip', True))}")
    print(f"LoRA                 : enabled={bool(lora_cfg.get('enabled', True))}, r={lora_cfg.get('r', 2)}, alpha={lora_cfg.get('alpha', 32)}, dropout={lora_cfg.get('dropout', 0.25)}")
    ae_enabled = bool(ae_cfg.get('enabled', False))
    ae_type = str(ae_cfg.get('type', 'ae')).strip().lower()
    ae_summary = f"enabled={ae_enabled}, type={ae_cfg.get('type', 'ae')}, detach_target={bool(ae_cfg.get('detach_target', True))}"
    if ae_enabled and ae_type == "vae":
        kl_weight = float(ae_cfg.get('kl_weight', 0.0))
        beta = float(ae_cfg.get('beta', 1.0))
        ae_summary += f", kl_weight={kl_weight:g}, beta={beta:g}, effective_kl_weight={kl_weight * beta:g}"
    print(f"AE/VAE               : {ae_summary}")
    contrastive_status = "enabled, feature=vae_latent, disables=l_sep,l_gsep" if vae_bottleneck_contrastive_enabled else "disabled"
    print(f"VAE bottleneck contrastive: {contrastive_status}")
    print(f"optimizer            : {optimizer_name}(lr={lr:g}, weight_decay={weight_decay:g})")
    print(f"feature keys          : fisher/prototype={embedding_key}, contrast={contrast_feature_key}")
    print(f"batch/epochs         : batch_size={config.get('batch_size')}, eval_batch_size={config.get('eval_batch_size', config.get('batch_size'))}, epochs={config.get('epochs')}")
    print(f"parameters           : total={_format_count(total_params)}, trainable={_format_count(trainable_params)} ({trainable_ratio:.2f}%), frozen={_format_count(frozen_params)}")
    print("loss weights         :")
    for name, weight in loss_weights:
        status = "on" if float(weight) != 0.0 else "off"
        print(f"  - {name:<12} weight={float(weight):g} [{status}]")
    if disabled_losses:
        print(f"disabled losses      : {', '.join(disabled_losses)}")
    print(f"sparse_fisher        : enabled={bool(sparse_cfg.get('enabled', False))}, entropy_weight={float(sparse_cfg.get('entropy_weight', 0.0)):g}, temperature={float(sparse_cfg.get('entropy_temperature', 0.5)):g}, topk={int(sparse_cfg.get('topk', 16))}")
    print(f"output_dir           : {output_dir}")
    print(f"save_best_auc_weight : {save_best_auc_weight}")
    if save_best_auc_weight:
        print(f"weight_dir           : {weight_dir}")
    print("=" * 72 + "\n")


def compute_ae_reconstruction_loss(outputs, enabled=False, detach_target=True):
    if not enabled:
        return outputs["embedding"].new_tensor(0.0) if "embedding" in outputs else outputs["clip_features"].new_tensor(0.0)
    reconstruction = outputs.get("reconstruction")
    if reconstruction is None:
        raise ValueError("AE is enabled but model outputs do not contain reconstruction")
    target = outputs["clip_features"]
    if detach_target:
        target = target.detach()
    return F.mse_loss(reconstruction, target)


def compute_vae_kl_loss(outputs, enabled=False):
    if not enabled:
        return outputs["embedding"].new_tensor(0.0) if "embedding" in outputs else outputs["clip_features"].new_tensor(0.0)
    mu = outputs.get("vae_mu")
    logvar = outputs.get("vae_logvar")
    if mu is None or logvar is None:
        raise ValueError("VAE is enabled but model outputs do not contain vae_mu/vae_logvar")
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def make_loader(config, split, train, processor, include_label_name=False, include_video_name=False, paired_transform=None, dataset_names=None):
    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()

    if dataset_mode == "universal":
        return _make_universal_loader(
            config, train=train, processor=processor,
            include_label_name=include_label_name, include_video_name=include_video_name,
            paired_transform=paired_transform, dataset_names=dataset_names,
        )

    # ---- DeepfakeBench (face) path ----
    dataset_key = "train_dataset" if train else "test_dataset"
    frame_num_key = "train_frame_num" if train else "test_frame_num"
    records = load_deepfake_records(
        dataset_json_folder=config["data_json_folder"],
        datasets=split_dataset_names(config[dataset_key]),
        split=split,
        compression=config.get("compression", "c23"),
        frame_num=config.get(frame_num_key, config.get("frame_num")),
        base_data_path=config.get("base_data_path"),
        label_dict=config.get("label_dict"),
        identity_regex=config.get("identity_regex"),
        fake_identity_role=config.get("fake_identity_role", "source"),
        is_training=train,
    )
    transform = build_clip_transform(
        processor,
        train=train,
        image_size=config.get("image_size"),
        augmentation_prob=config.get("augmentation_prob", 0.1),
    )
    strict_files = bool(config.get("strict_existing_files", False))

    dataset = FrameDataset(
        records,
        transform=transform,
        strict_existing_files=strict_files,
        include_label_name=include_label_name,
        include_video_name=include_video_name,
        paired_transform=paired_transform,
    )
    batch_size = int(config.get("batch_size", 16)) if train else int(config.get("eval_batch_size", config.get("batch_size", 16) * 2))
    drop_last = bool(config.get("drop_last", False)) if train else False
    use_balanced_sampler = bool(config.get("balanced_real_fake_batches", True)) and train

    loader_kwargs = {
        "num_workers": int(config.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_frames,
    }
    if use_balanced_sampler:
        loader_kwargs["batch_sampler"] = BalancedRealFakeBatchSampler(
            records,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=int(config.get("seed", 42)),
        )
    else:
        loader_kwargs.update(
            {
                "batch_size": batch_size,
                "shuffle": train,
                "drop_last": drop_last,
            }
        )

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, records


def _make_universal_loader(config, train, processor, include_label_name=False, include_video_name=False, paired_transform=None, dataset_names=None):
    """Build a DataLoader for UniversalFakeDetect directory-based data.

    When *dataset_names* is given (e.g. ``"progan"`` or ``"guided"``) it
    overrides the config-driven category list — useful for per-dataset
    evaluation in test scripts.
    """
    universal_cfg = config.get("universal", {}) or {}
    data_root = universal_cfg.get("data_root")
    if not data_root:
        raise ValueError("universal.data_root must be set in the config for UniversalFakeDetect mode")

    if train:
        categories = universal_cfg.get("train_categories", "all")
        exclude = list(universal_cfg.get("train_exclude", []))
        records = load_universal_records(
            data_root, split="train", categories=categories,
            exclude=exclude,
        )
    else:
        if dataset_names is not None:
            # Per-dataset evaluation – try generator sets first, then diffusion keys
            names = split_dataset_names(dataset_names)
            records = load_universal_records(
                data_root, split="test", categories=",".join(names),
                exclude=None,
            )
            diffusion_keys = {key for _, _, key in UNIVERSAL_DIFFUSION_PAIRS}
            diffusion_hits = [n for n in names if n in diffusion_keys]
            if diffusion_hits:
                diffusion_records = load_universal_diffusion_records(data_root)
                diffusion_records = [r for r in diffusion_records if r.dataset_name in set(diffusion_hits)]
                records.extend(diffusion_records)
        else:
            # Full test set
            test_generators = universal_cfg.get("test_generators", "all")
            records = load_universal_records(
                data_root, split="test", categories=test_generators,
                exclude=None,
            )
            if universal_cfg.get("include_diffusion_test", True):
                diffusion_records = load_universal_diffusion_records(data_root)
                records.extend(diffusion_records)

    # ---- Image transforms: Effort reference mode vs default CLIPMLPa mode ----
    effort_mode = bool(universal_cfg.get("reference_data_pipeline", False))
    visual_type = str(config.get("visual_model_type_resolved") or resolve_visual_model_config(config)["type"]).strip().lower()
    if effort_mode:
        transform = build_clip_transform(
            processor,
            train=train,
            load_size=int(universal_cfg.get("load_size", 256)),
            crop_size=int(universal_cfg.get("crop_size", 224)),
            reference_normalization=("clip" if visual_type == "clip" else "processor"),
        )
    else:
        transform = build_clip_transform(
            processor,
            train=train,
            image_size=config.get("image_size"),
            augmentation_prob=config.get("augmentation_prob", 0.1),
        )
    strict_files = bool(config.get("strict_existing_files", False))

    dataset = FrameDataset(
        records,
        transform=transform,
        strict_existing_files=strict_files,
        include_label_name=include_label_name,
        include_video_name=include_video_name,
        paired_transform=paired_transform,
    )
    batch_size = int(config.get("batch_size", 16)) if train else int(config.get("eval_batch_size", config.get("batch_size", 16) * 2))
    drop_last = bool(config.get("drop_last", False)) if train else False
    # Effort reference mode: plain shuffle; CLIPMLPa default: class-balanced sampling
    if effort_mode:
        use_balanced_sampler = False
    else:
        use_balanced_sampler = bool(config.get("balanced_real_fake_batches", True)) and train

    loader_kwargs = {
        "num_workers": int(config.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_frames,
    }
    if use_balanced_sampler:
        loader_kwargs["batch_sampler"] = BalancedRealFakeBatchSampler(
            records,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=int(config.get("seed", 42)),
        )
    else:
        loader_kwargs.update(
            {
                "batch_size": batch_size,
                "shuffle": train,
                "drop_last": drop_last,
            }
        )

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, records


# ---------------------------------------------------------------------------
#  Prototype helpers
# ---------------------------------------------------------------------------

def encode_string_groups(values, device):
    mapping = {}
    ids = []
    for value in values:
        value = str(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        ids.append(mapping[value])
    return torch.tensor(ids, dtype=torch.long, device=device), mapping


def select_embedding(outputs, key="embedding"):
    if key not in outputs:
        available_keys = ", ".join(sorted(outputs.keys()))
        raise ValueError(
            f"Model outputs do not contain requested embedding key: {key}. "
            f"Available keys: {available_keys}"
        )
    return outputs[key]


@torch.no_grad()
def classify_with_prototypes(embeddings, prototypes):
    """Return [N, 2] logits via cosine similarity to class prototypes."""
    emb = F.normalize(embeddings.float(), dim=1)
    proto_real = F.normalize(prototypes[0].float(), dim=0)
    proto_fake = F.normalize(prototypes[1].float(), dim=0)
    sim_real = emb @ proto_real
    sim_fake = emb @ proto_fake
    return torch.stack([sim_real, sim_fake], dim=1)


@torch.no_grad()
def update_prototypes(prototypes, embeddings, labels, momentum):
    """EMA-update prototypes dict in-place.  prototypes[cls] is a 1-D tensor."""
    for cls in (0, 1):
        mask = labels.eq(cls)
        if mask.any():
            batch_proto = embeddings[mask].float().mean(dim=0)
            if prototypes[cls] is None:
                prototypes[cls] = batch_proto.clone()
            else:
                prototypes[cls] = momentum * prototypes[cls] + (1.0 - momentum) * batch_proto


def compute_metrics(labels, probs):
    """Return dict with ACC, AUC, AP given binary labels and predicted probabilities."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {}
    if accuracy_score is not None:
        metrics["acc"] = float(accuracy_score(labels, preds))
    else:
        metrics["acc"] = float((preds == labels).mean())
    if roc_auc_score is not None and len(np.unique(labels)) == 2:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["auc"] = None
    if average_precision_score is not None and len(np.unique(labels)) == 2:
        metrics["ap"] = float(average_precision_score(labels, probs))
    else:
        metrics["ap"] = None
    return metrics


def compute_video_auc(labels, probs, video_names):
    """Return video-level AUC by averaging frame probabilities per video."""
    if not video_names or roc_auc_score is None:
        return None

    video_probs = {}
    video_labels = {}
    for label, prob, video_name in zip(labels, probs, video_names):
        key = str(video_name)
        video_probs.setdefault(key, []).append(float(prob))
        video_labels.setdefault(key, int(label))

    labels_video = np.asarray([video_labels[key] for key in video_probs])
    probs_video = np.asarray([np.mean(video_probs[key]) for key in video_probs])
    if len(np.unique(labels_video)) != 2:
        return None
    return float(roc_auc_score(labels_video, probs_video))


@torch.no_grad()
def evaluate(model, loader, prototypes, device, embedding_key="embedding"):
    """Run the test loader once and return metrics dict + average loss."""
    if prototypes[0] is None or prototypes[1] is None:
        return {"acc": None, "auc": None, "ap": None, "loss": None, "video_auc": None}

    was_training = model.training
    model.eval()
    try:
        all_labels = []
        all_probs = []
        all_video_names = []
        total_loss = 0.0
        total_count = 0

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)

            outputs = model(images)
            logits = classify_with_prototypes(select_embedding(outputs, embedding_key), prototypes)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss = F.cross_entropy(logits, labels)

            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            if "video_name" in batch:
                all_video_names.extend(batch["video_name"])

        metrics = compute_metrics(all_labels, all_probs)
        metrics["video_auc"] = compute_video_auc(all_labels, all_probs, all_video_names)
        metrics["loss"] = total_loss / max(1, total_count)
        return metrics
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
#  t-SNE  (unchanged from previous version)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_embeddings(model, loader, device, max_samples=None, config=None):
    was_training = model.training
    model.eval()
    try:
        embeddings = []
        labels = []
        paths = []
        identities = []
        total = 0
        sample_limit = None if max_samples is None else int(max_samples)

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            batch_embeddings = select_embedding(outputs, str(config.get("embedding_key", "embedding"))).detach().cpu()
            batch_labels = batch["label"].detach().cpu()

            if sample_limit is not None:
                remaining = sample_limit - total
                if remaining <= 0:
                    break
                batch_embeddings = batch_embeddings[:remaining]
                batch_labels = batch_labels[:remaining]
                batch_paths = batch["path"][:remaining]
                batch_identities = batch["identity"][:remaining]
            else:
                batch_paths = batch["path"]
                batch_identities = batch["identity"]

            embeddings.append(batch_embeddings)
            labels.append(batch_labels)
            paths.extend(batch_paths)
            identities.extend(batch_identities)
            total += int(batch_labels.numel())

        if not embeddings:
            return None

        return {
            "embeddings": torch.cat(embeddings, dim=0).numpy(),
            "labels": torch.cat(labels, dim=0).numpy(),
            "paths": np.asarray(paths),
            "identities": np.asarray(identities),
        }
    finally:
        model.train(was_training)


def save_tsne_visualization(model, loader, device, output_dir, epoch, config):
    if TSNE is None:
        return None
    if plt is None:
        return None

    max_samples = config.get("tsne_max_samples")
    collected = collect_embeddings(model, loader, device, max_samples=max_samples, config=config)
    if collected is None:
        return None

    embeddings = collected["embeddings"]
    labels = collected["labels"]
    if embeddings.shape[0] < 3:
        return None

    requested_perplexity = float(config.get("tsne_perplexity", 30.0))
    perplexity = min(requested_perplexity, max(1.0, (embeddings.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(config.get("seed", 42)),
    )
    coords = tsne.fit_transform(embeddings)

    stem = f"tsne_epoch_{epoch:03d}"
    npz_path = output_dir / f"{stem}.npz"
    png_path = output_dir / f"{stem}.png"
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        tsne=coords,
        labels=labels,
        paths=collected["paths"],
        identities=collected["identities"],
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    names = {0: "real", 1: "fake"}
    colors = {0: "#2b8cbe", 1: "#d95f02"}
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.72,
            c=colors.get(int(label), "#666666"),
            label=names.get(int(label), str(label)),
            linewidths=0,
        )
    ax.set_title(f"Embedding t-SNE epoch {epoch}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    return {
        "png": str(png_path),
        "npz": str(npz_path),
        "samples": int(embeddings.shape[0]),
        "perplexity": float(perplexity),
    }


def _serialize_prototypes(prototypes):
    """Convert prototype tensors to a JSON-serializable dict of lists."""
    return {str(k): v.detach().cpu().tolist() for k, v in prototypes.items() if v is not None}


def _prototypes_ready(prototypes):
    """Return True if both real (0) and fake (1) prototypes are non-None tensors."""
    return (prototypes is not None
            and prototypes.get(0) is not None
            and prototypes.get(1) is not None)


def save_checkpoint(path, model, optimizer, epoch, metrics, config, prototypes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    if _prototypes_ready(prototypes):
        save_dict["prototypes"] = _serialize_prototypes(prototypes)
    torch.save(save_dict, path)


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(config):
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_seed(config.get("seed", 42))
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- optional component config ----
    lora_cfg = config.get("lora", {}) or {}
    ae_cfg = config.get("ae", {}) or {}
    paired_view_cfg = config.get("paired_view_fisher_consistency", {}) or {}

    paired_view_active = bool(paired_view_cfg.get("enabled", False)) and float(paired_view_cfg.get("loss_weight", 0.0)) != 0.0
    visual_cfg = resolve_visual_model_config(config)
    config["visual_model_type_resolved"] = visual_cfg["type"]
    config["visual_model_name_resolved"] = visual_cfg["name"]
    config["feature_pooling_resolved"] = visual_cfg["feature_pooling"]

    processor = make_processor(
        visual_cfg["name"],
        model_type=visual_cfg["type"],
        cache_dir=config.get("cache_dir"),
        local_files_only=bool(config.get("local_files_only", False)),
    )
    paired_transform = None
    if paired_view_active:
        # In Effort reference mode, the second view uses the same reference transform
        dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()
        universal_cfg = config.get("universal", {}) or {}
        if dataset_mode == "universal" and bool(universal_cfg.get("reference_data_pipeline", False)):
            paired_transform = build_clip_transform(
                processor,
                train=True,
                load_size=int(universal_cfg.get("load_size", 256)),
                crop_size=int(universal_cfg.get("crop_size", 224)),
                reference_normalization=("clip" if visual_cfg["type"] == "clip" else "processor"),
            )
        else:
            paired_transform = build_clip_transform(
                processor,
                train=True,
                image_size=config.get("image_size"),
                augmentation_prob=float(paired_view_cfg.get("second_view_augmentation_prob", 1.0)),
            )
    train_loader, train_records = make_loader(
        config,
        split=config.get("train_split", "train"),
        train=True,
        processor=processor,
        include_label_name=False,
        paired_transform=paired_transform,
    )
    test_loader, test_records = make_loader(
        config,
        split=config.get("test_split", "test"),
        train=False,
        processor=processor,
        include_video_name=True,
    )

    # ---- per-dataset test loaders for universal mode ----
    _universal_test_loaders = None
    _universal_test_order = None  # ordered list of (internal_key, display_name, group)
    dataset_mode = str(config.get("dataset_mode", "deepfakebench")).strip().lower()
    if dataset_mode == "universal":
        universal_cfg = config.get("universal", {}) or {}
        data_root = universal_cfg.get("data_root")
    if not data_root:
        raise ValueError("universal.data_root must be set in the config for UniversalFakeDetect mode")

        # Build C2P-CLIP ordered test loader map
        _universal_test_loaders = {}
        _universal_test_order = []
        eval_workers = min(2, int(config.get("num_workers", 4)))

        for key, display, group in UNIVERSAL_TEST_ORDER:
            try:
                ds_loader, _ = make_loader(
                    config,
                    split="test",
                    train=False,
                    processor=processor,
                    include_video_name=True,
                    dataset_names=key,
                )
            except (ValueError, FileNotFoundError):
                continue  # skip datasets not available
            ds_loader.num_workers = eval_workers
            _universal_test_loaders[key] = ds_loader
            _universal_test_order.append((key, display, group))

    # ---- model (with optional LoRA) ----
    ae_enabled = bool(ae_cfg.get("enabled", False))
    ae_type = str(ae_cfg.get("type", "ae")).strip().lower()
    if ae_type not in {"ae", "vae"}:
        raise ValueError(f"ae.type must be 'ae' or 'vae', got {ae_type}")
    ae_recon_weight = float(ae_cfg.get("recon_weight", 0.0))
    vae_kl_weight = float(ae_cfg.get("kl_weight", 0.0))
    ae_beta = float(ae_cfg.get("beta", 1.0))
    if ae_beta < 0.0:
        raise ValueError(f"ae.beta must be >= 0, got {ae_beta}")
    ae_detach_target = bool(ae_cfg.get("detach_target", True))
    vae_enabled = ae_enabled and ae_type == "vae"
    vae_kl_effective_weight = vae_kl_weight * ae_beta if vae_enabled else 0.0
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
        use_ae=ae_enabled,
        ae_type=ae_type,
    ).to(device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    output_dir = Path(config.get("output_dir", "outputs/clipmlp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_best_auc_weight = bool(config.get("save_best_auc_weight", True))
    weight_dir = Path(config.get("weight_dir", output_dir / "weight"))
    if save_best_auc_weight:
        weight_dir.mkdir(parents=True, exist_ok=True)
    weight_name = str(config.get("weight_name", "best_auc.pt"))
    best_auc_path = weight_dir / weight_name
    best_auc = float("-inf")

    def save_best_auc_model(metrics, epoch, step, final=False, metric_key="auc"):
        nonlocal best_auc
        if not save_best_auc_weight:
            return
        value = metrics.get(metric_key)
        if value is None or float(value) <= best_auc:
            return

        best_auc = float(value)
        save_dict = {
            "epoch": epoch,
            "step": step,
            "final": final,
            "best_auc": best_auc,
            "metrics": metrics,
            "model_state": model.state_dict(),
            "config": config,
        }
        if _prototypes_ready(prototypes):
            save_dict["prototypes"] = _serialize_prototypes(prototypes)
        torch.save(save_dict, best_auc_path)


    l_contrast_weight = float(config.get("l_contrast_weight", 0.0))
    l_contrast_temperature = float(config.get("l_contrast_temperature", 0.07))
    contrast_feature_key = str(config.get("contrast_feature_key", "clip_features"))
    embedding_key = str(config.get("embedding_key", "embedding"))
    epochs = int(config.get("epochs", 10))
    l_sep_weight = float(config.get("l_sep_weight", 1.0))
    l_sep_gamma = float(config.get("l_sep_gamma", 1.0))
    l_sep_eps = float(config.get("l_sep_eps", 1e-8))
    l_global_sep_weight = float(config.get("l_global_sep_weight", 0.0))
    l_global_sep_enabled = bool(config.get("l_global_sep_enabled", l_global_sep_weight > 0.0))
    l_global_sep_gamma = float(config.get("l_global_sep_gamma", l_sep_gamma))
    l_global_sep_eps = float(config.get("l_global_sep_eps", l_sep_eps))

    vae_bottleneck_cfg = config.get("vae_bottleneck_contrastive", {}) or {}
    if isinstance(vae_bottleneck_cfg, bool):
        vae_bottleneck_contrastive_enabled = vae_bottleneck_cfg
        vae_bottleneck_loss_weight = 1.0
        vae_bottleneck_temperature = 0.07
    else:
        vae_bottleneck_contrastive_enabled = bool(vae_bottleneck_cfg.get("enabled", False))
        vae_bottleneck_loss_weight = float(vae_bottleneck_cfg.get("loss_weight", 1.0))
        vae_bottleneck_temperature = float(vae_bottleneck_cfg.get("temperature", 0.07))

    if vae_bottleneck_contrastive_enabled:
        if not ae_enabled:
            raise ValueError("vae_bottleneck_contrastive.enabled=True requires ae.enabled=True")
        if not vae_enabled:
            raise ValueError("vae_bottleneck_contrastive.enabled=True requires ae.type='vae'")
        if vae_bottleneck_loss_weight <= 0.0:
            raise ValueError("vae_bottleneck_contrastive.enabled=True requires loss_weight > 0")
        l_contrast_weight = vae_bottleneck_loss_weight
        l_contrast_temperature = vae_bottleneck_temperature
        contrast_feature_key = "vae_latent"
        l_sep_weight = 0.0
        l_global_sep_weight = 0.0
        l_global_sep_enabled = False

    proto_momentum = float(config.get("proto_momentum", 0.99))
    l_global_sep_momentum = float(config.get("l_global_sep_momentum", proto_momentum))
    log_interval = int(config.get("log_interval", 20))
    test_interval = int(config.get("test_interval", 400))
    tsne_interval = int(config.get("tsne_interval", 1))

    sparse_cfg = config.get("sparse_fisher", {}) or {}
    sparse_enabled = bool(sparse_cfg.get("enabled", False))
    sparse_eps = float(sparse_cfg.get("eps", l_sep_eps))
    sparse_topk = int(sparse_cfg.get("topk", 16))
    entropy_weight = float(sparse_cfg.get("entropy_weight", 0.0))
    entropy_temperature = float(sparse_cfg.get("entropy_temperature", 0.5))
    sparse_active = sparse_enabled and entropy_weight > 0.0

    l_global_sep_active = l_global_sep_enabled and l_global_sep_weight > 0.0
    loss_weights = [
        ("l_contrast", l_contrast_weight),
        ("l_sep", l_sep_weight),
        ("l_gsep", l_global_sep_weight if l_global_sep_active else 0.0),
        ("l_sparse", entropy_weight if sparse_active else 0.0),
        ("l_ae", ae_recon_weight if ae_enabled else 0.0),
        ("l_vae_kl", vae_kl_effective_weight),
        ("l_view_consistency", float(paired_view_cfg.get("loss_weight", 0.0)) if paired_view_active else 0.0),
    ]
    _print_training_summary(
        config=config,
        model=model,
        device=device,
        started_at=started_at,
        train_records=train_records,
        test_records=test_records,
        optimizer_name="AdamW",
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        output_dir=output_dir,
        weight_dir=weight_dir,
        save_best_auc_weight=save_best_auc_weight,
        loss_weights=loss_weights,
        ae_cfg=ae_cfg,
        lora_cfg=lora_cfg,
        sparse_cfg=sparse_cfg,
        embedding_key=embedding_key,
        contrast_feature_key=contrast_feature_key,
        vae_bottleneck_contrastive_enabled=vae_bottleneck_contrastive_enabled,
    )

    # ---- global prototypes (EMA over training batches) ----
    prototypes = {0: None, 1: None}  # each will be a 1-D tensor on the same device
    global_fisher_stats = GlobalFisherStats(l_global_sep_momentum) if l_global_sep_active else None

    # ---- accumulators for training metrics ----
    train_labels_acc = []
    train_probs_acc = []
    global_step = 0
    history = []
    total_train_steps = epochs * max(1, len(train_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = _make_running_loss()
        running_sparse_metrics = _make_running_sparse_metrics()
        for batch in train_loader:
            global_step += 1

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)

            # ---- forward ----
            outputs = model(images, detach_ae_input=ae_detach_target)
            outputs_view2 = None
            embeddings = select_embedding(outputs, embedding_key)
            contrast_features = select_embedding(outputs, contrast_feature_key)

            l_contrast = supervised_contrastive_loss(contrast_features, labels, temperature=l_contrast_temperature)
            l_sep = embeddings.new_tensor(0.0)
            if l_sep_weight > 0.0:
                l_sep = l_sep_loss(embeddings, labels, gamma=l_sep_gamma, eps=l_sep_eps)
            l_gsep = embeddings.new_tensor(0.0)
            if global_fisher_stats is not None:
                global_fisher_stats.update(embeddings.detach(), labels)
                l_gsep = global_fisher_loss(
                    embeddings,
                    labels,
                    global_fisher_stats,
                    gamma=l_global_sep_gamma,
                    eps=l_global_sep_eps,
                )
            fisher_ratio = None
            l_sparse = embeddings.new_tensor(0.0)
            l_entropy = embeddings.new_tensor(0.0)
            if sparse_active:
                fisher_ratio = compute_fisher_ratio(embeddings, labels, eps=sparse_eps)
                l_entropy = entropy_sparse_fisher_loss(fisher_ratio, temperature=entropy_temperature, eps=sparse_eps)
                l_sparse = entropy_weight * l_entropy

                sparse_metrics = sparse_fisher_metrics(fisher_ratio.detach(), topk=sparse_topk, eps=sparse_eps)
                for key, value in sparse_metrics.items():
                    running_sparse_metrics[key] += float(value) * int(labels.numel())
                running_sparse_metrics["count"] += int(labels.numel())

            l_view_consistency = embeddings.new_tensor(0.0)
            if paired_view_active:
                if "image_view2" not in batch:
                    raise ValueError("paired_view_fisher_consistency is enabled but batch does not contain image_view2")
                outputs_view2 = model(batch["image_view2"].to(device, non_blocking=True), detach_ae_input=ae_detach_target)
                embeddings_view2 = select_embedding(outputs_view2, embedding_key)
                ratio_view1 = compute_fisher_ratio(embeddings, labels, eps=float(paired_view_cfg.get("eps", sparse_eps)))
                ratio_view2 = compute_fisher_ratio(embeddings_view2, labels, eps=float(paired_view_cfg.get("eps", sparse_eps)))
                p1 = fisher_distribution(ratio_view1, temperature=float(paired_view_cfg.get("temperature", 0.5)), eps=float(paired_view_cfg.get("eps", 1e-8)))
                p2 = fisher_distribution(ratio_view2, temperature=float(paired_view_cfg.get("temperature", 0.5)), eps=float(paired_view_cfg.get("eps", 1e-8)))
                l_view_raw = symmetric_kl_distribution(p1, p2, eps=float(paired_view_cfg.get("eps", 1e-8)))
                feature_weight = float(paired_view_cfg.get("feature_consistency_weight", 0.0))
                if feature_weight != 0.0:
                    l_view_raw = l_view_raw + feature_weight * F.mse_loss(F.normalize(embeddings, dim=1), F.normalize(embeddings_view2, dim=1))
                l_view_consistency = float(paired_view_cfg.get("loss_weight", 0.0)) * l_view_raw
                overlap = fisher_topk_overlap(ratio_view1, ratio_view2, topk=int(paired_view_cfg.get("topk", 16)))
                running_loss["view_topk_overlap"] += overlap * int(labels.numel())
                running_loss["view_topk_overlap_count"] += int(labels.numel())

            l_ae = compute_ae_reconstruction_loss(outputs, enabled=ae_enabled, detach_target=ae_detach_target)
            l_vae_kl = compute_vae_kl_loss(outputs, enabled=vae_enabled)
            loss = (
                l_contrast_weight * l_contrast
                + l_sep_weight * l_sep
                + l_global_sep_weight * l_gsep
                + l_sparse
                + ae_recon_weight * l_ae
                + vae_kl_effective_weight * l_vae_kl
                + l_view_consistency
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ---- running loss stats ----
            bs = int(labels.numel())
            running_loss["loss"] += float(loss.item()) * bs
            running_loss["l_contrast"] += float(l_contrast.item()) * bs
            running_loss["l_sep"] += float(l_sep.item()) * bs
            running_loss["l_gsep"] += float(l_gsep.item()) * bs
            running_loss["l_sparse"] += float(l_sparse.item()) * bs
            running_loss["l_entropy"] += float(l_entropy.item()) * bs
            running_loss["l_ae"] += float(l_ae.item()) * bs
            running_loss["l_vae_kl"] += float(l_vae_kl.item()) * bs
            running_loss["l_view_consistency"] += float(l_view_consistency.item()) * bs
            running_loss["count"] += bs

            # ---- update EMA prototypes ----
            update_prototypes(prototypes, embeddings.detach(), labels, proto_momentum)

            # ---- accumulate training predictions (for 100-batch report) ----
            if prototypes[0] is not None and prototypes[1] is not None:
                logits_train = classify_with_prototypes(embeddings, prototypes)
                probs_train = torch.softmax(logits_train, dim=1)[:, 1]
                train_labels_acc.extend(labels.cpu().tolist())
                train_probs_acc.extend(probs_train.cpu().tolist())

            # =================================================================
            #  100-batch training-set metrics
            # =================================================================
            if global_step % log_interval == 0:
                if train_labels_acc:
                    train_metrics = compute_metrics(train_labels_acc, train_probs_acc)
                    train_loss_keys = [
                        "loss",
                        "l_contrast",
                        "l_sep",
                        "l_gsep",
                        "l_sparse",
                        "l_entropy",
                    ]
                    if ae_enabled:
                        train_loss_keys.append("l_ae")
                    if vae_enabled:
                        train_loss_keys.append("l_vae_kl")
                    if paired_view_active:
                        train_loss_keys.append("l_view_consistency")
                    train_loss_str = _average_running_values(running_loss, train_loss_keys)
                    if paired_view_active and running_loss.get("view_topk_overlap_count", 0) > 0:
                        train_loss_str["view_topk_overlap"] = f"{running_loss['view_topk_overlap'] / running_loss['view_topk_overlap_count']:.4f}"
                    sparse_metric_str = _average_running_values(
                        running_sparse_metrics,
                        ("d_eff", "gini", "topk_mass_ratio", "zero_ratio", "max_ratio"),
                    ) if sparse_active and running_sparse_metrics["count"] > 0 else {}
                    print(
                        f"[train] epoch={epoch} step={global_step}  mode={'train' if model.training else 'eval'}  "
                        + "  ".join(f"{k}={v}" for k, v in train_loss_str.items())
                        + ("  |  " + "  ".join(f"sf_{k}={v}" for k, v in sparse_metric_str.items()) if sparse_metric_str else "")
                        + f"  |  acc={train_metrics['acc']:.4f}  auc={_format_metric(train_metrics['auc'])}  ap={_format_metric(train_metrics['ap'])}"
                    )
                else:
                    print(f"[train] epoch={epoch} step={global_step}  prototypes not ready yet")
                # reset accumulators
                train_labels_acc = []
                train_probs_acc = []
                running_loss = _make_running_loss()
                running_sparse_metrics = _make_running_sparse_metrics()

            # =================================================================
            #  400-batch test evaluation
            # =================================================================
            if global_step % test_interval == 0:
                pre_eval_training = model.training
                if _universal_test_loaders is not None and _universal_test_order is not None:
                    # ---- universal mode: C2P-CLIP style per-dataset evaluation ----
                    per_dataset = {}
                    all_labels_agg = []
                    all_probs_agg = []
                    total_loss_agg = 0.0
                    total_count_agg = 0
                    for key, display, group in _universal_test_order:
                        ds_loader = _universal_test_loaders.get(key)
                        if ds_loader is None:
                            continue
                        ds_metrics = evaluate(model, ds_loader, prototypes, device, embedding_key=embedding_key)
                        per_dataset[key] = ds_metrics

                    # Aggregate
                    aucs = [m["auc"] for m in per_dataset.values() if m.get("auc") is not None]
                    agg_auc = sum(aucs) / len(aucs) if aucs else None
                    accs = [m["acc"] for m in per_dataset.values() if m.get("acc") is not None]
                    agg_acc = sum(accs) / len(accs) if accs else None
                    losses = [m["loss"] for m in per_dataset.values() if m.get("loss") is not None]
                    agg_loss = sum(losses) / len(losses) if losses else None
                    aps = [m["ap"] for m in per_dataset.values() if m.get("ap") is not None]
                    agg_ap = sum(aps) / len(aps) if aps else None

                    # C2P-CLIP grouped output
                    prev_group = None
                    print(f"[test]  epoch={epoch} step={global_step}  === C2P-CLIP evaluation ({len(per_dataset)} datasets) ===")
                    for key, display, group in _universal_test_order:
                        m = per_dataset.get(key)
                        if m is None:
                            continue
                        if group != prev_group:
                            print(f"  --- {group}-based ---")
                            prev_group = group
                        print(
                            f"  {display:<18s} "
                            f"acc={m['acc']:.4f}  auc={_format_metric(m['auc'])}  "
                            f"ap={_format_metric(m['ap'])}"
                        )
                    # Group averages
                    for grp in ("GAN", "Diffusion"):
                        grp_keys = [k for k, _, g in _universal_test_order if g == grp and k in per_dataset]
                        grp_aucs = [per_dataset[k]["auc"] for k in grp_keys if per_dataset[k].get("auc") is not None]
                        grp_accs = [per_dataset[k]["acc"] for k in grp_keys if per_dataset[k].get("acc") is not None]
                        grp_aps = [per_dataset[k]["ap"] for k in grp_keys if per_dataset[k].get("ap") is not None]
                        if grp_aucs:
                            print(f"  {'[' + grp + ' avg]':<18s} acc={sum(grp_accs)/len(grp_accs):.4f}  auc={sum(grp_aucs)/len(grp_aucs):.4f}  ap={sum(grp_aps)/len(grp_aps):.4f}  datasets={len(grp_aucs)}")
                    print(
                        f"[test]  epoch={epoch} step={global_step}  AGGREGATE  "
                        f"loss={_format_metric(agg_loss)}  "
                        f"acc={agg_acc:.4f}  auc={_format_metric(agg_auc)}  "
                        f"ap={_format_metric(agg_ap)}  "
                        f"datasets={len(per_dataset)}"
                    )
                    agg_metrics = {"acc": agg_acc, "auc": agg_auc, "ap": agg_ap, "loss": agg_loss,
                                   "per_dataset": per_dataset}
                    history.append({"step": global_step, "epoch": epoch, "test": agg_metrics})
                    save_best_auc_model(agg_metrics, epoch, global_step, metric_key="ap")
                else:
                    test_metrics = evaluate(model, test_loader, prototypes, device, embedding_key=embedding_key)
                    print(
                        f"[test]  epoch={epoch} step={global_step}  "
                        f"loss={_format_metric(test_metrics['loss'])}  "
                        f"acc={test_metrics['acc']:.4f}  auc={_format_metric(test_metrics['auc'])}  "
                        f"video_auc={_format_metric(test_metrics.get('video_auc'))}  ap={_format_metric(test_metrics['ap'])}"
                    )
                    history.append({"step": global_step, "epoch": epoch, "test": test_metrics})
                    save_best_auc_model(test_metrics, epoch, global_step)
                assert model.training == pre_eval_training, (
                    f"model.training changed during evaluation at epoch={epoch}, step={global_step}: "
                    f"{pre_eval_training} -> {model.training}"
                )

        # ---- end of epoch t-SNE ----
        tsne_info = None
        if tsne_interval > 0 and (epoch % tsne_interval == 0 or epoch == epochs):
            tsne_info = save_tsne_visualization(model, test_loader, device, output_dir, epoch, config)
            if tsne_info:
                pass

        # checkpoint
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, {"epoch": epoch, "tsne": tsne_info}, config, prototypes=prototypes)

    # ---- final test after training ----
    if _universal_test_loaders is not None and _universal_test_order is not None:
        per_dataset = {}
        for key, display, group in _universal_test_order:
            ds_loader = _universal_test_loaders.get(key)
            if ds_loader is None:
                continue
            per_dataset[key] = evaluate(model, ds_loader, prototypes, device, embedding_key=embedding_key)

        aucs = [m["auc"] for m in per_dataset.values() if m.get("auc") is not None]
        agg_auc = sum(aucs) / len(aucs) if aucs else None
        accs = [m["acc"] for m in per_dataset.values() if m.get("acc") is not None]
        agg_acc = sum(accs) / len(accs) if accs else None
        aps = [m["ap"] for m in per_dataset.values() if m.get("ap") is not None]
        agg_ap = sum(aps) / len(aps) if aps else None
        final_metrics = {"acc": agg_acc, "auc": agg_auc, "ap": agg_ap, "per_dataset": per_dataset}

        prev_group = None
        print(f"\n[final] === C2P-CLIP evaluation ({len(per_dataset)} datasets) ===")
        for key, display, group in _universal_test_order:
            m = per_dataset.get(key)
            if m is None:
                continue
            if group != prev_group:
                print(f"  --- {group}-based ---")
                prev_group = group
            print(
                f"  {display:<18s} "
                f"acc={m['acc']:.4f}  auc={_format_metric(m['auc'])}  "
                f"ap={_format_metric(m['ap'])}"
            )
        for grp in ("GAN", "Diffusion"):
            grp_keys = [k for k, _, g in _universal_test_order if g == grp and k in per_dataset]
            grp_aucs = [per_dataset[k]["auc"] for k in grp_keys if per_dataset[k].get("auc") is not None]
            grp_accs = [per_dataset[k]["acc"] for k in grp_keys if per_dataset[k].get("acc") is not None]
            grp_aps = [per_dataset[k]["ap"] for k in grp_keys if per_dataset[k].get("ap") is not None]
            if grp_aucs:
                print(f"  {'[' + grp + ' avg]':<18s} acc={sum(grp_accs)/len(grp_accs):.4f}  auc={sum(grp_aucs)/len(grp_aucs):.4f}  ap={sum(grp_aps)/len(grp_aps):.4f}  datasets={len(grp_aucs)}")
        print(f"[final] AGGREGATE  acc={agg_acc:.4f}  auc={_format_metric(agg_auc)}  ap={_format_metric(agg_ap)}  datasets={len(per_dataset)}")
    else:
        final_metrics = evaluate(model, test_loader, prototypes, device, embedding_key=embedding_key)
    history.append({"step": global_step, "epoch": epochs, "test": final_metrics, "final": True})
    if _universal_test_loaders is not None:
        save_best_auc_model(final_metrics, epochs, global_step, final=True, metric_key="ap")
    else:
        save_best_auc_model(final_metrics, epochs, global_step, final=True)

    # ---- save prototypes alongside the best-weights file so they always travel together ----
    if save_best_auc_weight and _prototypes_ready(prototypes):
        proto_path = weight_dir / "prototypes.json"
        with proto_path.open("w") as handle:
            json.dump({str(k): v.tolist() for k, v in prototypes.items()}, handle, indent=2)
        print(f"[save] prototypes → {proto_path}")

    with (output_dir / "metrics.json").open("w") as handle:
        json.dump({"history": history, "prototypes": _serialize_prototypes(prototypes)}, handle, indent=2)

    # ---- print historical best test metric ----
    if _universal_test_loaders is not None:
        metric_key = "ap"
        metric_label = "best_test_ap"
    else:
        metric_key = "auc"
        metric_label = "best_test_auc"
    best_metric_so_far = float("-inf")
    best_metric_step = None
    for entry in history:
        value = (entry.get("test") or {}).get(metric_key)
        if value is not None and float(value) > best_metric_so_far:
            best_metric_so_far = float(value)
            best_metric_step = entry.get("step")
    if best_metric_step is not None:
        print(f"\n[best]  step={best_metric_step}  {metric_label}={best_metric_so_far:.4f}")
    else:
        print(f"\n[best]  no valid test {metric_key.upper()} recorded")

    return final_metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    train(config)


if __name__ == "__main__":
    main()
