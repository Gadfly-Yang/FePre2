import torch
import torch.nn.functional as F


def _validate_feature_targets(features, targets, target_name):
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, dim]")
    if targets.ndim != 1:
        raise ValueError(f"{target_name} must have shape [batch]")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(f"features and {target_name} must have the same batch size")


def l_sep_loss(features, labels, gamma=1.0, eps=1e-8):
    """Exponential per-dimension separation loss normalized by class variance."""
    _validate_feature_targets(features, labels, "labels")

    real_mask = labels.eq(0)
    fake_mask = labels.eq(1)
    if not bool(real_mask.any()) or not bool(fake_mask.any()):
        return features.new_tensor(0.0)

    features = features.float()
    real_features = features[real_mask]
    fake_features = features[fake_mask]
    real_center = real_features.mean(dim=0)
    fake_center = fake_features.mean(dim=0)
    real_variance = real_features.var(dim=0, unbiased=False)
    fake_variance = fake_features.var(dim=0, unbiased=False)
    center_distance_squared = (fake_center - real_center).pow(2)
    variance_sum = fake_variance + real_variance + float(eps)
    return torch.exp(-float(gamma) * center_distance_squared / variance_sum).mean()


class GlobalFisherStats:
    """EMA class statistics for the global Fisher separation loss."""

    def __init__(self, momentum=0.99):
        self.momentum = float(momentum)
        self.means = {0: None, 1: None}
        self.variances = {0: None, 1: None}

    def ready(self):
        return all(self.means[cls] is not None and self.variances[cls] is not None for cls in (0, 1))

    @torch.no_grad()
    def update(self, features, labels):
        _validate_feature_targets(features, labels, "labels")

        features = features.detach().float()
        for cls in (0, 1):
            mask = labels.eq(cls)
            if not bool(mask.any()):
                continue

            cls_features = features[mask]
            batch_mean = cls_features.mean(dim=0)
            batch_variance = cls_features.var(dim=0, unbiased=False)
            if self.means[cls] is None:
                self.means[cls] = batch_mean.clone()
                self.variances[cls] = batch_variance.clone()
            else:
                momentum = self.momentum
                self.means[cls] = momentum * self.means[cls] + (1.0 - momentum) * batch_mean
                self.variances[cls] = momentum * self.variances[cls] + (1.0 - momentum) * batch_variance


def global_fisher_loss(features, labels, stats, gamma=1.0, eps=1e-8):
    """Global Fisher separation loss using EMA statistics as class references."""
    _validate_feature_targets(features, labels, "labels")
    if not stats.ready():
        return features.new_tensor(0.0)

    real_mask = labels.eq(0)
    fake_mask = labels.eq(1)
    losses = []
    features = features.float()

    if bool(real_mask.any()):
        real_features = features[real_mask]
        real_center = real_features.mean(dim=0)
        real_variance = real_features.var(dim=0, unbiased=False)
        center_distance_squared = (real_center - stats.means[1].to(features.device)).pow(2)
        variance_sum = real_variance + stats.variances[1].to(features.device) + float(eps)
        losses.append(torch.exp(-float(gamma) * center_distance_squared / variance_sum))

    if bool(fake_mask.any()):
        fake_features = features[fake_mask]
        fake_center = fake_features.mean(dim=0)
        fake_variance = fake_features.var(dim=0, unbiased=False)
        center_distance_squared = (fake_center - stats.means[0].to(features.device)).pow(2)
        variance_sum = fake_variance + stats.variances[0].to(features.device) + float(eps)
        losses.append(torch.exp(-float(gamma) * center_distance_squared / variance_sum))

    if not losses:
        return features.new_tensor(0.0)
    return torch.cat(losses).mean()


def compute_fisher_ratio(features, labels, eps=1e-8):
    """Return per-dimension Fisher ratios between fake and real embeddings."""
    _validate_feature_targets(features, labels, "labels")
    real_mask = labels.eq(0)
    fake_mask = labels.eq(1)
    if not bool(real_mask.any()) or not bool(fake_mask.any()):
        return features.new_zeros(features.shape[1])

    features = features.float()
    real_features = features[real_mask]
    fake_features = features[fake_mask]
    real_center = real_features.mean(dim=0)
    fake_center = fake_features.mean(dim=0)
    real_variance = real_features.var(dim=0, unbiased=False)
    fake_variance = fake_features.var(dim=0, unbiased=False)
    return (fake_center - real_center).pow(2) / (fake_variance + real_variance + float(eps))


def _ratio_from_features_or_ratio(features_or_ratio, labels=None, eps=1e-8):
    if labels is None:
        if features_or_ratio.ndim != 1:
            raise ValueError("ratio must have shape [dim] when labels is None")
        return features_or_ratio.float()
    return compute_fisher_ratio(features_or_ratio, labels, eps=eps)


def _safe_topk(ratio, topk):
    if ratio.numel() == 0:
        return ratio
    k = max(1, min(int(topk), int(ratio.numel())))
    return torch.topk(ratio, k=k, largest=True).values


def entropy_sparse_fisher_loss(features_or_ratio, labels=None, temperature=0.5, eps=1e-8):
    """Entropy concentration regularizer over the Fisher-ratio softmax distribution."""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    ratio = _ratio_from_features_or_ratio(features_or_ratio, labels, eps=eps)
    if torch.count_nonzero(ratio).item() == 0:
        return ratio.new_tensor(0.0)
    log_probs = F.log_softmax(ratio / temperature, dim=0)
    probs = log_probs.exp()
    return -(probs * log_probs).sum()


def fisher_ratio_gini(ratio, eps=1e-8):
    ratio = torch.clamp(ratio.float(), min=0.0)
    total = ratio.sum()
    if float(total.item()) <= float(eps):
        return ratio.new_tensor(0.0)
    sorted_ratio, _ = torch.sort(ratio, descending=False)
    d = ratio.numel()
    index = torch.arange(1, d + 1, dtype=ratio.dtype, device=ratio.device)
    return (2.0 * (index * sorted_ratio).sum()) / (d * total + float(eps)) - (d + 1.0) / d


@torch.no_grad()
def sparse_fisher_metrics(features_or_ratio, labels=None, topk=16, zero_threshold=1e-3, eps=1e-8):
    """Return monitoring metrics for Fisher-ratio sparsity."""
    ratio = _ratio_from_features_or_ratio(features_or_ratio, labels, eps=eps).detach().float()
    if ratio.numel() == 0:
        return {"d_eff": 0.0, "gini": 0.0, "topk_mass_ratio": 0.0, "zero_ratio": 0.0, "max_ratio": 0.0}
    total = ratio.clamp(min=0.0).sum()
    if float(total.item()) <= float(eps):
        return {"d_eff": float(ratio.numel()), "gini": 0.0, "topk_mass_ratio": 0.0, "zero_ratio": 1.0, "max_ratio": 0.0}
    probs = ratio.clamp(min=0.0) / (total + float(eps))
    d_eff = 1.0 / float((probs.pow(2).sum() + float(eps)).item())
    topk_mass = _safe_topk(ratio.clamp(min=0.0), topk).sum()
    return {
        "d_eff": d_eff,
        "gini": float(fisher_ratio_gini(ratio, eps=eps).item()),
        "topk_mass_ratio": float((topk_mass / (total + float(eps))).item()),
        "zero_ratio": float((ratio < float(zero_threshold)).float().mean().item()),
        "max_ratio": float(ratio.max().item()),
    }


def compute_fisher_ratio_from_masks(features, real_mask, fake_mask, eps=1e-8):
    """Return per-dimension Fisher ratios using explicit real/fake masks."""
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, dim]")
    if real_mask.ndim != 1 or fake_mask.ndim != 1:
        raise ValueError("real_mask and fake_mask must have shape [batch]")
    if features.shape[0] != real_mask.shape[0] or features.shape[0] != fake_mask.shape[0]:
        raise ValueError("features and masks must have the same batch size")
    if not bool(real_mask.any()) or not bool(fake_mask.any()):
        return features.new_zeros(features.shape[1])

    features = features.float()
    real_features = features[real_mask]
    fake_features = features[fake_mask]
    real_center = real_features.mean(dim=0)
    fake_center = fake_features.mean(dim=0)
    real_variance = real_features.var(dim=0, unbiased=False)
    fake_variance = fake_features.var(dim=0, unbiased=False)
    return (fake_center - real_center).pow(2) / (fake_variance + real_variance + float(eps))


def fisher_distribution(ratio, temperature=0.5, eps=1e-8):
    """Convert a Fisher-ratio vector into a dimension-importance distribution."""
    if ratio.ndim != 1:
        raise ValueError("ratio must have shape [dim]")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    ratio = ratio.float()
    if torch.count_nonzero(ratio).item() == 0:
        return torch.full_like(ratio, 1.0 / max(1, ratio.numel()))
    return F.softmax(ratio / temperature, dim=0).clamp_min(float(eps))


def kl_divergence_distribution(p, q, eps=1e-8):
    """KL(p || q) for probability vectors."""
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = p.float().clamp_min(float(eps))
    q = q.float().clamp_min(float(eps))
    p = p / p.sum().clamp_min(float(eps))
    q = q / q.sum().clamp_min(float(eps))
    return (p * (torch.log(p) - torch.log(q))).sum()


def symmetric_kl_distribution(p, q, eps=1e-8):
    """Symmetric KL divergence between two probability vectors."""
    return kl_divergence_distribution(p, q, eps=eps) + kl_divergence_distribution(q, p, eps=eps)




@torch.no_grad()
def fisher_topk_overlap(ratio_a, ratio_b, topk=16):
    """Jaccard overlap of top-k Fisher dimensions."""
    if ratio_a is None or ratio_b is None or ratio_a.numel() == 0 or ratio_b.numel() == 0:
        return 0.0
    k = max(1, min(int(topk), int(ratio_a.numel()), int(ratio_b.numel())))
    a = set(torch.topk(ratio_a.detach().float(), k=k).indices.cpu().tolist())
    b = set(torch.topk(ratio_b.detach().float(), k=k).indices.cpu().tolist())
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def cross_label_separation_loss(features, labels, gamma=1.0, eps=1e-8):
    """Backward-compatible alias for the per-dimension L_sep loss."""
    return l_sep_loss(features, labels, gamma=gamma, eps=eps)

class FisherDistributionEMA:
    """EMA target distribution for Fisher-dimension stability losses."""

    def __init__(self, momentum=0.99):
        self.momentum = float(momentum)
        self.distribution = None

    def ready(self):
        return self.distribution is not None

    @torch.no_grad()
    def update(self, distribution):
        distribution = distribution.detach().float()
        if self.distribution is None:
            self.distribution = distribution.clone()
        else:
            self.distribution = self.momentum * self.distribution.to(distribution.device) + (1.0 - self.momentum) * distribution

    def loss(self, distribution, detach_target=True, eps=1e-8):
        if self.distribution is None:
            return distribution.new_tensor(0.0)
        target = self.distribution.to(distribution.device)
        if detach_target:
            target = target.detach()
        return symmetric_kl_distribution(distribution, target, eps=eps)


def supervised_contrastive_loss(features, labels, temperature=0.07, eps=1e-8):
    """Supervised contrastive loss over same-label positives in a batch."""
    _validate_feature_targets(features, labels, "labels")
    features = F.normalize(features.float(), dim=1)
    labels = labels.view(-1)
    batch_size = int(features.shape[0])
    if batch_size <= 1:
        return features.new_tensor(0.0)

    logits = features @ features.t() / float(temperature)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(eye, -float("inf"))
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~eye
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return features.new_tensor(0.0)

    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probs = log_probs.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(float(eps))
    return -positive_log_probs[valid].mean()


def domain_classification_loss(logits, domain_labels, ignore_index=-1):
    mask = domain_labels.ne(int(ignore_index))
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return logits.new_tensor(0.0), 0
    return F.cross_entropy(logits[mask], domain_labels[mask]), valid_count


@torch.no_grad()
def domain_accuracy(logits, domain_labels, ignore_index=-1):
    mask = domain_labels.ne(int(ignore_index))
    if not bool(mask.any()):
        return 0.0
    preds = logits[mask].argmax(dim=1)
    return float(preds.eq(domain_labels[mask]).float().mean().item())


def compute_group_fisher_ratios(features, labels, group_ids, min_real=1, min_fake=1, eps=1e-8):
    _validate_feature_targets(features, labels, "labels")
    if group_ids.ndim != 1 or group_ids.shape[0] != features.shape[0]:
        raise ValueError("group_ids must have shape [batch]")
    results = []
    for group in torch.unique(group_ids.detach()).tolist():
        group_mask = group_ids.eq(int(group))
        real_mask = group_mask & labels.eq(0)
        fake_mask = group_mask & labels.eq(1)
        if int(real_mask.sum().item()) < int(min_real) or int(fake_mask.sum().item()) < int(min_fake):
            continue
        ratio = compute_fisher_ratio_from_masks(features, real_mask, fake_mask, eps=eps)
        results.append({"group": int(group), "ratio": ratio})
    return results


def semantic_conditioned_fisher_loss(features, labels, group_ids, gamma=1.0, eps=1e-8, min_real=1, min_fake=1):
    group_ratios = compute_group_fisher_ratios(features, labels, group_ids, min_real=min_real, min_fake=min_fake, eps=eps)
    losses = []
    for item in group_ratios:
        ratio = item["ratio"]
        losses.append(torch.exp(-float(gamma) * ratio).mean())
    if not losses:
        return features.new_tensor(0.0), group_ratios
    return torch.stack(losses).mean(), group_ratios


def fisher_group_distribution_consistency_loss(ratios, temperature=0.5, min_groups=2, eps=1e-8):
    if len(ratios) < int(min_groups):
        device = ratios[0].device if ratios else None
        return (ratios[0].new_tensor(0.0) if ratios else torch.tensor(0.0, device=device))
    distributions = [fisher_distribution(ratio, temperature=temperature, eps=eps) for ratio in ratios]
    losses = []
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            losses.append(symmetric_kl_distribution(distributions[i], distributions[j], eps=eps))
    if not losses:
        return distributions[0].new_tensor(0.0)
    return torch.stack(losses).mean()


def manipulation_wise_fisher_consistency_loss(
    features,
    labels,
    manipulation_ids,
    temperature=0.5,
    unknown_index=-1,
    min_real=1,
    min_fake=1,
    min_groups=2,
    eps=1e-8,
):
    valid_mask = manipulation_ids.ne(int(unknown_index))
    if not bool(valid_mask.any()):
        return features.new_tensor(0.0), []
    group_ratios = compute_group_fisher_ratios(
        features[valid_mask],
        labels[valid_mask],
        manipulation_ids[valid_mask],
        min_real=min_real,
        min_fake=min_fake,
        eps=eps,
    )
    loss = fisher_group_distribution_consistency_loss(
        [item["ratio"] for item in group_ratios],
        temperature=temperature,
        min_groups=min_groups,
        eps=eps,
    ) if group_ratios else features.new_tensor(0.0)
    return loss, group_ratios
