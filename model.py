import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoModel, CLIPModel

try:
    from peft import LoraConfig, get_peft_model

    _HAS_PEFT = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    _HAS_PEFT = False


def resolve_feature_layer(feature_layer):
    if isinstance(feature_layer, str):
        value = feature_layer.strip().lower()
        if not value:
            raise ValueError("feature_layer must not be empty")
        if value in {"final", "last", "output"}:
            return "final"
        cls_match = re.fullmatch(r"(?:l|layer)(\d+)[_-]cls", value)
        if cls_match:
            layer_number = int(cls_match.group(1))
            if layer_number < 1:
                raise ValueError(f"feature_layer CLS layer must be >= 1, got {layer_number}")
            return ("cls", layer_number - 1)
        layer_match = re.fullmatch(r"l(\d+)", value)
        if layer_match:
            layer_number = int(layer_match.group(1))
            if layer_number < 1:
                raise ValueError(f"feature_layer layer must be >= 1, got {layer_number}")
            return layer_number - 1
        return int(value)
    return int(feature_layer)


DEFAULT_VISUAL_MODEL_TYPE = "clip"
DEFAULT_VISUAL_MODEL_NAME = "openai/clip-vit-large-patch14"
DEFAULT_FEATURE_POOLING = "default"
DEFAULT_LORA_TARGETS = {
    "clip": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "siglip": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "dinov2": ["query", "key", "value"],
    "vit": ["query", "key", "value"],
    "swin": ["query", "key", "value"],
}


def resolve_visual_model_config(config):
    """Return a normalized visual backbone config while preserving legacy CLIP keys."""
    config = config or {}
    visual_cfg = config.get("visual_model", {}) or {}
    model_type = str(
        visual_cfg.get("type")
        or config.get("visual_model_type")
        or DEFAULT_VISUAL_MODEL_TYPE
    ).strip().lower()
    model_name = (
        visual_cfg.get("name")
        or config.get("visual_model_name")
        or config.get("clip_model")
        or DEFAULT_VISUAL_MODEL_NAME
    )
    feature_pooling = str(
        visual_cfg.get("feature_pooling")
        or config.get("feature_pooling")
        or DEFAULT_FEATURE_POOLING
    ).strip().lower()
    return {
        "type": model_type,
        "name": str(model_name),
        "feature_pooling": feature_pooling,
    }


def _get_config_value(config, name):
    if config is None:
        return None
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name)
    return None


def _infer_hidden_size(model, explicit_hidden_size=None):
    if explicit_hidden_size is not None:
        return int(explicit_hidden_size)
    candidates = [getattr(model, "config", None)]
    config = candidates[0]
    if config is not None:
        for nested_name in ("vision_config", "encoder_config", "backbone_config"):
            nested = _get_config_value(config, nested_name)
            if nested is not None:
                candidates.append(nested)
    for candidate in candidates:
        hidden_size = _get_config_value(candidate, "hidden_size")
        if hidden_size is not None:
            return int(hidden_size)
        embed_dim = _get_config_value(candidate, "embed_dim")
        if embed_dim is not None:
            return int(embed_dim)
    raise ValueError(
        "Could not infer visual backbone hidden size. "
        "Set visual_model.hidden_size in the config or use a HuggingFace vision model with hidden_size."
    )


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        latent_dim=128,
        hidden_dim=512,
        topk=16,
        dropout=0.1,
        activation="relu",
        normalize_code=False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.topk = max(1, min(int(topk), self.latent_dim))
        self.activation = str(activation).strip().lower()
        self.normalize_code = bool(normalize_code)

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def _activate(self, code):
        if self.activation in {"relu", ""}:
            return F.relu(code)
        if self.activation == "gelu":
            return F.gelu(code)
        if self.activation == "tanh":
            return torch.tanh(code)
        if self.activation in {"identity", "linear", "none"}:
            return code
        raise ValueError(f"Unsupported SAE activation: {self.activation}")

    def _topk_sparse(self, code):
        if self.topk >= self.latent_dim:
            return code
        topk_values, topk_indices = torch.topk(code, k=self.topk, dim=1)
        sparse_code = code.new_zeros(code.shape)
        sparse_code.scatter_(1, topk_indices, topk_values)
        return sparse_code

    def forward(self, features):
        code = self._activate(self.encoder(features))
        sparse_code = self._topk_sparse(code)
        if self.normalize_code:
            sparse_code = F.normalize(sparse_code, dim=1)
        reconstruction = self.decoder(sparse_code)
        return {
            "code": code,
            "sparse_code": sparse_code,
            "reconstruction": reconstruction,
        }


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, inputs, lambd):
        ctx.lambd = float(lambd)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class ClipMlpClassifier(nn.Module):
    def __init__(
        self,
        clip_model="openai/clip-vit-large-patch14",
        feature_layer="final",
        embedding_dim=128,
        hidden_dim=512,
        dropout=0.1,
        freeze_clip=True,
        normalize_clip_features=True,
        use_lora=False,
        lora_r=2,
        lora_alpha=32,
        lora_dropout=0.25,
        lora_target_modules=None,
        cache_dir=None,
        local_files_only=False,
        head_type="mlp",
        sae_enabled=False,
        sae_latent_dim=None,
        sae_hidden_dim=None,
        sae_topk=16,
        sae_activation="relu",
        sae_normalize_code=False,
        use_ae=False,
        ae_type="ae",
        use_domain_adversarial=False,
        domain_num_classes=5,
        domain_hidden_dim=64,
        domain_dropout=0.3,
        use_real_manifold_residual=False,
        residual_embedding_key="residual_embedding",
        detach_reconstruction_in_residual=False,
        normalize_residual_features=False,
        use_inter_layer_residual=False,
        inter_mid_layer="L12",
        inter_residual_mode="residual",
        inter_adapter="linear",
        inter_residual_direction="mid_minus_final",
        inter_output_embedding_key="inter_layer_residual_embedding",
        normalize_inter_residual_features=False,
        visual_model_type="clip",
        visual_model_name=None,
        feature_pooling="default",
        visual_hidden_size=None,
    ):
        super().__init__()
        clip_kwargs = {"local_files_only": bool(local_files_only)}
        if cache_dir:
            clip_kwargs["cache_dir"] = cache_dir
        self.visual_model_type = str(visual_model_type or "clip").strip().lower()
        self.visual_model_name = str(visual_model_name or clip_model)
        self.feature_pooling = str(feature_pooling or "default").strip().lower()
        if self.feature_pooling not in {"default", "cls", "mean", "pooler"}:
            raise ValueError(
                "feature_pooling must be one of: default, cls, mean, pooler; "
                f"got {feature_pooling}"
            )
        if self.visual_model_type == "clip":
            self.clip = CLIPModel.from_pretrained(self.visual_model_name, **clip_kwargs)
        else:
            self.clip = AutoModel.from_pretrained(self.visual_model_name, **clip_kwargs)
        self.feature_layer = resolve_feature_layer(feature_layer)
        self.normalize_clip_features = bool(normalize_clip_features)
        self.use_lora = bool(use_lora)
        self.use_ae = bool(use_ae)
        self.use_domain_adversarial = bool(use_domain_adversarial)
        self.use_real_manifold_residual = bool(use_real_manifold_residual)
        self.residual_embedding_key = str(residual_embedding_key)
        self.detach_reconstruction_in_residual = bool(detach_reconstruction_in_residual)
        self.normalize_residual_features = bool(normalize_residual_features)
        self.use_inter_layer_residual = bool(use_inter_layer_residual)
        self.inter_output_embedding_key = str(inter_output_embedding_key)
        self.normalize_inter_residual_features = bool(normalize_inter_residual_features)
        self.ae_type = str(ae_type).strip().lower()
        if self.ae_type not in {"ae", "vae"}:
            raise ValueError(f"ae_type must be 'ae' or 'vae', got {ae_type}")

        # ---- optional LoRA fine-tuning on the visual encoder ----
        self._lora_module_attr = None
        if self.use_lora:
            if not _HAS_PEFT:
                raise ImportError(
                    "LoRA requires the peft package.  Install it with `pip install peft` "
                    "or set use_lora: false in the config."
                )
            target_modules = list(lora_target_modules) if lora_target_modules else list(
                DEFAULT_LORA_TARGETS.get(self.visual_model_type, DEFAULT_LORA_TARGETS["clip"])
            )
            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=target_modules,
                bias="none",
            )
            try:
                if self.visual_model_type == "clip" or hasattr(self.clip, "vision_model"):
                    self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_config)
                    self._lora_module_attr = "vision_model"
                else:
                    self.clip = get_peft_model(self.clip, lora_config)
                    self._lora_module_attr = "clip"
            except ValueError as exc:
                raise ValueError(
                    f"Could not apply LoRA to visual_model.type={self.visual_model_type} "
                    f"with target_modules={target_modules}. Set lora.target_modules explicitly "
                    "after inspecting the selected model's module names."
                ) from exc

        # ---- freeze / unfreeze strategy ----
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad_(False)
        if self.use_lora:
            # only LoRA adapters stay trainable
            lora_module = self.clip.vision_model if self._lora_module_attr == "vision_model" else self.clip
            for name, param in lora_module.named_parameters():
                param.requires_grad_("lora_" in name)

        # ---- gradient checkpointing on ViT encoder to reduce peak memory ----
        # With paired_view_fisher_consistency, two forward passes coexist in autograd.
        # Checkpointing trades extra recomputation for lower activation memory.
        try:
            self.clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            from functools import partial
            from torch.utils.checkpoint import checkpoint

            checkpoint_func = partial(checkpoint, use_reentrant=False)
            for module in self.clip.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module._gradient_checkpointing_func = checkpoint_func
                    module.gradient_checkpointing = True
        except (AttributeError, TypeError, ValueError):
            pass  # not all CLIP vision model versions support this, silently skip

        if self.visual_model_type == "clip":
            if isinstance(self.feature_layer, tuple) and self.feature_layer[0] == "cls":
                feature_dim = int(self.clip.vision_model.config.hidden_size)
            else:
                feature_dim = int(self.clip.config.projection_dim)
        else:
            feature_dim = _infer_hidden_size(self.clip, visual_hidden_size)
        head_type = str(head_type).strip().lower()
        if head_type not in {"mlp", "sae"}:
            raise ValueError(f"head_type must be 'mlp' or 'sae', got {head_type}")
        self.use_sae_head = bool(sae_enabled) or head_type == "sae"

        if self.use_sae_head:
            sae_latent_dim = int(sae_latent_dim or feature_dim)
            self.embedding_dim = sae_latent_dim
            self.projector = None
            self.sae = SparseAutoencoder(
                feature_dim=feature_dim,
                latent_dim=sae_latent_dim,
                hidden_dim=int(sae_hidden_dim or hidden_dim),
                topk=int(sae_topk),
                dropout=float(dropout),
                activation=sae_activation,
                normalize_code=sae_normalize_code,
            )
        else:
            self.embedding_dim = int(embedding_dim)
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )
            self.sae = None

        self.ae_decoder = None
        self.vae_mu = None
        self.vae_logvar = None
        if self.use_ae and not self.use_sae_head:
            self.ae_decoder = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feature_dim),
            )
            if self.ae_type == "vae":
                self.vae_mu = nn.Linear(self.embedding_dim, self.embedding_dim)
                self.vae_logvar = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.domain_classifier = None
        if self.use_domain_adversarial:
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, int(domain_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(domain_dropout)),
                nn.Linear(int(domain_hidden_dim), int(domain_num_classes)),
            )

    def train(self, mode=True):
        super().train(mode)
        # keep CLIP in eval mode unless some parameters (e.g. LoRA) require gradients
        if not any(p.requires_grad for p in self.clip.parameters()):
            self.clip.eval()
        return self

    def _run_clip_with_checkpointing(self, fn):
        checkpoint_modules = []
        previous_training_flags = []
        if torch.is_grad_enabled():
            for module in self.clip.modules():
                if getattr(module, "gradient_checkpointing", False):
                    checkpoint_modules.append(module)
                    previous_training_flags.append(module.training)
                    module.training = True
        try:
            return fn()
        finally:
            for module, training in zip(checkpoint_modules, previous_training_flags):
                module.training = training

    def extract_clip_features(self, pixel_values):
        if self.visual_model_type == "clip":
            features = self._extract_clip_features(pixel_values)
        else:
            features = self._extract_auto_model_features(pixel_values)

        features = features.float()
        if self.normalize_clip_features:
            features = F.normalize(features, dim=1)
        return features

    def _extract_clip_features(self, pixel_values):
        if self.feature_layer == "final":
            return self._run_clip_with_checkpointing(lambda: self.clip.get_image_features(pixel_values=pixel_values))

        outputs = self._run_clip_with_checkpointing(lambda: self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True))
        num_layers = int(self.clip.vision_model.config.num_hidden_layers)
        raw_cls = isinstance(self.feature_layer, tuple) and self.feature_layer[0] == "cls"
        layer = int(self.feature_layer[1] if raw_cls else self.feature_layer)
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"feature_layer must be in [0, {num_layers - 1}], got {layer}")
        cls_features = outputs.hidden_states[layer + 1][:, 0, :]
        if raw_cls:
            return cls_features
        cls_features = self.clip.vision_model.post_layernorm(cls_features)
        return self.clip.visual_projection(cls_features)

    def _extract_auto_model_features(self, pixel_values):
        forward_model = self.clip.vision_model if hasattr(self.clip, "vision_model") else self.clip
        outputs = self._run_clip_with_checkpointing(
            lambda: forward_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        pooler_output = getattr(outputs, "pooler_output", None)

        if self.feature_layer == "final":
            if self.feature_pooling in {"default", "pooler"} and pooler_output is not None:
                return pooler_output
            if last_hidden_state is None:
                raise ValueError("Selected visual model did not return last_hidden_state")
            return self._pool_hidden_state(last_hidden_state)

        if hidden_states is None:
            raise ValueError("Selected visual model did not return hidden_states; cannot use intermediate feature_layer")
        raw_cls = isinstance(self.feature_layer, tuple) and self.feature_layer[0] == "cls"
        layer = int(self.feature_layer[1] if raw_cls else self.feature_layer)
        num_layers = len(hidden_states) - 1
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"feature_layer must be in [0, {num_layers - 1}], got {layer}")
        # Non-CLIP backbones do not have CLIP's visual_projection; L<N> and L<N>_CLS
        # both use the selected hidden state's pooled representation.
        return self._pool_hidden_state(hidden_states[layer + 1])

    def _pool_hidden_state(self, hidden_state):
        if self.feature_pooling == "mean":
            tokens = hidden_state[:, 1:, :] if hidden_state.size(1) > 1 else hidden_state
            return tokens.mean(dim=1)
        # default / cls / pooler fallback: use CLS token when no pooler_output is available.
        return hidden_state[:, 0, :]

    def forward(self, pixel_values, detach_ae_input=False, grl_lambda=None):
        clip_features = self.extract_clip_features(pixel_values)
        if self.use_sae_head:
            sae_outputs = self.sae(clip_features)
            outputs = {
                "embedding": sae_outputs["sparse_code"],
                "clip_features": clip_features,
                "visual_features": clip_features,
                "sae_code": sae_outputs["code"],
                "sae_sparse_code": sae_outputs["sparse_code"],
                "sae_reconstruction": sae_outputs["reconstruction"],
                "sae_reconstruction_target": clip_features,
            }
            return outputs
        embeddings = self.projector(clip_features)
        outputs = {
            "embedding": embeddings,
            "clip_features": clip_features,
            "visual_features": clip_features,
        }
        if self.use_ae:
            ae_embeddings = self.projector(clip_features.detach()) if detach_ae_input else embeddings
            if self.ae_type == "vae":
                mu = self.vae_mu(ae_embeddings)
                logvar = self.vae_logvar(ae_embeddings)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
                latent = mu + eps * std
                outputs.update(
                    {
                        "reconstruction": self.ae_decoder(latent),
                        "vae_mu": mu,
                        "vae_logvar": logvar,
                        "vae_latent": latent,
                    }
                )
            else:
                outputs["reconstruction"] = self.ae_decoder(ae_embeddings)

        if self.use_real_manifold_residual and outputs.get("reconstruction") is not None:
            reconstruction = outputs["reconstruction"].detach() if self.detach_reconstruction_in_residual else outputs["reconstruction"]
            residual = clip_features - reconstruction
            if self.normalize_residual_features:
                residual = F.normalize(residual, dim=1)
            outputs["residual_features"] = residual
            outputs[self.residual_embedding_key] = residual

        if self.use_inter_layer_residual:
            outputs[self.inter_output_embedding_key] = embeddings

        if self.domain_classifier is not None:
            lambd = 1.0 if grl_lambda is None else float(grl_lambda)
            reversed_embeddings = GradientReverse.apply(embeddings, lambd)
            outputs["domain_logits"] = self.domain_classifier(reversed_embeddings)
        return outputs
