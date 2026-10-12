import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPImageProcessor


# ---------------------------------------------------------------------------
#  CLIP normalization parameters from the Effort (UniversalFakeDetect_Benchmark) reference implementation
# ---------------------------------------------------------------------------
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def translate_duplicate(img, cropSize):
    """Tile a small image with itself, matching the UniversalFakeDetect_Benchmark
    reference implementation.

    If the shorter side of the image is smaller than ``cropSize``, tile the image
    until both sides are at least ``cropSize``.
    """
    if min(img.size) < cropSize:
        width, height = img.size
        new_width = width * math.ceil(cropSize / width)
        new_height = height * math.ceil(cropSize / height)
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img


REAL_LABEL_NAMES = {
    "FF-real",
    "DFD_real",
    "DFDCP_Real",
    "Real",
    "real",
    "CelebDFv1_real",
    "CelebDFv2_real",
    "Celeb-real",
    "YouTube-real",
    "DFDC_Real",
    "DFDC_real",
    "DeepfakeDetection_real",
    "DF_real",
    "UADFV_Real",
    "roop_Real",
}

IDENTITY_KEYS = (
    "identity",
    "identity_id",
    "person_id",
    "subject",
    "original",
    "original_video",
)

SUPPORTED_FAKE_IDENTITY_ROLES = {"source", "target"}

# ---------------------------------------------------------------------------
#  UniversalFakeDetect – directory-based data loading
# ---------------------------------------------------------------------------

# Diffusion cross-source test pairs: (real_subdir_in_diffusion, fake_subdir_in_diffusion, dataset_key)
UNIVERSAL_DIFFUSION_PAIRS = [
    ("imagenet", "guided", "guided"),
    ("laion", "ldm_200", "ldm_200"),
    ("laion", "ldm_200_cfg", "ldm_200_cfg"),
    ("laion", "ldm_100", "ldm_100"),
    ("laion", "glide_100_27", "glide_100_27"),
    ("laion", "glide_50_27", "glide_50_27"),
    ("laion", "glide_100_10", "glide_100_10"),
    ("laion", "dalle", "dalle"),
]

# C2P-CLIP style test dataset ordering (matches the paper's evaluation table)
# Each entry: (internal_key, display_name, group)
# internal_key matches the dataset_name in FrameRecord
UNIVERSAL_TEST_ORDER = [
    # ---- GAN-based ----
    ("progan",           "ProGAN",           "GAN"),
    ("cyclegan",         "CycleGAN",         "GAN"),
    ("biggan",           "BigGAN",           "GAN"),
    ("stylegan",         "StyleGAN",         "GAN"),
    ("gaugan",           "GauGAN",           "GAN"),
    ("stargan",          "StarGAN",          "GAN"),
    ("deepfake",         "DeepFake",         "GAN"),
    ("seeingdark",       "SITD",             "GAN"),
    ("san",              "SAN",              "GAN"),
    ("crn",              "CRN",              "GAN"),
    ("imle",             "IMLE",             "GAN"),
    ("stylegan2",        "StyleGAN2",        "GAN"),
    ("whichfaceisreal",  "WFIR",             "GAN"),
    # ---- Diffusion-based ----
    ("guided",           "Guided",           "Diffusion"),
    ("ldm_200",          "LDM_200",          "Diffusion"),
    ("ldm_200_cfg",      "LDM_200_cfg",      "Diffusion"),
    ("ldm_100",          "LDM_100",          "Diffusion"),
    ("glide_100_27",     "Glide_100_27",     "Diffusion"),
    ("glide_50_27",      "Glide_50_27",      "Diffusion"),
    ("glide_100_10",     "Glide_100_10",     "Diffusion"),
    ("dalle",            "DALL-E",           "Diffusion"),
]


def _scan_universal_subdir(records, dataset_name, real_dir, fake_dir, split):
    """Scan 0_real/ and 1_fake/ directories and append FrameRecord entries."""
    for label, img_dir in [(0, real_dir), (1, fake_dir)]:
        if img_dir is None or not img_dir.is_dir():
            continue
        paths = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
        for p in paths:
            records.append(
                FrameRecord(
                    path=str(p),
                    label=label,
                    label_name="real" if label == 0 else "fake",
                    dataset_name=dataset_name,
                    split=split,
                    video_name="",          # universal data has no video grouping
                    identity=p.stem,
                )
            )


def _collect_universal_subdirs(root, categories, exclude):
    """Return sorted list of Path objects for matching subdirectories under *root*.

    Handles nested names like ``"cyclegan/apple"`` — the first component must
    match a top-level directory, and the remainder narrows to nested children.
    *exclude* is only applied when *categories* is ``"all"``.
    """
    if not categories or categories == "all":
        exclude = set(exclude or [])
        return sorted([d for d in Path(root).iterdir() if d.is_dir() and d.name not in exclude])

    results = []
    for cat in split_dataset_names(categories):
        parts = cat.split("/")
        parent = Path(root) / parts[0]
        if not parent.is_dir():
            continue

        if len(parts) == 1:
            results.append(parent)
        else:
            nested_path = parent
            for part in parts[1:]:
                nested_path = nested_path / part
            if nested_path.is_dir():
                nr = nested_path / "0_real"
                nf = nested_path / "1_fake"
                if nr.is_dir() or nf.is_dir():
                    results.append(nested_path)

    # Deduplicate while preserving order
    unique = []
    for d in results:
        if d not in unique:
            unique.append(d)
    return sorted(unique)


def load_universal_records(
    data_root,
    split="train",
    categories=None,
    exclude=None,
):
    """Load FrameRecord list from a UniversalFakeDetect-style directory tree.
    All images under the matched directories are loaded — no subsampling.

    Directory convention
    --------------------
    datasets/train/{category}/0_real/   ← real images
    datasets/train/{category}/1_fake/   ← fake images
    datasets/test/{generator}/0_real/   ← real images
    datasets/test/{generator}/1_fake/   ← fake images
    datasets/test/cyclegan/{class}/0_real/  ← nested (cyclegan)
    datasets/test/cyclegan/{class}/1_fake/
    diffusion_datasets/{model}/0_real/  ← real images (imagenet, laion)
    diffusion_datasets/{model}/1_fake/  ← fake images (guided, ldm_*, etc.)

    Diffusion datasets use cross-source real/fake pairs defined in
    ``UNIVERSAL_DIFFUSION_PAIRS``.

    Parameters
    ----------
    data_root : str
        Root of the UniversalFakeDetect checkout.
    split : str
        One of ``"train"``, ``"test"``, or ``"diffusion"``.
    categories : str or list or None
        Comma-separated names, a list, or ``"all"`` to include every
        subdirectory (after applying *exclude*).
    exclude : list or None
        Directory names to skip.
    """
    records: List[FrameRecord] = []
    data_root = Path(data_root)

    if split == "train":
        root = data_root / "datasets" / "train"
        exclude = set(exclude or [])
    elif split == "test":
        root = data_root / "datasets" / "test"
        exclude = set(exclude or [])
    elif split == "diffusion":
        root = data_root / "diffusion_datasets"
        exclude = set(exclude or [])
    else:
        raise ValueError(f"Unknown universal split: {split}")

    subdirs = _collect_universal_subdirs(root, categories, exclude)

    for subdir in subdirs:
        # Detect if this is a nested path (e.g. "cyclegan/apple")
        subdir_rel = subdir.relative_to(root)
        if len(subdir_rel.parts) > 1:
            # Direct nested match — scan immediately
            ds_name = str(subdir_rel).replace("\\", "/")
            real_dir = subdir / "0_real"
            fake_dir = subdir / "1_fake"
            _scan_universal_subdir(records, ds_name, real_dir, fake_dir, split)
            continue

        real_dir = subdir / "0_real"
        fake_dir = subdir / "1_fake"
        has_direct = real_dir.is_dir() or fake_dir.is_dir()

        if has_direct:
            _scan_universal_subdir(records, subdir.name, real_dir, fake_dir, split)
        else:
            # Nested structure – e.g. cyclegan/{apple,horse,…}
            for nested in sorted(subdir.iterdir()):
                if not nested.is_dir():
                    continue
                nr = nested / "0_real"
                nf = nested / "1_fake"
                if nr.is_dir() or nf.is_dir():
                    ds_name = f"{subdir.name}/{nested.name}"
                    _scan_universal_subdir(records, ds_name, nr, nf, split)

    return records


def load_universal_diffusion_records(data_root):
    """Load cross-source diffusion test records.
    All images under the matched directories are loaded — no subsampling.

    Returns one flat list where each diffusion pair becomes a dataset named
    after its *key* (e.g. ``"guided"``, ``"ldm_200"``).
    """
    records: List[FrameRecord] = []
    diffusion_root = Path(data_root) / "diffusion_datasets"

    for real_sub, fake_sub, key in UNIVERSAL_DIFFUSION_PAIRS:
        real_dir = diffusion_root / real_sub / "0_real"
        fake_dir = diffusion_root / fake_sub / "1_fake"
        _scan_universal_subdir(records, key, real_dir, fake_dir, "diffusion")

    return records

@dataclass(frozen=True)
class FrameRecord:
    path: str
    label: int
    label_name: str
    dataset_name: str
    split: str
    video_name: str
    identity: str


def split_dataset_names(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _is_real_label(label_name, label_dict=None):
    if label_dict and label_name in label_dict:
        return int(label_dict[label_name]) == 0
    if label_name in REAL_LABEL_NAMES:
        return True
    lowered = str(label_name).lower()
    return "real" in lowered and "fake" not in lowered


def _deepfakebench_label(label_name, split, is_training=False, label_dict=None):
    if str(label_name).startswith("DFDC_"):
        return 0 if label_name == "DFDC_Real" else 1
    return 0 if _is_real_label(label_name, label_dict) else 1


def _resolve_json_path(dataset_json_folder, dataset_name):
    path = Path(dataset_name)
    if path.suffix == ".json" and path.exists():
        return path
    suffix_name = dataset_name if str(dataset_name).endswith(".json") else f"{dataset_name}.json"
    return Path(dataset_json_folder) / suffix_name


def _select_compression(split_entry, dataset_name, compression):
    if not isinstance(split_entry, dict):
        return split_entry
    if compression in split_entry:
        return split_entry[compression]
    if str(dataset_name).endswith("_c40") and "c40" in split_entry:
        return split_entry["c40"]
    return split_entry


def _normalize_path(path, base_data_path=None):
    path = str(path).replace("\\", "/")
    if base_data_path and not os.path.isabs(path):
        path = os.path.join(str(base_data_path), path)
    return path


def _stem(value):
    value = str(value).replace("\\", "/").rstrip("/")
    return Path(value).stem or Path(value).name


def _identity_from_video_name(video_name, label_name, fake_identity_role="source"):
    name = _stem(video_name)
    parts = [part for part in name.split("_") if part]
    if parts:
        if _is_real_label(label_name) or len(parts) < 2:
            return parts[0]
        role = str(fake_identity_role).lower()
        if role not in SUPPORTED_FAKE_IDENTITY_ROLES:
            raise ValueError(f"fake_identity_role must be one of {sorted(SUPPORTED_FAKE_IDENTITY_ROLES)}")
        return parts[0] if role == "source" else parts[1]

    return name


def _identity_from_info(video_name, video_info, label_name, regex=None, fake_identity_role="source"):
    for key in IDENTITY_KEYS:
        value = video_info.get(key)
        if value:
            return _stem(value)

    if not _is_real_label(label_name):
        role = str(fake_identity_role).lower()
        if role not in SUPPORTED_FAKE_IDENTITY_ROLES:
            raise ValueError(f"fake_identity_role must be one of {sorted(SUPPORTED_FAKE_IDENTITY_ROLES)}")
        role_keys = ("source", "source_id", "source_video") if role == "source" else ("target", "target_id")
        for key in role_keys:
            value = video_info.get(key)
            if value:
                return _stem(value)

    name = _stem(video_name)
    if regex:
        match = re.search(regex, name)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    return _identity_from_video_name(name, label_name, fake_identity_role=fake_identity_role)


def _subsample_frames(frames, frame_num):
    if frame_num is None or int(frame_num) <= 0 or len(frames) <= int(frame_num):
        return frames
    frame_num = int(frame_num)
    step = max(1, len(frames) // frame_num)
    return [frames[i] for i in range(0, len(frames), step)][:frame_num]


def load_deepfake_records(
    dataset_json_folder,
    datasets,
    split,
    compression="c23",
    frame_num=None,
    base_data_path=None,
    label_dict=None,
    identity_regex=None,
    fake_identity_role="source",
    is_training=False,
):
    records: List[FrameRecord] = []
    for dataset_name in split_dataset_names(datasets):
        json_path = _resolve_json_path(dataset_json_folder, dataset_name)
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        with json_path.open("r") as handle:
            data = json.load(handle)
        root_name = next(iter(data))
        for label_name, split_map in data[root_name].items():
            if split not in split_map:
                continue
            videos = _select_compression(split_map[split], root_name, compression)
            for video_name, video_info in videos.items():
                frames = video_info.get("frames", [])
                if not frames:
                    continue
                frames = [_normalize_path(path, base_data_path) for path in frames]
                frames = _subsample_frames(frames, frame_num)
                target = _deepfakebench_label(
                    label_name=label_name,
                    split=split,
                    is_training=is_training,
                    label_dict=label_dict,
                )
                identity = _identity_from_info(
                    video_name=video_name,
                    video_info=video_info,
                    label_name=label_name,
                    regex=identity_regex,
                    fake_identity_role=fake_identity_role,
                )
                records.extend(
                    FrameRecord(
                        path=path,
                        label=target,
                        label_name=label_name,
                        dataset_name=root_name,
                        split=split,
                        video_name=str(video_name),
                        identity=identity,
                    )
                    for path in frames
                )
    return records


class RandomAugmentReplace:
    def __init__(self, base_transform, augment_transform, p=0.1):
        self.base_transform = base_transform
        self.augment_transform = augment_transform
        self.p = float(p)

    def __call__(self, image):
        if random.random() < self.p:
            return self.augment_transform(image)
        return self.base_transform(image)


def _processor_image_size(processor, fallback=224):
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        for key in ("shortest_edge", "height", "width"):
            value = size.get(key)
            if value is not None:
                return int(value)
    if isinstance(size, int):
        return int(size)
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, dict):
        for key in ("height", "width", "shortest_edge"):
            value = crop_size.get(key)
            if value is not None:
                return int(value)
    if isinstance(crop_size, int):
        return int(crop_size)
    return int(fallback)


def _processor_mean_std(processor):
    mean = getattr(processor, "image_mean", None) or CLIP_MEAN
    std = getattr(processor, "image_std", None) or CLIP_STD
    return mean, std


def build_clip_transform(processor, train=True, image_size=None, augmentation_prob=0.1,
                         load_size=None, crop_size=None, reference_normalization="clip"):
    """Build the image preprocessing pipeline.

    When both ``load_size`` and ``crop_size`` are not None, use the Effort
    reference pipeline:
      train: translate_duplicate(load_size) → RandomCrop(crop_size) → RandomHorizontalFlip →
             ToTensor → Normalize(CLIP_MEAN, CLIP_STD)
      eval:  translate_duplicate(load_size) → CenterCrop(crop_size) → ToTensor →
             Normalize(CLIP_MEAN, CLIP_STD)

    Otherwise fall back to the original CLIPMLPa pipeline (Resize + CenterCrop +
    optional augmentation).
    """
    # ---- Effort reference pipeline ----
    if load_size is not None and crop_size is not None:
        if str(reference_normalization).strip().lower() == "processor":
            mean, std = _processor_mean_std(processor)
        else:
            mean, std = CLIP_MEAN, CLIP_STD
        if train:
            return transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, load_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            return transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, load_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    # ---- Original CLIPMLPa pipeline ----
    processor_size = _processor_image_size(processor)
    image_size = int(image_size or processor_size)
    mean, std = _processor_mean_std(processor)
    base_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    if train:
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return RandomAugmentReplace(base_transform, augment_transform, p=augmentation_prob)
    return base_transform


def load_rgb(path):
    return Image.open(path).convert("RGB")


class FrameDataset(Dataset):
    def __init__(
        self,
        records: Sequence[FrameRecord],
        transform=None,
        strict_existing_files=False,
        include_label_name=False,
        include_video_name=False,
        paired_transform=None,
    ):
        self.records = list(records)
        self.transform = transform
        self.strict_existing_files = bool(strict_existing_files)
        self.include_label_name = bool(include_label_name)
        self.include_video_name = bool(include_video_name)
        self.paired_transform = paired_transform
        if not self.records:
            raise ValueError("FrameDataset received no records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.strict_existing_files and not os.path.exists(record.path):
            raise FileNotFoundError(record.path)
        image = load_rgb(record.path)
        image_view2 = image.copy() if self.paired_transform is not None else None
        if self.transform:
            image = self.transform(image)
        if self.paired_transform is not None:
            image_view2 = self.paired_transform(image_view2)
        item = {
            "image": image,
            "label": torch.tensor(record.label, dtype=torch.long),
            "binary_label": torch.tensor(0 if record.label == 0 else 1, dtype=torch.long),
            "path": record.path,
            "identity": record.identity,
        }
        if image_view2 is not None:
            item["image_view2"] = image_view2
        if self.include_label_name:
            item["label_name"] = record.label_name
        if self.include_video_name:
            item["video_name"] = record.video_name
        return item


class BalancedRealFakeBatchSampler(Sampler):
    """Yield batches with the same number of real and fake frame indices."""

    def __init__(self, records: Sequence[FrameRecord], batch_size, drop_last=False, seed=42):
        batch_size = int(batch_size)
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("BalancedRealFakeBatchSampler requires an even batch_size >= 2")

        self.real_indices = [index for index, record in enumerate(records) if record.label == 0]
        self.fake_indices = [index for index, record in enumerate(records) if record.label != 0]
        if not self.real_indices or not self.fake_indices:
            raise ValueError(
                "BalancedRealFakeBatchSampler requires at least one real and one fake sample; "
                f"found {len(self.real_indices)} real and {len(self.fake_indices)} fake samples."
            )

        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        majority = max(len(self.real_indices), len(self.fake_indices))
        if self.drop_last:
            self.num_batches = max(1, majority // self.half_batch)
        else:
            self.num_batches = max(1, (majority + self.half_batch - 1) // self.half_batch)

    def __len__(self):
        return self.num_batches

    def _draw_indices(self, indices, total, rng):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) >= total:
            return shuffled[:total]

        drawn = []
        while len(drawn) < total:
            chunk = list(indices)
            rng.shuffle(chunk)
            drawn.extend(chunk)
        return drawn[:total]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        total_per_class = self.num_batches * self.half_batch
        real_epoch_indices = self._draw_indices(self.real_indices, total_per_class, rng)
        fake_epoch_indices = self._draw_indices(self.fake_indices, total_per_class, rng)

        for start in range(0, total_per_class, self.half_batch):
            batch = real_epoch_indices[start:start + self.half_batch]
            batch.extend(fake_epoch_indices[start:start + self.half_batch])
            rng.shuffle(batch)
            yield batch


def collate_frames(batch):
    collated = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "binary_label": torch.stack([item["binary_label"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "identity": [item["identity"] for item in batch],
    }
    if "image_view2" in batch[0]:
        collated["image_view2"] = torch.stack([item["image_view2"] for item in batch], dim=0)
    if "label_name" in batch[0]:
        collated["label_name"] = [item["label_name"] for item in batch]
    if "video_name" in batch[0]:
        collated["video_name"] = [item["video_name"] for item in batch]
    return collated


def make_processor(model_name, model_type="clip", cache_dir=None, local_files_only=False):
    kwargs = {"local_files_only": bool(local_files_only)}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if str(model_type).strip().lower() == "clip":
        return CLIPImageProcessor.from_pretrained(model_name, **kwargs)
    return AutoImageProcessor.from_pretrained(model_name, **kwargs)
