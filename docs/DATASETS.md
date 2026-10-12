# Dataset Layout

This repository does not redistribute datasets. Please follow the license of
each source dataset and place the files outside the repository if possible.

## DeepfakeBench

The loader needs two locations: a folder containing benchmark JSON metadata and
the root folder containing extracted frames. A common local layout is:

```text
DeepfakeBench/
  dataset_json/
    FaceForensics++.json
    Celeb-DF-v2.json
    DFDC.json
    DFDCP.json
  Datasets/
    FaceForensics++/
      c23/
        frames/
    Celeb-DF-v2/
      frames/
    DFDC/
      frames/
```

Each JSON file should contain real and fake videos with their extracted frames:

```json
{
  "FaceForensics++": {
    "Real": {
      "train": {
        "000_003": {"frames": ["FaceForensics++/c23/frames/000_003/frame_0000.png"]}
      },
      "test": {
        "000_003": {"frames": ["FaceForensics++/c23/frames/000_003/frame_0000.png"]}
      }
    },
    "Fake": {
      "train": {
        "000_003_000": {"frames": ["FaceForensics++/c23/frames/000_003_000/frame_0000.png"]}
      },
      "test": {
        "000_003_000": {"frames": ["FaceForensics++/c23/frames/000_003_000/frame_0000.png"]}
      }
    }
  }
}
```

Absolute frame paths are supported. Relative paths are resolved against the
`base_data_path` argument, for example the `DeepfakeBench/Datasets` folder shown
above. Real labels are mapped to `0` and fake labels to `1`.

The two required arguments are:

```text
--data-json-folder /path/to/DeepfakeBench/dataset_json
--base-data-path  /path/to/DeepfakeBench/Datasets
```

## UniversalFakeDetect

The expected root layout is:

```text
UniversalFakeDetect/
  datasets/
    train/
      <category>/
        0_real/
        1_fake/
    test/
      <generator>/
        0_real/
        1_fake/
      cyclegan/
        <class>/
          0_real/
          1_fake/
  diffusion_datasets/
    imagenet/
      0_real/
    laion/
      0_real/
    guided/
      1_fake/
    ldm_200/
      1_fake/
    dalle/
      1_fake/
```

The universal loader also supports diffusion test sets that pair one real source
with one fake source. It exposes them as separate evaluation keys, for example
`imagenet_guided` or `imagenet_ldm_200`.

The required argument is:

```text
--data-root /path/to/UniversalFakeDetect
```

## Suggested Local Checkout

To keep datasets out of Git, use a sibling directory such as:

```text
FePreR2/
data/
  DeepfakeBench/
  UniversalFakeDetect/
weights/
  best_auc_123.pt
  universal-final.pt
```

The `data/` and `weights/` folders are ignored by Git. Do not remove that ignore
rule.
