# XDL DenseCaps

Hybrid DenseNet121 + CapsNet training code for binary GI image classification.

The project currently predicts two classes:

- `normal`
- `lesion`

Images inside `data/raw/normal` or `data/raw/nomal` are labeled `normal`.
Images inside every other folder under `data/raw` are labeled `lesion`.

## Model

The default workflow is:

```text
image -> DenseNet121 feature extractor -> CapsNet head -> capsule margin loss
```

The implementation lives in:

```text
src/xdl_densecaps/models/densenet_capsnet.py
```

The main pieces are:

- `DenseNetCapsNetClassifier`
- `DenseNetCapsHead`
- `RoutingCapsuleLayer`
- `CapsuleMarginLoss`

The older plain DenseNet classifier is still available as `densenet121`, but the
default config now uses `densenet121_capsnet`.

## Setup

Use `uv` from the project root:

```powershell
uv sync --dev
```

You do not need to manually activate `.venv` when using `uv run`.

`uv.lock` is intentionally not tracked. PyTorch/CUDA builds can differ per PC,
so each machine should resolve or install the Torch build that matches its own
driver/CUDA setup.

## Data Layout

Expected dataset layout:

```text
data/raw/
  nomal/      -> normal class
  normal/     -> normal class, also supported
  uc_1/       -> lesion class
  uc_2/       -> lesion class
  uc_3/       -> lesion class
```

The folder names under the lesion class do not need to be exactly `uc_*`.
Anything that is not `normal` or `nomal` becomes label `1`. Older folders named
`lession` are still accepted because they are treated as non-normal folders.

## Config

Training settings live in:

```text
configs/config.yaml
```

Current important fields:

```yaml
data:
  root_dir: data/raw
  split_dir: data/splits/capsnet
  image_size: 128
  batch_size: 8
  val_ratio: 0.2
  test_ratio: 0.1

model:
  name: densenet121_capsnet
  pretrained: false
  backbone_checkpoint_path: artifacts/normal_lesion_densenet121/best.pt
  freeze_backbone: false
  feature_h: 4
  feature_w: 4
  num_capsules: 256
  capsule_routing_iters: 2
  digit_routing_iters: 3
  margin_lambda: 0.5

training:
  epochs: 100
  learning_rate: 0.00005
  weight_decay: 0.0001
  early_stopping_patience: 30

runtime:
  output_dir: artifacts/normal_lesion_densenet121_capsnet
  device: auto
  checkpoint_name: best.pt
```

The default config initializes the DenseNet backbone from:

```text
artifacts/normal_lesion_densenet121/best.pt
```

To train the hybrid model without that old DenseNet-only checkpoint, set:

```yaml
model:
  backbone_checkpoint_path: null
```

For the Kvasir v2 binary dataset, use:

```powershell
uv run xdl-train --config configs/kvasir_v2_capsnet.yaml
```

That config reads from `data/kvasir-v2` and trains from scratch by default, so it
does not require any local file under `artifacts/`.

Splits are duplicate-aware. The split generator hashes each image file and keeps
exact duplicate files in the same split, so identical image content cannot appear
in train, validation, and test at the same time.

## Train

```powershell
uv run xdl-train --config configs/config.yaml
```

Training validates after each epoch and saves the best checkpoint to:

```text
artifacts/normal_lesion_densenet121_capsnet/best.pt
```

The output folder also contains:

```text
train.log
run_metadata.json
class_names.txt
```

## Validate

```powershell
uv run xdl-val --config configs/config.yaml
```

This writes:

```text
artifacts/normal_lesion_densenet121_capsnet/val_metrics.json
artifacts/normal_lesion_densenet121_capsnet/val.log
```

## Test

```powershell
uv run xdl-test --config configs/config.yaml
```

This writes:

```text
artifacts/normal_lesion_densenet121_capsnet/test_metrics.json
artifacts/normal_lesion_densenet121_capsnet/test.log
```

## Filter Lesion Regions

After training, create a dataset of the selected lesion-region crops with:

```powershell
uv run xdl-filter-lesions --config configs/config.yaml
```

This processes every image under `data.root_dir`, builds KMeans normal-region
centroids from correctly predicted normal images, and saves one filtered lesion
crop per correctly predicted lesion image. It also saves one selected normal
crop per correctly predicted normal image. The default output is:

```text
artifacts/normal_lesion_densenet121_capsnet/filtered_lesion_regions/
  images/          -> selected lesion crops
  normal_images/   -> selected normal crops
  metadata.json
```

## Direct Python Commands

These are equivalent to the script commands:

```powershell
uv run python -m xdl_densecaps.train --config configs/config.yaml
uv run python -m xdl_densecaps.val --config configs/config.yaml
uv run python -m xdl_densecaps.test --config configs/config.yaml
uv run python -m xdl_densecaps.filter_lesion_regions --config configs/config.yaml
```

## Notebook

`stage1.ipynb` is the readable prototype/debug notebook for the DenseNet +
CapsNet path up to the capsule classifier. The reusable model code now lives in
`src/xdl_densecaps/models/densenet_capsnet.py`, so the notebook imports those
classes instead of redefining the full CapsNet implementation inline.

## Checks

Run the focused test suite with:

```powershell
uv run pytest
```
