# LangGraph Liver Tumor Segmentation

Research code for multi-phase liver tumor segmentation and registration-driven
inference on abdominal CT. The project is built around arterial, portal venous,
and delayed phase NIfTI volumes, with baseline segmentation models and a
LangGraph-style orchestration notebook for patient-level inference.

## Overview

This repository supports experiments for liver tumor segmentation using three
contrast-enhanced CT phases:

- **A**: arterial phase
- **P**: portal venous phase
- **D**: delayed phase

The current training and inference code covers:

- 3D U-Net
- SwinUNETR
- 3D SAM Adapter
- nnUNetv2 dataset preparation and CLI integration
- registration quality control using liver Dice
- tumor segmentation evaluation with Dice, IoU, and lesion-level detection metrics

The repository intentionally excludes medical images, generated nnUNet folders,
model checkpoints, and local spreadsheet metadata. See `.gitignore` for the data
and artifact policy.

## Repository Layout

```text
.
├── main.py                         # Training entry point for UNet, SwinUNETR, 3D SAM Adapter
├── inference.py                    # Single-case inference for direct PyTorch baselines
├── nnunetdataset.py                # nnUNet raw dataset conversion utility
├── pipeline.ipynb                  # LangGraph-style patient inference pipeline
├── logger.py                       # File and console logger setup
├── train/
│   ├── trainer.py                  # MONAI-based generic trainer
│   ├── sam_adapter_trainer.py      # Dedicated 3D SAM Adapter training loop
│   ├── data/dataset.py             # 3-phase CT dataset and MONAI preprocessing
│   └── models/
│       ├── UNet.py
│       ├── SwinUNetR.py
│       ├── SAMAdapter.py
│       └── 3DSAMAdapter/           # 3D SAM Adapter model components
└── langgraph/files/
    ├── config.py
    ├── confg.py
    ├── registration.py
    ├── resampler.py
    ├── liver_extractor.py
    └── tumor_extractor.py
```

## Data Assumptions

The training split is derived from `alldata_metrics.xlsx` and sorted by tumor
size. By default:

- test cases: first 34 sorted cases
- train cases: remaining cases

Two input modes are supported:

- `--data_type ct`: original CT phases
  - `A.nii.gz`
  - `P.nii.gz`
  - `D.nii.gz`
- `--data_type liver`: liver-extracted phase volumes
  - `AliverAv.nii.gz`
  - `PliverPv.nii.gz`
  - `DliverDv.nii.gz`

Labels are expected as:

```text
label.nii.gz
```

Local data roots are configured in `config.py`.

## Installation

Create an environment with PyTorch and the medical imaging dependencies used by
the pipeline:

```bash
pip install -r requirements.txt
```

For the 3D SAM Adapter baseline, install the Segment Anything package and place
the required pretrained checkpoints locally:

```text
ckpt/sam_vit_b_01ec64.pth
snapshot/lits/last.pth.tar
```

These checkpoint paths can also be overridden from the command line.

## Training

### 3D U-Net

```bash
python main.py \
  --model unet \
  --data_type liver \
  --epochs 100 \
  --batch_size 4 \
  --roi_size 96 128 128 \
  --augment
```

### SwinUNETR

```bash
python main.py \
  --model swinunetr \
  --data_type liver \
  --epochs 100 \
  --batch_size 1 \
  --roi_size 96 128 128 \
  --feature_size 48 \
  --use_checkpoint
```

### 3D SAM Adapter

The 3D SAM Adapter uses a dedicated training loop because the model is composed
of a SAM image encoder, prompt encoders, and a mask decoder rather than a single
`model(image)` forward pass.

```bash
python main.py \
  --model sam_adapter \
  --data_type liver \
  --epochs 100 \
  --batch_size 1 \
  --roi_size 128 128 128 \
  --lr 4e-4 \
  --sam_checkpoint ckpt/sam_vit_b_01ec64.pth \
  --sam_pretrained_ckpt snapshot/lits/last.pth.tar
```

For this baseline, cubic crops are required by the reference training flow.
`main.py` automatically overrides the crop size to `128 128 128` when needed.

## Inference

Direct PyTorch baselines can be run on one patient with:

```bash
python inference.py \
  --model unet \
  --checkpoint checkpoints/unet/best_model.pt \
  --A /path/to/A.nii.gz \
  --P /path/to/P.nii.gz \
  --D /path/to/D.nii.gz \
  --output_dir ./inference_outputs \
  --save_probability
```

Sliding-window inference is enabled by default through MONAI.

## nnUNetv2 Dataset Conversion

The conversion script creates nnUNet-compatible raw datasets for original CT
volumes and liver-extracted volumes:

```bash
python nnunetdataset.py --dataset all --dry_run
python nnunetdataset.py --dataset all --overwrite
```

The intended layout is:

```text
nnUNet/nnUNet_raw/
├── Dataset001/    # original CT phases
└── Dataset002/    # liver-extracted phase volumes
```

Generated nnUNet folders are ignored by git.

## LangGraph Pipeline

`pipeline.ipynb` sketches the full patient-level inference workflow:

1. Register A, P, and D phase volumes.
2. Evaluate registration quality with liver Dice.
3. Retry registration with alternate hyperparameters if liver Dice is below the threshold.
4. Run tumor segmentation only after registration passes quality control.
5. Save per-patient registration attempts, tumor predictions, metrics, and summaries.

The target output layout is:

```text
Results/
└── {subject}_{date}/
    ├── registration/
    │   └── attempt_*/
    ├── tumor/
    │   ├── unet/
    │   ├── swinunetr/
    │   ├── sam3d_adapter/
    │   └── nnunetv2/
    └── summary.json
```

## Metrics

Validation includes:

- Dice
- IoU
- lesion-level detection metrics
  - F1 score
  - sensitivity
  - thresholded overlap from 10% to 50%

## Version Control Notes

Large or sensitive files are excluded from git:

- NIfTI/DICOM medical images
- nnUNet raw, preprocessed, and result folders
- PyTorch checkpoints
- SAM checkpoints
- local Excel/CSV metadata
- generated result folders

Use local storage or an institutional artifact store for these files.

## Development Workflow

Optional pre-commit hooks are provided:

```bash
pip install pre-commit ruff
pre-commit install
```

After that, commits will automatically run lightweight formatting and safety
checks before changes are recorded.

### Optional Auto-Push

Two local automation helpers are available.

To automatically commit and push file changes while a watcher is running:

```bash
bash scripts/auto_push_on_change.sh 30
```

The number is the polling interval in seconds. The script respects `.gitignore`,
so medical images, checkpoints, nnUNet outputs, and local metadata are not added
to git.

To push automatically after every manual commit:

```bash
bash scripts/install_auto_push_hook.sh
```

This installs a local `.git/hooks/post-commit` hook. Git hooks are local machine
state and are not stored in the repository.
