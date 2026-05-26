"""Shared helpers for per-model tumor inference."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import SimpleITK as sitk

from train.data.dataset import normalize_volume


def resolve_phase_paths(attempt_dir, cfg) -> Tuple[Path, Path, Path]:
    """Return the (A, P, D) NIfTI paths a tumor model should consume.

    The pipeline always stages two variants under each registration folder:
      - ``attempt_dir/{A,P,D}.nii.gz``                  : the (registered) CT
      - ``attempt_dir/liver_images/{A,P,D}_liver.nii.gz``: liver-cropped (CT * liver mask),
        produced either by copying the training-side AliverAv/PliverPv/DliverDv
        files (no-registration shortcut) or by LiverExtractor after registration.

    ``cfg.data_type`` then selects which set the tumor models read.
    """
    attempt_dir = Path(attempt_dir)
    data_type = getattr(cfg, "data_type", "ct")
    if data_type == "liver":
        liver_dir = attempt_dir / "liver_images"
        return (
            liver_dir / "A_liver.nii.gz",
            liver_dir / "P_liver.nii.gz",
            liver_dir / "D_liver.nii.gz",
        )
    return (
        attempt_dir / "A.nii.gz",
        attempt_dir / "P.nii.gz",
        attempt_dir / "D.nii.gz",
    )


def read_volume(path):
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image)
    return image, volume


def load_case(a_path, p_path, d_path, window):
    reference_image, a_volume = read_volume(a_path)
    _, p_volume = read_volume(p_path)
    _, d_volume = read_volume(d_path)

    a_volume = normalize_volume(a_volume, window)
    p_volume = normalize_volume(p_volume, window)
    d_volume = normalize_volume(d_volume, window)

    image = np.stack([a_volume, p_volume, d_volume], axis=0)
    image = torch.from_numpy(image).unsqueeze(0).float()
    return image, reference_image


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned)
    return checkpoint


def save_nifti(array, reference_image, output_path):
    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(output_path))


@torch.no_grad()
def sliding_window_predict(model, image, roi_size, sw_batch_size, overlap, mode):
    from monai.inferers import sliding_window_inference

    return sliding_window_inference(
        inputs=image,
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode=mode,
    )
